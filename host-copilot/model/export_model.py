"""Export the final pricing model so Host Copilot can load it without a notebook.

This script does NOT change the training code. It imports the project's own feature code
(`airbnb-pricing/src/`) and repeats the final fit from
`airbnb-pricing/notebooks/boston_airbnb_final_pipeline.ipynb` with the same 141 features,
the same LightGBM settings and the same seed. It then checks the result against the
predictions the notebook stored in `airbnb-pricing/interface/models.js`.

Outputs (model/artifacts/):
  final_lgbm.txt      LightGBM booster (text format)
  meta.json           feature order, interval multipliers, CV metrics, snapshot dates
  supply.csv.gz       listing coordinates per quarter (for the "listings within 300 m / 1 km" features)
  segments.csv        typical values per neighbourhood x room type, used to fill fields a host leaves blank
  hood_centers.csv    median coordinates of each neighbourhood
  reference.csv.gz    the 7,039 training listings (key fields, amenity flags, price), used later for comparables

Run:  python model/export_model.py      (about 2 minutes on 4 CPUs)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
PRICING_DIR = Path(os.environ.get("AIRBNB_PRICING_DIR", HERE.parents[1] / "airbnb-pricing"))
ART = HERE / "artifacts"
sys.path.insert(0, str(PRICING_DIR / "src"))
import pricing_features as pf      # noqa: E402  (project code, unchanged)
import enriched_features as ef     # noqa: E402

SEED = 42
# Same settings as the final pipeline notebook (section 4, "the final fit").
LGB_PARAMS = dict(n_estimators=900, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0, random_state=SEED, verbose=-1)
SNAPDATE = {"2026Q2": "2026-06-16", "2026Q3": "2026-09-20"}
DETAILED = {"2026Q2": "listings_detailed_2026Q2_jun.csv.gz", "2026Q3": "listings_detailed_2026Q3_sep.csv.gz"}


def read_js_bundle(path):
    """Read the JSON object inside a `window.X = {...};` file written by the notebooks."""
    text = Path(path).read_text()
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


def load_training_data():
    """Build the same (rows, features, log price, host groups) the notebook trains on."""
    bundle = read_js_bundle(PRICING_DIR / "interface" / "data.js")
    zones = bundle["quarterly"]["zones"]["centers_km"]
    A, X, y, groups = ef.build_merged_matrix(PRICING_DIR / "data", zones)
    features = read_js_bundle(PRICING_DIR / "interface" / "models.js")["final"]["features"]
    assert set(features) == set(X.columns), "feature set differs from the notebook's final model"
    return A, X[features], np.asarray(y), np.asarray(groups), zones, bundle


def out_of_fold_residuals(X, y, groups):
    """Log-price residuals of 5-fold CV where all listings of a host sit in the same fold.
    These are errors on hosts the model never saw, which is what a new host gets."""
    oof = np.zeros(len(y))
    for train, test in GroupKFold(5).split(X, y, groups):
        oof[test] = lgb.LGBMRegressor(**LGB_PARAMS).fit(X.iloc[train], y[train]).predict(X.iloc[test])
    return y - oof, oof


def interval_multipliers(residuals, level):
    """Empirical interval: prediction x exp(lower quantile) to prediction x exp(upper quantile)."""
    tail = (1 - level) / 2
    lo, hi = np.quantile(residuals, [tail, 1 - tail])
    return {"level": level, "low_log": float(lo), "high_log": float(hi),
            "low_mult": float(np.exp(lo)), "high_mult": float(np.exp(hi))}


def supply_table():
    """All listings (priced or not) per quarter after the pre-check, rounded like the notebook."""
    official = pd.read_csv(PRICING_DIR / "data" / "neighbourhoods.csv")["neighbourhood"]
    geo = pf.load_geo(PRICING_DIR / "data" / "neighbourhoods.geojson")
    parts = []
    for q, f in DETAILED.items():
        raw = ef.load_detailed(PRICING_DIR / "data" / "detailed" / f)
        s, _ = pf.precheck(raw, official, geo, require_price=False)
        parts.append(s[["latitude", "longitude"]].round(5).assign(quarter=q))
    return pd.concat(parts, ignore_index=True)


def segment_defaults(A, X):
    """Median (or share) of each host-describable field per neighbourhood x room type.
    Used only for fields the host leaves blank; the app lists which fields were filled."""
    d = pd.DataFrame({
        "neighbourhood": A["neighbourhood"].values, "room_type": A["room_type"].values,
        "accommodates": X["accommodates"].values, "bedrooms": X["bedrooms_d"].values, "beds": X["beds"].values,
        "baths": X["baths"].values, "minimum_nights": A["minimum_nights"].values,
        "availability_365": A["availability_365"].values, "number_of_reviews": A["number_of_reviews"].values,
        "reviews_per_month": A["reviews_per_month"].fillna(0).values,
        "number_of_reviews_ltm": A["number_of_reviews_ltm"].values,
        "host_listings": A["calculated_host_listings_count"].values, "review_score": X["rs_rating"].values,
        "months_since_first_review": X["months_since_first_review"].values,
        "days_since_review": X["days_since_review"].values, "am_count": X["am_count"].values,
        "desc_len": X["desc_len"].values, "price": A["price"].values,
    })
    seg = d.groupby(["neighbourhood", "room_type"]).median(numeric_only=True)
    seg["n"] = d.groupby(["neighbourhood", "room_type"]).size()
    overall = d.groupby("room_type").median(numeric_only=True).assign(n=d.groupby("room_type").size())
    overall = overall.reset_index().assign(neighbourhood="*").set_index(["neighbourhood", "room_type"])
    return pd.concat([seg, overall]).round(3)


def reference_table(A, X, y):
    """Training listings with the fields comparables need (step 6)."""
    am_cols = [c for c in X.columns if c.startswith("am_") and c != "am_count"]
    ref = pd.DataFrame({"id": A["id"].values, "quarter": A["quarter"].values, "neighbourhood": A["neighbourhood"].values,
                        "room_type": A["room_type"].values, "latitude": A["latitude"].values, "longitude": A["longitude"].values,
                        "accommodates": X["accommodates"].values, "bedrooms": X["bedrooms_d"].values, "baths": X["baths"].values,
                        "price": np.round(np.exp(y), 2)})
    return pd.concat([ref, X[am_cols].reset_index(drop=True).astype(int)], axis=1)


def main():
    ART.mkdir(parents=True, exist_ok=True)
    A, X, y, groups, zones, bundle = load_training_data()
    print(f"training rows {len(y):,}, features {X.shape[1]}")

    res, oof = out_of_fold_residuals(X, y, groups)
    mae = float(np.mean(np.abs(np.exp(y) - np.exp(oof))))
    i90, i80 = interval_multipliers(res, 0.90), interval_multipliers(res, 0.80)
    nb80 = bundle["final"]["band_log"]
    print(f"out-of-fold MAE ${mae:.2f}; 90% interval x{i90['low_mult']:.2f}..x{i90['high_mult']:.2f}; "
          f"80% matches notebook: {np.allclose([i80['low_log'], i80['high_log']], nb80, atol=1e-6)}")

    final = lgb.LGBMRegressor(**LGB_PARAMS).fit(X, y)
    final.booster_.save_model(str(ART / "final_lgbm.txt"))

    ref = read_js_bundle(PRICING_DIR / "interface" / "models.js")["final_reference"]
    Xr = pd.DataFrame([[np.nan if v is None else v for v in r] for r in ref["X"]], columns=X.columns)
    gap = float(np.max(np.abs(final.predict(Xr) - np.asarray(ref["pred"]))))
    print(f"max |log difference| vs notebook predictions on 200 stored rows: {gap:.2e}")

    meta = {"features": list(X.columns), "lgb_params": LGB_PARAMS, "seed": SEED, "n_rows": int(len(y)),
            "n_hosts": int(len(set(groups))), "oof_mae": mae, "interval": {"90": i90, "80": i80},
            "residual_quantiles": [float(v) for v in np.quantile(res, np.linspace(0, 1, 201))],
            "interval_method": "Empirical quantiles of out-of-fold log-price residuals (5-fold CV, hosts held out). "
                               "Same multiplier for every listing.",
            "snapshot_dates": SNAPDATE, "zones_km": zones, "zone_names": bundle["quarterly"]["zones"]["names"], "notebook_gap_log": gap}
    (ART / "meta.json").write_text(json.dumps(meta, indent=1))
    supply_table().to_csv(ART / "supply.csv.gz", index=False)
    segment_defaults(A, X).to_csv(ART / "segments.csv")
    A.groupby("neighbourhood")[["latitude", "longitude"]].median().to_csv(ART / "hood_centers.csv")
    reference_table(A, X, y).to_csv(ART / "reference.csv.gz", index=False)
    print("wrote", sorted(p.name for p in ART.iterdir()))


if __name__ == "__main__":
    main()
