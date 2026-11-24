"""Which changes a host controls move the price, and where?

Changes are the 30 amenities and a few non-amenity levers (minimum nights, superhost, licence,
verified identity). For every real listing that could make a change (it lacks the amenity, or its
minimum nights are in the band the move starts from), apply the change and re-price the listing.
Doing this for all 7,039 training listings gives the model's estimated lift for each change in each
area, instead of for one sample listing.

Honesty checks
  * out-of-fold: each listing is re-priced by the CV model that never saw its host;
  * stability: the area median lift is computed with each of the 5 fold models; a lift whose
    sign flips between models is not trusted;
  * support: the area needs enough listings with AND without the amenity to learn from;
  * raw check: price gap between listings with and without the amenity inside the same area,
    room type and bedroom count (confounded, shown only as a sanity check).

Areas: the 8 zones used by the model (named after their main neighbourhoods) and the
neighbourhoods themselves; plus all of Boston.

Output: model/artifacts/change_effects.csv (one row per area x change)
Run:    python analysis/change_effects.py      (about 2 minutes)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
import export_model as em          # noqa: E402  (same training data and settings as the exported model)
import vocab                       # noqa: E402

OUT = ROOT / "model" / "artifacts" / "change_effects.csv"
AMENITIES = list(vocab.AMENITIES)


def min_nights(X):
    """Minimum nights of each feature row (the model stores log(1 + nights))."""
    return np.round(np.expm1(X["log_min_nights"].values))


def set_min_nights(n):
    """Transform that sets minimum nights to n."""
    def f(X):
        Z = X.copy()
        Z["log_min_nights"] = np.log1p(n)
        return Z
    return f


def set_flags(**values):
    """Transform that sets feature columns to fixed values."""
    def f(X):
        Z = X.copy()
        for k, v in values.items():
            Z[k] = v
        return Z
    return f


def switch_on(X, key):
    """Copy of the feature rows with one amenity switched on and one more amenity line."""
    Z = X.copy()
    Z[key] = 1
    if key == "am_central_ac":
        Z["am_any_ac"] = 1
    Z["am_count"] = Z["am_count"] + 1
    return Z


def fit_fold_models(X, y, groups):
    """The 5 cross-validation models (hosts held out) and the fold of each listing."""
    fold_of = np.zeros(len(y), dtype=int)
    models = []
    for k, (train, test) in enumerate(GroupKFold(5).split(X, y, groups)):
        models.append(lgb.LGBMRegressor(**em.LGB_PARAMS).fit(X.iloc[train], y[train]))
        fold_of[test] = k
    return models, fold_of


def changes():
    """key -> (who can make it: mask(X), apply it: transform(X), already there: state(X))."""
    C = {key: ((lambda X, k=key: X[k].values == 0), (lambda X, k=key: switch_on(X, k)), (lambda X, k=key: X[k].values == 1))
         for key in AMENITIES}
    mn = min_nights
    C.update({
        "lv_min_nights_1": (lambda X: (mn(X) >= 2) & (mn(X) <= 27), set_min_nights(1), lambda X: mn(X) == 1),
        "lv_min_nights_2": (lambda X: (mn(X) >= 3) & (mn(X) <= 27), set_min_nights(2), lambda X: mn(X) == 2),
        "lv_min_nights_5": (lambda X: mn(X) <= 2, set_min_nights(5), lambda X: (mn(X) >= 4) & (mn(X) <= 7)),
        "lv_monthly": (lambda X: mn(X) <= 27, set_min_nights(30), lambda X: mn(X) >= 28),
        "lv_nightly": (lambda X: mn(X) >= 28, set_min_nights(2), lambda X: mn(X) <= 27),
        "lv_superhost": (lambda X: X["superhost"].values == 0, set_flags(superhost=1), lambda X: X["superhost"].values == 1),
        "lv_licensed": (lambda X: X["licensed"].values == 0, set_flags(licensed=1, exempt=0), lambda X: X["licensed"].values == 1),
        "lv_verified": (lambda X: X["identity_verified"].values == 0, set_flags(identity_verified=1),
                        lambda X: X["identity_verified"].values == 1),
    })
    return C


def lifts(models, fold_of, X):
    """Per listing and change (only listings that can make it): lift in % and $ from each fold
    model and out-of-fold (the model that never saw the listing's host)."""
    rows = []
    base = np.column_stack([np.exp(m.predict(X)) for m in models])          # n x 5
    for key, (can, apply, _) in changes().items():
        idx = np.where(can(X))[0]
        if len(idx) == 0:
            continue
        Z = apply(X.iloc[idx])
        on = np.column_stack([np.exp(m.predict(Z)) for m in models])        # m x 5
        pct = on / base[idx] - 1
        dol = on - base[idx]
        own = fold_of[idx]
        rows.append(pd.DataFrame({
            "row": idx, "change": key,
            "oof_pct": pct[np.arange(len(idx)), own], "oof_usd": dol[np.arange(len(idx)), own],
            **{f"pct_m{k}": pct[:, k] for k in range(len(models))}}))
    return pd.concat(rows, ignore_index=True)


def raw_gap(A, X, y, rows, has):
    """Median log-price gap (already in the target state - not) inside cells of same room type and bedroom count,
    weighted by the smaller group, as %. NaN when no cell has 3+ listings on both sides."""
    d = pd.DataFrame({"room": A["room_type"].values[rows], "bed": np.minimum(X["bedrooms_d"].values[rows], 3),
                      "has": has[rows], "y": y[rows]})
    num = den = 0.0
    for _, g in d.groupby(["room", "bed"]):
        a, b = g[g.has == 1].y, g[g.has == 0].y
        if len(a) >= 3 and len(b) >= 3:
            w = min(len(a), len(b))
            num += w * (a.median() - b.median())
            den += w
    return float(np.expm1(num / den)) if den else np.nan


def summarise(A, X, y, L, areas):
    """One row per area x change. `areas` lists (row, level, area); each listing is in 3 areas."""
    out = []
    state = {key: st(X).astype(int) for key, (_, _, st) in changes().items()}
    L = L.merge(areas, on="row")
    for (level, area), g_area in L.groupby(["level", "area"]):
        members = areas[(areas.level == level) & (areas.area == area)]["row"].values
        for key, g in g_area.groupby("change"):
            fold_meds = [g[f"pct_m{k}"].median() for k in range(5)]
            med = g["oof_pct"].median()
            out.append({
                "level": level, "area": area, "change": key, "label": vocab.label(key),
                "n_area": len(members), "n_with": int(state[key][members].sum()), "n_without": len(g),
                "lift_pct": med, "lift_usd": g["oof_usd"].median(),
                "lift_pct_q25": g["oof_pct"].quantile(0.25), "lift_pct_q75": g["oof_pct"].quantile(0.75),
                "share_positive": (g["oof_pct"] > 0).mean(),
                "fold_min_pct": min(fold_meds), "fold_max_pct": max(fold_meds),
                "sign_agree": int(sum(np.sign(m) == np.sign(med) for m in fold_meds)),
                "raw_gap_pct": raw_gap(A, X, y, members, state[key]),
            })
    return pd.DataFrame(out)


def area_table(A, X, zone_names):
    """(row, level, area) for every listing at three levels: Boston, zone, neighbourhood."""
    zone_cols = [c for c in X.columns if c.startswith("zone_")]
    zone = X[zone_cols].values.argmax(1)
    n = len(X)
    return pd.concat([
        pd.DataFrame({"row": range(n), "level": "boston", "area": "Boston"}),
        pd.DataFrame({"row": range(n), "level": "zone", "area": [zone_names[str(z)] for z in zone]}),
        pd.DataFrame({"row": range(n), "level": "neighbourhood", "area": A["neighbourhood"].values}),
    ], ignore_index=True)


def main():
    A, X, y, groups, zones, bundle = em.load_training_data()
    models, fold_of = fit_fold_models(X, y, groups)
    L = lifts(models, fold_of, X)
    areas = area_table(A, X, bundle["quarterly"]["zones"]["names"])
    S = summarise(A, X, y, L, areas)
    S.round(4).to_csv(OUT, index=False)
    print("wrote", OUT, len(S), "rows")


if __name__ == "__main__":
    main()
