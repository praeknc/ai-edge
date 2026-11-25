"""Other things a host controls, beyond amenities and minimum stay: do they move the price?

Same honest test as analysis/change_effects.py: for every listing that could make the change, apply it to
the listing's features and re-price it with the cross-validation model that never saw the host; then check
the direction with all 5 fold models and count the zones where the effect is substantial (median lift
>= 3% and >= $8, same direction in all 5 models, >= 80% of the zone's listings move that way).

Output: model/artifacts/more_levers.csv (Boston-wide) and more_levers_by_zone.csv
Run:    python analysis/more_levers.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "model"))
import change_effects as ce        # noqa: E402  (fold models, area table)
import export_model as em          # noqa: E402

OUT = ROOT.parent / "model" / "artifacts"
SCORES = ["rs_rating", "rs_accuracy", "rs_cleanliness", "rs_checkin", "rs_communication", "rs_location", "rs_value"]


def setf(**values):
    def f(X):
        Z = X.copy()
        for k, v in values.items():
            Z[k] = v(Z) if callable(v) else v
        return Z
    return f


def set_night(day):
    """Price a different night of the week (0 = Monday ... 6 = Sunday)."""
    return setf(**{f"quote_dow_{d}": int(d == day) for d in range(7)})


# key -> (label, group, who can make it, transform)
LEVERS = {
    "title_luxury": ("Title words like 'modern', 'renovated', 'luxury'", "Listing text",
                     lambda X: X.kw_luxury == 0, setf(kw_luxury=1)),
    "title_view": ("Title mentions a view or rooftop", "Listing text", lambda X: X.kw_view == 0, setf(kw_view=1)),
    "title_parking": ("Title mentions parking", "Listing text", lambda X: X.kw_parking == 0, setf(kw_parking=1)),
    "title_amenities": ("Title mentions gym / pool / amenities", "Listing text",
                        lambda X: X.kw_amenities == 0, setf(kw_amenities=1)),
    "desc_full": ("Full description (about 500 characters)", "Listing text",
                  lambda X: X.desc_len < 300, setf(desc_len=500)),
    "guests_plus2": ("Sleep 2 more guests (sofa bed: +2 guests, +1 bed)", "Capacity",
                     lambda X: X.accommodates <= 6, setf(accommodates=lambda Z: Z.accommodates + 2,
                                                         beds=lambda Z: Z.beds + 1)),
    "beds_plus1": ("One more bed, same guests", "Capacity", lambda X: X.beds.notna(),
                   setf(beds=lambda Z: Z.beds + 1)),
    "private_bath": ("Private instead of shared bathroom", "Capacity", lambda X: X.baths_shared == 1,
                     setf(baths_shared=0)),
    "rating_up": ("Ratings 0.1 higher (all 7 scores, max 5)", "Guest experience",
                  lambda X: X.rs_rating.notna() & (X.rs_rating <= 4.85),
                  setf(**{s: (lambda Z, s=s: np.minimum(5.0, Z[s] + 0.1)) for s in SCORES})),
    "open_90_more": ("Calendar open 90 more days a year", "Calendar",
                     lambda X: X.availability_365 <= 275, setf(availability_365=lambda Z: Z.availability_365 + 90)),
    "night_saturday": ("Saturday night instead of a weekday (Mon-Thu)", "Pricing by night",
                       lambda X: X[[f"quote_dow_{d}" for d in range(4)]].sum(axis=1) == 1, set_night(5)),
    "night_friday": ("Friday night instead of a weekday (Mon-Thu)", "Pricing by night",
                     lambda X: X[[f"quote_dow_{d}" for d in range(4)]].sum(axis=1) == 1, set_night(4)),
    "summer": ("Summer (June) instead of fall (September)", "Season", lambda X: X.q_sep == 1, setf(q_sep=0)),
}


def main():
    A, X, y, groups, zones, bundle = em.load_training_data()
    models, fold_of = ce.fit_fold_models(X, y, groups)
    zone_cols = [c for c in X.columns if c.startswith("zone_")]
    zone = np.array([bundle["quarterly"]["zones"]["names"][str(z)] for z in X[zone_cols].values.argmax(1)])
    base = np.column_stack([np.exp(m.predict(X)) for m in models])
    rows, zrows = [], []
    for key, (label, group, can, apply) in LEVERS.items():
        idx = np.where(can(X))[0]
        on = np.column_stack([np.exp(m.predict(apply(X.iloc[idx]))) for m in models])
        pct, usd = on / base[idx] - 1, on - base[idx]
        own = fold_of[idx]
        oof_pct, oof_usd = pct[np.arange(len(idx)), own], usd[np.arange(len(idx)), own]

        def summary(mask):
            med = np.median(oof_pct[mask])
            fold_meds = [np.median(pct[mask, k]) for k in range(5)]
            share = np.mean(oof_pct[mask] > 0) if med >= 0 else np.mean(oof_pct[mask] < 0)
            return {"n": int(mask.sum()), "lift_pct": med, "lift_usd": np.median(oof_usd[mask]),
                    "q25_pct": np.quantile(oof_pct[mask], .25), "q75_pct": np.quantile(oof_pct[mask], .75),
                    "same_direction_share": share, "sign_agree": sum(np.sign(m) == np.sign(med) for m in fold_meds)}

        everyone = np.ones(len(idx), bool)
        r = {"lever": key, "label": label, "group": group, **summary(everyone)}
        n_sub = 0
        for zname in np.unique(zone):
            m = zone[idx] == zname
            if m.sum() < 15:
                continue
            s = summary(m)
            sub = (abs(s["lift_pct"]) >= 0.03 and abs(s["lift_usd"]) >= 8 and s["sign_agree"] == 5
                   and s["same_direction_share"] >= 0.8)
            n_sub += sub
            zrows.append({"lever": key, "zone": zname, **s, "substantial": sub})
        r["zones_substantial"] = n_sub
        rows.append(r)
    pd.DataFrame(rows).round(4).to_csv(OUT / "more_levers.csv", index=False)
    pd.DataFrame(zrows).round(4).to_csv(OUT / "more_levers_by_zone.csv", index=False)
    print(pd.DataFrame(rows)[["label", "n", "lift_pct", "lift_usd", "q25_pct", "q75_pct", "same_direction_share",
                              "sign_agree", "zones_substantial"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
