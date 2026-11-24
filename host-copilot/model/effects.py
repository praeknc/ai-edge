"""Area-specific effects of changes a host controls (amenities and levers), from analysis/change_effects.py.

For each area and change the table holds the median out-of-fold price lift of making the change,
over all listings in that area that could make it, plus stability and support checks.
An effect is SUBSTANTIAL in an area when:
  * the median lift is at least 3% and at least $8 per night (up or down),
  * all 5 cross-validation models agree on its direction,
  * at least 80% of the area's listings move in that direction,
  * at least 15 listings in the area are already in the target state and at least 15 are not.
The listing's neighbourhood is used when it has that support; otherwise its zone (one of the
model's 8 location zones); otherwise the effect is reported as "not enough data".
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd

ART = Path(__file__).resolve().parent / "artifacts"
MIN_PCT, MIN_USD, MIN_SHARE, MIN_SUPPORT = 0.03, 8.0, 0.80, 15
RULE_TEXT = (f"median change at least {MIN_PCT:.0%} and ${MIN_USD:.0f}/night in the area, the same direction in all 5 "
             f"cross-validation models, at least {MIN_SHARE:.0%} of the area's listings move that way, and at least "
             f"{MIN_SUPPORT} listings with and without it there")


@lru_cache(maxsize=1)
def table():
    """The effects table indexed by (level, area, amenity)."""
    return pd.read_csv(ART / "change_effects.csv").set_index(["level", "area", "change"])


def _supported(row):
    return row["n_with"] >= MIN_SUPPORT and row["n_without"] >= MIN_SUPPORT


def is_substantial(row):
    """Apply the rule above to one table row."""
    share = row["share_positive"] if row["lift_pct"] >= 0 else 1 - row["share_positive"]
    return bool(_supported(row) and abs(row["lift_pct"]) >= MIN_PCT and abs(row["lift_usd"]) >= MIN_USD
                and row["sign_agree"] == 5 and share >= MIN_SHARE)


def area_effect(change, neighbourhood, zone):
    """Effect of one change (amenity or lever key) for a listing in this neighbourhood / zone.

    Returns {area, level, lift_usd, lift_pct, n_with, n_without, substantial, reason}."""
    t = table()
    for level, area in (("neighbourhood", neighbourhood), ("zone", zone)):
        key = (level, area, change)
        if key in t.index and _supported(t.loc[key]):
            row = t.loc[key]
            sub = is_substantial(row)
            reason = "substantial" if sub else (
                "direction not stable across models" if row["sign_agree"] < 5 else
                f"too small here (median ${row['lift_usd']:.0f}, {row['lift_pct']:.1%})")
            return {"area": area, "level": level, "lift_usd": round(float(row["lift_usd"]), 1),
                    "lift_pct": round(float(row["lift_pct"]), 4), "n_with": int(row["n_with"]),
                    "n_without": int(row["n_without"]), "substantial": sub, "reason": reason}
    return {"area": neighbourhood, "level": "none", "lift_usd": None, "lift_pct": None, "n_with": None,
            "n_without": None, "substantial": False, "reason": "not enough listings in this area to judge"}


def substantial_in_area(neighbourhood, zone, increases_only=True):
    """Change keys with a substantial effect for this neighbourhood / zone (price increases by default)."""
    out = []
    for c in table().index.get_level_values("change").unique():
        e = area_effect(c, neighbourhood, zone)
        if e["substantial"] and (e["lift_usd"] > 0 or not increases_only):
            out.append(c)
    return out
