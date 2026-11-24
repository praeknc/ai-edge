"""Amenity costs from data/amenity_costs.csv.

ALL COSTS ARE PLACEHOLDER ASSUMPTIONS for a student project, not quotes. The midpoint of the
low-high range is used for payback; monthly_cost (power, subscriptions, cleaning) is subtracted
from the monthly revenue lift.
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd

from model import vocab

COSTS_CSV = Path(__file__).resolve().parents[2] / "data" / "amenity_costs.csv"


@lru_cache(maxsize=1)
def cost_table():
    """The cost table with a midpoint column."""
    t = pd.read_csv(COSTS_CSV)
    t["cost_mid"] = (t["cost_low"] + t["cost_high"]) / 2
    return t


def cost_of(key):
    """Cost row (dict) for a model amenity key, or None if the table has no cost for it."""
    t = cost_table()
    rows = t[t["model_key"] == key]
    return None if rows.empty else rows.iloc[0].to_dict()


def payback_months(cost, lift_per_night, nights_per_year, monthly_cost=0.0):
    """Months until the upfront cost is earned back: cost / (lift x nights per month - monthly cost).
    0 when there is no upfront cost and the net gain is positive; None when the net gain is not positive."""
    monthly_net = lift_per_night * nights_per_year / 12 - monthly_cost
    if monthly_net <= 0:
        return None
    return round(cost / monthly_net, 1)


def get_amenity_cost(name):
    """Look up the assumed cost of an amenity by name ('hot tub', 'fast wifi', 'am_ev_charger')."""
    t = cost_table()
    key = vocab.resolve(name)
    rows = t[t["model_key"] == key] if key else t[t["name"].str.lower() == str(name).lower().strip()]
    if rows.empty:
        return {"component": "cost_card", "data": {
            "name": str(name), "found": False, "in_model": key is not None,
            "message": "No cost assumption for this amenity in data/amenity_costs.csv."}}
    r = rows.iloc[0]
    return {"component": "cost_card", "data": {
        "name": r["name"], "found": True, "model_key": key, "in_model": key is not None,
        "cost_low": int(r["cost_low"]), "cost_high": int(r["cost_high"]), "cost_mid": int(r["cost_mid"]),
        "monthly_cost": int(r["monthly_cost"]), "notes": r["notes"], "assumption": True}}
