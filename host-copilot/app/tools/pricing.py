"""Pricing tools: predict_price and what_if.

Every tool returns {"component": <UI component name>, "data": {...}}. Dollar amounts are rounded to
whole dollars so the agent's narration can quote them exactly.
"""
from itertools import combinations
from math import factorial

from model import effects, vocab
from model.loader import get_model
from app.tools.costs import cost_of, payback_months

CAVEAT = ("Estimates show what similar Boston listings charge, learned from 7,039 listings. "
          "They are statistical associations, not guaranteed effects of a change.")
MAX_CHANGES = 8          # what_if enumerates all 2^k combinations of the requested changes
NIGHTS_PER_YEAR = 200


def money(v):
    """Round a dollar amount for display and narration."""
    return int(round(float(v)))


def priced(p):
    """{price, low, high} rounded to whole dollars."""
    return {"price": money(p["price"]), "low": money(p["low"]), "high": money(p["high"])}


def interval_note(model):
    """How the 90% interval is made (shown under every price)."""
    lo, hi, method = model.interval("90")
    return {"level": 0.90, "low_mult": round(lo, 3), "high_mult": round(hi, 3), "method": method}


def listing_context(listing):
    """Complete a host listing and find its area: (model, model listing, filled fields, zone name)."""
    model = get_model()
    L, filled = model.complete_listing(listing)
    zone = model.zone_name(model.features([L]))[0]
    return model, L, filled, zone


def summary(L, zone):
    """Short description of a model listing for the UI header."""
    return {"neighbourhood": L["neighbourhood"], "zone": zone, "room_type": L["room_type"], "bedrooms": L["bedrooms"],
            "accommodates": L["accommodates"], "season": "summer" if L["quarter"] == "2026Q2" else "fall",
            "amenities": [k for k, v in L["amenities"].items() if v]}


def apply_changes(L, adds, removes):
    """Copy of a model listing with amenity flags switched on/off; the amenity line count follows."""
    new = {**L, "amenities": dict(L["amenities"])}
    for k in adds:
        new["amenities"][k] = True
        if k == "am_central_ac":
            new["amenities"]["am_any_ac"] = True
    for k in removes:
        new["amenities"][k] = False
    new["am_count"] = max(L["am_count"] + len(adds) - len(removes), sum(new["amenities"].values()))
    return new


def area_opportunities(L, zone):
    """Amenities the listing lacks that a host can add and that move prices substantially in its area."""
    out = []
    for key in effects.substantial_in_area(L["neighbourhood"], zone):
        if not L["amenities"].get(key) and vocab.kind(key) != "building":
            e = effects.area_effect(key, L["neighbourhood"], zone)
            out.append({"key": key, "label": vocab.label(key), "kind": vocab.kind(key),
                        "area_lift_usd": money(e["lift_usd"]), "area": e["area"]})
    return sorted(out, key=lambda o: -o["area_lift_usd"])


def predict_price(listing):
    """Recommended nightly price with a 90% interval.

    listing: host listing dict (see data/sample_listing.json); only `neighbourhood` is required.
    The interval multiplies the prediction by the 5th and 95th percentiles of out-of-fold errors
    on held-out hosts, so roughly 9 in 10 comparable listings price inside it."""
    model, L, filled, zone = listing_context(listing)
    p = priced(model.price([L])[0])
    data = {**p, "interval": interval_note(model), "listing": summary(L, zone), "filled_fields": filled,
            "area_opportunities": area_opportunities(L, zone), "caveat": CAVEAT}
    current = listing.get("current_price")
    if current:
        data["current_price"] = money(current)
        data["current_vs_range"] = ("below" if current < p["low"] else "above" if current > p["high"] else "within")
        data["current_gap"] = money(current - p["price"])
    return {"component": "price_card", "data": data}


def shapley(values, k):
    """Exact Shapley value of each change from v(S) for every subset S (dict: frozenset -> price).
    phi_i = sum over S not containing i of |S|!(k-|S|-1)!/k! * (v(S + i) - v(S))."""
    phi = []
    for i in range(k):
        total = 0.0
        others = [j for j in range(k) if j != i]
        for r in range(k):
            w = factorial(r) * factorial(k - r - 1) / factorial(k)
            for S in combinations(others, r):
                total += w * (values[frozenset(S) | {i}] - values[frozenset(S)])
        phi.append(total)
    return phi


def what_if(listing, add=None, remove=None, nights_per_year=NIGHTS_PER_YEAR):
    """Price after adding and/or removing amenities, and how much each change contributes.

    * contributions: exact Shapley values over the requested changes for THIS listing (each change's
      average effect over every order of making the changes); they add up to the total change.
    * Only changes that are substantial in the listing's area are listed one by one; the others are
      summed into one "no reliable effect here" line with the reason for each (see model/effects.py).
    * Amenities the model does not know are returned under `unsupported` with suggestions."""
    model, L, filled, zone = listing_context(listing)
    add_keys, unk_add = vocab.resolve_many(add)
    rem_keys, unk_rem = vocab.resolve_many(remove)
    present = {k for k, v in L["amenities"].items() if v}
    already = [k for k in add_keys if k in present]
    missing = [k for k in rem_keys if k not in present]
    changes = [("add", k) for k in add_keys if k not in present] + [("remove", k) for k in rem_keys if k in present]
    if len(changes) > MAX_CHANGES:
        raise ValueError(f"at most {MAX_CHANGES} amenity changes at a time")

    k = len(changes)
    subsets = [frozenset(S) for r in range(k + 1) for S in combinations(range(k), r)]
    variants = [apply_changes(L, [changes[i][1] for i in S if changes[i][0] == "add"],
                              [changes[i][1] for i in S if changes[i][0] == "remove"]) for S in subsets]
    prices = model.price(variants)
    value = {S: p["price"] for S, p in zip(subsets, prices)}
    base, new = priced(prices[0]), priced(prices[-1])
    phi = shapley(value, k)

    listed, minor = [], []
    for i, (action, key) in enumerate(changes):
        e = effects.area_effect(key, L["neighbourhood"], zone)
        item = {"amenity": key, "label": vocab.label(key), "action": action, "kind": vocab.kind(key),
                "contribution": money(phi[i]), "alone": money(value[frozenset([i])] - value[frozenset()]),
                "area": e["area"], "area_lift_usd": None if e["lift_usd"] is None else money(e["lift_usd"]),
                "substantial_in_area": e["substantial"], "reason": e["reason"]}
        (listed if e["substantial"] else minor).append(item)
    listed.sort(key=lambda c: -abs(c["contribution"]))

    adds = [key for a, key in changes if a == "add"]
    costs = [cost_of(key) for key in adds]
    known = [c for c in costs if c]
    upfront = money(sum(c["cost_mid"] for c in known)) if known else 0
    monthly = sum(c["monthly_cost"] for c in known)
    lift = new["price"] - base["price"]
    final = apply_changes(L, adds, [key for a, key in changes if a == "remove"])["amenities"]
    data = {
        "base": base, "new": new, "delta": lift, "delta_pct": round(100 * (new["price"] / base["price"] - 1), 1),
        "contributions": listed,
        "no_reliable_effect": {"items": minor, "contribution": money(sum(phi[changes.index((m["action"], m["amenity"]))]
                                                                         for m in minor))} if minor else None,
        "upfront_cost": upfront, "monthly_cost": int(monthly),
        "costs_missing_for": [vocab.label(key) for key, c in zip(adds, costs) if not c],
        "payback_months": payback_months(upfront, lift, nights_per_year, monthly) if adds else None,
        "nights_per_year": nights_per_year,
        "changes": [{"action": a, "amenity": key, "label": vocab.label(key)} for a, key in changes],
        "already_present": [vocab.label(x) for x in already], "not_present": [vocab.label(x) for x in missing],
        "unsupported": unk_add + unk_rem,
        "amenities_after": [{"key": key, "label": vocab.label(key)} for key, on in final.items() if on],
        "area_opportunities": area_opportunities(L, zone),
        "interval": interval_note(model), "listing": summary(L, zone), "filled_fields": filled,
        "rule": effects.RULE_TEXT, "costs_are_assumptions": True,
        "method": "Shapley values over the requested changes (exact, all 2^k combinations scored by the model).",
        "caveat": CAVEAT,
    }
    return {"component": "whatif_panel", "data": data}
