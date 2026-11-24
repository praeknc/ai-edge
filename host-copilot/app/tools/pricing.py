"""Pricing tools: predict_price and what_if.

Every tool returns {"component": <UI component name>, "data": {...}}. Dollar amounts are rounded to
whole dollars so the agent's narration can quote them exactly.
"""
from itertools import combinations
from math import factorial

from model import vocab
from model.loader import get_model

CAVEAT = ("Estimates show what similar Boston listings charge, learned from 7,039 listings. "
          "They are statistical associations, not guaranteed effects of a change.")
MAX_CHANGES = 8   # what_if enumerates all 2^k combinations of the requested changes


def _money(v):
    """Round a dollar amount for display and narration."""
    return int(round(float(v)))


def _priced(p):
    """{price, low, high} rounded to whole dollars."""
    return {"price": _money(p["price"]), "low": _money(p["low"]), "high": _money(p["high"])}


def _summary(L):
    """Short description of a model listing for the UI header."""
    return {"neighbourhood": L["neighbourhood"], "room_type": L["room_type"], "bedrooms": L["bedrooms"],
            "accommodates": L["accommodates"], "season": "summer" if L["quarter"] == "2026Q2" else "fall",
            "amenities": [k for k, v in L["amenities"].items() if v]}


def _interval_note(model):
    lo, hi, method = model.interval("90")
    return {"level": 0.90, "low_mult": round(lo, 3), "high_mult": round(hi, 3), "method": method}


def predict_price(listing):
    """Recommended nightly price with a 90% interval.

    listing: host listing dict (see data/sample_listing.json); only `neighbourhood` is required.
    The interval multiplies the prediction by the 5th and 95th percentiles of out-of-fold errors
    on held-out hosts, so roughly 9 in 10 comparable listings price inside it."""
    model = get_model()
    L, filled = model.complete_listing(listing)
    p = _priced(model.price([L])[0])
    data = {**p, "interval": _interval_note(model), "listing": _summary(L), "filled_fields": filled, "caveat": CAVEAT}
    current = listing.get("current_price")
    if current:
        data["current_price"] = _money(current)
        data["current_vs_range"] = ("below" if current < p["low"] else "above" if current > p["high"] else "within")
        data["current_gap"] = _money(current - p["price"])
    return {"component": "price_card", "data": data}


def _apply(L, adds, removes):
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


def _shapley(values, k):
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


def what_if(listing, add=None, remove=None):
    """Price after adding and/or removing amenities, and how much each change contributes.

    Contributions are exact Shapley values over the requested changes: each change's average
    effect over every order in which the changes could be made. They add up to the total change.
    `alone` is the effect of that one change by itself. Amenities the model does not know are
    returned under `unsupported` with the closest supported suggestions."""
    model = get_model()
    L, filled = model.complete_listing(listing)
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
    variants = [_apply(L, [changes[i][1] for i in S if changes[i][0] == "add"],
                       [changes[i][1] for i in S if changes[i][0] == "remove"]) for S in subsets]
    prices = model.price(variants)
    value = {S: p["price"] for S, p in zip(subsets, prices)}
    base, new = prices[0], prices[-1]
    phi = _shapley(value, k)
    contributions = [{"amenity": key, "label": vocab.label(key), "action": action, "contribution": _money(phi[i]),
                      "alone": _money(value[frozenset([i])] - value[frozenset()])}
                     for i, (action, key) in enumerate(changes)]
    contributions.sort(key=lambda c: -abs(c["contribution"]))
    final = [{"key": key, "label": vocab.label(key)} for key, on in _apply(L, [c[1] for c in changes if c[0] == "add"],
             [c[1] for c in changes if c[0] == "remove"])["amenities"].items() if on]
    data = {
        "base": _priced(base), "new": _priced(new),
        "delta": _priced(new)["price"] - _priced(base)["price"],   # difference of the two prices as shown
        "delta_pct": round(100 * (new["price"] / base["price"] - 1), 1),
        "contributions": contributions,
        "changes": [{"action": a, "amenity": key, "label": vocab.label(key)} for a, key in changes],
        "already_present": [vocab.label(x) for x in already], "not_present": [vocab.label(x) for x in missing],
        "unsupported": unk_add + unk_rem,
        "amenities_after": final,
        "interval": _interval_note(model), "listing": _summary(L), "filled_fields": filled,
        "method": "Shapley values over the requested changes (exact, all 2^k combinations scored by the model).",
        "caveat": CAVEAT,
    }
    return {"component": "whatif_panel", "data": data}
