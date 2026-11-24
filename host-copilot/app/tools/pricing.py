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
            "amenities": [k for k, v in L["amenities"].items() if v],
            "minimum_nights": L["minimum_nights"], "superhost": L["superhost"], "license": L["license"],
            "identity_verified": L["identity_verified"]}


SETTABLE = {"minimum_nights", "superhost", "license", "identity_verified"}
MARKET_SWITCH = {"lv_monthly", "lv_nightly"}
MARKET_NOTE = ("Switching between monthly and nightly stays changes the market, not just the price: occupancy, "
               "turnover costs and the rules for short stays all differ, so the nightly price is not comparable revenue.")
REGULATORY_NOTE = ("Whether a listing can register, and what it costs, depends on city rules. This tool does not give "
                   "legal or regulatory advice.")


def apply_changes(L, adds=(), removes=(), settings=None):
    """Copy of a model listing with amenity flags switched on/off and lever fields set
    (settings: {field: value}); the amenity line count follows the amenity changes."""
    new = {**L, "amenities": dict(L["amenities"])}
    for k in adds:
        new["amenities"][k] = True
        if k == "am_central_ac":
            new["amenities"]["am_any_ac"] = True
    for k in removes:
        new["amenities"][k] = False
    new["am_count"] = max(L["am_count"] + len(adds) - len(removes), sum(new["amenities"].values()))
    new.update(settings or {})
    return new


def apply_keys(L, keys):
    """Apply a bundle of change keys: amenity keys are added, lever keys set their fields."""
    settings = {}
    for k in keys:
        if k in vocab.LEVERS:
            settings.update(vocab.LEVERS[k][2])
    return apply_changes(L, [k for k in keys if k in vocab.AMENITIES], [], settings)


def area_opportunities(L, zone):
    """Amenities the listing lacks, and levers it could use, that a host can choose and that raise
    prices substantially in its area (building features, earned status and market switches excluded)."""
    out = []
    for key in effects.substantial_in_area(L["neighbourhood"], zone):
        if key in vocab.LEVERS:
            if key in MARKET_SWITCH or vocab.kind(key) == "earned" or not lever_applies(L, key):
                continue
        elif L["amenities"].get(key) or vocab.kind(key) == "building":
            continue
        e = effects.area_effect(key, L["neighbourhood"], zone)
        out.append({"key": key, "label": vocab.label(key), "kind": vocab.kind(key),
                    "area_lift_usd": money(e["lift_usd"]), "area": e["area"]})
    return sorted(out, key=lambda o: -o["area_lift_usd"])


def lever_applies(L, key):
    """Can this listing make the lever move (is it in the band the move starts from)?"""
    mn = L["minimum_nights"]
    return {"lv_min_nights_1": 2 <= mn <= 27, "lv_min_nights_2": 3 <= mn <= 27, "lv_min_nights_5": mn <= 2,
            "lv_monthly": mn <= 27, "lv_nightly": mn >= 28, "lv_superhost": not L["superhost"],
            "lv_licensed": L["license"] != "licensed", "lv_verified": not L["identity_verified"]}[key]


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


def _lever_change(L, field, value):
    """One requested lever setting -> change dict (key may be None when no checked move matches)."""
    if field not in SETTABLE:
        raise ValueError(f"settings can change only {sorted(SETTABLE)}")
    if field == "minimum_nights":
        value = int(value)
        if not 1 <= value <= 365:
            raise ValueError("minimum_nights must be between 1 and 365")
    if field == "license" and value not in ("licensed", "exempt", "none"):
        raise ValueError("license must be 'licensed', 'exempt' or 'none'")
    if field in ("superhost", "identity_verified"):
        value = bool(value)
    key = vocab.lever_for(field, L[field], value)
    label = vocab.label(key) if key else f"{field.replace('_', ' ')}: {L[field]} -> {value}"
    if key == "lv_min_nights_5" or (key and field == "minimum_nights" and vocab.LEVERS[key][2]["minimum_nights"] != value):
        label = f"Minimum stay {value} night{'s' if value != 1 else ''}"
    return {"action": "set", "key": key, "label": label, "settings": {field: value}}


def what_if(listing, add=None, remove=None, settings=None, nights_per_year=NIGHTS_PER_YEAR):
    """Price after adding/removing amenities and changing levers, and what each change contributes.

    settings: {field: value} for minimum_nights, superhost, license, identity_verified.
    * contributions: exact Shapley values over the requested changes for THIS listing (each change's
      average effect over every order of making the changes); they add up to the total change.
    * Only changes that are substantial in the listing's area are listed one by one; the others are
      summed into one "no reliable effect here" line with the reason for each (see model/effects.py).
    * Amenities the model does not know are returned under `unsupported` with suggestions."""
    model, L, filled, zone = listing_context(listing)
    add_keys, unk_add = vocab.resolve_many(add)
    rem_keys, unk_rem = vocab.resolve_many(remove)
    present = {k for k, v in L["amenities"].items() if v}
    already = [vocab.label(k) for k in add_keys if k in present]
    missing = [vocab.label(k) for k in rem_keys if k not in present]
    changes = ([{"action": "add", "key": k, "label": vocab.label(k)} for k in add_keys if k not in present]
               + [{"action": "remove", "key": k, "label": vocab.label(k)} for k in rem_keys if k in present])
    for field, value in (settings or {}).items():
        ch = _lever_change(L, field, value)
        if L[field] == ch["settings"][field]:
            already.append(f"{field.replace('_', ' ')} = {value}")
        else:
            changes.append(ch)
    if len(changes) > MAX_CHANGES:
        raise ValueError(f"at most {MAX_CHANGES} changes at a time")

    def variant(S):
        chosen = [changes[i] for i in S]
        merged = {}
        for c in chosen:
            merged.update(c.get("settings", {}))
        return apply_changes(L, [c["key"] for c in chosen if c["action"] == "add"],
                             [c["key"] for c in chosen if c["action"] == "remove"], merged)

    k = len(changes)
    subsets = [frozenset(S) for r in range(k + 1) for S in combinations(range(k), r)]
    prices = model.price([variant(S) for S in subsets])
    value = {S: p["price"] for S, p in zip(subsets, prices)}
    base, new = priced(prices[0]), priced(prices[-1])
    phi = shapley(value, k)

    listed, minor = [], []
    for i, c in enumerate(changes):
        e = (effects.area_effect(c["key"], L["neighbourhood"], zone) if c["key"] else
             {"area": L["neighbourhood"], "lift_usd": None, "substantial": False, "reason": "no area check for this move"})
        item = {"key": c["key"], "label": c["label"], "action": c["action"], "kind": vocab.kind(c["key"]) if c["key"] else "policy",
                "contribution": money(phi[i]), "alone": money(value[frozenset([i])] - value[frozenset()]),
                "area": e["area"], "area_lift_usd": None if e["lift_usd"] is None else money(e["lift_usd"]),
                "substantial_in_area": e["substantial"], "reason": e["reason"],
                "market_switch": c["key"] in MARKET_SWITCH}
        (listed if e["substantial"] else minor).append((i, item))
    listed.sort(key=lambda t: -abs(t[1]["contribution"]))

    adds = [c["key"] for c in changes if c["action"] == "add"]
    costs = [cost_of(c["key"]) for c in changes if c["action"] == "add" or (c["action"] == "set" and c["key"])]
    known = [c for c in costs if c]
    upfront = money(sum(c["cost_mid"] for c in known)) if known else 0
    monthly = sum(c["monthly_cost"] for c in known)
    lift = new["price"] - base["price"]
    final = variant(range(k))
    data = {
        "base": base, "new": new, "delta": lift, "delta_pct": round(100 * (new["price"] / base["price"] - 1), 1),
        "contributions": [item for _, item in listed],
        "no_reliable_effect": {"items": [item for _, item in minor],
                               "contribution": money(sum(phi[i] for i, _ in minor))} if minor else None,
        "upfront_cost": upfront, "monthly_cost": int(monthly),
        "costs_missing_for": [vocab.label(a) for a in adds if not cost_of(a)],
        "payback_months": payback_months(upfront, lift, nights_per_year, monthly) if changes else None,
        "nights_per_year": nights_per_year,
        "changes": [{"action": c["action"], "key": c["key"], "label": c["label"], **({"settings": c["settings"]} if "settings" in c else {})}
                    for c in changes],
        "already_present": already, "not_present": missing,
        "unsupported": unk_add + unk_rem,
        "amenities_after": [{"key": key, "label": vocab.label(key)} for key, on in final["amenities"].items() if on],
        "levers_after": {f: final[f] for f in sorted(SETTABLE)},
        "area_opportunities": area_opportunities(L, zone),
        "notes": ([MARKET_NOTE] if any(c["key"] in MARKET_SWITCH for c in changes) else [])
                 + ([REGULATORY_NOTE] if any(c["key"] == "lv_licensed" for c in changes) else []),
        "interval": interval_note(model), "listing": summary(L, zone), "filled_fields": filled,
        "rule": effects.RULE_TEXT, "costs_are_assumptions": True,
        "method": "Shapley values over the requested changes (exact, all 2^k combinations scored by the model).",
        "caveat": CAVEAT,
    }
    return {"component": "whatif_panel", "data": data}
