"""Planning tools: goal_seek (reach a target price within a budget) and roi_rank (best value amenities).

Both plan only with changes that
  * the listing can still make (it lacks the amenity / its minimum nights are in the move's band),
  * a host can choose: amenities to buy, free policies, minimum-stay moves and registering a licence
    (not building features, not superhost status, which is earned, and not switching between
    monthly and nightly stays, which changes the market),
  * RAISE the price substantially in the listing's area (model/effects.py), and
  * have a cost assumption in data/amenity_costs.csv.
A bundle holds at most one minimum-stay move.
Prices of bundles are always scored by the model on the whole bundle, not by adding up single effects.
"""
from itertools import combinations

from model import effects, vocab
from app.tools.costs import cost_of, payback_months
from app.tools.pricing import (CAVEAT, MARKET_SWITCH, NIGHTS_PER_YEAR, REGULATORY_NOTE, apply_keys, interval_note,
                               lever_applies, listing_context, money, priced, summary)

MAX_POOL = 10      # candidates kept for goal_seek: 2^10 = 1,024 bundles
OTHER_LEVERS = ("Amenities explain little of Boston prices (3.6% of the model's split gain). Size (guests, bedrooms, "
                "baths), private vs shared bath, minimum nights and location matter far more.")


def exclusion_reason(L, key, e):
    """Why a change is not planned with (None if it can be)."""
    if key in vocab.LEVERS:
        if key in MARKET_SWITCH:
            return "switches between monthly and nightly stays (a different market)"
        if vocab.kind(key) == "earned":
            return "earned through guest ratings, not a direct choice"
    elif vocab.kind(key) == "building":
        return "building feature, not something to buy"
    if not e["substantial"]:
        return e["reason"]
    if e["lift_usd"] < 0:
        return "lowers prices in this area"
    if cost_of(key) is None:
        return "no cost assumption"
    return None


def candidate_pool(L, zone):
    """Changes the listing can make that raise prices substantially in its area and have a known cost,
    plus the excluded ones and why."""
    pool, excluded = [], []
    keys = [k for k in vocab.AMENITIES if not L["amenities"].get(k)] + [k for k in vocab.LEVERS if lever_applies(L, k)]
    for key in keys:
        e = effects.area_effect(key, L["neighbourhood"], zone)
        reason = exclusion_reason(L, key, e)
        if reason:
            excluded.append({"key": key, "label": vocab.label(key), "reason": reason, "kind": vocab.kind(key),
                             "substantial_in_area": e["substantial"]})
        else:
            pool.append({"key": key, "label": vocab.label(key), "kind": vocab.kind(key), "cost": cost_of(key), "effect": e})
    return pool, excluded


def valid_bundle(keys):
    """At most one minimum-stay move per bundle."""
    return sum(k.startswith("lv_min_nights") for k in keys) <= 1


def single_lifts(model, L, keys):
    """Price of the listing with each amenity added on its own, and the base price."""
    prices = model.price([L] + [apply_keys(L, [k]) for k in keys])
    return prices[0], {k: p for k, p in zip(keys, prices[1:])}


def bundle_card(keys, pool_by_key, p, base, nights_per_year):
    """One option card: amenities, price with interval, lift, cost and payback."""
    costs = [pool_by_key[k]["cost"] for k in keys]
    cost = sum(c["cost_mid"] for c in costs)
    monthly = sum(c["monthly_cost"] for c in costs)
    lift = priced(p)["price"] - base["price"]
    return {"amenities": [{"key": k, "label": vocab.label(k), "kind": vocab.kind(k)} for k in keys],
            **{"price": priced(p)}, "lift": lift, "cost": money(cost),
            "cost_low": money(sum(c["cost_low"] for c in costs)), "cost_high": money(sum(c["cost_high"] for c in costs)),
            "monthly_cost": money(monthly), "payback_months": payback_months(cost, lift, nights_per_year, monthly)}


def goal_seek(listing, target_price, budget, nights_per_year=NIGHTS_PER_YEAR):
    """Up to 3 bundles of changes (amenities and levers) that reach `target_price` (model's central price) within `budget`,
    cheapest first. If none reaches it, the 3 highest-priced bundles within budget, marked unreachable."""
    model, L, filled, zone = listing_context(listing)
    pool, excluded = candidate_pool(L, zone)
    base_p, single = single_lifts(model, L, [c["key"] for c in pool])
    base = priced(base_p)
    pool = sorted((c for c in pool if single[c["key"]]["price"] > base_p["price"]),
                  key=lambda c: -single[c["key"]]["price"])[:MAX_POOL]
    by_key = {c["key"]: c for c in pool}

    bundles = [list(S) for r in range(1, len(pool) + 1) for S in combinations(by_key, r)
               if valid_bundle(S) and sum(by_key[k]["cost"]["cost_mid"] for k in S) <= budget]
    prices = model.price([apply_keys(L, b) for b in bundles]) if bundles else []
    cards = [bundle_card(b, by_key, p, base, nights_per_year) for b, p in zip(bundles, prices)]
    reach = sorted([c for c in cards if c["price"]["price"] >= target_price], key=lambda c: (c["cost"], -c["price"]["price"]))
    best = sorted(cards, key=lambda c: (-c["price"]["price"], c["cost"]))

    status = ("already_met" if base["price"] >= target_price else
              "reachable" if reach else "unreachable")
    data = {
        "target_price": money(target_price), "budget": money(budget), "nights_per_year": nights_per_year,
        "base": base, "status": status,
        "target_above_range": target_price > base["high"],
        "options": (reach if reach else best)[:3],
        "best_possible": best[0]["price"]["price"] if best else base["price"],
        "candidates": [{"key": c["key"], "label": c["label"], "alone_lift": money(single[c["key"]]["price"] - base_p["price"]),
                        "cost": money(c["cost"]["cost_mid"])} for c in pool],
        "excluded": excluded, "bundles_scored": len(bundles),
        "notes": [REGULATORY_NOTE] if any(k == "lv_licensed" for k in by_key) else [],
        "other_levers": OTHER_LEVERS, "interval": interval_note(model), "listing": summary(L, zone),
        "filled_fields": filled, "rule": effects.RULE_TEXT, "costs_are_assumptions": True, "caveat": CAVEAT,
    }
    return {"component": "plan_cards", "data": data}


def roi_rank(listing, budget=None, nights_per_year=NIGHTS_PER_YEAR):
    """Changes a host can make (amenities and levers) ranked by payback time: cost, lift per night, yearly gain and payback months.
    Only amenities with a substantial effect in the listing's area and a positive lift for this listing."""
    model, L, filled, zone = listing_context(listing)
    pool, excluded = candidate_pool(L, zone)
    base_p, single = single_lifts(model, L, [c["key"] for c in pool])
    base = priced(base_p)
    items = []
    for c in pool:
        lift = priced(single[c["key"]])["price"] - base["price"]
        if lift <= 0:
            excluded.append({"key": c["key"], "label": c["label"], "kind": c["kind"], "substantial_in_area": True,
                             "reason": "no gain for this particular listing"})
            continue
        cost = c["cost"]
        items.append({"key": c["key"], "label": c["label"], "kind": c["kind"], "cost": money(cost["cost_mid"]),
                      "cost_low": int(cost["cost_low"]), "cost_high": int(cost["cost_high"]),
                      "monthly_cost": int(cost["monthly_cost"]), "lift_per_night": lift,
                      "yearly_gain": money(lift * nights_per_year - 12 * cost["monthly_cost"]),
                      "payback_months": payback_months(cost["cost_mid"], lift, nights_per_year, cost["monthly_cost"]),
                      "area": c["effect"]["area"], "area_lift_usd": money(c["effect"]["lift_usd"]),
                      "within_budget": budget is None or cost["cost_mid"] <= budget})
    items.sort(key=lambda i: (i["payback_months"] is None, i["payback_months"] or 0, -i["lift_per_night"]))
    data = {"base": base, "budget": None if budget is None else money(budget), "nights_per_year": nights_per_year,
            "items": items, "excluded": excluded, "other_levers": OTHER_LEVERS,
            "notes": [REGULATORY_NOTE] if any(i["key"] == "lv_licensed" for i in items) else [], "interval": interval_note(model),
            "listing": summary(L, zone), "filled_fields": filled, "rule": effects.RULE_TEXT,
            "costs_are_assumptions": True, "caveat": CAVEAT}
    return {"component": "roi_bubble", "data": data}
