"""Clickable suggestions: what the assistant can do (new conversation) and next steps after each answer.

A suggestion is an *action*: {"label", "tool", "args"}. Clicking it calls the tool directly with the
sidebar listing, so it never depends on how well a question is understood.
"""
import math

from model import vocab

CAPABILITIES = [
    ("Price check", "What a listing like yours charges, with a 90% range.", "💲"),
    ("Test changes", "Switch amenities, minimum stay, licence and superhost on and off; see the price move.", "🎛️"),
    ("Reach a target", "Is your target price realistic, and which changes get closest within your budget?", "🎯"),
    ("Best upgrades", "Changes that pay back fastest in your area, with cost and payback months.", "📈"),
    ("Cost lookup", "Assumed cost of an amenity (placeholder figures).", "🧾"),
]


def action(label, tool, args=None):
    return {"label": label, "tool": tool, "args": args or {}}


def _round_up(v, step=10):
    return int(math.ceil(v / step) * step)


def capability_actions(goals):
    """One action per capability, for the 'What I can do' box."""
    target = goals.get("target_price") or 600
    return [
        action("What should I charge?", "predict_price"),
        action("Open a what-if panel", "what_if"),
        action(f"Can I reach ${target:,}?", "goal_seek", {"target_price": target, "budget": goals.get("budget", 2000)}),
        action("Which upgrades pay back fastest?", "roi_rank", {"budget": goals.get("budget")}),
        action("What does a hot tub cost?", "get_amenity_cost", {"name": "hot tub"}),
    ]


def _try_change(key):
    """What-if action for one amenity or lever key."""
    if key in vocab.LEVERS:
        return action(f"What if: {vocab.label(key).lower()}?", "what_if", {"settings": dict(vocab.LEVERS[key][2])})
    return action(f"What if I add {vocab.label(key).lower()}?", "what_if", {"add": [vocab.label(key)]})


def _goal(goals, price):
    """Target to suggest: the host's goal if it is above today's price, else about 10% above today."""
    t = goals.get("target_price") or 0
    target = t if t > price else _round_up(price * 1.10)
    return action(f"Can I reach ${target:,}?", "goal_seek", {"target_price": target, "budget": goals.get("budget", 2000)})


def follow_ups(block, goals):
    """Up to 3 next steps that make sense after this answer."""
    c, d = block["component"], block["data"]
    roi = action("Which upgrades pay back fastest?", "roi_rank", {"budget": goals.get("budget")})
    out = []
    if c == "price_card":
        out += [_try_change(o["key"]) for o in d.get("area_opportunities", [])[:1]]
        out += [_goal(goals, d["price"]), roi]
    elif c == "whatif_panel":
        for u in d.get("unsupported", [])[:1]:
            out += [_try_change(s["key"]) for s in u["suggestions"][:1]]
        out += [_goal(goals, d["new"]["price"]), roi]
    elif c == "plan_cards":
        f, base = d["feasibility"], d["base"]["price"]
        if f["verdict"] in ("stretch", "unlikely", "very_unlikely", "out_of_range"):
            easier = _round_up(base + max(10, (d["target_price"] - base) / 2))
            out.append(action(f"What about ${easier:,} instead?", "goal_seek",
                              {"target_price": easier, "budget": d["budget"]}))
        out += [roi, action("Open a what-if panel", "what_if")]
    elif c == "roi_bubble":
        out += [_try_change(i["key"]) for i in d["items"][:1]]
        out += [_goal(goals, d["base"]["price"]), action("What should I charge?", "predict_price")]
    elif c == "cost_card":
        if d.get("found") and d.get("model_key"):
            out.append(_try_change(d["model_key"]))
        out += [roi]
    return out[:3] or capability_actions(goals)[:3]
