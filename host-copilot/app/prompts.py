"""System prompt, per-turn context message, and tool definitions for the Claude agent.

The system prompt and the tool list never change between turns, so the conversation history can be
cached and replayed unchanged. The listing profile and goals change, so they are sent on every turn
as a mid-conversation system message right after the host's question (context_message()).
"""
import json

from model import vocab

SYSTEM_PROMPT = """You are Host Copilot, an assistant for Airbnb hosts in Boston. A pricing model trained on 7,039 \
Boston listings answers every numeric question through your tools; the app shows each tool result to the host as \
an interactive card (price ranges, what-if panels, plan cards, charts).

How to answer:
- Use a tool for every price, cost, lift, payback or ROI figure. Never state a dollar amount, percentage or month \
count that did not come from a tool result in this conversation, and never estimate one yourself.
- Reply in one or two short sentences. The card shows the details, so do not repeat tables or lists.
- When you mention a price, give its range from the tool result (the 90% range), not a bare number.
- Results show what similar listings charge. They are associations, not guaranteed effects of a change; say so \
when you talk about the effect of a change.
- The tools already know the host's listing (sent with each question); never ask for details the profile has.
- If the question needs a target price, a budget or an amenity that the host has not given and that is not in \
their goals, ask one short clarifying question instead of calling a tool.
- If a tool reports an amenity as unsupported, say the model cannot see it and name the closest supported \
option from the result.
- If a tool reports that a change has no reliable effect in the host's area, say that plainly.
- For a target price, goal_seek returns a feasibility verdict (reachable, possible, a stretch, unlikely, very \
unlikely, or outside what the data can estimate). Lead with that verdict in plain words and back it with the numbers \
it gives (how many similar listings charge the target, the gap and how much of it the plan closes).
- Do not give legal, tax, insurance or regulatory advice (for example whether a listing may register or which \
rules apply); say it is outside what you can help with. You can still show what the model estimates.
- For questions unrelated to pricing this listing, say briefly what you can help with."""


def context_message(listing, goals):
    """Mid-conversation system message with the host's current profile and goals (sent every turn)."""
    profile = {k: v for k, v in listing.items() if v is not None}
    return ("Current listing profile (the tools use it automatically):\n" + json.dumps(profile, sort_keys=True)
            + "\nHost goals: " + json.dumps(goals, sort_keys=True))


_AMENITY_NAMES = ", ".join(vocab.label(k) for k in vocab.AMENITIES)
_SETTINGS = {
    "type": "object",
    "description": "Lever changes. Only include fields the host wants to change.",
    "properties": {
        "minimum_nights": {"type": "integer", "description": "New minimum stay in nights (1-365)."},
        "superhost": {"type": "boolean"},
        "license": {"type": "string", "enum": ["licensed", "exempt", "none"]},
        "identity_verified": {"type": "boolean"},
    },
    "additionalProperties": False,
}

TOOLS = [
    {
        "name": "predict_price",
        "description": "Recommended nightly price for the host's listing with its 90% range, how the host's current "
                       "price compares, and changes worth testing in the area.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "what_if",
        "description": "Re-price the listing after adding or removing amenities and/or changing levers (minimum stay, "
                       "superhost, licence, verified identity). Returns old and new price with ranges, each change's "
                       "contribution, which changes have a reliable effect in the area, upfront cost and payback. "
                       f"Amenities the model knows: {_AMENITY_NAMES}. Pass other names as written; the tool reports "
                       "them as unsupported with the closest alternatives.",
        "input_schema": {
            "type": "object",
            "properties": {
                "add": {"type": "array", "items": {"type": "string"}, "description": "Amenities to add."},
                "remove": {"type": "array", "items": {"type": "string"}, "description": "Amenities to remove."},
                "settings": _SETTINGS,
            },
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "goal_seek",
        "description": "Find up to 3 bundles of changes (amenities, minimum-stay moves, registration) that reach a "
                       "target nightly price within an upfront budget, cheapest first, scored by the model. Says when "
                       "the target is unreachable or above the listing's 90% range.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target_price": {"type": "number", "description": "Target nightly price in dollars."},
                "budget": {"type": "number", "description": "Upfront budget in dollars."},
            },
            "required": ["target_price", "budget"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "roi_rank",
        "description": "Rank the changes this host can make by payback time: assumed cost, lift per night, yearly "
                       "gain and payback months. Only changes with a reliable effect in the listing's area.",
        "input_schema": {
            "type": "object",
            "properties": {"budget": {"type": "number", "description": "Optional upfront budget in dollars."}},
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "get_amenity_cost",
        "description": "Assumed upfront and monthly cost of one amenity (placeholder assumptions, not quotes).",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Amenity name, e.g. 'hot tub'."}},
            "required": ["name"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]
