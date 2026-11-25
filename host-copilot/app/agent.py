"""The agent: one host question in, short narration plus UI blocks out.

respond() returns {"text": str, "blocks": [{"tool", "args", "component", "data"}]}.
Step 4: mock mode only (keyword routing, app/mock_llm.py). Step 5 adds the Claude tool-use loop.
"""
import os

from app import mock_llm
from app.tools import get_amenity_cost, goal_seek, predict_price, roi_rank, what_if

TOOLS = {"predict_price": predict_price, "what_if": what_if, "goal_seek": goal_seek, "roi_rank": roi_rank,
         "get_amenity_cost": get_amenity_cost}
NEEDS_LISTING = {"predict_price", "what_if", "goal_seek", "roi_rank"}
USES_NIGHTS = {"what_if", "goal_seek", "roi_rank"}


def mock_mode():
    """True when the app should not call the Claude API."""
    return os.environ.get("MOCK_LLM", "0") == "1" or not os.environ.get("ANTHROPIC_API_KEY")


def call_tool(name, args, listing, goals=None):
    """Run one tool; errors come back as an 'error' component instead of an exception.
    The listing (and the host's nights per year) come from the sidebar, not from the model."""
    full = {"listing": listing, **args} if name in NEEDS_LISTING else dict(args)
    if name in USES_NIGHTS and goals and goals.get("nights_per_year") and "nights_per_year" not in full:
        full["nights_per_year"] = int(goals["nights_per_year"])
    try:
        result = TOOLS[name](**full)
    except (ValueError, KeyError) as e:
        result = {"component": "error", "data": {"message": str(e).strip("'\"")}}
    return {"tool": name, "args": full, **result}


def respond(question, listing, goals):
    """Answer one free-text question (mock mode)."""
    tool, args = mock_llm.route(question, goals)
    if tool is None:
        return {"text": args, "blocks": []}
    block = call_tool(tool, args, listing, goals)
    return {"text": mock_llm.narrate(tool, block), "blocks": [block]}
