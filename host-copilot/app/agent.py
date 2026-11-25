"""The agent: one host question in, short narration plus UI blocks out.

respond() returns {"text", "blocks": [{"tool", "args", "component", "data"}], "unverified": [...]}.

Two modes:
  * mock  (MOCK_LLM=1, or no ANTHROPIC_API_KEY): keyword routing in app/mock_llm.py.
  * Claude: a tool-use loop with the Anthropic SDK. The model is CLAUDE_MODEL (default claude-sonnet-5-5).
    At most MAX_TOOL_CALLS tool calls per user turn; extra calls get an error result and the model is
    asked to answer with what it has.
Every turn (question, tool calls with arguments and outputs, final text) is appended to logs/turns.jsonl.
"""
import json
import os
import time
from pathlib import Path

from app import mock_llm, prompts, suggestions
from app.faithfulness import unverified_figures
from app.tools import get_amenity_cost, goal_seek, predict_price, roi_rank, what_if

TOOLS = {"predict_price": predict_price, "what_if": what_if, "goal_seek": goal_seek, "roi_rank": roi_rank,
         "get_amenity_cost": get_amenity_cost}
NEEDS_LISTING = {"predict_price", "what_if", "goal_seek", "roi_rank"}
USES_NIGHTS = {"what_if", "goal_seek", "roi_rank"}
MAX_TOOL_CALLS = 5
MAX_REQUESTS = MAX_TOOL_CALLS + 2         # one request per tool round, plus the final answer, plus one spare
DEFAULT_MODEL = "claude-sonnet-5-5"
LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "turns.jsonl"


def mock_mode():
    """True when the app should not call the Claude API."""
    return os.environ.get("MOCK_LLM", "0") == "1" or not os.environ.get("ANTHROPIC_API_KEY")


def model_name():
    return os.environ.get("CLAUDE_MODEL", DEFAULT_MODEL)


# ---------------------------------------------------------------- tools
def call_tool(name, args, listing, goals=None):
    """Run one tool; errors come back as an 'error' component instead of an exception.
    The listing (and the host's nights per year) come from the sidebar, never from the model."""
    if name not in TOOLS:
        return {"tool": name, "args": args, "component": "error", "data": {"message": f"Unknown tool '{name}'."}}
    args = {k: v for k, v in args.items() if k != "listing"}           # the model never supplies the listing
    full = {**args, "listing": listing} if name in NEEDS_LISTING else dict(args)
    if name in USES_NIGHTS and goals and goals.get("nights_per_year") and "nights_per_year" not in full:
        full["nights_per_year"] = int(goals["nights_per_year"])
    try:
        result = TOOLS[name](**full)
    except (ValueError, KeyError, TypeError) as e:
        result = {"component": "error", "data": {"message": str(e).strip("'\"")}}
    return {"tool": name, "args": full, **result}


def tool_result_for_model(block):
    """What the model reads back: the component name and its data (the listing itself is left out)."""
    return json.dumps({"component": block["component"], "data": block["data"]}, separators=(",", ":"), default=str)


# ---------------------------------------------------------------- logging
def log_turn(record):
    """Append one turn to logs/turns.jsonl (for evaluation)."""
    LOG_PATH.parent.mkdir(exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _log(mode, question, listing, goals, blocks, text, extra=None):
    log_turn({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "mode": mode, "model": None if mode == "mock" else model_name(),
              "question": question, "listing": listing, "goals": goals,
              "tool_calls": [{"name": b["tool"], "args": {k: v for k, v in b["args"].items() if k != "listing"},
                              "component": b["component"], "output": b["data"]} for b in blocks],
              "final_text": text, **(extra or {})})


# ---------------------------------------------------------------- mock mode
def respond_mock(question, listing, goals):
    """Mock mode: keyword scoring picks a tool; unclear questions get clickable options instead of a guess."""
    tool, args = mock_llm.route(question, goals)
    if tool == "clarify":
        by_tool = {a["tool"]: a for a in suggestions.capability_actions(goals)}
        options = [by_tool[t] for t in args if t in by_tool] or suggestions.capability_actions(goals)[:3]
        return {"text": mock_llm.CLARIFY_TEXT, "blocks": [], "suggestions": options}
    if tool is None:
        return {"text": args, "blocks": [], "suggestions": suggestions.capability_actions(goals)}
    block = call_tool(tool, args, listing, goals)
    return {"text": mock_llm.narrate(tool, block), "blocks": [block]}


def run_action(act, listing, goals):
    """A clicked suggestion or capability card: call its tool directly (no language model) and narrate."""
    block = call_tool(act["tool"], act["args"], listing, goals)
    text = mock_llm.narrate(act["tool"], block)
    _log("action", act["label"], listing, goals, [block], text)
    return {"text": text, "blocks": [block], "unverified": []}


# ---------------------------------------------------------------- Claude mode
def make_client():
    import anthropic
    return anthropic.Anthropic()


def _request(client, messages):
    """One Messages API call. Effort `low` suits short chat answers. With CLAUDE_SERVER_FALLBACK=1 (default),
    refusals in some categories are retried server-side on a fallback model; set it to 0 when the API is
    reached through a proxy (ANTHROPIC_BASE_URL) that does not accept that beta."""
    extra = {}
    if os.environ.get("CLAUDE_SERVER_FALLBACK", "1") == "1":
        extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    return client.beta.messages.create(
        model=model_name(),
        max_tokens=16000,
        system=prompts.SYSTEM_PROMPT,
        tools=prompts.TOOLS,
        messages=messages,
        output_config={"effort": os.environ.get("CLAUDE_EFFORT", "low")},
        **extra,
    )


def run_claude_turn(client, history, question, listing, goals):
    """Run one user turn through the tool-use loop.

    history: the conversation so far (list of message dicts); it is extended in place, append-only,
    so earlier turns (including thinking blocks) are replayed unchanged."""
    history.append({"role": "user", "content": question})
    history.append({"role": "system", "content": prompts.context_message(listing, goals)})
    blocks, calls, stop = [], 0, None
    for _ in range(MAX_REQUESTS):
        response = _request(client, history)
        stop = response.stop_reason
        history.append({"role": "assistant", "content": response.content})
        if stop == "refusal":
            raise Refused()
        if stop == "pause_turn":
            continue
        uses = [b for b in response.content if b.type == "tool_use"]
        if stop != "tool_use" or not uses:
            text = " ".join(b.text for b in response.content if b.type == "text").strip()
            return text, blocks, stop
        results = []
        for use in uses:
            if calls >= MAX_TOOL_CALLS:
                results.append({"type": "tool_result", "tool_use_id": use.id, "is_error": True,
                                "content": f"Tool call limit ({MAX_TOOL_CALLS} per question) reached. "
                                           "Answer with the results you already have."})
                continue
            calls += 1
            block = call_tool(use.name, dict(use.input or {}), listing, goals)
            blocks.append(block)
            results.append({"type": "tool_result", "tool_use_id": use.id, "content": tool_result_for_model(block),
                            **({"is_error": True} if block["component"] == "error" else {})})
        history.append({"role": "user", "content": results})
    text = "I couldn't finish that answer within the tool limit; the cards below show what I found."
    history.append({"role": "assistant", "content": text})
    return text, blocks, stop


def respond(question, listing, goals, history=None, client=None):
    """Answer one free-text question. `history` (Claude mode) is the session's message list."""
    if mock_mode() and client is None:
        out = respond_mock(question, listing, goals)
        # Only narration built from tool results is checked; the fixed help text has no tool behind it.
        out["unverified"] = unverified_figures(out["text"], [b["data"] for b in out["blocks"]]) if out["blocks"] else []
        _log("mock", question, listing, goals, out["blocks"], out["text"], {"unverified": out["unverified"]})
        return out
    import anthropic
    history = [] if history is None else history
    start = len(history)
    text, blocks, stop = None, [], None
    try:
        text, blocks, stop = run_claude_turn(client or make_client(), history, question, listing, goals)
    except Refused:
        text, stop = "I can't help with that request.", "refusal"
    except anthropic.RateLimitError:
        text, blocks, stop = "The language model is busy (rate limited). Try again in a minute.", [], "rate_limited"
    except anthropic.AuthenticationError:
        text, blocks, stop = "The API key was rejected. Check ANTHROPIC_API_KEY, or run with MOCK_LLM=1.", [], "auth_error"
    except anthropic.APIStatusError as e:
        text, blocks, stop = f"The language model returned an error ({e.status_code}). Try again.", [], "api_error"
    except anthropic.APIConnectionError:
        text, blocks, stop = "Could not reach the language model. Check the network, or run with MOCK_LLM=1.", [], "network_error"
    if stop in ("refusal", "rate_limited", "auth_error", "api_error", "network_error"):
        del history[start:]                     # drop the unfinished turn; earlier turns stay unchanged
    unverified = unverified_figures(text, [b["data"] for b in blocks])
    _log("claude", question, listing, goals, blocks, text, {"stop_reason": stop, "unverified": unverified})
    return {"text": text, "blocks": blocks, "unverified": unverified}
