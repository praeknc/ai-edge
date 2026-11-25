"""The Claude tool-use loop, run against a scripted fake client (no API calls, no key needed).

Run from host-copilot/:   python -m unittest tests.test_agent
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import agent, prompts                     # noqa: E402

SAMPLE = json.loads((ROOT / "data" / "sample_listing.json").read_text())
GOALS = {"target_price": 600, "budget": 2000, "nights_per_year": 200}


def tool_use(name, args, i=0):
    return NS(type="tool_use", id=f"tu_{name}_{i}", name=name, input=args)


def reply(*content, stop="end_turn"):
    return NS(stop_reason=stop, content=list(content))


def text(t):
    return NS(type="text", text=t)


class FakeClient:
    """Mimics client.beta.messages.create; returns scripted responses and records each request."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        self.beta = NS(messages=NS(create=self.create))

    def create(self, **kw):
        self.requests.append({**kw, "messages": list(kw["messages"])})
        return self.script.pop(0)


class Loop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        agent.LOG_PATH = Path(self.tmp.name) / "turns.jsonl"
        os.environ.pop("CLAUDE_MODEL", None)

    def tearDown(self):
        self.tmp.cleanup()

    def test_tool_call_then_answer(self):
        c = FakeClient([reply(tool_use("predict_price", {}), stop="tool_use"),
                        reply(text("About $543 a night (90% range $324–$942)."))])
        out = agent.respond("What should I charge?", SAMPLE, GOALS, history=[], client=c)
        self.assertEqual([b["component"] for b in out["blocks"]], ["price_card"])
        self.assertEqual(out["unverified"], [])
        req = c.requests[0]
        self.assertEqual(req["model"], "claude-sonnet-5-5")
        self.assertEqual(req["system"], prompts.SYSTEM_PROMPT)
        self.assertEqual(req["tools"], prompts.TOOLS)
        self.assertEqual(req["fallbacks"], "default")
        self.assertEqual([m["role"] for m in req["messages"]], ["user", "system"])   # profile sent as context

    def test_tool_limit(self):
        many = reply(*[tool_use("get_amenity_cost", {"name": "tv"}, i) for i in range(2)], stop="tool_use")
        c = FakeClient([many] * agent.MAX_REQUESTS)
        out = agent.respond("costs?", SAMPLE, GOALS, history=[], client=c)
        self.assertEqual(len(out["blocks"]), agent.MAX_TOOL_CALLS)
        last_results = c.requests[-1]["messages"][-1]["content"]
        self.assertTrue(all(r.get("is_error") for r in last_results))

    def test_refusal(self):
        c = FakeClient([reply(stop="refusal")])
        history = []
        out = agent.respond("...", SAMPLE, GOALS, history=history, client=c)
        self.assertIn("can't help", out["text"])
        self.assertEqual(history, [])                       # the refused turn is not kept

    def test_invented_figure_is_flagged(self):
        c = FakeClient([reply(tool_use("predict_price", {}), stop="tool_use"), reply(text("Charge $999."))])
        out = agent.respond("price?", SAMPLE, GOALS, history=[], client=c)
        self.assertEqual(out["unverified"], [999.0])

    def test_model_cannot_replace_the_listing(self):
        c = FakeClient([reply(tool_use("predict_price", {"listing": {"neighbourhood": "Roxbury"}}), stop="tool_use"),
                        reply(text("ok"))])
        out = agent.respond("price?", SAMPLE, GOALS, history=[], client=c)
        self.assertEqual(out["blocks"][0]["data"]["listing"]["neighbourhood"], "South End")

    def test_history_is_append_only_and_logged(self):
        history = []
        c = FakeClient([reply(text("Hi.")), reply(text("Hello again."))])
        agent.respond("hi", SAMPLE, GOALS, history=history, client=c)
        first = list(history)
        agent.respond("hi again", SAMPLE, GOALS, history=history, client=c)
        self.assertEqual(c.requests[1]["messages"][:len(first)], first)
        self.assertEqual(c.requests[0]["system"], c.requests[1]["system"])
        lines = agent.LOG_PATH.read_text().splitlines()
        self.assertEqual(len(lines), 2)
        rec = json.loads(lines[0])
        self.assertEqual(set(rec) >= {"question", "tool_calls", "final_text", "model", "stop_reason"}, True)

    def test_model_from_environment(self):
        os.environ["CLAUDE_MODEL"] = "claude-opus-5-5"
        c = FakeClient([reply(text("ok"))])
        agent.respond("hi", SAMPLE, GOALS, history=[], client=c)
        self.assertEqual(c.requests[0]["model"], "claude-opus-5-5")
        os.environ.pop("CLAUDE_MODEL")

    def test_server_fallback_can_be_switched_off(self):
        os.environ["CLAUDE_SERVER_FALLBACK"] = "0"
        c = FakeClient([reply(text("ok"))])
        agent.respond("hi", SAMPLE, GOALS, history=[], client=c)
        self.assertNotIn("fallbacks", c.requests[0])
        self.assertNotIn("betas", c.requests[0])
        os.environ.pop("CLAUDE_SERVER_FALLBACK")

    def test_mock_mode_without_key(self):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ["MOCK_LLM"] = "0"
        self.assertTrue(agent.mock_mode())
        out = agent.respond("What should I charge?", SAMPLE, GOALS)
        self.assertEqual(out["blocks"][0]["component"], "price_card")
        self.assertEqual(json.loads(agent.LOG_PATH.read_text().splitlines()[-1])["mode"], "mock")


class MockRouting(unittest.TestCase):
    def test_goal_questions_in_plain_words(self):
        from app import mock_llm
        for q in ["if i want $700 per night is it possible and what should i improve", "Can I make $650 a night?",
                  "How do I get to $450 with a $2,000 budget?"]:
            self.assertEqual(mock_llm.route(q, GOALS)[0], "goal_seek", q)
        self.assertEqual(mock_llm.route("what should I improve?", GOALS)[0], "roi_rank")
        for q, tool in [("how much is a hot tub", "get_amenity_cost"), ("would a jacuzzi help", "what_if"),
                        ("is a dishwasher worth it", "what_if"), ("can you help me price my place", "predict_price"),
                        ("what about going monthly", "what_if"), ("tell me about boston", "clarify")]:
            self.assertEqual(mock_llm.route(q, GOALS)[0], tool, q)

    def test_help_text_is_not_flagged_and_offers_actions(self):
        os.environ["MOCK_LLM"] = "1"
        out = agent.respond("what can i ask", SAMPLE, GOALS)
        self.assertEqual(out["blocks"], [])
        self.assertEqual(out["unverified"], [])
        self.assertEqual(len(out["suggestions"]), 5)

    def test_every_capability_action_runs(self):
        from app.suggestions import capability_actions, follow_ups
        for act in capability_actions(GOALS):
            out = agent.run_action(act, SAMPLE, GOALS)
            self.assertNotEqual(out["blocks"][0]["component"], "error", act["label"])
            for nxt in follow_ups(out["blocks"][0], GOALS):
                self.assertNotEqual(agent.run_action(nxt, SAMPLE, GOALS)["blocks"][0]["component"], "error", nxt["label"])


if __name__ == "__main__":
    unittest.main()
