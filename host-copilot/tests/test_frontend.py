"""End-to-end UI checks in mock-LLM mode with Streamlit's headless AppTest.

Run from host-copilot/:   python -m unittest tests.test_frontend
"""
import os
import sys
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["MOCK_LLM"] = "1"
APP = str(ROOT / "app" / "frontend.py")
IDS = [item["id"] for item in __import__("json").loads((ROOT / "data" / "host_portfolio.json").read_text())["listings"]]


def start(listing=0):
    """Open the app, sign in as the demo host and open one of the 5 listings (0 = South End condo)."""
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    next(b for b in at.button if b.label.startswith("Sign in as")).click().run()
    at.button(key=f"open-{IDS[listing]}").click().run()
    assert not at.exception, at.exception
    return at


def ask(at, text):
    at.chat_input[0].set_value(text).run()
    assert not at.exception, at.exception
    return at


def metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


class Frontend(unittest.TestCase):
    def test_price_question_shows_interval(self):
        at = ask(start(), "What should I charge?")
        self.assertIn("90% range", at.chat_message[-1].markdown[0].value)
        self.assertTrue(any("90% range" in c.value for c in at.caption))

    def test_whatif_toggle_reprices_without_llm(self):
        at = ask(start(), "What if I add a dishwasher?")
        before = metric(at, "Predicted price")
        n_messages = len(at.chat_message)
        next(w for w in at.number_input if str(w.key).endswith("-minimum_nights")).set_value(30).run()
        self.assertFalse(at.exception)
        self.assertNotEqual(metric(at, "Predicted price"), before)
        self.assertEqual(len(at.chat_message), n_messages)          # no new chat turn: the tool was called directly

    def test_plan_slider_and_apply_to_whatif(self):
        at = ask(start(), "How do I get to $600 with a budget of $3,000?")
        next(w for w in at.slider if str(w.key).endswith("-target")).set_value(560).run()
        self.assertFalse(at.exception)
        apply = [b for b in at.button if b.label == "Apply to what-if"]
        self.assertTrue(apply)                                        # the best bundle is shown even when unreachable
        apply[0].click().run()
        self.assertFalse(at.exception)
        self.assertIn("Loaded option 1", at.chat_message[-1].markdown[0].value)
        self.assertTrue(any(m.label == "Payback" for m in at.metric))

    def test_out_of_scope_and_unsupported(self):
        at = ask(start(), "Can you do my taxes?")
        self.assertIn("can't help", at.chat_message[-1].markdown[0].value)
        at = ask(at, "What if I add a fire pit?")
        self.assertIn("closest", at.chat_message[-1].markdown[0].value)


class HostFlow(unittest.TestCase):
    """Sign in, choose a listing, switch listings, restart, reset, and come back after a refresh."""

    def test_portfolio_has_five_synthetic_listings(self):
        at = AppTest.from_file(APP, default_timeout=60)
        at.run()
        next(b for b in at.button if b.label.startswith("Sign in as")).click().run()
        self.assertEqual(len([b for b in at.button if b.label == "Open copilot →"]), 5)
        self.assertTrue(any("Your listings (5)" in t.value for t in at.title))

    def test_each_listing_keeps_its_chat(self):
        at = ask(start(0), "What should I charge?")
        n = len(at.chat_message)
        next(b for b in at.button if b.label == "← My listings").click().run()
        at.button(key=f"open-{IDS[2]}").click().run()                               # Dorchester room
        self.assertEqual(len(at.chat_message), 0)
        self.assertEqual(at.session_state["p_hood"], "Dorchester")
        next(b for b in at.button if b.label == "← My listings").click().run()
        at.button(key=f"open-{IDS[0]}").click().run()                               # back to South End
        self.assertEqual(len(at.chat_message), n)
        self.assertEqual(at.session_state["p_hood"], "South End")

    def test_restart_and_reset(self):
        at = ask(start(0), "What should I charge?")
        at.number_input(key="p_bed").set_value(3).run()
        next(b for b in at.button if b.label == "⟲ Reset details").click().run()
        self.assertEqual(at.session_state["p_bed"], 2)
        self.assertGreater(len(at.chat_message), 0)                                   # reset keeps the chat
        next(b for b in at.button if b.label == "↻ Restart chat").click().run()
        self.assertEqual(len(at.chat_message), 0)
        self.assertTrue(any("What I can do" in m.value for m in at.markdown))

    def test_refresh_returns_to_listing_in_url(self):
        at = AppTest.from_file(APP, default_timeout=60)
        at.query_params["host"] = "demo"
        at.query_params["listing"] = "990000000000000003"
        at.run()
        self.assertFalse(at.exception)
        self.assertTrue(any("Dorchester room" in t.value for t in at.title))


class Flow(unittest.TestCase):
    """Capability cards, next-step buttons, clarification and new conversations."""

    def click(self, at, label):
        next(b for b in at.button if b.label == label).click().run()
        self.assertFalse(at.exception)
        return at

    def test_capability_card_runs_tool_and_offers_next_steps(self):
        at = start()
        self.assertTrue(any("What I can do" in m.value for m in at.markdown))
        at = self.click(at, "What should I charge?")
        self.assertIn("90% range", at.chat_message[-1].markdown[0].value)
        labels = [b.label for b in at.button]
        self.assertIn("Which upgrades pay back fastest?", labels)        # a next step under the answer
        at = self.click(at, "Which upgrades pay back fastest?")
        self.assertIn("payback", at.chat_message[-1].markdown[0].value.lower())

    def test_unclear_question_gets_options(self):
        at = ask(start(), "tell me about boston")
        self.assertIn("not sure", at.chat_message[-1].markdown[0].value)
        self.assertIn("What should I charge?", [b.label for b in at.button])

    def test_new_conversation_shows_cards_again(self):
        at = ask(start(), "What should I charge?")
        at = self.click(at, "↻ Restart chat")
        self.assertEqual(len(at.chat_message), 0)
        self.assertTrue(any("What I can do" in m.value for m in at.markdown))


if __name__ == "__main__":
    unittest.main()
