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


def start():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
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


if __name__ == "__main__":
    unittest.main()
