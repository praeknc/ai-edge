"""Checks that Host Copilot prices exactly like the final notebook, and that the tools behave.

Run from host-copilot/:   python -m unittest -v        (or: python -m pytest -q)
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.loader import PRICING_DIR, get_model            # noqa: E402
from model import vocab                                     # noqa: E402
from app.tools import predict_price, what_if                # noqa: E402

SAMPLE = json.loads((ROOT / "data" / "sample_listing.json").read_text())


def read_js_bundle(path):
    """JSON object inside a `window.X = {...};` file written by the notebooks."""
    text = Path(path).read_text()
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


class MatchesNotebook(unittest.TestCase):
    """The exported model and feature path reproduce boston_airbnb_final_pipeline.ipynb."""

    @classmethod
    def setUpClass(cls):
        cls.model = get_model()

    def test_saved_model_matches_notebook_predictions(self):
        """200 feature rows stored by the notebook -> same log price."""
        ref = read_js_bundle(PRICING_DIR / "interface" / "models.js")["final_reference"]
        X = pd.DataFrame([[np.nan if v is None else v for v in r] for r in ref["X"]], columns=self.model.feature_names)
        gap = np.max(np.abs(self.model.predict_log(X) - np.asarray(ref["pred"])))
        self.assertLess(gap, 1e-6)

    def test_raw_listings_build_notebook_features_and_prices(self):
        """40 listings typed as raw inputs (the notebook's parity cases) -> same 141 features and price."""
        cases = json.loads((PRICING_DIR / "scripts" / "final_cases.json").read_text())
        X = self.model.features([c["input"] for c in cases])
        for i, c in enumerate(cases):
            expected = np.array([np.nan if v is None else v for v in c["x"]], dtype=float)
            np.testing.assert_allclose(X.iloc[i].values.astype(float), expected, rtol=1e-6, atol=1e-9, equal_nan=True,
                                       err_msg=f"case {i}")
        np.testing.assert_allclose(self.model.predict_log(X), [c["pred"] for c in cases], atol=1e-6)

    def test_80_percent_interval_matches_notebook(self):
        """The out-of-fold residuals are reproduced: the 80% band equals the notebook's."""
        band = read_js_bundle(PRICING_DIR / "interface" / "data.js")["final"]["band_log"]
        iv = self.model.meta["interval"]["80"]
        np.testing.assert_allclose([iv["low_log"], iv["high_log"]], band, atol=1e-6)

    def test_shap_values_add_up_to_prediction(self):
        """TreeSHAP contributions plus the base value equal the log prediction."""
        L, _ = self.model.complete_listing(SAMPLE)
        X = self.model.features([L])
        c = self.model.contributions_log(X)
        self.assertAlmostEqual(float(c.sum(axis=1).iloc[0]), float(self.model.predict_log(X)[0]), places=6)


class PredictPrice(unittest.TestCase):
    def test_returns_price_card_with_interval(self):
        r = predict_price(SAMPLE)
        self.assertEqual(r["component"], "price_card")
        d = r["data"]
        self.assertLess(d["low"], d["price"])
        self.assertLess(d["price"], d["high"])
        self.assertEqual(d["interval"]["level"], 0.90)

    def test_reports_filled_fields_and_current_price(self):
        d = predict_price({"neighbourhood": "Back Bay", "current_price": 150})["data"]
        self.assertIn("bedrooms", d["filled_fields"])
        self.assertIn(d["current_vs_range"], {"below", "within", "above"})

    def test_unknown_neighbourhood_suggests_closest(self):
        with self.assertRaises(ValueError) as ctx:
            predict_price({"neighbourhood": "Back Bey"})
        self.assertIn("Back Bay", str(ctx.exception))

    def test_merged_neighbourhood_is_accepted(self):
        d = predict_price({"neighbourhood": "Leather District"})["data"]
        self.assertEqual(d["listing"]["neighbourhood"], "Chinatown")


class WhatIf(unittest.TestCase):
    def test_contributions_add_up_to_total_change(self):
        r = what_if(SAMPLE, add=["hot tub", "free parking", "balcony"], remove=["dishwasher"])
        self.assertEqual(r["component"], "whatif_panel")
        d = r["data"]
        total = sum(c["contribution"] for c in d["contributions"])
        self.assertLessEqual(abs(total - d["delta"]), len(d["contributions"]))   # rounding: <= $1 per item
        self.assertEqual(d["new"]["price"] - d["base"]["price"], d["delta"])

    def test_already_present_amenity_changes_nothing(self):
        d = what_if(SAMPLE, add=["tv"])["data"]
        self.assertEqual(d["delta"], 0)
        self.assertEqual(d["already_present"], ["TV"])
        self.assertEqual(d["contributions"], [])

    def test_unsupported_amenity_gets_suggestions(self):
        d = what_if(SAMPLE, add=["fire pit"])["data"]
        self.assertEqual(d["unsupported"][0]["name"], "fire pit")
        self.assertEqual([s["key"] for s in d["unsupported"][0]["suggestions"]], ["am_bbq", "am_backyard"])

    def test_add_then_remove_is_symmetric(self):
        # the total amenity count moves with the change (36 -> 37), so state it explicitly
        up = what_if({**SAMPLE, "amenity_count": 36}, add=["hot tub"])["data"]
        listing2 = {**SAMPLE, "amenities": SAMPLE["amenities"] + ["hot tub"], "amenity_count": 37}
        down = what_if(listing2, remove=["hot tub"])["data"]
        self.assertEqual(up["new"]["price"], down["base"]["price"])
        self.assertEqual(up["base"]["price"], down["new"]["price"])

    def test_single_change_contribution_equals_alone(self):
        d = what_if(SAMPLE, add=["hot tub"])["data"]
        c = d["contributions"][0]
        self.assertEqual(c["contribution"], c["alone"])
        self.assertEqual(c["contribution"], d["delta"])


class Vocabulary(unittest.TestCase):
    def test_synonyms_resolve(self):
        self.assertEqual(vocab.resolve("Jacuzzi"), "am_hot_tub")
        self.assertEqual(vocab.resolve("am_ev_charger"), "am_ev_charger")
        self.assertEqual(vocab.resolve("AC"), "am_any_ac")

    def test_not_in_model(self):
        self.assertIsNone(vocab.resolve("fast wifi"))
        self.assertEqual(vocab.suggest("fast wifi"), ["am_workspace"])


if __name__ == "__main__":
    unittest.main()
