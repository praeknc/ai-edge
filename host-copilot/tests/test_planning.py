"""goal_seek, roi_rank, costs and the area-effect rule.

Run from host-copilot/:   python -m unittest -v
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import effects, vocab                                   # noqa: E402
from model.loader import get_model                                 # noqa: E402
from app.tools import get_amenity_cost, goal_seek, roi_rank        # noqa: E402
from app.tools.pricing import apply_changes                        # noqa: E402

SAMPLE = json.loads((ROOT / "data" / "sample_listing.json").read_text())
DOWNTOWN = {"neighbourhood": "Downtown", "room_type": "Entire home/apt", "bedrooms": 1, "accommodates": 2,
            "amenities": ["kitchen", "tv", "self check-in"]}


def rescore(listing, option):
    """Price a goal_seek option again, independently of the tool."""
    model = get_model()
    L, _ = model.complete_listing(listing)
    return round(model.price([apply_changes(L, [a["key"] for a in option["amenities"]], [])])[0]["price"])


class EffectRule(unittest.TestCase):
    def test_rule_matches_table(self):
        e = effects.area_effect("am_dishwasher", "Back Bay", "Back Bay / South End")
        self.assertTrue(e["substantial"])
        self.assertEqual(e["level"], "neighbourhood")
        self.assertGreaterEqual(e["lift_usd"], effects.MIN_USD)

    def test_small_effect_is_not_substantial(self):
        e = effects.area_effect("am_dishwasher", "Dorchester", "Dorchester / Mattapan")
        self.assertFalse(e["substantial"])
        self.assertTrue(e["reason"].startswith("too small"))

    def test_rare_amenity_reports_missing_data(self):
        self.assertEqual(effects.area_effect("am_hot_tub", "Back Bay", "Back Bay / South End")["level"], "none")


class GoalSeek(unittest.TestCase):
    def check_options(self, listing, target, budget):
        d = goal_seek(listing, target_price=target, budget=budget)["data"]
        for o in d["options"]:
            self.assertLessEqual(o["cost"], budget)
            self.assertEqual(o["price"]["price"], rescore(listing, o))
            for a in o["amenities"]:
                self.assertNotEqual(vocab.kind(a["key"]), "building")
            if d["status"] == "reachable":
                self.assertGreaterEqual(o["price"]["price"], target)
        costs = [o["cost"] for o in d["options"]]
        if d["status"] == "reachable":
            self.assertEqual(costs, sorted(costs))
        return d

    def test_reachable_target(self):
        base = goal_seek(DOWNTOWN, target_price=1, budget=0)["data"]["base"]["price"]
        d = self.check_options(DOWNTOWN, base + 5, 5000)
        self.assertEqual(d["status"], "reachable")
        self.assertTrue(d["options"])

    def test_unrealistic_target(self):
        d = self.check_options(SAMPLE, 5000, 100000)
        self.assertEqual(d["status"], "unreachable")
        self.assertTrue(d["target_above_range"])

    def test_budget_is_respected(self):
        d = self.check_options(DOWNTOWN, 10000, 1000)
        self.assertTrue(all(o["cost"] <= 1000 for o in d["options"]))

    def test_already_met(self):
        self.assertEqual(goal_seek(SAMPLE, target_price=10, budget=500)["data"]["status"], "already_met")


class RoiRank(unittest.TestCase):
    def test_items_are_substantial_addable_and_sorted(self):
        d = roi_rank(DOWNTOWN, budget=2000)["data"]
        self.assertTrue(d["items"])
        pay = [i["payback_months"] for i in d["items"] if i["payback_months"] is not None]
        self.assertEqual(pay, sorted(pay))
        for i in d["items"]:
            self.assertGreater(i["lift_per_night"], 0)
            self.assertNotEqual(i["kind"], "building")
            self.assertEqual(i["within_budget"], i["cost"] <= 2000)

    def test_payback_formula(self):
        for i in roi_rank(DOWNTOWN)["data"]["items"]:
            monthly = i["lift_per_night"] * 200 / 12 - i["monthly_cost"]
            self.assertAlmostEqual(i["payback_months"], round(i["cost"] / monthly, 1), places=1)


class Costs(unittest.TestCase):
    def test_known_and_out_of_vocabulary(self):
        hot = get_amenity_cost("Jacuzzi")["data"]
        self.assertEqual((hot["cost_low"], hot["cost_high"], hot["in_model"]), (6000, 12000, True))
        wifi = get_amenity_cost("fast wifi")["data"]
        self.assertTrue(wifi["found"])
        self.assertFalse(wifi["in_model"])
        self.assertFalse(get_amenity_cost("sauna")["data"]["found"])

    def test_every_addable_amenity_has_a_cost(self):
        from app.tools.costs import cost_of
        missing = [k for k in vocab.AMENITIES if vocab.kind(k) != "building" and cost_of(k) is None]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
