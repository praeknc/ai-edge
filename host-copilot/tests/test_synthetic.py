"""The 5 synthetic listings: Inside Airbnb format, not in the training data, read the same way by the
project's pipeline and by the app.

Run from host-copilot/:   python -m unittest tests.test_synthetic
"""
import json
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from model.loader import DATA, get_model                     # noqa: E402
import make_synthetic_listings as gen                        # noqa: E402

CSV = ROOT / "data" / "synthetic_listings.csv"
PORTFOLIO = json.loads((ROOT / "data" / "host_portfolio.json").read_text())


class SyntheticListings(unittest.TestCase):
    def test_same_columns_as_training_files(self):
        train_cols = list(pd.read_csv(DATA / "detailed" / "listings_detailed_2026Q3_sep.csv.gz", nrows=1).columns)
        self.assertEqual(list(pd.read_csv(CSV, nrows=1).columns), train_cols)

    def test_not_in_training_data(self):
        gen.check_not_training(pd.read_csv(CSV))

    def test_pipeline_and_app_build_the_same_features(self):
        model = get_model()
        listings = [model.complete_listing(item["listing"])[0] for item in PORTFOLIO["listings"]]
        gen.check_parity(CSV, listings, model)

    def test_portfolio_matches_csv(self):
        df = pd.read_csv(CSV)
        self.assertEqual([str(i) for i in df["id"]], [item["id"] for item in PORTFOLIO["listings"]])
        self.assertTrue((df["host_name"] == PORTFOLIO["host"]["host_name"]).all())


if __name__ == "__main__":
    unittest.main()
