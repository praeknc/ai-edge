"""Load the exported pricing model and turn a host's listing description into model features.

The feature code is the project's own (`airbnb-pricing/src/`), used unchanged. A host's listing is
first written as a raw Inside Airbnb row (the same `raw_row` the final notebook uses for its parity
cases), then sent through the same pre-check and feature pipeline as the training data. That path is
what `tests/test_pricing.py` checks against the notebook.

Two listing formats appear here:
  * host listing  - what the app and the tools use (see data/sample_listing.json). Friendly names,
                    every field optional except `neighbourhood`.
  * model listing - every field filled, in the exact format of the notebook's parity cases.
`complete_listing()` converts the first into the second and records which fields were filled in.
"""
import difflib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
PRICING_DIR = Path(os.environ.get("AIRBNB_PRICING_DIR", HERE.parents[1] / "airbnb-pricing"))
DATA = PRICING_DIR / "data"
sys.path.insert(0, str(PRICING_DIR / "src"))
sys.path.insert(0, str(HERE))
import pricing_features as pf      # noqa: E402  (project code, unchanged)
import enriched_features as ef     # noqa: E402
import vocab                       # noqa: E402

# ---------------------------------------------------------------- raw-row conventions (copied from the final notebook)
SCORES = ["rating", "accuracy", "cleanliness", "checkin", "communication", "location", "value"]
PROPERTY_GROUPS = sorted(set(ef.PROPERTY_GROUPS.values())) + ["other"]
REP_TYPE = {g: t for t, g in reversed(list(ef.PROPERTY_GROUPS.items()))}
REP_TYPE["other"] = "Tiny home"
AMEN_LABEL = {
    "am_free_parking": "Free parking on premises", "am_paid_parking": "Paid parking garage off premises",
    "am_washer_in_unit": "Free washer – In unit", "am_dryer": "Dryer", "am_dishwasher": "Dishwasher",
    "am_central_ac": "Central air conditioning", "am_any_ac": "Air conditioning", "am_gym": "Gym", "am_pool": "Pool",
    "am_hot_tub": "Hot tub", "am_elevator": "Elevator", "am_workspace": "Dedicated workspace",
    "am_self_checkin": "Self check-in", "am_balcony_patio": "Patio or balcony", "am_view": "City skyline view",
    "am_kitchen": "Kitchen", "am_bathtub": "Bathtub", "am_fireplace": "Indoor fireplace", "am_ev_charger": "EV charger",
    "am_pets": "Pets allowed", "am_long_term": "Long term stays allowed", "am_luggage": "Luggage dropoff allowed",
    "am_crib": "Crib", "am_private_entry": "Private entrance", "am_backyard": "Backyard", "am_bbq": "BBQ grill",
    "am_tv": "TV", "am_streaming": "Netflix", "am_game_console": "Game console", "am_sound_system": "Bluetooth sound system"}
KW_WORD = {"kw_parking": "parking", "kw_luxury": "modern", "kw_view": "view", "kw_amenities": "gym"}
KW_FROM_HOST = {"parking": "kw_parking", "luxury": "kw_luxury", "modern": "kw_luxury", "renovated": "kw_luxury",
                "view": "kw_view", "amenities": "kw_amenities", "gym": "kw_amenities", "pool": "kw_amenities"}
ROOM_TYPES = ["Entire home/apt", "Private room", "Hotel room", "Shared room"]
SEASON_TO_QUARTER = {"summer": "2026Q2", "june": "2026Q2", "2026Q2": "2026Q2",
                     "fall": "2026Q3", "autumn": "2026Q3", "september": "2026Q3", "2026Q3": "2026Q3"}
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
LICENSES = ["licensed", "exempt", "none"]


def raw_row(L, i, snapshot_dates):
    """Write a model listing as one raw Inside Airbnb row (same convention as the notebook's parity cases)."""
    scraped = pd.Timestamp(snapshot_dates[L["quarter"]])
    am = [AMEN_LABEL[k] for k, v in L["amenities"].items() if v]
    if L["amenities"].get("am_central_ac"):
        am.append("Air conditioning")
    am += [f"Essentials item {j}" for j in range(max(0, L["am_count"] - len(am)))]
    title = ("Studio" if L["bedrooms"] == 0 else f"{L['bedrooms']}BR") + " " + " ".join(KW_WORD[k] for k, v in L["title_kw"].items() if v)
    has_rev = L["number_of_reviews"] > 0
    lic = {"licensed": "STR-123456", "exempt": "Exempt", "none": np.nan}[L["license"]]
    checkin = scraped + pd.Timedelta(days=1)
    checkin = checkin + pd.Timedelta(days=(L["quote_dow"] - checkin.dayofweek) % 7)
    if L["baths"] == 0.5:
        baths_text = "Shared half-bath" if L["baths_shared"] else "Half-bath"
    else:
        baths_text = f"{L['baths']:g} {'shared ' if L['baths_shared'] else ''}baths"
    row = {
        "id": -1 - i, "name": title, "host_id": -1 - i, "neighbourhood": L["neighbourhood"],
        "neighbourhood_cleansed": L["neighbourhood"], "room_type": L["room_type"], "latitude": L["lat"],
        "longitude": L["lon"], "price": 100.0, "minimum_nights": L["minimum_nights"],
        "number_of_reviews": L["number_of_reviews"], "reviews_per_month": L["reviews_per_month"] if has_rev else np.nan,
        "number_of_reviews_ltm": min(L["number_of_reviews_ltm"], L["number_of_reviews"]),
        "calculated_host_listings_count": L["host_listings"], "availability_365": L["availability_365"], "license": lic,
        "accommodates": L["accommodates"], "bedrooms": L["bedrooms"], "beds": L["beds"], "bathrooms_text": baths_text,
        "property_type": REP_TYPE[L["property_group"]], "amenities": json.dumps(am),
        "host_is_superhost": "t" if L["superhost"] else "f", "host_identity_verified": "t" if L["identity_verified"] else "f",
        "description": "x" * L["desc_len"], "last_scraped": str(scraped.date()),
        "last_review": str((scraped - pd.Timedelta(days=L["days_since_review"])).date()) if has_rev else np.nan,
        "first_review": str((scraped - pd.Timedelta(days=int(np.floor(L["months_since_first_review"] * 365.25 / 12 + 0.5)))).date()) if has_rev else np.nan,
        "price_quote_checkin_date": str(checkin.date()), "quarter": L["quarter"]}
    for s in SCORES:
        row[f"review_scores_{s}"] = L["scores"][s] if has_rev else np.nan
    return row


# ---------------------------------------------------------------- the model
class PricingModel:
    """The exported LightGBM model plus everything needed to build its 141 features."""

    def __init__(self, artifacts=ART):
        artifacts = Path(artifacts)
        self.meta = json.loads((artifacts / "meta.json").read_text())
        self.feature_names = self.meta["features"]
        self.booster = lgb.Booster(model_file=str(artifacts / "final_lgbm.txt"))
        self.supply = pd.read_csv(artifacts / "supply.csv.gz")
        self.segments = pd.read_csv(artifacts / "segments.csv").set_index(["neighbourhood", "room_type"])
        self.hood_centers = pd.read_csv(artifacts / "hood_centers.csv").set_index("neighbourhood")
        self.official = pd.read_csv(DATA / "neighbourhoods.csv")["neighbourhood"]
        self.geo = pf.load_geo(DATA / "neighbourhoods.geojson")

    # ---- interval
    def interval(self, level="90"):
        """(low multiplier, high multiplier, description) of the prediction interval."""
        iv = self.meta["interval"][level]
        return iv["low_mult"], iv["high_mult"], self.meta["interval_method"]

    # ---- host listing -> model listing
    def neighbourhoods(self):
        """Neighbourhood names the model knows."""
        return list(self.hood_centers.index)

    def _hood(self, name):
        """Validate a neighbourhood name (case-insensitive, merges applied)."""
        name = pf.HOOD_MERGE.get(name, name)
        names = {h.lower(): h for h in self.neighbourhoods()}
        if str(name).lower() in names:
            return names[str(name).lower()]
        close = difflib.get_close_matches(str(name), self.neighbourhoods(), n=3, cutoff=0.4)
        raise ValueError(f"Unknown neighbourhood '{name}'. Closest: {', '.join(close) or 'none'}")

    def _segment(self, hood, room):
        """Typical values for this neighbourhood x room type (falls back to all of Boston)."""
        room = pf.ROOM_MERGE.get(room, room)
        if (hood, room) in self.segments.index and self.segments.loc[(hood, room), "n"] >= 10:
            return self.segments.loc[(hood, room)]
        return self.segments.loc[("*", room)]

    def complete_listing(self, host):
        """Fill a host listing into a full model listing.

        Returns (model_listing, filled) where `filled` names the fields taken from the
        neighbourhood x room-type medians because the host did not give them."""
        filled = []

        def get(key, default):
            if host.get(key) is None:
                filled.append(key)
                return default
            return host[key]

        hood = self._hood(host["neighbourhood"])
        room = get("room_type", "Entire home/apt")
        if room not in ROOM_TYPES:
            raise ValueError(f"room_type must be one of {ROOM_TYPES}")
        seg = self._segment(hood, room)
        amen_keys, _unknown = vocab.resolve_many(host.get("amenities") or [])
        amenities = {k: (k in amen_keys) for k in AMEN_LABEL}
        if amenities["am_central_ac"]:
            amenities["am_any_ac"] = True
        n_lines = sum(amenities.values())
        am_count = int(get("amenity_count", max(int(round(seg["am_count"])), n_lines)))
        kws = {KW_FROM_HOST[w.lower()] for w in (host.get("title_keywords") or []) if w.lower() in KW_FROM_HOST}
        night = get("night_of_week", "friday")
        dow = night if isinstance(night, int) else DAYS.index(str(night).lower())
        season = str(get("season", "fall")).lower()
        n_rev = int(get("number_of_reviews", int(round(seg["number_of_reviews"]))))
        score = float(get("review_score", float(seg["review_score"]) if not np.isnan(seg["review_score"]) else 4.8))
        lic = get("license", "licensed")
        if lic not in LICENSES:
            raise ValueError(f"license must be one of {LICENSES}")
        prop = get("property_group", "rental unit" if room == "Entire home/apt" else "room in unit")
        if prop not in PROPERTY_GROUPS:
            raise ValueError(f"property_group must be one of {PROPERTY_GROUPS}")
        L = {
            "neighbourhood": hood,
            "lat": float(get("latitude", float(self.hood_centers.loc[hood, "latitude"]))),
            "lon": float(get("longitude", float(self.hood_centers.loc[hood, "longitude"]))),
            "room_type": room, "property_group": prop,
            "accommodates": int(get("accommodates", int(round(seg["accommodates"])))),
            "bedrooms": int(get("bedrooms", int(round(seg["bedrooms"])))),
            "beds": int(get("beds", int(round(seg["beds"])))),
            "baths": float(get("baths", float(seg["baths"]))),
            "baths_shared": bool(get("baths_shared", room != "Entire home/apt")),
            "amenities": amenities, "am_count": max(am_count, n_lines),
            "title_kw": {k: (k in kws) for k in KW_WORD},
            "minimum_nights": int(get("minimum_nights", int(round(seg["minimum_nights"])))),
            "availability_365": int(get("availability_365", int(round(seg["availability_365"])))),
            "license": lic, "host_listings": int(get("host_listings", 1)),
            "superhost": bool(get("superhost", False)), "identity_verified": bool(get("identity_verified", True)),
            "number_of_reviews": n_rev,
            "reviews_per_month": float(get("reviews_per_month", float(seg["reviews_per_month"]))),
            "number_of_reviews_ltm": int(get("number_of_reviews_ltm", int(round(seg["number_of_reviews_ltm"])))),
            "scores": {s: score for s in SCORES},
            "months_since_first_review": float(get("months_since_first_review", float(seg["months_since_first_review"]))),
            "days_since_review": int(get("days_since_review", int(round(seg["days_since_review"])))),
            "quarter": SEASON_TO_QUARTER[season], "tenure_q": int(get("tenure_quarters", 2)), "quote_dow": dow,
            "desc_len": int(get("description_length", int(round(seg["desc_len"])))),
        }
        return L, filled

    # ---- model listing -> features
    def features(self, listings):
        """141-column feature frame for a list of model listings (row order kept).

        Listings are grouped by location, season and tenure; each group goes through the
        project's pipeline once, with that location added to the nearby-listings supply
        (the density code subtracts the listing itself)."""
        parts = []
        keyed = pd.Series([(L["lat"], L["lon"], L["quarter"], L["tenure_q"]) for L in listings])
        for key, idx in keyed.groupby(keyed).groups.items():
            lat, lon, q, tenure = key
            rows = pd.DataFrame([raw_row(listings[i], i, self.meta["snapshot_dates"]) for i in idx])
            clean, _ = pf.precheck(rows, self.official, self.geo)
            if len(clean) != len(rows):
                raise ValueError("listing failed the data pre-check (check reviews, price range or neighbourhood)")
            supply = pd.concat([self.supply[self.supply["quarter"] == q],
                                pd.DataFrame({"latitude": [lat], "longitude": [lon], "quarter": [q]})], ignore_index=True)
            X = ef.features_for_rows(clean.reset_index(drop=True), supply, self.meta["zones_km"],
                                     DATA / "quarterly", self.feature_names, tenure=tenure)
            X.index = list(idx)
            parts.append(X)
        return pd.concat(parts).sort_index()

    def zone_name(self, X):
        """Name of the model's location zone for each feature row (e.g. 'Back Bay / South End')."""
        zone_cols = [f"zone_{z}" for z in range(len(self.meta["zone_names"]))]
        return [self.meta["zone_names"][str(z)] for z in X[zone_cols].values.argmax(1)]

    # ---- predictions
    def predict_log(self, X):
        """Log nightly price for a feature frame."""
        return self.booster.predict(X[self.feature_names])

    def contributions_log(self, X):
        """Exact TreeSHAP values on the log-price scale (LightGBM's own implementation).
        Columns: the 141 features plus 'base'; each row sums to the log prediction."""
        c = self.booster.predict(X[self.feature_names], pred_contrib=True)
        return pd.DataFrame(c, columns=self.feature_names + ["base"], index=X.index)

    def price(self, listings, level="90"):
        """[{price, low, high}] in dollars for a list of model listings."""
        lo, hi, _ = self.interval(level)
        p = np.exp(self.predict_log(self.features(listings)))
        return [{"price": float(v), "low": float(v * lo), "high": float(v * hi)} for v in p]


@lru_cache(maxsize=1)
def get_model():
    """Shared model instance (loading takes about a second)."""
    return PricingModel()
