"""Create 5 synthetic (fake) listings for the demo host, in the Inside Airbnb format of the training data.

None of these listings exists or was used for training. Each one is designed as a host profile (situations a
host copilot should handle: underpriced, unregistered, shared bath, monthly-only, family house), then written
as a full 90-column Inside Airbnb row with the same converter the app uses (model/loader.raw_row), plus
readable titles, descriptions and generic amenity lines.

Checks before saving:
  * the project's own pipeline (pricing_features.precheck + enriched_features.features_for_rows) reads each
    CSV row into exactly the features the app builds from the host profile;
  * no id or host id appears in the training data, and no training listing sits at the same spot.

Outputs (host-copilot/data/):
  synthetic_listings.csv   5 rows, same 90 columns as listings_detailed_*.csv.gz
  host_portfolio.json      the demo host and the 5 listings as host profiles (what the app loads)
Run:  python scripts/make_synthetic_listings.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.loader import DATA, ef, get_model, pf, raw_row      # noqa: E402

SEED = 7
OUT = ROOT / "data"
HOST = {"host_id": 990000001, "host_name": "Demo Host", "host_since": "2022-03-14"}
SAMPLE = json.loads((ROOT / "data" / "sample_listing.json").read_text())

# Generic amenity lines that belong to none of the model's 30 groups (used to fill the amenity count).
GENERIC = ["Wifi", "Hair dryer", "Hangers", "Iron", "Smoke alarm", "Carbon monoxide alarm", "Hot water", "Shampoo",
           "Bed linens", "Extra pillows and blankets", "Microwave", "Refrigerator", "Coffee maker", "Dishes and silverware",
           "Cooking basics", "Oven", "Stove", "Toaster", "Wine glasses", "Heating", "Fire extinguisher", "First aid kit",
           "Room-darkening shades", "Body soap", "Conditioner", "Shower gel", "Cleaning products", "Clothing storage",
           "Essentials", "Freezer", "Kettle", "Dining table", "Laundromat nearby", "Ethernet connection",
           "Outdoor furniture", "Books and reading material", "Board games", "Window guards", "Ceiling fan",
           "Portable fans", "Lock on bedroom door", "Single level home", "Baking sheet", "Rice maker", "Blender",
           "Paid washer – In building", "Exercise equipment", "Bidet", "Safe", "Hot water kettle"]

PERSONAS = [
    {"nickname": "South End condo", "title": "Sunny 2BR renovated condo steps from Tremont St",
     "story": "Underpriced: listed well below what similar listings charge.",
     "listing": {**SAMPLE, "host_listings": 5, "description_length": 448}},
    {"nickname": "Downtown 1BR", "title": "Downtown 1BR by Boston Common",
     "story": "New listing, not registered yet, 3-night minimum.",
     "listing": {"neighbourhood": "Downtown", "room_type": "Entire home/apt", "property_group": "rental unit",
                 "accommodates": 2, "bedrooms": 1, "beds": 1, "baths": 1, "baths_shared": False,
                 "amenities": ["kitchen", "tv", "self check-in", "air conditioning"], "amenity_count": 28,
                 "title_keywords": [], "season": "fall", "night_of_week": "friday", "minimum_nights": 3,
                 "availability_365": 220, "license": "none", "host_listings": 5, "superhost": False,
                 "identity_verified": True, "number_of_reviews": 8, "reviews_per_month": 1.1,
                 "number_of_reviews_ltm": 8, "review_score": 4.7, "months_since_first_review": 7,
                 "days_since_review": 12, "tenure_quarters": 2, "description_length": 310, "current_price": 340}},
    {"nickname": "Dorchester room", "title": "Quiet private room (1 bedroom) in a Dorchester house",
     "story": "Private room with a shared bathroom.",
     "listing": {"neighbourhood": "Dorchester", "room_type": "Private room", "property_group": "room in house",
                 "accommodates": 2, "bedrooms": 1, "beds": 1, "baths": 1, "baths_shared": True,
                 "amenities": ["kitchen", "tv", "in-unit washer", "dryer", "workspace", "free parking"],
                 "amenity_count": 34, "title_keywords": [], "season": "fall", "night_of_week": "friday",
                 "minimum_nights": 2, "availability_365": 300, "license": "licensed", "host_listings": 5,
                 "superhost": False, "identity_verified": True, "number_of_reviews": 60, "reviews_per_month": 2.4,
                 "number_of_reviews_ltm": 22, "review_score": 4.85, "months_since_first_review": 25,
                 "days_since_review": 9, "tenure_quarters": 3, "description_length": 420, "current_price": 85}},
    {"nickname": "East Boston monthly", "title": "East Boston 2BR for monthly stays near the Blue Line",
     "story": "Monthly-only (30-night minimum) and not registered.",
     "listing": {"neighbourhood": "East Boston", "room_type": "Entire home/apt", "property_group": "rental unit",
                 "accommodates": 4, "bedrooms": 2, "beds": 2, "baths": 1, "baths_shared": False,
                 "amenities": ["kitchen", "in-unit washer", "dryer", "tv", "workspace", "long term stays"],
                 "amenity_count": 31, "title_keywords": [], "season": "fall", "night_of_week": "friday",
                 "minimum_nights": 30, "availability_365": 250, "license": "none", "host_listings": 5,
                 "superhost": False, "identity_verified": True, "number_of_reviews": 12, "reviews_per_month": 0.6,
                 "number_of_reviews_ltm": 5, "review_score": 4.6, "months_since_first_review": 20,
                 "days_since_review": 45, "tenure_quarters": 3, "description_length": 380, "current_price": 150}},
    {"nickname": "Jamaica Plain house", "title": "Jamaica Plain 3BR family house with backyard and parking",
     "story": "Large family house, superhost, many reviews.",
     "listing": {"neighbourhood": "Jamaica Plain", "room_type": "Entire home/apt", "property_group": "house",
                 "accommodates": 6, "bedrooms": 3, "beds": 4, "baths": 2, "baths_shared": False,
                 "amenities": ["kitchen", "in-unit washer", "dryer", "dishwasher", "backyard", "bbq", "free parking",
                               "crib", "tv", "air conditioning"], "amenity_count": 45,
                 "title_keywords": ["parking"], "season": "fall", "night_of_week": "friday", "minimum_nights": 2,
                 "availability_365": 180, "license": "licensed", "host_listings": 5, "superhost": True,
                 "identity_verified": True, "number_of_reviews": 140, "reviews_per_month": 3.1,
                 "number_of_reviews_ltm": 35, "review_score": 4.95, "months_since_first_review": 45,
                 "days_since_review": 6, "tenure_quarters": 3, "description_length": 500, "current_price": 380}},
]

DESCRIPTION = ("Welcome to our {kind} in {hood}. The space has {bed} bedroom(s) and sleeps {guests}. You'll find a "
               "fully set up home with fresh linens, fast check-in instructions and local tips for coffee, parks and "
               "the T. We look after every stay ourselves and reply quickly. ")


def place(rng, hood, centers, ref, min_gap_m=20):
    """A point about 100-250 m from the neighbourhood's median listing location, redrawn until it is at least
    `min_gap_m` metres from every training listing."""
    lat0, lon0 = centers.loc[hood, "latitude"], centers.loc[hood, "longitude"]
    while True:
        angle, dist_km = rng.uniform(0, 2 * np.pi), rng.uniform(0.1, 0.25)
        lat = round(float(lat0 + dist_km * np.sin(angle) / 111.0), 5)
        lon = round(float(lon0 + dist_km * np.cos(angle) / 82.0), 5)
        gap = np.sqrt(((ref.latitude - lat) * 111000) ** 2 + ((ref.longitude - lon) * 82000) ** 2).min()
        if gap >= min_gap_m:
            return lat, lon


def description(L, length):
    text = DESCRIPTION.format(kind=L["property_group"], hood=L["neighbourhood"], bed=L["bedrooms"],
                              guests=L["accommodates"])
    while len(text) < length:
        text += text
    return text[:length].rstrip()[: length]


def to_inside_airbnb(L, persona, listing_id, rng, columns):
    """One 90-column Inside Airbnb row for a completed model listing."""
    row = raw_row(L, 0, get_model().meta["snapshot_dates"])
    labelled = json.loads(row["amenities"])
    real = [a for a in labelled if not a.startswith("Essentials item")]
    filler = list(rng.choice(GENERIC, size=len(labelled) - len(real), replace=False))
    row.update({
        "id": listing_id, "listing_url": f"https://example.invalid/rooms/{listing_id}", "source": "synthetic",
        "name": persona["title"], "description": description(L, L["desc_len"]).ljust(L["desc_len"], "."),
        "amenities": json.dumps(real + filler), "price": f"${persona['listing']['current_price']:.2f}",
        "host_id": HOST["host_id"], "host_name": HOST["host_name"], "host_since": HOST["host_since"],
        "calculated_host_listings_count": L["host_listings"], "host_listings_count": L["host_listings"],
        "has_availability": "t", "instant_bookable": "f", "property_type": row["property_type"],
    })
    row.pop("quarter", None)
    return {c: row.get(c, np.nan) for c in columns}


def check_parity(csv_path, listings, model):
    """The project's pipeline must read each CSV row (with its own loader) into the same features the app builds."""
    df = ef.load_detailed(csv_path)
    for i, (_, r) in enumerate(df.iterrows()):
        q = {v: k for k, v in model.meta["snapshot_dates"].items()}[r["last_scraped"]]
        raw = pd.DataFrame([{**r.to_dict(), "quarter": q}])
        clean, _ = pf.precheck(raw, model.official, model.geo)
        assert len(clean) == 1, f"listing {i} failed the pre-check"
        L = listings[i]
        supply = pd.concat([model.supply[model.supply.quarter == q],
                            pd.DataFrame({"latitude": [L["lat"]], "longitude": [L["lon"]], "quarter": [q]})],
                           ignore_index=True)
        X_csv = ef.features_for_rows(clean.reset_index(drop=True), supply, model.meta["zones_km"], DATA / "quarterly",
                                     model.feature_names, tenure=L["tenure_q"])
        X_app = model.features([L])
        np.testing.assert_allclose(X_csv.values.astype(float), X_app.values.astype(float), rtol=1e-9, equal_nan=True,
                                   err_msg=f"listing {i}: CSV row and app features differ")


def check_not_training(df):
    """Synthetic ids, host and locations must not coincide with training listings."""
    ref = pd.read_csv(ROOT / "model" / "artifacts" / "reference.csv.gz")
    assert not set(df["id"]) & set(ref["id"]), "a synthetic id exists in the training data"
    for _, r in df.iterrows():
        d_m = np.sqrt(((ref.latitude - r.latitude) * 111000) ** 2 + ((ref.longitude - r.longitude) * 82000) ** 2).min()
        assert d_m >= 20, f"synthetic listing {r['id']} is within 20 m of a training listing"
        print(f"  {r['id']}: nearest training listing {d_m:.0f} m away")


def main():
    rng = np.random.default_rng(SEED)
    model = get_model()
    columns = list(pd.read_csv(DATA / "detailed" / "listings_detailed_2026Q3_sep.csv.gz", nrows=1).columns)
    ref = pd.read_csv(ROOT / "model" / "artifacts" / "reference.csv.gz")
    rows, listings, portfolio = [], [], []
    for i, p in enumerate(PERSONAS):
        lat, lon = place(rng, p["listing"]["neighbourhood"], model.hood_centers, ref)
        host_listing = {**p["listing"], "latitude": lat, "longitude": lon, "amenity_count": p["listing"].get("amenity_count")}
        L, filled = model.complete_listing(host_listing)
        listing_id = 990000000000000001 + i
        rows.append(to_inside_airbnb(L, p, listing_id, rng, columns))
        listings.append(L)
        portfolio.append({"id": str(listing_id), "nickname": p["nickname"], "title": p["title"], "story": p["story"],
                          "listing": {**host_listing, "amenity_count": L["am_count"]}, "filled_fields": filled})
    df = pd.DataFrame(rows, columns=columns)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "synthetic_listings.csv"
        df.to_csv(path, index=False)
        check_parity(path, listings, model)
    check_not_training(df)
    df.to_csv(OUT / "synthetic_listings.csv", index=False)
    (OUT / "host_portfolio.json").write_text(json.dumps({"host": HOST, "listings": portfolio}, indent=1) + "\n")
    print(f"wrote {len(df)} synthetic listings ({len(columns)} columns) and host_portfolio.json")
    for p in portfolio:
        price = model.price([model.complete_listing(p["listing"])[0]])[0]
        print(f"  {p['nickname']:22s} listed ${p['listing']['current_price']:>4}  model ${price['price']:.0f} "
              f"(${price['low']:.0f}-${price['high']:.0f})  filled: {p['filled_fields']}")


if __name__ == "__main__":
    main()
