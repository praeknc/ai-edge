"""Amenity vocabulary of the pricing model.

The model does not read free text. `airbnb-pricing/src/enriched_features.py` turns each listing's
amenity list into 30 yes/no group flags (e.g. `am_hot_tub`), plus `am_count`, the total number of
amenity lines. Host Copilot can only reason about these 30 groups. Anything else (fast WiFi, fire pit,
sauna...) is "not in the model's vocabulary" and gets the closest supported suggestions.
"""
import difflib

# key -> (display label, extra words a host might type)
AMENITIES = {
    "am_free_parking": ("Free parking", ["parking", "free parking", "driveway", "garage"]),
    "am_paid_parking": ("Paid parking", ["paid parking", "parking garage"]),
    "am_washer_in_unit": ("In-unit washer", ["washer", "washing machine", "laundry"]),
    "am_dryer": ("Dryer", ["dryer", "tumble dryer"]),
    "am_dishwasher": ("Dishwasher", ["dishwasher"]),
    "am_central_ac": ("Central air conditioning", ["central ac", "central air"]),
    "am_any_ac": ("Air conditioning", ["ac", "a/c", "air conditioning", "aircon", "air con", "split ac", "window ac"]),
    "am_gym": ("Gym", ["gym", "fitness", "fitness center"]),
    "am_pool": ("Pool", ["pool", "swimming pool"]),
    "am_hot_tub": ("Hot tub", ["hot tub", "jacuzzi", "spa tub", "whirlpool"]),
    "am_elevator": ("Elevator", ["elevator", "lift"]),
    "am_workspace": ("Dedicated workspace", ["workspace", "desk", "office", "work space", "home office"]),
    "am_self_checkin": ("Self check-in", ["self check-in", "self checkin", "smart lock", "keypad", "lockbox", "keyless entry"]),
    "am_balcony_patio": ("Patio or balcony", ["balcony", "patio", "terrace", "deck"]),
    "am_view": ("View", ["view", "city view", "water view", "skyline view"]),
    "am_kitchen": ("Kitchen", ["kitchen"]),
    "am_bathtub": ("Bathtub", ["bathtub", "tub"]),
    "am_fireplace": ("Indoor fireplace", ["fireplace"]),
    "am_ev_charger": ("EV charger", ["ev charger", "electric vehicle charger", "car charger", "tesla charger"]),
    "am_pets": ("Pets allowed", ["pets", "pet friendly", "dogs allowed", "pets allowed"]),
    "am_long_term": ("Long-term stays allowed", ["long term stays", "long-term stays", "monthly stays"]),
    "am_luggage": ("Luggage drop-off", ["luggage", "luggage dropoff", "bag storage"]),
    "am_crib": ("Crib", ["crib", "cot", "baby crib"]),
    "am_private_entry": ("Private entrance", ["private entrance", "private entry", "separate entrance"]),
    "am_backyard": ("Backyard", ["backyard", "yard", "garden"]),
    "am_bbq": ("BBQ grill", ["bbq", "grill", "barbecue"]),
    "am_tv": ("TV", ["tv", "television", "smart tv", "hdtv"]),
    "am_streaming": ("Streaming services", ["netflix", "streaming", "hulu", "disney+", "chromecast", "roku"]),
    "am_game_console": ("Game console", ["game console", "ps5", "xbox", "nintendo", "playstation"]),
    "am_sound_system": ("Sound system", ["sound system", "bluetooth speaker", "speaker"]),
}

# Common requests the model cannot see, with the nearest supported groups.
NOT_IN_MODEL = {
    "fast wifi": ["am_workspace"], "wifi": ["am_workspace"], "internet": ["am_workspace"],
    "fire pit": ["am_bbq", "am_backyard"], "firepit": ["am_bbq", "am_backyard"],
    "sauna": ["am_hot_tub"], "heating": ["am_any_ac"], "heat": ["am_any_ac"],
    "coffee maker": ["am_kitchen"], "espresso machine": ["am_kitchen"], "microwave": ["am_kitchen"],
    "smart tv": ["am_tv", "am_streaming"], "projector": ["am_tv", "am_streaming"],
    "parking spot": ["am_free_parking"], "ev charging": ["am_ev_charger"],
}

_LOOKUP = {}
for _key, (_label, _words) in AMENITIES.items():
    for _w in [_key, _key[3:].replace("_", " "), _label] + _words:
        _LOOKUP[_w.lower().strip()] = _key


def label(key):
    """Display label for an amenity or lever key, e.g. 'am_hot_tub' -> 'Hot tub'."""
    return AMENITIES[key][0] if key in AMENITIES else LEVERS[key][0]


def resolve(name):
    """Map what a host typed ('hot tub', 'Jacuzzi', 'am_hot_tub') to a model key, or None."""
    text = str(name).lower().strip()
    if text in NOT_IN_MODEL:
        return None
    return _LOOKUP.get(text)


def suggest(name, n=2):
    """Closest supported amenity keys for a name the model does not know."""
    text = str(name).lower().strip()
    if text in NOT_IN_MODEL:
        return NOT_IN_MODEL[text][:n]
    close = difflib.get_close_matches(text, list(_LOOKUP), n=5, cutoff=0.5)
    keys = []
    for c in close:
        if _LOOKUP[c] not in keys:
            keys.append(_LOOKUP[c])
    return keys[:n]


def resolve_many(names):
    """Split a list of names into (known keys, unknown [{name, suggestions}])."""
    known, unknown = [], []
    for nm in names or []:
        key = resolve(nm)
        if key:
            if key not in known:
                known.append(key)
        else:
            unknown.append({"name": str(nm), "suggestions": [{"key": k, "label": label(k)} for k in suggest(nm)]})
    return known, unknown


# How a host could add each amenity. goal_seek and roi_rank plan only with "purchase" and "policy";
# "building" features (elevator, pool, garage...) are shown as context, not as something to buy.
KIND = {
    "am_washer_in_unit": "purchase", "am_dryer": "purchase", "am_dishwasher": "purchase", "am_central_ac": "purchase",
    "am_any_ac": "purchase", "am_hot_tub": "purchase", "am_workspace": "purchase", "am_self_checkin": "purchase",
    "am_ev_charger": "purchase", "am_crib": "purchase", "am_bbq": "purchase", "am_tv": "purchase",
    "am_streaming": "purchase", "am_game_console": "purchase", "am_sound_system": "purchase",
    "am_pets": "policy", "am_long_term": "policy", "am_luggage": "policy",
    "am_free_parking": "building", "am_paid_parking": "building", "am_gym": "building", "am_pool": "building",
    "am_elevator": "building", "am_balcony_patio": "building", "am_view": "building", "am_kitchen": "building",
    "am_bathtub": "building", "am_fireplace": "building", "am_private_entry": "building", "am_backyard": "building",
}


def kind(key):
    """'purchase', 'policy' (free rule change), 'registration', 'earned' (superhost: not a direct choice)
    or 'building' (not realistically added by a host)."""
    return KIND[key]


# Non-amenity levers a host controls. Each is one fixed move, so its effect can be checked the same
# way as an amenity (analysis/change_effects.py). Minimum-nights moves start from a band:
#   to 1 night (from 2-27), to 2 nights (from 3-27), to 5 nights (from 1-2),
#   monthly = 30 nights (from 1-27), nightly = 2 nights (from 28+).
LEVERS = {
    "lv_min_nights_1": ("Minimum stay 1 night", "policy", {"minimum_nights": 1}),
    "lv_min_nights_2": ("Minimum stay 2 nights", "policy", {"minimum_nights": 2}),
    "lv_min_nights_5": ("Minimum stay 5 nights", "policy", {"minimum_nights": 5}),
    "lv_monthly": ("Monthly stays only (30 nights)", "policy", {"minimum_nights": 30}),
    "lv_nightly": ("Switch to nightly stays (2 nights)", "policy", {"minimum_nights": 2}),
    "lv_superhost": ("Superhost status", "earned", {"superhost": True}),
    "lv_licensed": ("Registered (licence number)", "registration", {"license": "licensed"}),
    "lv_verified": ("Verified identity", "policy", {"identity_verified": True}),
}
KIND.update({k: v[1] for k, v in LEVERS.items()})


def lever_for(field, current, target):
    """The lever key that matches changing one listing field, or None.
    Minimum nights: the move is chosen from the current and target values (see LEVERS)."""
    if field == "minimum_nights":
        if current >= 28 and target <= 27:
            return "lv_nightly"
        if current <= 27 and target >= 28:
            return "lv_monthly"
        if current <= 27 and target < current:
            return "lv_min_nights_1" if target == 1 else "lv_min_nights_2"
        if current <= 27 and target > current:
            return "lv_min_nights_5"
        return None
    for key, (_, _, setting) in LEVERS.items():
        if field in setting and setting[field] == target:
            return key
    return None
