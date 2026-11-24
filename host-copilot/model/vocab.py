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
    """Display label for an amenity key, e.g. 'am_hot_tub' -> 'Hot tub'."""
    return AMENITIES[key][0]


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
