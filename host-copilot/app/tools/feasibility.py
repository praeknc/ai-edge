"""How realistic is a target price? A plain-language verdict built from three pieces of evidence.

1. Model chance: the model's estimate for the listing, combined with how far real prices landed from the
   model's estimates on held-out hosts (out-of-fold errors, model/artifacts/meta.json), gives the share of
   listings like this one that charge at least the target.
2. Comparable listings: real listings in the training data with the same neighbourhood, room type and
   bedroom count (widened when there are fewer than 15), and how many of them charge at least the target.
3. Best plan: how much of the gap the best plan of changes closes (from goal_seek).

Verdicts, from most to least likely:
  already_met    today's estimate is at or above the target
  reachable      a plan within budget reaches it
  possible       >= 25% chance: similar listings often charge this; the gap is in things the model can't see
  stretch        10-25% chance
  unlikely       2-10% chance
  very_unlikely  < 2% chance
  out_of_range   above every comparable listing in the data, beyond every error the model made on held-out
                 hosts, or above the $2,000 price cap used to train the model: the data can't support an estimate
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path(__file__).resolve().parents[2] / "model" / "artifacts"
PRICE_CAP = 2000          # training data kept prices from $30 to $2,000 a night (pricing_features.PRICE_RANGE)
MIN_PEERS = 15
LABELS = {"already_met": "Already met", "reachable": "Reachable", "possible": "Possible",
          "stretch": "A stretch", "unlikely": "Unlikely", "very_unlikely": "Very unlikely",
          "out_of_range": "Outside what the data can estimate"}


@lru_cache(maxsize=1)
def reference():
    """Training listings with their real nightly prices."""
    return pd.read_csv(ART / "reference.csv.gz")


def chance_at_least(model, predicted, target):
    """Share of listings like this one priced at or above `target`, from the out-of-fold error spread.
    0.0 when the target is further above the estimate than any error the model made on held-out hosts."""
    q = np.asarray(model.meta["residual_quantiles"])
    x = np.log(target / predicted)
    if x > q[-1]:
        return 0.0
    if x <= q[0]:
        return 1.0
    return float(1 - np.interp(x, q, np.linspace(0, 1, len(q))))


ROOM_WORDS = {"Entire home/apt": "entire homes", "Private room": "private rooms", "Hotel room": "hotel rooms"}
NUMBER_WORDS = ["studio", "one-bedroom", "two-bedroom", "three-bedroom", "four-or-more-bedroom"]


def comparable_listings(L):
    """Real listings like this one: same neighbourhood, room type and bedrooms, widened step by step
    when fewer than MIN_PEERS match. Returns (prices, description)."""
    ref = reference()
    room = "Private room" if L["room_type"] == "Shared room" else L["room_type"]
    beds = min(L["bedrooms"], 4)
    ref_beds = ref["bedrooms"].clip(upper=4)
    kind = f"{NUMBER_WORDS[beds]} {ROOM_WORDS[room]}" if room == "Entire home/apt" else ROOM_WORDS[room]
    steps = [
        ((ref.neighbourhood == L["neighbourhood"]) & (ref.room_type == room) & (ref_beds == beds),
         f"{kind} in {L['neighbourhood']}"),
        ((ref.neighbourhood == L["neighbourhood"]) & (ref.room_type == room),
         f"{ROOM_WORDS[room]} in {L['neighbourhood']}"),
        ((ref.room_type == room) & (ref_beds == beds), f"{kind} across Boston"),
    ]
    for mask, desc in steps:
        if mask.sum() >= MIN_PEERS:
            return ref.loc[mask, "price"].to_numpy(), desc
    mask, desc = steps[-1]
    return ref.loc[mask, "price"].to_numpy(), desc


def _one_in(chance):
    """'about 1 in 8' for a chance between 0 and 1."""
    return f"about 1 in {max(2, int(round(1 / chance)))}" if chance > 0 else "none"


def assess_target(model, L, base_price, target, best_price=None, plan_reaches=False):
    """Verdict, evidence and one plain sentence for a target nightly price."""
    target, base_price = float(target), float(base_price)
    chance = chance_at_least(model, base_price, target)
    prices, peer_desc = comparable_listings(L)
    n_peers, n_at = len(prices), int((prices >= target).sum())
    gap = target - base_price
    closed = None if best_price is None else max(0.0, float(best_price) - base_price)

    if base_price >= target:
        verdict = "already_met"
    elif plan_reaches:
        verdict = "reachable"
    elif target > PRICE_CAP or chance == 0.0 or (n_peers and target > prices.max()):
        verdict = "out_of_range"
    elif chance >= 0.25:
        verdict = "possible"
    elif chance >= 0.10:
        verdict = "stretch"
    elif chance >= 0.02:
        verdict = "unlikely"
    else:
        verdict = "very_unlikely"

    t, b = f"${target:,.0f}", f"${base_price:,.0f}"
    peers = f"{n_at} of {n_peers} {peer_desc} in the data do"
    if verdict == "already_met":
        text = f"Already met: listings like yours are estimated at {b}, at or above {t}."
    elif verdict == "reachable":
        text = f"Reachable: a plan within your budget gets the estimate from {b} to {t} or more."
    elif verdict == "out_of_range":
        reason = (f"above the ${PRICE_CAP:,} cap of the prices the model was trained on" if target > PRICE_CAP else
                  f"higher than any of the {n_peers} {peer_desc} in the data (top price ${prices.max():,.0f})"
                  if n_peers and target > prices.max() else
                  "further above the estimate than any listing priced in the model's tests")
        text = (f"Outside what the data can estimate: {t} is {reason}, so the model has no evidence a listing like "
                f"yours earns it (today's estimate {b}).")
    else:
        text = (f"{LABELS[verdict]}: listings like yours are estimated at {b}, and {_one_in(chance)} similar listings "
                f"charge {t} or more ({peers}).")
        if closed is not None:
            text += f" The changes you can make close ${closed:,.0f} of the ${gap:,.0f} gap"
            text += ("; the rest comes from things like size, finish and reviews, not upgrades." if verdict != "possible"
                     else "; listings that charge it usually differ in size, finish or reviews.")
    return {"verdict": verdict, "label": LABELS[verdict], "chance_pct": round(100 * chance, 1),
            "peers": {"description": peer_desc, "n": n_peers, "n_at_or_above": n_at,
                      "median": round(float(np.median(prices))) if n_peers else None,
                      "p90": round(float(np.quantile(prices, 0.9))) if n_peers else None,
                      "max": round(float(prices.max())) if n_peers else None},
            "gap": round(gap), "closed_by_plan": None if closed is None else round(closed), "text": text}
