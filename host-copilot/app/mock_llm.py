"""Mock LLM: routes a host's question to a tool with keyword rules, then narrates the result.

Used when MOCK_LLM=1 or no ANTHROPIC_API_KEY is set, so the UI can be built and demoed offline.
It understands a handful of question shapes, for example:
  "What should I charge?"                         -> predict_price
  "What if I add a hot tub and a dishwasher?"     -> what_if
  "What if I set a 1-night minimum and register?" -> what_if (levers)
  "How do I get to $450 with a $2,000 budget?"    -> goal_seek
  "Which upgrades pay back fastest?"              -> roi_rank
  "How much does an EV charger cost?"             -> get_amenity_cost
The narration only repeats numbers found in the tool output.
"""
import re

from model import vocab

OUT_OF_SCOPE = re.compile(r"\btax(es)?\b|\blawyer\b|legal advice|\bsue\b|\binsurance\b|\bzoning\b|\bmortgage\b", re.I)
REMOVE_WORDS = re.compile(r"\b(remove|without|drop|get rid of|take out)\b", re.I)
_PHRASES = sorted(list(vocab._LOOKUP) + list(vocab.NOT_IN_MODEL), key=len, reverse=True)


def find_amenities(text):
    """Amenity phrases in the text, longest first, as (position, phrase). Includes unsupported ones
    (fast wifi, fire pit...) so the tool can answer with suggestions."""
    found, taken = [], []
    low = text.lower()
    for ph in _PHRASES:
        if len(ph) < 2 or ph.startswith("am_"):
            continue
        for m in re.finditer(r"(?<![a-z])" + re.escape(ph) + r"(?![a-z])", low):
            span = range(m.start(), m.end())
            if not any(set(span) & set(t) for t in taken):
                found.append((m.start(), ph))
                taken.append(span)
    return sorted(found)


def dollars(text):
    """Dollar amounts in the text ('$450', '$2,000', '3k' after a $ or 'budget')."""
    out = []
    for m in re.finditer(r"\$\s?([\d,]+(?:\.\d+)?)\s*(k)?", text, re.I):
        v = float(m.group(1).replace(",", ""))
        out.append((m.start(), v * 1000 if m.group(2) else v))
    return out


def parse_settings(text):
    """Lever settings mentioned in the text."""
    s = {}
    m = re.search(r"(\d+)[- ]?night", text, re.I)
    if m:
        s["minimum_nights"] = int(m.group(1))
    elif re.search(r"\bmonthly\b", text, re.I):
        s["minimum_nights"] = 30
    if re.search(r"licen[cs]e|regist", text, re.I):
        s["license"] = "licensed"
    if re.search(r"superhost", text, re.I):
        s["superhost"] = True
    return s


HELP = re.compile(r"what can (you|i)|^\s*help\s*\??$|how does (this|it) work|what do you do|^\s*(options|menu)\b", re.I)
INTENT_WORDS = {
    "predict_price": [(r"what should i charge|how much should i (charge|ask)|my price|nightly (price|rate)", 3),
                      (r"\bprice\b|\bcharge\b|\brate\b|worth|estimate|value|pricing", 1)],
    "what_if": [(r"what if|what happens if|should i add|if i (add|remove|install|drop|get|switch|set|go)", 3),
                (r"\badd\b|install|\bput in\b|\bbuy\b|\bget a\b|remove|without|drop|switch|change my|set a", 1)],
    "goal_seek": [(r"\breach\b|\btarget\b|\bgoal\b|get to|hit|achieve|is it possible|can i (get|make|charge|earn)", 3),
                  (r"\bwant\b|make|earn|charge", 1)],
    "roi_rank": [(r"\broi\b|pay ?back|best value|fastest|which (upgrade|amenit|change)|what should i (add|change|do|improve)", 3),
                 (r"upgrade|invest|improve|increase|boost|raise|worth (it|investing)|best", 1)],
    "get_amenity_cost": [(r"how much (does|do|would|is|are) (a|an|the)? ?\w+( \w+)? cost|\bcost of\b|price of|"
                          r"how much (is|are|for) (a|an|the)? ?\w+", 3),
                         (r"\bcost\b|\bcosts\b|expensive|cheap", 1)],
}
CLARIFY_TEXT = "I'm not sure which of these you mean. Pick one below, or rephrase with a little more detail."


def score_intents(q, amen, settings, targets):
    """Score each tool for a question: keyword weights plus what the question mentions."""
    sc = {t: sum(w for pat, w in pats if re.search(pat, q, re.I)) for t, pats in INTENT_WORDS.items()}
    if amen:
        sc["what_if"] += 2
        if re.search(r"worth|help|pay off|good idea|should i (add|get|buy|install)", q, re.I):
            sc["what_if"] += 2              # judging one amenity: the what-if panel shows its effect, cost and payback
        sc["get_amenity_cost"] += 1 if sc["get_amenity_cost"] else 0
    if settings:
        sc["what_if"] += 2
    if targets:
        sc["goal_seek"] += 2
        sc["predict_price"] -= 1
    if not amen:
        sc["get_amenity_cost"] = 0          # a cost lookup needs an amenity
    if not (amen or settings):
        sc["what_if"] = min(sc["what_if"], 1)
    return sc


def route(question, goals):
    """Pick (tool, arguments without the listing) for a question.
    Returns (None, reply text) for help and out-of-scope questions, and ("clarify", [tools]) when unsure."""
    q = question.strip()
    if OUT_OF_SCOPE.search(q):
        return None, ("I can't help with legal, tax or insurance questions. I can estimate prices, test changes "
                      "and plan upgrades for your listing.")
    if HELP.search(q):
        return None, "Here is what I can do. Pick one below, or ask in your own words."
    amen = find_amenities(q)
    money = dollars(q)
    budget_m = re.search(r"(budget|under|spend|up to|with)\D{0,12}\$\s?([\d,]+)\s*(k)?", q, re.I)
    budget = None
    if budget_m:
        budget = float(budget_m.group(2).replace(",", "")) * (1000 if budget_m.group(3) else 1)
    targets = [v for pos, v in money if not budget_m or pos < budget_m.start() or pos > budget_m.end()]
    settings = parse_settings(q)

    sc = score_intents(q, amen, settings, targets)
    ranked = sorted(sc, key=sc.get, reverse=True)
    best, second = ranked[0], ranked[1]
    clear_winner = sc[best] >= 2 and sc[best] - sc[second] >= 1
    only_match = sc[best] >= 1 and sc[second] <= 0
    if not (clear_winner or only_match):
        return "clarify", [t for t in ranked if sc[t] > 0][:3] or ranked[:3]

    if best == "get_amenity_cost":
        return best, {"name": amen[0][1]}
    if best == "goal_seek":
        target = targets[0] if targets else goals.get("target_price")
        if not target:
            return "clarify", ["goal_seek"]
        return best, {"target_price": target, "budget": budget if budget is not None else goals.get("budget", 2000)}
    if best == "roi_rank":
        return best, {"budget": budget if budget is not None else goals.get("budget")}
    if best == "what_if":
        rm = REMOVE_WORDS.search(q)
        args = {"add": [p for pos, p in amen if not rm or pos < rm.start()],
                "remove": [p for pos, p in amen if rm and pos > rm.start()]}
        if settings:
            args["settings"] = settings
        return best, args
    return "predict_price", {}


def usd(v):
    """$1,234 formatting used in every narration."""
    return f"${int(v):,}"


def _range(p):
    return f"{usd(p['low'])}–{usd(p['high'])}"


def _unsupported(d):
    parts = []
    for u in d.get("unsupported", []):
        alt = ", ".join(s["label"] for s in u["suggestions"]) or "none"
        parts.append(f"{u['name']} isn't something the model can see (closest: {alt})")
    return ("; ".join(parts) + ".") if parts else ""


def narrate(tool, result):
    """One or two sentences built only from numbers in the tool result."""
    c, d = result["component"], result["data"]
    if c == "error":
        return d["message"]
    if tool == "predict_price":
        s = f"Listings like yours charge about {usd(d['price'])} a night (90% range {_range(d)})."
        if "current_price" in d:
            s += f" Your {usd(d['current_price'])} is {d['current_vs_range']} that range."
        return s
    if tool == "what_if":
        extra = _unsupported(d)
        if not d["changes"] and not d["already_present"] and not d.get("unsupported"):
            return (f"Here is your what-if panel at {usd(d['base']['price'])} a night (range {_range(d['base'])}). "
                    "Click amenities or change the levers to re-price.")
        if not d["changes"]:
            already = ", ".join(d["already_present"])
            return " ".join(x for x in [f"Already in place: {already}." if already else "", extra] if x) or \
                "I didn't find a change I can test in that question."
        s = (f"With these changes the estimate moves from {usd(d['base']['price'])} to {usd(d['new']['price'])} a night "
             f"(range {_range(d['new'])}).")
        if d["already_present"]:
            s += f" Already in place: {', '.join(d['already_present'])}."
        if any(c["market_switch"] for c in d["contributions"] + ((d["no_reliable_effect"] or {}).get("items") or [])):
            s += " That switches between monthly and nightly stays, a different market."
        elif not d["contributions"]:
            s += f" None of them has a reliable effect in {d['listing']['neighbourhood']}."
        return " ".join(x for x in [s, extra] if x)
    if tool == "goal_seek":
        f = d["feasibility"]
        if d["status"] == "reachable":
            o = d["options"][0]
            names = ", ".join(a["label"] for a in o["amenities"])
            return (f"Reachable: the cheapest plan to reach {usd(d['target_price'])} is {names} for about {usd(o['cost'])} "
                    f"(estimate {usd(o['price']['price'])}, range {_range(o['price'])}).")
        return f["text"]
    if tool == "roi_rank":
        if not d["items"]:
            return f"No change you can make has a reliable price effect in {d['listing']['neighbourhood']}."
        i = d["items"][0]
        return f"Fastest payback here: {i['label']}, about +{usd(i['lift_per_night'])} a night for about {usd(i['cost'])}."
    if tool == "get_amenity_cost":
        if not d["found"]:
            return f"I have no cost assumption for {d['name']}."
        return f"{d['name']}: assumed {usd(d['cost_low'])}–{usd(d['cost_high'])} upfront (a placeholder, not a quote)."
    return ""
