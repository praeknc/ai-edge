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


def route(question, goals):
    """Pick (tool name, arguments without the listing) for a question, or (None, reply text)."""
    q = question.strip()
    if OUT_OF_SCOPE.search(q):
        return None, ("I can't help with legal, tax or insurance questions. I can estimate prices, test changes "
                      "and plan upgrades for your listing.")
    amen = find_amenities(q)
    money = dollars(q)
    budget_m = re.search(r"(budget|under|spend|up to)\D{0,12}\$\s?([\d,]+)\s*(k)?", q, re.I)
    budget = None
    if budget_m:
        budget = float(budget_m.group(2).replace(",", "")) * (1000 if budget_m.group(3) else 1)
    targets = [v for pos, v in money if not budget_m or pos < budget_m.start() or pos > budget_m.end()]

    if re.search(r"\bcost\b|how much (does|would|is|are)", q, re.I) and amen and not re.search(r"payback|roi", q, re.I):
        return "get_amenity_cost", {"name": amen[0][1]}
    goal_words = r"\b(reach|get to|target|hit|goal|raise .* to|want|charge|make|earn|possible|achieve|get)\b"
    if (targets and re.search(goal_words, q, re.I)) or \
            (re.search(r"\b(reach|target|goal)\b", q, re.I) and goals.get("target_price")):
        return "goal_seek", {"target_price": targets[0] if targets else goals["target_price"],
                             "budget": budget if budget is not None else goals.get("budget", 2000)}
    if re.search(r"\broi\b|best value|pay ?back|worth (it|investing)|invest|upgrade|which amenit|improve|increase|"
                 r"raise my|boost|what should i (change|add|do)", q, re.I):
        return "roi_rank", {"budget": budget if budget is not None else goals.get("budget")}
    settings = parse_settings(q)
    if re.search(r"what if|\badd\b|install|\bget a\b|remove|without|minimum|licen|regist|superhost|monthly", q, re.I) \
            and (amen or settings):
        rm = REMOVE_WORDS.search(q)
        add = [p for pos, p in amen if not rm or pos < rm.start()]
        remove = [p for pos, p in amen if rm and pos > rm.start()]
        args = {"add": add, "remove": remove}
        if settings:
            args["settings"] = settings
        return "what_if", args
    if re.search(r"price|charge|worth|rate|how much should|estimate", q, re.I):
        return "predict_price", {}
    return None, ("Try: \"What should I charge?\", \"What if I add a dishwasher?\", \"What if I set a 1-night minimum?\", "
                  "\"How do I get to $450 with a $2,000 budget?\" or \"Which upgrades pay back fastest?\"")


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
