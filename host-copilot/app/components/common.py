"""Shared helpers for UI components: block storage, money formatting, footnotes."""
import itertools

import streamlit as st

_ids = itertools.count()


def usd(v):
    """$1,234."""
    return f"${int(round(v)):,}"


def signed_usd(v):
    """+$13 / −$8."""
    v = int(round(v))
    return f"{'+' if v >= 0 else '-'}${abs(v):,}"          # ASCII minus: Streamlit reads it as a decrease


def interval_text(p):
    """'90% range $324–$942' for a {price, low, high} dict."""
    return f"90% range {usd(p['low'])}–{usd(p['high'])}"


def blocks():
    """All UI blocks of this session: id -> {tool, args, component, data}."""
    return st.session_state.setdefault("blocks", {})


def add_block(block):
    """Store a tool result as a UI block and return its id."""
    bid = f"b{len(blocks())}-{next(_ids)}"
    blocks()[bid] = block
    return bid


def add_message(role, text, block_ids=()):
    """Append a chat message that shows some blocks under its text."""
    st.session_state.setdefault("messages", []).append({"role": role, "text": text, "blocks": list(block_ids)})


def listing_caption(d):
    """Which listing a block was computed for (blocks keep their own copy of the listing)."""
    s = d.get("listing", {})
    if not s:
        return
    room = s.get("room_type", "").replace("Entire home/apt", "entire home").lower()
    st.caption(f"For: {s.get('neighbourhood')} · {room} · {s.get('bedrooms')} bedroom(s) · {s.get('season')} · "
               f"area zone {s.get('zone')}")


def footnote(d):
    """Caveat, interval method and filled-in fields, small and grey."""
    parts = [d.get("caveat", "")]
    if d.get("interval"):
        parts.append(f"90% range = prediction × {d['interval']['low_mult']:.2f} to × {d['interval']['high_mult']:.2f}: "
                     + d["interval"]["method"])
    if d.get("filled_fields"):
        parts.append("Filled in from similar listings: " + ", ".join(f.replace("_", " ") for f in d["filled_fields"]) + ".")
    if d.get("costs_are_assumptions"):
        parts.append("Costs are placeholder assumptions (data/amenity_costs.csv).")
    st.caption(" ".join(p for p in parts if p))
