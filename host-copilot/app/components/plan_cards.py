"""plan_cards: target and budget sliders, up to 3 option cards, and "Apply to what-if"."""
import streamlit as st

from app import agent
from app.components.common import add_block, add_message, blocks, footnote, interval_text, listing_caption, signed_usd, usd
from app.tools import goal_seek
from model import vocab

VERDICT_STYLE = {"already_met": "info", "reachable": "success", "possible": "info", "stretch": "warning",
                 "unlikely": "warning", "very_unlikely": "error", "out_of_range": "error"}


def recompute(bid):
    """Slider callback: run goal_seek again with the new target and budget."""
    b = blocks()[bid]
    args = {**b["args"], "target_price": st.session_state[f"{bid}-target"], "budget": st.session_state[f"{bid}-budget"]}
    b.update(goal_seek(**args), args=args)


def apply_to_whatif(bid, i):
    """Button callback: open option i in a new what-if panel (tool call, no LLM)."""
    b = blocks()[bid]
    keys = [a["key"] for a in b["data"]["options"][i]["amenities"]]
    settings = {}
    for k in keys:
        if k in vocab.LEVERS:
            settings.update(vocab.LEVERS[k][2])
    args = {"add": [k for k in keys if k in vocab.AMENITIES], "settings": settings}
    if b["args"].get("nights_per_year"):
        args["nights_per_year"] = b["args"]["nights_per_year"]
    new = agent.call_tool("what_if", args, b["args"]["listing"])
    add_message("assistant", f"Loaded option {i + 1} into a what-if panel.", [add_block(new)])


def _card(col, bid, i, o):
    with col.container(border=True):
        st.markdown(f"**Option {i + 1}**")
        st.markdown("\n".join(f"- {a['label']}" for a in o["amenities"]))
        st.metric("Price / night", usd(o["price"]["price"]), signed_usd(o["lift"]))
        st.caption(interval_text(o["price"]))
        st.markdown(f"Cost **{usd(o['cost'])}** ({usd(o['cost_low'])}–{usd(o['cost_high'])})"
                    + (f" plus {usd(o['monthly_cost'])}/month" if o["monthly_cost"] else ""))
        pay = o["payback_months"]
        st.markdown("Payback **" + ("—" if pay is None else f"{pay:g} months") + "**")
        st.button("Apply to what-if", key=f"{bid}-apply-{i}", on_click=apply_to_whatif, args=(bid, i),
                  width="stretch")


def render(bid, block):
    d = block["data"]
    base = d["base"]
    if f"{bid}-target" not in st.session_state:
        st.session_state[f"{bid}-target"] = int(d["target_price"])
        st.session_state[f"{bid}-budget"] = int(d["budget"])
    with st.container(border=True):
        listing_caption(d)
        c1, c2 = st.columns(2)
        hi = max(int(base["high"]), int(d["target_price"]))
        c1.slider("Target price / night", int(base["low"]), hi, key=f"{bid}-target", step=5, on_change=recompute,
                  args=(bid,), help=f"Today's estimate {usd(base['price'])} ({interval_text(base)})")
        c2.slider("Budget", 0, max(20000, int(d["budget"])), key=f"{bid}-budget", step=100, on_change=recompute,
                  args=(bid,))
        d = blocks()[bid]["data"]
        f = d["feasibility"]
        kind = VERDICT_STYLE[f["verdict"]]
        detail = f["text"].split(": ", 1)[-1]
        getattr(st, kind)(f"**{f['label']}.** " + detail[:1].upper() + detail[1:])
        c1, c2, c3 = st.columns(3)
        c1.metric("Chance similar listings charge it", f"{f['chance_pct']:g}%",
                  help="From the model's errors on hosts it never saw")
        p = f["peers"]
        c2.metric("Comparable listings at or above", f"{p['n_at_or_above']} of {p['n']}", help=p["description"])
        c3.metric("Best plan within budget", usd(d["best_possible"]), signed_usd(d["best_possible"] - base["price"]))
        st.caption(f"Today: {usd(base['price'])} ({interval_text(base)}). Comparable listings ({p['description']}): "
                   f"median {usd(p['median'])}, top 10% from {usd(p['p90'])}, highest {usd(p['max'])}.")
        if d["options"]:
            cols = st.columns(3)
            for i, (col, o) in enumerate(zip(cols, d["options"])):
                _card(col, bid, i, o)
        else:
            st.info("No change that can be made has a reliable effect in your area. " + d["other_levers"])
        for n in d.get("notes", []):
            st.caption(n)
        with st.expander(f"Why other changes are not in the plan ({len(d['excluded'])})"):
            st.markdown("\n".join(f"- **{e['label']}**: {e['reason']}" for e in d["excluded"]))
            st.caption(d["other_levers"])
        footnote(d)
