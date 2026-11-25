"""whatif_panel: toggle amenities and levers; the panel re-prices through the what_if tool directly
(no LLM). Shows 4 metric cards, the contribution of each change, and caveats."""
import plotly.graph_objects as go
import streamlit as st

from app.components.common import blocks, footnote, interval_text, listing_caption, signed_usd, usd
from app.tools import what_if
from model import vocab

LICENSES = ["licensed", "exempt", "none"]


def _base(d):
    """The listing's starting amenities and lever values (before any change)."""
    s = d["listing"]
    return set(s["amenities"]), {"minimum_nights": s["minimum_nights"], "license": s["license"], "superhost": s["superhost"]}


def recompute(bid):
    """Read this panel's widgets and call what_if again."""
    b = blocks()[bid]
    base_amen, base_lev = _base(b["data"])
    chosen = set(st.session_state[f"{bid}-am"] or [])
    settings = {f: st.session_state[f"{bid}-{f}"] for f in base_lev}
    settings = {f: v for f, v in settings.items() if v != base_lev[f]}
    args = {**b["args"], "add": sorted(chosen - base_amen), "remove": sorted(base_amen - chosen), "settings": settings}
    try:
        b.update(what_if(**args), args=args)
        b.pop("error", None)
    except ValueError as e:
        b["error"] = str(e)


def apply_opportunity(bid, key):
    """Button callback: add an area opportunity (amenity or lever) to the panel."""
    if key in vocab.LEVERS:
        for f, v in vocab.LEVERS[key][2].items():
            st.session_state[f"{bid}-{f}"] = v
    else:
        st.session_state[f"{bid}-am"] = sorted(set(st.session_state[f"{bid}-am"] or []) | {key})
    recompute(bid)


def _init_widgets(bid, d):
    """First render: set widget values from the tool result (amenities and levers after the changes)."""
    if f"{bid}-am" in st.session_state:
        return
    st.session_state[f"{bid}-am"] = [a["key"] for a in d["amenities_after"]]
    for f in ("minimum_nights", "license", "superhost"):
        st.session_state[f"{bid}-{f}"] = d["levers_after"][f]


def contribution_chart(d):
    """Horizontal bars: each substantial change, plus one grey bar for everything else."""
    rows = [(c["label"], c["contribution"], f"alone {c['alone']:+,} · {c['area']} median "
             f"{'' if c['area_lift_usd'] is None else format(c['area_lift_usd'], '+,')}", None)
            for c in d["contributions"]]
    other = d.get("no_reliable_effect")
    if other:
        rows.append(("Other changes (no reliable effect here)", other["contribution"],
                     "; ".join(f"{m['label']}: {m['reason']}" for m in other["items"]), "grey"))
    if not rows:
        return None
    rows = rows[::-1]
    colors = [c or ("#2f6db5" if v >= 0 else "#c0504d") for _, v, _, c in rows]
    fig = go.Figure(go.Bar(x=[v for _, v, _, _ in rows], y=[n for n, _, _, _ in rows], orientation="h",
                           marker_color=colors, customdata=[h for _, _, h, _ in rows],
                           text=[f"{v:+,}" for _, v, _, _ in rows], textposition="outside",
                           hovertemplate="%{y}: %{x:+,} $/night<br>%{customdata}<extra></extra>"))
    fig.update_layout(height=60 + 42 * len(rows), margin=dict(l=10, r=40, t=10, b=30),
                      xaxis_title="$ per night (Shapley contribution)", showlegend=False)
    fig.update_xaxes(zeroline=True)
    return fig


def render(bid, block):
    d = block["data"]
    _init_widgets(bid, d)
    with st.container(border=True):
        listing_caption(d)
        st.pills("Amenities (click to switch on/off)", options=list(vocab.AMENITIES), format_func=vocab.label,
                 selection_mode="multi", key=f"{bid}-am", on_change=recompute, args=(bid,))
        c1, c2, c3 = st.columns(3)
        c1.number_input("Minimum stay (nights)", 1, 365, key=f"{bid}-minimum_nights", on_change=recompute, args=(bid,))
        c2.selectbox("Licence", LICENSES, key=f"{bid}-license", on_change=recompute, args=(bid,))
        c3.toggle("Superhost", key=f"{bid}-superhost", on_change=recompute, args=(bid,))
        if block.get("error"):
            st.warning(block["error"])
        d = blocks()[bid]["data"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted price", usd(d["new"]["price"]), help=f"Before changes: {usd(d['base']['price'])}")
        m1.caption(interval_text(d["new"]))
        m2.metric("Lift / night", signed_usd(d["delta"]), f"{d['delta_pct']:+.1f}%")
        m3.metric("Upfront cost", usd(d["upfront_cost"]), help="Midpoint of the assumed cost range")
        if d["monthly_cost"]:
            m3.caption(f"plus {usd(d['monthly_cost'])}/month")
        pay = d["payback_months"]
        m4.metric("Payback", "—" if pay is None else f"{pay:g} months",
                  help=f"Upfront cost / (lift × {d['nights_per_year']} nights / 12 − monthly cost)")
        if pay is None and d["changes"]:
            m4.caption("No net gain" if d["delta"] <= 0 else "")

        fig = contribution_chart(d)
        if fig is not None:
            st.plotly_chart(fig, width="stretch", key=f"{bid}-chart", config={"displayModeBar": False})
        for u in d["unsupported"]:
            st.info(f"**{u['name']}** isn't something the model can see. Closest: "
                    + (", ".join(s["label"] for s in u["suggestions"]) or "none") + ".")
        for n in d.get("notes", []):
            st.warning(n)
        opps = [o for o in d.get("area_opportunities", [])
                if o["key"] not in st.session_state[f"{bid}-am"]]
        if opps:
            st.markdown("**Worth testing in your area**")
            cols = st.columns(min(len(opps), 4))
            for col, o in zip(cols, opps):
                col.button(f"＋ {o['label']} (median +{usd(o['area_lift_usd'])})", key=f"{bid}-opp-{o['key']}",
                           on_click=apply_opportunity, args=(bid, o["key"]), width="stretch")
        footnote(d)
        st.caption("Contributions are Shapley values for this listing; a change is listed on its own only if it is "
                   "substantial in your area: " + d["rule"] + ".")
