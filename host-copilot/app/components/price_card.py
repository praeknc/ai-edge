"""price_card: the recommended price with its 90% interval."""
import streamlit as st

from app.components.common import footnote, interval_text, listing_caption, usd


def render(bid, block):
    d = block["data"]
    with st.container(border=True):
        listing_caption(d)
        c1, c2 = st.columns([1, 2])
        c1.metric("Recommended price / night", usd(d["price"]))
        c1.caption(interval_text(d))
        if "current_price" in d:
            c2.metric("Your current price", usd(d["current_price"]))
            gap = d["current_gap"]
            side = "below" if gap < 0 else "above"
            c2.caption(f"{usd(abs(gap))} {side} the estimate; **{d['current_vs_range']}** the 90% range."
                       if gap else "Same as the estimate.")
        if d.get("area_opportunities"):
            st.markdown("**Worth testing in your area:** " + ", ".join(
                f"{o['label']} (median +{usd(o['area_lift_usd'])} in {o['area']})" for o in d["area_opportunities"]))
        footnote(d)
