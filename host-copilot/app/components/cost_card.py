"""cost_card: assumed cost of one amenity."""
import streamlit as st

from app.components.common import usd


def render(bid, block):
    d = block["data"]
    with st.container(border=True):
        if not d["found"]:
            st.info(d["message"])
            return
        c1, c2 = st.columns(2)
        c1.metric(d["name"], f"{usd(d['cost_low'])}–{usd(d['cost_high'])}")
        c2.metric("Monthly running cost", usd(d["monthly_cost"]))
        if not d["in_model"]:
            st.caption("The pricing model cannot see this amenity, so it has no price effect estimate.")
        st.caption(f"{d['notes']}")
