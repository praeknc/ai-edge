"""roi_bubble: cost (x) vs lift per night (y); bubble size = payback months."""
import plotly.graph_objects as go
import streamlit as st

from app.components.common import footnote, listing_caption, usd


def render(bid, block):
    d = block["data"]
    with st.container(border=True):
        listing_caption(d)
        items = d["items"]
        if not items:
            st.info("No change you can make has a reliable price effect in your area. " + d["other_levers"])
        else:
            pay = [i["payback_months"] if i["payback_months"] is not None else 0 for i in items]
            fig = go.Figure(go.Scatter(
                x=[i["cost"] for i in items], y=[i["lift_per_night"] for i in items], mode="markers+text",
                text=[i["label"] for i in items], textposition="top center",
                marker=dict(size=[14 + 3 * p for p in pay], sizemode="diameter", opacity=0.75,
                            color=["#2f6db5" if i["within_budget"] else "#9aa5b1" for i in items]),
                customdata=[[i["cost_low"], i["cost_high"], i["yearly_gain"],
                             "—" if i["payback_months"] is None else i["payback_months"], i["area"], i["area_lift_usd"]]
                            for i in items],
                hovertemplate=("<b>%{text}</b><br>cost $%{x:,} ($%{customdata[0]:,}–$%{customdata[1]:,})"
                               "<br>+$%{y}/night · $%{customdata[2]:,}/year<br>payback %{customdata[3]} months"
                               "<br>%{customdata[4]} median +$%{customdata[5]}<extra></extra>")))
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=40), xaxis_title="Upfront cost ($, assumed)",
                              yaxis_title="Lift per night ($)", showlegend=False)
            st.plotly_chart(fig, width="stretch", key=f"{bid}-chart", config={"displayModeBar": False})
            st.caption(f"Bubble size = payback months (assuming {d['nights_per_year']} booked nights a year); "
                       "grey = over budget.")
        for n in d.get("notes", []):
            st.caption(n)
        with st.expander(f"Changes not ranked ({len(d['excluded'])})"):
            st.markdown("\n".join(f"- **{e['label']}**: {e['reason']}" for e in d["excluded"]))
        footnote(d)
