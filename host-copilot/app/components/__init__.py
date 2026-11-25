"""Renderer: maps a tool result's "component" name to the Streamlit code that draws it."""
from app.components import cost_card, error, plan_cards, price_card, roi_bubble, whatif_panel

RENDERERS = {
    "price_card": price_card.render,
    "whatif_panel": whatif_panel.render,
    "plan_cards": plan_cards.render,
    "roi_bubble": roi_bubble.render,
    "cost_card": cost_card.render,
    "error": error.render,
}


def render_block(bid, block):
    """Draw one UI block; unknown components fall back to raw JSON."""
    fn = RENDERERS.get(block["component"])
    if fn is None:
        import streamlit as st
        st.json(block["data"])
    else:
        fn(bid, block)
