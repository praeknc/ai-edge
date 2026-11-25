"""error: a tool refused the request (unknown neighbourhood, invalid setting...)."""
import streamlit as st


def render(bid, block):
    st.warning(block["data"]["message"])
