"""Host Copilot: Streamlit front end.

Run from host-copilot/:   streamlit run app/frontend.py
Free-text chat goes to the agent (mock LLM in this step). Sliders, toggles and buttons inside the
answers call the Python tools directly and never go back through the LLM.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import agent  # noqa: E402
from app.components import render_block  # noqa: E402
from app.components.common import add_block, add_message, blocks  # noqa: E402
from model import vocab  # noqa: E402
from model.loader import PROPERTY_GROUPS, ROOM_TYPES, get_model  # noqa: E402

SAMPLE = json.loads((ROOT / "data" / "sample_listing.json").read_text())
EXAMPLES = ["What should I charge?", "What if I add a dishwasher and a hot tub?",
            "How do I get to $560 with a budget of $2,000?", "Which upgrades pay back fastest?",
            "What if I set a 1-night minimum?"]
CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
[data-testid="stMetricValue"] {font-size: 1.6rem;}
</style>
"""


def sidebar_listing():
    """Listing profile and host goals from the sidebar form (kept in st.session_state by widget keys)."""
    model = get_model()
    hoods = model.neighbourhoods()
    sb = st.sidebar
    sb.header("Your listing")
    hood = sb.selectbox("Neighbourhood", hoods, index=hoods.index(SAMPLE["neighbourhood"]), key="p_hood")
    room = sb.selectbox("Room type", ROOM_TYPES, index=ROOM_TYPES.index(SAMPLE["room_type"]), key="p_room")
    prop = sb.selectbox("Property type", PROPERTY_GROUPS, index=PROPERTY_GROUPS.index(SAMPLE["property_group"]), key="p_prop")
    c1, c2 = sb.columns(2)
    bedrooms = c1.number_input("Bedrooms", 0, 8, SAMPLE["bedrooms"], key="p_bed")
    guests = c2.number_input("Guests", 1, 16, SAMPLE["accommodates"], key="p_guests")
    beds = c1.number_input("Beds", 0, 16, SAMPLE["beds"], key="p_beds")
    baths = c2.number_input("Baths", 0.0, 8.0, float(SAMPLE["baths"]), 0.5, key="p_baths")
    shared = sb.toggle("Shared bathroom", SAMPLE["baths_shared"], key="p_shared")
    default_am = vocab.resolve_many(SAMPLE["amenities"])[0]
    amen = sb.multiselect("Current amenities", list(vocab.AMENITIES), default=default_am, format_func=vocab.label, key="p_am")
    c1, c2 = sb.columns(2)
    min_n = c1.number_input("Minimum nights", 1, 365, SAMPLE["minimum_nights"], key="p_min")
    lic = c2.selectbox("Licence", ["licensed", "exempt", "none"], key="p_lic")
    superhost = sb.toggle("Superhost", SAMPLE["superhost"], key="p_super")
    season = sb.radio("Season to price", ["fall", "summer"], horizontal=True, key="p_season")
    price = sb.number_input("Your current price ($/night)", 0, 5000, SAMPLE["current_price"], 5, key="p_price")
    with sb.expander("Reviews and history"):
        n_rev = st.number_input("Number of reviews", 0, 2000, SAMPLE["number_of_reviews"], key="p_nrev")
        score = st.slider("Average rating", 1.0, 5.0, SAMPLE["review_score"], 0.05, key="p_score")
        host_n = st.number_input("Listings you host", 1, 500, SAMPLE["host_listings"], key="p_hostn")
    sb.header("Your goals")
    target = sb.number_input("Target price ($/night)", 0, 5000, 600, 10, key="g_target")
    budget = sb.number_input("Upgrade budget ($)", 0, 100000, 2000, 100, key="g_budget")
    nights = sb.slider("Booked nights per year", 30, 365, 200, 5, key="g_nights")
    listing = {**SAMPLE, "neighbourhood": hood, "room_type": room, "property_group": prop, "bedrooms": int(bedrooms),
               "accommodates": int(guests), "beds": int(beds), "baths": float(baths), "baths_shared": shared,
               "amenities": amen, "minimum_nights": int(min_n), "license": lic, "superhost": superhost,
               "season": season, "current_price": int(price) or None, "number_of_reviews": int(n_rev),
               "review_score": float(score), "host_listings": int(host_n), "amenity_count": None}
    goals = {"target_price": int(target), "budget": int(budget), "nights_per_year": int(nights)}
    return listing, goals


def ask(question, listing, goals):
    """Send free text to the agent and store its answer and UI blocks."""
    add_message("user", question)
    result = agent.respond(question, listing, goals)
    add_message("assistant", result["text"], [add_block(b) for b in result["blocks"]])


def main():
    st.set_page_config(page_title="Host Copilot", page_icon="🏠", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    listing, goals = sidebar_listing()
    st.title("Host Copilot")
    mode = "Mock LLM (offline keyword routing)" if agent.mock_mode() else "Claude"
    st.caption(f"Ask about your Boston Airbnb's price. Numbers come from the pricing model, never from the chat model. · {mode}")

    prompt = st.chat_input("Ask about your price, upgrades or a target…")
    if prompt:
        ask(prompt, listing, goals)
    messages = st.session_state.get("messages", [])
    if not messages:
        st.markdown("**Try one of these:**")
        cols = st.columns(len(EXAMPLES))
        for col, q in zip(cols, EXAMPLES):
            if col.button(q, width="stretch"):
                ask(q, listing, goals)
                st.rerun()
    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])
            for bid in msg["blocks"]:
                render_block(bid, blocks()[bid])


main()
