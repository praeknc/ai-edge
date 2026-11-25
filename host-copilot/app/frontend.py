"""Host Copilot: Streamlit front end.

Run from host-copilot/:   streamlit run app/frontend.py
Free-text chat goes to the agent (Claude, or the offline mock router). Capability cards, suggestion buttons,
sliders, toggles and buttons inside the answers call the Python tools directly and never go through the LLM.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import agent  # noqa: E402
from app.suggestions import CAPABILITIES, capability_actions, follow_ups  # noqa: E402
from app.components import render_block  # noqa: E402
from app.components.common import add_block, add_message, blocks  # noqa: E402
from model import vocab  # noqa: E402
from model.loader import PROPERTY_GROUPS, ROOM_TYPES, get_model  # noqa: E402

SAMPLE = json.loads((ROOT / "data" / "sample_listing.json").read_text())
CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
[data-testid="stMetricValue"] {font-size: 1.6rem;}
.cap-title {font-weight: 600; font-size: 1.05rem;}
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


def _with_flag(result):
    """Narration plus a visible flag for any dollar figure no tool returned."""
    text = result["text"]
    if result.get("unverified"):
        text += ("\n\n:orange[Unverified figure(s) not found in any tool result: "
                 + ", ".join(f"${v:,.0f}" for v in result["unverified"]) + ".]")
    return text


def ask(question, listing, goals):
    """Send free text to the agent and store its answer, UI blocks and suggested next steps."""
    add_message("user", question)
    history = st.session_state.setdefault("llm_history", [])      # Claude mode: append-only conversation
    result = agent.respond(question, listing, goals, history=history)
    _store_answer(result, goals)


def run(act, listing, goals):
    """A clicked card or suggestion: call the tool directly, no language model involved."""
    add_message("user", act["label"])
    _store_answer(agent.run_action(act, listing, goals), goals)


def _store_answer(result, goals):
    ids = [add_block(b) for b in result["blocks"]]
    if result.get("suggestions") is not None:
        nxt = result["suggestions"]
    elif result["blocks"]:
        nxt = follow_ups(result["blocks"][-1], goals)
    else:
        nxt = capability_actions(goals)[:3]
    add_message("assistant", _with_flag(result), ids)
    st.session_state["messages"][-1]["suggestions"] = nxt


def new_conversation():
    for k in ("messages", "blocks", "llm_history"):
        st.session_state.pop(k, None)


def capability_box(listing, goals):
    """'What I can do' cards, shown at the start of every conversation."""
    st.markdown("#### What I can do")
    st.caption("Click a card to run it on the listing in the sidebar, or type a question below.")
    acts = capability_actions(goals)
    rows = [st.columns(3), st.columns(3)]
    cells = [c for row in rows for c in row]
    for cell, (title, desc, icon), act in zip(cells, CAPABILITIES, acts):
        with cell.container(border=True, height=190):
            st.markdown(f"<div class='cap-title'>{icon} {title}</div>", unsafe_allow_html=True)
            st.caption(desc)
            if st.button(act["label"], key=f"cap-{title}", width="stretch"):
                run(act, listing, goals)
                st.rerun()
    with cells[5].container(border=True, height=190):
        st.markdown("<div class='cap-title'>💬 Ask in your own words</div>", unsafe_allow_html=True)
        st.caption("For example: \"Is a dishwasher worth it?\", \"Can I make $650 a night?\", "
                   "\"What if I set a 1-night minimum and register?\"")


def suggestion_buttons(msg_index, acts, listing, goals):
    """Next-step buttons under the latest answer."""
    if not acts:
        return
    st.caption("Next steps")
    cols = st.columns(len(acts))
    for i, (col, act) in enumerate(zip(cols, acts)):
        if col.button(act["label"], key=f"sug-{msg_index}-{i}", width="stretch"):
            run(act, listing, goals)
            st.rerun()


def mode_notice():
    """Say plainly how free text is understood in the current mode."""
    if agent.mock_mode():
        st.info("**Offline mode.** Typed questions are matched with keyword rules, so wording outside a few patterns "
                "gets a 'did you mean' choice. The cards and next-step buttons always work. For full chat, set "
                "ANTHROPIC_API_KEY and run with MOCK_LLM=0.", icon="ℹ️")


def main():
    st.set_page_config(page_title="Host Copilot", page_icon="🏠", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    listing, goals = sidebar_listing()
    st.sidebar.button("＋ New conversation", on_click=new_conversation, width="stretch")
    st.title("Host Copilot")
    mode = "Offline mock (keyword rules)" if agent.mock_mode() else f"Claude ({agent.model_name()})"
    st.caption(f"Pricing help for Boston Airbnb hosts. Numbers come from the pricing model, never from the chat model. "
               f"· Chat: {mode}")

    prompt = st.chat_input("Ask about your price, an upgrade, or a target…")
    if prompt:
        ask(prompt, listing, goals)
    messages = st.session_state.get("messages", [])
    if not messages:
        mode_notice()
        capability_box(listing, goals)
    last_assistant = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=None)
    for i, msg in enumerate(messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])
            for bid in msg["blocks"]:
                render_block(bid, blocks()[bid])
            if i == last_assistant:
                suggestion_buttons(i, msg.get("suggestions"), listing, goals)


main()
