"""Host Copilot: Streamlit front end.

Run from host-copilot/:   streamlit run app/frontend.py

Flow: sign in as the demo host -> "Your listings" (5 synthetic listings, data/host_portfolio.json) -> the copilot
for one listing. Each listing keeps its own chat while you switch; the open listing is kept in the URL so a
browser refresh returns to it.
Free-text chat goes to the agent (Claude, or the offline mock router). Capability cards, suggestion buttons,
sliders, toggles and buttons inside the answers call the Python tools directly and never go through the LLM.
"""
import math
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

SECRET_KEYS = ["ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "MOCK_LLM", "CLAUDE_MODEL", "CLAUDE_EFFORT",
               "CLAUDE_SERVER_FALLBACK"]


def load_secrets():
    """Copy settings from Streamlit secrets (.streamlit/secrets.toml, or the Secrets box on Streamlit Cloud)
    into environment variables, where the Anthropic SDK and the agent read them. Values already set in the
    environment win."""
    try:
        for key in SECRET_KEYS:
            if key in st.secrets and not os.environ.get(key):
                os.environ[key] = str(st.secrets[key])
    except Exception:                     # no secrets file: run with the environment as it is
        pass


load_secrets()

from app import agent  # noqa: E402
from app.suggestions import CAPABILITIES, capability_actions, follow_ups  # noqa: E402
from app.components import render_block  # noqa: E402
from app.components.common import add_block, add_message, blocks  # noqa: E402
from model import vocab  # noqa: E402
from model.loader import PROPERTY_GROUPS, ROOM_TYPES, get_model  # noqa: E402

PORTFOLIO = json.loads((ROOT / "data" / "host_portfolio.json").read_text())
LISTINGS = {item["id"]: item for item in PORTFOLIO["listings"]}
CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
[data-testid="stMetricValue"] {font-size: 1.6rem;}
.cap-title {font-weight: 600; font-size: 1.05rem;}
</style>
"""
FORM = {  # widget key -> host-listing field
    "p_hood": "neighbourhood", "p_room": "room_type", "p_prop": "property_group", "p_bed": "bedrooms",
    "p_guests": "accommodates", "p_beds": "beds", "p_baths": "baths", "p_shared": "baths_shared", "p_am": "amenities",
    "p_min": "minimum_nights", "p_lic": "license", "p_super": "superhost", "p_season": "season",
    "p_price": "current_price", "p_nrev": "number_of_reviews", "p_score": "review_score", "p_hostn": "host_listings",
}
GOALS = ("g_target", "g_budget", "g_nights")


# ---------------------------------------------------------------- per-listing state
def chats():
    """listing id -> {"messages", "llm_history", "form"}: each listing keeps its own conversation and edits."""
    return st.session_state.setdefault("chats", {})


@st.cache_data(show_spinner=False)
def estimate(listing_id):
    """Model estimate for a portfolio listing as stored (for the listing cards)."""
    return agent.call_tool("predict_price", {}, LISTINGS[listing_id]["listing"])["data"]


def form_values(listing_id):
    """Widget values for a listing straight from the portfolio, plus a default goal about 10% above the estimate."""
    L = LISTINGS[listing_id]["listing"]
    vals = {k: L[f] for k, f in FORM.items()}
    vals["p_am"] = vocab.resolve_many(L["amenities"])[0]
    vals["p_baths"] = float(vals["p_baths"])
    est = estimate(listing_id)["price"]
    vals.update({"g_target": int(math.ceil(est * 1.1 / 10) * 10), "g_budget": 2000, "g_nights": 200})
    return vals


def _save_open_listing():
    lid = st.session_state.get("open_listing")
    if lid:
        chats()[lid] = {"messages": st.session_state.get("messages", []),
                        "llm_history": st.session_state.get("llm_history", []),
                        "form": {k: st.session_state[k] for k in list(FORM) + list(GOALS) if k in st.session_state}}


def open_listing(listing_id):
    """Switch the copilot to a listing: keep the current listing's chat, restore (or start) the new one's."""
    _save_open_listing()
    saved = chats().get(listing_id, {})
    st.session_state["messages"] = saved.get("messages", [])
    st.session_state["llm_history"] = saved.get("llm_history", [])
    for k, v in (saved.get("form") or form_values(listing_id)).items():
        st.session_state[k] = v
    st.session_state["open_listing"] = listing_id
    st.query_params.update({"host": "demo", "listing": listing_id})


def back_to_listings():
    _save_open_listing()
    st.session_state["open_listing"] = None
    st.query_params.clear()
    st.query_params["host"] = "demo"


def restart_chat():
    """New conversation for the open listing (its details and goals stay)."""
    st.session_state["messages"], st.session_state["llm_history"] = [], []


def reset_details():
    """Put the open listing's details and goals back to what is stored on the platform."""
    for k, v in form_values(st.session_state["open_listing"]).items():
        st.session_state[k] = v


def sign_in():
    st.session_state["signed_in"] = True
    st.query_params["host"] = "demo"


def sign_out():
    for k in ("signed_in", "open_listing", "chats", "messages", "llm_history", "blocks"):
        st.session_state.pop(k, None)
    st.query_params.clear()


def restore_from_url():
    """After a browser refresh, return to the signed-in host and the listing in the URL."""
    if st.session_state.get("signed_in") or st.query_params.get("host") != "demo":
        return
    st.session_state["signed_in"] = True
    lid = st.query_params.get("listing")
    if lid in LISTINGS:
        open_listing(lid)


# ---------------------------------------------------------------- sidebar: listing details and goals
def sidebar_listing(listing_id):
    """Editable details of the open listing and the host's goals (values live in st.session_state)."""
    model = get_model()
    base = LISTINGS[listing_id]["listing"]
    sb = st.sidebar
    sb.caption(f"Signed in as **{PORTFOLIO['host']['host_name']}**")
    sb.header("Listing details")
    sb.caption("Prefilled from the platform. Edit to test a scenario; ⟲ Reset details restores them.")
    hood = sb.selectbox("Neighbourhood", model.neighbourhoods(), key="p_hood")
    room = sb.selectbox("Room type", ROOM_TYPES, key="p_room")
    prop = sb.selectbox("Property type", PROPERTY_GROUPS, key="p_prop")
    c1, c2 = sb.columns(2)
    bedrooms = c1.number_input("Bedrooms", 0, 8, key="p_bed")
    guests = c2.number_input("Guests", 1, 16, key="p_guests")
    beds = c1.number_input("Beds", 0, 16, key="p_beds")
    baths = c2.number_input("Baths", 0.0, 8.0, step=0.5, key="p_baths")
    shared = sb.toggle("Shared bathroom", key="p_shared")
    amen = sb.multiselect("Amenities", list(vocab.AMENITIES), format_func=vocab.label, key="p_am")
    c1, c2 = sb.columns(2)
    min_n = c1.number_input("Minimum nights", 1, 365, key="p_min")
    lic = c2.selectbox("Licence", ["licensed", "exempt", "none"], key="p_lic")
    superhost = sb.toggle("Superhost", key="p_super")
    season = sb.radio("Season to price", ["fall", "summer"], horizontal=True, key="p_season")
    price = sb.number_input("Your listed price ($/night)", 0, 5000, step=5, key="p_price")
    with sb.expander("Reviews and history"):
        n_rev = st.number_input("Number of reviews", 0, 2000, key="p_nrev")
        score = st.slider("Average rating", 1.0, 5.0, step=0.05, key="p_score")
        host_n = st.number_input("Listings you host", 1, 500, key="p_hostn")
    sb.header("Your goals")
    target = sb.number_input("Target price ($/night)", 0, 5000, step=10, key="g_target")
    budget = sb.number_input("Upgrade budget ($)", 0, 100000, step=100, key="g_budget")
    nights = sb.slider("Booked nights per year", 30, 365, step=5, key="g_nights")
    sb.divider()
    sb.button("Sign out", on_click=sign_out, width="stretch")

    base_keys = set(vocab.resolve_many(base["amenities"])[0])
    listing = {**base, "neighbourhood": hood, "room_type": room, "property_group": prop, "bedrooms": int(bedrooms),
               "accommodates": int(guests), "beds": int(beds), "baths": float(baths), "baths_shared": shared,
               "amenities": amen, "minimum_nights": int(min_n), "license": lic, "superhost": superhost,
               "season": season, "current_price": int(price) or None, "number_of_reviews": int(n_rev),
               "review_score": float(score), "host_listings": int(host_n),
               "amenity_count": base["amenity_count"] + len(set(amen) - base_keys) - len(base_keys - set(amen))}
    if hood != base["neighbourhood"]:                 # moved: use the new neighbourhood's typical location
        listing.pop("latitude", None)
        listing.pop("longitude", None)
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


# ---------------------------------------------------------------- pages
def login_page():
    st.title("🏠 Host Copilot")
    st.caption("Pricing help for Boston Airbnb hosts, built on a model trained on 7,039 Boston listings.")
    with st.container(border=True):
        st.markdown("#### Sign in")
        st.text_input("Host account", PORTFOLIO["host"]["host_name"], disabled=True)
        st.caption("Demo sign-in, no password. The account's 5 listings are synthetic: made for this demo, "
                   "not part of the data the model was trained on.")
        st.button(f"Sign in as {PORTFOLIO['host']['host_name']}", on_click=sign_in, type="primary")


def price_status(current, est):
    """Card line comparing the listed price with the model estimate (15% either way counts as off)."""
    gap = current - est["price"]
    if gap <= -0.15 * est["price"]:
        return f":orange[${-gap:,} below the model estimate]" + (" (below its 90% range)" if current < est["low"] else "")
    if gap >= 0.15 * est["price"]:
        return f":orange[${gap:,} above the model estimate]" + (" (above its 90% range)" if current > est["high"] else "")
    return ":green[Close to the model estimate]"


def portfolio_page():
    st.title(f"Your listings ({len(LISTINGS)})")
    st.caption(f"Signed in as {PORTFOLIO['host']['host_name']}. Choose a listing to open its copilot.")
    cols = st.columns(2)
    for i, (lid, item) in enumerate(LISTINGS.items()):
        L, est = item["listing"], estimate(lid)
        room = "Entire home" if L["room_type"] == "Entire home/apt" else L["room_type"]
        with cols[i % 2].container(border=True):
            st.markdown(f"**{item['nickname']}**  \n{item['title']}")
            st.caption(f"📍 {L['neighbourhood']} · {room} · {L['bedrooms']} bedroom(s) · ★ {L['review_score']} "
                       f"({L['number_of_reviews']} reviews)" + (" · Superhost" if L["superhost"] else ""))
            c1, c2 = st.columns(2)
            c1.metric("Your price", f"${L['current_price']:,}")
            c2.metric("Model estimate", f"${est['price']:,}")
            c2.caption(f"90% range ${est['low']:,}–${est['high']:,}")
            st.markdown(f"{price_status(L['current_price'], est)}  \n{item['story']}")
            if any(m["role"] == "user" for m in chats().get(lid, {}).get("messages", [])):
                st.caption("💬 Conversation in progress")
            st.button("Open copilot →", key=f"open-{lid}", on_click=open_listing, args=(lid,), width="stretch")


def copilot_page(listing_id):
    item = LISTINGS[listing_id]
    listing, goals = sidebar_listing(listing_id)
    nav1, nav2, nav3, _ = st.columns([1.2, 1.2, 1.2, 3])
    nav1.button("← My listings", on_click=back_to_listings, width="stretch")
    nav2.button("↻ Restart chat", on_click=restart_chat, width="stretch")
    nav3.button("⟲ Reset details", on_click=reset_details, width="stretch")
    st.title(item["nickname"])
    mode = "Offline mock (keyword rules)" if agent.mock_mode() else f"Claude ({agent.model_name()})"
    st.caption(f"{item['title']} · Numbers come from the pricing model, never from the chat model. · Chat: {mode}")

    prompt = st.chat_input(f"Ask about {item['nickname']}: price, an upgrade, or a target…")
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


def main():
    st.set_page_config(page_title="Host Copilot", page_icon="🏠", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    restore_from_url()
    if not st.session_state.get("signed_in"):
        login_page()
    elif st.session_state.get("open_listing") in LISTINGS:
        copilot_page(st.session_state["open_listing"])
    else:
        portfolio_page()


main()
