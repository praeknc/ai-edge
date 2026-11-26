# Host Copilot

Chat assistant for Boston Airbnb hosts, built on the pricing model in `../airbnb-pricing`.
(Full documentation comes with the final step; this page covers setup.)

## Setup (Python 3.10+)

Keep the whole repository: Host Copilot reads `../airbnb-pricing/src` and `../airbnb-pricing/data`.

```bash
cd host-copilot
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**macOS:** LightGBM needs OpenMP. Install it once with Homebrew, otherwise importing LightGBM fails with
`Library not loaded: @rpath/libomp.dylib`:

```bash
brew install libomp
```

## Run

```bash
MOCK_LLM=1 streamlit run app/frontend.py          # offline: keyword routing, no API key
ANTHROPIC_API_KEY=... MOCK_LLM=0 streamlit run app/frontend.py   # Claude (model: CLAUDE_MODEL, default claude-sonnet-5-5)
```

## Demo host and synthetic listings

The app opens as a host would: sign in as the demo host, choose one of 5 listings, then chat with the copilot
about that listing. Each listing keeps its own conversation; **← My listings**, **↻ Restart chat** and
**⟲ Reset details** move around without starting over, and the open listing is kept in the URL so a browser
refresh returns to it.

The 5 listings are **synthetic**: designed for the demo, not taken from Inside Airbnb and not used for training.
`scripts/make_synthetic_listings.py` writes them as `data/synthetic_listings.csv` (the same 90 columns as the
Inside Airbnb detailed files) and `data/host_portfolio.json` (what the app loads). It checks that the project's
own pipeline reads each CSV row into exactly the features the app builds, and that no id, host or location
coincides with a training listing (`tests/test_synthetic.py` repeats these checks).

## Settings and secrets

Settings come from environment variables or from Streamlit secrets (copied into the environment at start-up):

| Setting | Meaning |
|---|---|
| `ANTHROPIC_API_KEY` | API key (never commit it) |
| `ANTHROPIC_BASE_URL` | Optional proxy, e.g. MIT Parley `https://parley.api.mit.edu` |
| `MOCK_LLM` | `1` = offline keyword rules, `0` = Claude (falls back to offline if there is no key) |
| `CLAUDE_MODEL` | Default `claude-sonnet-5-5` |
| `CLAUDE_EFFORT` | Default `low` |
| `CLAUDE_SERVER_FALLBACK` | `1` = server-side refusal fallback (Anthropic beta); use `0` behind a proxy |

Locally: copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` (git-ignored) and fill in the key.

## Deploy on Streamlit Community Cloud

1. Push the repository to GitHub (the whole repo: the app reads `../airbnb-pricing`).
2. On share.streamlit.io: **Create app** > choose the repo and branch > **Main file path** `host-copilot/streamlit_app.py`.
   `requirements.txt` and `packages.txt` (installs `libgomp1` for LightGBM) sit next to it.
3. **Advanced settings**: Python 3.11 or 3.12; paste the contents of `secrets.toml.example`, with your real key,
   into **Secrets**.

## Test

```bash
python -m unittest discover tests
```
