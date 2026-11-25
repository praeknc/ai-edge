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

## Test

```bash
python -m unittest discover tests
```
