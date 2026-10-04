# VendorWise: AI Vendor Selection Assistant

End Term Project of AI for Managers (PGDM, FORE School of Management), Use Case #6: Vendor selection / procurement recommender.

A buyer describes what they need, sets how much cost, quality, lead time, reliability and payment terms matter, and gets a
ranked vendor shortlist with an AI-written rationale. **All scores are calculated in Python; Google Gemini (free tier) only
reads the requirement and explains the result.** The app checks every figure the AI writes against the data.

## Features
- Plain-English requirement → form auto-filled by Gemini (keyword fallback if AI is off)
- Hard filters: MOQ, max lead time, ISO certification, price cap, with the reason each vendor was excluded
- Adjustable weights + presets (Balanced, Cost-first, Quality-first, Urgent order)
- Ranked shortlist, score-contribution chart, winner-stability test (±10-point weight changes)
- Procurement rule-of-thumb checks (premium over L1, defect > 2%, on-time < 85%, disputes, close calls)
- AI recommendation memo, validated: must match rank 1; figures not found in the data are flagged
- Chat about the shortlist with conversation memory; refuses off-topic and prompt-injection requests
- Privacy: e-mail / phone / PAN / GSTIN / card numbers masked before any API call; disclosure in sidebar
- Basic mode when there's no key, the API is down or the quota is used up; per-session AI call cap; memo caching

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI |
| `scoring.py` | Data validation, filters, scoring, sensitivity, rule checks |
| `ai.py` | Gemini prompts, guardrails, model fallback, output validation, no-AI fallbacks |
| `data/vendors.csv` | Sample data: 30 vendors in 5 categories for a fictional appliance maker |
| `tests/` | 14 automated tests (Gemini mocked) |

## Run locally
```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # paste your Gemini key
streamlit run app.py
python -m pytest tests -q
```

## Deploy free (Streamlit Community Cloud)
1. Push this folder to a public GitHub repo (don't upload `secrets.toml`).
2. share.streamlit.io → Create app → pick the repo, branch `main`, file `app.py`.
3. Advanced settings → Secrets: `GEMINI_API_KEY = "your-key"` → Deploy.
