# ⚽ +EV Soccer Dashboard

A small Streamlit app that exposes the pi-rating +EV workflow end-to-end.

You pick a matchup, paste the book's American odds, and the app shows:

- the **pi-rating** % for each market (Home / Draw / Away)
- the **book's no-vig (fair)** %
- the **calibrated pi** % (corrected via the 9,678-match backtest)
- **+EV flags** (markets where `pi% − book% ≥ threshold`)
- a **confidence tier** (🟢 A / 🟡 B / 🟠 C / 🔴 D) with any warnings

It's a thin UI on top of `soccer_ev_model.ev_workflow.evaluate_match()`,
which does the actual work. The UI is single-column and mobile-friendly
so you can run it from a phone.

---

## Install

Streamlit is an **optional** dependency. From the project root:

```bash
source .venv/bin/activate

# Option A: install streamlit alone
pip install streamlit

# Option B: install as an optional extra (after `pip install -e .`)
pip install -e ".[dashboard]"

# Option C: install everything (phase2 models + streamlit)
pip install -e ".[all]"
```

Verify the install:

```bash
python -c "import streamlit; print('streamlit', streamlit.__version__)"
```

## Run

From the project root:

```bash
source .venv/bin/activate
streamlit run dashboard/app.py
```

…or use the helper script (defaults to port 8501, headless mode):

```bash
./scripts/run_dashboard.sh
PORT=9000 ./scripts/run_dashboard.sh   # custom port
HEADLESS=false ./scripts/run_dashboard.sh   # auto-open browser
```

Then open <http://localhost:8501> on your phone or laptop.

> If you want to access the dashboard from your phone on the same Wi-Fi,
> run the server with `--server.address 0.0.0.0` (the helper script
> already does this) and visit `http://<laptop-lan-ip>:8501`.

## Example inputs

These three matches are the known 2026 WC group-stage openers. Drop them
into the form to see the dashboard light up:

| Home       | Away      | Date       | Home odds | Draw odds | Away odds |
|------------|-----------|------------|-----------|-----------|-----------|
| Iraq       | Norway    | 2026-06-16 | +1300     | +550      | −450      |
| Argentina  | Algeria   | 2026-06-16 | −230      | +350      | +700      |
| Austria    | Jordan    | 2026-06-16 | −280      | +400      | +800      |

**Steps:** Home team = "Argentina", Away team = "Algeria", date = 2026-06-16,
odds `-230 / +350 / +700`, hit **Run Analysis**.

## What you'll see

1. A coloured confidence banner with the tier emoji and any warnings.
2. A 3-column probability comparison: **Pi (raw)** vs **Pi (calibrated)** vs **Book fair**.
3. The **edges** (pi − book) per market.
4. **+EV flags** as `st.metric` cards — one per flagged market, with the
   delta and the underlying percentages.
5. An "Input odds & raw values" expander with the full payload for copy-paste.

### Confidence tiers

| Tier | Meaning                                                        |
|------|----------------------------------------------------------------|
| 🟢 A | Model is well-calibrated on this matchup                       |
| 🟡 B | Decent, minor caveats                                          |
| 🟠 C | Overconfident pi **or** limited team data — signal unreliable  |
| 🔴 D | Insufficient data, pi is roughly a coin flip                   |

The dashboard also surfaces any textual warnings from
`assess_match_confidence()` (e.g. "home team has only 8 matches of data").

## Files

- `dashboard/app.py` — the Streamlit app
- `scripts/run_dashboard.sh` — runner that activates the venv and starts streamlit
- `../soccer_ev_model/ev_workflow.py` — `evaluate_match()` (the actual work)

## Caching

The training corpus (~32k matches) and the name→id map are cached via
`@st.cache_data` and load once. The pi-rating computation is also cached
and keyed on the match date — re-submitting the form for the same date
is instant.
