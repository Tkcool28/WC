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
6. A **Squad strength context** panel (Phase 4) — display-only, never
   feeds the probability model.

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
- `dashboard/context_loader.py` — manual squad-strength CSV loader (Phase 4)
- `dashboard/data_loader.py` — auto-populate cache loader
- `scripts/run_dashboard.sh` — runner that activates the venv and starts streamlit
- `../soccer_ev_model/ev_workflow.py` — `evaluate_match()` (the actual work)

## Caching

The training corpus (~32k matches) and the name→id map are cached via
`@st.cache_data` and load once. The pi-rating computation is also cached
and keyed on the match date — re-submitting the form for the same date
is instant.

## Manual squad-strength updates (Phase 4)

The **Squad strength context** panel is **display-only** — it never feeds
the probability model. The numbers come from three manually curated CSV
files under `data/manual/`:

| File | Purpose |
|------|---------|
| `squad_strength_snapshot.csv` | Per-team Transfermarkt-style squad market values |
| `fifa_ranking_snapshot.csv`   | FIFA rank fallback when squad value is missing |
| `team_context_notes.csv`      | Free-text notes (injury / absence / rotation / motivation) |

### Squad strength schema

```csv
canonical_team_id,squad_market_value_eur,avg_player_value_eur,top_5_player_value_eur,most_valuable_player,snapshot_date,source_url
ARG,1180000000,26300000,265000000,Lautaro Martinez,2026-05-15,https://www.transfermarkt.com/argentinien/kader
```

- `canonical_team_id` must match an entry in `data/team_identity.json`
  (3-letter code: `ARG`, `BRA`, `FRA`, …). The loader does **not**
  auto-resolve aliases.
- `squad_market_value_eur` is the **estimated total squad value** in
  euros. This drives the value-tier band:
  - `elite`  ≥ €800M
  - `high`   ≥ €400M
  - `mid`    ≥ €150M
  - `low`    <  €150M
  - `unknown` if the row is missing
- `avg_player_value_eur` and `top_5_player_value_eur` are context-only
  numbers shown in the panel.
- `most_valuable_player` is rendered as plain text (HTML-escaped).
- `snapshot_date` (YYYY-MM-DD) is the date **you** looked the values up.
  Update this every time you refresh.
- `source_url` is the Transfermarkt-style page (or any other source) you
  read the numbers from. Kept for auditability.

### FIFA ranking schema

```csv
canonical_team_id,fifa_rank,fifa_points,snapshot_date,source_url
ARG,2,1872,2026-05-15,https://www.fifa.com/fifa-world-ranking
```

Used as a **fallback** in the panel when a team has no squad-value row.

### Notes schema

```csv
canonical_team_id,snapshot_date,note_category,note_text
ARG,2026-06-10,injury,Lautaro Martinez — hamstring, expected return group stage
ARG,2026-06-12,rotation,Expected 4-3-3 with Lo Celso at CAM
```

`note_category` is one of `injury`, `absence`, `rotation`, `motivation`,
or `other`. All note text is HTML-escaped before render — `<script>`,
quotes, and other HTML are passed through as literal text.

### How to refresh

1. Visit Transfermarkt (or your preferred source) for each team.
2. Edit `data/manual/squad_strength_snapshot.csv` — keep the header row.
3. Update `snapshot_date` on every row you touch (use the date you did
   the lookup, not today).
4. For new squads (e.g. call-ups), append a row; do not change the
   schema.
5. `pytest -q` runs the loader tests against the CSVs, so a missing
   column or bad date will fail CI rather than break the dashboard.

### Hard guarantees

- The loader is **read-only** — it never modifies any other file.
- The panel is **additive** — removing the CSVs leaves the dashboard
  working; the panel simply shows "Unknown" for every team.
- The probability model (`evaluate_match`, `pi_ratings`, `elo_ratings`,
  `confidence`, `prediction_summary`) is **never called** from the
  panel. Verified by `git diff origin/main -- soccer_ev_model/`.
