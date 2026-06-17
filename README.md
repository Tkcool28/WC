# soccer-ev-model-lab

Local-first World Cup soccer outcome probability model. Uses pi-ratings and Elo
on historical FIFA World Cup matches (2010-2026) to predict P(home win / draw /
away win) for upcoming matches. Designed for **probabilities, not +EV bets** —
the user compares model %s to bookmaker %s separately.

**Phase 1 (this build):** Data fetching, no-vig odds utility, pi-rating
computation. **Phase 2 (next):** Feature engineering + CatBoost/XGBoost model.
**Phase 3 (later):** Walk-forward backtest harness + reporting.

## Quick start
```bash
cd /root/soccer-model-lab
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pytest -v
```

## Data
Cached WC match data lives in `data/raw/matches_<year>.json`. Large data
files are gitignored — after a fresh clone, fetch them:

```bash
# Fetch the Elo ratings cache (used by the Pi+Elo blend)
python3 scripts/fetch_elo_ratings.py

# Fetch the live 2026 WC fixture list (used by the auto-populate view)
python3 scripts/fetch_live_2026.py
```

The dashboard also needs a `.env` file at the project root with
`FOOTBALL_DATA_API_KEY=*** for `scripts/fetch_live_2026.py`. Get a free
key at https://www.football-data.org/.

## Dashboard
```bash
./scripts/run_dashboard.sh          # http://127.0.0.1:8501
```
