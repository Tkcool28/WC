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

## VPS Disaster Recovery

The runtime data files are committed to this repo as a disaster-recovery
backup for the VPS at `wcusa.duckdns.org`. After a fresh clone on a new
host:

```bash
git clone https://github.com/Tkcool28/WC.git
cd WC
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dashboard]"
python scripts/verify_runtime_data.py   # sanity-check the data backup
./scripts/run_dashboard.sh              # serves the dashboard on :8501 (or :8510 on the VPS)
```

### Secrets
`.env` is gitignored and **not** committed. Recreate it manually if you
need live data refresh:

```bash
echo "FOOTBALL_DATA_API_KEY=your_key_here" > .env
```

Get a free key at https://www.football-data.org/.

### Data sources
The 26 tracked files in `data/` are the exact files the dashboard and
the model load at runtime. If you ever need to regenerate them, the
fetchers live in `scripts/`:

- `scripts/fetch_elo_ratings.py` → `data/raw/elo_ratings.json`
- `scripts/fetch_live_2026.py` → `data/raw/matches_2026.json`
- `scripts/fetch_historical_wc.py` → `data/raw/matches_<year>_openfootball.json`
- `scripts/parse_international.py` → `data/processed/international_matches.json`

`data/team_identity.json` is hand-curated (see `soccer_ev_model/team_identity.py`).
