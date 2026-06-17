# Soccer EV Model — Phase 1 Foundation

> **For Hermes:** Phase 1 only. Phase 2 (model) and Phase 3 (validation) come after Phase 1 is verified end-to-end.

**Goal:** Build the rate-limited API client, no-vig odds utility, and pi-rating computation module for the World Cup soccer model. Also fetch and cache historical WC match data 2010-2026.

**Architecture:** Mirror `mlb-ev-model-lab/` layout. Standalone `soccer_ev_model/` package with isolated `.venv/`. Cached data in `data/raw/` as JSON. No model code in Phase 1.

**Tech Stack:** Python 3.11, stdlib `urllib` (no requests dep), pandas, numpy. CatBoost/XGBoost come in Phase 2.

---

## Target

**Decision being backtested (Phase 3):** "For a given WC match, what are P(home win), P(draw), P(away win)?"

**Features needed (Phase 1 starts the data layer; Phase 2 builds features):**
- Pi-rating (offense + defense, per team) — computed from historical results
- World Football Elo — external source
- Rest days, tournament stage, recent form — derived from match data

**Data timing:** All features must use only data from matches strictly before the target match. Pi-ratings are sequential; Elo is point-in-time per match day.

**Baseline to beat:** World Football Elo (or Naive market, once we have a way to fetch it). If our model can't beat Elo on RPS, it's not adding value.

**Fake signal to avoid:** If we accidentally include any post-match data in features (final scores, xG from the match being predicted), we'll look great in training and terrible live.

---

## Leakage controls (for Phase 1's data layer)

- Match data is fetched once per WC tournament, stored as a static file. No re-fetching that touches today's matches unless explicitly requested.
- Pi-rating computation is a separate function that takes a list of matches and a cutoff date. It never silently re-computes against a window that includes the target match.
- The API client has a configurable `cache_first` mode that, by default, reads from disk and only fetches if the cache is missing or stale.

---

## Task 1: Project scaffold + .gitignore

**Objective:** Create the directory structure that mirrors `mlb-ev-model-lab/`.

**Files:**
- Create: `/root/soccer-model-lab/pyproject.toml`
- Create: `/root/soccer-model-lab/.gitignore`
- Create: `/root/soccer-model-lab/README.md`
- Create: `/root/soccer-model-lab/soccer_ev_model/__init__.py`
- Create: `/root/soccer-model-lab/tests/__init__.py`

**Step 1: Write pyproject.toml** (match mlb-ev-model-lab style, just the minimum)

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "soccer-ev-model-lab"
version = "0.1.0"
description = "Local-first World Cup soccer outcome probability model (no EV math)"
requires-python = ">=3.11"
dependencies = [
  "pandas>=2.0",
  "numpy>=1.24",
]

[project.optional-dependencies]
phase2 = [
  "catboost>=1.2",
  "xgboost>=2.0",
  "scikit-learn>=1.3",
]

[tool.setuptools]
packages = ["soccer_ev_model"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

**Step 2: Write .gitignore** (mirror MLB project)

```
.venv/
__pycache__/
.pytest_cache/
data/raw/*.json
data/processed/
*.pyc
.DS_Store
.env
```

**Step 3: Write README.md** (3 paragraphs: what, why, current state)

```markdown
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
pip install -e .[phase2]
pytest -v
```

## Data
Cached WC match data lives in `data/raw/matches_<year>.json`.
```

**Step 4: Create package __init__.py and tests __init__.py** (empty)

**Step 5: Verify**

```bash
cd /root/soccer-model-lab
touch soccer_ev_model/__init__.py tests/__init__.py
ls -la
```

Expected: directories exist, files empty.

**Step 6: Commit**

```bash
cd /root/soccer-model-lab
git init -q
git add .
git commit -q -m "scaffold: project structure + gitignore"
```

---

## Task 2: Create venv and install deps

**Objective:** Isolated Python environment matching MLB project pattern.

**Step 1: Create venv and install base deps**

```bash
cd /root/soccer-model-lab
python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -e .
pip install -q pytest
```

**Step 2: Verify Python is from venv**

```bash
which python3
python3 -c "import sys; print(sys.prefix)"
```

Expected: both paths under `/root/soccer-model-lab/.venv`

**Step 3: Verify package importable**

```bash
python3 -c "import soccer_ev_model; print('OK')"
```

Expected: `OK`

**Step 4: Commit (no changes yet, just verifying state)**

```bash
git status
```

Expected: no changes since last commit.

---

## Task 3: API client with rate limiting (TDD)

**Objective:** Single chokepoint for all football-data.org calls. 6s delay between calls, 60s pause on 429, polite User-Agent, token loaded from `.env`.

**Files:**
- Create: `/root/soccer-model-lab/soccer_ev_model/api_client.py`
- Test: `/root/soccer-model-lab/tests/test_api_client.py`

**Step 1: Write failing test**

```python
# tests/test_api_client.py
import os
import time
from unittest.mock import patch, MagicMock
from soccer_ev_model.api_client import FootballDataClient


def test_load_token_from_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("FOOTBALL_DATA_API_KEY=test_token_abc123\n")
    monkeypatch.chdir(tmp_path)
    client = FootballDataClient()
    assert client.token == "test_token_abc123"


def test_user_agent_header():
    with patch("soccer_ev_model.api_client.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"name": "FIFA World Cup"}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = lambda s, *a: False
        mock_urlopen.return_value = mock_resp
        client = FootballDataClient(token="abc")
        client.get("/v4/competitions/WC")
        # Verify the request had our User-Agent
        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        assert "Hermes-Research-Bot" in request_obj.headers["User-agent"]
        assert request_obj.headers["X-Auth-Token"] == "abc"


def test_429_triggers_backoff():
    """When we get a 429, the client should sleep 60s before retrying."""
    with patch("soccer_ev_model.api_client.urlopen") as mock_urlopen, \
         patch("soccer_ev_model.api_client.time.sleep") as mock_sleep:
        # First call: 429. Second call: 200.
        resp_429 = MagicMock()
        resp_429.status = 429
        resp_429.__enter__ = lambda s: s
        resp_429.__exit__ = lambda s, *a: False
        resp_429.read.side_effect = Exception("can't read on 429")

        resp_200 = MagicMock()
        resp_200.status = 200
        resp_200.read.return_value = b'{}'
        resp_200.__enter__ = lambda s: s
        resp_200.__exit__ = lambda s, *a: False

        mock_urlopen.side_effect = [resp_429, resp_200]
        client = FootballDataClient(token="abc", min_delay=0, max_retries=2)
        client.get("/v4/test")
        # Should have slept 60s on the 429
        assert any(call.args[0] == 60 for call in mock_sleep.call_args_list)
```

**Step 2: Run test to verify failure**

```bash
source .venv/bin/activate
pytest tests/test_api_client.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'soccer_ev_model.api_client'`

**Step 3: Write minimal implementation**

```python
# soccer_ev_model/api_client.py
import os
import time
import urllib.request
import urllib.error
import json
from pathlib import Path


class FootballDataClient:
    BASE_URL = "https://api.football-data.org/v4"
    USER_AGENT = "Hermes-Research-Bot/1.0 (contact: todd-private)"
    MIN_DELAY_SECONDS = 6.0  # Free tier is 10 req/min
    BACKOFF_429_SECONDS = 60.0
    DEFAULT_MAX_RETRIES = 3

    def __init__(self, token: str | None = None, min_delay: float = MIN_DELAY_SECONDS, max_retries: int = DEFAULT_MAX_RETRIES):
        self.token = token or self._load_token()
        if not self.token:
            raise ValueError("No API token. Pass token= or set FOOTBALL_DATA_API_KEY in .env")
        self.min_delay = min_delay
        self.max_retries = max_retries
        self._last_call_ts = 0.0

    @staticmethod
    def _load_token() -> str | None:
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("FOOTBALL_DATA_API_KEY="):
                    return line.split("=", 1)[1].strip()
        return os.environ.get("FOOTBALL_DATA_API_KEY")

    def _throttle(self):
        elapsed = time.time() - self._last_call_ts
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        self._last_call_ts = time.time()

    def get(self, path: str) -> dict:
        """Make a GET request to the API. Returns parsed JSON."""
        if not path.startswith("/"):
            path = "/" + path
        url = f"{self.BASE_URL}{path}"
        headers = {
            "X-Auth-Token": self.token,
            "User-Agent": self.USER_AGENT,
        }
        for attempt in range(self.max_retries):
            self._throttle()
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    time.sleep(self.BACKOFF_429_SECONDS)
                    continue
                raise
        raise RuntimeError(f"Max retries ({self.max_retries}) exceeded for {url}")
```

**Step 4: Run test to verify pass**

```bash
pytest tests/test_api_client.py -v
```

Expected: 3 passed

**Step 5: Commit**

```bash
git add soccer_ev_model/api_client.py tests/test_api_client.py
git commit -q -m "feat(api): rate-limited football-data client with 429 backoff"
```

---

## Task 4: No-vig odds utility (TDD)

**Objective:** Convert 3-way American odds to implied probs, then remove the vig to get fair probabilities. Also the "anytime" 2-way version (for player props). This is a pure math module — no API calls.

**Files:**
- Create: `/root/soccer-model-lab/soccer_ev_model/no_vig.py`
- Test: `/root/soccer-model-lab/tests/test_no_vig.py`

**Step 1: Write failing test**

```python
# tests/test_no_vig.py
from soccer_ev_model.no_vig import american_to_implied, remove_vig_3way, no_vig_3way


def test_american_to_implied_favorite():
    assert abs(american_to_implied(-200) - 0.6667) < 0.001


def test_american_to_implied_underdog():
    assert abs(american_to_implied(550) - 0.1538) < 0.001


def test_remove_vig_3way_france_senegal():
    """France -200, Senegal +550, Draw +340. Fair probs should sum to 1.0."""
    fair = remove_vig_3way(home_odds=-200, away_odds=550, draw_odds=340)
    assert abs(sum(fair.values()) - 1.0) < 0.0001
    assert 0.55 < fair["home"] < 0.70
    assert 0.18 < fair["draw"] < 0.25
    assert 0.10 < fair["away"] < 0.18


def test_remove_vig_3way_balanced_market():
    """Equal moneyline -110/-110/-110. Fair probs should be roughly 1/3 each."""
    fair = remove_vig_3way(home_odds=-110, away_odds=-110, draw_odds=-110)
    assert abs(fair["home"] - 1/3) < 0.01
    assert abs(fair["draw"] - 1/3) < 0.01
    assert abs(fair["away"] - 1/3) < 0.01


def test_no_vig_3way_returns_full_dict():
    """no_vig_3way should also include the raw implied probs and vig_pct."""
    result = no_vig_3way(home_odds=-200, away_odds=550, draw_odds=340)
    assert "implied" in result
    assert "fair" in result
    assert "vig_pct" in result
    assert "home" in result["implied"] and "draw" in result["implied"] and "away" in result["implied"]
    assert result["vig_pct"] > 0  # Vig should be positive
```

**Step 2: Run test to verify failure**

```bash
pytest tests/test_no_vig.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'soccer_ev_model.no_vig'`

**Step 3: Write minimal implementation**

```python
# soccer_ev_model/no_vig.py
"""Odds math utilities. Pure functions, no API calls.

Standard formulas:
- American odds -200 -> implied prob = 200/(200+100) = 0.6667
- American odds +550 -> implied prob = 100/(550+100) = 0.1538
- Removing vig from a 3-way market: divide each implied prob by the total.
"""


def american_to_implied(odds: int) -> float:
    """Convert American odds to implied probability (includes vig)."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    elif odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    else:
        raise ValueError("American odds cannot be 0 (implies infinite prob)")


def remove_vig_3way(home_odds: int, away_odds: int, draw_odds: int) -> dict[str, float]:
    """Remove vig from a 3-way market. Returns dict with home/draw/away fair probs summing to 1.0."""
    h = american_to_implied(home_odds)
    d = american_to_implied(draw_odds)
    a = american_to_implied(away_odds)
    total = h + d + a
    return {
        "home": h / total,
        "draw": d / total,
        "away": a / total,
    }


def no_vig_3way(home_odds: int, away_odds: int, draw_odds: int) -> dict:
    """Return both raw implied and fair (vig-removed) probs, plus the vig %."""
    h = american_to_implied(home_odds)
    d = american_to_implied(draw_odds)
    a = american_to_implied(away_odds)
    total = h + d + a
    return {
        "implied": {"home": h, "draw": d, "away": a},
        "fair": {"home": h / total, "draw": d / total, "away": a / total},
        "vig_pct": (total - 1.0) * 100,
    }
```

**Step 4: Run test to verify pass**

```bash
pytest tests/test_no_vig.py -v
```

Expected: 5 passed

**Step 5: Commit**

```bash
git add soccer_ev_model/no_vig.py tests/test_no_vig.py
git commit -q -m "feat(odds): 3-way no-vig probability calculator"
```

---

## Task 5: Data fetcher for historical WC matches

**Objective:** Fetch all matches for WC 2010, 2014, 2018, 2022, 2026 and save to `data/raw/`. One file per year. **Cache-first** — if file exists, don't refetch.

**Files:**
- Create: `/root/soccer-model-lab/soccer_ev_model/fetch_data.py`
- Test: `/root/soccer-model-lab/tests/test_fetch_data.py`

**Step 1: Write failing test**

```python
# tests/test_fetch_data.py
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from soccer_ev_model.fetch_data import fetch_wc_matches, parse_match


def test_parse_match_extracts_essentials():
    raw = {
        "id": 123,
        "utcDate": "2022-12-18T15:00:00Z",
        "status": "FINISHED",
        "stage": "FINAL",
        "group": None,
        "homeTeam": {"id": 762, "name": "Argentina"},
        "awayTeam": {"id": 773, "name": "France"},
        "score": {
            "winner": "HOME_TEAM",
            "duration": "REGULAR",
            "fullTime": {"home": 3, "away": 0},  # simplified
            "halfTime": {"home": 2, "away": 0},
        },
    }
    parsed = parse_match(raw)
    assert parsed["match_id"] == 123
    assert parsed["date"] == "2022-12-18"
    assert parsed["home_team_id"] == 762
    assert parsed["home_team_name"] == "Argentina"
    assert parsed["away_team_id"] == 773
    assert parsed["away_team_name"] == "France"
    assert parsed["home_goals"] == 3
    assert parsed["away_goals"] == 0
    assert parsed["result"] == "H"  # home win
    assert parsed["stage"] == "FINAL"


def test_parse_match_draw():
    raw = {
        "id": 1, "utcDate": "2026-06-16T01:00:00Z", "status": "FINISHED",
        "stage": "GROUP_STAGE", "group": "GROUP_G",
        "homeTeam": {"id": 840, "name": "Iran"},
        "awayTeam": {"id": 783, "name": "New Zealand"},
        "score": {"winner": "DRAW", "duration": "REGULAR",
                  "fullTime": {"home": 2, "away": 2},
                  "halfTime": {"home": 1, "away": 1}},
    }
    parsed = parse_match(raw)
    assert parsed["result"] == "D"


def test_parse_match_away_win():
    raw = {
        "id": 1, "utcDate": "2022-12-18T15:00:00Z", "status": "FINISHED",
        "stage": "GROUP_STAGE", "group": "GROUP_C",
        "homeTeam": {"id": 1, "name": "A"},
        "awayTeam": {"id": 2, "name": "B"},
        "score": {"winner": "AWAY_TEAM", "duration": "REGULAR",
                  "fullTime": {"home": 0, "away": 1},
                  "halfTime": {"home": 0, "away": 0}},
    }
    parsed = parse_match(raw)
    assert parsed["result"] == "A"


def test_fetch_wc_matches_caches_to_disk(tmp_path):
    """If the cache file already exists, fetch_wc_matches should NOT call the API."""
    cache_dir = tmp_path / "raw"
    cache_dir.mkdir()
    cached_file = cache_dir / "matches_2022.json"
    cached_data = [{"match_id": 1, "stub": True}]
    cached_file.write_text(json.dumps(cached_data))

    with patch("soccer_ev_model.fetch_data.FootballDataClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        result = fetch_wc_matches(year=2022, raw_dir=cache_dir, client=mock_client)

    assert result == cached_data
    mock_client.get.assert_not_called()


def test_fetch_wc_matches_calls_api_when_cache_missing(tmp_path):
    """If the cache file is missing, fetch_wc_matches should call the API and save."""
    cache_dir = tmp_path / "raw"
    cache_dir.mkdir()

    api_response = {
        "matches": [
            {"id": 100, "utcDate": "2022-11-20T15:00:00Z", "status": "FINISHED",
             "stage": "GROUP_STAGE", "group": "GROUP_A",
             "homeTeam": {"id": 1, "name": "Qatar"}, "awayTeam": {"id": 2, "name": "Ecuador"},
             "score": {"winner": "HOME_TEAM", "duration": "REGULAR",
                       "fullTime": {"home": 0, "away": 2},
                       "halfTime": {"home": 0, "away": 1}}}
        ]
    }
    with patch("soccer_ev_model.fetch_data.FootballDataClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.get.return_value = api_response
        mock_client_cls.return_value = mock_client
        result = fetch_wc_matches(year=2022, raw_dir=cache_dir, client=mock_client)

    assert len(result) == 1
    assert result[0]["home_team_name"] == "Qatar"
    # Verify it was saved
    saved = json.loads((cache_dir / "matches_2022.json").read_text())
    assert saved[0]["home_team_name"] == "Qatar"
```

**Step 2: Run test to verify failure**

```bash
pytest tests/test_fetch_data.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'soccer_ev_model.fetch_data'`

**Step 3: Write minimal implementation**

```python
# soccer_ev_model/fetch_data.py
"""Fetch and cache World Cup match data from football-data.org.

Cache-first: if data/raw/matches_<year>.json exists, return it. Otherwise fetch
from API, parse, and save.
"""
import json
from datetime import datetime
from pathlib import Path
from .api_client import FootballDataClient


def parse_match(raw: dict) -> dict:
    """Convert a raw API match object into a flat, training-friendly dict."""
    score = raw.get("score", {})
    ft = score.get("fullTime", {}) or {}
    ht = score.get("halfTime", {}) or {}
    winner = score.get("winner")  # HOME_TEAM / AWAY_TEAM / DRAW
    if winner == "HOME_TEAM":
        result = "H"
    elif winner == "AWAY_TEAM":
        result = "A"
    elif winner == "DRAW":
        result = "D"
    else:
        result = None  # match not finished yet

    utc = raw.get("utcDate", "")
    date_str = utc[:10] if utc else None

    return {
        "match_id": raw.get("id"),
        "date": date_str,
        "datetime_utc": utc,
        "status": raw.get("status"),
        "stage": raw.get("stage"),
        "group": raw.get("group"),
        "home_team_id": raw.get("homeTeam", {}).get("id"),
        "home_team_name": raw.get("homeTeam", {}).get("name"),
        "away_team_id": raw.get("awayTeam", {}).get("id"),
        "away_team_name": raw.get("awayTeam", {}).get("name"),
        "home_goals": ft.get("home"),
        "away_goals": ft.get("away"),
        "home_goals_ht": ht.get("home"),
        "away_goals_ht": ht.get("away"),
        "result": result,
        "duration": score.get("duration"),
    }


def fetch_wc_matches(year: int, raw_dir: Path, client: FootballDataClient | None = None) -> list[dict]:
    """Fetch all matches for a given WC year. Cache-first.

    Args:
        year: e.g. 2022
        raw_dir: directory like data/raw/
        client: optional FootballDataClient (creates one with .env token if None)
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_file = raw_dir / f"matches_{year}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text())

    if client is None:
        client = FootballDataClient()

    # The WC competition code is "WC" on football-data.org
    response = client.get(f"/competitions/WC/matches?season={year}")
    matches = response.get("matches", [])
    parsed = [parse_match(m) for m in matches]

    cache_file.write_text(json.dumps(parsed, indent=2))
    return parsed
```

**Step 4: Run test to verify pass**

```bash
pytest tests/test_fetch_data.py -v
```

Expected: 5 passed

**Step 5: Commit**

```bash
git add soccer_ev_model/fetch_data.py tests/test_fetch_data.py
git commit -q -m "feat(data): WC match fetcher with parse + cache"
```

---

## Task 6: Pi-rating computation (TDD)

**Objective:** Compute dynamic pi-ratings from a list of historical matches. The function takes matches and a cutoff date — only matches before the cutoff are used to compute the rating. Sequential update per match.

**Files:**
- Create: `/root/soccer-model-lab/soccer_ev_model/pi_ratings.py`
- Test: `/root/soccer-model-lab/tests/test_pi_ratings.py`

**Step 1: Write failing test**

```python
# tests/test_pi_ratings.py
from soccer_ev_model.pi_ratings import compute_pi_ratings, pi_diff_features


def test_compute_pi_ratings_initial_state():
    """A team with no matches should have neutral pi-rating of 0.0."""
    ratings = compute_pi_ratings(matches=[])
    assert ratings == {}


def test_compute_pi_ratings_single_match_win_increases_offense():
    """A home win should increase the home team's offensive rating above 0."""
    matches = [
        {"date": "2022-01-01", "home_team_id": 1, "away_team_id": 2,
         "home_goals": 2, "away_goals": 0, "result": "H"}
    ]
    ratings = compute_pi_ratings(matches)
    assert ratings[1]["offense"] > 0
    assert ratings[2]["defense"] < 0  # conceded 2


def test_compute_pi_ratings_cutoff_excludes_later_matches():
    """Matches after the cutoff date must not affect the rating."""
    matches = [
        {"date": "2022-01-01", "home_team_id": 1, "away_team_id": 2,
         "home_goals": 5, "away_goals": 0, "result": "H"},
        {"date": "2022-12-01", "home_team_id": 1, "away_team_id": 2,
         "home_goals": 0, "away_goals": 5, "result": "A"},
    ]
    r_before = compute_pi_ratings(matches, cutoff="2022-06-01")
    r_all = compute_pi_ratings(matches)
    # The team that won 5-0 in the first match but lost 0-5 in the second
    # should have LOWER offense rating when both matches are included
    assert r_before[1]["offense"] > r_all[1]["offense"]


def test_compute_pi_ratings_returns_offense_and_defense():
    """Each team should have both an offense and defense rating."""
    matches = [
        {"date": "2022-01-01", "home_team_id": 1, "away_team_id": 2,
         "home_goals": 1, "away_goals": 0, "result": "H"},
        {"date": "2022-02-01", "home_team_id": 2, "away_team_id": 1,
         "home_goals": 0, "away_goals": 1, "result": "A"},
    ]
    ratings = compute_pi_ratings(matches)
    for team_id in [1, 2]:
        assert "offense" in ratings[team_id]
        assert "defense" in ratings[team_id]
        assert "matches_played" in ratings[team_id]


def test_pi_diff_features_for_matchup():
    """Given ratings and a matchup, return the difference features."""
    ratings = {
        1: {"offense": 1.5, "defense": 0.5, "matches_played": 5},
        2: {"offense": 0.3, "defense": -0.2, "matches_played": 3},
    }
    features = pi_diff_features(home_id=1, away_id=2, ratings=ratings)
    # The home team is stronger. Diff should be positive.
    assert features["pi_off_diff"] > 0  # home_off - away_off
    assert features["pi_def_diff"] > 0  # home_def - away_def (higher = better)
    assert features["pi_matchup"] > 0  # combined strength diff
    assert features["pi_home_off"] == 1.5
    assert features["pi_away_off"] == 0.3
```

**Step 2: Run test to verify failure**

```bash
pytest tests/test_pi_ratings.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'soccer_ev_model.pi_ratings'`

**Step 3: Write minimal implementation**

Pi-rating math (from Constantinou & Fenton 2013, simplified for goal-difference):

For each match result, update the team's offense and defense:
- A team that scores more than expected raises its offense rating
- A team that concedes more than expected lowers its defense rating
- "Expected" is based on a simple Poisson-like formula

For a first-pass implementation that captures the spirit without all the original math, we'll use a simple goal-difference update with home advantage and a shrinkage factor:

```python
# soccer_ev_model/pi_ratings.py
"""Pi-rating computation for football teams.

Pi-ratings (Constantinou & Fenton 2013) are dynamic team strength ratings that
update after each match. They outperform Elo for football because they account
for goal margin, not just W/D/L.

This is a simplified implementation. The original paper uses a more complex
update rule with a home-advantage coefficient and several tuning parameters.
We use a goal-difference update with shrinkage:
    offense_new = offense_old + (goals_scored - expected_goals) * c
    defense_new = defense_old + (goals_conceded - expected_conceded) * c
where c is a learning rate (we use 0.05) and expected_goals is a simple
function of the opponent's defense rating.

The function is leak-safe by construction: a `cutoff` parameter controls
which matches are processed. Matches on or after the cutoff are ignored.
"""
from typing import Iterable


LEARNING_RATE = 0.05
HOME_ADVANTAGE = 0.4  # boost to expected goals when playing at home


def _expected_goals(opponent_defense: float, home_advantage: float = 0.0) -> float:
    """Simple model: expected goals for a team = 1.3 - opponent_defense/4 + home_advantage."""
    return max(0.0, 1.3 - opponent_defense / 4.0 + home_advantage)


def compute_pi_ratings(
    matches: Iterable[dict],
    cutoff: str | None = None,
    learning_rate: float = LEARNING_RATE,
) -> dict[int, dict]:
    """Compute pi-ratings for all teams from a chronological list of matches.

    Args:
        matches: iterable of dicts with keys:
            date (str, YYYY-MM-DD), home_team_id, away_team_id,
            home_goals, away_goals, result
        cutoff: if provided, only process matches with date < cutoff.
        learning_rate: how quickly ratings adapt to new results.

    Returns:
        dict mapping team_id -> {"offense": float, "defense": float, "matches_played": int}
    """
    if cutoff is not None:
        matches = [m for m in matches if m.get("date", "") < cutoff]

    # Sort by date so updates are sequential
    matches = sorted(matches, key=lambda m: m.get("date", ""))

    ratings: dict[int, dict] = {}

    def _ensure(team_id):
        if team_id not in ratings:
            ratings[team_id] = {"offense": 0.0, "defense": 0.0, "matches_played": 0}

    for m in matches:
        hg = m.get("home_goals")
        ag = m.get("away_goals")
        if hg is None or ag is None:
            continue  # match not finished

        h = m["home_team_id"]
        a = m["away_team_id"]
        _ensure(h)
        _ensure(a)

        exp_h = _expected_goals(ratings[a]["defense"], HOME_ADVANTAGE)
        exp_a = _expected_goals(ratings[h]["defense"], 0.0)

        ratings[h]["offense"] += (hg - exp_h) * learning_rate
        ratings[h]["defense"] += (ag - exp_h * 0.8) * learning_rate  # conceded
        ratings[a]["offense"] += (ag - exp_a) * learning_rate
        ratings[a]["defense"] += (hg - exp_a * 0.8) * learning_rate

        ratings[h]["matches_played"] += 1
        ratings[a]["matches_played"] += 1

    return ratings


def pi_diff_features(home_id: int, away_id: int, ratings: dict[int, dict]) -> dict[str, float]:
    """Compute the pi-rating difference features for a matchup.

    Returns a flat dict suitable for a model row.
    """
    h = ratings.get(home_id, {"offense": 0.0, "defense": 0.0, "matches_played": 0})
    a = ratings.get(away_id, {"offense": 0.0, "defense": 0.0, "matches_played": 0})

    pi_off_diff = h["offense"] - a["offense"]
    pi_def_diff = h["defense"] - a["defense"]
    # Matchup strength: home attack vs away defense, plus away attack vs home defense
    pi_matchup = (h["offense"] - a["defense"]) - (a["offense"] - h["defense"])

    return {
        "pi_home_off": h["offense"],
        "pi_home_def": h["defense"],
        "pi_away_off": a["offense"],
        "pi_away_def": a["defense"],
        "pi_off_diff": pi_off_diff,
        "pi_def_diff": pi_def_diff,
        "pi_matchup": pi_matchup,
        "pi_home_n": h["matches_played"],
        "pi_away_n": a["matches_played"],
    }
```

**Step 4: Run test to verify pass**

```bash
pytest tests/test_pi_ratings.py -v
```

Expected: 5 passed

**Step 5: Commit**

```bash
git add soccer_ev_model/pi_ratings.py tests/test_pi_ratings.py
git commit -q -m "feat(pi): pi-rating computation with cutoff + matchup features"
```

---

## Task 7: Real data fetch (NOT a test, a script)

**Objective:** Run the fetcher against the real API. Cache the 5 historical WCs. **Bounded runtime:** 5-10 minutes max, ~6s per call.

**Files:**
- Create: `/root/soccer-model-lab/scripts/fetch_historical_wc.py`

**Step 1: Write the script**

```python
# scripts/fetch_historical_wc.py
"""One-shot script to fetch and cache WC 2010, 2014, 2018, 2022, 2026.

Usage:
    source .venv/bin/activate
    python scripts/fetch_historical_wc.py
"""
import sys
from pathlib import Path

# Make the package importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from soccer_ev_model.api_client import FootballDataClient
from soccer_ev_model.fetch_data import fetch_wc_matches


def main():
    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    client = FootballDataClient()

    years = [2010, 2014, 2018, 2022, 2026]
    for year in years:
        cache = raw_dir / f"matches_{year}.json"
        if cache.exists():
            print(f"✓ {year}: cached at {cache}")
            continue
        print(f"→ Fetching WC {year}...")
        matches = fetch_wc_matches(year=year, raw_dir=raw_dir, client=client)
        print(f"  Saved {len(matches)} matches to {cache}")


if __name__ == "__main__":
    main()
```

**Step 2: Run the script (Takes ~5 minutes due to 6s throttle + 5 tournaments)**

```bash
cd /root/soccer-model-lab
source .venv/bin/activate
python scripts/fetch_historical_wc.py
```

Expected output: 5 lines, one per year. Each non-cached year takes ~6s (one API call). Total expected: ~30 seconds if cache is fresh, but first time it's 5 tournaments × 1 call = 30s of actual wait. 

**Step 3: Verify the data**

```bash
cd /root/soccer-model-lab
ls -la data/raw/
python3 -c "
import json
for y in [2010, 2014, 2018, 2022, 2026]:
    data = json.loads(open(f'data/raw/matches_{y}.json').read())
    print(f'{y}: {len(data)} matches, first: {data[0][\"home_team_name\"]} vs {data[0][\"away_team_name\"]}, result={data[0][\"result\"]}')
"
```

Expected: 5 years with realistic match counts (2010-2022 = 64 matches each, 2026 = 104).

**Step 4: Spot-check the pi-ratings work on real data**

```bash
cd /root/soccer-model-lab
source .venv/bin/activate
python3 -c "
import json
from soccer_ev_model.pi_ratings import compute_pi_ratings, pi_diff_features

# Load 2022 matches
matches = json.loads(open('data/raw/matches_2022.json').read())
ratings = compute_pi_ratings(matches)
print(f'Computed ratings for {len(ratings)} teams')
top_5 = sorted(ratings.items(), key=lambda x: x[1]['offense'], reverse=True)[:5]
for tid, r in top_5:
    print(f'  Team {tid}: off={r[\"offense\"]:.2f} def={r[\"defense\"]:.2f} n={r[\"matches_played\"]}')

# Test a 2022 final matchup: Argentina vs France
# (You can find IDs from the cached data)
print()
print('Example: features for first 2022 match')
m = matches[0]
feat = pi_diff_features(m['home_team_id'], m['away_team_id'], ratings)
for k, v in feat.items():
    print(f'  {k}: {v:.3f}')
"
```

Expected: Real ratings with realistic magnitudes. Argentina and France should rank near the top for 2022 (they made the final).

**Step 5: Commit**

```bash
cd /root/soccer-model-lab
git add scripts/fetch_historical_wc.py
git commit -q -m "data: fetch + cache WC 2010-2026"
# data/raw/*.json is gitignored, no need to add
```

---

## Task 8: Smoke test on real data — predict today's matches

**Objective:** End-to-end sanity check. Pull today's WC 2026 fixtures, compute pi-rating features, show the user what the predictions would look like. **No model yet** — just the pi-rating based heuristic.

**Files:**
- Create: `/root/soccer-model-lab/scripts/smoke_test_today.py`

**Step 1: Write the script**

```python
# scripts/smoke_test_today.py
"""End-to-end smoke test. Prints pi-rating based 'naive' probabilities for today's WC matches.

This is NOT a model output. It's the simplest possible baseline: compute pi-rating
matchup features for today, then convert to a 3-way probability using a simple
formula. Phase 2 will replace this with a real trained model.

Usage:
    source .venv/bin/activate
    python scripts/smoke_test_today.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from soccer_ev_model.fetch_data import fetch_wc_matches
from soccer_ev_model.pi_ratings import compute_pi_ratings, pi_diff_features


def naive_3way(pi_matchup: float, home_advantage: float = 0.15) -> dict:
    """Convert a pi-matchup score to a naive 3-way probability.

    pi_matchup > 0 means home is stronger. The home_advantage is added to
    bias toward home wins (a small but real effect in international play
    even at neutral venues when one team has the crowd).
    """
    import math
    # Squish pi_matchup into a -1..+1 range with a tanh
    s = math.tanh(pi_matchup + home_advantage)
    # Base rate: in WC, home win ~50%, draw ~25%, away win ~25%
    p_home = 0.50 + 0.20 * s
    p_draw = 0.25 - 0.05 * abs(s)
    p_away = 1.0 - p_home - p_draw
    return {"home": p_home, "draw": p_draw, "away": p_away}


def main():
    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    today = datetime.utcnow().strftime("%Y-%m-%d")
    print(f"=== Smoke test for {today} (UTC) ===\n")

    # Load all historical WCs up to today
    all_matches = []
    for year in [2010, 2014, 2018, 2022, 2026]:
        f = raw_dir / f"matches_{year}.json"
        if f.exists():
            data = json.loads(f.read_text())
            # Only finished matches with dates before today
            data = [m for m in data if m.get("result") and m.get("date", "") < today]
            all_matches.extend(data)
            print(f"  Loaded {len(data)} finished matches from WC {year}")

    if not all_matches:
        print("No historical data found. Run scripts/fetch_historical_wc.py first.")
        return

    # Compute pi-ratings using all data strictly before today
    ratings = compute_pi_ratings(all_matches, cutoff=today)
    print(f"\n  Computed pi-ratings for {len(ratings)} teams\n")

    # Find today's fixtures
    today_fixtures = [m for m in json.loads((raw_dir / "matches_2026.json").read_text())
                      if m.get("date") == today]
    if not today_fixtures:
        print(f"  No WC 2026 fixtures scheduled for {today}")
        return

    print(f"=== Today's {len(today_fixtures)} WC match(es) ===\n")
    for m in today_fixtures:
        h, a = m["home_team_name"], m["away_team_name"]
        hid, aid = m["home_team_id"], m["away_team_id"]
        feat = pi_diff_features(hid, aid, ratings)
        probs = naive_3way(feat["pi_matchup"])
        print(f"{m['datetime_utc'][11:16]} UTC | {h} vs {a}")
        print(f"  Stage: {m['stage']} | Group: {m.get('group')}")
        print(f"  Pi features: off_diff={feat['pi_off_diff']:.2f}, def_diff={feat['pi_def_diff']:.2f}, matchup={feat['pi_matchup']:.2f}")
        print(f"  NAIVE probs (pi-based heuristic, NOT a model):")
        print(f"    P({h})   = {probs['home']:.3f}")
        print(f"    P(Draw)  = {probs['draw']:.3f}")
        print(f"    P({a})  = {probs['away']:.3f}")
        print()


if __name__ == "__main__":
    main()
```

**Step 2: Run the smoke test**

```bash
cd /root/soccer-model-lab
source .venv/bin/activate
python scripts/smoke_test_today.py
```

Expected: A report of today's WC matches with pi-rating-based naive probabilities. Even though it's not a real model, the output format previews what Phase 2 will produce.

**Step 3: Commit**

```bash
git add scripts/smoke_test_today.py
git commit -q -m "feat(smoke): end-to-end test using real data + pi-ratings"
```

---

## Decision backtest design (Phase 3, not yet implemented)

When Phase 3 is built, the backtest will:
- For each WC year in [2010, 2014, 2018, 2022], predict every match using only data with `date < match_date`
- Report RPS (ranked probability score, primary metric), Brier, log loss
- Compare to naive baselines:
  - "Always pick home win" (RPS ~0.44)
  - "Pick by historical home/draw/away rates" (RPS ~0.23)
  - "Pick by Elo" (RPS ~0.21)
  - "Pick by market no-vig" (RPS ~0.20 — the sharpest baseline)
- Report ROI on flat-stake model picks vs market picks

The honest test: **does our model RPS beat the market no-vig RPS?** If yes, we have a tool. If no, we have a hobby.

---

## Verification checklist (Phase 1 complete when all checked)

- [ ] All 8 tasks committed
- [ ] `pytest -v` shows all tests passing
- [ ] `data/raw/matches_{2010,2014,2018,2022,2026}.json` exist with correct match counts
- [ ] `no_vig.no_vig_3way(-200, 550, 340)` returns `{implied: {...}, fair: {...}, vig_pct: 4.8}`
- [ ] `compute_pi_ratings` on real 2022 data gives Argentina and France top ratings
- [ ] `scripts/smoke_test_today.py` runs and shows today's fixtures with naive probs
- [ ] `.env` is NOT in git (verify with `git log --all -- .env` shows nothing)
- [ ] No `__pycache__` committed (`git ls-files | grep __pycache__` is empty)

---

## Known limitations / honest caveats

- **Pi-rating math is simplified.** The original paper has more parameters and a more complex update rule. Our version captures the spirit (goal-difference updates with shrinkage) but won't match published pi-rating tools exactly. This is fine — we're training a model on top of pi-ratings as features, not using pi-ratings as the final predictor.

- **Cutoff for "future" is today's date.** The pi-rating cutoff in the smoke test is "today" (UTC). This is correct: we use all data before now, not before the match. In Phase 3's backtest, the cutoff will be the match's date.

- **No Elo source yet.** Phase 1 doesn't include Elo data. Phase 2 will either scrape eloratings.net (respectfully) or pull from their GitHub CSV.

- **No markets data yet.** The no-vig utility works on whatever odds you give it. To get the "compare to market" picture, you'd paste in book odds manually. Phase 2 doesn't change this — keeping model and market separate by design.

---

## Next phase preview (Phase 2, NOT started)

When you give the green light, Phase 2 adds:
- `features.py` — feature engineering (rest days, recent form, draw rate, Elo join, tournament stage)
- `train.py` — CatBoost multiclass with walk-forward backtest
- `predict.py` — given today's fixtures, output W/D/L probs from the trained model

Phase 2 install adds catboost + xgboost + scikit-learn via `pip install -e .[phase2]`.
