"""Fetch and cache historical World Cup match data from football-data.org.

We pull WC match data for selected years and save it as flat JSON files in
data/raw/. The parser converts the API's nested JSON into a list of clean
match records with stable field names. All date math and team-id joins
downstream rely on this shape, so the parser is well-tested.

Why a fetcher module (vs. doing it inline in a script):
- The parser is the only place with non-trivial logic, so it lives in the
  package and gets unit tests.
- The actual HTTP fetching is one short function; it's exercised by a real
  smoke script in scripts/ rather than by mocked tests.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from soccer_ev_model.api_client import FootballDataClient


# World Cup season codes used by football-data.org (the API uses the year of
# the final, not the year the tournament started).
WC_SEASONS = [2010, 2014, 2018, 2022, 2026]

# openfootball/worldcup.json has WC data 1930-2026 as static JSON on GitHub.
# Free, no API key, no rate limit (it's GitHub raw content).
# Used for HISTORICAL data (1930-2022). Current 2026 is fetched from
# football-data.org (which has live scores and today's games on free tier).
OPENFOOTBALL_RAW_BASE = (
    "https://raw.githubusercontent.com/openfootball/worldcup.json/master"
)


def normalize_team_name(name: str) -> str:
    """Canonical form of a team name for joining across data sources.

    Lowercase, strip whitespace, remove dashes, ampersands, and the word
    'and'. This makes 'Bosnia-Herzegovina', 'Bosnia & Herzegovina', and
    'Bosnia and Herzegovina' all map to 'bosniaherzegovina', so the
    football-data.org source and the openfootball source can be joined
    on (year, normalized_team_name).
    """
    if not name:
        return ""
    s = name.lower().strip()
    s = s.replace("-", "").replace("&", "").replace(" and ", "").replace(" ", "")
    return s


def team_id_from_name(name: str) -> int:
    """Synthesize a stable integer id from a team name.

    openfootball has no team IDs, so we hash the normalized name. This
    gives us a stable id that is consistent across calls and years. It
    will NOT match football-data.org's real team ids, but that's fine
    because we never need to join across both sources at the same time
    (historical = openfootball, live = football-data.org).
    """
    norm = normalize_team_name(name)
    # Python's hash() is randomized per process. Use a stable hash instead.
    import hashlib
    h = hashlib.md5(norm.encode("utf-8")).hexdigest()
    # Take first 8 hex chars as int (32-bit). Stable across runs.
    return int(h[:8], 16)


def parse_matches(api_response: dict) -> list[dict]:
    """Convert one football-data.org /matches response into flat match records.

    Each record is a dict with the following stable fields (all downstream
    code is written against these names):
        id               int   API match id
        date             str   ISO 8601 UTC, e.g. "2022-11-20T15:00:00Z"
        status           str   API status, e.g. "FINISHED", "TIMED", "IN_PLAY"
        matchday         int   Round within stage
        stage            str   e.g. "GROUP_STAGE", "LAST_16", "QUARTER_FINALS"
        group            str|None  e.g. "GROUP_A" or None for knockout
        home_team_id     int
        home_team_name   str
        away_team_id     int
        away_team_name   str
        home_goals       int|None  full-time home goals, or None if not played
        away_goals       int|None  full-time away goals, or None if not played
        result           str|None  "H" / "D" / "A" / None if not played

    The output order matches the input order (which is the API's order).
    An empty API response returns an empty list.
    """
    out: list[dict] = []
    for m in api_response.get("matches", []):
        score = m.get("score") or {}
        full = score.get("fullTime") or {}
        winner = score.get("winner")

        # Map the API's "HOME_TEAM" / "AWAY_TEAM" / "DRAW" to single-letter codes.
        if winner == "HOME_TEAM":
            result = "H"
        elif winner == "AWAY_TEAM":
            result = "A"
        elif winner == "DRAW":
            result = "D"
        else:
            result = None

        out.append(
            {
                "id": m.get("id"),
                "date": m.get("utcDate"),
                "status": m.get("status"),
                "matchday": m.get("matchday"),
                "stage": m.get("stage"),
                "group": m.get("group"),
                "home_team_id": (m.get("homeTeam") or {}).get("id"),
                "home_team_name": (m.get("homeTeam") or {}).get("name"),
                "away_team_id": (m.get("awayTeam") or {}).get("id"),
                "away_team_name": (m.get("awayTeam") or {}).get("name"),
                "home_goals": full.get("home"),
                "away_goals": full.get("away"),
                "result": result,
            }
        )
    return out


def fetch_year(
    client: FootballDataClient,
    year: int,
    out_dir: Path,
    force: bool = False,
) -> Path:
    """Fetch all WC matches for one season and save to data/raw/matches_<year>.json.

    Returns the path to the saved file. If the file already exists and
    force=False, the existing file is returned without making an API call.

    For 2026 the endpoint /v4/competitions/WC/matches?season=2026 returns
    both played and unplayed matches. We save them all; downstream code
    filters by status when it needs only finished games.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"matches_{year}.json"

    if out_path.exists() and not force:
        return out_path

    api_response = client.get(f"/competitions/WC/matches?season={year}")
    matches = parse_matches(api_response)
    out_path.write_text(
        json.dumps(
            {
                "year": year,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "count": len(matches),
                "matches": matches,
            },
            indent=2,
        )
    )
    return out_path


def parse_openfootball_matches(json_doc: dict) -> list[dict]:
    """Convert one openfootball/worldcup.json document into flat match records.

    openfootball's shape is:
        {"name": "World Cup 2022",
         "matches": [
            {"round": "Matchday 1",
             "date": "2022-11-20",
             "time": "19:00",
             "team1": "Qatar",         (string name, not id)
             "team2": "Ecuador",        (string name, not id)
             "score": {"ft": [0, 2], "ht": [0, 2]},
             "group": "Group A",
             "goals1": [...], "goals2": [...],  # optional, not used
             ...},
            ...
         ]}

    Output record fields match parse_matches() where possible:
        id, date, status, matchday, stage, group,
        home_team_id, home_team_name, away_team_id, away_team_name,
        home_goals, away_goals, result, source

    Differences from football-data.org records:
        - team IDs are synthesized from normalized names (see team_id_from_name)
        - stage is derived from 'round' (Matchday 1-3 -> GROUP_STAGE, etc.)
        - matchday is parsed from round text
        - source = "openfootball"
    """
    out: list[dict] = []
    matches = json_doc.get("matches", [])
    for m in matches:
        score = m.get("score") or {}
        ft = score.get("ft")
        home_goals = ft[0] if ft else None
        away_goals = ft[1] if ft else None

        if home_goals is None or away_goals is None:
            result = None
        elif home_goals > away_goals:
            result = "H"
        elif home_goals < away_goals:
            result = "A"
        else:
            result = "D"

        # Combine date + time into ISO 8601 (openfootball's time is local-ish, we
        # just append :00 if missing and Z if missing — exact TZ not critical for
        # pi-rating order).
        date = m.get("date", "")
        time_str = m.get("time", "")
        if time_str:
            # Strip any timezone suffix like "UTC+3" since we don't track that
            time_str = re.sub(r"\s*UTC[+\-]\d+", "", time_str)
            if ":" not in time_str:
                time_str = time_str + ":00"
            iso_date = f"{date}T{time_str}:00Z" if "Z" not in time_str else f"{date}T{time_str}"
        else:
            iso_date = f"{date}T00:00:00Z"

        # Stage: GROUP_STAGE if round is "Matchday N" (or "Group N"), else KO stage.
        round_name = (m.get("round") or "").lower()
        if "matchday" in round_name or "group" in round_name:
            stage = "GROUP_STAGE"
            matchday = _parse_matchday(round_name)
        elif "round of 16" in round_name or "last 16" in round_name or "eighth" in round_name:
            stage = "LAST_16"
            matchday = None
        elif "quarter" in round_name:
            stage = "QUARTER_FINALS"
            matchday = None
        elif "semi" in round_name:
            stage = "SEMI_FINALS"
            matchday = None
        elif "final" in round_name and "semi" not in round_name:
            stage = "FINAL"
            matchday = None
        else:
            stage = round_name.upper() or "UNKNOWN"
            matchday = None

        home_name = m.get("team1", "") or ""
        away_name = m.get("team2", "") or ""

        out.append(
            {
                "id": None,  # openfootball has no id
                "date": iso_date,
                "status": "FINISHED" if result is not None else "TIMED",
                "matchday": matchday,
                "stage": stage,
                "group": m.get("group"),
                "home_team_id": team_id_from_name(home_name) if home_name else None,
                "home_team_name": home_name,
                "away_team_id": team_id_from_name(away_name) if away_name else None,
                "away_team_name": away_name,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": result,
                "source": "openfootball",
            }
        )
    return out


def _parse_matchday(round_text: str) -> int | None:
    """Extract matchday number from text like 'matchday 1' or 'group b matchday 3'."""
    m = re.search(r"matchday\s*(\d+)", round_text)
    if m:
        return int(m.group(1))
    return None


def fetch_openfootball_year(year: int, out_dir: Path, force: bool = False) -> Path | None:
    """Fetch one year of openfootball WC data and save to data/raw/matches_<year>_openfootball.json.

    Returns the saved path, or None if the year is not in the repo (404).
    Does not use the football-data.org client — this is a one-shot GitHub
    raw fetch. We use urllib directly and do NOT pass a User-Agent that
    identifies us as a model. (We use a standard browser UA since this
    is a public static file.)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"matches_{year}_openfootball.json"

    if out_path.exists() and not force:
        return out_path

    url = f"{OPENFOOTBALL_RAW_BASE}/{year}/worldcup.json"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Hermes-Research"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise

    json_doc = json.loads(raw)
    matches = parse_openfootball_matches(json_doc)
    out_path.write_text(
        json.dumps(
            {
                "year": year,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "count": len(matches),
                "matches": matches,
            },
            indent=2,
        )
    )
    return out_path
