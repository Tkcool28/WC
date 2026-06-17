"""Tests for the dashboard's 2026 WC cached data loader.

The dashboard reads `data/raw/matches_2026.json` (a read-only file fetched
out-of-band by `scripts/fetch_live_2026.py`) and auto-fills the day's
matchups so the user only has to type in book odds. These tests pin the
filtering rules: unplayed games on the given date, non-NULL team names,
sorted by kickoff time.

Date semantics
--------------
The "day" in this loader is **Mountain Time (America/Denver)**, not UTC.
The cache stores kickoffs in UTC, but the user is in Denver, so a game
at 2026-06-17T01:00:00Z (7 PM MDT on the 16th) is filtered under
"2026-06-16". Each game is converted to MT for the date comparison; the
original UTC kickoff is preserved in `kickoff_iso`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dashboard.data_loader import (
    CACHE_FILENAME,
    UnplayedMatch,
    get_unplayed_matches,
    list_dates_with_unplayed,
    load_matches_cache,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Fixtures: synthetic caches that mimic matches_2026.json
# --------------------------------------------------------------------------- #

def _m(
    match_id: int,
    iso: str,
    home: str | None,
    away: str | None,
    status: str = "TIMED",
    home_goals=None,
    away_goals=None,
    result=None,
    home_id: int = 1,
    away_id: int = 2,
    stage: str = "GROUP_STAGE",
    group: str = "GROUP_A",
) -> dict:
    return {
        "id": match_id,
        "date": iso,
        "status": status,
        "matchday": 1,
        "stage": stage,
        "group": group,
        "home_team_id": home_id,
        "home_team_name": home,
        "away_team_id": away_id,
        "away_team_name": away,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "result": result,
    }


@pytest.fixture
def synthetic_cache(tmp_path: Path) -> Path:
    """Build a small in-memory matches_2026.json with edge cases.

    All kickoffs are in UTC; the loader converts them to MT before comparing
    against the requested date. Layout for "2026-06-16" MT:

      • 2026-06-16T01:00:00Z  = 2026-06-15 19:00 MDT  → previous MT day
      • 2026-06-16T19:00:00Z  = 2026-06-16 13:00 MDT  → Tuesday MT
      • 2026-06-16T22:00:00Z  = 2026-06-16 16:00 MDT  → Tuesday MT
      • 2026-06-16T23:30:00Z  = 2026-06-16 17:30 MDT  → TBD, dropped
      • 2026-06-17T01:00:00Z  = 2026-06-16 19:00 MDT  → Tuesday MT (Argentina)
      • 2026-06-17T04:00:00Z  = 2026-06-16 22:00 MDT  → Tuesday MT (Austria)
      • 2026-06-17T20:00:00Z  = 2026-06-17 14:00 MDT  → Wednesday MT
      • 2026-06-15T18:00:00Z  = 2026-06-15 12:00 MDT  → previous MT day

    So filtering by "2026-06-16" (MT) should yield ids [2, 3, 5, 6]
    (France/Senegal, Iraq/Norway, Argentina/Algeria, Austria/Jordan).
    """
    matches = [
        # FINISHED on the day before — should be filtered out (status)
        _m(1, "2026-06-16T01:00:00Z", "Iran", "New Zealand",
           status="FINISHED", home_goals=1, away_goals=1, result="D"),
        # Unplayed on the day, both teams present — INCLUDE (1 PM MDT)
        _m(2, "2026-06-16T19:00:00Z", "France", "Senegal",
           home_id=10, away_id=11),
        # Unplayed on the day, both teams present — INCLUDE (4 PM MDT)
        _m(3, "2026-06-16T22:00:00Z", "Iraq", "Norway",
           home_id=12, away_id=13),
        # TBD knockout (NULL names) — should be filtered out
        _m(4, "2026-06-16T23:30:00Z", None, None,
           home_id=14, away_id=15),
        # 2026-06-17T01:00:00Z = 7 PM MDT on 2026-06-16 — INCLUDE (Argentina)
        _m(5, "2026-06-17T01:00:00Z", "Argentina", "Algeria",
           home_id=20, away_id=21),
        # 2026-06-17T04:00:00Z = 10 PM MDT on 2026-06-16 — INCLUDE (Austria)
        _m(6, "2026-06-17T04:00:00Z", "Austria", "Jordan",
           status="IN_PLAY", home_id=22, away_id=23),
        # Wednesday MT, not Tuesday — belongs to 2026-06-17
        _m(7, "2026-06-17T20:00:00Z", "Portugal", "Congo DR",
           status="PAUSED", home_id=24, away_id=25),
        # Earlier MT day — should not appear on the 16th
        _m(8, "2026-06-15T18:00:00Z", "Mexico", "South Africa",
           home_id=30, away_id=31),
        # Only one team name is NULL — partial TBD, drop it
        _m(9, "2026-06-17T22:00:00Z", "England", None,
           home_id=40, away_id=41),
    ]
    payload = {"year": 2026, "fetched_at": "2026-06-16T11:44:15Z",
               "count": len(matches), "matches": matches}
    p = tmp_path / "matches_2026.json"
    p.write_text(json.dumps(payload))
    return p


@pytest.fixture
def empty_cache(tmp_path: Path) -> Path:
    p = tmp_path / "matches_2026.json"
    p.write_text(json.dumps({"year": 2026, "fetched_at": "x", "count": 0, "matches": []}))
    return p


@pytest.fixture
def missing_cache(tmp_path: Path) -> Path:
    # A directory that does NOT exist
    return tmp_path / "does_not_exist.json"


# --------------------------------------------------------------------------- #
# load_matches_cache
# --------------------------------------------------------------------------- #

def test_load_matches_cache_returns_full_payload(synthetic_cache: Path):
    payload = load_matches_cache(synthetic_cache)
    assert payload["year"] == 2026
    assert payload["count"] == 9
    assert len(payload["matches"]) == 9


def test_load_matches_cache_missing_file_returns_empty(missing_cache: Path):
    payload = load_matches_cache(missing_cache)
    # When the cache is missing we still get a dict-shape back, never a crash.
    assert payload == {"matches": [], "year": 2026}


def test_load_matches_cache_malformed_returns_empty(tmp_path: Path):
    bad = tmp_path / "matches_2026.json"
    bad.write_text("{not valid json")
    assert load_matches_cache(bad) == {"matches": [], "year": 2026}


# --------------------------------------------------------------------------- #
# get_unplayed_matches — core filter (Mountain Time)
# --------------------------------------------------------------------------- #

def test_get_unplayed_matches_filters_by_date(synthetic_cache: Path):
    """Filtering by '2026-06-16' (MT) must include late-UTC kickoffs whose MT
    date is the 16th, even though their UTC date prefix is 17th."""
    out = get_unplayed_matches("2026-06-16", cache_path=synthetic_cache)
    ids = [m.match_id for m in out]
    # Iran/NZ is FINISHED → out. France/Senegal + Iraq/Norway (13:00/16:00 MDT)
    # are TIMED → in. TBD (None/None) → out. Argentina/Algeria (19:00 MDT on
    # the 16th, but UTC 2026-06-17T01:00:00Z) → in. Austria/Jordan (22:00 MDT
    # on the 16th, but UTC 2026-06-17T04:00:00Z) → in. Portugal (Wed MT) → out.
    assert ids == [2, 3, 5, 6]


def test_get_unplayed_matches_excludes_finished(synthetic_cache: Path):
    """Finished matches (status=FINISHED) must be excluded even on the same date."""
    out = get_unplayed_matches("2026-06-16", cache_path=synthetic_cache)
    for m in out:
        assert m.status != "FINISHED"


def test_get_unplayed_matches_excludes_null_team_names(synthetic_cache: Path):
    """TBD knockout games have None team names — drop them silently."""
    out = get_unplayed_matches("2026-06-16", cache_path=synthetic_cache)
    for m in out:
        assert m.home_team_name
        assert m.away_team_name


def test_get_unplayed_matches_excludes_partial_null(synthetic_cache: Path):
    """If only one team name is null, drop the match too."""
    out = get_unplayed_matches("2026-06-17", cache_path=synthetic_cache)
    ids = [m.match_id for m in out]
    # 7 (Portugal/Congo DR PAUSED) is on Wed MT. 9 (England/None) is dropped.
    assert 9 not in ids
    assert ids == [7]


def test_get_unplayed_matches_includes_in_play_and_paused(synthetic_cache: Path):
    """IN_PLAY / PAUSED (no result yet) count as unplayed."""
    # Austria/Jordan is IN_PLAY on Tue MT 2026-06-16
    out = get_unplayed_matches("2026-06-16", cache_path=synthetic_cache)
    statuses = [m.status for m in out]
    assert "IN_PLAY" in statuses
    assert "TIMED" in statuses


def test_get_unplayed_matches_sorted_by_kickoff(synthetic_cache: Path):
    """Results are returned in chronological order (by MT kickoff) so the
    dashboard lists them left-to-right in the order they kick off for the
    user in Denver."""
    out = get_unplayed_matches("2026-06-16", cache_path=synthetic_cache)
    times = [m.kickoff_mt_iso for m in out]
    assert times == sorted(times)


def test_get_unplayed_matches_empty_date(synthetic_cache: Path):
    """A date with no games returns an empty list, not an error."""
    out = get_unplayed_matches("2026-07-01", cache_path=synthetic_cache)
    assert out == []


def test_get_unplayed_matches_date_format_tolerant(synthetic_cache: Path):
    """Accept either 'YYYY-MM-DD' or full ISO 'YYYY-MM-DDTHH:MM:SSZ'."""
    out_a = get_unplayed_matches("2026-06-16", cache_path=synthetic_cache)
    out_b = get_unplayed_matches("2026-06-16T19:00:00Z", cache_path=synthetic_cache)
    assert [m.match_id for m in out_a] == [m.match_id for m in out_b]


def test_get_unplayed_matches_missing_cache(missing_cache: Path):
    """If the cache file isn't there, return an empty list (not raise)."""
    out = get_unplayed_matches("2026-06-16", cache_path=missing_cache)
    assert out == []


# --------------------------------------------------------------------------- #
# Timezone fix — required tests for the MT date filter
# --------------------------------------------------------------------------- #

def test_tz_argentina_algeria_appears_under_tuesday_mt(tmp_path: Path):
    """Argentina/Algeria kicks off at 2026-06-17T01:00:00Z (UTC) = 7 PM MDT
    on 2026-06-16. The user expects it under "2026-06-16" — the date in MT,
    not the date in UTC.
    """
    cache = tmp_path / "matches_2026.json"
    cache.write_text(json.dumps({
        "year": 2026, "fetched_at": "x", "count": 1,
        "matches": [_m(99, "2026-06-17T01:00:00Z", "Argentina", "Algeria",
                       home_id=20, away_id=21)],
    }))
    out = get_unplayed_matches("2026-06-16", cache_path=cache)
    assert len(out) == 1
    m = out[0]
    assert m.match_id == 99
    assert m.home_team_name == "Argentina"
    assert m.away_team_name == "Algeria"
    # The MT date stored on the record is the 16th (what the user picked).
    assert m.match_date_iso == "2026-06-16"
    # The original UTC kickoff is preserved.
    assert m.kickoff_iso == "2026-06-17T01:00:00Z"
    # And the MT-side timestamp is 7 PM MDT on the 16th.
    assert m.kickoff_mt_iso.startswith("2026-06-16T19:00:00")


def test_tz_austria_jordan_appears_under_tuesday_mt(tmp_path: Path):
    """Austria/Jordan at 2026-06-17T04:00:00Z (UTC) = 10 PM MDT on 2026-06-16.
    Should appear under "2026-06-16" — the late-evening MT case."""
    cache = tmp_path / "matches_2026.json"
    cache.write_text(json.dumps({
        "year": 2026, "fetched_at": "x", "count": 1,
        "matches": [_m(100, "2026-06-17T04:00:00Z", "Austria", "Jordan",
                       home_id=22, away_id=23)],
    }))
    out = get_unplayed_matches("2026-06-16", cache_path=cache)
    assert len(out) == 1
    m = out[0]
    assert m.match_id == 100
    assert m.match_date_iso == "2026-06-16"
    assert m.kickoff_mt_iso.startswith("2026-06-16T22:00:00")


def test_tz_just_after_midnight_mt_lands_on_next_day(tmp_path: Path):
    """A kickoff at 2026-06-17T07:00:00Z (UTC) is 1 AM MDT on 2026-06-17 —
    the user expects it under "2026-06-17", not the 16th."""
    cache = tmp_path / "matches_2026.json"
    cache.write_text(json.dumps({
        "year": 2026, "fetched_at": "x", "count": 1,
        "matches": [_m(101, "2026-06-17T07:00:00Z", "Portugal", "Congo DR",
                       home_id=24, away_id=25)],
    }))
    out_16 = get_unplayed_matches("2026-06-16", cache_path=cache)
    out_17 = get_unplayed_matches("2026-06-17", cache_path=cache)
    assert out_16 == []
    assert len(out_17) == 1
    assert out_17[0].match_id == 101
    assert out_17[0].match_date_iso == "2026-06-17"


def test_tz_list_dates_returns_mt_date_not_utc_date(tmp_path: Path):
    """list_dates_with_unplayed() must bucket Argentina/Algeria (kickoff
    2026-06-17T01:00:00Z) under '2026-06-16' — the MT date the user sees,
    not the UTC date the cache uses."""
    cache = tmp_path / "matches_2026.json"
    cache.write_text(json.dumps({
        "year": 2026, "fetched_at": "x", "count": 2,
        "matches": [
            _m(5, "2026-06-17T01:00:00Z", "Argentina", "Algeria",
               home_id=20, away_id=21),
            # Already-finished game on the 16th — should be skipped entirely.
            _m(1, "2026-06-16T01:00:00Z", "Iran", "New Zealand",
               status="FINISHED", home_goals=1, away_goals=1, result="D"),
        ],
    }))
    dates = list_dates_with_unplayed(cache_path=cache)
    # The Argentina game should put 2026-06-16 in the set (MT date),
    # NOT 2026-06-17 (which would be the UTC-date mistake).
    assert "2026-06-16" in dates
    assert "2026-06-17" not in dates


def test_tz_5pm_mt_sanity(tmp_path: Path):
    """A kickoff at 2026-06-16T23:00:00Z (UTC) is 5 PM MDT on 2026-06-16.
    Sanity check that a normal daytime MT match still lands on the 16th."""
    cache = tmp_path / "matches_2026.json"
    cache.write_text(json.dumps({
        "year": 2026, "fetched_at": "x", "count": 1,
        "matches": [_m(50, "2026-06-16T23:00:00Z", "France", "Senegal",
                       home_id=10, away_id=11)],
    }))
    out = get_unplayed_matches("2026-06-16", cache_path=cache)
    assert len(out) == 1
    m = out[0]
    assert m.match_date_iso == "2026-06-16"
    assert m.kickoff_mt_iso.startswith("2026-06-16T17:00:00")


def test_tz_dst_fallback_handled(tmp_path: Path):
    """After the fall-back (Sun 2026-11-01 02:00 MDT → 01:00 MST), Denver
    switches to UTC-7. A kickoff at 2026-11-02T01:00:00Z should be
    2026-11-01 18:00 MST (NOT 17:00 MDT). The loader must follow DST
    automatically so the date still maps to 2026-11-01."""
    cache = tmp_path / "matches_2026.json"
    cache.write_text(json.dumps({
        "year": 2026, "fetched_at": "x", "count": 1,
        "matches": [_m(200, "2026-11-02T01:00:00Z", "Brazil", "Italy",
                       home_id=30, away_id=31)],
    }))
    out = get_unplayed_matches("2026-11-01", cache_path=cache)
    assert len(out) == 1
    m = out[0]
    # MST is UTC-7, so 01:00Z = 18:00 the day before in MST.
    assert m.kickoff_mt_iso.startswith("2026-11-01T18:00:00")


# --------------------------------------------------------------------------- #
# Output typing & shape
# --------------------------------------------------------------------------- #

def test_unplayed_match_dataclass_fields(synthetic_cache: Path):
    """The dataclass exposes the fields the dashboard renders."""
    out = get_unplayed_matches("2026-06-16", cache_path=synthetic_cache)
    assert len(out) == 4
    m = out[0]
    assert isinstance(m, UnplayedMatch)
    assert m.home_team_name == "France"
    assert m.away_team_name == "Senegal"
    assert m.match_id == 2
    assert m.kickoff_iso == "2026-06-16T19:00:00Z"
    # match_date_iso is the MT date (2026-06-16), same as the picker value.
    assert m.match_date_iso == "2026-06-16"


def test_unplayed_match_to_dict_round_trips(synthetic_cache: Path):
    """The dashboard reads matches as dicts; the dataclass must serialize cleanly."""
    out = get_unplayed_matches("2026-06-16", cache_path=synthetic_cache)
    d = out[0].to_dict()
    assert d["home_team_name"] == "France"
    assert d["match_id"] == 2
    # Round-trips through stdlib json
    assert json.loads(json.dumps(d)) == d


# --------------------------------------------------------------------------- #
# list_dates_with_unplayed — used for cache invalidation / debugging
# --------------------------------------------------------------------------- #

def test_list_dates_with_unplayed(synthetic_cache: Path):
    dates = list_dates_with_unplayed(cache_path=synthetic_cache)
    # Tuesday MT 2026-06-16 has 4 games (ids 2, 3, 5, 6).
    # Wednesday MT 2026-06-17 has 1 game (id 7).
    # Monday MT 2026-06-15 has 1 game (id 8).
    assert "2026-06-16" in dates
    assert "2026-06-17" in dates
    assert "2026-06-15" in dates
    assert dates == sorted(dates)


def test_list_dates_with_unplayed_missing_cache(missing_cache: Path):
    assert list_dates_with_unplayed(cache_path=missing_cache) == []


# --------------------------------------------------------------------------- #
# Default cache path points at the real file
# --------------------------------------------------------------------------- #

def test_default_cache_filename_is_matches_2026():
    """The loader's default file name must match the cached WC data file."""
    assert CACHE_FILENAME == "matches_2026.json"


# --------------------------------------------------------------------------- #
# evaluate_one_game (dashboard app helper) — validation paths
# --------------------------------------------------------------------------- #

import importlib  # noqa: E402


@pytest.fixture
def app_module():
    """Import the dashboard app with streamlit.cache_data stubbed.

    The app uses @st.cache_data heavily; we don't need the real Streamlit
    runtime for these tests, just the function surface.
    """
    import streamlit as st
    st.cache_data = lambda *a, **k: (lambda f: f)
    import sys
    sys.path.insert(0, str(_PROJECT_ROOT))
    if "dashboard.app" in sys.modules:
        return sys.modules["dashboard.app"]
    return importlib.import_module("dashboard.app")


def test_evaluate_one_game_rejects_empty_team_names(app_module):
    out = app_module.evaluate_one_game(
        home_name="",
        away_name="France",
        home_team_id=1,
        away_team_id=2,
        cutoff_iso="2026-06-16",
        home_odds_txt="-150",
        draw_odds_txt="+300",
        away_odds_txt="+400",
        ratings={1: {"offense": 0.0, "defense": 0.0, "matches_played": 10},
                 2: {"offense": 0.0, "defense": 0.0, "matches_played": 10}},
        min_edge=0.03,
        name_to_id={},
    )
    assert out["ok"] is False
    assert "home" in out["error"].lower() or "away" in out["error"].lower()


def test_evaluate_one_game_rejects_identical_teams(app_module):
    out = app_module.evaluate_one_game(
        home_name="France",
        away_name="france",
        home_team_id=1,
        away_team_id=2,
        cutoff_iso="2026-06-16",
        home_odds_txt="-150",
        draw_odds_txt="+300",
        away_odds_txt="+400",
        ratings={1: {}, 2: {}},
        min_edge=0.03,
        name_to_id={"France": 1},
    )
    assert out["ok"] is False
    assert "differ" in out["error"].lower()


def test_evaluate_one_game_rejects_unparseable_odds(app_module):
    out = app_module.evaluate_one_game(
        home_name="France",
        away_name="Senegal",
        home_team_id=1,
        away_team_id=2,
        cutoff_iso="2026-06-16",
        home_odds_txt="not-a-number",
        draw_odds_txt="+300",
        away_odds_txt="+400",
        ratings={1: {}, 2: {}},
        min_edge=0.03,
        name_to_id={},
    )
    assert out["ok"] is False
    assert "odds" in out["error"].lower()


def test_evaluate_one_game_rejects_unknown_team(app_module):
    out = app_module.evaluate_one_game(
        home_name="Atlantis",
        away_name="Senegal",
        home_team_id=None,  # cache didn't have it; we fall back to map
        away_team_id=2,
        cutoff_iso="2026-06-16",
        home_odds_txt="-150",
        draw_odds_txt="+300",
        away_odds_txt="+400",
        ratings={1: {}, 2: {}},
        min_edge=0.03,
        name_to_id={"Senegal": 2},  # Atlantis missing
    )
    assert out["ok"] is False
    assert "atlantis" in out["error"].lower()


def test_evaluate_one_game_rejects_empty_ratings(app_module):
    out = app_module.evaluate_one_game(
        home_name="France",
        away_name="Senegal",
        home_team_id=1,
        away_team_id=2,
        cutoff_iso="2026-06-16",
        home_odds_txt="-150",
        draw_odds_txt="+300",
        away_odds_txt="+400",
        ratings={},  # empty ratings (no teams rated)
        min_edge=0.03,
        name_to_id={"France": 1, "Senegal": 2},
    )
    assert out["ok"] is False
    assert "ratings" in out["error"].lower() or "cutoff" in out["error"].lower()


def test_evaluate_one_game_happy_path(app_module):
    """With valid inputs and a tiny synthetic ratings dict, evaluation succeeds."""
    from soccer_ev_model.pi_ratings import compute_pi_ratings

    # Build a tiny training corpus and compute real pi-ratings
    train = []
    for i in range(40):
        train.append({
            "match_id": f"x{i}", "date": f"2020-{(i % 9) + 1:02d}-01",
            "home_team": "France", "away_team": "Weak",
            "home_team_id": 1, "away_team_id": 2,
            "home_goals": 2, "away_goals": 0, "result": "H",
        })
        train.append({
            "match_id": f"y{i}", "date": f"2020-{(i % 9) + 1:02d}-02",
            "home_team": "Senegal", "away_team": "Weak",
            "home_team_id": 2, "away_team_id": 3,
            "home_goals": 1, "away_goals": 1, "result": "D",
        })
    ratings = compute_pi_ratings(train, cutoff="2020-12-01")

    out = app_module.evaluate_one_game(
        home_name="France",
        away_name="Senegal",
        home_team_id=1,
        away_team_id=2,
        cutoff_iso="2020-12-01",
        home_odds_txt="-150",
        draw_odds_txt="+300",
        away_odds_txt="+400",
        ratings=ratings,
        min_edge=0.03,
        name_to_id={"France": 1, "Senegal": 2},
    )
    assert out["ok"] is True
    r = out["result"]
    assert r["home_team"] == "France"
    assert r["away_team"] == "Senegal"
    assert "pi_probs" in r
    assert "edges" in r


def test_evaluate_one_game_strips_plus_sign_on_odds(app_module):
    """'+350' must parse the same as '350' (American odds convention)."""
    from soccer_ev_model.pi_ratings import compute_pi_ratings
    train = [{
        "match_id": f"x{i}", "date": f"2020-{i+1:02d}-01",
        "home_team": "France", "away_team": "Weak",
        "home_team_id": 1, "away_team_id": 2,
        "home_goals": 2, "away_goals": 0, "result": "H",
    } for i in range(40)]
    ratings = compute_pi_ratings(train, cutoff="2020-12-01")

    out = app_module.evaluate_one_game(
        home_name="France", away_name="Weak",
        home_team_id=1, away_team_id=2,
        cutoff_iso="2020-12-01",
        home_odds_txt="-150", draw_odds_txt="+300", away_odds_txt="400",
        ratings=ratings, min_edge=0.03,
        name_to_id={"France": 1, "Weak": 2},
    )
    assert out["ok"] is True, out
