"""Tests for the Elo ratings loader and feature integration."""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

# Make `soccer_ev_model` importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soccer_ev_model.elo_ratings import (  # noqa: E402
    DEFAULT_ELO,
    elo_at,
    load_elo_ratings,
    normalize_team_name,
)
from soccer_ev_model.features import build_feature_matrix  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic cache fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def synthetic_cache(tmp_path: Path) -> Path:
    """Write a tiny synthetic Elo cache for unit tests."""
    cache = {
        "source": "test",
        "fetched_at": "2026-06-16T00:00:00Z",
        "years_covered": [2020, 2021, 2022],
        "teams": {
            "Argentina": [
                {"date": "2020-01-01", "elo": 2000, "rank": 1},
                {"date": "2021-01-01", "elo": 2050, "rank": 1},
                {"date": "2022-01-01", "elo": 2100, "rank": 1},
            ],
            "Algeria": [
                {"date": "2020-01-01", "elo": 1500, "rank": 50},
                {"date": "2022-01-01", "elo": 1600, "rank": 40},
            ],
        },
    }
    p = tmp_path / "elo.json"
    p.write_text(json.dumps(cache))
    return p


# --------------------------------------------------------------------------- #
# normalize_team_name
# --------------------------------------------------------------------------- #

def test_normalize_team_name_known_aliases():
    """Common variants in our data should map to eloratings canonical names."""
    assert normalize_team_name("United States") == "USA"
    assert normalize_team_name("South Korea") == "Korea Republic"
    assert normalize_team_name("Iran") == "IR Iran"
    assert normalize_team_name("Ivory Coast") == "Côte d'Ivoire"
    assert normalize_team_name("Cape Verde") == "Cabo Verde"
    assert normalize_team_name("DR Congo") == "Congo DR"


def test_normalize_team_name_passthrough():
    """Names already in eloratings form should pass through unchanged."""
    assert normalize_team_name("Argentina") == "Argentina"
    assert normalize_team_name("France") == "France"
    assert normalize_team_name("Brazil") == "Brazil"


# --------------------------------------------------------------------------- #
# load_elo_ratings + elo_at
# --------------------------------------------------------------------------- #

def test_load_elo_ratings_returns_dict_with_dates(synthetic_cache: Path):
    snaps = load_elo_ratings(synthetic_cache)
    assert "Argentina" in snaps
    assert len(snaps["Argentina"]) == 3
    assert snaps["Argentina"][0]["date"] == date(2020, 1, 1)
    assert snaps["Argentina"][0]["elo"] == 2000


def test_elo_at_strict_less_than(synthetic_cache: Path):
    """elo_at must return the most recent snapshot with date < match_date."""
    snaps = load_elo_ratings(synthetic_cache)
    # Match on 2021-06-01 should see Argentina's 2021-01-01 snapshot (2050),
    # NOT the 2022-01-01 one.
    elo, missing = elo_at(snaps, "Argentina", "2021-06-01")
    assert elo == 2050
    assert missing is False


def test_elo_at_returns_default_when_no_prior_snapshot(synthetic_cache: Path):
    """If the team's first snapshot is AFTER the match date, return default."""
    snaps = load_elo_ratings(synthetic_cache)
    elo, missing = elo_at(snaps, "Algeria", "2019-12-31")
    assert elo == DEFAULT_ELO
    assert missing is True


def test_elo_at_returns_default_for_unknown_team(synthetic_cache: Path):
    snaps = load_elo_ratings(synthetic_cache)
    elo, missing = elo_at(snaps, "Wakanda", "2022-06-01")
    assert elo == DEFAULT_ELO
    assert missing is True


# --------------------------------------------------------------------------- #
# build_feature_matrix: label-encoding fix
# --------------------------------------------------------------------------- #

def test_build_feature_matrix_accepts_international_result_codes():
    """Matches with 'home'/'draw'/'away' must be included (not silently dropped)."""
    matches = [
        {"date": "2000-01-01", "home_team": "A", "away_team": "B",
         "home_team_id": 1, "away_team_id": 2,
         "home_goals": 2, "away_goals": 1, "result": "home"},
        {"date": "2001-01-01", "home_team": "A", "away_team": "B",
         "home_team_id": 1, "away_team_id": 2,
         "home_goals": 1, "away_goals": 1, "result": "draw"},
        {"date": "2002-01-01", "home_team": "A", "away_team": "B",
         "home_team_id": 1, "away_team_id": 2,
         "home_goals": 0, "away_goals": 3, "result": "away"},
    ]
    X, y = build_feature_matrix(matches)
    assert len(X) == 3, "all 3 matches should be included (was the bug)"
    assert y.tolist() == ["H", "D", "A"], "results should normalize to H/D/A"


def test_build_feature_matrix_accepts_wc_result_codes():
    """The old 'H'/'D'/'A' codes still work after the fix."""
    matches = [
        {"date": "2010-01-01", "home_team": "A", "away_team": "B",
         "home_team_id": 1, "away_team_id": 2,
         "home_goals": 1, "away_goals": 0, "result": "H"},
        {"date": "2011-01-01", "home_team": "A", "away_team": "B",
         "home_team_id": 1, "away_team_id": 2,
         "home_goals": 1, "away_goals": 1, "result": "D"},
    ]
    X, y = build_feature_matrix(matches)
    assert len(X) == 2
    assert y.tolist() == ["H", "D"]


# --------------------------------------------------------------------------- #
# build_feature_matrix: Elo integration
# --------------------------------------------------------------------------- #

def test_build_feature_matrix_adds_elo_columns_when_snapshots_provided(synthetic_cache: Path):
    """When elo_snapshots is passed, the feature matrix should have elo columns."""
    snaps = load_elo_ratings(synthetic_cache)
    matches = [
        {"date": "2021-06-01", "home_team": "Argentina", "away_team": "Algeria",
         "home_team_id": 1, "away_team_id": 2,
         "home_goals": 2, "away_goals": 1, "result": "H"},
    ]
    X, y = build_feature_matrix(matches, elo_snapshots=snaps)
    # Match is on 2021-06-01 → Argentina's latest prior snapshot is 2021-01-01 (2050).
    # Algeria's latest prior snapshot is 2020-01-01 (1500) - the 2022 one is in the future.
    row = X.iloc[0]
    assert row["home_elo"] == 2050
    assert row["away_elo"] == 1500
    assert row["elo_diff"] == 550
    assert row["home_elo_missing"] == 0
    assert row["away_elo_missing"] == 0


def test_build_feature_matrix_marks_missing_elo(synthetic_cache: Path):
    """Unknown team → default 1500 + missing flag = 1."""
    snaps = load_elo_ratings(synthetic_cache)
    matches = [
        {"date": "2021-06-01", "home_team": "Wakanda", "away_team": "Argentina",
         "home_team_id": 99, "away_team_id": 1,
         "home_goals": 1, "away_goals": 1, "result": "D"},
    ]
    X, _ = build_feature_matrix(matches, elo_snapshots=snaps)
    row = X.iloc[0]
    assert row["home_elo"] == DEFAULT_ELO
    assert row["home_elo_missing"] == 1
    assert row["away_elo"] == 2050
    assert row["away_elo_missing"] == 0


def test_build_feature_matrix_elo_strictly_before_match_date(synthetic_cache: Path):
    """The Elo used must be strictly before the match date (no leakage)."""
    snaps = load_elo_ratings(synthetic_cache)
    # Match on 2022-01-01: the snapshot ON that date is a future result.
    # We must NOT use the 2022-01-01 value (2100); we use 2021-01-01 (2050).
    matches = [
        {"date": "2022-01-01", "home_team": "Argentina", "away_team": "Algeria",
         "home_team_id": 1, "away_team_id": 2,
         "home_goals": 2, "away_goals": 1, "result": "H"},
    ]
    X, _ = build_feature_matrix(matches, elo_snapshots=snaps)
    row = X.iloc[0]
    assert row["home_elo"] == 2050  # 2021 snapshot, not the 2022 one
    assert row["away_elo"] == 1500  # Algeria's 2020 snapshot


def test_build_feature_matrix_no_elo_columns_when_snapshots_none():
    """If elo_snapshots=None (backwards compat), the function still works
    but uses the default-1500-with-missing-flag path. This is a behavior
    change from the old version (which would have crashed trying to fill
    home_elo); we accept it for consistency."""
    matches = [
        {"date": "2010-01-01", "home_team": "A", "away_team": "B",
         "home_team_id": 1, "away_team_id": 2,
         "home_goals": 1, "away_goals": 0, "result": "H"},
    ]
    X, y = build_feature_matrix(matches)  # no elo_snapshots
    row = X.iloc[0]
    assert row["home_elo"] == DEFAULT_ELO
    assert row["home_elo_missing"] == 1
