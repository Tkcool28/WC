"""Tests for the WC data fetcher's parse logic.

We test the parser directly with a small sample of the football-data.org API
response shape. The HTTP path is covered by the api_client tests; this module
focuses on the bit that actually has logic: turning the API's nested JSON into
a flat list of clean match records.
"""

import pytest
from soccer_ev_model.fetch_data import (
    parse_matches,
    parse_openfootball_matches,
    normalize_team_name,
)


SAMPLE_API_RESPONSE = {
    "matches": [
        {
            "id": 1,
            "utcDate": "2022-11-20T15:00:00Z",
            "status": "FINISHED",
            "matchday": 1,
            "stage": "GROUP_STAGE",
            "group": "GROUP_A",
            "homeTeam": {"id": 759, "name": "Germany"},
            "awayTeam": {"id": 760, "name": "Spain"},
            "score": {
                "winner": "HOME_TEAM",
                "duration": "REGULAR",
                "fullTime": {"home": 2, "away": 1},
                "halfTime": {"home": 1, "away": 0},
            },
        },
        {
            "id": 2,
            "utcDate": "2022-11-20T18:00:00Z",
            "status": "FINISHED",
            "matchday": 1,
            "stage": "GROUP_STAGE",
            "group": "GROUP_B",
            "homeTeam": {"id": 762, "name": "Argentina"},
            "awayTeam": {"id": 763, "name": "Brazil"},
            "score": {
                "winner": "DRAW",
                "duration": "REGULAR",
                "fullTime": {"home": 1, "away": 1},
                "halfTime": {"home": 0, "away": 0},
            },
        },
        {
            "id": 3,
            "utcDate": "2026-06-16T19:00:00Z",
            "status": "TIMED",
            "matchday": 1,
            "stage": "GROUP_STAGE",
            "group": "GROUP_I",
            "homeTeam": {"id": 773, "name": "France"},
            "awayTeam": {"id": 841, "name": "Senegal"},
            "score": {
                "winner": None,
                "duration": "REGULAR",
                "fullTime": {"home": None, "away": None},
                "halfTime": {"home": None, "away": None},
            },
        },
    ]
}


def test_parse_matches_returns_a_list():
    """Happy path: a well-formed API response parses to a list of 3 records."""
    matches = parse_matches(SAMPLE_API_RESPONSE)
    assert isinstance(matches, list)
    assert len(matches) == 3


def test_parse_matches_includes_core_fields():
    """Each record has date, teams, ids, stage, group, matchday, and final score."""
    matches = parse_matches(SAMPLE_API_RESPONSE)
    m = matches[0]
    for field in [
        "id", "date", "home_team_id", "home_team_name",
        "away_team_id", "away_team_name",
        "stage", "group", "matchday", "status",
        "home_goals", "away_goals", "result",
    ]:
        assert field in m, f"Missing field: {field}"


def test_parse_matches_extracts_team_ids_and_names():
    """Team ids are integers, names are strings, both home and away are present."""
    matches = parse_matches(SAMPLE_API_RESPONSE)
    m = matches[0]
    assert m["home_team_id"] == 759
    assert m["home_team_name"] == "Germany"
    assert m["away_team_id"] == 760
    assert m["away_team_name"] == "Spain"


def test_parse_matches_home_win_result():
    """A match with winner=HOME_TEAM should be encoded as 'H'."""
    matches = parse_matches(SAMPLE_API_RESPONSE)
    assert matches[0]["result"] == "H"
    assert matches[0]["home_goals"] == 2
    assert matches[0]["away_goals"] == 1


def test_parse_matches_draw_result():
    """A match with winner=DRAW should be encoded as 'D'."""
    matches = parse_matches(SAMPLE_API_RESPONSE)
    assert matches[1]["result"] == "D"
    assert matches[1]["home_goals"] == 1
    assert matches[1]["away_goals"] == 1


def test_parse_matches_unplayed_match_has_null_score():
    """A TIMED match with no score yet should have None for goals and result."""
    matches = parse_matches(SAMPLE_API_RESPONSE)
    m = matches[2]
    assert m["status"] == "TIMED"
    assert m["home_goals"] is None
    assert m["away_goals"] is None
    assert m["result"] is None


def test_parse_matches_date_is_iso_string():
    """Date stays as ISO 8601 string for easy sorting and parsing later."""
    matches = parse_matches(SAMPLE_API_RESPONSE)
    assert matches[0]["date"] == "2022-11-20T15:00:00Z"


def test_parse_matches_empty_response_returns_empty_list():
    """If the API returns 0 matches, we get an empty list, not an error."""
    matches = parse_matches({"matches": []})
    assert matches == []


def test_parse_matches_handles_missing_score_field():
    """A match with no 'score' key at all (some endpoint variants) gets nulls."""
    api_response = {
        "matches": [
            {
                "id": 99,
                "utcDate": "2022-11-20T15:00:00Z",
                "status": "TIMED",
                "stage": "GROUP_STAGE",
                "group": "GROUP_A",
                "homeTeam": {"id": 1, "name": "TeamA"},
                "awayTeam": {"id": 2, "name": "TeamB"},
            }
        ]
    }
    matches = parse_matches(api_response)
    assert matches[0]["home_goals"] is None
    assert matches[0]["away_goals"] is None
    assert matches[0]["result"] is None


def test_parse_matches_preserves_order():
    """The output order matches the input order (which is the API's order)."""
    matches = parse_matches(SAMPLE_API_RESPONSE)
    assert [m["id"] for m in matches] == [1, 2, 3]


# ---- openfootball parser ----
#
# The openfootball/worldcup.json repo on GitHub has World Cup data 1930-2026
# as static JSON files. Different shape from football-data.org:
#   {"name": "World Cup 2022", "matches": [...]}
#   each match: {round, date, time, team1, team2, score: {ft: [home, away], ht: [...]}, goals1, goals2, group, ...}
# No team IDs (just names), so we synthesize an id from a normalized name.


SAMPLE_OPENFOOTBALL = {
    "name": "World Cup 2022",
    "matches": [
        {
            "round": "Matchday 1",
            "date": "2022-11-20",
            "time": "19:00",
            "team1": "Qatar",
            "team2": "Ecuador",
            "score": {"ft": [0, 2], "ht": [0, 2]},
            "group": "Group A",
        },
        {
            "round": "Matchday 1",
            "date": "2022-11-21",
            "time": "16:00",
            "team1": "England",
            "team2": "Iran",
            "score": {"ft": [6, 2], "ht": [3, 0]},
            "group": "Group B",
        },
        {
            "round": "Final",
            "date": "2022-12-18",
            "time": "18:00",
            "team1": "Argentina",
            "team2": "France",
            "score": {"ft": [3, 3], "ht": [2, 0]},
            "group": None,
        },
    ],
}


def test_parse_openfootball_returns_a_list():
    matches = parse_openfootball_matches(SAMPLE_OPENFOOTBALL)
    assert isinstance(matches, list)
    assert len(matches) == 3


def test_parse_openfootball_emits_same_record_shape_as_football_data():
    """The two parsers must produce records with overlapping field names so
    downstream code (pi-ratings, features) can use either source transparently."""
    fd_matches = parse_matches(SAMPLE_API_RESPONSE)
    of_matches = parse_openfootball_matches(SAMPLE_OPENFOOTBALL)
    fd_fields = set(fd_matches[0].keys())
    of_fields = set(of_matches[0].keys())
    # The fields pi-ratings need must be present in BOTH sources
    required = {
        "date", "home_team_name", "away_team_name",
        "home_goals", "away_goals", "result",
    }
    missing_in_of = required - of_fields
    missing_in_fd = required - fd_fields
    assert not missing_in_of, f"openfootball parser missing: {missing_in_of}"
    assert not missing_in_fd, f"football-data parser missing: {missing_in_fd}"


def test_parse_openfootball_home_win():
    """England 6, Iran 2 -> result 'H'."""
    matches = parse_openfootball_matches(SAMPLE_OPENFOOTBALL)
    assert matches[1]["result"] == "H"
    assert matches[1]["home_goals"] == 6
    assert matches[1]["away_goals"] == 2


def test_parse_openfootball_draw():
    """Argentina 3, France 3 (final) -> result 'D'."""
    matches = parse_openfootball_matches(SAMPLE_OPENFOOTBALL)
    assert matches[2]["result"] == "D"
    assert matches[2]["home_goals"] == 3
    assert matches[2]["away_goals"] == 3


def test_parse_openfootball_handles_unplayed_match():
    """If 'ft' is missing or null, the match is unplayed. Goals and result are None."""
    sample = {
        "name": "World Cup 2026",
        "matches": [
            {
                "round": "Matchday 1",
                "date": "2026-06-16",
                "time": "19:00",
                "team1": "France",
                "team2": "Senegal",
                "score": {},
                "group": "Group I",
            }
        ]
    }
    matches = parse_openfootball_matches(sample)
    assert matches[0]["home_goals"] is None
    assert matches[0]["away_goals"] is None
    assert matches[0]["result"] is None


def test_parse_openfootball_date_uses_iso_format():
    """Date is stored as YYYY-MM-DDTHH:MM:SSZ so it sorts and parses cleanly."""
    matches = parse_openfootball_matches(SAMPLE_OPENFOOTBALL)
    assert matches[0]["date"] == "2022-11-20T19:00:00Z"


def test_parse_openfootball_synthesizes_team_ids_from_names():
    """openfootball has no team IDs, so we hash the normalized name to an int.
    Same team name must always produce the same id (across calls and years)."""
    matches_a = parse_openfootball_matches(SAMPLE_OPENFOOTBALL)
    matches_b = parse_openfootball_matches(SAMPLE_OPENFOOTBALL)
    assert matches_a[0]["home_team_id"] == matches_b[0]["home_team_id"]
    # And the id is a stable integer
    assert isinstance(matches_a[0]["home_team_id"], int)


def test_parse_openfootball_team_ids_differ_for_different_teams():
    """England and Iran should have different ids even though they appear in the same match."""
    matches = parse_openfootball_matches(SAMPLE_OPENFOOTBALL)
    assert matches[1]["home_team_id"] != matches[1]["away_team_id"]


def test_parse_openfootball_empty_response():
    matches = parse_openfootball_matches({"name": "World Cup 2099", "matches": []})
    assert matches == []


# ---- team name normalization ----
#
# football-data.org uses names like "Bosnia-Herzegovina".
# openfootball uses names like "Bosnia and Herzegovina" or "Bosnia & Herzegovina".
# We need a stable mapping so a team is one team across both sources.


def test_normalize_team_name_lowercase():
    assert normalize_team_name("France") == "france"
    assert normalize_team_name("FRANCE") == "france"


def test_normalize_team_name_strips_dashes_and_ampersand():
    """Different sources spell the same country differently. We standardize."""
    assert normalize_team_name("Bosnia-Herzegovina") == "bosniaherzegovina"
    assert normalize_team_name("Bosnia & Herzegovina") == "bosniaherzegovina"
    assert normalize_team_name("Bosnia and Herzegovina") == "bosniaherzegovina"
    assert normalize_team_name("Bosnia-Herzegovina") == "bosniaherzegovina"


def test_normalize_team_name_strips_whitespace():
    assert normalize_team_name("  France  ") == "france"
    assert normalize_team_name("South Korea") == "southkorea"


def test_normalize_team_name_known_aliases_is_out_of_scope():
    """Aliasing (South Korea <-> Korea Republic) is intentionally out of scope
    for normalize_team_name. The function only does string-level cleanup.
    Aliases are a separate concern: a TEAM_ALIASES dict applied after
    normalization. We document this with a test that pins the current
    behavior, so it doesn't drift silently.

    If/when we add aliasing, this test will start failing and prompt
    the change to be intentional.
    """
    assert normalize_team_name("South Korea") != normalize_team_name("Korea Republic")
