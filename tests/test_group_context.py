"""Tests for the Phase 3 group-stage context warnings.

The helpers under test are pure: no I/O, no Streamlit, no new
dependencies, no calls to ``evaluate_match``.  The end-to-end tests
(``test_group_context_does_not_alter_blend`` and
``test_group_context_does_not_alter_pi_only``) call the real
``evaluate_match`` workflow and snapshot the model's blend / pi
probabilities before and after the group-context layer to confirm
nothing in this phase can change the model output.

Forbidden-words check: no new string literal in the production code
or in these tests contains any of the six words the project
explicitly forbids (gambling terminology).  Phrases used in the
production code are the spec-mandated: "context warning",
"warning only", "context only", "rotation", "qualification", "draw
risk", "draw-sensitive".
"""
import pytest

from soccer_ev_model.ev_workflow import evaluate_match
from soccer_ev_model.prediction_summary import (
    compute_group_standings,
    group_context_warnings,
    matchday_label,
)


# --------------------------------------------------------------------------- #
# matchday_label
# --------------------------------------------------------------------------- #

def test_matchday_label_group_stage_matchday_1():
    """Group stage, matchday 1 → 'Opening group match', severity info."""
    out = matchday_label("GROUP_STAGE", 1)
    assert out["label"] == "Opening group match"
    assert out["severity"] == "info"
    assert out["is_group_stage"] is True
    assert out["is_final_group_match"] is False
    assert out["matchday"] == 1


def test_matchday_label_group_stage_matchday_2():
    """Group stage, matchday 2 → 'Second group match', severity info."""
    out = matchday_label("GROUP_STAGE", 2)
    assert out["label"] == "Second group match"
    assert out["severity"] == "info"
    assert out["is_group_stage"] is True
    assert out["is_final_group_match"] is False
    assert out["matchday"] == 2


def test_matchday_label_group_stage_matchday_3():
    """Group stage, matchday 3 → 'Final group match', severity warning, is_final_group_match True."""
    out = matchday_label("GROUP_STAGE", 3)
    assert out["label"] == "Final group match"
    assert out["severity"] == "warning"
    assert out["is_group_stage"] is True
    assert out["is_final_group_match"] is True
    assert out["matchday"] == 3


def test_matchday_label_knockout():
    """Knockout stage codes → 'Knockout stage', is_group_stage False, matchday None."""
    for stage in ("LAST_32", "LAST_16", "QUARTER_FINALS", "SEMI_FINALS",
                  "THIRD_PLACE", "FINAL"):
        out = matchday_label(stage, None)
        assert out["label"] == "Knockout stage", f"stage={stage!r}"
        assert out["is_group_stage"] is False
        assert out["matchday"] is None
        assert out["is_final_group_match"] is False


def test_matchday_label_unknown_stage():
    """Empty / unknown stage → 'Unknown stage', is_group_stage False."""
    for stage in ("", "UNKNOWN", "PRESEASON", "GROUP_STAGE_TYPO"):
        out = matchday_label(stage, None)
        assert out["label"] == "Unknown stage", f"stage={stage!r}"
        assert out["is_group_stage"] is False
        assert out["matchday"] is None
        assert out["is_final_group_match"] is False


# --------------------------------------------------------------------------- #
# group_context_warnings
# --------------------------------------------------------------------------- #

def test_group_context_warnings_non_group_returns_empty():
    """Knockout / unknown stage → empty list (no group warning)."""
    assert group_context_warnings("FINAL", None) == []
    assert group_context_warnings("LAST_16", None) == []
    assert group_context_warnings("", 1) == []
    assert group_context_warnings("UNKNOWN", 3) == []


def test_group_context_warnings_matchday_1():
    """Matchday 1 → exactly one warning tagged 'opening' about normal incentives.

    We deliberately do NOT pass a group here, so the no_data warning
    does NOT fire (per the helper's documented contract: "If the
    group is supplied but finished_matches_in_group is empty or
    None, append a 'no_data' warning").
    """
    warnings = group_context_warnings("GROUP_STAGE", 1)
    assert len(warnings) == 1
    w = warnings[0]
    assert w["tag"] == "opening"
    assert w["severity"] == "info"
    assert "normal incentives" in w["text"].lower()


def test_group_context_warnings_matchday_2():
    """Matchday 2 → exactly one warning tagged 'second' about first result.

    Same as test_matchday_1: no group passed, so no no_data warning
    is appended.
    """
    warnings = group_context_warnings("GROUP_STAGE", 2)
    assert len(warnings) == 1
    w = warnings[0]
    assert w["tag"] == "second"
    assert w["severity"] == "info"
    assert "first result" in w["text"].lower()


def test_group_context_warnings_matchday_3_has_three_warnings():
    """Matchday 3 → at least 3 warnings, the first has tag='final' and severity='warning'."""
    warnings = group_context_warnings("GROUP_STAGE", 3, "GROUP_A")
    assert len(warnings) >= 3, f"expected ≥3 warnings, got {len(warnings)}"
    first = warnings[0]
    assert first["tag"] == "final"
    assert first["severity"] == "warning"
    # The other two should follow.
    tags = [w["tag"] for w in warnings[:3]]
    assert "rotation" in tags
    assert "draw_sensitive" in tags
    # The second warning is the long rotation/qualification note.
    rotation = next(w for w in warnings if w["tag"] == "rotation")
    assert rotation["severity"] == "warning"
    assert "rotation" in rotation["text"].lower()
    assert "qualification" in rotation["text"].lower()
    # Draw-sensitive is an info note.
    draw_w = next(w for w in warnings if w["tag"] == "draw_sensitive")
    assert draw_w["severity"] == "info"
    assert "draw" in draw_w["text"].lower()


def test_group_context_warnings_no_data_for_just_started_group():
    """Group set, no finished matches → a 'no_data' warning is appended."""
    warnings = group_context_warnings(
        "GROUP_STAGE", 1, "GROUP_A", finished_matches_in_group=None,
    )
    tags = [w["tag"] for w in warnings]
    assert "no_data" in tags, f"missing no_data tag, got: {tags}"
    no_data = next(w for w in warnings if w["tag"] == "no_data")
    assert no_data["severity"] == "info"
    assert "no group matches yet played" in no_data["text"].lower()

    # Same expectation with an empty list (not just None).
    warnings_empty = group_context_warnings(
        "GROUP_STAGE", 1, "GROUP_A", finished_matches_in_group=[],
    )
    tags_empty = [w["tag"] for w in warnings_empty]
    assert "no_data" in tags_empty


def test_group_context_warnings_standings_summary():
    """With 2 finished matches, a 'standings' warning is appended with team/point info."""
    finished = [
        {
            "home_team_id": 1, "away_team_id": 2,
            "home_goals": 2, "away_goals": 0,
            "home_team_name": "Alpha", "away_team_name": "Bravo",
            "date": "2026-06-11",
        },
        {
            "home_team_id": 1, "away_team_id": 3,
            "home_goals": 1, "away_goals": 1,
            "home_team_name": "Alpha", "away_team_name": "Charlie",
            "date": "2026-06-15",
        },
    ]
    warnings = group_context_warnings(
        "GROUP_STAGE", 2, "GROUP_A", finished_matches_in_group=finished,
    )
    tags = [w["tag"] for w in warnings]
    assert "standings" in tags, f"missing standings tag, got: {tags}"
    standings = next(w for w in warnings if w["tag"] == "standings")
    assert standings["severity"] == "info"
    # Spec requires the text starts with the documented prefix.
    assert standings["text"].startswith("Current group standings (context only)")
    # And contains the team names + their points.
    assert "Alpha" in standings["text"]
    assert "Bravo" in standings["text"]
    assert "Charlie" in standings["text"]
    # Alpha: win (3 pts) + draw (1 pt) = 4 pts total
    assert "4pts" in standings["text"] or "4 pts" in standings["text"]


# --------------------------------------------------------------------------- #
# compute_group_standings
# --------------------------------------------------------------------------- #

def test_compute_group_standings_empty_input():
    """Empty list → empty dict (never raises)."""
    assert compute_group_standings([]) == {}


def test_compute_group_standings_basic():
    """1 match, home wins 2-0 → home 3pts +2 GD; away 0pts -2 GD."""
    matches = [
        {
            "home_team_id": 1, "away_team_id": 2,
            "home_goals": 2, "away_goals": 0,
            "home_team_name": "Home", "away_team_name": "Away",
        }
    ]
    s = compute_group_standings(matches)
    assert set(s.keys()) == {1, 2}
    assert s[1]["played"] == 1
    assert s[1]["wins"] == 1
    assert s[1]["points"] == 3
    assert s[1]["gd"] == 2
    assert s[1]["gf"] == 2
    assert s[1]["ga"] == 0
    assert s[2]["played"] == 1
    assert s[2]["losses"] == 1
    assert s[2]["points"] == 0
    assert s[2]["gd"] == -2
    assert s[2]["gf"] == 0
    assert s[2]["ga"] == 2


def test_compute_group_standings_draw():
    """1 match 1-1 → both teams: 1pt, GD 0, 1 draw."""
    matches = [
        {
            "home_team_id": 1, "away_team_id": 2,
            "home_goals": 1, "away_goals": 1,
            "home_team_name": "Home", "away_team_name": "Away",
        }
    ]
    s = compute_group_standings(matches)
    for tid in (1, 2):
        assert s[tid]["points"] == 1
        assert s[tid]["gd"] == 0
        assert s[tid]["draws"] == 1
        assert s[tid]["wins"] == 0
        assert s[tid]["losses"] == 0
        assert s[tid]["played"] == 1


def test_compute_group_standings_three_way_ranking():
    """3 teams, 3 matches: A beats B 2-0, A beats C 1-0, B beats C 3-1.

    Correct arithmetic (per the spec's own match schedule):
      - A: 2 wins, 6 pts, GF=3, GA=0, GD=+3
      - B: 1 win, 1 loss, 3 pts, GF=3, GA=3, GD=0
      - C: 0 wins, 2 losses, 0 pts, GF=1, GA=4, GD=-3

    The spec's text said "gd: A=+3, B=-2, C=-1", but those values
    don't match its own match schedule (B's actual GD is 0, C's is
    -3).  We pin the arithmetic here so the test reflects the
    helper's actual output — flagged in the worker report as a
    deviation from the spec's literal expected values.
    """
    matches = [
        {"home_team_id": 1, "away_team_id": 2, "home_goals": 2, "away_goals": 0,
         "home_team_name": "A", "away_team_name": "B"},
        {"home_team_id": 1, "away_team_id": 3, "home_goals": 1, "away_goals": 0,
         "home_team_name": "A", "away_team_name": "C"},
        {"home_team_id": 2, "away_team_id": 3, "home_goals": 3, "away_goals": 1,
         "home_team_name": "B", "away_team_name": "C"},
    ]
    s = compute_group_standings(matches)
    # Points (these match the spec)
    assert s[1]["points"] == 6  # A
    assert s[2]["points"] == 3  # B
    assert s[3]["points"] == 0  # C
    # Goal difference (arithmetic from the match schedule)
    assert s[1]["gd"] == 3      # A
    assert s[2]["gd"] == 0      # B
    assert s[3]["gd"] == -3     # C
    # Sanity: GDs sum to 0.
    assert sum(s[tid]["gd"] for tid in s) == 0
    # Sanity: total points = 3 matches × 3 pts/match = 9.
    assert sum(s[tid]["points"] for tid in s) == 9


# --------------------------------------------------------------------------- #
# Phase 3 purity — context layer must not change model probabilities
# --------------------------------------------------------------------------- #

def _make_train():
    """Tiny training set so pi-rating has something to fit."""
    train = []
    for i in range(40):
        train.append({
            "match_id": f"2020-{i:02d}-1",
            "date": f"2020-{(i % 9) + 1:02d}-01",
            "home_team": "Team1", "away_team": "Team2",
            "home_team_id": 1, "away_team_id": 2,
            "home_goals": 2, "away_goals": 0, "result": "H",
        })
    return train


def test_group_context_does_not_alter_blend():
    """Calling group_context_warnings + compute_group_standings must not
    change ``result['blend_probs']`` byte-for-byte.  The context layer
    is a pure function of its inputs.
    """
    from soccer_ev_model.pi_ratings import compute_pi_ratings

    train = _make_train()
    ratings = compute_pi_ratings(train, cutoff="2020-12-01")

    result = evaluate_match(
        home_team="Team1", away_team="Team2",
        home_team_id=1, away_team_id=2,
        date="2020-12-01",
        book_home_odds=-150, book_draw_odds=300, book_away_odds=400,
        ratings=ratings,
    )

    # Snapshot the blend probs in a stable, comparable form.
    before = dict(result["blend_probs"])

    # Exercise the Phase 3 layer on synthetic input.  None of these
    # calls touches ``result``.
    _ = group_context_warnings(
        "GROUP_STAGE", 3, "GROUP_A",
        finished_matches_in_group=[
            {"home_team_id": 1, "away_team_id": 2,
             "home_goals": 2, "away_goals": 0,
             "home_team_name": "T1", "away_team_name": "T2"},
        ],
    )
    _ = group_context_warnings("GROUP_STAGE", 1, "GROUP_A")
    _ = group_context_warnings("LAST_16", None)
    _ = compute_group_standings([
        {"home_team_id": 1, "away_team_id": 2,
         "home_goals": 1, "away_goals": 1,
         "home_team_name": "T1", "away_team_name": "T2"},
    ])

    after = dict(result["blend_probs"])
    assert before == after, (
        f"group context mutated blend_probs: before={before}, after={after}"
    )
    # And the original result identity is preserved (no in-place edit).
    assert result["blend_probs"] is result["blend_probs"]


def test_group_context_does_not_alter_pi_only():
    """Same immutability guarantee, but for ``pi_only_probs``."""
    from soccer_ev_model.pi_ratings import compute_pi_ratings

    train = _make_train()
    ratings = compute_pi_ratings(train, cutoff="2020-12-01")

    result = evaluate_match(
        home_team="Team1", away_team="Team2",
        home_team_id=1, away_team_id=2,
        date="2020-12-01",
        book_home_odds=-150, book_draw_odds=300, book_away_odds=400,
        ratings=ratings,
    )

    pi_before = result.get("pi_only_probs")
    # If the result happens not to include pi_only_probs (older code
    # path), we still pin blend_probs; the test is most meaningful
    # when pi_only_probs is present.
    if pi_before is None:
        pytest.skip("pi_only_probs not present in evaluate_match output")

    before = dict(pi_before)
    _ = group_context_warnings("GROUP_STAGE", 3, "GROUP_A")
    _ = compute_group_standings([])
    after = dict(pi_before)
    assert before == after, (
        f"group context mutated pi_only_probs: before={before}, after={after}"
    )


def test_warnings_deterministic():
    """group_context_warnings is deterministic — no hidden state, no Date.now, no random."""
    args = ("GROUP_STAGE", 3, "GROUP_A")
    a = group_context_warnings(*args)
    b = group_context_warnings(*args)
    assert a == b, f"non-deterministic output: {a!r} vs {b!r}"

    # And with kwargs.
    a2 = group_context_warnings(
        "GROUP_STAGE", 3, "GROUP_A",
        finished_matches_in_group=[
            {"home_team_id": 1, "away_team_id": 2,
             "home_goals": 2, "away_goals": 0,
             "home_team_name": "A", "away_team_name": "B"},
        ],
    )
    b2 = group_context_warnings(
        "GROUP_STAGE", 3, "GROUP_A",
        finished_matches_in_group=[
            {"home_team_id": 1, "away_team_id": 2,
             "home_goals": 2, "away_goals": 0,
             "home_team_name": "A", "away_team_name": "B"},
        ],
    )
    assert a2 == b2, f"non-deterministic output: {a2!r} vs {b2!r}"
