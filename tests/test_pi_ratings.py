"""Tests for pi-rating computation.

Pi-ratings are a dynamic team-strength system for football (Constantinou &
Fenton 2013) that improves on Elo by using goal-margin information. Each
team has separate offense and defense ratings that update after every match.

The key property we test HARD is the **cutoff date**: a rating computed for
a target match must use only matches strictly before that target. Any leak
here destroys the model's value.

References:
- Constantinou & Fenton, "Determining the level of ability of football teams
  by dynamic ratings based on the relative discrepancies in scores between
  adversaries" (2013)
- Razali et al. 2022 (CatBoost + pi-ratings, top performer in 2023 SPC)
"""

import pytest
from soccer_ev_model.pi_ratings import (
    compute_pi_ratings,
    compute_pi_ratings_walk_forward,
    get_team_experience,
    pi_diff_features,
    LEARNING_RATE,
)


# ---- empty / neutral state ----


def test_empty_matches_returns_empty_ratings():
    """No matches -> no ratings. Empty dict, not an error."""
    ratings = compute_pi_ratings(matches=[])
    assert ratings == {}


def test_team_with_no_matches_is_not_in_ratings():
    """A team that has never played doesn't get a rating. This prevents
    downstream code from reading a misleading 0.0 for a new team."""
    matches = [
        {"date": "2022-01-01", "home_team_id": 1, "away_team_id": 2,
         "home_goals": 2, "away_goals": 0, "result": "H"}
    ]
    ratings = compute_pi_ratings(matches)
    # Only teams 1 and 2 should have ratings
    assert set(ratings.keys()) == {1, 2}
    # Team 3 (never played) should NOT be present
    assert 3 not in ratings


# ---- single match effects ----


def test_home_win_increases_home_offense():
    """A team that wins at home should have its offense rating go UP (positive)."""
    matches = [
        {"date": "2022-01-01", "home_team_id": 1, "away_team_id": 2,
         "home_goals": 2, "away_goals": 0, "result": "H"}
    ]
    ratings = compute_pi_ratings(matches)
    assert ratings[1]["offense"] > 0
    # Defense for a team that conceded 0 should also be > 0 (good defense)
    assert ratings[1]["defense"] > 0


def test_home_win_decreases_away_defense():
    """A team that loses away (conceded 2 goals) should have its defense go DOWN."""
    matches = [
        {"date": "2022-01-01", "home_team_id": 1, "away_team_id": 2,
         "home_goals": 2, "away_goals": 0, "result": "H"}
    ]
    ratings = compute_pi_ratings(matches)
    # Away team conceded 2, so their defense rating drops
    assert ratings[2]["defense"] < 0
    # Away team's offense didn't score, so offense drops too
    assert ratings[2]["offense"] < 0


def test_draw_keeps_ratings_near_zero():
    """A 0-0 or 1-1 draw should produce small (near zero) rating changes
    because both teams performed as expected against similarly-rated opponents."""
    matches = [
        {"date": "2022-01-01", "home_team_id": 1, "away_team_id": 2,
         "home_goals": 1, "away_goals": 1, "result": "D"}
    ]
    ratings = compute_pi_ratings(matches)
    # Small absolute values - not strict but should be small
    assert abs(ratings[1]["offense"]) < 0.5
    assert abs(ratings[2]["offense"]) < 0.5


def test_offense_and_defense_are_separate():
    """A team that scores a lot but concedes a lot should have
    high offense AND low defense (they are independent dimensions)."""
    matches = [
        {"date": "2022-01-01", "home_team_id": 1, "away_team_id": 2,
         "home_goals": 5, "away_goals": 4, "result": "H"}
    ]
    ratings = compute_pi_ratings(matches)
    # Team 1 scored 5 (offense up) but conceded 4 (defense down)
    assert ratings[1]["offense"] > 0
    assert ratings[1]["defense"] < 0


# ---- CRITICAL: cutoff date is leak-safe ----


def test_cutoff_excludes_later_matches():
    """Matches AFTER the cutoff must NOT affect the rating.

    This is the single most important test in the suite. If this fails,
    the model has future-knowledge leakage and backtest results are meaningless.
    """
    matches = [
        {"date": "2022-01-01", "home_team_id": 1, "away_team_id": 2,
         "home_goals": 5, "away_goals": 0, "result": "H"},
        # This match is AFTER the cutoff. Team 1 loses 0-5.
        # If the cutoff is honored, this match has zero effect on the rating.
        {"date": "2022-12-01", "home_team_id": 1, "away_team_id": 2,
         "home_goals": 0, "away_goals": 5, "result": "A"},
    ]
    r_before = compute_pi_ratings(matches, cutoff="2022-06-01")
    r_all = compute_pi_ratings(matches)  # no cutoff, uses both
    # The team that won 5-0 first but lost 0-5 second has LOWER offense
    # when both matches are included.
    assert r_before[1]["offense"] > r_all[1]["offense"], (
        f"Cutoff failed: before={r_before[1]['offense']}, "
        f"all={r_all[1]['offense']}. "
        f"Later match is leaking into the rating."
    )


def test_cutoff_inclusive_only_of_strictly_before():
    """A match on the cutoff date itself is INCLUDED (it's not the future)."""
    matches = [
        {"date": "2022-06-01", "home_team_id": 1, "away_team_id": 2,
         "home_goals": 3, "away_goals": 0, "result": "H"}
    ]
    r_with = compute_pi_ratings(matches, cutoff="2022-06-01")
    r_without = compute_pi_ratings(matches, cutoff="2022-05-31")
    # With cutoff=06-01: the match counts
    assert r_with[1]["offense"] > 0
    # With cutoff=05-31: the match is excluded
    assert 1 not in r_without or abs(r_without[1]["offense"]) < 1e-9


# ---- matchup features ----


def test_pi_diff_features_for_strong_vs_weak():
    """A strong home team vs a weak away team should have positive diffs."""
    ratings = {
        1: {"offense": 1.5, "defense": 1.0, "matches_played": 5},
        2: {"offense": -0.5, "defense": -1.0, "matches_played": 5},
    }
    feats = pi_diff_features(home_id=1, away_id=2, ratings=ratings)
    # Home is much stronger, so all diffs positive
    assert feats["pi_off_diff"] == pytest.approx(2.0, abs=1e-9)
    assert feats["pi_def_diff"] == pytest.approx(2.0, abs=1e-9)
    assert feats["pi_matchup"] > 0
    # The individual ratings are also exposed
    assert feats["pi_home_off"] == 1.5
    assert feats["pi_away_off"] == -0.5


def test_pi_diff_features_for_close_matchup():
    """Two evenly-matched teams should have near-zero diffs."""
    ratings = {
        1: {"offense": 0.3, "defense": 0.2, "matches_played": 5},
        2: {"offense": 0.3, "defense": 0.2, "matches_played": 5},
    }
    feats = pi_diff_features(home_id=1, away_id=2, ratings=ratings)
    assert abs(feats["pi_off_diff"]) < 1e-9
    assert abs(feats["pi_def_diff"]) < 1e-9


def test_pi_diff_features_handles_missing_team():
    """If a team has no rating (e.g., debut match), use 0.0 as a neutral default."""
    ratings = {
        1: {"offense": 1.0, "defense": 0.5, "matches_played": 3},
        # team 2 not in ratings - never played before
    }
    feats = pi_diff_features(home_id=1, away_id=2, ratings=ratings)
    # Away defaults to 0.0, so diff is positive
    assert feats["pi_off_diff"] == pytest.approx(1.0, abs=1e-9)
    assert feats["pi_away_off"] == 0.0


# ---- matches_played tracking ----


def test_matches_played_is_counted():
    """The matches_played counter should reflect how many matches each team had."""
    matches = [
        {"date": "2022-01-01", "home_team_id": 1, "away_team_id": 2, "home_goals": 1, "away_goals": 0, "result": "H"},
        {"date": "2022-02-01", "home_team_id": 1, "away_team_id": 3, "home_goals": 1, "away_goals": 0, "result": "H"},
        {"date": "2022-03-01", "home_team_id": 2, "away_team_id": 3, "home_goals": 1, "away_goals": 0, "result": "H"},
    ]
    ratings = compute_pi_ratings(matches)
    # Team 1 played 2 matches, teams 2 and 3 each played 2
    assert ratings[1]["matches_played"] == 2
    assert ratings[2]["matches_played"] == 2
    assert ratings[3]["matches_played"] == 2


# ---- sanity: learning rate is exported and reasonable ----


def test_learning_rate_is_in_safe_range():
    """LR is a model hyperparameter. Pin it so we don't accidentally tune it
    to something crazy (like 0.5 = too jumpy, or 0.001 = doesn't move).

    Note: the empirically-best LR (0.005) is BELOW this old range.
    The original 0.07-0.15 range was tuned for small-sample use cases
    (256 WC matches); 0.005 is correct for the 25k+ intl dataset. We
    lower the lower bound to accommodate.
    """
    assert 0.001 <= LEARNING_RATE <= 0.15


# ---- walk-forward batch: matches the cutoff-based path for first match ----


def _wf_match(date, home_id, away_id, hg, ag, result="H"):
    return {
        "match_id": f"{date}_{home_id}_{away_id}",
        "date": date,
        "home_team_id": home_id,
        "away_team_id": away_id,
        "home_goals": hg,
        "away_goals": ag,
        "result": result,
    }


def test_walk_forward_first_match_equals_cutoff_path():
    """For the first test match, walk-forward (no prior test consumption)
    must give the same ratings as the cutoff-based path with cutoff=match.date.
    """
    train = [_wf_match("2020-01-01", 1, 2, 2, 0),
             _wf_match("2020-02-01", 1, 3, 1, 1, "D")]
    test = [_wf_match("2020-06-01", 1, 2, 0, 0)]
    snaps = compute_pi_ratings_walk_forward(train, test, consume_test_results=True)
    assert len(snaps) == 1
    m, ratings = snaps[0]
    expected = compute_pi_ratings(train, cutoff=m["date"])
    assert set(ratings.keys()) == set(expected.keys())
    for tid in ratings:
        assert ratings[tid]["offense"] == pytest.approx(expected[tid]["offense"], abs=1e-9)
        assert ratings[tid]["defense"] == pytest.approx(expected[tid]["defense"], abs=1e-9)


def test_walk_forward_consume_test_results_updates_ratings():
    """With consume_test_results=True, the second test match sees the first
    test match's outcome already folded into ratings. With it False, both
    test matches see the same end-of-training snapshot.
    """
    train = [_wf_match("2020-01-01", 1, 2, 1, 0)]
    test = [_wf_match("2020-06-01", 1, 2, 0, 0),
            _wf_match("2020-07-01", 1, 2, 0, 0)]
    snap_consume = compute_pi_ratings_walk_forward(train, test, consume_test_results=True)
    snap_freeze = compute_pi_ratings_walk_forward(train, test, consume_test_results=False)
    # Frozen path: both test matches see the same snapshot
    assert snap_freeze[0][1] == snap_freeze[1][1]
    # Consume path: the second test sees the result of the first already applied
    # (so team 1's offense will be lower for the second snapshot because the
    # first test match was a 0-0 draw which is below the expected score from
    # team 1's strong 1-0 win in training)
    assert snap_consume[0][1] != snap_consume[1][1]


def test_walk_forward_is_leak_free():
    """A test match must NEVER see its own result in its ratings snapshot.
    This is the leak-protection guarantee of the walk-forward batch path.
    """
    train = [_wf_match("2020-01-01", 1, 2, 0, 0)]
    # Test match has team 1 winning 5-0
    test = [_wf_match("2020-06-01", 1, 2, 5, 0)]
    snaps = compute_pi_ratings_walk_forward(train, test, consume_test_results=True)
    _, ratings = snaps[0]
    # Team 1's offense would be very high if the 5-0 win were leaked.
    # With the only training match being 0-0 (no signal), team 1's offense
    # should be near 0.
    assert abs(ratings[1]["offense"]) < 0.1
    assert abs(ratings[2]["offense"]) < 0.1


def test_walk_forward_empty_train_works():
    """With no training matches, all test snapshots should be empty dicts."""
    test = [_wf_match("2020-06-01", 1, 2, 0, 0)]
    snaps = compute_pi_ratings_walk_forward([], test)
    assert len(snaps) == 1
    _, ratings = snaps[0]
    assert ratings == {}


# ---- get_team_experience ----


def test_get_team_experience_returns_matches_played_and_ratings():
    """For a team in the ratings dict, return their matches_played and rating values."""
    ratings = {
        1: {"offense": 0.42, "defense": 0.31, "matches_played": 7},
        2: {"offense": -0.10, "defense": 0.05, "matches_played": 4},
    }
    exp = get_team_experience(ratings, 1)
    assert exp == {
        "matches_played": 7,
        "offense": pytest.approx(0.42, abs=1e-9),
        "defense": pytest.approx(0.31, abs=1e-9),
    }


def test_get_team_experience_returns_zeros_for_unknown_team():
    """If the team is not in ratings, return zeros (not raise)."""
    ratings = {
        1: {"offense": 0.5, "defense": 0.2, "matches_played": 10},
    }
    exp = get_team_experience(ratings, 999)
    assert exp == {"matches_played": 0, "offense": 0.0, "defense": 0.0}


def test_get_team_experience_with_empty_ratings():
    """With an empty ratings dict, every lookup returns zeros."""
    exp = get_team_experience({}, 42)
    assert exp == {"matches_played": 0, "offense": 0.0, "defense": 0.0}


def test_get_team_experience_used_in_walk_forward_snapshot():
    """The function must work on a snapshot from walk_forward (not just compute_pi_ratings)."""
    train = [_wf_match("2020-01-01", 1, 2, 2, 0),
             _wf_match("2020-02-01", 1, 2, 1, 0)]
    test = [_wf_match("2020-06-01", 1, 2, 0, 0)]
    snaps = compute_pi_ratings_walk_forward(train, test)
    _, ratings = snaps[0]
    # Both teams should have 2 matches in their history
    assert get_team_experience(ratings, 1)["matches_played"] == 2
    assert get_team_experience(ratings, 2)["matches_played"] == 2
