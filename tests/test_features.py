"""Tests for the feature pipeline.

The feature pipeline turns a list of historical matches + their dates into
a feature matrix suitable for training a classifier. The CRITICAL invariant
is that features for match M are computed using only matches strictly
before M's date (no leakage).

Features computed per (match, team) pair:
- pi-rating offense, defense (via cutoff-dated pi-rating computation)
- days since previous match (rest)
- wins/draws/losses in last 5 matches (form)
- goal diff in last 5 matches

Features computed per match (matchup-level, not per-team):
- pi-rating diffs (offense_diff, defense_diff, matchup_strength)
- total pi-rating strength gap

The pipeline is strict about ordering: it processes matches in chronological
order and computes each match's features from a pi-rating snapshot taken
just before that match.
"""

import random

import numpy as np
import pytest

from soccer_ev_model.features import (
    _FeatureBuilderState,
    build_feature_matrix,
    compute_recent_form,
    rest_days,
)


# ---- recent form ----


def test_recent_form_no_history_returns_zeros():
    """A team with no prior matches has no form. Return zeros, not errors."""
    form = compute_recent_form(team_id=1, target_date="2022-01-01",
                               prior_matches=[])
    assert form == {"wins": 0, "draws": 0, "losses": 0, "goal_diff": 0,
                    "matches_used": 0}


def test_recent_form_counts_only_last_5():
    """Even if a team has 100 prior matches, we only look at the last 5."""
    # 8 wins in 2020, 2 losses in 2022.
    matches = [
        *[_make_match(f"2020-0{i+1}-01", home_id=1, away_id=99, result="H",
                      home_goals=2, away_goals=0) for i in range(8)],
        _make_match("2022-01-01", home_id=1, away_id=88, result="A",
                    home_goals=0, away_goals=2),
        _make_match("2022-02-01", home_id=99, away_id=1, result="A",
                    home_goals=3, away_goals=1),
    ]
    form = compute_recent_form(team_id=1, target_date="2022-03-01",
                               prior_matches=matches, n=5)
    # 10 total matches. Most recent 5 by date = 2020-04..08 (5 wins) + the 2
    # 2022 losses. Sorted newest first, last 5 are: 2022-02, 2022-01, 2020-08,
    # 2020-07, 2020-06. That's 2 losses + 3 wins.
    assert form["matches_used"] == 5
    assert form["losses"] == 2
    assert form["wins"] == 3
    assert form["draws"] == 0


def test_recent_form_uses_strictly_before_target_date():
    """Matches on or after the target date are NOT used."""
    matches = [
        _make_match("2022-01-01", home_id=1, away_id=99, result="H",
                    home_goals=5, away_goals=0),  # before
        _make_match("2022-06-01", home_id=1, away_id=99, result="A",
                    home_goals=0, away_goals=5),  # AFTER
    ]
    form = compute_recent_form(team_id=1, target_date="2022-03-01",
                               prior_matches=matches)
    # Only the 2022-01-01 match counts (won 5-0)
    assert form["matches_used"] == 1
    assert form["wins"] == 1
    assert form["goal_diff"] == 5  # +5 from the win


def test_recent_form_includes_both_home_and_away_perspective():
    """A team can be home in one match and away in another. Both count."""
    matches = [
        _make_match("2022-01-01", home_id=1, away_id=99, result="H",
                    home_goals=3, away_goals=1),  # team 1 won at home
        _make_match("2022-02-01", home_id=99, away_id=1, result="A",
                    home_goals=0, away_goals=2),  # team 1 won away
    ]
    form = compute_recent_form(team_id=1, target_date="2022-03-01",
                               prior_matches=matches)
    assert form["matches_used"] == 2
    assert form["wins"] == 2
    assert form["goal_diff"] == 4  # +2 + +2


# ---- rest days ----


def test_rest_days_no_prior_match_returns_large_number():
    """If a team has never played, 'rest' is effectively infinite.
    Use a large finite number (e.g. 999) so the model can still train on it."""
    rest = rest_days(team_id=1, target_date="2022-06-01", prior_matches=[])
    assert rest == 999


def test_rest_days_returns_days_since_last_match():
    """Simple case: last match was 10 days before target."""
    matches = [
        _make_match("2022-05-22", home_id=1, away_id=99, result="H",
                    home_goals=1, away_goals=0),
    ]
    rest = rest_days(team_id=1, target_date="2022-06-01", prior_matches=matches)
    assert rest == 10


def test_rest_days_ignores_other_teams_matches():
    """Only count matches where this team actually played."""
    matches = [
        _make_match("2022-05-25", home_id=2, away_id=3, result="H",  # not team 1
                    home_goals=1, away_goals=0),
        _make_match("2022-05-22", home_id=1, away_id=99, result="H",  # team 1
                    home_goals=1, away_goals=0),
    ]
    rest = rest_days(team_id=1, target_date="2022-06-01", prior_matches=matches)
    # The team-1 match is older (5/22 < 5/25), so it's the relevant one
    assert rest == 10


# ---- full feature matrix ----


def test_build_feature_matrix_emits_one_row_per_match():
    """The matrix has one row per input match. No aggregation."""
    matches = [
        _make_match("2022-01-01", home_id=1, away_id=2, result="H",
                    home_goals=2, away_goals=1),
        _make_match("2022-01-15", home_id=1, away_id=3, result="D",
                    home_goals=1, away_goals=1),
    ]
    X, y = build_feature_matrix(matches)
    assert len(X) == 2


def test_build_feature_matrix_target_is_result_code():
    """y is the result code (H, D, A) for each match."""
    matches = [
        _make_match("2022-01-01", home_id=1, away_id=2, result="H",
                    home_goals=2, away_goals=1),
        _make_match("2022-01-15", home_id=1, away_id=3, result="D",
                    home_goals=1, away_goals=1),
        _make_match("2022-02-01", home_id=4, away_id=1, result="A",
                    home_goals=0, away_goals=3),
    ]
    X, y = build_feature_matrix(matches)
    assert list(y) == ["H", "D", "A"]


def test_build_feature_matrix_first_match_has_neutral_features():
    """The very first match (no history) should have neutral feature values,
    not NaN or missing. Models can't handle NaN."""
    matches = [
        _make_match("2022-01-01", home_id=1, away_id=2, result="H",
                    home_goals=2, away_goals=1),
    ]
    X, y = build_feature_matrix(matches)
    # No NaN values
    assert not X.isna().any().any(), f"Found NaN in features: {X}"
    # First match: pi-ratings both 0 (no prior matches), form 0/0/0, etc.
    row = X.iloc[0]
    assert row["pi_off_diff"] == 0.0
    assert row["pi_def_diff"] == 0.0
    assert row["home_form_wins"] == 0
    assert row["away_form_wins"] == 0


def test_build_feature_matrix_second_match_uses_first_match_data():
    """The second match's features should reflect the first match's outcome.
    If team 1 won the first match, team 1's pi-rating should be positive
    when entering the second match."""
    matches = [
        _make_match("2022-01-01", home_id=1, away_id=2, result="H",
                    home_goals=5, away_goals=0),
        _make_match("2022-01-15", home_id=1, away_id=3, result="D",
                    home_goals=1, away_goals=1),
    ]
    X, y = build_feature_matrix(matches)
    second = X.iloc[1]
    # Team 1 scored 5 in the first match, so its offense rating should be > 0
    # for the second match. Team 3 had no prior matches -> 0.
    assert second["pi_home_off"] > 0
    assert second["pi_away_off"] == 0.0
    assert second["pi_off_diff"] > 0


def test_build_feature_matrix_orders_matches_chronologically():
    """Input matches can be in any order. Output is in chronological order
    (oldest first). This is critical for temporal splits downstream."""
    matches = [
        _make_match("2022-02-01", home_id=4, away_id=1, result="A",
                    home_goals=0, away_goals=3),
        _make_match("2022-01-01", home_id=1, away_id=2, result="H",
                    home_goals=2, away_goals=1),
    ]
    X, y = build_feature_matrix(matches)
    # Should be sorted by date: 2022-01-01 first
    assert y.iloc[0] == "H"
    assert y.iloc[1] == "A"


# ---- helper: a tiny match factory ----


def _make_match(date, home_id, away_id, result, home_goals, away_goals):
    """Tiny helper to build a match record in the format features.py expects."""
    return {
        "date": f"{date}T00:00:00Z",
        "home_team_id": home_id,
        "away_team_id": away_id,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "result": result,
    }


# ---- numerical equivalence: incremental build == batch rebuild ----


def _synthetic_match_list(seed: int = 7, n: int = 200, n_teams: int = 12):
    """Generate a small synthetic match list with a mix of teams, dates, scores.

    Used by the equivalence test below to compare the new incremental
    build path against a reference batch rebuild of pi-ratings / form /
    rest-days. The synthetic data has overlapping teams, multiple matches
    per team, draws, and varied goal margins — enough surface area to
    exercise the math.
    """
    rng = random.Random(seed)
    matches = []
    # Spread over ~2 years so rest_days varies
    base_year = 2020
    for i in range(n):
        month = (i // 15) % 12 + 1
        day = (i % 28) + 1
        h = rng.randint(1, n_teams)
        a = rng.randint(1, n_teams)
        # Avoid self-matches (data uses team ids, not home/away semantics
        # that prevent home==away, but real data has none so keep clean)
        if h == a:
            a = (a % n_teams) + 1
        hg = rng.randint(0, 4)
        ag = rng.randint(0, 4)
        if hg == ag:
            result = "D"
        elif hg > ag:
            result = "H"
        else:
            result = "A"
        date = f"{base_year + (i // 200):04d}-{month:02d}-{day:02d}"
        matches.append(_make_match(date, h, a, result, hg, ag))
    return matches


def _reference_features(matches):
    """Brute-force reference build that calls rest_days / compute_recent_form
    and compute_pi_ratings on a per-match basis (the OLD O(N^2) path).

    Used to verify the new incremental build_feature_matrix produces
    numerically identical output.
    """
    from soccer_ev_model.pi_ratings import compute_pi_ratings, pi_diff_features

    sorted_matches = sorted(matches, key=lambda m: m.get("date", ""))
    prior: list[dict] = []
    rows: list[dict] = []
    labels: list[str] = []
    for m in sorted_matches:
        date = m.get("date", "")
        home_id = m.get("home_team_id")
        away_id = m.get("away_team_id")
        ratings = compute_pi_ratings(prior, cutoff=date)
        pi_feats = pi_diff_features(home_id, away_id, ratings)
        home_r = ratings.get(home_id, {"offense": 0.0, "defense": 0.0})
        away_r = ratings.get(away_id, {"offense": 0.0, "defense": 0.0})
        home_form = compute_recent_form(home_id, date, prior)
        away_form = compute_recent_form(away_id, date, prior)
        home_rest = rest_days(home_id, date, prior)
        away_rest = rest_days(away_id, date, prior)
        rows.append({
            "pi_home_off": home_r["offense"],
            "pi_away_off": away_r["offense"],
            "pi_home_def": home_r["defense"],
            "pi_away_def": away_r["defense"],
            "pi_off_diff": pi_feats["pi_off_diff"],
            "pi_def_diff": pi_feats["pi_def_diff"],
            "pi_matchup": pi_feats["pi_matchup"],
            "home_form_wins": home_form["wins"],
            "home_form_draws": home_form["draws"],
            "home_form_losses": home_form["losses"],
            "home_form_goal_diff": home_form["goal_diff"],
            "home_form_matches_used": home_form["matches_used"],
            "away_form_wins": away_form["wins"],
            "away_form_draws": away_form["draws"],
            "away_form_losses": away_form["losses"],
            "away_form_goal_diff": away_form["goal_diff"],
            "away_form_matches_used": away_form["matches_used"],
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
        })
        labels.append(m["result"])
        prior.append(m)
    return rows, labels


def test_incremental_build_matches_batch_within_tolerance():
    """The new incremental build_feature_matrix must produce numerically
    identical features to the old per-match batch path (within 1e-6).

    This is the regression guard: any future refactor that breaks the
    math will trip this test before it reaches the backtest.
    """
    matches = _synthetic_match_list(seed=42, n=150, n_teams=10)
    ref_rows, ref_labels = _reference_features(matches)
    X, y = build_feature_matrix(matches)

    # Same shape and ordering
    assert len(X) == len(ref_rows) == len(ref_labels)
    assert list(y) == ref_labels

    # Compare every numeric column
    numeric_cols = [
        "pi_home_off", "pi_away_off", "pi_home_def", "pi_away_def",
        "pi_off_diff", "pi_def_diff", "pi_matchup",
        "home_form_wins", "home_form_draws", "home_form_losses",
        "home_form_goal_diff", "home_form_matches_used",
        "away_form_wins", "away_form_draws", "away_form_losses",
        "away_form_goal_diff", "away_form_matches_used",
        "home_rest_days", "away_rest_days",
    ]
    for col in numeric_cols:
        ref = np.array([r[col] for r in ref_rows], dtype=float)
        got = X[col].to_numpy(dtype=float)
        assert ref.shape == got.shape, f"shape mismatch for {col}"
        # The two paths sort by date, so row order is identical.
        np.testing.assert_allclose(
            got, ref, atol=1e-6,
            err_msg=f"mismatch in column {col}",
        )


def test_incremental_build_handles_out_of_order_input():
    """Shuffled input must produce the same result as sorted input."""
    matches = _synthetic_match_list(seed=99, n=80, n_teams=6)
    X_sorted, y_sorted = build_feature_matrix(matches)

    # Shuffle and rebuild
    shuffled = matches[:]
    random.Random(123).shuffle(shuffled)
    X_shuf, y_shuf = build_feature_matrix(shuffled)

    # Output is in chronological order either way
    assert list(y_sorted) == list(y_shuf)
    for col in X_sorted.columns:
        if col == "date":
            continue
        np.testing.assert_array_equal(
            X_sorted[col].to_numpy(), X_shuf[col].to_numpy(),
        )


def test_state_snapshot_does_not_mutate():
    """Reading form/rest/pi-ratings from the state must NOT mutate it.
    This is the contract that lets us snapshot then update.
    """
    state = _FeatureBuilderState()
    # Add one match worth of state
    m = _make_match("2022-01-01", home_id=1, away_id=2, result="H",
                    home_goals=2, away_goals=1)
    state.update_after_match(m)

    # Snapshot — should be repeatable
    s1_form = state.home_form(1)
    s1_rest = state.rest_days(1, "2022-06-01")
    s1_pi = state.pi_snapshot()
    # Snapshot again — must be identical
    s2_form = state.home_form(1)
    s2_rest = state.rest_days(1, "2022-06-01")
    s2_pi = state.pi_snapshot()
    assert s1_form == s2_form
    assert s1_rest == s2_rest
    assert s1_pi[1]["offense"] == s2_pi[1]["offense"]
    assert s1_pi[1]["defense"] == s2_pi[1]["defense"]


def test_incremental_build_is_fast_on_synthetic_medium():
    """Sanity check: 1000 matches should build in well under 1 second.

    This is a soft smoke test; the real perf bar is the 33k run done
    outside the test suite.
    """
    matches = _synthetic_match_list(seed=0, n=1000, n_teams=20)
    import time
    t0 = time.time()
    X, y = build_feature_matrix(matches)
    dt = time.time() - t0
    assert dt < 1.0, f"1000-match build took {dt:.2f}s, expected < 1s"
    assert len(X) == 1000
