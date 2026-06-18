"""Tests for the independent goal model.

Covers data invariants, probability math, chronology, shrinkage,
Dixon–Coles, tournament classification, and recency/importance weighting.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np

from soccer_ev_model.goal_model import (
    EloPoissonModel,
    GlobalPoissonModel,
    RegularizedTeamPoissonModel,
    dixon_coles_correction,
    importance_weights,
    recency_weights,
    scoreline_matrix,
    summarize_prediction,
)
from soccer_ev_model.goal_model_data import (
    GoalMatch,
    build_goal_matches,
    classify_tournament,
    tournament_class_counts,
)


def sample_matches():
    """Minimal match set for testing."""
    return [
        GoalMatch(date(2020, 1, 1), "A", "B", 1, 2, 2, 0, "Friendly", False),
        GoalMatch(date(2020, 2, 1), "B", "A", 2, 1, 1, 1, "Friendly", False),
        GoalMatch(date(2020, 3, 1), "A", "C", 1, 3, 1, 0, "Cup", True),
        GoalMatch(date(2020, 4, 1), "C", "B", 3, 2, 0, 2, "Cup", True),
    ]


def larger_match_set():
    """Generate a larger match set with clear home advantage."""
    matches = []
    rng = np.random.RandomState(42)
    for i in range(200):
        h = i % 10 + 1
        a = (i + 1) % 10 + 1
        d = date(2018, 1, 1) + timedelta(days=i * 7)
        home_goals = int(rng.poisson(1.6))
        away_goals = int(rng.poisson(1.1))
        is_neutral = (i % 3 == 0)
        tournament = "Friendly" if i % 4 == 0 else "FIFA World Cup qualification"
        matches.append(GoalMatch(
            d, f"T{h}", f"T{a}", h, a,
            home_goals, away_goals,
            tournament, is_neutral,
        ))
    return matches


def make_elo_correlated_matches():
    """Create matches where Elo correlates with goal scoring."""
    matches = []
    elo = {}
    rng = np.random.RandomState(42)
    base_date = date(2018, 1, 1)
    for i in range(200):
        h = i % 10 + 1
        a = (i + 1) % 10 + 1
        d = base_date + timedelta(days=i * 7)
        # Assign Elo: higher Elo → more goals
        home_elo = 1400 + (i % 5) * 50  # 1400-1600
        away_elo = 1400 + ((i + 2) % 5) * 50
        # Goals correlated with Elo
        home_xg = 0.8 + (home_elo - 1400) / 400 * 0.8  # 0.8 to 1.6
        away_xg = 0.8 + (away_elo - 1400) / 400 * 0.6  # 0.8 to 1.2
        hg = int(rng.poisson(home_xg))
        ag = int(rng.poisson(away_xg))
        is_neutral = (i % 3 == 0)
        matches.append(GoalMatch(d, f"T{h}", f"T{a}", h, a, hg, ag, "Friendly", is_neutral))
        elo[(d.isoformat(), f"T{h}", f"T{a}")] = (home_elo, away_elo)
    return matches, elo


def _unnormalized_matrix(home_xg, away_xg, max_goals=16):
    """Build an unnormalized Poisson scoreline matrix."""
    hp = np.array([math.exp(-home_xg) * home_xg ** i / math.factorial(i) for i in range(max_goals)])
    ap = np.array([math.exp(-away_xg) * away_xg ** i / math.factorial(i) for i in range(max_goals)])
    return np.outer(hp, ap)


# ===========================================================================
# Scoreline matrix tests
# ===========================================================================


def test_scoreline_matrix_is_normalized_and_nonnegative():
    m, raw_mass, tail_mass = scoreline_matrix(1.6, 0.9)
    assert m.shape == (16, 16)
    assert np.isclose(m.sum(), 1.0)
    assert np.all(m >= 0)
    assert raw_mass > 0.99
    assert tail_mass < 0.01


def test_scoreline_matrix_orientation():
    m, _, _ = scoreline_matrix(2.0, 0.5)
    assert m[2, 0] > m[0, 2]


def test_scoreline_matrix_hda_sums_to_one():
    m, _, _ = scoreline_matrix(1.5, 1.2)
    h = float(np.tril(m, -1).sum())
    d = float(np.trace(m))
    a = float(np.triu(m, 1).sum())
    assert abs(h + d + a - 1.0) < 1e-10
    assert h > a


def test_scoreline_matrix_neutral_equal_rates():
    m, _, _ = scoreline_matrix(1.3, 1.3)
    h = float(np.tril(m, -1).sum())
    a = float(np.triu(m, 1).sum())
    assert abs(h - a) < 1e-10


def test_scoreline_matrix_tail_mass_small():
    for hx, ax in [(1.0, 1.0), (2.0, 0.5), (1.5, 1.2), (3.0, 2.0)]:
        _, raw_mass, tail_mass = scoreline_matrix(hx, ax)
        assert tail_mass < 0.001


def test_scoreline_matrix_rejects_negative():
    try:
        scoreline_matrix(-1.0, 1.0)
        assert False
    except ValueError:
        pass


def test_scoreline_matrix_rejects_nan():
    try:
        scoreline_matrix(float("nan"), 1.0)
        assert False
    except ValueError:
        pass


# ===========================================================================
# Global Poisson tests
# ===========================================================================


def test_global_neutral_removes_home_advantage():
    model = GlobalPoissonModel.fit(larger_match_set())
    pred = model.predict(neutral=True)
    assert pred["home_xg"] == pred["away_xg"]
    assert abs(sum(pred["hda_probs"].values()) - 1.0) < 1e-12


def test_global_non_neutral_has_home_advantage():
    model = GlobalPoissonModel.fit(larger_match_set())
    pred = model.predict(neutral=False)
    assert pred["home_xg"] > pred["away_xg"]


def test_global_hda_sums_to_one():
    model = GlobalPoissonModel.fit(sample_matches())
    pred = model.predict()
    assert abs(sum(pred["hda_probs"].values()) - 1.0) < 1e-12


def test_global_home_win_more_likely_than_away():
    model = GlobalPoissonModel.fit(larger_match_set())
    pred = model.predict(neutral=False)
    assert pred["hda_probs"]["home"] > pred["hda_probs"]["away"]


def test_global_deterministic():
    m1 = GlobalPoissonModel.fit(sample_matches())
    m2 = GlobalPoissonModel.fit(sample_matches())
    assert m1 == m2
    assert m1.predict() == m2.predict()


def test_global_raw_and_normalized_mass():
    model = GlobalPoissonModel.fit(sample_matches())
    pred = model.predict()
    assert pred["raw_matrix_mass"] > 0.99
    assert pred["tail_mass"] < 0.01


# ===========================================================================
# Elo Poisson tests
# ===========================================================================


def test_elo_stronger_home_increases_home_xg():
    matches, elo = make_elo_correlated_matches()
    model = EloPoissonModel.fit(matches, elo)
    pred_strong = model.predict(home_elo=1700, away_elo=1500)
    pred_even = model.predict(home_elo=1500, away_elo=1500)
    assert pred_strong["home_xg"] > pred_even["home_xg"]


def test_elo_stronger_away_reduces_home_xg():
    matches, elo = make_elo_correlated_matches()
    model = EloPoissonModel.fit(matches, elo)
    pred = model.predict(home_elo=1500, away_elo=1600)
    pred_same = model.predict(home_elo=1500, away_elo=1500)
    assert pred["home_xg"] < pred_same["home_xg"]


def test_elo_neutral_keeps_elo_effect():
    matches, elo = make_elo_correlated_matches()
    model = EloPoissonModel.fit(matches, elo)
    pred = model.predict(home_elo=1600, away_elo=1500, neutral=True)
    pred_rev = model.predict(home_elo=1500, away_elo=1600, neutral=True)
    assert pred["home_xg"] > pred_rev["home_xg"]


def test_elo_hda_sums_to_one():
    matches, elo = make_elo_correlated_matches()
    model = EloPoissonModel.fit(matches, elo)
    pred = model.predict(home_elo=1550, away_elo=1500)
    assert abs(sum(pred["hda_probs"].values()) - 1.0) < 1e-12


def test_elo_slope_grid_recorded():
    matches, elo = make_elo_correlated_matches()
    grid = (-0.002, -0.001, 0.0, 0.001, 0.002)
    model = EloPoissonModel.fit(matches, elo, slope_grid=grid)
    assert model.slope_grid_tested == grid
    assert len(model.slope_nll_scores) == len(grid)


def test_elo_deterministic():
    matches, elo = make_elo_correlated_matches()
    m1 = EloPoissonModel.fit(matches, elo)
    m2 = EloPoissonModel.fit(matches, elo)
    assert m1 == m2
    assert m1.predict(home_elo=1550, away_elo=1500) == m2.predict(home_elo=1550, away_elo=1500)


# ===========================================================================
# Regularized team Poisson tests
# ===========================================================================


def test_regularized_model_is_deterministic():
    first = RegularizedTeamPoissonModel.fit(sample_matches(), shrinkage=10)
    second = RegularizedTeamPoissonModel.fit(sample_matches(), shrinkage=10)
    assert first == second
    assert first.predict(home_team_id=1, away_team_id=2) == second.predict(home_team_id=1, away_team_id=2)


def test_unseen_team_falls_back_safely():
    model = RegularizedTeamPoissonModel.fit(sample_matches())
    pred = model.predict(home_team_id=999, away_team_id=2, neutral=True)
    assert "home_unseen" in pred["low_data_flags"]
    assert pred["home_xg"] >= 0
    assert pred["away_xg"] >= 0
    assert abs(sum(pred["hda_probs"].values()) - 1.0) < 1e-12


def test_regularized_neutral_reduces_advantage():
    """Neutral matches should have smaller home-away gap than non-neutral."""
    matches = larger_match_set()
    model = RegularizedTeamPoissonModel.fit(matches)
    pred_neutral = model.predict(home_team_id=1, away_team_id=2, neutral=True)
    pred_home = model.predict(home_team_id=1, away_team_id=2, neutral=False)
    diff_neutral = abs(pred_neutral["home_xg"] - pred_neutral["away_xg"])
    diff_home = abs(pred_home["home_xg"] - pred_home["away_xg"])
    # With shrinkage the home advantage should be smaller or equal for neutral
    # Use generous tolerance since team effects are estimated from limited data
    assert diff_neutral <= diff_home + 0.02


def test_regularized_shrinkage_grid():
    matches = larger_match_set()
    model_low = RegularizedTeamPoissonModel.fit(matches, shrinkage=5)
    model_high = RegularizedTeamPoissonModel.fit(matches, shrinkage=80)
    spread_low = max(model_low.attacks.values()) - min(model_low.attacks.values())
    spread_high = max(model_high.attacks.values()) - min(model_high.attacks.values())
    assert spread_high <= spread_low


def test_regularized_expected_goals_positive():
    model = RegularizedTeamPoissonModel.fit(sample_matches())
    pred = model.predict(home_team_id=1, away_team_id=2)
    assert pred["home_xg"] > 0
    assert pred["away_xg"] > 0
    assert pred["home_xg"] <= 6.0
    assert pred["away_xg"] <= 6.0


def test_regularized_convergence_reported():
    matches = larger_match_set()
    model = RegularizedTeamPoissonModel.fit(matches, shrinkage=20)
    assert model.iterations_run > 0
    assert model.iterations_run <= 50


# ===========================================================================
# Data builder tests
# ===========================================================================


def test_data_builder_deduplicates_without_mutating_raw():
    row = {"date": "2020-01-01", "home_team": "A", "away_team": "B",
           "home_team_id": 1, "away_team_id": 2, "home_goals": 1,
           "away_goals": 0, "tournament": "Friendly", "neutral": False}
    raw = [dict(row), dict(row)]
    clean, excluded = build_goal_matches(raw)
    assert len(clean) == 1
    assert excluded["exact_duplicate"] == 1
    assert len(raw) == 2


def test_data_builder_preserves_raw():
    row = {"date": "2020-01-01", "home_team": "A", "away_team": "B",
           "home_team_id": 1, "away_team_id": 2, "home_goals": 1,
           "away_goals": 0, "tournament": "Friendly", "neutral": True}
    raw_orig = dict(row)
    raw = [row]
    build_goal_matches(raw)
    assert row == raw_orig


# ===========================================================================
# Tournament classification tests
# ===========================================================================


def test_classify_world_cup():
    assert classify_tournament("FIFA World Cup") == "world_cup"


def test_classify_world_cup_qualifier():
    assert classify_tournament("FIFA World Cup qualification") == "world_cup_qualifier"
    assert classify_tournament("FIFA World Cup qualification") != "world_cup"


def test_classify_continental_championship():
    assert classify_tournament("UEFA Euro") == "continental_championship"
    assert classify_tournament("Copa América") == "continental_championship"
    assert classify_tournament("African Cup of Nations") == "continental_championship"
    assert classify_tournament("AFC Asian Cup") == "continental_championship"
    assert classify_tournament("Gold Cup") == "continental_championship"


def test_classify_continental_qualifier():
    assert classify_tournament("UEFA Euro qualification") == "continental_qualifier"
    assert classify_tournament("AFC Asian Cup qualification") == "continental_qualifier"
    assert classify_tournament("African Cup of Nations qualification") == "continental_qualifier"


def test_classify_nations_league():
    assert classify_tournament("CONCACAF Nations League") == "nations_league"
    assert classify_tournament("UEFA Nations League") == "nations_league"


def test_classify_friendly():
    assert classify_tournament("Friendly") == "friendly"


def test_classify_unknown_maps_to_other():
    assert classify_tournament("Mystery Tournament 2025") == "other"


def test_classify_deterministic():
    for _ in range(10):
        assert classify_tournament("FIFA World Cup") == "world_cup"


def test_classify_case_insensitive():
    assert classify_tournament("fifa world cup") == "world_cup"
    assert classify_tournament("FIFA WORLD CUP") == "world_cup"
    assert classify_tournament("Fifa World Cup") == "world_cup"


def test_tournament_class_counts():
    counts = tournament_class_counts(sample_matches())
    assert counts.get("friendly", 0) >= 2


# ===========================================================================
# Summarize prediction tests
# ===========================================================================


def test_summarize_prediction_hda_sums_to_one():
    pred = summarize_prediction(1.5, 1.2)
    assert abs(sum(pred["hda_probs"].values()) - 1.0) < 1e-12


def test_summarize_prediction_home_gt_away():
    pred = summarize_prediction(2.0, 0.8)
    assert pred["hda_probs"]["home"] > pred["hda_probs"]["away"]


def test_summarize_prediction_neutral():
    pred = summarize_prediction(1.3, 1.3)
    assert abs(pred["hda_probs"]["home"] - pred["hda_probs"]["away"]) < 1e-10


def test_summarize_prediction_expected_total():
    pred = summarize_prediction(1.5, 1.2)
    assert abs(pred["expected_total_goals"] - 2.7) < 1e-10


# ===========================================================================
# Dixon–Coles tests
# ===========================================================================


def test_dixon_coles_only_adjusts_low_scores():
    unnorm = _unnormalized_matrix(1.5, 1.0, max_goals=11)
    corrected = dixon_coles_correction(unnorm, 1.5, 1.0, rho=-0.1)
    original_norm = unnorm / unnorm.sum()
    for i in range(2, 11):
        for j in range(2, 11):
            assert np.isclose(corrected[i, j], original_norm[i, j]), \
                f"Cell ({i},{j}) should not be adjusted"


def test_dixon_coles_rho_zero_unchanged():
    unnorm = _unnormalized_matrix(1.5, 1.0)
    corrected = dixon_coles_correction(unnorm, 1.5, 1.0, rho=0.0)
    original_norm = unnorm / unnorm.sum()
    assert np.allclose(corrected, original_norm)


def test_dixon_coles_probabilities_nonnegative():
    unnorm = _unnormalized_matrix(1.5, 1.0)
    for rho in [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1]:
        corrected = dixon_coles_correction(unnorm, 1.5, 1.0, rho=rho)
        assert np.all(corrected >= 0), f"negative prob at rho={rho}"


def test_dixon_coles_renormalizes():
    unnorm = _unnormalized_matrix(1.5, 1.0)
    for rho in [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1]:
        corrected = dixon_coles_correction(unnorm, 1.5, 1.0, rho=rho)
        assert abs(corrected.sum() - 1.0) < 1e-10


def test_dixon_coles_rejects_extreme_rho():
    unnorm = _unnormalized_matrix(1.5, 1.0)
    try:
        dixon_coles_correction(unnorm, 1.5, 1.0, rho=0.5)
        assert False
    except ValueError:
        pass


def test_dixon_coles_negative_rho_increases_00():
    unnorm = _unnormalized_matrix(1.5, 1.0)
    corrected_pos = dixon_coles_correction(unnorm, 1.5, 1.0, rho=0.1)
    corrected_neg = dixon_coles_correction(unnorm, 1.5, 1.0, rho=-0.1)
    assert corrected_neg[0, 0] > corrected_pos[0, 0]


# ===========================================================================
# Recency weight tests
# ===========================================================================


def test_recency_weights_uniform_when_no_decay():
    matches = sample_matches()
    w = recency_weights(matches, date(2021, 1, 1), half_life_days=None)
    assert np.allclose(w, 1.0)


def test_recency_weights_decay():
    matches = sample_matches()
    cutoff = date(2021, 1, 1)
    w = recency_weights(matches, cutoff, half_life_days=365)
    ages = [(cutoff - m.match_date).days for m in matches]
    min_age_idx = ages.index(min(ages))
    max_age_idx = ages.index(max(ages))
    assert w[min_age_idx] > w[max_age_idx]


def test_recency_weights_all_positive():
    matches = sample_matches()
    w = recency_weights(matches, date(2021, 1, 1), half_life_days=365)
    assert np.all(w > 0)


# ===========================================================================
# Importance weight tests
# ===========================================================================


def test_importance_weights_world_cup_higher_than_friendly():
    matches = [
        GoalMatch(date(2020, 1, 1), "A", "B", 1, 2, 1, 0, "FIFA World Cup", True),
        GoalMatch(date(2020, 2, 1), "C", "D", 3, 4, 1, 1, "Friendly", False),
    ]
    from soccer_ev_model.goal_model_data import MATCH_IMPORTANCE_WEIGHTS
    w = importance_weights(matches, MATCH_IMPORTANCE_WEIGHTS)
    assert w[0] > w[1]


def test_importance_weights_unknown_class_maps_to_other():
    matches = [GoalMatch(date(2020, 1, 1), "A", "B", 1, 2, 1, 0, "MysteryTournament", False)]
    from soccer_ev_model.goal_model_data import MATCH_IMPORTANCE_WEIGHTS
    w = importance_weights(matches, MATCH_IMPORTANCE_WEIGHTS)
    assert w[0] == 0.9  # 'other' class weight
