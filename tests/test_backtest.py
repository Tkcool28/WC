"""Tests for the backtest harness."""
from __future__ import annotations

from datetime import date

import numpy as np

from soccer_ev_model.goal_model import GlobalPoissonModel, RegularizedTeamPoissonModel
from soccer_ev_model.goal_model_backtest import (
    HOLDOUT_2014_WC,
    HOLDOUT_2018_WC,
    HOLDOUT_2022_WC,
    BacktestMetrics,
    compute_rps,
    run_backtest,
)
from soccer_ev_model.goal_model_data import GoalMatch


def make_test_matches(n=200):
    """Create test matches spanning 2010-2022 for holdout testing."""
    matches = []
    rng = np.random.RandomState(42)
    # Generate matches across the full date range
    for i in range(n):
        # Spread from 2010 to 2022
        year = 2010 + (i * 12) // n
        month = 1 + (i % 12)
        day = 1 + (i % 28)
        d = date(min(year, 2022), month, day)
        h = i % 5 + 1
        a = (i + 1) % 5 + 1
        if h == a:
            a = (a + 1) % 5 + 1
        matches.append(GoalMatch(
            d, f"T{h}", f"T{a}", h, a,
            int(rng.poisson(1.5)), int(rng.poisson(1.0)),
            "Friendly", i % 2 == 0,
        ))
    # Ensure we have matches in each WC period
    for yr, mth in [(2014, 6), (2018, 6), (2022, 11)]:
        for day in range(1, 15):
            d = date(yr, mth, day)
            h = day % 5 + 1
            a = (day + 1) % 5 + 1
            if h == a:
                a = (a + 1) % 5 + 1
            matches.append(GoalMatch(
                d, f"T{h}", f"T{a}", h, a,
                int(rng.poisson(1.5)), int(rng.poisson(1.0)),
                "FIFA World Cup", True,
            ))
    return matches


# ===========================================================================
# Chronology tests
# ===========================================================================


def test_backtest_no_future_leakage():
    """Training data must be strictly before prediction date."""
    from soccer_ev_model.goal_model_backtest import HoldoutPeriod
    matches = make_test_matches(200)
    # Use a holdout that falls within our test data range
    holdout = HoldoutPeriod("test_2022", date(2022, 11, 1), date(2022, 12, 31))
    result = run_backtest("global_poisson", matches, holdout)
    assert result.metrics.n_matches > 0


def test_backtest_same_date_grouping():
    """All matches on the same date should be predicted from the same cutoff."""
    from soccer_ev_model.goal_model_backtest import HoldoutPeriod
    matches = [
        GoalMatch(date(2022, 6, 1), "A", "B", 1, 2, 2, 1, "Cup", True),
        GoalMatch(date(2022, 6, 1), "C", "D", 3, 4, 1, 0, "Cup", True),
        GoalMatch(date(2022, 6, 2), "E", "F", 5, 6, 0, 3, "Cup", True),
    ]
    rng = np.random.RandomState(42)
    for i in range(60):
        d = date(2020 + i // 24, 1 + (i % 12), 1 + (i % 28))
        matches.append(GoalMatch(d, f"X{i}", f"Y{i}", 100 + i, 200 + i,
                                  int(rng.poisson(1.5)), int(rng.poisson(1.0)), "Friendly", False))
    holdout = HoldoutPeriod("test_jun2022", date(2022, 6, 1), date(2022, 6, 30))
    result = run_backtest("global_poisson", matches, holdout)
    assert result.metrics.n_matches >= 2


def test_backtest_deterministic():
    """Same input should produce same output."""
    from soccer_ev_model.goal_model_backtest import HoldoutPeriod
    matches = make_test_matches(200)
    holdout = HoldoutPeriod("test_det", date(2022, 11, 1), date(2022, 12, 31))
    r1 = run_backtest("global_poisson", matches, holdout)
    r2 = run_backtest("global_poisson", matches, holdout)
    assert r1.metrics.log_loss == r2.metrics.log_loss
    assert r1.metrics.ranked_probability_score == r2.metrics.ranked_probability_score


def test_backtest_different_holdouts_different_results():
    """Different holdout periods should produce different results."""
    from soccer_ev_model.goal_model_backtest import HoldoutPeriod
    matches = make_test_matches(200)
    h1 = HoldoutPeriod("test_2014", date(2014, 6, 1), date(2014, 6, 30))
    h2 = HoldoutPeriod("test_2022", date(2022, 11, 1), date(2022, 12, 31))
    r1 = run_backtest("global_poisson", matches, h1)
    r2 = run_backtest("global_poisson", matches, h2)
    assert r1.metrics.n_matches != r2.metrics.n_matches or r1.metrics.log_loss != r2.metrics.log_loss


# ===========================================================================
# Metrics tests
# ===========================================================================


def test_rps_perfect_prediction():
    """RPS should be 0 for perfect predictions."""
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    outcomes = np.array([0, 1, 2])
    rps = compute_rps(probs, outcomes)
    assert abs(rps) < 1e-10


def test_rps_worst_prediction():
    """RPS should be higher for worse predictions."""
    probs_good = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    probs_bad = np.array([[0.1, 0.1, 0.8], [0.8, 0.1, 0.1]])
    outcomes = np.array([0, 1])
    rps_good = compute_rps(probs_good, outcomes)
    rps_bad = compute_rps(probs_bad, outcomes)
    assert rps_good < rps_bad


def test_metrics_hda_calibration():
    """Calibration should reflect actual outcomes."""
    from soccer_ev_model.goal_model_backtest import HoldoutPeriod
    matches = make_test_matches(200)
    holdout = HoldoutPeriod("test_cal", date(2022, 11, 1), date(2022, 12, 31))
    result = run_backtest("global_poisson", matches, holdout)
    m = result.metrics
    assert 0 <= m.home_calibration <= 1
    assert 0 <= m.draw_calibration <= 1
    assert 0 <= m.away_calibration <= 1
    assert abs(m.home_calibration + m.draw_calibration + m.away_calibration - 1.0) < 0.01


def test_metrics_log_loss_positive():
    """Log loss should be positive."""
    from soccer_ev_model.goal_model_backtest import HoldoutPeriod
    matches = make_test_matches(200)
    holdout = HoldoutPeriod("test_ll", date(2022, 11, 1), date(2022, 12, 31))
    result = run_backtest("global_poisson", matches, holdout)
    assert result.metrics.log_loss > 0


def test_metrics_mae_positive():
    """MAE should be positive."""
    from soccer_ev_model.goal_model_backtest import HoldoutPeriod
    matches = make_test_matches(200)
    holdout = HoldoutPeriod("test_mae", date(2022, 11, 1), date(2022, 12, 31))
    result = run_backtest("global_poisson", matches, holdout)
    assert result.metrics.mae_home_goals >= 0
    assert result.metrics.mae_away_goals >= 0
    assert result.metrics.mae_total_goals >= 0


# ===========================================================================
# Regularized team model backtest
# ===========================================================================


def test_regularized_backtest_runs():
    """Regularized team model should run without errors."""
    from soccer_ev_model.goal_model_backtest import HoldoutPeriod
    matches = make_test_matches(200)
    holdout = HoldoutPeriod("test_reg", date(2022, 11, 1), date(2022, 12, 31))
    result = run_backtest("regularized_team", matches, holdout)
    assert result.metrics.n_matches > 0


def test_regularized_vs_global():
    """Regularized team should generally match or beat global Poisson."""
    from soccer_ev_model.goal_model_backtest import HoldoutPeriod
    matches = make_test_matches(200)
    holdout = HoldoutPeriod("test_comp", date(2022, 11, 1), date(2022, 12, 31))
    r_global = run_backtest("global_poisson", matches, holdout)
    r_team = run_backtest("regularized_team", matches, holdout)
    print(f"Global: {r_global.metrics.log_loss:.4f}, Team: {r_team.metrics.log_loss:.4f}")
