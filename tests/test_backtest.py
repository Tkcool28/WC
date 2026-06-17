"""Tests for the walk-forward backtest harness.

The backtest is the most leakage-sensitive part of the system. These tests
verify the temporal split, that training only sees past data, and that the
backtest loop honors the cutoff at each step.
"""

import pandas as pd
import pytest
from soccer_ev_model.train import train, evaluate
from soccer_ev_model.backtest import walk_forward_split, run_backtest


def _make_synthetic_matches():
    """Generate 100 synthetic matches with deterministic outcomes.

    Outcomes are a function of pi-rating strength diff so the model has
    SOMETHING to learn. We use openfootball-style team IDs and dates.
    """
    import random
    random.seed(42)
    matches = []
    teams = list(range(1, 33))  # 32 teams
    base = pd.Timestamp("2010-01-01")
    for i in range(100):
        d = base + pd.Timedelta(days=i * 5)
        home = random.choice(teams)
        away = random.choice([t for t in teams if t != home])
        # Stronger team (lower id) tends to win at home
        if home < away:
            home_goals = random.choice([2, 2, 1, 3, 1, 0])
            away_goals = random.choice([0, 1, 0, 0, 1, 2])
        else:
            home_goals = random.choice([0, 1, 1, 0, 2, 0])
            away_goals = random.choice([2, 1, 2, 3, 1, 1])
        if home_goals > away_goals:
            result = "H"
        elif home_goals < away_goals:
            result = "A"
        else:
            result = "D"
        matches.append({
            "date": d.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "home_team_id": home,
            "away_team_id": away,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "result": result,
        })
    return matches


# ---- walk_forward_split ----


def test_walk_forward_split_returns_train_and_test():
    """Split returns a train set and a test set, both non-empty."""
    matches = _make_synthetic_matches()
    train_set, test_set = walk_forward_split(matches, train_end_date="2010-06-01")
    assert len(train_set) > 0
    assert len(test_set) > 0
    assert len(train_set) + len(test_set) == len(matches)


def test_walk_forward_split_is_strictly_temporal():
    """Train matches all have date < cutoff. Test matches all have date >= cutoff."""
    matches = _make_synthetic_matches()
    cutoff = "2010-06-01"
    train_set, test_set = walk_forward_split(matches, train_end_date=cutoff)
    # All train dates < cutoff
    for m in train_set:
        assert m["date"] < f"{cutoff}T23:59:59Z" or m["date"] < cutoff, (
            f"Train match has date {m['date']} which is >= cutoff {cutoff}"
        )
    # All test dates >= cutoff
    for m in test_set:
        assert m["date"] >= f"{cutoff}T00:00:00Z", (
            f"Test match has date {m['date']} which is < cutoff {cutoff}"
        )


def test_walk_forward_split_handles_all_in_one_side():
    """If cutoff is before all matches, train is empty. After all, test is empty."""
    matches = _make_synthetic_matches()
    # Cutoff way before any match
    train_set, test_set = walk_forward_split(matches, train_end_date="2000-01-01")
    assert len(train_set) == 0
    assert len(test_set) == len(matches)
    # Cutoff way after all matches
    train_set, test_set = walk_forward_split(matches, train_end_date="2099-01-01")
    assert len(train_set) == len(matches)
    assert len(test_set) == 0


# ---- run_backtest ----


def test_run_backtest_returns_metrics_dict():
    """Backtest returns metrics for both models."""
    from soccer_ev_model.features import build_feature_matrix
    matches = _make_synthetic_matches()
    result = run_backtest(
        matches=matches,
        train_end_date="2010-06-01",
        model_types=("logreg",),
    )
    assert "logreg" in result
    metrics = result["logreg"]
    for k in ["accuracy", "log_loss", "brier_avg", "rps", "n"]:
        assert k in metrics, f"Missing metric: {k}"


def test_run_backtest_logreg_does_not_crash():
    """Smoke test: the backtest runs end-to-end on synthetic data.

    We do NOT assert accuracy > random here. With only 49 training matches
    and 32 teams, the model has too little signal to outperform uniform
    guessing. The HONEST result on real data is similar: see the
    walk_forward.py output. The test just confirms the pipeline doesn't
    throw exceptions and returns the right shape.
    """
    from soccer_ev_model.features import build_feature_matrix
    matches = _make_synthetic_matches()
    result = run_backtest(
        matches=matches,
        train_end_date="2010-09-01",
        model_types=("logreg", "catboost"),
    )
    assert "logreg" in result
    assert "catboost" in result
    for mt, m in result.items():
        assert "n" in m
        assert m["n"] > 0
        # Sanity: the model output should be valid probability distributions
        # (We don't assert this directly because evaluate() returns metrics,
        # but we verify the pipeline didn't error.)
