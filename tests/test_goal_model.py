from datetime import date

import numpy as np

from soccer_ev_model.goal_model import (
    GlobalPoissonModel,
    RegularizedTeamPoissonModel,
    scoreline_matrix,
)
from soccer_ev_model.goal_model_data import GoalMatch, build_goal_matches


def sample_matches():
    return [
        GoalMatch(date(2020,1,1), "A", "B", 1, 2, 2, 0, "Friendly", False),
        GoalMatch(date(2020,2,1), "B", "A", 2, 1, 1, 1, "Friendly", False),
        GoalMatch(date(2020,3,1), "A", "C", 1, 3, 1, 0, "Cup", True),
        GoalMatch(date(2020,4,1), "C", "B", 3, 2, 0, 2, "Cup", True),
    ]


def test_scoreline_matrix_is_normalized_and_nonnegative():
    m = scoreline_matrix(1.6, 0.9)
    assert m.shape == (11, 11)
    assert np.isclose(m.sum(), 1.0)
    assert np.all(m >= 0)


def test_global_neutral_removes_home_advantage():
    model = GlobalPoissonModel.fit(sample_matches())
    pred = model.predict(neutral=True)
    assert pred["home_xg"] == pred["away_xg"]
    assert abs(sum(pred["hda_probs"].values()) - 1.0) < 1e-12


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


def test_data_builder_deduplicates_without_mutating_raw():
    row = {"date":"2020-01-01", "home_team":"A", "away_team":"B",
           "home_team_id":1, "away_team_id":2, "home_goals":1,
           "away_goals":0, "tournament":"Friendly", "neutral":False}
    raw = [dict(row), dict(row)]
    clean, excluded = build_goal_matches(raw)
    assert len(clean) == 1
    assert excluded["exact_duplicate"] == 1
    assert len(raw) == 2
