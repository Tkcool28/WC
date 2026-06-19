"""Tests for Phase 5 — Goal model integration.

Covers:

* ``predict_match`` with ``goal_probs`` — Elo60/Goal40 blend produces
  correct ``primary_probs``.
* ``predict_match`` without ``goal_probs`` — backward compatible.
* ``prediction_why_text`` — goal-model-aware explanations.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dashboard.ux_presenters import prediction_why_text


# --------------------------------------------------------------------------- #
# predict_match: Elo60/Goal40 blend
# --------------------------------------------------------------------------- #
class TestPredictMatchGoalBlend:
    """Verify that goal_probs + Elo produces the Elo60/Goal40 blend."""

    def _make_ratings(self, home_id: int = 1, away_id: int = 2) -> dict:
        """Minimal ratings dict with two teams (string keys, offense/defense)."""
        return {
            str(home_id): {"matches_played": 50, "offense": 0.6, "defense": -0.3},
            str(away_id): {"matches_played": 50, "offense": 0.4, "defense": -0.2},
        }

    def _make_elo_snapshots(self, home_elo: float = 1100.0, away_elo: float = 1000.0) -> dict:
        return {
            "teams": {
                "Home": {"rating": home_elo},
                "Away": {"rating": away_elo},
            }
        }

    def test_goal_probs_with_elo_produces_blended_primary_probs(self) -> None:
        """When goal_probs and Elo are both present, primary_probs = Elo60/Goal40."""
        from soccer_ev_model.ev_workflow import predict_match

        ratings = self._make_ratings()
        goal_probs = {"home": 0.55, "draw": 0.20, "away": 0.25}

        result = predict_match(
            home_team="Home",
            away_team="Away",
            home_team_id=1,
            away_team_id=2,
            date="2026-06-18",
            ratings=ratings,
            home_elo=1100.0,
            away_elo=1000.0,
            goal_probs=goal_probs,
        )

        # primary_probs should be the Elo60/Goal40 blend, NOT the raw pi probs
        assert result["_goal_model_used"] is True
        pp = result["primary_probs"]
        pi = result["pi_probs"]
        # primary must differ from pure pi (the blend shifts toward goal_probs)
        assert pp != pi, "primary_probs should differ from pure pi when Goal+Elo blend is active"
        assert result["fallback_case"] == "A", "Case A expected with Elo+Goal+sufficient data"
        assert abs(pp["home"] + pp["draw"] + pp["away"] - 1.0) < 0.01
        # Verify pi_probs is pure pi (diagnostic), not the blend
        assert result["pi_probs"] == result["pi_only_probs"]

    def test_goal_probs_without_elo_uses_goal_only(self) -> None:
        """When goal_probs is provided but Elo is None, primary_probs = Goal-only (Case D)."""
        from soccer_ev_model.ev_workflow import predict_match

        ratings = self._make_ratings()
        goal_probs = {"home": 0.55, "draw": 0.20, "away": 0.25}

        result = predict_match(
            home_team="Home",
            away_team="Away",
            home_team_id=1,
            away_team_id=2,
            date="2026-06-18",
            ratings=ratings,
            home_elo=None,
            away_elo=None,
            goal_probs=goal_probs,
        )

        # Without Elo but with Goal, fallback Case D: 100% Goal
        assert result["_goal_model_used"] is True
        assert result["fallback_case"] == "D", "Case D expected with Goal but no Elo"
        # primary_probs should equal goal_probs (not pi_probs)
        for k in ("home", "draw", "away"):
            assert result["primary_probs"][k] == pytest.approx(goal_probs[k], abs=0.01)

    def test_no_goal_probs_uses_elo_only_fallback(self) -> None:
        """Without goal_probs but with Elo, primary_probs = Elo-only (Case C)."""
        from soccer_ev_model.ev_workflow import predict_match

        ratings = self._make_ratings()

        result = predict_match(
            home_team="Home",
            away_team="Away",
            home_team_id=1,
            away_team_id=2,
            date="2026-06-18",
            ratings=ratings,
            home_elo=1100.0,
            away_elo=1000.0,
        )

        assert result["_goal_model_used"] is False
        assert result["fallback_case"] == "C", "Case C expected with Elo but no Goal"
        # primary_probs should equal elo_only_probs (100% Elo, not pi)
        eo = result["elo_only_probs"]
        pp = result["primary_probs"]
        assert eo is not None
        for k in ("home", "draw", "away"):
            assert pp[k] == pytest.approx(eo[k], abs=0.01)
        # pi_probs is pure pi — must differ from primary (which is Elo-only)
        assert result["pi_probs"] != pp

    def test_goal_probs_normalizes_when_sum_is_not_1(self) -> None:
        """Elo60/Goal40 blend is normalized to sum to 1.0."""
        from soccer_ev_model.ev_workflow import predict_match

        ratings = self._make_ratings()
        # Deliberately extreme goal_probs that would not sum to 1 after blend
        goal_probs = {"home": 0.8, "draw": 0.1, "away": 0.1}

        result = predict_match(
            home_team="Home",
            away_team="Away",
            home_team_id=1,
            away_team_id=2,
            date="2026-06-18",
            ratings=ratings,
            home_elo=1100.0,
            away_elo=1000.0,
            goal_probs=goal_probs,
        )

        pp = result["primary_probs"]
        total = pp["home"] + pp["draw"] + pp["away"]
        assert abs(total - 1.0) < 0.001

    def test_goal_model_xg_preserved_in_result(self) -> None:
        """When goal_model_xg is provided, it appears in the result."""
        from soccer_ev_model.ev_workflow import predict_match

        ratings = self._make_ratings()
        goal_probs = {"home": 0.45, "draw": 0.25, "away": 0.30}

        result = predict_match(
            home_team="Home",
            away_team="Away",
            home_team_id=1,
            away_team_id=2,
            date="2026-06-18",
            ratings=ratings,
            home_elo=1100.0,
            away_elo=1000.0,
            goal_probs=goal_probs,
            goal_model_xg={"home_xg": 1.8, "away_xg": 0.9},
        )

        assert result["_goal_model_xg"] == {"home_xg": 1.8, "away_xg": 0.9}

    def test_goal_model_low_data_flag_preserved(self) -> None:
        """When goal_model_low_data is True, it appears in the result."""
        from soccer_ev_model.ev_workflow import predict_match

        ratings = self._make_ratings()
        goal_probs = {"home": 0.45, "draw": 0.25, "away": 0.30}

        result = predict_match(
            home_team="Home",
            away_team="Away",
            home_team_id=1,
            away_team_id=2,
            date="2026-06-18",
            ratings=ratings,
            home_elo=1100.0,
            away_elo=1000.0,
            goal_probs=goal_probs,
            goal_model_low_data=True,
        )

        assert result["_goal_model_low_data"] is True


# --------------------------------------------------------------------------- #
# prediction_why_text: goal-model-aware explanations
# --------------------------------------------------------------------------- #
class TestPredictionWhyTextGoalModel:
    """Verify the new goal-model-aware why text variants."""

    def _make_result(
        self,
        primary_probs=None,
        pi_probs=None,
        elo_only_probs=None,
        goal_model_used=False,
        goal_model_expected=False,
        goal_model_xg=None,
        goal_model_low_data=False,
        blend_was_used=False,
        tier="B",
        agreement="agree",
    ):
        pi_probs = pi_probs or {"home": 0.45, "draw": 0.25, "away": 0.30}
        primary_probs = primary_probs or dict(pi_probs)
        return {
            "home_team": "England",
            "away_team": "Croatia",
            "primary_probs": primary_probs,
            "pi_probs": pi_probs,
            "blend_probs": dict(pi_probs),
            "pi_only_probs": dict(pi_probs),
            "elo_only_probs": elo_only_probs,
            "blend_was_used": blend_was_used,
            "_goal_model_used": goal_model_used,
            "_goal_model_expected": goal_model_expected,
            "_goal_model_xg": goal_model_xg,
            "_goal_model_low_data": goal_model_low_data,
            "confidence": {
                "tier": tier,
                "low_data": False,
                "warnings": [],
            },
        }

    def test_goal_model_low_data_returns_limited_history(self) -> None:
        """When goal model ran with low data, why text mentions limited history."""
        result = self._make_result(
            goal_model_used=True,
            goal_model_low_data=True,
            elo_only_probs={"home": 0.50, "draw": 0.25, "away": 0.25},
        )
        text = prediction_why_text(result, warnings=[], identity_warnings=[])
        assert text == "Limited goal-model history is available for this matchup."

    def test_goal_model_agreement_returns_both_favor(self) -> None:
        """When goal model + Elo agree, why text mentions both favoring the team."""
        result = self._make_result(
            primary_probs={"home": 0.55, "draw": 0.20, "away": 0.25},
            pi_probs={"home": 0.45, "draw": 0.25, "away": 0.30},
            elo_only_probs={"home": 0.50, "draw": 0.25, "away": 0.25},
            goal_model_used=True,
            blend_was_used=True,
            tier="B",
            agreement="agree",
        )
        text = prediction_why_text(result, warnings=[], identity_warnings=[])
        assert "Elo and the goal model both favor England" in text

    def test_goal_model_agreement_draw_returns_draw_favor(self) -> None:
        """When goal model + Elo agree on a draw."""
        result = self._make_result(
            primary_probs={"home": 0.20, "draw": 0.55, "away": 0.25},
            pi_probs={"home": 0.20, "draw": 0.45, "away": 0.35},
            elo_only_probs={"home": 0.20, "draw": 0.50, "away": 0.30},
            goal_model_used=True,
            blend_was_used=True,
            tier="B",
            agreement="agree",
        )
        text = prediction_why_text(result, warnings=[], identity_warnings=[])
        assert "Elo and the goal model both favor a draw" in text

    def test_goal_model_xg_returns_expected_goals(self) -> None:
        """When goal model ran with xG, why text mentions expected goals."""
        # No Elo available (only_pi), so the "both favor" branch doesn't fire.
        # The xG branch (step 2c) fires instead.
        result = self._make_result(
            goal_model_used=True,
            goal_model_xg={"home_xg": 1.8, "away_xg": 0.9},
            elo_only_probs=None,
            blend_was_used=False,
            tier="B",
            agreement="only_pi",
        )
        text = prediction_why_text(result, warnings=[], identity_warnings=[])
        assert "Goal model projects 1.8-0.9 expected goals" in text

    def test_elo_only_fallback_returns_fallback_text(self) -> None:
        """When goal model expected but not used and Elo was, why text mentions fallback."""
        result = self._make_result(
            goal_model_used=False,
            goal_model_expected=True,
            blend_was_used=True,
            tier="B",
        )
        text = prediction_why_text(result, warnings=[], identity_warnings=[])
        assert text == "Elo-only fallback used."

    def test_goal_model_not_used_no_elo_returns_only_one_method(self) -> None:
        """When neither goal model nor Elo used, why text mentions single method."""
        result = self._make_result(
            primary_probs={"home": 0.34, "draw": 0.33, "away": 0.33},
            pi_probs={"home": 0.34, "draw": 0.33, "away": 0.33},
            goal_model_used=False,
            blend_was_used=False,
            elo_only_probs=None,
            tier="B",
        )
        text = prediction_why_text(result, warnings=[], identity_warnings=[])
        assert "Only one prediction method was available" in text

    def test_primary_probs_used_for_most_likely(self) -> None:
        """prediction_card._extract_most_likely uses primary_probs (blend result)."""
        from dashboard.prediction_card import _extract_most_likely

        # primary_probs (blend) says home, pi_probs says away
        pred = {
            "primary_probs": {"home": 0.51, "draw": 0.24, "away": 0.25},
            "blend_probs": {"home": 0.30, "draw": 0.30, "away": 0.40},
            "pi_probs": {"home": 0.30, "draw": 0.30, "away": 0.40},
        }
        assert _extract_most_likely(pred) == "home"

    def test_primary_probs_used_for_top_prob(self) -> None:
        """prediction_card._extract_top_prob uses primary_probs (blend result)."""
        from dashboard.prediction_card import _extract_top_prob

        pred = {
            "primary_probs": {"home": 0.513, "draw": 0.244, "away": 0.243},
            "blend_probs": {"home": 0.300, "draw": 0.300, "away": 0.400},
        }
        assert abs(_extract_top_prob(pred) - 0.513) < 1e-9
