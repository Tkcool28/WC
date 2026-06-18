"""
Tests for the fallback and low-data policy (Phase 3).

Covers all five fallback branches (A–E), edge cases, and the required
test teams: Cape Verde, DR Congo, Curacao, one fully covered major team,
and one unseen-team fixture.

Verification goals:
  - Every fallback branch is tested.
  - Warnings are explicit (non-empty for B–E, empty for A).
  - No silent weight change — weights always match the resolved case.
  - No crash on missing team (Elo or Goal).
  - No divide-by-zero in blend helper.
  - Pi is diagnostic only — never influences primary_probs.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Ensure the WC package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soccer_ev_model.blend_fallback import (
    BlendResult,
    ModelAvailability,
    _blend_probs,
    _check_goal_availability,
    predict_with_fallback,
    resolve_blend_weights,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _elo_probs(home_elo: float = 1700, away_elo: float = 1500) -> dict[str, float]:
    """Simple Elo-only probability via the logistic mapping (same as production)."""
    from soccer_ev_model.ev_workflow import _logistic_matchup
    diff = (home_elo - away_elo) / 400.0
    return _logistic_matchup(diff)


def _goal_probs(home: float = 0.55, draw: float = 0.25, away: float = 0.20) -> dict[str, float]:
    """Synthetic goal model H/D/A probs."""
    return {"home": home, "draw": draw, "away": away}


def _pi_probs(home: float = 0.50, draw: float = 0.30, away: float = 0.20) -> dict[str, float]:
    """Synthetic pi-rating H/D/A probs (diagnostic only)."""
    return {"home": home, "draw": draw, "away": away}


def _make_goal_model(
    home_count: int = 50,
    away_count: int = 50,
    home_id: int = 1,
    away_id: int = 2,
) -> SimpleNamespace:
    """Build a minimal goal-model-like object with a counts dict."""
    return SimpleNamespace(
        counts={str(home_id): home_count, str(away_id): away_count}
    )


# ===================================================================
# Case A: Elo valid + Goal valid (sufficient data) → 60/40, no warning
# ===================================================================

class TestCaseA:
    def test_weights_are_60_40(self):
        w_elo, w_goal, case, warnings = resolve_blend_weights(
            elo_available=True, goal_available=True, goal_low_data=False,
        )
        assert case == "A"
        assert w_elo == 0.6
        assert w_goal == 0.4
        assert warnings == []

    def test_blend_is_weighted_average(self):
        elo = _elo_probs(1700, 1500)
        goal = _goal_probs(0.55, 0.25, 0.20)
        blended = _blend_probs(elo, goal, 0.6, 0.4)
        for k in ("home", "draw", "away"):
            expected = 0.6 * elo[k] + 0.4 * goal[k]
            assert abs(blended[k] - expected) < 1e-6, f"mismatch on {k}"

    def test_blend_sums_to_one(self):
        elo = _elo_probs(1700, 1500)
        goal = _goal_probs(0.55, 0.25, 0.20)
        blended = _blend_probs(elo, goal, 0.6, 0.4)
        assert abs(sum(blended.values()) - 1.0) < 1e-6

    def test_predict_with_fallback_case_a(self):
        elo = _elo_probs(1700, 1500)
        goal = _goal_probs(0.55, 0.25, 0.20)
        result = predict_with_fallback(
            home_team="Argentina", away_team="Brazil",
            home_team_id=1, away_team_id=2,
            match_date="2026-07-01",
            elo_probs=elo, goal_probs=goal,
            elo_available=True, goal_available=True, goal_low_data=False,
        )
        assert result.case == "A"
        assert result.w_elo == 0.6
        assert result.w_goal == 0.4
        assert result.warnings == []
        assert result.elo_probs is not None
        assert result.goal_probs is not None
        assert abs(sum(result.primary_probs.values()) - 1.0) < 1e-4

    def test_fully_covered_major_team(self):
        """Simulate a well-known team (e.g. Argentina) with full data."""
        elo = _elo_probs(2000, 1800)  # Strong Elo favourite
        goal = _goal_probs(0.60, 0.22, 0.18)
        result = predict_with_fallback(
            home_team="Argentina", away_team="Mexico",
            home_team_id=10, away_team_id=20,
            match_date="2026-06-15",
            elo_probs=elo, goal_probs=goal,
            elo_available=True, goal_available=True, goal_low_data=False,
        )
        assert result.case == "A"
        assert result.warnings == []
        # Blend should be between the two inputs
        for k in ("home", "draw", "away"):
            lo = min(elo[k], goal[k])
            hi = max(elo[k], goal[k])
            assert lo <= result.primary_probs[k] <= hi, (
                f"primary_probs[{k}]={result.primary_probs[k]} not in [{lo}, {hi}]"
            )


# ===================================================================
# Case B: Goal valid but low-data → 60/40, warning about Goal coverage
# ===================================================================

class TestCaseB:
    def test_weights_are_60_40_with_warning(self):
        w_elo, w_goal, case, warnings = resolve_blend_weights(
            elo_available=True, goal_available=True, goal_low_data=True,
        )
        assert case == "B"
        assert w_elo == 0.6
        assert w_goal == 0.4
        assert len(warnings) > 0
        assert "Goal model has limited historical coverage" in warnings[0]

    def test_predict_with_fallback_case_b(self):
        elo = _elo_probs(1700, 1500)
        goal = _goal_probs(0.55, 0.25, 0.20)
        result = predict_with_fallback(
            home_team="Cape Verde", away_team="DR Congo",
            home_team_id=100, away_team_id=101,
            match_date="2026-06-20",
            elo_probs=elo, goal_probs=goal,
            elo_available=True, goal_available=True, goal_low_data=True,
        )
        assert result.case == "B"
        assert result.w_elo == 0.6
        assert result.w_goal == 0.4
        assert len(result.warnings) == 1
        assert "Goal model has limited historical coverage" in result.warnings[0]
        assert abs(sum(result.primary_probs.values()) - 1.0) < 1e-4

    def test_cape_verde_low_data_scenario(self):
        """Cape Verde: Elo available, goal model has <10 matches."""
        model = _make_goal_model(home_count=6, away_count=50, home_id=200, away_id=201)
        home_a, away_a = _check_goal_availability(model, 200, 201)
        assert home_a.available is True
        assert home_a.low_data is True
        assert away_a.available is True
        assert away_a.low_data is False

        w_elo, w_goal, case, warnings = resolve_blend_weights(
            elo_available=True, goal_available=True, goal_low_data=True,
        )
        assert case == "B"

    def test_dr_congo_low_data_scenario(self):
        """DR Congo: Elo available, goal model has <10 matches."""
        model = _make_goal_model(home_count=50, away_count=7, home_id=201, away_id=202)
        home_a, away_a = _check_goal_availability(model, 201, 202)
        assert home_a.available is True
        assert home_a.low_data is False
        assert away_a.available is True
        assert away_a.low_data is True

    def test_curacao_low_data_scenario(self):
        """Curacao: Elo available, goal model has <10 matches."""
        model = _make_goal_model(home_count=3, away_count=40, home_id=203, away_id=204)
        home_a, away_a = _check_goal_availability(model, 203, 204)
        assert home_a.available is True
        assert home_a.low_data is True


# ===================================================================
# Case C: Goal unavailable → 100% Elo, warning
# ===================================================================

class TestCaseC:
    def test_weights_are_100_elo(self):
        w_elo, w_goal, case, warnings = resolve_blend_weights(
            elo_available=True, goal_available=False, goal_low_data=False,
        )
        assert case == "C"
        assert w_elo == 1.0
        assert w_goal == 0.0
        assert len(warnings) > 0
        assert "Goal model unavailable" in warnings[0]

    def test_predict_with_fallback_case_c(self):
        elo = _elo_probs(1700, 1500)
        result = predict_with_fallback(
            home_team="Argentina", away_team="Curacao",
            home_team_id=10, away_team_id=999,
            match_date="2026-06-25",
            elo_probs=elo, goal_probs=None,
            elo_available=True, goal_available=False,
        )
        assert result.case == "C"
        assert result.w_elo == 1.0
        assert result.w_goal == 0.0
        assert result.goal_probs is None
        assert result.elo_probs is not None
        # primary_probs must equal elo_probs exactly
        for k in ("home", "draw", "away"):
            assert result.primary_probs[k] == result.elo_probs[k]
        assert "Goal model unavailable" in result.warnings[0]

    def test_goal_unseen_team(self):
        """A team with 0 matches in goal model counts → unavailable."""
        model = _make_goal_model(home_count=50, away_count=0, home_id=1, away_id=999)
        home_a, away_a = _check_goal_availability(model, 1, 999)
        assert home_a.available is True
        assert away_a.available is False
        assert "unseen" in away_a.reason.lower() or "0 training" in away_a.reason

    def test_no_crash_on_missing_goal_probs(self):
        """Must not crash when goal_probs is None and goal_available=False."""
        elo = _elo_probs(1700, 1500)
        result = predict_with_fallback(
            home_team="X", away_team="Y",
            home_team_id=1, away_team_id=2,
            match_date="2026-01-01",
            elo_probs=elo, goal_probs=None,
            elo_available=True, goal_available=False,
        )
        assert abs(sum(result.primary_probs.values()) - 1.0) < 1e-4


# ===================================================================
# Case D: Elo unavailable → 100% Goal, warning
# ===================================================================

class TestCaseD:
    def test_weights_are_100_goal(self):
        w_elo, w_goal, case, warnings = resolve_blend_weights(
            elo_available=False, goal_available=True, goal_low_data=False,
        )
        assert case == "D"
        assert w_elo == 0.0
        assert w_goal == 1.0
        assert len(warnings) > 0
        assert "Elo unavailable" in warnings[0]

    def test_predict_with_fallback_case_d(self):
        goal = _goal_probs(0.55, 0.25, 0.20)
        result = predict_with_fallback(
            home_team="Curacao", away_team="Cape Verde",
            home_team_id=203, away_team_id=200,
            match_date="2026-06-20",
            elo_probs=None, goal_probs=goal,
            elo_available=False, goal_available=True,
        )
        assert result.case == "D"
        assert result.w_elo == 0.0
        assert result.w_goal == 1.0
        assert result.elo_probs is None
        assert result.goal_probs is not None
        # primary_probs must equal goal_probs exactly
        for k in ("home", "draw", "away"):
            assert result.primary_probs[k] == result.goal_probs[k]
        assert "Elo unavailable" in result.warnings[0]

    def test_no_crash_on_missing_elo_probs(self):
        """Must not crash when elo_probs is None and elo_available=False."""
        goal = _goal_probs(0.55, 0.25, 0.20)
        result = predict_with_fallback(
            home_team="X", away_team="Y",
            home_team_id=1, away_team_id=2,
            match_date="2026-01-01",
            elo_probs=None, goal_probs=goal,
            elo_available=False, goal_available=True,
        )
        assert abs(sum(result.primary_probs.values()) - 1.0) < 1e-4


# ===================================================================
# Case E: Both unavailable → uniform baseline, warning
# ===================================================================

class TestCaseE:
    def test_weights_are_zero(self):
        w_elo, w_goal, case, warnings = resolve_blend_weights(
            elo_available=False, goal_available=False, goal_low_data=False,
        )
        assert case == "E"
        assert w_elo == 0.0
        assert w_goal == 0.0
        assert len(warnings) > 0
        assert "Primary models unavailable" in warnings[0]

    def test_predict_with_fallback_case_e(self):
        result = predict_with_fallback(
            home_team="UnknownA", away_team="UnknownB",
            home_team_id=9998, away_team_id=9999,
            match_date="2026-06-15",
            elo_probs=None, goal_probs=None,
            elo_available=False, goal_available=False,
        )
        assert result.case == "E"
        assert result.w_elo == 0.0
        assert result.w_goal == 0.0
        assert result.elo_probs is None
        assert result.goal_probs is None
        # Uniform baseline
        for k in ("home", "draw", "away"):
            assert abs(result.primary_probs[k] - 1 / 3) < 1e-4
        assert "Primary models unavailable" in result.warnings[0]

    def test_unseen_team_fixture(self):
        """A fixture where neither team has any data in either model."""
        model = _make_goal_model(home_count=0, away_count=0, home_id=9998, away_id=9999)
        home_a, away_a = _check_goal_availability(model, 9998, 9999)
        assert home_a.available is False
        assert away_a.available is False

        result = predict_with_fallback(
            home_team="UnknownA", away_team="UnknownB",
            home_team_id=9998, away_team_id=9999,
            match_date="2026-06-15",
            elo_probs=None, goal_probs=None,
            elo_available=False, goal_available=False,
        )
        assert result.case == "E"
        assert abs(sum(result.primary_probs.values()) - 1.0) < 1e-4


# ===================================================================
# Edge cases: no divide-by-zero, no silent weight change
# ===================================================================

class TestBlendEdgeCases:
    def test_both_probs_none_returns_uniform(self):
        result = _blend_probs(None, None, 0.0, 0.0)
        for k in ("home", "draw", "away"):
            assert abs(result[k] - 1 / 3) < 1e-10

    def test_only_elo_returns_elo(self):
        elo = _elo_probs(1700, 1500)
        result = _blend_probs(elo, None, 0.0, 1.0)
        for k in ("home", "draw", "away"):
            assert result[k] == elo[k]

    def test_only_goal_returns_goal(self):
        goal = _goal_probs(0.55, 0.25, 0.20)
        result = _blend_probs(None, goal, 1.0, 0.0)
        for k in ("home", "draw", "away"):
            assert result[k] == goal[k]

    def test_zero_weights_with_both_probs_returns_uniform(self):
        """When both weights are 0, _blend_probs must not divide by zero."""
        elo = _elo_probs(1700, 1500)
        goal = _goal_probs(0.55, 0.25, 0.20)
        result = _blend_probs(elo, goal, 0.0, 0.0)
        for k in ("home", "draw", "away"):
            assert abs(result[k] - 1 / 3) < 1e-10

    def test_blend_preserves_ordering(self):
        """If Elo and Goal agree on the top market, blend must too."""
        elo = _elo_probs(2000, 1500)   # Strong home favourite
        goal = _goal_probs(0.65, 0.20, 0.15)  # Also home favourite
        blended = _blend_probs(elo, goal, 0.6, 0.4)
        elo_top = max(elo, key=elo.get)
        goal_top = max(goal, key=goal.get)
        blend_top = max(blended, key=blended.get)
        assert elo_top == goal_top == blend_top == "home"

    def test_blend_with_equal_inputs_returns_same(self):
        """If both models produce identical probs, blend must match."""
        probs = {"home": 0.50, "draw": 0.25, "away": 0.25}
        blended = _blend_probs(probs, probs, 0.6, 0.4)
        for k in ("home", "draw", "away"):
            assert abs(blended[k] - probs[k]) < 1e-10


# ===================================================================
# Pi is diagnostic only — never influences primary_probs
# ===================================================================

class TestPiDiagnosticOnly:
    def test_pi_does_not_affect_blend(self):
        elo = _elo_probs(1700, 1500)
        goal = _goal_probs(0.55, 0.25, 0.20)
        pi = _pi_probs(0.90, 0.05, 0.05)  # Extreme pi — should be ignored

        result_no_pi = predict_with_fallback(
            home_team="A", away_team="B",
            home_team_id=1, away_team_id=2,
            match_date="2026-01-01",
            elo_probs=elo, goal_probs=goal,
            elo_available=True, goal_available=True,
        )
        result_with_pi = predict_with_fallback(
            home_team="A", away_team="B",
            home_team_id=1, away_team_id=2,
            match_date="2026-01-01",
            elo_probs=elo, goal_probs=goal,
            elo_available=True, goal_available=True,
            pi_probs=pi,
        )
        # primary_probs must be identical regardless of pi
        for k in ("home", "draw", "away"):
            assert result_no_pi.primary_probs[k] == result_with_pi.primary_probs[k]
        # pi_probs should be present in the diagnostic output
        assert result_with_pi.pi_probs is not None
        assert result_with_pi.pi_probs["home"] == 0.90

    def test_pi_present_in_all_cases(self):
        """Pi can be attached in any case — it's always diagnostic."""
        pi = _pi_probs(0.70, 0.20, 0.10)
        for case_kwargs in [
            dict(elo_available=True, goal_available=True, goal_low_data=False),   # A
            dict(elo_available=True, goal_available=True, goal_low_data=True),    # B
            dict(elo_available=True, goal_available=False),                       # C
            dict(elo_available=False, goal_available=True),                       # D
            dict(elo_available=False, goal_available=False),                      # E
        ]:
            result = predict_with_fallback(
                home_team="A", away_team="B",
                home_team_id=1, away_team_id=2,
                match_date="2026-01-01",
                elo_probs=_elo_probs() if case_kwargs["elo_available"] else None,
                goal_probs=_goal_probs() if case_kwargs["goal_available"] else None,
                pi_probs=pi,
                **case_kwargs,
            )
            assert result.pi_probs is not None
            assert result.pi_probs["home"] == 0.70


# ===================================================================
# Availability checker tests
# ===================================================================

class TestGoalAvailability:
    def test_both_teams_well_covered(self):
        model = _make_goal_model(home_count=50, away_count=60)
        ha, aa = _check_goal_availability(model, 1, 2)
        assert ha.available and not ha.low_data
        assert aa.available and not aa.low_data

    def test_home_unseen(self):
        model = _make_goal_model(home_count=0, away_count=50)
        ha, aa = _check_goal_availability(model, 1, 2)
        assert not ha.available
        assert aa.available

    def test_away_unseen(self):
        model = _make_goal_model(home_count=50, away_count=0)
        ha, aa = _check_goal_availability(model, 1, 2)
        assert ha.available
        assert not aa.available

    def test_both_unseen(self):
        model = _make_goal_model(home_count=0, away_count=0)
        ha, aa = _check_goal_availability(model, 1, 2)
        assert not ha.available
        assert not aa.available

    def test_low_data_boundary(self):
        """Exactly at GOAL_MIN_MATCHES_LOW threshold."""
        from soccer_ev_model.blend_fallback import GOAL_MIN_MATCHES_LOW
        # At threshold - 1 → low_data
        model = _make_goal_model(home_count=GOAL_MIN_MATCHES_LOW - 1, away_count=50)
        ha, aa = _check_goal_availability(model, 1, 2)
        assert ha.available and ha.low_data

        # At threshold → NOT low_data
        model2 = _make_goal_model(home_count=GOAL_MIN_MATCHES_LOW, away_count=50)
        ha2, aa2 = _check_goal_availability(model2, 1, 2)
        assert ha2.available and not ha2.low_data

    def test_empty_model_counts(self):
        """Model with empty counts dict → both teams unseen."""
        model = SimpleNamespace(counts={})
        ha, aa = _check_goal_availability(model, 1, 2)
        assert not ha.available
        assert not aa.available


# ===================================================================
# BlendResult structure tests
# ===================================================================

class TestBlendResult:
    def test_all_fields_present(self):
        result = predict_with_fallback(
            home_team="A", away_team="B",
            home_team_id=1, away_team_id=2,
            match_date="2026-01-01",
            elo_probs=_elo_probs(), goal_probs=_goal_probs(),
            elo_available=True, goal_available=True,
        )
        assert isinstance(result, BlendResult)
        assert hasattr(result, "primary_probs")
        assert hasattr(result, "elo_probs")
        assert hasattr(result, "goal_probs")
        assert hasattr(result, "w_elo")
        assert hasattr(result, "w_goal")
        assert hasattr(result, "case")
        assert hasattr(result, "warnings")
        assert hasattr(result, "pi_probs")
        assert hasattr(result, "elo_available")
        assert hasattr(result, "goal_available")
        assert hasattr(result, "goal_low_data")

    def test_primary_probs_always_sum_to_one(self):
        """Exhaustive: every case must produce probs summing to 1.0."""
        cases = [
            (True, True, False),   # A
            (True, True, True),    # B
            (True, False, False),  # C
            (False, True, False),  # D
            (False, False, False), # E
        ]
        for elo_av, goal_av, goal_low in cases:
            result = predict_with_fallback(
                home_team="A", away_team="B",
                home_team_id=1, away_team_id=2,
                match_date="2026-01-01",
                elo_probs=_elo_probs() if elo_av else None,
                goal_probs=_goal_probs() if goal_av else None,
                elo_available=elo_av, goal_available=goal_av, goal_low_data=goal_low,
            )
            total = sum(result.primary_probs.values())
            assert abs(total - 1.0) < 1e-4, (
                f"case={result.case}: primary_probs sum={total}"
            )

    def test_weights_sum_to_one_for_blend_cases(self):
        """Cases A and B: w_elo + w_goal == 1.0."""
        for elo_av, goal_av, goal_low in [(True, True, False), (True, True, True)]:
            result = predict_with_fallback(
                home_team="A", away_team="B",
                home_team_id=1, away_team_id=2,
                match_date="2026-01-01",
                elo_probs=_elo_probs(), goal_probs=_goal_probs(),
                elo_available=elo_av, goal_available=goal_av, goal_low_data=goal_low,
            )
            assert abs(result.w_elo + result.w_goal - 1.0) < 1e-10

    def test_no_silent_weight_change(self):
        """Weights must always match the resolved case — never silently different."""
        # Case A
        r = predict_with_fallback(
            "A", "B", 1, 2, "2026-01-01",
            _elo_probs(), _goal_probs(),
            elo_available=True, goal_available=True, goal_low_data=False,
        )
        assert r.w_elo == 0.6 and r.w_goal == 0.4 and r.case == "A"

        # Case C
        r = predict_with_fallback(
            "A", "B", 1, 2, "2026-01-01",
            _elo_probs(), None,
            elo_available=True, goal_available=False,
        )
        assert r.w_elo == 1.0 and r.w_goal == 0.0 and r.case == "C"

        # Case D
        r = predict_with_fallback(
            "A", "B", 1, 2, "2026-01-01",
            None, _goal_probs(),
            elo_available=False, goal_available=True,
        )
        assert r.w_elo == 0.0 and r.w_goal == 1.0 and r.case == "D"

        # Case E
        r = predict_with_fallback(
            "A", "B", 1, 2, "2026-01-01",
            None, None,
            elo_available=False, goal_available=False,
        )
        assert r.w_elo == 0.0 and r.w_goal == 0.0 and r.case == "E"
