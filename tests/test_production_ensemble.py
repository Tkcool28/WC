"""Tests for soccer_ev_model.production_ensemble.blend_ensemble.

Covers all 13 required validation scenarios plus the exact 60/40 math,
normalization, determinism, pi-diagnostic-only, and schema stability.
"""
from __future__ import annotations

import copy

import pytest

from soccer_ev_model.production_ensemble import (
    MODEL_NAME,
    WEIGHTS,
    blend_ensemble,
    EnsembleInputError,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _valid_elo(**overrides):
    base = {"home": 0.50, "draw": 0.25, "away": 0.25}
    base.update(overrides)
    return base


def _valid_goal(**overrides):
    base = {"home": 0.40, "draw": 0.30, "away": 0.30}
    base.update(overrides)
    return base


def _valid_pi(**overrides):
    base = {"home": 0.55, "draw": 0.20, "away": 0.25}
    base.update(overrides)
    return base


def _valid_details(**overrides):
    base = {
        "home_xg": 1.8,
        "away_xg": 1.2,
        "expected_total_goals": 3.0,
        "most_likely_score": [2, 1],
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# 1. Exact 60/40 blend
# --------------------------------------------------------------------------- #

def test_exact_6040_blend():
    elo = _valid_elo(home=0.60, draw=0.20, away=0.20)
    goal = _valid_goal(home=0.30, draw=0.40, away=0.30)

    result = blend_ensemble(elo, goal)

    # primary = 0.6 * elo + 0.4 * goal, then normalize
    raw_home = 0.60 * 0.60 + 0.40 * 0.30  # = 0.48
    raw_draw = 0.60 * 0.20 + 0.40 * 0.40  # = 0.28
    raw_away = 0.60 * 0.20 + 0.40 * 0.30  # = 0.24
    total = raw_home + raw_draw + raw_away  # = 1.0

    assert result["primary_probs"]["home"] == pytest.approx(raw_home / total)
    assert result["primary_probs"]["draw"] == pytest.approx(raw_draw / total)
    assert result["primary_probs"]["away"] == pytest.approx(raw_away / total)


def test_exact_6040_blend_unnormalized_inputs():
    """Unnormalized but valid inputs should be normalized before blending."""
    elo = _valid_elo(home=0.60, draw=0.20, away=0.20)
    goal = _valid_goal(home=0.30, draw=0.40, away=0.30)

    # Pass unnormalized goal (sums to 2.0)
    goal_unnorm = {"home": 0.60, "draw": 0.80, "away": 0.60}
    result = blend_ensemble(elo, goal_unnorm)

    # After normalization, goal becomes {home: 0.30, draw: 0.40, away: 0.30}
    # so result should match the exact test above
    assert result["primary_probs"]["home"] == pytest.approx(0.48)
    assert result["primary_probs"]["draw"] == pytest.approx(0.28)
    assert result["primary_probs"]["away"] == pytest.approx(0.24)


# --------------------------------------------------------------------------- #
# 2. Final normalization
# --------------------------------------------------------------------------- #

def test_final_primary_probs_sum_to_one():
    elo = _valid_elo(home=0.45, draw=0.30, away=0.25)
    goal = _valid_goal(home=0.35, draw=0.35, away=0.30)

    result = blend_ensemble(elo, goal)
    total = sum(result["primary_probs"].values())
    assert total == pytest.approx(1.0, abs=1e-9)


def test_final_primary_probs_all_finite():
    elo = _valid_elo()
    goal = _valid_goal()

    result = blend_ensemble(elo, goal)
    for k, v in result["primary_probs"].items():
        assert isinstance(v, float)
        assert v >= 0.0
        # No NaN or inf
        assert v == v  # NaN != NaN
        assert abs(v) != float("inf")


# --------------------------------------------------------------------------- #
# 3. Deterministic repeated output
# --------------------------------------------------------------------------- #

def test_deterministic_repeated_output():
    elo = _valid_elo()
    goal = _valid_goal()

    r1 = blend_ensemble(elo, goal)
    r2 = blend_ensemble(elo, goal)
    r3 = blend_ensemble(elo, goal)

    assert r1 == r2 == r3


# --------------------------------------------------------------------------- #
# 4. Pi does not affect primary probabilities
# --------------------------------------------------------------------------- #

def test_pi_does_not_affect_primary():
    elo = _valid_elo()
    goal = _valid_goal()
    pi = _valid_pi(home=0.90, draw=0.05, away=0.05)

    without_pi = blend_ensemble(elo, goal)
    with_pi = blend_ensemble(elo, goal, pi_probs=pi)

    for k in ("home", "draw", "away"):
        assert without_pi["primary_probs"][k] == pytest.approx(
            with_pi["primary_probs"][k], abs=1e-12
        )


def test_extreme_pi_does_not_affect_primary():
    """Even an extreme pi vector must not change primary_probs."""
    elo = _valid_elo()
    goal = _valid_goal()

    without = blend_ensemble(elo, goal)
    extreme_pi = {"home": 0.99, "draw": 0.005, "away": 0.005}
    with_extreme = blend_ensemble(elo, goal, pi_probs=extreme_pi)

    for k in ("home", "draw", "away"):
        assert without["primary_probs"][k] == pytest.approx(
            with_extreme["primary_probs"][k], abs=1e-12
        )


# --------------------------------------------------------------------------- #
# 5. Component values remain separately available
# --------------------------------------------------------------------------- #

def test_component_values_separately_available():
    elo = _valid_elo(home=0.50, draw=0.25, away=0.25)
    goal = _valid_goal(home=0.40, draw=0.30, away=0.30)
    pi = _valid_pi()

    result = blend_ensemble(elo, goal, pi_probs=pi)

    assert result["elo_probs"] == elo
    assert result["goal_probs"] == goal
    assert result["pi_probs"] == pi


def test_goal_details_passed_through():
    details = _valid_details()
    result = blend_ensemble(_valid_elo(), _valid_goal(), goal_details=details)
    assert result["goal_details"] == details


# --------------------------------------------------------------------------- #
# 6. Input dictionaries are not mutated
# --------------------------------------------------------------------------- #

def test_input_dicts_not_mutated():
    elo = _valid_elo()
    goal = _valid_goal()
    pi = _valid_pi()
    details = _valid_details()

    elo_copy = copy.deepcopy(elo)
    goal_copy = copy.deepcopy(goal)
    pi_copy = copy.deepcopy(pi)
    details_copy = copy.deepcopy(details)

    blend_ensemble(elo, goal, pi_probs=pi, goal_details=details)

    assert elo == elo_copy
    assert goal == goal_copy
    assert pi == pi_copy
    assert details == details_copy


# --------------------------------------------------------------------------- #
# 7. Negative probability rejection
# --------------------------------------------------------------------------- #

def test_negative_probability_rejected_in_elo():
    with pytest.raises(EnsembleInputError, match="negative"):
        blend_ensemble({"home": -0.1, "draw": 0.6, "away": 0.5}, _valid_goal())


def test_negative_probability_rejected_in_goal():
    with pytest.raises(EnsembleInputError, match="negative"):
        blend_ensemble(_valid_elo(), {"home": 0.5, "draw": -0.1, "away": 0.6})


# --------------------------------------------------------------------------- #
# 8. NaN rejection
# --------------------------------------------------------------------------- #

def test_nan_rejected_in_elo():
    with pytest.raises(EnsembleInputError, match="NaN"):
        blend_ensemble({"home": float("nan"), "draw": 0.5, "away": 0.5}, _valid_goal())


def test_nan_rejected_in_goal():
    with pytest.raises(EnsembleInputError, match="NaN"):
        blend_ensemble(_valid_elo(), {"home": 0.5, "draw": 0.5, "away": float("nan")})


# --------------------------------------------------------------------------- #
# 9. Infinity rejection
# --------------------------------------------------------------------------- #

def test_positive_infinity_rejected():
    with pytest.raises(EnsembleInputError, match="infinity"):
        blend_ensemble(
            {"home": float("inf"), "draw": 0.0, "away": 0.0}, _valid_goal()
        )


def test_negative_infinity_rejected():
    with pytest.raises(EnsembleInputError, match="infinity"):
        blend_ensemble(
            {"home": float("-inf"), "draw": 0.5, "away": 0.5}, _valid_goal()
        )


# --------------------------------------------------------------------------- #
# 10. Missing-key rejection
# --------------------------------------------------------------------------- #

def test_missing_home_key_rejected():
    with pytest.raises(EnsembleInputError, match="missing keys"):
        blend_ensemble({"draw": 0.5, "away": 0.5}, _valid_goal())


def test_missing_draw_key_rejected():
    with pytest.raises(EnsembleInputError, match="missing keys"):
        blend_ensemble({"home": 0.5, "away": 0.5}, _valid_goal())


def test_missing_away_key_rejected():
    with pytest.raises(EnsembleInputError, match="missing keys"):
        blend_ensemble({"home": 0.5, "draw": 0.5}, _valid_goal())


def test_empty_dict_rejected():
    with pytest.raises(EnsembleInputError, match="missing keys"):
        blend_ensemble({}, _valid_goal())


# --------------------------------------------------------------------------- #
# 11. Zero-total rejection
# --------------------------------------------------------------------------- #

def test_zero_total_rejected_in_elo():
    with pytest.raises(EnsembleInputError, match="total probability"):
        blend_ensemble({"home": 0.0, "draw": 0.0, "away": 0.0}, _valid_goal())


def test_zero_total_rejected_in_goal():
    with pytest.raises(EnsembleInputError, match="total probability"):
        blend_ensemble(_valid_elo(), {"home": 0.0, "draw": 0.0, "away": 0.0})


# --------------------------------------------------------------------------- #
# 12. Stable output schema
# --------------------------------------------------------------------------- #

EXPECTED_TOP_LEVEL_KEYS = {
    "primary_probs",
    "elo_probs",
    "goal_probs",
    "pi_probs",
    "goal_details",
    "model_name",
    "weights",
    "fallback_used",
    "warnings",
}

EXPECTED_PROB_KEYS = {"home", "draw", "away"}


def test_stable_output_schema_no_pi_no_details():
    result = blend_ensemble(_valid_elo(), _valid_goal())

    assert set(result.keys()) == EXPECTED_TOP_LEVEL_KEYS
    assert set(result["primary_probs"].keys()) == EXPECTED_PROB_KEYS
    assert set(result["elo_probs"].keys()) == EXPECTED_PROB_KEYS
    assert set(result["goal_probs"].keys()) == EXPECTED_PROB_KEYS
    assert result["pi_probs"] is None
    assert result["goal_details"] is None
    assert result["fallback_used"] is None
    assert result["warnings"] == []


def test_stable_output_schema_with_pi_and_details():
    pi = _valid_pi()
    details = _valid_details()
    warnings = ["low_data"]

    result = blend_ensemble(
        _valid_elo(), _valid_goal(),
        pi_probs=pi, goal_details=details, warnings=warnings,
    )

    assert set(result.keys()) == EXPECTED_TOP_LEVEL_KEYS
    assert set(result["pi_probs"].keys()) == EXPECTED_PROB_KEYS
    assert result["goal_details"] == details
    assert result["warnings"] == ["low_data"]


# --------------------------------------------------------------------------- #
# 13. model_name and weights are exact
# --------------------------------------------------------------------------- #

def test_model_name_is_exact():
    result = blend_ensemble(_valid_elo(), _valid_goal())
    assert result["model_name"] == "elo60_goal40"


def test_weights_are_exact():
    result = blend_ensemble(_valid_elo(), _valid_goal())
    assert result["weights"] == {"elo": 0.60, "goal": 0.40}
    assert result["weights"]["elo"] == pytest.approx(0.60)
    assert result["weights"]["goal"] == pytest.approx(0.40)


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #

def test_non_dict_input_rejected():
    with pytest.raises(EnsembleInputError, match="expected dict"):
        blend_ensemble("not a dict", _valid_goal())


def test_non_numeric_value_rejected():
    with pytest.raises(EnsembleInputError, match="expected numeric"):
        blend_ensemble({"home": "high", "draw": 0.5, "away": 0.5}, _valid_goal())


def test_goal_details_non_dict_rejected():
    with pytest.raises(EnsembleInputError, match="expected dict"):
        blend_ensemble(_valid_elo(), _valid_goal(), goal_details="bad")


def test_warnings_are_copied_not_aliased():
    """Mutating the returned warnings list must not affect the input."""
    src = ["warn1"]
    result = blend_ensemble(_valid_elo(), _valid_goal(), warnings=src)
    result["warnings"].append("warn2")
    assert src == ["warn1"]


def test_integer_probabilities_accepted():
    """Integer values (0, 1) should be accepted as numeric."""
    elo = {"home": 1, "draw": 0, "away": 0}  # sums to 1
    goal = {"home": 0, "draw": 0, "away": 1}  # sums to 1
    result = blend_ensemble(elo, goal)
    # primary = 0.6 * {1,0,0} + 0.4 * {0,0,1} = {0.6, 0, 0.4}
    assert result["primary_probs"]["home"] == pytest.approx(0.6)
    assert result["primary_probs"]["draw"] == pytest.approx(0.0)
    assert result["primary_probs"]["away"] == pytest.approx(0.4)
