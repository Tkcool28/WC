"""
Phase 9A — Focused unit tests proving the Elo60/Goal40 integration contracts.

Five test categories:
  1. Ensemble math: exact 60/40 blend, normalization, H/D/A sum=1, no invalid probs
  2. Artifact loader: successful load, missing file, invalid version, cache reuse
  3. Fallbacks: all 5 cases A-E plus low-data warning
  4. Contract: primary_probs exists, Pi separate, components preserved, fallback metadata
  5. Identity resolution: schedule ID, canonical ID, corpus ID, COD/CPV edge cases

These tests are additive — they prove the integration contracts end-to-end
without duplicating the fine-grained unit tests already in test_production_ensemble.py,
test_blend_fallback_policy.py, test_goal_model_cached.py, test_prediction_contract.py,
and test_team_identity.py.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

_WC_ROOT = Path(__file__).resolve().parent.parent
if str(_WC_ROOT) not in sys.path:
    sys.path.insert(0, str(_WC_ROOT))

# ── Shared helpers ────────────────────────────────────────────────────────────

def _valid_elo(home=0.50, draw=0.25, away=0.25):
    return {"home": home, "draw": draw, "away": away}

def _valid_goal(home=0.40, draw=0.30, away=0.30):
    return {"home": home, "draw": draw, "away": away}

def _valid_pi(home=0.55, draw=0.20, away=0.25):
    return {"home": home, "draw": draw, "away": away}


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 1: Ensemble Math
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnsembleMath:
    """Verify the exact 60/40 blend, normalization, and probability validity."""

    def test_exact_6040_blend_with_simple_inputs(self):
        """Elo60/Goal40 with simple probabilities produces exact weighted average."""
        from soccer_ev_model.production_ensemble import blend_ensemble

        elo = _valid_elo(home=0.60, draw=0.20, away=0.20)
        goal = _valid_goal(home=0.30, draw=0.40, away=0.30)

        result = blend_ensemble(elo, goal)

        # raw = 0.6*elo + 0.4*goal = {0.48, 0.28, 0.24}, sum=1.0, no renormalization needed
        assert result["primary_probs"]["home"] == pytest.approx(0.48, abs=1e-9)
        assert result["primary_probs"]["draw"] == pytest.approx(0.28, abs=1e-9)
        assert result["primary_probs"]["away"] == pytest.approx(0.24, abs=1e-9)

    def test_hda_sum_exactly_one(self):
        """H/D/A probabilities must always sum to 1.0 within floating-point tolerance."""
        from soccer_ev_model.production_ensemble import blend_ensemble

        # Use inputs that will produce a non-trivial blend
        elo = _valid_elo(home=0.70, draw=0.15, away=0.15)
        goal = _valid_goal(home=0.20, draw=0.50, away=0.30)

        result = blend_ensemble(elo, goal)
        total = sum(result["primary_probs"].values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_no_invalid_probabilities_nan(self):
        """blend_ensemble must never produce NaN in primary_probs."""
        from soccer_ev_model.production_ensemble import blend_ensemble, EnsembleInputError

        # Valid inputs should never produce NaN
        result = blend_ensemble(_valid_elo(), _valid_goal())
        for k, v in result["primary_probs"].items():
            assert v == v, f"NaN detected in primary_probs[{k}]"

    def test_no_invalid_probabilities_negative(self):
        """blend_ensemble must never produce negative probabilities in primary_probs."""
        from soccer_ev_model.production_ensemble import blend_ensemble

        result = blend_ensemble(_valid_elo(), _valid_goal())
        for k, v in result["primary_probs"].items():
            assert v >= 0.0, f"Negative probability in primary_probs[{k}]: {v}"

    def test_no_invalid_probabilities_inf(self):
        """blend_ensemble must never produce infinity in primary_probs."""
        from soccer_ev_model.production_ensemble import blend_ensemble

        result = blend_ensemble(_valid_elo(), _valid_goal())
        for k, v in result["primary_probs"].items():
            assert abs(v) != float("inf"), f"Infinity in primary_probs[{k}]"

    def test_blend_with_unnormalized_inputs_normalizes_first(self):
        """Inputs that don't sum to 1.0 are normalized before blending."""
        from soccer_ev_model.production_ensemble import blend_ensemble

        elo = {"home": 6.0, "draw": 2.0, "away": 2.0}  # sums to 10 → {0.6, 0.2, 0.2}
        goal = {"home": 3.0, "draw": 4.0, "away": 3.0}  # sums to 10 → {0.3, 0.4, 0.3}

        result = blend_ensemble(elo, goal)
        # Same as normalized test: 0.6*{0.6,0.2,0.2} + 0.4*{0.3,0.4,0.3} = {0.48,0.28,0.24}
        assert result["primary_probs"]["home"] == pytest.approx(0.48, abs=1e-9)
        assert result["primary_probs"]["draw"] == pytest.approx(0.28, abs=1e-9)
        assert result["primary_probs"]["away"] == pytest.approx(0.24, abs=1e-9)

    def test_blend_is_deterministic(self):
        """Identical inputs must produce identical outputs every time."""
        from soccer_ev_model.production_ensemble import blend_ensemble

        elo = _valid_elo()
        goal = _valid_goal()

        results = [blend_ensemble(elo, goal) for _ in range(10)]
        for r in results[1:]:
            assert r == results[0]

    def test_blend_preserves_agreement_ordering(self):
        """If both models agree on the favorite, the blend must agree too."""
        from soccer_ev_model.production_ensemble import blend_ensemble

        elo = _valid_elo(home=0.70, draw=0.15, away=0.15)   # Elo says home
        goal = _valid_goal(home=0.65, draw=0.20, away=0.15)  # Goal says home

        result = blend_ensemble(elo, goal)
        assert result["primary_probs"]["home"] > result["primary_probs"]["away"]
        assert result["primary_probs"]["home"] > result["primary_probs"]["draw"]

    def test_blend_with_equal_inputs_returns_same(self):
        """If both models produce identical probabilities, the blend matches exactly."""
        from soccer_ev_model.production_ensemble import blend_ensemble

        probs = {"home": 0.50, "draw": 0.25, "away": 0.25}
        result = blend_ensemble(probs, probs)
        for k in ("home", "draw", "away"):
            assert result["primary_probs"][k] == pytest.approx(probs[k], abs=1e-9)

    def test_weights_are_exactly_60_40(self):
        """The WEIGHTS constant must be exactly 0.60/0.40."""
        from soccer_ev_model.production_ensemble import WEIGHTS
        assert WEIGHTS["elo"] == pytest.approx(0.60)
        assert WEIGHTS["goal"] == pytest.approx(0.40)
        assert WEIGHTS["elo"] + WEIGHTS["goal"] == pytest.approx(1.0)

    def test_model_name_is_elo60_goal40(self):
        """The MODEL_NAME constant must be 'elo60_goal40'."""
        from soccer_ev_model.production_ensemble import MODEL_NAME
        assert MODEL_NAME == "elo60_goal40"


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 2: Artifact Loader
# ═══════════════════════════════════════════════════════════════════════════════

class TestArtifactLoader:
    """Verify artifact loading: success, missing file, invalid version, cache reuse."""

    def test_successful_load_returns_predictor(self):
        """load_and_validate() returns a working GoalModelPredictor."""
        from soccer_ev_model.goal_model_cached import load_and_validate

        predictor = load_and_validate()
        assert predictor is not None
        assert predictor.artifact is not None
        assert len(predictor.artifact.attacks) > 0
        assert len(predictor.artifact.defenses) > 0

    def test_loaded_predictor_can_predict(self):
        """A loaded predictor produces valid H/D/A probabilities."""
        from soccer_ev_model.goal_model_cached import load_and_validate

        predictor = load_and_validate()
        sample_id = list(predictor.artifact.attacks.keys())[0]
        pred = predictor.predict(
            home_team_id=int(sample_id),
            away_team_id=int(sample_id),
            match_date="2026-06-18",
        )
        assert pred.home_xg > 0
        assert pred.away_xg > 0
        assert set(pred.hda_probs.keys()) == {"home", "draw", "away"}
        total = sum(pred.hda_probs.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_missing_file_raises_structured_error(self):
        """Missing artifact file produces GoalModelLoadError with reason='missing'."""
        from soccer_ev_model.goal_model_cached import GoalModelLoadError, load_and_validate

        with pytest.raises(GoalModelLoadError) as exc_info:
            load_and_validate(path="/nonexistent/path/goal_model.json")
        assert exc_info.value.reason == "missing"

    def test_invalid_version_raises_structured_error(self):
        """Wrong artifact_version produces GoalModelLoadError with reason='invalid_version'."""
        from soccer_ev_model.goal_model_cached import GoalModelLoadError, load_and_validate

        # Read the real artifact, modify version, write to temp
        art_path = _WC_ROOT / "data" / "artifacts" / "goal_model_sh5.json"
        data = json.loads(art_path.read_text(encoding="utf-8"))
        data["artifact_version"] = "wrong-version-999"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmp_path = f.name

        try:
            with pytest.raises(GoalModelLoadError) as exc_info:
                load_and_validate(path=tmp_path)
            assert exc_info.value.reason == "invalid_version"
            assert "wrong-version-999" in exc_info.value.message
        finally:
            Path(tmp_path).unlink()

    def test_malformed_missing_field_raises_structured_error(self):
        """Missing required field produces GoalModelLoadError with reason='malformed'."""
        from soccer_ev_model.goal_model_cached import GoalModelLoadError, load_and_validate

        art_path = _WC_ROOT / "data" / "artifacts" / "goal_model_sh5.json"
        data = json.loads(art_path.read_text(encoding="utf-8"))
        del data["attacks"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmp_path = f.name

        try:
            with pytest.raises(GoalModelLoadError) as exc_info:
                load_and_validate(path=tmp_path)
            assert exc_info.value.reason == "malformed"
        finally:
            Path(tmp_path).unlink()

    def test_session_cache_reuse_no_reload(self):
        """get_goal_predictor() loads artifact exactly once per session (cache reuse)."""
        from soccer_ev_model.goal_model_cached import (
            get_goal_predictor, get_load_count, reset_session_predictor,
        )

        fake_state = {}

        class FakeSessionState:
            def __init__(self):
                self._data = fake_state
            def get(self, key, default=None):
                return self._data.get(key, default)
            def pop(self, key, default=None):
                return self._data.pop(key, default)
            def __setitem__(self, key, value):
                self._data[key] = value

        with patch("streamlit.session_state", FakeSessionState()):
            reset_session_predictor()

            # First call loads
            pred1, err1 = get_goal_predictor()
            assert pred1 is not None
            assert err1 is None
            assert get_load_count() == 1

            # Second call returns cached (same object, no reload)
            pred2, err2 = get_goal_predictor()
            assert pred2 is pred1
            assert err2 is None
            assert get_load_count() == 1  # still 1

    def test_predictor_is_stateless_no_io_on_predict(self):
        """Prediction calls must not re-read the artifact file."""
        from soccer_ev_model.goal_model_cached import load_and_validate
        from pathlib import Path

        predictor = load_and_validate()
        sample_id = list(predictor.artifact.attacks.keys())[0]

        original_read = Path.read_text
        read_count = {"n": 0}

        def counting_read(self, *args, **kwargs):
            if "goal_model_sh5" in str(self):
                read_count["n"] += 1
            return original_read(self, *args, **kwargs)

        with patch.object(Path, "read_text", counting_read):
            for _ in range(5):
                predictor.predict(
                    home_team_id=int(sample_id),
                    away_team_id=int(sample_id),
                    match_date="2026-06-18",
                )

        assert read_count["n"] == 0, (
            f"Artifact was read {read_count['n']} times during prediction"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 3: Fallbacks — All 5 Cases A-E + Low-Data Warning
# ═══════════════════════════════════════════════════════════════════════════════

class TestFallbackCases:
    """Verify all 5 fallback cases (A-E) and the low-data warning."""

    def _make_goal_model(self, home_count=50, away_count=50, home_id=1, away_id=2):
        return SimpleNamespace(
            counts={str(home_id): home_count, str(away_id): away_count}
        )

    # ── Case A: Both valid, sufficient data ──

    def test_case_a_weights_60_40_no_warning(self):
        """Case A: Elo+Goal both valid → 60/40 blend, no warnings."""
        from soccer_ev_model.blend_fallback import resolve_blend_weights

        w_elo, w_goal, case, warnings = resolve_blend_weights(
            elo_available=True, goal_available=True, goal_low_data=False,
        )
        assert case == "A"
        assert w_elo == 0.6
        assert w_goal == 0.4
        assert warnings == []

    def test_case_a_predict_produces_blend(self):
        """Case A: predict_with_fallback produces 60/40 blend."""
        from soccer_ev_model.blend_fallback import predict_with_fallback

        elo = _valid_elo(home=0.60, draw=0.20, away=0.20)
        goal = _valid_goal(home=0.30, draw=0.40, away=0.30)

        result = predict_with_fallback(
            home_team="A", away_team="B",
            home_team_id=1, away_team_id=2,
            match_date="2026-01-01",
            elo_probs=elo, goal_probs=goal,
            elo_available=True, goal_available=True, goal_low_data=False,
        )
        assert result.case == "A"
        assert result.w_elo == 0.6
        assert result.w_goal == 0.4
        assert result.warnings == []
        assert abs(sum(result.primary_probs.values()) - 1.0) < 1e-4

    # ── Case B: Goal valid but low-data ──

    def test_case_b_weights_60_40_with_warning(self):
        """Case B: Goal low-data → 60/40 blend, warning about limited coverage."""
        from soccer_ev_model.blend_fallback import resolve_blend_weights

        w_elo, w_goal, case, warnings = resolve_blend_weights(
            elo_available=True, goal_available=True, goal_low_data=True,
        )
        assert case == "B"
        assert w_elo == 0.6
        assert w_goal == 0.4
        assert len(warnings) == 1
        assert "Goal model has limited historical coverage" in warnings[0]

    def test_case_b_predict_produces_blend_with_warning(self):
        """Case B: predict_with_fallback produces blend + low-data warning."""
        from soccer_ev_model.blend_fallback import predict_with_fallback

        elo = _valid_elo()
        goal = _valid_goal()

        result = predict_with_fallback(
            home_team="A", away_team="B",
            home_team_id=1, away_team_id=2,
            match_date="2026-01-01",
            elo_probs=elo, goal_probs=goal,
            elo_available=True, goal_available=True, goal_low_data=True,
        )
        assert result.case == "B"
        assert result.w_elo == 0.6
        assert result.w_goal == 0.4
        assert len(result.warnings) == 1
        assert "Goal model has limited historical coverage" in result.warnings[0]
        assert abs(sum(result.primary_probs.values()) - 1.0) < 1e-4

    def test_case_b_goal_availability_checker(self):
        """Goal model with <10 matches triggers low_data flag."""
        from soccer_ev_model.blend_fallback import _check_goal_availability

        model = self._make_goal_model(home_count=6, away_count=50, home_id=1, away_id=2)
        home_a, away_a = _check_goal_availability(model, 1, 2)
        assert home_a.available is True
        assert home_a.low_data is True
        assert away_a.available is True
        assert away_a.low_data is False

    # ── Case C: Goal unavailable → Elo-only ──

    def test_case_c_weights_100_elo(self):
        """Case C: Goal unavailable → 100% Elo, warning."""
        from soccer_ev_model.blend_fallback import resolve_blend_weights

        w_elo, w_goal, case, warnings = resolve_blend_weights(
            elo_available=True, goal_available=False, goal_low_data=False,
        )
        assert case == "C"
        assert w_elo == 1.0
        assert w_goal == 0.0
        assert len(warnings) == 1
        assert "Goal model unavailable" in warnings[0]

    def test_case_c_primary_equals_elo(self):
        """Case C: primary_probs must equal elo_probs exactly."""
        from soccer_ev_model.blend_fallback import predict_with_fallback

        elo = _valid_elo(home=0.60, draw=0.25, away=0.15)

        result = predict_with_fallback(
            home_team="A", away_team="B",
            home_team_id=1, away_team_id=2,
            match_date="2026-01-01",
            elo_probs=elo, goal_probs=None,
            elo_available=True, goal_available=False,
        )
        assert result.case == "C"
        assert result.goal_probs is None
        for k in ("home", "draw", "away"):
            assert result.primary_probs[k] == result.elo_probs[k]

    # ── Case D: Elo unavailable → Goal-only ──

    def test_case_d_weights_100_goal(self):
        """Case D: Elo unavailable → 100% Goal, warning."""
        from soccer_ev_model.blend_fallback import resolve_blend_weights

        w_elo, w_goal, case, warnings = resolve_blend_weights(
            elo_available=False, goal_available=True, goal_low_data=False,
        )
        assert case == "D"
        assert w_elo == 0.0
        assert w_goal == 1.0
        assert len(warnings) == 1
        assert "Elo unavailable" in warnings[0]

    def test_case_d_primary_equals_goal(self):
        """Case D: primary_probs must equal goal_probs exactly."""
        from soccer_ev_model.blend_fallback import predict_with_fallback

        goal = _valid_goal(home=0.55, draw=0.25, away=0.20)

        result = predict_with_fallback(
            home_team="A", away_team="B",
            home_team_id=1, away_team_id=2,
            match_date="2026-01-01",
            elo_probs=None, goal_probs=goal,
            elo_available=False, goal_available=True,
        )
        assert result.case == "D"
        assert result.elo_probs is None
        for k in ("home", "draw", "away"):
            assert result.primary_probs[k] == result.goal_probs[k]

    # ── Case E: Both unavailable → uniform baseline ──

    def test_case_e_weights_zero(self):
        """Case E: Both unavailable → 0/0 weights, warning."""
        from soccer_ev_model.blend_fallback import resolve_blend_weights

        w_elo, w_goal, case, warnings = resolve_blend_weights(
            elo_available=False, goal_available=False, goal_low_data=False,
        )
        assert case == "E"
        assert w_elo == 0.0
        assert w_goal == 0.0
        assert len(warnings) == 1
        assert "Primary models unavailable" in warnings[0]

    def test_case_e_uniform_baseline(self):
        """Case E: primary_probs must be uniform 1/3 each."""
        from soccer_ev_model.blend_fallback import predict_with_fallback

        result = predict_with_fallback(
            home_team="X", away_team="Y",
            home_team_id=9998, away_team_id=9999,
            match_date="2026-01-01",
            elo_probs=None, goal_probs=None,
            elo_available=False, goal_available=False,
        )
        assert result.case == "E"
        assert result.elo_probs is None
        assert result.goal_probs is None
        for k in ("home", "draw", "away"):
            assert abs(result.primary_probs[k] - 1 / 3) < 1e-4

    # ── Exhaustive: every case produces valid probabilities ──

    def test_all_cases_produce_valid_probabilities(self):
        """Every fallback case (A-E) must produce primary_probs summing to 1.0."""
        from soccer_ev_model.blend_fallback import predict_with_fallback

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
                elo_probs=_valid_elo() if elo_av else None,
                goal_probs=_valid_goal() if goal_av else None,
                elo_available=elo_av, goal_available=goal_av, goal_low_data=goal_low,
            )
            total = sum(result.primary_probs.values())
            assert abs(total - 1.0) < 1e-4, (
                f"case={result.case}: sum={total}"
            )

    def test_no_silent_weight_change(self):
        """Weights must always match the resolved case — never silently different."""
        from soccer_ev_model.blend_fallback import predict_with_fallback

        # Case A
        r = predict_with_fallback(
            "A", "B", 1, 2, "2026-01-01",
            _valid_elo(), _valid_goal(),
            elo_available=True, goal_available=True, goal_low_data=False,
        )
        assert r.w_elo == 0.6 and r.w_goal == 0.4 and r.case == "A"

        # Case C
        r = predict_with_fallback(
            "A", "B", 1, 2, "2026-01-01",
            _valid_elo(), None,
            elo_available=True, goal_available=False,
        )
        assert r.w_elo == 1.0 and r.w_goal == 0.0 and r.case == "C"

        # Case D
        r = predict_with_fallback(
            "A", "B", 1, 2, "2026-01-01",
            None, _valid_goal(),
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


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 4: Contract — primary_probs, Pi separate, components, fallback metadata
# ═══════════════════════════════════════════════════════════════════════════════

class TestContract:
    """Verify the integration contract: primary_probs exists, Pi separate, etc."""

    def test_primary_probs_exists_in_blend_output(self):
        """blend_ensemble must always include primary_probs in output."""
        from soccer_ev_model.production_ensemble import blend_ensemble

        result = blend_ensemble(_valid_elo(), _valid_goal())
        assert "primary_probs" in result
        assert set(result["primary_probs"].keys()) == {"home", "draw", "away"}

    def test_pi_probs_remains_separate_from_primary(self):
        """pi_probs must be preserved separately and never influence primary_probs."""
        from soccer_ev_model.production_ensemble import blend_ensemble

        elo = _valid_elo()
        goal = _valid_goal()
        pi_extreme = {"home": 0.99, "draw": 0.005, "away": 0.005}

        without_pi = blend_ensemble(elo, goal)
        with_pi = blend_ensemble(elo, goal, pi_probs=pi_extreme)

        # primary_probs must be identical regardless of pi
        for k in ("home", "draw", "away"):
            assert without_pi["primary_probs"][k] == pytest.approx(
                with_pi["primary_probs"][k], abs=1e-12
            )

        # pi_probs must be present in the output
        assert with_pi["pi_probs"] is not None
        assert with_pi["pi_probs"]["home"] == 0.99

    def test_component_probs_preserved(self):
        """elo_probs and goal_probs must be preserved separately in the output."""
        from soccer_ev_model.production_ensemble import blend_ensemble

        elo = _valid_elo(home=0.55, draw=0.25, away=0.20)
        goal = _valid_goal(home=0.35, draw=0.35, away=0.30)

        result = blend_ensemble(elo, goal)
        assert result["elo_probs"] == elo
        assert result["goal_probs"] == goal

    def test_fallback_metadata_preserved_in_blend_result(self):
        """BlendResult must carry case, warnings, and availability flags."""
        from soccer_ev_model.blend_fallback import predict_with_fallback

        result = predict_with_fallback(
            home_team="A", away_team="B",
            home_team_id=1, away_team_id=2,
            match_date="2026-01-01",
            elo_probs=_valid_elo(), goal_probs=_valid_goal(),
            elo_available=True, goal_available=True, goal_low_data=True,
        )
        # Case B metadata
        assert result.case == "B"
        assert result.elo_available is True
        assert result.goal_available is True
        assert result.goal_low_data is True
        assert len(result.warnings) == 1
        assert "Goal model has limited historical coverage" in result.warnings[0]

    def test_output_schema_stable_keys(self):
        """blend_ensemble output must always have exactly the expected top-level keys."""
        from soccer_ev_model.production_ensemble import blend_ensemble

        EXPECTED_KEYS = {
            "primary_probs", "elo_probs", "goal_probs", "pi_probs",
            "goal_details", "model_name", "weights", "fallback_used", "warnings",
        }
        result = blend_ensemble(_valid_elo(), _valid_goal())
        assert set(result.keys()) == EXPECTED_KEYS

    def test_fallback_used_is_none_in_production_ensemble(self):
        """production_ensemble.blend_ensemble must set fallback_used=None (not its responsibility)."""
        from soccer_ev_model.production_ensemble import blend_ensemble

        result = blend_ensemble(_valid_elo(), _valid_goal())
        assert result["fallback_used"] is None

    def test_blend_result_has_all_required_attributes(self):
        """BlendResult dataclass must carry all required fields."""
        from soccer_ev_model.blend_fallback import BlendResult, predict_with_fallback

        result = predict_with_fallback(
            home_team="A", away_team="B",
            home_team_id=1, away_team_id=2,
            match_date="2026-01-01",
            elo_probs=_valid_elo(), goal_probs=_valid_goal(),
            elo_available=True, goal_available=True,
        )
        assert isinstance(result, BlendResult)
        for attr in ("primary_probs", "elo_probs", "goal_probs", "w_elo", "w_goal",
                      "case", "warnings", "pi_probs", "elo_available",
                      "goal_available", "goal_low_data"):
            assert hasattr(result, attr), f"BlendResult missing attribute: {attr}"

    def test_pi_diagnostic_in_all_fallback_cases(self):
        """Pi can be attached in any fallback case — it's always diagnostic."""
        from soccer_ev_model.blend_fallback import predict_with_fallback

        pi = _valid_pi(0.70, 0.20, 0.10)
        cases = [
            dict(elo_available=True, goal_available=True, goal_low_data=False),   # A
            dict(elo_available=True, goal_available=True, goal_low_data=True),    # B
            dict(elo_available=True, goal_available=False),                       # C
            dict(elo_available=False, goal_available=True),                       # D
            dict(elo_available=False, goal_available=False),                      # E
        ]
        for case_kwargs in cases:
            result = predict_with_fallback(
                home_team="A", away_team="B",
                home_team_id=1, away_team_id=2,
                match_date="2026-01-01",
                elo_probs=_valid_elo() if case_kwargs["elo_available"] else None,
                goal_probs=_valid_goal() if case_kwargs["goal_available"] else None,
                pi_probs=pi,
                **case_kwargs,
            )
            assert result.pi_probs is not None
            assert result.pi_probs["home"] == 0.70

    def test_input_dicts_not_mutated(self):
        """blend_ensemble must not mutate caller-supplied input dicts."""
        from soccer_ev_model.production_ensemble import blend_ensemble
        import copy

        elo = _valid_elo()
        goal = _valid_goal()
        pi = _valid_pi()
        elo_copy = copy.deepcopy(elo)
        goal_copy = copy.deepcopy(goal)
        pi_copy = copy.deepcopy(pi)

        blend_ensemble(elo, goal, pi_probs=pi)

        assert elo == elo_copy
        assert goal == goal_copy
        assert pi == pi_copy


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 5: Identity Resolution — Schedule ID, Canonical ID, Corpus ID, COD/CPV
# ═══════════════════════════════════════════════════════════════════════════════

class TestIdentityResolution:
    """Verify identity resolution: schedule ID, canonical ID, corpus ID, COD/CPV edge cases."""

    def test_football_data_id_resolves_to_canonical(self):
        """football-data.org ID 762 (Argentina) resolves to canonical 'ARG'."""
        from soccer_ev_model.team_identity import canonical_id_for_football_data_id
        assert canonical_id_for_football_data_id(762) == "ARG"

    def test_corpus_id_resolves_to_canonical(self):
        """Corpus ID 710061511 (Argentina) resolves to canonical 'ARG'."""
        from soccer_ev_model.team_identity import canonical_id_for_corpus_id
        assert canonical_id_for_corpus_id(710061511) == "ARG"

    def test_name_resolves_to_canonical_case_insensitive(self):
        """Team name 'Argentina' (case-insensitive, whitespace-tolerant) resolves to 'ARG'."""
        from soccer_ev_model.team_identity import canonical_id_for_name
        assert canonical_id_for_name("Argentina") == "ARG"
        assert canonical_id_for_name("  argentina  ") == "ARG"
        assert canonical_id_for_name("ARGENTINA") == "ARG"

    def test_resolve_team_returns_resolved_with_corpus_id(self):
        """resolve_team returns 'resolved' status with correct corpus_id for known teams."""
        from soccer_ev_model.team_identity import resolve_team

        res = resolve_team(football_data_id=762, name="Argentina")
        assert res["canonical_id"] == "ARG"
        assert res["status"] == "resolved"
        assert res["corpus_id"] == 710061511
        assert res["display_name"] == "Argentina"
        assert res["source"] == "football_data"

    def test_resolve_team_unknown_returns_unresolved(self):
        """resolve_team returns 'identity_unresolved' for unknown inputs."""
        from soccer_ev_model.team_identity import resolve_team

        res = resolve_team(football_data_id=999999999, name="Atlantis")
        assert res["canonical_id"] is None
        assert res["status"] == "identity_unresolved"
        assert res["corpus_id"] is None
        assert res["display_name"] is None

    def test_cape_verde_history_missing(self):
        """Cape Verde (CPV) resolves to canonical ID but with history_missing status."""
        from soccer_ev_model.team_identity import resolve_team

        res = resolve_team(football_data_id=1930, name="Cape Verde Islands")
        assert res["canonical_id"] == "CPV"
        assert res["status"] == "history_missing"
        assert res["corpus_id"] is None
        assert res["display_name"] == "Cape Verde Islands"

    def test_congo_dr_history_missing(self):
        """DR Congo (COD) resolves to canonical ID but with history_missing status."""
        from soccer_ev_model.team_identity import resolve_team

        res = resolve_team(football_data_id=1934, name="Congo DR")
        assert res["canonical_id"] == "COD"
        assert res["status"] == "history_missing"
        assert res["corpus_id"] is None

    def test_curacao_history_missing(self):
        """Curacao (CUW) resolves to canonical ID but with history_missing status."""
        from soccer_ev_model.team_identity import resolve_team

        res = resolve_team(football_data_id=9460, name="Curaçao")
        assert res["canonical_id"] == "CUW"
        assert res["status"] == "history_missing"
        assert res["corpus_id"] is None

    def test_corpus_id_for_canonical_arg(self):
        """corpus_id_for_canonical returns the correct integer for 'ARG'."""
        from soccer_ev_model.team_identity import corpus_id_for_canonical
        assert corpus_id_for_canonical("ARG") == 710061511

    def test_display_name_for_canonical(self):
        """display_name returns the human-readable name for a canonical ID."""
        from soccer_ev_model.team_identity import display_name
        assert display_name("ARG") == "Argentina"

    def test_all_canonical_ids_includes_major_teams(self):
        """all_canonical_ids includes at least 48 entries with ARG, USA, BRA, ALG."""
        from soccer_ev_model.team_identity import all_canonical_ids

        ids = all_canonical_ids()
        assert len(ids) >= 48
        for expected in ("ARG", "USA", "BRA", "ALG"):
            assert expected in ids, f"Missing canonical ID: {expected}"

    def test_identity_unresolved_with_no_inputs(self):
        """resolve_team with no inputs returns identity_unresolved."""
        from soccer_ev_model.team_identity import resolve_team

        res = resolve_team()
        assert res["canonical_id"] is None
        assert res["status"] == "identity_unresolved"

    def test_corpus_id_for_history_missing_team_is_none(self):
        """corpus_id_for_canonical returns None for history-missing teams (CPV, COD, CUW)."""
        from soccer_ev_model.team_identity import corpus_id_for_canonical

        # These teams have canonical IDs but no corpus_id (history_missing)
        for cid in ("CPV", "COD", "CUW"):
            assert corpus_id_for_canonical(cid) is None, (
                f"{cid} should have None corpus_id (history_missing)"
            )

    def test_football_data_id_priority_over_name(self):
        """football_data_id takes priority over name in resolution."""
        from soccer_ev_model.team_identity import resolve_team

        # fd_id=762 is Argentina, but name="Brazil" would resolve to BRA
        # football_data_id has higher priority
        res = resolve_team(football_data_id=762, name="Brazil")
        assert res["canonical_id"] == "ARG"
        assert res["source"] == "football_data"

    def test_corpus_id_fallback_when_no_football_data_id(self):
        """When football_data_id is None, corpus_id is used for resolution."""
        from soccer_ev_model.team_identity import resolve_team

        res = resolve_team(corpus_id=710061511, name="Brazil")
        assert res["canonical_id"] == "ARG"
        assert res["source"] == "corpus"

    def test_name_fallback_when_no_ids(self):
        """When both IDs are None, name is used for resolution."""
        from soccer_ev_model.team_identity import resolve_team

        res = resolve_team(name="Argentina")
        assert res["canonical_id"] == "ARG"
        assert res["source"] == "name"
