"""Tests for the cached goal model artifact loader.

Verifies:
  * Artifact is loaded exactly once per session (no per-prediction reload).
  * Missing artifact produces a structured GoalModelLoadError.
  * Invalid artifact (wrong version) produces a structured GoalModelLoadError.
  * Valid artifact produces a working predictor.
  * Predictor is stateless (no I/O on predict calls).
  * Fresh clone can locate the artifact at the expected path.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure the WC project root is on sys.path for imports
_WC_ROOT = Path(__file__).resolve().parent.parent
if str(_WC_ROOT) not in sys.path:
    sys.path.insert(0, str(_WC_ROOT))


# ── Helpers ──────────────────────────────────────────────────────────────────

def _read_artifact() -> dict:
    """Read the on-disk artifact for test reference."""
    art_path = _WC_ROOT / "data" / "artifacts" / "goal_model_sh5.json"
    return json.loads(art_path.read_text(encoding="utf-8"))


# ── Tests: fresh clone can locate the artifact ──────────────────────────────

def test_artifact_exists_on_disk():
    """The artifact file exists at the expected path in a fresh clone."""
    art_path = _WC_ROOT / "data" / "artifacts" / "goal_model_sh5.json"
    assert art_path.exists(), f"Artifact missing: {art_path}"
    assert art_path.stat().st_size > 1000, "Artifact suspiciously small"


def test_artifact_loads_from_default_path():
    """load_and_validate() works with no path argument (default)."""
    from soccer_ev_model.goal_model_cached import load_and_validate
    predictor = load_and_validate()
    assert predictor is not None
    assert predictor.artifact is not None
    assert len(predictor.artifact.attacks) > 0


# ── Tests: valid artifact produces a working predictor ──────────────────────

def test_valid_predictor_predicts():
    """A loaded predictor can produce predictions without I/O."""
    from soccer_ev_model.goal_model_cached import load_and_validate
    predictor = load_and_validate()
    # Pick a team ID known to be in the artifact
    sample_id = list(predictor.artifact.attacks.keys())[0]
    pred = predictor.predict(
        home_team_id=int(sample_id),
        away_team_id=int(sample_id),  # same team — degenerate but tests the math
        match_date="2026-06-18",
    )
    assert pred.home_xg > 0
    assert pred.away_xg > 0
    assert pred.home_xg <= 6.0
    assert pred.away_xg <= 6.0


# ── Tests: missing artifact ─────────────────────────────────────────────────

def test_missing_artifact_raises_structured_error():
    """Missing file → GoalModelLoadError(reason='missing')."""
    from soccer_ev_model.goal_model_cached import (
        GoalModelLoadError,
        load_and_validate,
    )
    with pytest.raises(GoalModelLoadError) as exc_info:
        load_and_validate(path="/nonexistent/goal_model_sh5.json")
    assert exc_info.value.reason == "missing"
    assert "not found" in exc_info.value.message.lower()


def test_missing_artifact_from_invalid_path():
    """A directory that doesn't exist also produces 'missing'."""
    from soccer_ev_model.goal_model_cached import (
        GoalModelLoadError,
        load_and_validate,
    )
    with pytest.raises(GoalModelLoadError) as exc_info:
        load_and_validate(path=Path("/tmp/no_such_dir_12345/artifact.json"))
    assert exc_info.value.reason == "missing"


# ── Tests: invalid artifact ─────────────────────────────────────────────────

def test_invalid_version_raises_structured_error():
    """Wrong artifact_version → GoalModelLoadError(reason='invalid_version')."""
    from soccer_ev_model.goal_model_cached import (
        GoalModelLoadError,
        load_and_validate,
    )
    data = _read_artifact()
    data["artifact_version"] = "wrong-version-999"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(data, f)
        tmp_path = f.name

    try:
        with pytest.raises(GoalModelLoadError) as exc_info:
            load_and_validate(path=tmp_path)
        assert exc_info.value.reason == "invalid_version"
        assert "wrong-version-999" in exc_info.value.message
    finally:
        Path(tmp_path).unlink()


def test_malformed_artifact_missing_field():
    """Missing required field → GoalModelLoadError(reason='malformed')."""
    from soccer_ev_model.goal_model_cached import (
        GoalModelLoadError,
        load_and_validate,
    )
    data = _read_artifact()
    del data["attacks"]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(data, f)
        tmp_path = f.name

    try:
        with pytest.raises(GoalModelLoadError) as exc_info:
            load_and_validate(path=tmp_path)
        assert exc_info.value.reason == "malformed"
        assert "attacks" in exc_info.value.message
    finally:
        Path(tmp_path).unlink()


def test_malformed_artifact_attacks_not_dict():
    """Non-dict attacks → GoalModelLoadError(reason='malformed')."""
    from soccer_ev_model.goal_model_cached import (
        GoalModelLoadError,
        load_and_validate,
    )
    data = _read_artifact()
    data["attacks"] = [1, 2, 3]  # list instead of dict

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(data, f)
        tmp_path = f.name

    try:
        with pytest.raises(GoalModelLoadError) as exc_info:
            load_and_validate(path=tmp_path)
        assert exc_info.value.reason == "malformed"
    finally:
        Path(tmp_path).unlink()


def test_malformed_artifact_team_id_mismatch():
    """attacks/defenses with different team sets → malformed error."""
    from soccer_ev_model.goal_model_cached import (
        GoalModelLoadError,
        load_and_validate,
    )
    data = _read_artifact()
    # Remove one team from defenses but leave in attacks
    key = list(data["attacks"].keys())[0]
    del data["defenses"][key]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(data, f)
        tmp_path = f.name

    try:
        with pytest.raises(GoalModelLoadError) as exc_info:
            load_and_validate(path=tmp_path)
        assert exc_info.value.reason == "malformed"
        assert "mismatch" in exc_info.value.message.lower()
    finally:
        Path(tmp_path).unlink()


# ── Tests: no retrain/reread per prediction ─────────────────────────────────

def test_predictor_is_stateless_no_io_on_predict():
    """Prediction calls do not read the artifact file again."""
    from soccer_ev_model.goal_model_cached import load_and_validate

    predictor = load_and_validate()
    sample_id = list(predictor.artifact.attacks.keys())[0]

    # Patch the artifact module's file read to detect I/O
    import soccer_ev_model.goal_model_production as gmp
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
        f"Artifact was read {read_count['n']} times during prediction — "
        "predictor should be stateless."
    )


# ── Tests: session-scoped caching (mocked st.session_state) ─────────────────

def test_session_predictor_loads_once():
    """get_goal_predictor() loads the artifact exactly once per session."""
    from soccer_ev_model.goal_model_cached import (
        get_goal_predictor,
        get_load_count,
        reset_session_predictor,
    )

    # Mock a minimal st.session_state dict
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
        predictor1, err1 = get_goal_predictor()
        assert predictor1 is not None
        assert err1 is None
        assert get_load_count() == 1

        # Second call returns cached (no new load)
        predictor2, err2 = get_goal_predictor()
        assert predictor2 is predictor1  # same object
        assert err2 is None
        assert get_load_count() == 1  # still 1


def test_session_predictor_missing_artifact():
    """Missing artifact → get_goal_predictor returns (None, error)."""
    from soccer_ev_model.goal_model_cached import (
        GoalModelLoadError,
        get_goal_predictor,
        reset_session_predictor,
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
        with patch(
            "soccer_ev_model.goal_model_cached.load_and_validate",
            side_effect=GoalModelLoadError(
                reason="missing",
                message="Artifact not found",
                detail="/fake/path.json",
            ),
        ):
            predictor, err = get_goal_predictor()
            assert predictor is None
            assert err is not None
            assert err.reason == "missing"

        # Second call should return the cached error, not re-raise
        predictor2, err2 = get_goal_predictor()
        assert predictor2 is None
        assert err2 is err  # same error object


def test_session_predictor_invalid_version():
    """Invalid version → get_goal_predictor returns (None, error)."""
    from soccer_ev_model.goal_model_cached import (
        GoalModelLoadError,
        get_goal_predictor,
        reset_session_predictor,
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
        with patch(
            "soccer_ev_model.goal_model_cached.load_and_validate",
            side_effect=GoalModelLoadError(
                reason="invalid_version",
                message="version mismatch",
                detail="v2 vs v1",
            ),
        ):
            predictor, err = get_goal_predictor()
            assert predictor is None
            assert err.reason == "invalid_version"


def test_reset_session_predictor():
    """reset_session_predictor() clears the cached state."""
    from soccer_ev_model.goal_model_cached import (
        get_goal_predictor,
        get_load_count,
        reset_session_predictor,
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
        predictor, _ = get_goal_predictor()
        assert predictor is not None
        assert get_load_count() == 1

        reset_session_predictor()
        assert get_load_count() == 0
        assert "_goal_model_predictor" not in fake_state


# ── Tests: structured error properties ──────────────────────────────────────

def test_goal_model_load_error_str():
    """GoalModelLoadError.__str__ formats message correctly."""
    from soccer_ev_model.goal_model_cached import GoalModelLoadError

    err = GoalModelLoadError(
        reason="missing",
        message="not found",
        detail="/some/path.json",
    )
    assert str(err) == "not found (/some/path.json)"

    err_no_detail = GoalModelLoadError(
        reason="malformed",
        message="bad format",
    )
    assert str(err_no_detail) == "bad format"


def test_goal_model_load_error_is_frozen():
    """GoalModelLoadError is immutable (frozen dataclass)."""
    from soccer_ev_model.goal_model_cached import GoalModelLoadError

    err = GoalModelLoadError(reason="missing", message="x")
    with pytest.raises(AttributeError):
        err.reason = "something_else"
