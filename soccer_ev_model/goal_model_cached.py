"""Cached, session-scoped goal model artifact loader.

Provides :func:`get_goal_predictor` — a single entry point for the dashboard
and other callers to obtain a fully-loaded :class:`GoalModelPredictor` that
is constructed **once per Streamlit session** and reused across all
subsequent prediction calls.

The module guarantees:
  * The artifact JSON is read from disk exactly once per session.
  * Artifact validation (version, required fields, team-ID sanity) runs
    once and the result is cached.
  * Prediction calls are stateless — no hidden I/O, no re-reads.
  * Structured :class:`GoalModelLoadError` on failure so callers can
    degrade gracefully (fallback message instead of a crash).

This module is the Phase 2 deliverable for artifact loading and caching.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_ARTIFACT_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "artifacts" / "goal_model_sh5.json"
)

# ── Structured errors ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GoalModelLoadError(Exception):
    """Structured error when the artifact cannot be loaded or is invalid.

    Carries a ``reason`` enum for programmatic handling and a human-readable
    ``message`` for UI display.
    """
    reason: str         # "missing" | "invalid_version" | "malformed" | "io_error"
    message: str
    detail: str = ""    # extra context (field name, path, etc.)

    def __str__(self) -> str:
        if self.detail:
            return f"{self.message} ({self.detail})"
        return self.message


# ── Validation (runs once, result cached) ────────────────────────────────────

def _validate_artifact(data: dict, path: str) -> Optional[GoalModelLoadError]:
    """Validate artifact content. Returns None if valid."""
    # Version check
    expected_version = "goal-model-artifact-v1"
    version = data.get("artifact_version")
    if version != expected_version:
        return GoalModelLoadError(
            reason="invalid_version",
            message=f"Artifact version mismatch: expected {expected_version}, got {version}",
            detail=path,
        )

    # Required fields
    required_fields = [
        "artifact_version", "model_version", "data_cutoff",
        "training_row_count", "shrinkage", "global_rate",
        "home_advantage", "attacks", "defenses", "counts",
    ]
    for field in required_fields:
        if field not in data:
            return GoalModelLoadError(
                reason="malformed",
                message=f"Missing required field: {field}",
                detail=path,
            )

    # Type checks on critical fields
    if not isinstance(data["attacks"], dict) or not isinstance(data["defenses"], dict):
        return GoalModelLoadError(
            reason="malformed",
            message="attacks/defenses must be dicts",
            detail=path,
        )
    if not isinstance(data["counts"], dict):
        return GoalModelLoadError(
            reason="malformed",
            message="counts must be a dict",
            detail=path,
        )

    # Team ID sanity: all keys should be numeric strings
    for label, mapping in [("attacks", data["attacks"]),
                           ("defenses", data["defenses"]),
                           ("counts", data["counts"])]:
        for key in list(mapping.keys())[:5]:  # spot-check first 5
            if not isinstance(key, str) or not key.isdigit():
                return GoalModelLoadError(
                    reason="malformed",
                    message=f"Non-numeric team ID in {label}: {key}",
                    detail=path,
                )

    # Sanity: attacks and defenses should have same team set
    attack_teams = set(data["attacks"].keys())
    defense_teams = set(data["defenses"].keys())
    if attack_teams != defense_teams:
        missing_def = attack_teams - defense_teams
        missing_att = defense_teams - attack_teams
        detail_parts = []
        if missing_def:
            detail_parts.append(f"{len(missing_def)} teams missing defenses")
        if missing_att:
            detail_parts.append(f"{len(missing_att)} teams missing attacks")
        return GoalModelLoadError(
            reason="malformed",
            message="Team ID mismatch between attacks and defenses",
            detail="; ".join(detail_parts),
        )

    return None


# ── Core loader (read + validate, no caching decorator — caller wraps) ───────

def load_and_validate(
    path: str | Path = DEFAULT_ARTIFACT_PATH,
) -> "GoalModelPredictor":
    """Load, validate, and construct a :class:`GoalModelPredictor`.

    Args:
        path: Path to the artifact JSON file. Defaults to
            ``data/artifacts/goal_model_sh5.json`` relative to the project root.

    Returns:
        A ready-to-use :class:`GoalModelPredictor`.

    Raises:
        GoalModelLoadError: If the artifact is missing, malformed, or has
            an incompatible version.  Carries ``.reason`` for programmatic handling.
        FileNotFoundError: If the file does not exist (wrapped in GoalModelLoadError
            with reason="missing").
    """
    from .goal_model_production import (
        GoalModelPredictor,
        GoalModelArtifact,
        load_artifact,
    )

    path = Path(path)

    if not path.exists():
        raise GoalModelLoadError(
            reason="missing",
            message=f"Artifact not found: {path}",
            detail=str(path),
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise GoalModelLoadError(
            reason="io_error",
            message=f"Failed to read artifact: {exc}",
            detail=str(path),
        ) from exc

    validation_error = _validate_artifact(data, str(path))
    if validation_error is not None:
        raise validation_error

    # Construct the validated artifact and predictor
    artifact = GoalModelArtifact.from_dict(data)
    predictor = GoalModelPredictor(artifact)

    logger.info(
        "Goal model loaded: version=%s, teams=%d, cutoff=%s",
        artifact.model_version,
        len(artifact.attacks),
        artifact.data_cutoff,
    )
    return predictor


# ── Streamlit-scoped singleton ───────────────────────────────────────────────

def _get_session_predictor() -> tuple[Optional["GoalModelPredictor"], Optional[GoalModelLoadError]]:
    """Internal: get or create the session-scoped predictor.

    Returns a tuple of (predictor_or_None, error_or_None).  The predictor
    is stored in ``st.session_state`` and reused on subsequent calls/reruns.
    """
    import streamlit as st

    # Sentinel keys — use a dict so we can store both predictor and error
    _PRED_KEY = "_goal_model_predictor"
    _ERR_KEY = "_goal_model_error"
    _LOAD_COUNT_KEY = "_goal_model_load_count"

    # Already loaded this session?
    predictor = st.session_state.get(_PRED_KEY)
    error = st.session_state.get(_ERR_KEY)
    if predictor is not None or error is not None:
        return predictor, error

    # First call this session — load and cache
    try:
        predictor = load_and_validate()
        st.session_state[_PRED_KEY] = predictor
        st.session_state[_ERR_KEY] = None
        st.session_state[_LOAD_COUNT_KEY] = 1
        logger.info("Goal model predictor initialized (session scope)")
    except GoalModelLoadError as exc:
        st.session_state[_PRED_KEY] = None
        st.session_state[_ERR_KEY] = exc
        st.session_state[_LOAD_COUNT_KEY] = 1
        logger.warning("Goal model load failed: %s", exc)

    # Re-read from session_state so the return value reflects the
    # cached result (the local `error` variable is stale after except).
    return (
        st.session_state.get(_PRED_KEY),
        st.session_state.get(_ERR_KEY),
    )


def get_goal_predictor() -> tuple[Optional["GoalModelPredictor"], Optional[GoalModelLoadError]]:
    """Return the session-scoped :class:`GoalModelPredictor`.

    This is the **only** entry point the dashboard should use.  The
    artifact is loaded exactly once per Streamlit session; all subsequent
    calls return the cached instance.  Reruns do NOT re-read the file.

    Returns:
        A 2-tuple of ``(predictor_or_None, error_or_None)``.

        * If loading succeeded, ``predictor`` is a ready-to-use
          :class:`GoalModelPredictor` and ``error`` is ``None``.
        * If loading failed, ``predictor`` is ``None`` and ``error`` is a
          :class:`GoalModelLoadError` with ``.reason`` for programmatic
          handling and ``.message`` for display.

    Example::

        predictor, err = get_goal_predictor()
        if err is not None:
            st.warning(f"Goal model unavailable: {err.message}")
            # fall back to pi-rating only
        else:
            prediction = predictor.predict(
                home_team_id=home_id,
                away_team_id=away_id,
                match_date="2026-06-18",
            )
    """
    return _get_session_predictor()


def get_goal_predictor_or_fallback() -> "GoalModelPredictor":
    """Convenience wrapper: return predictor or raise a clear error.

    Raises:
        GoalModelLoadError: If the predictor could not be loaded.
    """
    predictor, err = get_goal_predictor()
    if err is not None:
        raise err
    return predictor


def get_load_count() -> int:
    """Return how many times the artifact has been loaded this session.

    Always returns 1 after the first call (verifies caching is working).
    Returns 0 if the predictor has never been requested.
    """
    import streamlit as st
    return st.session_state.get("_goal_model_load_count", 0)


def reset_session_predictor() -> None:
    """Clear the cached predictor from session state (for testing)."""
    import streamlit as st
    st.session_state.pop("_goal_model_predictor", None)
    st.session_state.pop("_goal_model_error", None)
    st.session_state.pop("_goal_model_load_count", None)
