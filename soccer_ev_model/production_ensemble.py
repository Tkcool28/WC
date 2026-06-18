"""Deterministic production ensemble core for the Elo60/Goal40 blend.

This module implements the validated blend formula:
    primary = 0.60 * Elo + 0.40 * Goal

It accepts already-computed component probability vectors and returns a
stable, normalized output contract.  No model loading, team resolution,
dashboard imports, or side effects.

The function is pure-deterministic: identical inputs always produce
identical outputs.  Pi probabilities are diagnostic-only and never
influence ``primary_probs``.
"""
from __future__ import annotations

import math
from typing import Any

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_NAME = "elo60_goal40"
WEIGHTS = {"elo": 0.60, "goal": 0.40}

_PROB_KEYS = ("home", "draw", "away")
_SUM_TOLERANCE = 1e-9


# ── Validation ───────────────────────────────────────────────────────────────

class EnsembleInputError(ValueError):
    """Raised when component probability vectors are invalid."""


def _validate_probs(probs: Any, label: str) -> dict[str, float]:
    """Validate and return a normalized 3-outcome probability dict.

    Args:
        probs: caller-supplied dict with home/draw/away keys.
        label: source name for error messages (e.g. "elo_probs").

    Returns:
        A new dict with float values, defensively normalized.

    Raises:
        EnsembleInputError: on any invalid input.
    """
    if not isinstance(probs, dict):
        raise EnsembleInputError(
            f"{label}: expected dict, got {type(probs).__name__}"
        )

    # Check required keys
    missing = set(_PROB_KEYS) - set(probs.keys())
    if missing:
        raise EnsembleInputError(
            f"{label}: missing keys {sorted(missing)}"
        )

    # Reject extra keys silently — only extract what we need
    values: dict[str, float] = {}
    for k in _PROB_KEYS:
        v = probs[k]
        if not isinstance(v, (int, float)):
            raise EnsembleInputError(
                f"{label}[{k}]: expected numeric, got {type(v).__name__}"
            )
        if math.isnan(v):
            raise EnsembleInputError(f"{label}[{k}]: NaN not allowed")
        if math.isinf(v):
            raise EnsembleInputError(f"{label}[{k}]: infinity not allowed")
        if v < 0:
            raise EnsembleInputError(f"{label}[{k}]: negative value {v}")
        values[k] = float(v)

    total = sum(values.values())
    if total <= 0:
        raise EnsembleInputError(
            f"{label}: total probability must be > 0, got {total}"
        )

    # Defensive normalization
    if abs(total - 1.0) > _SUM_TOLERANCE:
        values = {k: v / total for k, v in values.items()}

    return values


# ── Core blend ───────────────────────────────────────────────────────────────

def blend_ensemble(
    elo_probs: dict[str, float],
    goal_probs: dict[str, float],
    *,
    pi_probs: dict[str, float] | None = None,
    goal_details: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Blend Elo and Goal probabilities into a deterministic ensemble output.

    Args:
        elo_probs: Elo-derived home/draw/away probabilities.
        goal_probs: Goal-model-derived home/draw/away probabilities.
        pi_probs: optional diagnostic pi-rating probabilities (not blended).
        goal_details: optional goal model metadata (home_xg, away_xg, etc.).
        warnings: optional list of warning strings.

    Returns:
        Dict conforming to the Phase 1 output schema.

    Raises:
        EnsembleInputError: on invalid component inputs.
    """
    # Validate inputs (creates new dicts — no mutation)
    elo = _validate_probs(elo_probs, "elo_probs")
    goal = _validate_probs(goal_probs, "goal_probs")

    pi = None
    if pi_probs is not None:
        pi = _validate_probs(pi_probs, "pi_probs")

    details = None
    if goal_details is not None:
        if not isinstance(goal_details, dict):
            raise EnsembleInputError(
                f"goal_details: expected dict, got {type(goal_details).__name__}"
            )
        details = dict(goal_details)  # shallow copy, no mutation

    warn_list = list(warnings) if warnings else []

    # Weighted blend
    blended: dict[str, float] = {}
    for k in _PROB_KEYS:
        blended[k] = WEIGHTS["elo"] * elo[k] + WEIGHTS["goal"] * goal[k]

    # Final normalization (ensures sum == 1.0 even after float arithmetic)
    total = sum(blended.values())
    if total <= 0:
        # Should not happen given validation, but guard defensively
        raise EnsembleInputError("blended total probability is zero")
    primary = {k: blended[k] / total for k in _PROB_KEYS}

    # Final safety: clamp tiny negatives from float error, re-normalize
    primary = {k: max(0.0, v) for k, v in primary.items()}
    total = sum(primary.values())
    primary = {k: v / total for k, v in primary.items()}

    return {
        "primary_probs": primary,
        "elo_probs": elo,
        "goal_probs": goal,
        "pi_probs": pi,
        "goal_details": details,
        "model_name": MODEL_NAME,
        "weights": dict(WEIGHTS),
        "fallback_used": None,
        "warnings": warn_list,
    }
