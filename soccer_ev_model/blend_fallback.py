"""
Fallback and low-data policy for the Elo + Goal model blend.

This module implements deterministic behaviour when one or both primary
models (Elo, Goal) have missing or weak data.  The policy is:

  Case A — Elo valid + Goal valid (sufficient data):
      60/40 Elo/Goal blend, no warning.

  Case B — Goal valid but low-data (limited historical coverage):
      60/40 Elo/Goal blend, warning that Goal model has limited coverage.

  Case C — Goal unavailable (team unseen by goal model):
      100% Elo, warning that Goal model is unavailable.

  Case D — Elo unavailable (no Elo snapshot for team):
      100% Goal, warning that Elo is unavailable.

  Case E — Both unavailable:
      Conservative baseline (uniform 1/3 each), low-confidence warning.

Pi-ratings are diagnostic only — they are NEVER used as an automatic
fallback.  The blend is always between Elo and Goal; pi only appears
in the diagnostic output.

All branches produce an explicit ``warnings`` list.  No weight change
is ever silent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ModelAvailability:
    """Signals produced by the data-availability check for one model."""
    available: bool
    low_data: bool = False          # True when model runs but on thin data
    reason: str = ""                # human-readable explanation


@dataclass
class BlendResult:
    """Full output of the fallback-aware blend."""
    # The blended probabilities (always present, always sum to 1.0)
    primary_probs: dict[str, float]

    # Per-model probabilities (None when that model is unavailable)
    elo_probs: dict[str, float] | None
    goal_probs: dict[str, float] | None

    # Blend weights actually used (sum to 1.0)
    w_elo: float
    w_goal: float

    # Which case was selected
    case: str                       # "A" | "B" | "C" | "D" | "E"

    # Warnings (empty list for Case A; non-empty for B–E)
    warnings: list[str] = field(default_factory=list)

    # Diagnostics — pi-rating probs, NEVER used for fallback
    pi_probs: dict[str, float] | None = None

    # Per-model availability signals
    elo_available: bool = True
    goal_available: bool = True
    goal_low_data: bool = False


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Below this many training matches the goal model is "low-data"
GOAL_MIN_MATCHES_LOW = 10
# Below this the goal model is treated as "unavailable" (unseen team)
GOAL_MIN_MATCHES_UNSEEN = 1


# ---------------------------------------------------------------------------
# Availability helpers
# ---------------------------------------------------------------------------

def _check_elo_availability(
    elo_snapshots: dict[str, Any],
    home_team: str,
    away_team: str,
    match_date: str,
) -> tuple[ModelAvailability, ModelAvailability]:
    """Return (home_avail, away_avail) for the Elo model.

    Uses the same lookup convention as ``elo_ratings.elo_at``: a team is
    unavailable when no snapshot exists before the match date.
    """
    from .elo_ratings import elo_at as _elo_at

    home_elo, home_missing = _elo_at(elo_snapshots, home_team, match_date)
    away_elo, away_missing = _elo_at(elo_snapshots, away_team, match_date)

    home_avail = ModelAvailability(
        available=not home_missing,
        reason="" if not home_missing else f"No Elo data for {home_team} before {match_date}",
    )
    away_avail = ModelAvailability(
        available=not away_missing,
        reason="" if not away_missing else f"No Elo data for {away_team} before {match_date}",
    )
    return home_avail, away_avail


def _check_goal_availability(
    goal_model: Any,
    home_team_id: int,
    away_team_id: int,
) -> tuple[ModelAvailability, ModelAvailability]:
    """Return (home_avail, away_avail) for the Goal model.

    A team is *unavailable* when it has 0 matches in the goal model's
    training counts (unseen).  It is *low-data* when it has between 1 and
    GOAL_MIN_MATCHES_LOW matches.
    """
    counts: dict[str, int] = getattr(goal_model, "counts", {})

    home_count = counts.get(str(home_team_id), 0)
    away_count = counts.get(str(away_team_id), 0)

    def _avail(count: int, label: str) -> ModelAvailability:
        if count < GOAL_MIN_MATCHES_UNSEEN:
            return ModelAvailability(
                available=False,
                reason=f"{label} unseen by goal model (0 training matches)",
            )
        if count < GOAL_MIN_MATCHES_LOW:
            return ModelAvailability(
                available=True,
                low_data=True,
                reason=f"{label} has only {count} training matches (low-data)",
            )
        return ModelAvailability(available=True)

    return _avail(home_count, f"home team {home_team_id}"), _avail(
        away_count, f"away team {away_team_id}"
    )


# ---------------------------------------------------------------------------
# Blend weight resolver — the core policy
# ---------------------------------------------------------------------------

def resolve_blend_weights(
    elo_available: bool,
    goal_available: bool,
    goal_low_data: bool,
) -> tuple[float, float, str, list[str]]:
    """Deterministically resolve (w_elo, w_goal, case, warnings).

    Pure function — no side effects, no I/O.  Every branch is explicit.

    Returns:
        (w_elo, w_goal, case, warnings)
    """
    warnings: list[str] = []

    if elo_available and goal_available and not goal_low_data:
        # Case A: both valid, goal has sufficient data
        w_elo, w_goal = 0.6, 0.4
        return w_elo, w_goal, "A", warnings

    if elo_available and goal_available and goal_low_data:
        # Case B: goal valid but low-data — still blend, but warn
        w_elo, w_goal = 0.6, 0.4
        warnings.append(
            "Goal model has limited historical coverage — "
            "blend uses 60/40 Elo/Goal but confidence is reduced."
        )
        return w_elo, w_goal, "B", warnings

    if elo_available and not goal_available:
        # Case C: goal unavailable — fall back to Elo-only
        w_elo, w_goal = 1.0, 0.0
        warnings.append(
            "Goal model unavailable — Elo-only fallback used."
        )
        return w_elo, w_goal, "C", warnings

    if not elo_available and goal_available:
        # Case D: Elo unavailable — fall back to Goal-only
        w_elo, w_goal = 0.0, 1.0
        warnings.append(
            "Elo unavailable — Goal-only fallback used."
        )
        return w_elo, w_goal, "D", warnings

    # Case E: both unavailable — conservative baseline
    w_elo, w_goal = 0.0, 0.0
    warnings.append(
        "Primary models unavailable — low-confidence baseline used."
    )
    return w_elo, w_goal, "E", warnings


# ---------------------------------------------------------------------------
# Probability blend helper
# ---------------------------------------------------------------------------

def _blend_probs(
    elo_probs: dict[str, float] | None,
    goal_probs: dict[str, float] | None,
    w_elo: float,
    w_goal: float,
) -> dict[str, float]:
    """Blend two H/D/A probability dicts with the given weights.

    Handles all edge cases:
      - Both present: weighted average, renormalised.
      - Only one present: returns that one (weights ignored).
      - Neither present: returns uniform 1/3.
    - No divide-by-zero: weights are checked before division.
    """
    if elo_probs is not None and goal_probs is not None:
        total_w = w_elo + w_goal
        if total_w <= 0:
            # Shouldn't happen if resolve_blend_weights is used, but guard.
            return {"home": 1 / 3, "draw": 1 / 3, "away": 1 / 3}
        nw_elo = w_elo / total_w
        nw_goal = w_goal / total_w
        blended = {
            "home": nw_elo * elo_probs["home"] + nw_goal * goal_probs["home"],
            "draw": nw_elo * elo_probs["draw"] + nw_goal * goal_probs["draw"],
            "away": nw_elo * elo_probs["away"] + nw_goal * goal_probs["away"],
        }
        # Renormalize for floating-point safety
        s = sum(blended.values())
        if s > 0:
            blended = {k: v / s for k, v in blended.items()}
        return blended

    if elo_probs is not None:
        return dict(elo_probs)

    if goal_probs is not None:
        return dict(goal_probs)

    # Case E: neither available — uniform baseline
    return {"home": 1 / 3, "draw": 1 / 3, "away": 1 / 3}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def predict_with_fallback(
    # Identifiers
    home_team: str,
    away_team: str,
    home_team_id: int,
    away_team_id: int,
    match_date: str,
    # Model outputs (pre-computed by caller)
    elo_probs: dict[str, float] | None,
    goal_probs: dict[str, float] | None,
    # Availability signals
    elo_available: bool = True,
    goal_available: bool = True,
    goal_low_data: bool = False,
    # Diagnostics (never used for fallback)
    pi_probs: dict[str, float] | None = None,
) -> BlendResult:
    """Full fallback-aware prediction.

    This is the single entry point for the production pipeline.  It
    resolves blend weights deterministically, blends the probabilities,
    and returns a ``BlendResult`` with the case label and warnings.

    Args:
        home_team, away_team: human-readable team names.
        home_team_id, away_team_id: integer team IDs (for goal model lookup).
        match_date: ISO date string.
        elo_probs: H/D/A dict from Elo model, or None if Elo unavailable.
        goal_probs: H/D/A dict from Goal model, or None if Goal unavailable.
        elo_available: whether Elo data exists for both teams.
        goal_available: whether Goal model has data for both teams.
        goal_low_data: whether Goal model has <10 matches for either team.
        pi_probs: optional H/D/A dict from pi-rating (diagnostic only).

    Returns:
        BlendResult with primary_probs, case, warnings, and all diagnostics.
    """
    w_elo, w_goal, case, warnings = resolve_blend_weights(
        elo_available=elo_available,
        goal_available=goal_available,
        goal_low_data=goal_low_data,
    )

    primary = _blend_probs(elo_probs, goal_probs, w_elo, w_goal)

    return BlendResult(
        primary_probs={k: round(v, 6) for k, v in primary.items()},
        elo_probs={k: round(v, 6) for k, v in elo_probs.items()} if elo_probs is not None else None,
        goal_probs={k: round(v, 6) for k, v in goal_probs.items()} if goal_probs is not None else None,
        w_elo=w_elo,
        w_goal=w_goal,
        case=case,
        warnings=warnings,
        pi_probs={k: round(v, 6) for k, v in pi_probs.items()} if pi_probs is not None else None,
        elo_available=elo_available,
        goal_available=goal_available,
        goal_low_data=goal_low_data,
    )


# ---------------------------------------------------------------------------
# Convenience: availability check + predict in one call
# ---------------------------------------------------------------------------

def predict_with_availability(
    home_team: str,
    away_team: str,
    home_team_id: int,
    away_team_id: int,
    match_date: str,
    elo_snapshots: dict[str, Any],
    goal_model: Any,
    elo_probs_fn,          # (home_elo, away_elo) -> dict[str, float]
    goal_probs_fn,         # (home_id, away_id) -> dict[str, float]
    pi_probs: dict[str, float] | None = None,
) -> BlendResult:
    """Convenience wrapper that checks availability then predicts.

    This is the entry point when you have the raw snapshots / model and
    want the module to determine availability internally.

    Args:
        home_team, away_team: human-readable team names.
        home_team_id, away_team_id: integer team IDs.
        match_date: ISO date string.
        elo_snapshots: Elo snapshot dict (as returned by load_elo_ratings).
        goal_model: fitted goal model artifact or RegularizedTeamPoissonModel.
        elo_probs_fn: callable(home_elo: float, away_elo: float) -> H/D/A dict.
        goal_probs_fn: callable(home_id: int, away_id: int) -> H/D/A dict.
        pi_probs: optional diagnostic pi-rating probs.

    Returns:
        BlendResult.
    """
    # Check Elo availability
    elo_home_a, elo_away_a = _check_elo_availability(
        elo_snapshots, home_team, away_team, match_date
    )
    elo_available = elo_home_a.available and elo_away_a.available

    # Check Goal availability
    goal_home_a, goal_away_a = _check_goal_availability(
        goal_model, home_team_id, away_team_id
    )
    goal_available = goal_home_a.available and goal_away_a.available
    goal_low_data = goal_home_a.low_data or goal_away_a.low_data

    # Compute model probs only when available
    elo_probs = None
    if elo_available:
        from .elo_ratings import elo_at as _elo_at
        he, _ = _elo_at(elo_snapshots, home_team, match_date)
        ae, _ = _elo_at(elo_snapshots, away_team, match_date)
        elo_probs = elo_probs_fn(he, ae)

    goal_probs = None
    if goal_available:
        goal_probs = goal_probs_fn(home_team_id, away_team_id)

    return predict_with_fallback(
        home_team=home_team,
        away_team=away_team,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        match_date=match_date,
        elo_probs=elo_probs,
        goal_probs=goal_probs,
        elo_available=elo_available,
        goal_available=goal_available,
        goal_low_data=goal_low_data,
        pi_probs=pi_probs,
    )
