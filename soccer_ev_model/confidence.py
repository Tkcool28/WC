"""
Confidence assessment for pi-rating predictions.

Pi-rating is well-calibrated in general, but degrades when:
  1. A team has too few matches in the training set (no signal)
  2. The prediction is high-confidence (overconfidence, see calibration table)
  3. The teams have wildly different recent activity (one side is rusty)

This module exposes a single function `assess_match_confidence()` that returns
a structured assessment the dashboard can render visually.

The key principle: NEVER block. Always show the pi-rating output, but flag
when confidence is low so the user can decide.

Calibration source: 9,678 intl matches 2015-2024, pi-rating at LR=0.005
(see scripts/calibration_run.py for reproduction).
"""
from __future__ import annotations

from typing import Any


# Calibration table: bucket of top_p -> actual hit rate
# Built from 9,678 international matches 2015-2024.
# (lo, hi) -> actual_hit_rate
# Pairs with confidence_label below.
CALIBRATION_TABLE: list[tuple[tuple[float, float], float, str]] = [
    ((0.00, 0.40), 0.377, "high"),       # Honest, well-calibrated
    ((0.40, 0.50), 0.458, "high"),       # Honest
    ((0.50, 0.60), 0.532, "high"),       # Honest
    ((0.60, 0.70), 0.560, "medium"),     # Mild overconfidence
    ((0.70, 0.80), 0.646, "low"),        # Significant overconfidence
    ((0.80, 0.90), 0.721, "low"),        # Heavy overconfidence
    ((0.90, 1.01), 0.834, "low"),        # Heavy overconfidence
]

# Minimum matches_played for "trusted" data
# Below this, the team's pi-rating is basically a guess.
# Empirically: teams with <30 matches have pi-ratings that don't track results well.
MIN_MATCHES_TRUSTED = 30
MIN_MATCHES_HIGH_CONF = 100  # Below this, we don't trust high-confidence predictions


def calibration_lookup(top_p: float) -> float:
    """Look up actual hit rate for a given raw pi-rating confidence.

    Args:
        top_p: the highest probability from pi-rating (e.g. 0.75 for a "75% confident" pick)

    Returns:
        The empirically-observed hit rate for matches in that bucket.
    """
    for (lo, hi), actual_hit, _ in CALIBRATION_TABLE:
        if lo <= top_p < hi:
            return actual_hit
    return 0.5  # fallback for edge cases


def calibration_confidence_label(top_p: float) -> str:
    """Bucket-based confidence label from calibration table.

    Returns: "high" | "medium" | "low"
    """
    for (lo, hi), _, label in CALIBRATION_TABLE:
        if lo <= top_p < hi:
            return label
    return "low"


def matches_played_confidence_label(matches_played: int) -> str:
    """Data-volume-based confidence label.

    Returns: "high" | "medium" | "low" | "insufficient"
    """
    if matches_played < 5:
        return "insufficient"
    if matches_played < MIN_MATCHES_TRUSTED:
        return "low"
    if matches_played < MIN_MATCHES_HIGH_CONF:
        return "medium"
    return "high"


# Overall confidence tier and what action is reasonable
# This is what the dashboard displays.
# Tiers (best to worst):
#   A: model is well-calibrated AND team has plenty of data -> trust the number
#   B: one of calibration or data is good, not both         -> moderate trust
#   C: model is overconfident AND/OR data is sparse         -> be skeptical
#   D: insufficient data (team never played)                -> the model is a coin flip
TIER_DESCRIPTIONS = {
    "A": "High confidence — the model is well-calibrated and the team has ample match history. Trust the number.",
    "B": "Moderate confidence — mild calibration or data caveats apply.",
    "C": "Low confidence — the model tends to be overconfident at this probability level OR the team has limited history. Treat as a rough estimate.",
    "D": "Insufficient data — one or both teams have <5 prior matches. The model is essentially a coin flip here.",
}


def assess_match_confidence(
    home_matches_played: int,
    away_matches_played: int,
    pi_probs: dict[str, float],
) -> dict[str, Any]:
    """Full confidence assessment for a pi-rating prediction.

    Args:
        home_matches_played: int, the home team's matches_played in pi-rating dict
        away_matches_played: int, the away team's matches_played
        pi_probs: {"home": float, "draw": float, "away": float} from pi-rating

    Returns:
        dict with:
            tier: "A" | "B" | "C" | "D"
            tier_description: human-readable explanation
            top_p: highest probability in pi_probs
            calibrated_p: actual hit rate per calibration table
            calibration_diff: calibrated_p - top_p (negative = overconfident)
            calib_label: "high" | "medium" | "low" (calibration quality)
            data_label: "high" | "medium" | "low" | "insufficient" (data quality)
            warnings: list of human-readable warning strings
            edge_warning: bool, True if the +EV signal might be unreliable
    """
    top_p = max(pi_probs.values())

    calib_label = calibration_confidence_label(top_p)
    calibrated_p = calibration_lookup(top_p)
    calibration_diff = calibrated_p - top_p

    # Worst data label between home and away (the weak link)
    home_label = matches_played_confidence_label(home_matches_played)
    away_label = matches_played_confidence_label(away_matches_played)
    label_rank = {"high": 3, "medium": 2, "low": 1, "insufficient": 0}
    data_label = home_label if label_rank[home_label] < label_rank[away_label] else away_label
    min_matches = min(home_matches_played, away_matches_played)

    # Build warnings list (always populated if any concerns)
    warnings: list[str] = []

    if data_label == "insufficient":
        warnings.append(
            f"One or both teams have <5 prior matches in training. "
            f"Pi-rating is essentially a coin flip here (home: {home_matches_played}, away: {away_matches_played})."
        )
    elif data_label == "low":
        warnings.append(
            f"Limited data: min matches played = {min_matches} (recommended: ≥{MIN_MATCHES_TRUSTED}). "
            f"Pi-rating for this matchup is directionally useful but not precise."
        )
    elif data_label == "medium":
        warnings.append(
            f"Moderate data: min matches played = {min_matches}. "
            f"Pi-rating is reliable but not as precise as for high-data teams."
        )

    if calib_label == "low":
        warnings.append(
            f"Pi-rating is overconfident at this probability level. "
            f"Raw {top_p:.0%} but actual hit rate is closer to {calibrated_p:.0%} per 9,678-match backtest."
        )
    elif calib_label == "medium":
        warnings.append(
            f"Pi-rating is mildly overconfident here. "
            f"Raw {top_p:.0%} but actual hit rate is closer to {calibrated_p:.0%}."
        )

    # Tier assignment
    if data_label == "insufficient":
        tier = "D"
    elif calib_label == "low" and data_label in ("low", "medium"):
        tier = "C"
    elif calib_label == "low" or data_label in ("low",):
        tier = "C"
    elif calib_label == "medium" or data_label == "medium":
        tier = "B"
    else:
        tier = "A"

    tier_description = TIER_DESCRIPTIONS[tier]

    # Edge warning: +EV signal might be unreliable
    # True when the prediction is in a low-data / low-calibration regime
    edge_warning = tier in ("C", "D")

    return {
        "tier": tier,
        "tier_description": tier_description,
        "top_p": top_p,
        "calibrated_p": calibrated_p,
        "calibration_diff": calibration_diff,
        "calib_label": calib_label,
        "data_label": data_label,
        "home_matches_played": home_matches_played,
        "away_matches_played": away_matches_played,
        "min_matches_played": min_matches,
        "warnings": warnings,
        "edge_warning": edge_warning,
    }


def render_warning_banner(assessment: dict[str, Any]) -> str:
    """Render a single-line warning banner for the dashboard.

    The dashboard can prepend this to the model output to give the user
    a quick visual cue. Empty string if no warnings.
    """
    tier = assessment["tier"]
    tier_emoji = {
        "A": "🟢",
        "B": "🟡",
        "C": "🟠",
        "D": "🔴",
    }
    banner = f"{tier_emoji[tier]} Tier {tier} confidence — {assessment['tier_description']}"

    if assessment["warnings"]:
        banner += "\n" + "\n".join(f"  ⚠️  {w}" for w in assessment["warnings"])

    return banner
