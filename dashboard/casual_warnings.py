"""Consolidate raw model warnings into a single concise user-facing message.

The casual Predictions card (Phase 3 / Phase 6) is allowed to surface at
most ONE warning per game.  Behind the scenes the model emits several
distinct warnings — a history-missing line on the identity-warning
pipeline, an "<5 prior matches" / "coin flip" line from the confidence
assessor, a calibration-caution line referring to a 9,678-match
backtest, etc.  Showing all of them at once would stack three or four
``st.info`` blocks per card, leak internal counts, and read like a log
file.  This module picks the single highest-priority message and
absorbs lower-priority calibration detail into it.

Selection priority (highest first; first match wins):

1. ``identity_unresolved`` — a team could not be resolved at all
   (e.g. canonical=None, fd_id=...).  Show the calm "We couldn't find
   this team in our training data" message naming the team.
2. ``history_missing`` — the team has no / very little training-corpus
   history (the canonical=CPV/COD case).  Name the team and absorb the
   calibration caveat into one message.
3. ``limited_data`` — at least one team has <5 / <30 prior matches;
   pi-rating is unreliable.  Name the team.
4. ``calibration_caution`` — pi-rating is overconfident vs the
   9,678-match backtest.  No team name required.
5. ``other`` — pass-through of any already-translated sentence, no
   internal codes, no raw counts.

The function MUST NOT return warnings that contain:
  * raw internal codes (``canonical=``, ``status=``, ``fd_id=``)
  * raw counts (``home: 429, away: 0``, ``<5 prior matches``,
    ``9,678-match``)
  * technical jargon (``Pi-rating is essentially a coin flip``,
    ``calibration bucket``)
  * canonical IDs in the user-visible text

The Analysis tab is unchanged — it still prints the full raw warning
list under "Calibration and Data Quality" and "Raw Diagnostics".
"""
from __future__ import annotations

import re
from typing import Any

# --------------------------------------------------------------------------- #
# Patterns that identify an internal / raw warning (NEVER pass through)
# --------------------------------------------------------------------------- #
_INTERNAL_PATTERNS = (
    re.compile(r"canonical=", re.IGNORECASE),
    re.compile(r"\bstatus=", re.IGNORECASE),
    re.compile(r"\bfd_id=", re.IGNORECASE),
    re.compile(r"\bcanonical_id\s*=", re.IGNORECASE),
    re.compile(r"\bhistory_missing\b", re.IGNORECASE),
    re.compile(r"\bidentity_unresolved\b", re.IGNORECASE),
    re.compile(r"home:\d+\s+away:\d+", re.IGNORECASE),
    re.compile(r"<5\s+prior\s+matches", re.IGNORECASE),
    re.compile(r"9,?678-match", re.IGNORECASE),
    re.compile(r"coin\s+flip", re.IGNORECASE),
    re.compile(r"calibration\s+bucket", re.IGNORECASE),
    re.compile(r"neutral\s+pi-rating", re.IGNORECASE),
    re.compile(r"no\s+training-corpus\s+history", re.IGNORECASE),
    re.compile(r"could\s+not\s+be\s+resolved", re.IGNORECASE),
)


def _is_internal(raw: str) -> bool:
    """True if ``raw`` is an internal warning (must not pass through)."""
    if not raw:
        return False
    return any(p.search(raw) for p in _INTERNAL_PATTERNS)


# --------------------------------------------------------------------------- #
# Classification helpers
# --------------------------------------------------------------------------- #
def _is_identity_unresolved(raw: str) -> bool:
    """A team could not be resolved via the canonical identity registry.

    Distinguish from ``_is_history_missing``: a missing canonical id
    (e.g. ``canonical_id=None``) is unresolved; a resolved team with no
    training data is *history_missing*, not unresolved.
    """
    low = (raw or "").lower()
    if "identity_unresolved" in low:
        return True
    if "could not be resolved" in low:
        return True
    # The ``canonical=None`` / ``canonical_id=None`` form means the
    # registry lookup failed for this team.  ``(canonical=COD,
    # status=history_missing)`` is NOT this case — that team is
    # resolved, it just has no training data.
    if "canonical=none" in low or "canonical_id=none" in low:
        return True
    if re.search(r"\bfd_id=\d+\b", low) and "resolved" in low:
        return True
    return False


def _is_history_missing(raw: str) -> bool:
    """The team has no / very little training-corpus history."""
    low = (raw or "").lower()
    return (
        "history_missing" in low
        or "no training-corpus history" in low
        # The "<5 prior matches" raw warning is also a "history_missing"
        # signal in our casual wording.
        or "prior matches" in low
        or "coin flip" in low
    )


def _is_limited_data(raw: str) -> bool:
    """A limited-data / insufficient-data / coin-flip signal."""
    low = (raw or "").lower()
    return (
        "limited data" in low
        or "insufficient data" in low
        or "low data" in low
        or "min matches" in low
    )


def _is_calibration_caution(raw: str) -> bool:
    """The model's raw probability is more confident than backtest hit rates."""
    low = (raw or "").lower()
    return (
        "calibration" in low
        or "overconfident" in low
        or "hit rate" in low
        or "9,678-match" in low
        or "9,678 match" in low
    )


# --------------------------------------------------------------------------- #
# Display-name resolution
# --------------------------------------------------------------------------- #
def _display_team_names(prediction: dict, override: dict | None) -> dict[str, str]:
    """Return ``{"home": str, "away": str}`` preferring override names."""
    out = {
        "home": (
            (override or {}).get("home")
            or prediction.get("home_team")
            or "the home team"
        ),
        "away": (
            (override or {}).get("away")
            or prediction.get("away_team")
            or "the away team"
        ),
    }
    return out


def _which_team_lacks_history(
    prediction: dict,
    raw_warnings: list[str],
    identity_warnings: list[str],
) -> str:
    """Return ``"home"`` or ``"away"`` — which side has the missing data?

    Resolution priority (first hit wins):

    1. If ``prediction["canonical_home_id"]`` is ``None``/empty and
       ``canonical_away_id`` is set → ``"home"``.
    2. Vice versa → ``"away"``.
    3. If both are missing → scan the raw warnings for the home /
       away team names; first match wins.
    4. As a final fallback, default to ``"home"``.
    """
    pred = prediction or {}
    home_can = pred.get("canonical_home_id")
    away_can = pred.get("canonical_away_id")
    home_missing = home_can in (None, "", "None")
    away_missing = away_can in (None, "", "None")
    if home_missing and not away_missing:
        return "home"
    if away_missing and not home_missing:
        return "away"

    # Both or neither missing — scan the warning blob for team names.
    home_name = (pred.get("home_team") or "").strip().lower()
    away_name = (pred.get("away_team") or "").strip().lower()
    blob = " ".join(
        [str(w) for w in (raw_warnings + identity_warnings) if w]
    ).lower()
    if home_name and home_name in blob:
        return "home"
    if away_name and away_name in blob:
        return "away"

    # Last-resort: scan for "home" / "away" keywords.
    if not blob:
        return "home"
    mentions_home = "home" in blob or "home:" in blob
    mentions_away = "away" in blob or "away:" in blob
    if mentions_home and not mentions_away:
        return "home"
    if mentions_away and not mentions_home:
        return "away"
    return "home"


# --------------------------------------------------------------------------- #
# Wording templates
# --------------------------------------------------------------------------- #
def _wording_identity_unresolved(team: str) -> str:
    return (
        f"We couldn't find this team in our training data ({team}). "
        f"This prediction is unavailable."
    )


def _wording_history_missing(team: str) -> str:
    return (
        f"Limited confidence: {team} has little historical match data, so "
        f"this prediction relies more heavily on the other available rating "
        f"information and should be treated cautiously."
    )


def _wording_limited_data(team: str) -> str:
    return (
        f"Limited data available for {team} — treat this prediction "
        f"cautiously."
    )


def _wording_calibration_caution() -> str:
    return (
        "Calibration caution: this prediction's raw probability is somewhat "
        "more confident than long-term hit rates support."
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def consolidate_casual_warnings(
    prediction: dict,
    *,
    confidence: dict | None = None,
    identity_warnings: list[str] | None = None,
    display_team_names: dict | None = None,
) -> list[str]:
    """Pick at most ONE concise user-facing warning for the casual
    Prediction card.

    Returns a list with 0 or 1 element.  NEVER returns multiple
    stacked warnings for the same underlying limitation.

    Parameters
    ----------
    prediction
        Output of :func:`soccer_ev_model.ev_workflow.predict_match`.  Used
        only to resolve display team names — the function does NOT
        recompute anything.
    confidence
        Optional confidence assessment dict (defaults to
        ``prediction.get("confidence")``).  The raw
        ``confidence["warnings"]`` list is the second source of
        warnings.
    identity_warnings
        Optional list of raw identity-warning strings (defaults to
        ``prediction.get("identity_warnings")``).
    display_team_names
        Optional ``{"home": str, "away": str}`` override for display
        names.  Falls back to ``prediction["home_team"]`` /
        ``prediction["away_team"]``, then to a generic placeholder.

    Returns
    -------
    list[str]
        0 or 1 concise, jargon-free warning sentences.
    """
    prediction = prediction or {}
    assessment = confidence if confidence is not None else (
        prediction.get("confidence") or {}
    )
    raw_warnings = list(assessment.get("warnings") or [])
    identity = list(
        identity_warnings
        if identity_warnings is not None
        else (prediction.get("identity_warnings") or [])
    )

    names = _display_team_names(prediction, display_team_names)

    # Build a single combined raw-warning blob for classification.
    # Each side gets its own list so we can pick the highest-priority
    # classification.
    combined = [
        *((str(w) for w in identity if w)),
        *((str(w) for w in raw_warnings if w)),
    ]

    if not combined:
        return []

    # Priority 1 — identity_unresolved (highest).  Always show the
    # "could not be resolved" line; it absorbs any calibration / data
    # detail below it.
    for w in combined:
        if _is_identity_unresolved(w):
            team = _which_team_lacks_history(prediction, raw_warnings, identity)
            return [_wording_identity_unresolved(names[team])]

    # Priority 2 — history_missing (the canonical=CPV/COD case).
    # Absorb any calibration_caution raw warning into this message.
    has_history_missing = any(_is_history_missing(w) for w in combined)
    has_calibration = any(_is_calibration_caution(w) for w in combined)
    if has_history_missing:
        team = _which_team_lacks_history(prediction, raw_warnings, identity)
        return [_wording_history_missing(names[team])]

    # Priority 3 — limited_data (without the explicit "no history" flag).
    if any(_is_limited_data(w) for w in combined):
        team = _which_team_lacks_history(prediction, raw_warnings, identity)
        return [_wording_limited_data(names[team])]

    # Priority 4 — calibration_caution (no data/identity signals).
    # Only fire when the tier warrants it. Tier A/B high-history teams
    # have already paid the calibration tax in the pi-rating math; the
    # casual screen should NOT emit an extra warning for them. The
    # full calibration detail is still available in Analysis.
    if has_calibration:
        tier = assessment.get("tier") if isinstance(assessment, dict) else None
        if tier not in ("A", "B"):
            return [_wording_calibration_caution()]

    # Priority 5 — other.  Pass-through of any user-facing sentence.
    # Reject internal strings as a safety net.  Skip pass-through for
    # tier A/B teams too — the casual card should stay quiet for
    # well-supported predictions; raw detail is still available in
    # Analysis via _render_calibration / _render_raw_diagnostics.
    tier = assessment.get("tier") if isinstance(assessment, dict) else None
    if tier in ("A", "B"):
        return []
    for w in combined:
        if _is_internal(w):
            continue
        if w and w[0].isupper() and w.rstrip().endswith("."):
            return [w]
    return []


__all__ = ["consolidate_casual_warnings"]