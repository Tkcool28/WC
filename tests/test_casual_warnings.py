"""Tests for the Phase 9 casual-warning consolidator.

Covers dashboard.casual_warnings.consolidate_casual_warnings — the
single source of truth for the user-facing warning on the casual
Prediction card.

The consolidator MUST:
* return at most ONE concise string,
* never leak raw internal codes (canonical=, status=, fd_id=),
* never leak raw counts (home: 429, away: 0; <5 prior matches),
* never leak calibration jargon (9,678-match, coin flip, calibration bucket),
* absorb a lower-priority calibration_caution into a higher-priority
  history_missing / limited_data warning,
* return [] for normal high-history teams (England vs Croatia case),
* never emit canonical IDs in the user-visible text.
"""
from __future__ import annotations

import pytest

from dashboard.casual_warnings import (
    _INTERNAL_PATTERNS,
    _is_calibration_caution,
    _is_history_missing,
    _is_identity_unresolved,
    _is_limited_data,
    consolidate_casual_warnings,
)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_internal_patterns_list_is_not_empty() -> None:
    assert len(_INTERNAL_PATTERNS) > 0


def test_is_identity_unresolved_canonical_none() -> None:
    assert _is_identity_unresolved("canonical=None status=identity_unresolved fd_id=42") is True


def test_is_identity_unresolved_history_missing_is_not() -> None:
    """history_missing is a resolved team with no corpus, NOT unresolved."""
    assert _is_identity_unresolved(
        "canonical=COD status=history_missing fd_id=1934 name=Congo DR"
    ) is False


def test_is_history_missing_matches_known_signals() -> None:
    assert _is_history_missing("canonical=COD status=history_missing") is True
    assert _is_history_missing("One or both teams have <5 prior matches") is True
    assert _is_history_missing("Pi-rating is essentially a coin flip") is True


def test_is_limited_data_matches_known_signals() -> None:
    assert _is_limited_data("Limited data available for this team") is True
    assert _is_limited_data("Insufficient data") is True


def test_is_calibration_caution_matches_known_signals() -> None:
    assert _is_calibration_caution("Pi-rating is overconfident at this level") is True
    assert _is_calibration_caution("9,678-match backtest") is True


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def test_consolidate_no_warnings_returns_empty() -> None:
    pred = {
        "home_team": "England", "away_team": "Croatia",
        "confidence": {"tier": "A", "warnings": []},
        "identity_warnings": [],
    }
    assert consolidate_casual_warnings(pred) == []


def test_consolidate_returns_at_most_one_string() -> None:
    """Even with multiple raw warnings, only one concise string is returned."""
    pred = {
        "home_team": "England", "away_team": "Croatia",
        "confidence": {
            "tier": "A",
            "warnings": [
                "One or both teams have <5 prior matches in training. home: 429, away: 0",
                "Pi-rating is overconfident. Raw 78% but actual hit rate is closer to 65% per 9,678-match backtest.",
            ],
        },
        "identity_warnings": [],
    }
    out = consolidate_casual_warnings(pred)
    assert isinstance(out, list)
    assert len(out) <= 1


def test_consolidate_does_not_leak_home_429_away_0() -> None:
    pred = {
        "home_team": "Portugal", "away_team": "Congo DR",
        "confidence": {
            "tier": "D",
            "warnings": [
                "One or both teams have <5 prior matches in training. home: 429, away: 0",
            ],
        },
        "identity_warnings": [],
    }
    out = consolidate_casual_warnings(pred)
    assert out, "expected at least one warning for Congo DR history-missing case"
    for w in out:
        assert "429" not in w, f"raw match count leaked: {w!r}"
        assert "0)" not in w.replace(".", "").replace(",", ""), f"raw away count leaked: {w!r}"


def test_consolidate_does_not_leak_5_prior_matches_text() -> None:
    pred = {
        "home_team": "Portugal", "away_team": "Congo DR",
        "confidence": {
            "tier": "D",
            "warnings": ["One or both teams have <5 prior matches in training. home: 429, away: 0"],
        },
        "identity_warnings": [],
    }
    out = consolidate_casual_warnings(pred)
    for w in out:
        assert "<5" not in w, f"raw '<5 prior matches' leaked: {w!r}"
        assert "prior matches" not in w.lower(), f"'prior matches' phrase leaked: {w!r}"


def test_consolidate_does_not_leak_9678_match_text() -> None:
    pred = {
        "home_team": "Portugal", "away_team": "Congo DR",
        "confidence": {
            "tier": "D",
            "warnings": [
                "Pi-rating is overconfident. Raw 78% but actual hit rate is closer to 65% per 9,678-match backtest.",
            ],
        },
        "identity_warnings": [],
    }
    out = consolidate_casual_warnings(pred)
    for w in out:
        assert "9,678" not in w, f"raw '9,678-match' leaked: {w!r}"
        assert "9678" not in w, f"raw 9678 leaked: {w!r}"
        assert "backtest" not in w.lower(), f"'backtest' jargon leaked: {w!r}"


def test_consolidate_does_not_leak_coin_flip_text() -> None:
    pred = {
        "home_team": "Portugal", "away_team": "Congo DR",
        "confidence": {
            "tier": "D",
            "warnings": ["Pi-rating is essentially a coin flip here."],
        },
        "identity_warnings": [],
    }
    out = consolidate_casual_warnings(pred)
    for w in out:
        assert "coin flip" not in w.lower(), f"'coin flip' jargon leaked: {w!r}"


def test_consolidate_does_not_leak_canonical_codes() -> None:
    pred = {
        "home_team": "Portugal", "away_team": "Congo DR",
        "confidence": {"tier": "D", "warnings": []},
        "identity_warnings": ["canonical=COD status=history_missing fd_id=1934 name=Congo DR"],
    }
    out = consolidate_casual_warnings(pred)
    for w in out:
        assert "canonical=" not in w, f"raw 'canonical=' leaked: {w!r}"
        assert "status=" not in w, f"raw 'status=' leaked: {w!r}"
        assert "fd_id=" not in w, f"raw 'fd_id=' leaked: {w!r}"
        assert "COD" not in w, f"canonical id leaked: {w!r}"
        assert "POR" not in w, f"canonical id leaked: {w!r}"


def test_consolidate_history_missing_for_cod_names_team() -> None:
    pred = {
        "home_team": "Portugal", "away_team": "Congo DR",
        "confidence": {"tier": "D", "warnings": []},
        "identity_warnings": ["canonical=COD status=history_missing fd_id=1934 name=Congo DR"],
    }
    out = consolidate_casual_warnings(pred)
    assert len(out) == 1
    assert "Congo DR" in out[0], f"warning should name the team: {out[0]!r}"
    assert "limited" in out[0].lower() or "cautious" in out[0].lower()


def test_consolidate_history_missing_does_not_emit_calibration_separately() -> None:
    """When both history_missing and calibration_caution raw warnings exist,
    only the history_missing concise message is emitted (calibration absorbed)."""
    pred = {
        "home_team": "Portugal", "away_team": "Congo DR",
        "confidence": {
            "tier": "D",
            "warnings": [
                "One or both teams have <5 prior matches. Pi-rating is essentially a coin flip here. home: 429, away: 0",
                "Pi-rating is overconfident at this level. 9,678-match backtest.",
            ],
        },
        "identity_warnings": [],
    }
    out = consolidate_casual_warnings(pred)
    assert len(out) == 1, f"expected exactly one concise message, got: {out!r}"
    assert "9,678" not in out[0]
    assert "backtest" not in out[0].lower()
    assert "calibration" not in out[0].lower()


def test_consolidate_england_croatia_returns_empty() -> None:
    """Normal high-history teams: no casual warning at all."""
    pred = {
        "home_team": "England", "away_team": "Croatia",
        "confidence": {
            "tier": "A",
            "warnings": [
                "Pi-rating is overconfident at this level.",
            ],
        },
        "identity_warnings": [],
    }
    out = consolidate_casual_warnings(pred)
    assert out == [], (
        f"high-history team should have no casual warning; got: {out!r}"
    )


def test_consolidate_identity_unresolved_names_team() -> None:
    """Identity_unresolved: name the team in the warning."""
    pred = {
        "home_team": "Atlantis FC", "away_team": "Brazil",
        "confidence": {"tier": "D", "warnings": []},
        "identity_warnings": ["canonical=None status=identity_unresolved fd_id=999999999"],
    }
    out = consolidate_casual_warnings(pred)
    assert len(out) == 1
    assert "Atlantis FC" in out[0]
    assert "couldn" in out[0].lower() or "could not" in out[0].lower()


def test_consolidate_priority_identity_beats_calibration() -> None:
    """If identity_unresolved AND calibration_caution both present, the
    identity message wins (higher priority)."""
    pred = {
        "home_team": "Atlantis", "away_team": "Brazil",
        "confidence": {
            "tier": "D",
            "warnings": ["9,678-match calibration caution"],
        },
        "identity_warnings": ["canonical=None status=identity_unresolved fd_id=99"],
    }
    out = consolidate_casual_warnings(pred)
    assert len(out) == 1
    assert "Atlantis" in out[0]
    assert "9,678" not in out[0]


def test_consolidate_calibration_only_team_names_not_required() -> None:
    """When only calibration_caution is present (no history_missing),
    emit the concise calibration message — no team name required."""
    pred = {
        "home_team": "Argentina", "away_team": "Brazil",
        "confidence": {
            "tier": "C",
            "warnings": ["Pi-rating is overconfident at this level"],
        },
        "identity_warnings": [],
    }
    out = consolidate_casual_warnings(pred)
    assert len(out) == 1
    msg = out[0].lower()
    assert "calibration" in msg or "long-term" in msg
    assert "9,678" not in out[0]
    assert "backtest" not in msg