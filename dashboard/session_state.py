"""Centralized Streamlit session state keys and accessors for the WC dashboard.

Keys are namespaced (e.g. ``"predictions.date"``, ``"bets.min_edge"``) so the
global Predictions / Bets / Analysis switcher cannot accidentally clobber
another section's state.

Phase 2 introduces this module. Phase 3 and Phase 4 will replace the
legacy ``auto_*`` keys with namespaced equivalents as the new Predictions
and Bets renderers land.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import streamlit as st


@dataclass(frozen=True)
class SessionKeys:
    """Namespaced session state keys for the dashboard sections.

    Frozen dataclass: keys are immutable and cannot be reassigned at
    runtime. The dotted-namespace convention is enforced by inspection
    (see tests in ``tests/test_dashboard_nav.py``).
    """

    # Active section (one of "🎯 Predictions" | "💰 Bets" | "🔬 Analysis").
    ACTIVE_VIEW: str = "active_view"

    # ---- shared (used by >=2 sections) ----
    # Loaded date (ISO string) — shared across Predictions / Bets / Analysis.
    SELECTED_DATE: str = "selected_date"
    # Loaded matches for selected_date (list of dicts).
    LOADED_MATCHES: str = "loaded_matches"
    # Per-match predictions (dict keyed by match_id -> predict_match dict).
    PREDICTIONS_BY_MATCH: str = "predictions_by_match"
    # Per-match market results (dict keyed by match_id -> evaluate_market dict).
    MARKET_BY_MATCH: str = "market_by_match"
    # Per-match odds (dict keyed by match_id -> {"home", "draw", "away"} strs).
    ODDS_BY_MATCH: str = "odds_by_match"

    # ---- Predictions-section keys (legacy compat) ----
    PRED_LEGACY_PICKED_DATE: str = "auto_picked_date"
    PRED_LEGACY_LOADED_DATE: str = "auto_loaded_date"
    PRED_LEGACY_MATCHES: str = "auto_matches"
    PRED_LEGACY_MIN_EDGE: str = "auto_min_edge"

    # ---- Bets-section keys (legacy compat) ----
    BETS_MIN_EDGE: str = "bets.min_edge"

    # ---- Analysis-section keys ----
    ANALYSIS_GAME: str = "analysis.game"

    # ---- Custom-matchup inputs (used by Predictions + Bets expanders) ----
    CUSTOM_HOME: str = "custom.home"
    CUSTOM_AWAY: str = "custom.away"
    CUSTOM_DATE: str = "custom.date"
    CUSTOM_HOME_ODDS: str = "custom.home_odds"
    CUSTOM_DRAW_ODDS: str = "custom.draw_odds"
    CUSTOM_AWAY_ODDS: str = "custom.away_odds"


# Module-level singleton. Import as: from dashboard.session_state import KEYS
KEYS = SessionKeys()


def get(key: str, default: Any = None) -> Any:
    """Return ``st.session_state[key]`` or ``default`` if absent.

    Thin wrapper kept for symmetry with :func:`set_` and to give the
    test suite a single seam to monkey-patch.
    """
    return st.session_state.get(key, default)


def set_(key: str, value: Any) -> None:
    """Set ``st.session_state[key] = value``.

    The trailing underscore avoids shadowing the built-in :func:`set`.
    """
    st.session_state[key] = value


def pop(key: str, default: Any = None) -> Any:
    """Pop ``key`` from ``st.session_state``, returning ``default`` if absent."""
    return st.session_state.pop(key, default)
