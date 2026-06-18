"""Unit tests for the dashboard navigation primitives (Phase 2).

These tests cover the pure-Python parts of the new nav system:

* :class:`dashboard.session_state.SessionKeys` is a frozen dataclass with
  unique, namespaced keys (no two keys collide; all keys follow the
  dotted-or-simple-string convention used by the rest of the dashboard).
* :func:`get` / :func:`set_` / :func:`pop` are thin wrappers around
  ``streamlit.session_state`` and behave correctly when a real session
  is available.
* The top-level app starts cleanly under all three view slugs
  (``predictions``, ``bets``, ``analysis``) and renders the expected
  stub content in each.

We deliberately avoid running under a real ``streamlit run`` — that is
covered by the smoke test in the Phase 2 verification gate. The
session_state helpers do not require a live ScriptRunContext for
construction or for key-list inspection; ``get``/``set_``/``pop`` do
require one and are exercised via ``streamlit.testing.v1.AppTest``.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from dashboard.session_state import (
    KEYS,
    SessionKeys,
    get,
    pop,
    set_,
)


# --------------------------------------------------------------------------- #
# Pure-Python: SessionKeys is a frozen dataclass with namespaced unique keys
# --------------------------------------------------------------------------- #
def test_session_keys_is_frozen_dataclass() -> None:
    """KEYS is a SessionKeys instance, which is a frozen dataclass."""
    assert isinstance(KEYS, SessionKeys)
    # frozen=True means attribute assignment is rejected by the dataclass.
    with pytest.raises((AttributeError, Exception)):
        KEYS.ACTIVE_VIEW = "mutated"  # type: ignore[misc]


def test_session_keys_attribute_constants_are_strings() -> None:
    """Every key constant is a non-empty string."""
    for name in vars(KEYS):
        value = getattr(KEYS, name)
        assert isinstance(value, str), f"{name} -> {value!r} is not a str"
        assert value, f"{name} is empty"


def test_session_keys_are_unique() -> None:
    """No two SessionKeys fields share the same value (no key collisions)."""
    values = [getattr(KEYS, n) for n in vars(KEYS)]
    assert len(values) == len(set(values)), (
        f"Duplicate keys in SessionKeys: {values}"
    )


def test_session_keys_follow_namespace_convention() -> None:
    """All keys are snake_case, dotted, or single-word lowercase tokens.

    We accept either:
      * dotted namespace: ``"section.something"`` e.g. ``"bets.min_edge"``
      * snake_case / single-word: ``"active_view"``, ``"auto_matches"``
    The only disallowed characters are spaces and uppercase letters.
    """
    pattern = re.compile(r"^[a-z0-9_]+(?:\.[a-z0-9_]+)?$")
    for name in vars(KEYS):
        value = getattr(KEYS, name)
        assert pattern.match(value), (
            f"Key {name}={value!r} does not match namespaced convention"
        )


# --------------------------------------------------------------------------- #
# AppTest-backed: get/set_/pop round-trip through a real Streamlit session
# --------------------------------------------------------------------------- #
def test_session_state_helpers_round_trip() -> None:
    """``set_`` writes, ``get`` reads, ``pop`` removes a value."""
    at = AppTest.from_string(
        """
        import streamlit as st
        from dashboard.session_state import KEYS, get, set_, pop

        set_(KEYS.BETS_MIN_EDGE, 0.05)
        st.session_state["__probe__"] = get(KEYS.BETS_MIN_EDGE)
        st.session_state["__popped__"] = pop(KEYS.BETS_MIN_EDGE, default="missing")
        st.session_state["__after__"] = get(KEYS.BETS_MIN_EDGE, default="gone")
        """
    )
    at.run()
    # The script ran without raising.
    assert not at.exception
    assert at.session_state["__probe__"] == 0.05
    assert at.session_state["__popped__"] == 0.05
    assert at.session_state["__after__"] == "gone"


# --------------------------------------------------------------------------- #
# AppTest-backed: full app boots in all three top-level views
# --------------------------------------------------------------------------- #
# Path to dashboard/app.py for AppTest.from_file. Anchored at repo root.
_DASHBOARD_APP = (
    Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
)


def test_app_renders_predictions_view_by_default() -> None:
    """Default landing is the Phase 3 Predictions renderer (model-only).

    The new view shows a date picker, a single primary "Show Predictions"
    button, and a closed Custom-matchup expander. The legacy
    "Load games" button from Phase 2 is gone — Phase 3 collapsed the
    two-step flow (Load games → per-game Run analysis) into a single
    tap.
    """
    at = AppTest.from_file(str(_DASHBOARD_APP), default_timeout=60)
    at.run()
    assert not at.exception, f"app raised in default view: {at.exception}"
    # Phase 3 single primary button must be present.
    button_texts = [b.label for b in at.button]
    assert any("Show Predictions" in (t or "") for t in button_texts), (
        f"Phase 3 'Show Predictions' button missing; got: {button_texts!r}"
    )
    # The "min edge" slider that Phase 2's auto-populate view exposed
    # must NOT leak into the Phase 3 Predictions view.
    slider_texts = [s.label for s in at.slider]
    assert not any("Minimum edge" in (t or "") for t in slider_texts), (
        f"min-edge slider leaked into Predictions view: {slider_texts!r}"
    )


def test_app_renders_bets_view_via_query_param() -> None:
    """``?view=bets`` lands on the real Phase 4 Bets renderer.

    Phase 2 shipped this as a stub that delegated to the legacy Manual
    form. Phase 4 replaces the stub with a real mobile-first,
    odds-gated Bets experience:

      * Date picker + a single primary **"Show Bets"** button
      * One Bets card per matchup with three odds inputs and a
        **"Check Betting Value"** button
      * A closed-by-default **"Advanced settings"** expander that
        hides the ``min_edge`` slider
      * A closed-by-default **"Custom bet"** expander for non-2026
        fixtures

    We assert all of these here, plus that the legacy "Run Analysis"
    button (Phase 1 / 2 manual flow) is gone.
    """
    at = AppTest.from_file(str(_DASHBOARD_APP), default_timeout=60)
    at.query_params["view"] = "bets"
    at.run()
    assert not at.exception, f"app raised in bets view: {at.exception}"
    # Single primary "Show Bets" button is present.
    button_texts = [b.label for b in at.button]
    assert any("Show Bets" in (t or "") for t in button_texts), (
        f"Phase 4 'Show Bets' button missing; got: {button_texts!r}"
    )
    # Legacy "Run Analysis" button (Phase 1 / 2 manual form) is gone.
    assert not any("Run Analysis" in (t or "") for t in button_texts), (
        f"Legacy 'Run Analysis' button leaked into Bets view: "
        f"{button_texts!r}"
    )
    # Both the Advanced settings and Custom bet expanders are present.
    expander_labels = [e.label or "" for e in at.expander]
    assert any("Advanced settings" in t for t in expander_labels), (
        f"'Advanced settings' expander missing from Bets view: "
        f"{expander_labels!r}"
    )
    assert any("Custom bet" in t for t in expander_labels), (
        f"'Custom bet' expander missing from Bets view: "
        f"{expander_labels!r}"
    )


def test_app_renders_analysis_view_via_query_param() -> None:
    """``?view=analysis`` lands on the real Phase 5 Analysis renderer.

    Phase 2 shipped this as a stub with a "coming in Phase 5" notice.
    Phase 5 replaces the stub with a real, mobile-first, technical
    Analysis experience.  This test asserts the new flow:

      * Date picker is present (the Analysis view shares the same
        SELECTED_DATE as Predictions / Bets).
      * A single primary **"Show Analysis"** button is visible.
      * All 11 expanders are present.
      * **Prediction Details** is the only expander that opens by
        default; the other 10 are collapsed.
    """
    at = AppTest.from_file(str(_DASHBOARD_APP), default_timeout=60)
    at.query_params["view"] = "analysis"
    at.run()
    assert not at.exception, f"app raised in analysis view: {at.exception}"
    # Single primary "Show Analysis" button is present.
    button_texts = [b.label for b in at.button]
    assert any("Show Analysis" in (t or "") for t in button_texts), (
        f"Phase 5 'Show Analysis' button missing; got: {button_texts!r}"
    )
    # Legacy "Phase 5 lands here" stub is gone.
    info_texts = [i.value for i in at.info]
    assert not any("Phase 5 lands here" in (t or "") for t in info_texts), (
        f"Phase 5 stub still leaking into Analysis view: {info_texts!r}"
    )
    # All 11 Analysis expanders are present.
    expander_labels = [e.label or "" for e in at.expander]
    required = [
        "Prediction Details",
        "Model Breakdown",
        "Pi-Rating",
        "Elo Rating",
        "Blend",
        "Market Comparison",
        "Poisson View",
        "Squad Context",
        "Group Context",
        "Calibration and Data Quality",
        "Raw Diagnostics",
    ]
    for needle in required:
        assert any(needle in t for t in expander_labels), (
            f"Required expander {needle!r} missing from Analysis view: "
            f"{expander_labels!r}"
        )
    # Prediction Details is the only default-open expander.
    open_expanders = [
        e.label for e in at.expander
        if getattr(e.proto, "expanded", False)
    ]
    assert open_expanders == ["🎯 Prediction Details"], (
        f"Expected only 'Prediction Details' to default-open; got "
        f"{open_expanders!r}"
    )
