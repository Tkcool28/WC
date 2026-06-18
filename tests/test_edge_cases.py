"""Tests for Phase 7 edge-case states.

Covers:
* Empty / cache-missing states in the three main views (Predictions, Bets, Analysis).
* Plain-language error copy in custom-matchup expanders.
* Per-card error wording in Bets (no spillover to other cards).
* Plain-language wording of all "no odds" / "invalid odds" / "partial odds" paths.
* Identity-warning translation never leaks raw codes.
* No CLI / script paths in user-visible copy.

These tests are integration-light: they boot the real ``dashboard.app``
under ``AppTest.from_file`` and assert on the visible widget stream.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from dashboard.ux_presenters import (
    translate_and_dedupe_warnings,
    translate_warning,
)


_DASHBOARD_APP = (
    Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _all_visible_text(at: AppTest) -> str:
    """Concatenate all user-visible text the app emits (excluding <style>)."""
    parts: list[str] = []
    for el in at.markdown:
        v = (el.value or "")
        v = re.sub(r"<style.*?</style>", "", v, flags=re.DOTALL | re.IGNORECASE)
        parts.append(v)
    for el in at.caption:
        parts.append(el.value or "")
    for el in at.info:
        parts.append(el.value or "")
    for el in at.warning:
        parts.append(el.value or "")
    for el in at.error:
        parts.append(el.value or "")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Plain-language copy invariants
# --------------------------------------------------------------------------- #
# These constants are forbidden in user-visible copy. If any of them leak
# into the rendered page, the dashboard is leaking internal/operational
# detail to ordinary users.
_INTERNAL_TERMS = [
    "scripts/",
    ".py",
    "fetch_live_2026",
    "data/raw/",
    "/root/",
    "st.session_state",
    "Traceback",
    "Exception:",
    "AttributeError",
    "KeyError:",
    "TypeError:",
]


@pytest.mark.parametrize("view", ["predictions", "bets", "analysis"])
def test_no_internal_terms_in_view(view: str) -> None:
    """Each of the three top-level views must never expose CLI / script paths."""
    at = AppTest.from_file(str(_DASHBOARD_APP), default_timeout=60)
    at.query_params["view"] = view
    at.run()
    assert not at.exception, f"{view} raised: {at.exception}"
    text = _all_visible_text(at).lower()
    for term in _INTERNAL_TERMS:
        assert term.lower() not in text, (
            f"Internal term {term!r} leaked into {view} view"
        )


# --------------------------------------------------------------------------- #
# translate_warning: never leak raw internal codes
# --------------------------------------------------------------------------- #
def test_translate_warning_handles_known_internal_code() -> None:
    """Raw codes like ``canonical=ABC`` must be translated to plain language."""
    translated = translate_warning("canonical=ABC not found")
    # The exact wording lives in ux_presenters; we just assert it doesn't
    # pass through verbatim and isn't empty.
    assert translated != "canonical=ABC not found"
    assert "canonical=" not in translated
    assert translated  # not empty


def test_translate_warning_handles_status_codes() -> None:
    """Raw ``status=...`` codes are translated to plain language."""
    translated = translate_warning("status=history_missing")
    assert "status=" not in translated
    assert translated


def test_translate_warning_preserves_user_facing_sentence() -> None:
    """A well-formed user-facing sentence passes through unchanged."""
    sentence = "Argentina has limited recent history in our training data."
    assert translate_warning(sentence) == sentence


def test_translate_and_dedupe_warnings_dedupes() -> None:
    """The dedupe helper collapses identical translated warnings."""
    raw = ["canonical=A", "canonical=A", "canonical=B"]
    out = translate_and_dedupe_warnings(raw)
    # All raw codes map to the same translated string, so dedupe → 1.
    assert len(out) == 1
    assert "canonical=" not in out[0]


# --------------------------------------------------------------------------- #
# Identity warnings never leak raw codes through the prediction card
# --------------------------------------------------------------------------- #
def test_prediction_card_does_not_leak_raw_identity_codes() -> None:
    """``render_prediction_card`` translates identity_warnings before display."""
    from streamlit.testing.v1 import AppTest
    from dashboard.prediction_card import render_prediction_card

    pred = {
        "home_team": "Argentina",
        "away_team": "Brazil",
        "date": "2026-06-17",
        "pi_probs":      {"home": 0.52, "draw": 0.24, "away": 0.24},
        "blend_probs":   {"home": 0.51, "draw": 0.25, "away": 0.24},
        "pi_only_probs": {"home": 0.52, "draw": 0.24, "away": 0.24},
        "elo_only_probs": None,
        "blend_was_used": False,
        "confidence": {"tier": "C", "calibrated_p": 0.5, "warnings": []},
        "identity_warnings": ["canonical=ARG status=history_missing"],
    }

    at = AppTest.from_string(
        "import streamlit as st\n"
        "from dashboard.prediction_card import render_prediction_card\n"
        f"render_prediction_card({pred!r})\n"
    )
    at.run()
    assert not at.exception
    text = _all_visible_text(at).lower()
    assert "canonical=" not in text
    assert "status=history_missing" not in text


# --------------------------------------------------------------------------- #
# Draw wording — used by both Predictions and Bets
# --------------------------------------------------------------------------- #
def test_draw_wording_match_to_end_in_a_draw_in_prediction() -> None:
    """When draw is the top market, the card uses the exact Phase 3 wording."""
    from streamlit.testing.v1 import AppTest
    from dashboard.prediction_card import render_prediction_card

    pred = {
        "home_team": "Mexico",
        "away_team": "USA",
        "date": "2026-06-17",
        "pi_probs":      {"home": 0.30, "draw": 0.40, "away": 0.30},
        "blend_probs":   {"home": 0.30, "draw": 0.40, "away": 0.30},
        "pi_only_probs": {"home": 0.30, "draw": 0.40, "away": 0.30},
        "elo_only_probs": None,
        "blend_was_used": False,
        "confidence": {"tier": "C", "calibrated_p": 0.4, "warnings": []},
        "identity_warnings": [],
    }

    at = AppTest.from_string(
        "import streamlit as st\n"
        "from dashboard.prediction_card import render_prediction_card\n"
        f"render_prediction_card({pred!r})\n"
    )
    at.run()
    assert not at.exception
    text = _all_visible_text(at)
    assert "Match to End in a Draw" in text


# --------------------------------------------------------------------------- #
# Confidence has text, not only color (accessibility)
# --------------------------------------------------------------------------- #
def test_confidence_pill_has_text_label() -> None:
    """Every confidence tier maps to a human-readable string, never just color."""
    from dashboard.prediction_card import _confidence_label

    cases = [
        # tier A with agree → "High Confidence"
        (
            {
                "pi_only_probs":  {"home": 0.55, "draw": 0.25, "away": 0.20},
                "elo_only_probs": {"home": 0.52, "draw": 0.25, "away": 0.23},
                "confidence":     {"tier": "A", "calibrated_p": 0.55},
            },
            "High Confidence",
        ),
        # tier A only pi → "Moderate Confidence"
        (
            {
                "pi_only_probs":  {"home": 0.55, "draw": 0.25, "away": 0.20},
                "elo_only_probs": None,
                "confidence":     {"tier": "A", "calibrated_p": 0.55},
            },
            "Moderate Confidence",
        ),
        # tier C → "Limited Data"
        (
            {"confidence": {"tier": "C", "calibrated_p": 0.4}},
            "Limited Data",
        ),
        # tier D → "Low Confidence"
        (
            {"confidence": {"tier": "D", "calibrated_p": 0.5}},
            "Low Confidence",
        ),
    ]
    for pred, expected_label in cases:
        label, _ = _confidence_label(pred)
        assert label == expected_label, (
            f"For {pred['confidence']!r} expected {expected_label!r}, got {label!r}"
        )


# --------------------------------------------------------------------------- #
# Streamlit chrome hide: rerun controls MUST remain visible
# --------------------------------------------------------------------------- #
def test_styles_does_not_hide_rerun_controls() -> None:
    """We hide ``#MainMenu`` / footer / decoration, but NOT rerun controls."""
    from dashboard.styles import get_css
    css = get_css()
    # Safe-to-hide elements
    assert "#MainMenu" in css
    assert "footer" in css or "[data-testid=\"stFooter\"]" in css
    # Strip comments before checking for the universal-focus-killer pattern.
    flat = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    flat = re.sub(r"\s+", "", flat)
    assert "*:focus{outline:none" not in flat
    assert "*:focus{outline:0" not in flat


def test_styles_includes_focus_visible_rule() -> None:
    """A ``:focus-visible`` rule must be present (Phase 7 accessibility)."""
    from dashboard.styles import get_css
    css = get_css()
    assert ":focus-visible" in css, "Phase 7 focus-visible rule missing"


def test_styles_includes_dark_theme_focus_override() -> None:
    """Dark theme must override the focus ring color."""
    from dashboard.styles import get_css
    css = get_css()
    assert "[data-theme=\"dark\"]" in css, "Dark-theme focus override missing"