"""Tests for Phase 7 accessibility invariants.

The dashboard must remain usable by keyboard-only users and screen-reader
users. These tests assert the structural and stylistic commitments the
rearchitecture made:

* Every interactive widget has a ``label=`` (not relying on placeholder
  alone).
* CSS contains a visible ``:focus-visible`` rule.
* CSS does NOT include ``*:focus { outline: none; }`` (the universal
  focus killer).
* Body text never drops below the 0.85em accessibility floor.
* Dark-theme focus override exists.
* Emoji supplements (not replaces) meaning — every emoji has text.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


_DASHBOARD_APP = (
    Path(__file__).resolve().parent.parent / "dashboard" / "app.py"


# --------------------------------------------------------------------------- #
# CSS invariants
# --------------------------------------------------------------------------- #
)


def test_css_has_focus_visible_rule() -> None:
    """A ``:focus-visible`` rule must be present in the stylesheet."""
    from dashboard.styles import get_css
    css = get_css()
    assert ":focus-visible" in css


def test_css_has_no_universal_focus_killer() -> None:
    """The dashboard must not contain ``*:focus { outline: none; }``.

    That pattern is the universal accessibility killer — keyboard users
    lose all focus indication. Phase 7 explicitly preserved focus rings.
    The check strips CSS comments first so the docstring's reference to
    the pattern doesn't trigger a false positive.
    """
    from dashboard.styles import get_css
    css = get_css()
    flat = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    flat = re.sub(r"\s+", "", flat)
    assert "*:focus{outline:none" not in flat
    assert "*:focus{outline:0" not in flat


def test_css_dark_theme_focus_override_present() -> None:
    """Dark theme must override the focus ring color for visibility."""
    from dashboard.styles import get_css
    css = get_css()
    assert "[data-theme=\"dark\"]" in css


def test_css_body_text_meets_floor() -> None:
    """Body text must not drop below the 0.85em accessibility floor.

    The Phase 7 worker explicitly raised the floor to 1em with ``!important``
    on the common text containers.
    """
    from dashboard.styles import get_css
    css = get_css()
    # Phase 7 rule: ``.stMarkdown p, .stCaption, .stInfo, ... { font-size: 1em !important }``
    assert ".stMarkdown p" in css
    # We look for the floor (1em) being explicitly set; rule below 0.85em would
    # be an accessibility regression.
    for match in re.finditer(r"font-size:\s*([\d.]+)em", css):
        em = float(match.group(1))
        assert em >= 0.85, f"CSS font-size {em}em is below the 0.85em floor"


def test_css_keeps_running_controls_visible() -> None:
    """The Phase 6 hide-list targets only safe chrome — NOT rerun controls.

    We assert that ``#MainMenu`` / footer / decoration are explicitly
    hidden, while the rerun controls are not mentioned in a ``display: none``
    or ``visibility: hidden`` rule.
    """
    from dashboard.styles import get_css
    css = get_css()
    # Safe-to-hide elements
    assert "#MainMenu" in css
    assert "footer" in css or "[data-testid=\"stFooter\"]" in css
    # The rerun control class on Streamlit is typically ``[data-testid="stStatusWidget"]``
    # or similar — we just assert we haven't explicitly targeted it for hiding.
    assert "stStatusWidget" not in css or "display: none" not in css.split("stStatusWidget")[1].split("}")[0]


def test_css_48px_min_height_on_inputs() -> None:
    """Inputs should hit the 48px iOS/Material touch-target floor."""
    from dashboard.styles import get_css
    css = get_css()
    # Look for ``min-height: 48px`` near a text/number input selector.
    assert "stTextInput" in css or "stNumberInput" in css
    assert "min-height: 48px" in css


def test_css_primary_button_has_visible_focus() -> None:
    """Primary buttons must have a visible focus state."""
    from dashboard.styles import get_css
    css = get_css()
    assert "primary" in css
    assert ":focus-visible" in css or ":focus" in css


# --------------------------------------------------------------------------- #
# AppTest: every interactive widget has a label
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("view", ["predictions", "bets", "analysis"])
def test_no_unlabeled_widgets_in_view(view: str) -> None:
    """Every text_input / selectbox / slider / date_input has a label.

    We assert this for each top-level view at boot time (before any user
    interaction). The Custom-matchup expanders may add more widgets when
    opened, but the top-level widgets must be labeled.
    """
    at = AppTest.from_file(str(_DASHBOARD_APP), default_timeout=60)
    at.query_params["view"] = view
    at.run()
    assert not at.exception, f"{view} raised: {at.exception}"

    # Every text_input has a label.
    for inp in at.text_input:
        assert inp.label, (
            f"Unlabeled text_input in {view} view (placeholder={inp.placeholder!r})"
        )
    # Every selectbox has a label.
    for sel in at.selectbox:
        assert sel.label, f"Unlabeled selectbox in {view} view"
    # Every slider has a label.
    for sl in at.slider:
        assert sl.label, f"Unlabeled slider in {view} view"
    # Every date_input has a label.
    for di in at.date_input:
        assert di.label, f"Unlabeled date_input in {view} view"


# --------------------------------------------------------------------------- #
# AppTest: every button has text (not emoji-only)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("view", ["predictions", "bets", "analysis"])
def test_buttons_have_text_labels(view: str) -> None:
    """Buttons have text content (emoji + word), not bare icons."""
    at = AppTest.from_file(str(_DASHBOARD_APP), default_timeout=60)
    at.query_params["view"] = view
    at.run()
    assert not at.exception
    for btn in at.button:
        label = btn.label or ""
        # Strip emojis (very rough — match any non-ASCII character)
        text_only = label.encode("ascii", "ignore").decode().strip()
        assert text_only, (
            f"Button has no ASCII label in {view} view: {label!r}"
        )


# --------------------------------------------------------------------------- #
# Confidence pill has text, not only color
# --------------------------------------------------------------------------- #
def test_confidence_label_returns_text_not_color() -> None:
    """``_confidence_label`` returns a human-readable text label.

    A color key is fine as a *secondary* signal, but the primary output
    must be text so screen readers can announce it.
    """
    from dashboard.prediction_card import _confidence_label
    label, _ = _confidence_label({"confidence": {"tier": "A", "calibrated_p": 0.55}})
    assert isinstance(label, str)
    assert label  # non-empty
    assert not label.startswith("#")  # not a hex color


# --------------------------------------------------------------------------- #
# Best Value / No Clear Value have distinct text
# --------------------------------------------------------------------------- #
def test_best_value_and_no_clear_value_use_distinct_text() -> None:
    """``_render_best_value`` and ``_render_no_clear_value`` emit different text."""
    from streamlit.testing.v1 import AppTest
    from dashboard.bet_card import _render_best_value, _render_no_clear_value

    # Best value case.
    at1 = AppTest.from_string(
        "from dashboard.bet_card import _render_best_value\n"
        "_render_best_value('Argentina', -150)\n"
    )
    at1.run()
    best_text = " ".join((m.value or "") for m in at1.markdown)

    # No clear value case.
    at2 = AppTest.from_string(
        "from dashboard.bet_card import _render_no_clear_value\n"
        "_render_no_clear_value()\n"
    )
    at2.run()
    no_text = " ".join((m.value or "") for m in at2.markdown)

    # Both must emit some text.
    assert best_text.strip()
    assert no_text.strip()
    # And they must differ.
    assert "Best Value" in best_text or "best value" in best_text.lower()
    assert "No Clear Value" in no_text