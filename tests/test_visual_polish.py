"""Phase 6 — Mobile-first visual system polish.

These tests guard the *visual layer* — i.e. the CSS payload that the
dashboard injects on every page — and assert that the dashboard still
boots cleanly across all three views after the polish pass.

Why CSS-level tests and not a per-pixel screenshot test?
   The dashboard runs in headless Streamlit tests which don't render to
   a real browser.  What we CAN assert reliably is:

   1. The CSS payload contains the stable, scoped selectors a mobile
      user actually needs (segmented control padding, 48 px inputs,
      unified card class, no-scroll rules, hide-streamlit-chrome
      rules).
   2. Each of the three top-level views boots without exceptions, so
      the CSS injection didn't break the Streamlit script-run.
   3. The Predictions / Bets / Analysis views still emit the surface
      the existing tests already verify (no regression).

The tests deliberately avoid asserting exact colours / pixel values
because those are theme-dependent.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


# --------------------------------------------------------------------------- #
# Path constants
# --------------------------------------------------------------------------- #
_REPO = Path(__file__).resolve().parent.parent
_DASHBOARD_APP = _REPO / "dashboard" / "app.py"


# --------------------------------------------------------------------------- #
# Helpers — text extractors (copied pattern from test_predictions_view.py so
# the new tests don't accidentally diverge from the established convention)
# --------------------------------------------------------------------------- #
def _css_payload(at: AppTest) -> str:
    """Return the full text of the <style> block the dashboard injected.

    Streamlit emits the CSS payload via ``st.markdown`` with
    ``unsafe_allow_html=True``, which lands in ``at.markdown``.  We
    pull out the first ``<style>...</style>`` block and return its
    body — that is what the live browser will see.
    """
    for el in at.markdown:
        v = el.value or ""
        m = re.search(r"<style[^>]*>(.*?)</style>", v, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1)
    return ""


def _visible_text(at: AppTest) -> str:
    """Concatenate user-visible text (markdown + caption + info + warning + error)."""
    parts: list[str] = []
    for el in at.markdown:
        v = (el.value or "")
        # Strip the <style> block — it's not user-visible.
        v = re.sub(r"<style.*?</style>", "", v, flags=re.DOTALL | re.IGNORECASE)
        parts.append(v)
    for el in at.caption:
        parts.append((el.value or ""))
    for el in at.info:
        parts.append((el.value or ""))
    for el in at.warning:
        parts.append((el.value or ""))
    for el in at.error:
        parts.append((el.value or ""))
    return "\n".join(parts)


def _boot_app(view: str) -> AppTest:
    """Boot the dashboard in the given ``?view=...`` mode."""
    at = AppTest.from_file(str(_DASHBOARD_APP), default_timeout=60)
    at.query_params["view"] = view
    at.run()
    return at


# --------------------------------------------------------------------------- #
# 1) CSS payload presence + structure
# --------------------------------------------------------------------------- #
def test_styles_module_emits_a_non_empty_css_payload() -> None:
    """``dashboard.styles.get_css()`` returns a non-trivial CSS string.

    This is the surface unit tests use; the dashboard itself uses
    :func:`dashboard.styles.inject_css` to push the same payload into
    the page.  Both paths must agree.
    """
    from dashboard.styles import get_css
    css = get_css()
    assert isinstance(css, str)
    assert len(css) > 500, (
        "CSS payload suspiciously small — Phase 6 should have grown it"
    )


def test_css_payload_contains_unified_card_classes() -> None:
    """The Phase 6 unified-card class system is in the injected stylesheet."""
    at = _boot_app("predictions")
    assert not at.exception, f"app raised: {at.exception}"
    css = _css_payload(at)
    assert css, "Expected the dashboard to inject a <style> block"
    # All three card classes must be present so the views can opt-in.
    for cls in ("wc-card", "wc-prediction-card", "wc-bet-card"):
        assert f".{cls}" in css, (
            f"Missing card CSS class .{cls} in injected stylesheet"
        )


def test_css_payload_targets_segmented_control_for_mobile_padding() -> None:
    """The top-level nav (segmented_control) is styled for mobile.

    Streamlit ≥ 1.58 emits a ``[data-testid="stSegmentedControl"]``
    wrapper.  Phase 6 must pad it and emphasise the active option;
    otherwise the nav looks cramped on 360 / 390 / 430 px.
    """
    at = _boot_app("predictions")
    assert not at.exception, f"app raised: {at.exception}"
    css = _css_payload(at)
    assert "stSegmentedControl" in css, (
        "Expected segmented_control selector in CSS payload"
    )
    # Active-state emphasis must be present (background or aria-checked).
    assert ("aria-checked=\"true\"" in css) or ("data-checked=\"true\"" in css), (
        "Expected an active-state rule for the segmented_control"
    )


def test_css_payload_raises_inputs_to_48px_min_height() -> None:
    """text_inputs and number_inputs get the iOS/Material 48 px floor."""
    at = _boot_app("predictions")
    assert not at.exception, f"app raised: {at.exception}"
    css = _css_payload(at)
    # Both input selectors must appear with a 48 px min-height rule.
    assert 'data-testid="stTextInput"' in css, (
        "Expected stTextInput selector in CSS payload"
    )
    assert 'data-testid="stNumberInput"' in css, (
        "Expected stNumberInput selector in CSS payload"
    )
    # The 48 px floor must be present in the same payload.  We look
    # for any ``min-height: 48px`` line; the input rule is one of
    # several 48 px rules in the payload (primary buttons get the
    # same floor).
    assert re.search(r"min-height:\s*48px\s*!important", css), (
        "Expected a 48 px min-height rule somewhere in the CSS payload"
    )


def test_css_payload_hides_only_safe_streamlit_chrome() -> None:
    """``#MainMenu``, footer, and the ``stDecoration`` badge are hidden.

    The hide-pattern must NOT touch the running-man / rerun controls.
    """
    at = _boot_app("predictions")
    assert not at.exception, f"app raised: {at.exception}"
    css = _css_payload(at)
    # Required hides
    assert "#MainMenu" in css, "Expected #MainMenu hide rule"
    assert "footer" in css, "Expected footer hide rule"
    assert "stDecoration" in css, "Expected stDecoration hide rule"
    # We must not have hidden rerun controls; we never added a rule
    # for them, but defensively assert the payload doesn't reference
    # the rerun widget's testid with ``display: none``.
    assert not re.search(
        r"stRerun|rerun-button.*display:\s*none",
        css,
        flags=re.IGNORECASE,
    ), "CSS must not hide the rerun control"


def test_css_payload_caps_block_container_width_to_prevent_horizontal_scroll() -> None:
    """A 360 px viewport must not introduce a horizontal scrollbar.

    The Phase 6 stylesheet caps ``.main .block-container`` width and
    forces ``min-width: 0`` on flex children.
    """
    at = _boot_app("predictions")
    assert not at.exception, f"app raised: {at.exception}"
    css = _css_payload(at)
    assert ".main .block-container" in css, (
        "Expected .main .block-container rule in CSS payload"
    )
    assert "min-width: 0" in css, (
        "Expected min-width: 0 rule to prevent flex overflow"
    )


def test_css_payload_has_focus_state_for_why_popover_bubble() -> None:
    """The Why popover's focus ring is visible (3:1 contrast floor)."""
    at = _boot_app("predictions")
    assert not at.exception, f"app raised: {at.exception}"
    css = _css_payload(at)
    # Phase 3 already styled the bubble; Phase 6 adds :focus-visible.
    assert "stPopover" in css, "Expected stPopover selector in CSS payload"
    assert re.search(
        r"stPopover.*?button:focus-visible",
        css,
        flags=re.DOTALL,
    ), "Expected a focus-visible rule for the popover button"


def test_css_payload_preserves_phase_3_and_phase_4_hooks() -> None:
    """The polish pass must not have removed Phase 3 / Phase 4 rules.

    Regression guard for downstream test contracts that assert these
    class names are still in the injected stylesheet.
    """
    at = _boot_app("predictions")
    assert not at.exception, f"app raised: {at.exception}"
    css = _css_payload(at)
    for needle in (
        "wc-confidence-pill",   # Phase 3
        "wc-best-value",         # Phase 4
        "wc-no-value-badge",     # Phase 4
        "wc-mlr-headline",       # Phase 4
        "wc-edge",               # Phase 4
    ):
        assert needle in css, (
            f"Phase 6 dropped the .{needle} CSS hook"
        )


# --------------------------------------------------------------------------- #
# 2) Each top-level view boots without exceptions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("view", ["predictions", "bets", "analysis"])
def test_app_boots_cleanly_in_every_view(view: str) -> None:
    """The CSS injection must not break the script run for any tab.

    Phase 6 added a lot of new rules — if any of them referenced a
    non-existent selector and Streamlit's CSS parser choked, the run
    would still succeed but downstream widgets could be unstyled.
    This test catches a hard failure: the script raised an exception.
    """
    at = _boot_app(view)
    assert not at.exception, (
        f"App raised in ?view={view}: {at.exception}"
    )


def test_segmented_control_widget_is_emitted_with_all_three_options() -> None:
    """The top-level nav (segmented_control) shows Predictions / Bets / Analysis.

    Mobile users rely on the segmented_control being present and
    correctly labelled.  Phase 6 styles this widget but must not
    remove it.
    """
    at = _boot_app("predictions")
    assert not at.exception, f"app raised: {at.exception}"
    # AppTest exposes ``at.segmented_control`` (a list of ButtonGroup).
    assert len(at.segmented_control) >= 1, (
        "Expected the dashboard to render at least one segmented_control"
    )
    sc = at.segmented_control[0]
    # The options list should contain all three view labels.  Use
    # substring match so emoji / label drift don't break the test.
    opts_blob = " | ".join(sc.options or [])
    for needle in ("Predictions", "Bets", "Analysis"):
        assert needle in opts_blob, (
            f"Expected '{needle}' in segmented_control options; got {opts_blob!r}"
        )


def test_css_payload_is_injected_as_a_style_markdown_element() -> None:
    """The dashboard injects the CSS payload as the first markdown element.

    Some downstream tests parse the first ``at.markdown`` entry to find
    the ``<style>`` block.  Phase 6 must keep emitting the CSS through
    the same surface so those tests don't regress.
    """
    at = _boot_app("predictions")
    assert not at.exception, f"app raised: {at.exception}"
    assert at.markdown, "Expected at least one markdown element"
    first = at.markdown[0].value or ""
    assert "<style" in first.lower(), (
        "Expected the first markdown element to contain a <style> block "
        "so the dashboard's CSS is actually emitted"
    )


# --------------------------------------------------------------------------- #
# 3) No regressions in user-visible text
# --------------------------------------------------------------------------- #
def test_predictions_view_still_renders_headlines_and_confidence_pill() -> None:
    """Sanity: the predictions view surface is unchanged after Phase 6."""
    at = _boot_app("predictions")
    assert not at.exception, f"app raised: {at.exception}"
    text = _visible_text(at)
    assert "Most Likely Result" in text, (
        "Predictions view dropped 'Most Likely Result' header"
    )
    assert "wc-confidence-pill" in text, (
        "Predictions view dropped the wc-confidence-pill class hook"
    )


def test_bets_view_still_renders_most_likely_and_value_blocks() -> None:
    """Sanity: the bets view surface is unchanged after Phase 6."""
    at = _boot_app("bets")
    assert not at.exception, f"app raised: {at.exception}"
    text = _visible_text(at)
    assert "Most Likely Result" in text, (
        "Bets view dropped 'Most Likely Result' header"
    )
    # The wc-best-value / wc-no-value-badge classes may not appear
    # until the user submits odds, so we don't assert them here.
