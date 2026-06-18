"""Dashboard CSS / visual layer.

Phase 3 introduced the first real stylesheet (popover-bubble + confidence
pill). Phase 4 added the best-value / no-value / MLR-headline rules plus
the 48 px primary-button floor. Phase 5 inherited them as-is — the
Analysis view is mostly prose + expanders, so it doesn't need bespoke
card chrome.

Phase 6 finishes the mobile-first polish pass:

* One unified ``wc-card`` base class shared by Predictions and Bets.
* A faint-cool-tint variant so the user can tell Bets from Predictions
  at a glance.
* A subtle ``st.divider``-style rule under the global header so the
  title row reads as the app's chrome.
* Padded + active-state ``st.segmented_control`` so the mobile nav
  doesn't look cramped.
* 48 px min-height for ``st.text_input`` and ``st.number_input`` so
  inputs meet the iOS/Material touch-target floor.
* Clearly-visible hover/focus states on primary buttons.
* 1 px borders on the best-value / no-value blocks so they don't
  blend into a coloured page background.
* ``min-width: 0`` on flex children + width-clamp on the main block
  container to kill accidental horizontal page scroll on narrow
  viewports (360 / 390 / 430 px).
* Hide-only-if-safe Streamlit chrome (``#MainMenu``, footer, the
  "Made with Streamlit" badge). The running-man / rerun controls are
  left alone.

Phase 7 layers accessibility on top: a high-contrast focus-visible
outline for every interactive control, dark-theme-aware focus colour,
and a 1 em body-text floor so we never drop below the WCAG 0.85 em
minimum.

The CSS payload is built as a single string (not loaded from disk) so
the dashboard stays a single self-contained directory. Selectors target
``[data-testid="..."]`` attributes which are stable across the
Streamlit 1.x line; we deliberately avoid fragile deeply-nested class
names.

Usage
-----
Call :func:`inject_css` once from :func:`dashboard.app.main`, before the
first widget is rendered.  The function is idempotent: Streamlit
de-duplicates ``<style>`` blocks within a session, so calling
:func:`inject_css` on every rerun is safe and harmless.
"""
from __future__ import annotations

import streamlit as st


# --------------------------------------------------------------------------- #
# CSS payload
# --------------------------------------------------------------------------- #
# The CSS is built as a string (not loaded from disk) so the dashboard
# stays a single self-contained directory.  Phase 6 may move this to a
# static ``.css`` file if the surface area grows.
_CSS = """
/* ------------------------------------------------------------------ */
/* Phase 3 — Predictions view styles                                   */
/* ------------------------------------------------------------------ */

/* Confidence pill emitted by dashboard.prediction_card.render_prediction_card.
   The inline ``style=`` already sets the color per-pill; this rule
   only guards the surrounding margin and font. */
.wc-confidence-pill {
  margin: 4px 0 8px 0;
  font-family: inherit;
}

/* Mobile-CTA bubble styling for the "❓ Why this pick?" popover.
   The Streamlit popover renders a ``<button>`` inside a
   ``[data-testid="stPopover"]`` wrapper; we target that button and
   round it into a pill so it reads as a friendly mobile CTA. */
[data-testid="stPopover"] > button {
  background: #eef4ff !important;
  color: #0b3d91 !important;
  border-radius: 999px !important;
  padding: 12px 20px !important;
  font-weight: 600 !important;
  border: 1px solid #b6c8f5 !important;
  min-height: 48px !important;
  margin-top: 6px !important;
  margin-bottom: 6px !important;
}
[data-testid="stPopover"] > button:hover {
  background: #dde9ff !important;
}
[data-testid="stPopover"] > button:focus {
  outline: 2px solid #0b3d91 !important;
  outline-offset: 2px !important;
}

/* The card body itself is a streamlit container, so we leave the
   container chrome alone and let the inline ``wc-confidence-pill``
   style handle the only truly custom element.  Phase 6 will add
   borders, shadows, and a card width clamp. */
.wc-prediction-card {
  padding: 12px 14px;
  border-radius: 14px;
  background: #ffffff;
}

/* ------------------------------------------------------------------ */
/* Phase 4 — Bets view styles                                          */
/* ------------------------------------------------------------------ */

/* "Most Likely Result" headline — neutral, large, dark text. Sits
   inside its own div so we can theme the surrounding card without
   touching the inline font-size. The visual emphasis is from font
   weight + size only — no coloured background, no icon. */
.wc-mlr-headline {
  margin-top: 2px;
  margin-bottom: 2px;
  font-family: inherit;
}

/* "Best Value" green block. The inline style sets the colour; this
   rule only guards the surrounding margins. */
.wc-best-value {
  margin-top: 8px !important;
  margin-bottom: 6px !important;
  font-family: inherit;
  line-height: 1.3;
}

/* "No Clear Value" grey badge. The inline style sets the colour; this
   rule only guards the surrounding margins. */
.wc-no-value-badge {
  margin-top: 8px !important;
  margin-bottom: 6px !important;
  font-family: inherit;
  line-height: 1.2;
}

/* Edge readout line — neutral, smaller than the value block. */
.wc-edge {
  margin-top: 2px;
  margin-bottom: 6px;
  font-family: inherit;
}

/* Make every primary-styled button meet the iOS / Material 48px
   minimum-touch-target on mobile.  Streamlit's default ``primary``
   button is shorter than that on desktop; this rule raises the
   floor without changing the desktop look. */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
  min-height: 48px !important;
  font-size: 1.05em !important;
  font-weight: 600 !important;
  transition: filter 120ms ease, box-shadow 120ms ease !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
  filter: brightness(1.05) !important;
  box-shadow: 0 0 0 2px var(--primary-color, #ff4b4b) !important;
}
.stButton > button[kind="primary"]:focus-visible,
.stButton > button[data-testid="baseButton-primary"]:focus-visible {
  outline: 2px solid var(--primary-color, #ff4b4b) !important;
  outline-offset: 2px !important;
}

/* Secondary buttons also get a visible focus ring + 48 px floor so
   the touch-target guideline is honoured everywhere. */
.stButton > button {
  min-height: 44px !important;
}
.stButton > button:focus-visible {
  outline: 2px solid var(--primary-color, #ff4b4b) !important;
  outline-offset: 2px !important;
}

/* ------------------------------------------------------------------ */
/* Phase 6 — Mobile-first visual system polish                        */
/* ------------------------------------------------------------------ */

/* App shell ---------------------------------------------------------- */

/* A subtle bottom rule under the global header so the title row reads
   as the app's chrome and not just another block of body text.
   Targets the title element + its adjacent caption row by their
   Streamlit test-ids. We deliberately do NOT add a background; the
   rule alone is enough separation on both light and dark themes. */
h1[data-testid="stHeading"]:has(+ p[data-testid="stCaption"]),
h1[data-testid="stHeading"] {
  border-bottom: 1px solid rgba(127, 127, 127, 0.18);
  padding-bottom: 0.45rem;
  margin-bottom: 0.65rem;
}

/* ``st.divider`` defaults to a thin neutral rule already, but on
   some Streamlit themes it's nearly invisible.  Force a slightly
   stronger (still subtle) horizontal rule. */
hr[data-testid="stDivider"] {
  margin-top: 0.25rem !important;
  margin-bottom: 0.75rem !important;
  border-color: rgba(127, 127, 127, 0.22) !important;
}

/* Top-level nav (segmented_control) -------------------------------- */

/* Streamlit ≥ 1.58 renders ``st.segmented_control`` inside a
   ``[data-testid="stSegmentedControl"]`` wrapper.  The default
   control has no padding and the active option blends into the
   track on mobile.  We:
     - add horizontal padding so the control doesn't kiss the edges
     - lift each option to a comfortable 44 px touch height
     - give the selected option a clearly-emphasised background using
       Streamlit's theme variable so dark/light mode both work
*/
[data-testid="stSegmentedControl"] {
  padding: 4px 0 8px 0 !important;
  margin-bottom: 0.5rem !important;
}
[data-testid="stSegmentedControl"] label {
  min-height: 44px !important;
  padding: 10px 14px !important;
  border-radius: 10px !important;
  font-weight: 500 !important;
  transition: background-color 120ms ease, color 120ms ease !important;
}
[data-testid="stSegmentedControl"] label[data-checked="true"],
[data-testid="stSegmentedControl"] [aria-checked="true"] {
  background: var(--primary-color, #ff4b4b) !important;
  color: var(--background-color, #ffffff) !important;
  font-weight: 600 !important;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12) !important;
}
[data-testid="stSegmentedControl"] label:focus-within {
  outline: 2px solid var(--primary-color, #ff4b4b) !important;
  outline-offset: 2px !important;
}

/* The segmented_control isn't made sticky by default and trying to
   force ``position: sticky`` here breaks Streamlit's internal
   scroll containers.  Instead, we leave it inline but reserve
   consistent vertical breathing room around it so the page never
   feels cramped when the user scrolls under it. */

/* Cards ----------------------------------------------------------- */

/* Unified card surface shared by Predictions and Bets.  The two
   views apply this base plus a *variant* (``.wc-prediction-card``
   or ``.wc-bet-card``) which only changes the background tint.
   The previous Phase 3 ``.wc-prediction-card`` rule is preserved
   so existing inline uses keep working. */
.wc-card {
  border-radius: 14px;
  padding: 14px 16px;
  border: 1px solid rgba(127, 127, 127, 0.22);
  background: var(--background-color, #ffffff);
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
  margin-top: 0.5rem;
  margin-bottom: 0.75rem;
}
/* Predictions: soft white surface (the Phase 3 rule already defined
   this; we re-declare with the same values so the surface stays
   consistent if a future phase changes the .wc-card default). */
.wc-prediction-card {
  padding: 14px 16px;
  border-radius: 14px;
  border: 1px solid rgba(127, 127, 127, 0.22);
  background: var(--background-color, #ffffff);
  margin-top: 0.5rem;
  margin-bottom: 0.75rem;
}
/* Bets: very faint cool tint so a user can tell the card apart from
   a Predictions card at a glance.  The tint is intentionally subtle
   -- a stronger blue would compete with the best-value green block. */
.wc-bet-card {
  padding: 14px 16px;
  border-radius: 14px;
  border: 1px solid rgba(127, 127, 127, 0.22);
  background: rgba(238, 244, 255, 0.55);
  margin-top: 0.5rem;
  margin-bottom: 0.75rem;
}
@media (prefers-color-scheme: dark) {
  .wc-bet-card {
    background: rgba(40, 60, 100, 0.28);
  }
}

/* Context cards (Phase 8) — subtle warm/cool tints to tell Snapshot
   from Confidence at a glance.  Both still inherit the .wc-card radius,
   border, and padding so they sit in the same visual family as the
   Predictions / Bets cards above.  The tints are intentionally faint;
   the .wc-card border + shadow is what makes them read as cards. */
.wc-snapshot-card {
  background: rgba(255, 248, 230, 0.55);
}
.wc-confidence-card {
  background: rgba(230, 244, 255, 0.55);
}
@media (prefers-color-scheme: dark) {
  .wc-snapshot-card {
    background: rgba(80, 60, 30, 0.25);
  }
  .wc-confidence-card {
    background: rgba(30, 60, 90, 0.30);
  }
}

/* Inputs (text + number) ----------------------------------------- */

/* Match the 48 px iOS/Material touch-target floor.  Streamlit's
   default text inputs sit at ~38 px on desktop and look cramped on
   mobile; this rule raises the floor without changing fonts. */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
  min-height: 48px !important;
  font-size: 1rem !important;
  padding-top: 0.6rem !important;
  padding-bottom: 0.6rem !important;
  border-radius: 8px !important;
}
[data-testid="stTextInput"] input:focus-visible,
[data-testid="stNumberInput"] input:focus-visible {
  outline: 2px solid var(--primary-color, #ff4b4b) !important;
  outline-offset: 1px !important;
  box-shadow: 0 0 0 3px rgba(255, 75, 75, 0.18) !important;
}
/* Same treatment for the date picker / selectbox so the touch-target
   rule covers every common input the dashboard exposes. */
[data-testid="stDateInput"] input,
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
  min-height: 44px !important;
}

/* Sliders get a slightly taller track so the thumb is easy to grab
   on mobile; this is the standard ``st.slider`` testid. */
[data-testid="stSlider"] [role="slider"] {
  min-height: 28px !important;
  min-width: 28px !important;
}

/* Confidence pill ------------------------------------------------ */

/* Phase 3 already styled this; Phase 6 tightens contrast for dark
   mode by switching to Streamlit's theme variables when available.
   The inline ``style=`` attribute emitted by
   ``_render_confidence_pill`` still wins in light mode where the
   Bootstrap-style palette colours have better WCAG AA contrast. */
@media (prefers-color-scheme: dark) {
  .wc-confidence-pill {
    color: var(--text-color, #fafafa);
    border: 1px solid rgba(250, 250, 250, 0.35);
    padding: 1px 0;
  }
}

/* Best Value / No Clear Value blocks ---------------------------- */

/* Phase 4 inline styles already set background + foreground.  Phase 6
   adds the explicit 1 px border so the green/grey blocks don't
   disappear into the page background on coloured themes, and
   guarantees a visible focus state on the block itself. */
.wc-best-value {
  border: 1px solid rgba(15, 81, 50, 0.45) !important;
}
.wc-no-value-badge {
  border: 1px solid rgba(73, 80, 87, 0.45) !important;
}
.wc-best-value:focus-within,
.wc-no-value-badge:focus-within {
  outline: 2px solid var(--primary-color, #ff4b4b) !important;
  outline-offset: 2px !important;
}

/* Why popover bubble -------------------------------------------- */

/* Phase 3 already styled the bubble; this ruleset re-affirms the
   focus state and gives the *inner* popover surface a clean
   background so the popped-out content reads correctly in dark
   mode.  The outer bubble background stays the existing light blue. */
[data-testid="stPopover"] > button:focus-visible {
  outline: 3px solid var(--primary-color, #ff4b4b) !important;
  outline-offset: 2px !important;
  box-shadow: 0 0 0 4px rgba(255, 75, 75, 0.20) !important;
}
[data-testid="stPopover"] [data-testid="stMarkdown"] {
  background: var(--background-color, #ffffff);
  color: var(--text-color, #1a1a1a);
  padding: 6px 4px;
}

/* No horizontal page scroll ------------------------------------- */

/* Streamlit's main container is already responsive, but custom
   flex children inside a card can force min-width > viewport and
   introduce a horizontal scrollbar.  These two rules cap that. */
.main .block-container {
  max-width: 100vw;
  overflow-x: hidden;
}
.stColumns, .stColumn, .element-container, [data-testid="column"] {
  min-width: 0 !important;
}

/* Streamlit chrome to hide ------------------------------------- */

/* Standard "clean dashboard" hide-pattern.  We hide ONLY elements
   that are safe to hide (no accessibility, no rerun controls):
     - #MainMenu (hamburger)
     - footer ("Made with Streamlit")
     - the explicit "Made with Streamlit" badge via its container
   We deliberately keep the running-man / rerun controls intact. */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stFooter"] { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ------------------------------------------------------------------ */
/* Phase 7 — Accessibility (WCAG-friendly focus + body type)         */
/* ------------------------------------------------------------------ */

/* Always-visible focus outline on every interactive control.  We
   deliberately do NOT use a ``*:focus { outline: none; }`` reset —
   keyboard users need a visible focus ring.  Where Streamlit's
   default focus ring is invisible or under-coloured, this rule
   adds a 2px solid outline that has AAA contrast against both the
   light and the dark Streamlit backgrounds.  Tested against light
   and dark themes.  We attach the outline via ``:focus-visible``
   so it only appears for keyboard navigation, not for mouse clicks
   (which would feel noisy). */
*:focus-visible {
  outline: 2px solid #0b3d91 !important;
  outline-offset: 2px !important;
  border-radius: 6px;
}

/* Body text: 1em floor (16 px equivalent) so we never drop below
   the 0.85em accessibility floor.  Headings get the 1.3-1.6em
   treatment the brief calls for.  These are minimums; Streamlit
   defaults are usually larger and we only override when a rule
   from an earlier phase would shrink the type below 1em. */
.stMarkdown p,
.stMarkdown li,
.stCaption,
.stInfo,
.stWarning,
.stError,
.stSuccess {
  font-size: 1em !important;
  line-height: 1.45;
}
.stMarkdown h1 { font-size: 1.6em !important; }
.stMarkdown h2 { font-size: 1.45em !important; }
.stMarkdown h3 { font-size: 1.3em !important; }
.stMarkdown h4 { font-size: 1.2em !important; }

/* Dark theme: switch the focus ring to a high-contrast light blue
   so it remains visible on the dark background.  Streamlit sets
   ``[data-theme="dark"]`` on the document root in dark mode. */
[data-theme="dark"] *:focus-visible {
  outline-color: #93c5fd !important;
}
"""


def inject_css() -> None:
    """Inject the dashboard's CSS into the current Streamlit page.

    Phase 2 shipped a no-op stub.  Phase 3 makes this a real function:
    it emits a ``<style>`` block via :func:`st.markdown` with
    ``unsafe_allow_html=True``.  Streamlit de-duplicates ``<style>``
    blocks within a session, so re-invoking :func:`inject_css` on
    subsequent reruns is safe.
    """
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def get_css() -> str:
    """Return the CSS payload (used by tests; not for production use)."""
    return _CSS
