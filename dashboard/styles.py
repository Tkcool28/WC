"""Dashboard CSS / visual layer.

Phase 3 introduces the first real stylesheet. The rules here target the
``st.popover`` button (rebranded as a colored mobile CTA bubble for the
"❓ Why this pick?" control) and the ``wc-confidence-pill`` div emitted
by :mod:`dashboard.prediction_card`. Phase 6 will refine the design
system; the rules below are intentionally minimal and don't override
Streamlit's default typography.

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
