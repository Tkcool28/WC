"""Renders one prediction card in the casual Predictions view.

A "card" is one game's prediction: kickoff time, group/stage metadata,
the model's most-likely outcome, its probability, a confidence pill, and
a "Why?" popover with a short plain-language reason.

This module is the single source of truth for the *card* visual; the
*flow* (date picker → primary button → list of cards) lives in
:func:`dashboard.app._render_predictions_view`. Phase 6 will refine
the visuals but the card API is expected to stay stable.

Design constraints (Phase 3 brief):

* NO odds fields (``book_odds``, American-format numbers, +EV, etc.).
* NO ``min_edge`` slider, NO sportsbook terminology.
* NO raw ISO timestamps or raw ``GROUP_X`` strings.
* The "Why?" control is a :func:`streamlit.popover` styled as a
  colored mobile CTA bubble.  Plain bordered ``st.expander`` for
  "Why" is explicitly forbidden.
* Confidence is shown as a colored pill (green / blue / yellow / red)
  whose text maps to the existing confidence tier (``A``/``B``/``C``/``D``)
  plus the model-agreement status (only_pi / agree / disagree / …).
"""
from __future__ import annotations

import math
from typing import Any

import streamlit as st

from dashboard.text_format import (
    format_group_label,
    format_kickoff,
    format_matchday_label,
    format_team_matchup,
)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _format_probability(p: float | None) -> str:
    """Format a 0..1 probability as ``"42.3%"``, or ``"—"`` if unknown."""
    if p is None:
        return "—"
    try:
        v = float(p)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(v):
        return "—"
    return f"{v * 100:.1f}%"


def _extract_most_likely(prediction: dict) -> str:
    """Return ``"home"`` / ``"draw"`` / ``"away"`` for the highest-prob market.

    Prefers ``primary_probs`` (the official blended prediction), then
    ``blend_probs``, then ``pi_probs`` as fallback.  Returns ``""``
    if no probability dict is present.
    """
    probs = (
        prediction.get("primary_probs")
        or prediction.get("blend_probs")
        or prediction.get("pi_probs")
        or {}
    )
    if not probs:
        return ""
    return max(probs.items(), key=lambda kv: kv[1])[0]


def _extract_top_prob(prediction: dict) -> float | None:
    """Return the top-market probability (``0..1``) or ``None``."""
    probs = (
        prediction.get("primary_probs")
        or prediction.get("blend_probs")
        or prediction.get("pi_probs")
        or {}
    )
    if not probs:
        return None
    try:
        return max(float(v) for v in probs.values())
    except (TypeError, ValueError):
        return None


def _outcome_headline_text(mlr: str, prediction: dict) -> str:
    """Convert the top market into a human-readable headline.

    Examples
    --------
    >>> _outcome_headline_text("home", {"home_team": "England", "away_team": "Croatia"})
    'England to Win'
    >>> _outcome_headline_text("draw", {})
    'Match to End in a Draw'
    >>> _outcome_headline_text("away", {"home_team": "Mexico", "away_team": "USA"})
    'USA to Win'
    """
    if not mlr:
        return "TBD"
    s = mlr.lower()
    if s == "draw":
        return "Match to End in a Draw"
    if s == "home":
        name = prediction.get("home_team") or "Home"
        return f"{name} to Win"
    if s == "away":
        name = prediction.get("away_team") or "Away"
        return f"{name} to Win"
    # Unknown key — render the key verbatim so the user can see it but
    # still get a "to Win" suffix (a draw is special-cased above).
    return f"{s.title()} to Win"


def _confidence_tier(prediction: dict) -> str:
    """Return the existing confidence tier letter ``"A"``/``"B"``/``"C"``/``"D"``.

    Falls back to ``"C"`` (the "limited data" tier) when the assessment
    dict is missing or malformed.
    """
    assessment = prediction.get("confidence") or {}
    tier = assessment.get("tier")
    if tier in ("A", "B", "C", "D"):
        return tier
    return "C"


def _agreement_label(prediction: dict) -> str:
    """Return the model-agreement status (only_pi / agree / disagree / ...).

    Re-uses :func:`dashboard.ux_presenters.agreement_status` so the card
    matches the per-game Analysis view exactly.  Falls back to ``""`` if
    the helper is not importable (defensive — should never happen in
    production).
    """
    try:
        from dashboard.ux_presenters import agreement_status
        return agreement_status(prediction) or ""
    except Exception:
        return ""


def _confidence_label(prediction: dict) -> tuple[str, str]:
    """Map a prediction to (text_label, css_color_key).

    The mapping is a deliberately small, human-friendly set:

    * ``"High Confidence"``       — tier A and both models agree.
    * ``"Moderate Confidence"``   — tier A/B with fragile agreement, or
      tier B with agree.  (Tone matches the existing dashboard's
      "High / Medium / Low" copy but with slightly less jargon.)
    * ``"Limited Data"``          — tier C.
    * ``"Low Confidence"``        — tier D, or the two models disagree.

    ``css_color_key`` is one of ``"high"``, ``"moderate"``,
    ``"limited"``, ``"low"`` and is consumed by
    :func:`_render_confidence_pill` to pick foreground/background colors.
    """
    tier = _confidence_tier(prediction)
    agreement = _agreement_label(prediction)

    if tier == "A" and agreement == "agree":
        return "High Confidence", "high"
    if tier == "A" and agreement in ("fragile", "only_pi", "only_elo", ""):
        return "Moderate Confidence", "moderate"
    if tier == "B" and agreement == "disagree":
        return "Low Confidence", "low"
    if tier == "B":
        return "Moderate Confidence", "moderate"
    if tier == "C":
        return "Limited Data", "limited"
    # tier D or anything unrecognised
    return "Low Confidence", "low"


# --------------------------------------------------------------------------- #
# Confidence pill
# --------------------------------------------------------------------------- #
_CONFIDENCE_PALETTE: dict[str, tuple[str, str]] = {
    # css_color_key -> (foreground_hex, background_hex)
    "high":     ("#0f5132", "#d1e7dd"),
    "moderate": ("#055160", "#cff4fc"),
    "limited":  ("#664d03", "#fff3cd"),
    "low":      ("#842029", "#f8d7da"),
}


def _render_confidence_pill(label: str, color_key: str) -> None:
    """Render a colored confidence pill using inline HTML.

    The pill is wrapped in a ``<div class="wc-confidence-pill">`` so
    :mod:`dashboard.styles` can target it.  Inline ``style=`` attributes
    set the actual color so the pill is correct even if a future phase
    swaps the global stylesheet.
    """
    fg, bg = _CONFIDENCE_PALETTE.get(color_key, _CONFIDENCE_PALETTE["limited"])
    label_html = (
        label
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    html = (
        f"<div class='wc-confidence-pill' "
        f"style='display:inline-block;padding:6px 14px;border-radius:999px;"
        f"background:{bg};color:{fg};font-weight:600;font-size:0.95em;'>"
        f"{label_html}</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Why-text builder
# --------------------------------------------------------------------------- #
def _build_why_text(prediction: dict, identity_warnings: list[str] | None = None) -> str:
    """Return plain-language "Why this pick?" text.

    Delegates to :func:`dashboard.ux_presenters.prediction_why_text` so
    the rule ordering (limited data → disagree → margin → agree → ...)
    stays in one place.  Falls back to a generic line if the helper
    raises (defensive — should not happen in practice).
    """
    try:
        from dashboard.ux_presenters import prediction_why_text
        assessment = prediction.get("confidence") or {}
        raw_warnings = list(assessment.get("warnings") or [])
        return prediction_why_text(
            prediction,
            warnings=raw_warnings,
            identity_warnings=list(identity_warnings or []),
        )
    except Exception:
        return (
            "The model rates this as the most likely outcome based on team "
            "history and current form."
        )


def _consolidated_warnings(prediction: dict) -> list[str]:
    """Return at most ONE concise user-facing warning for this card.

    Delegates to :func:`dashboard.casual_warnings.consolidate_casual_warnings`
    so the rule ordering (identity_unresolved > history_missing >
    limited_data > calibration_caution > other) lives in one place.  A
    broken / missing helper falls back to an empty list — never to the
    raw translated stack.
    """
    try:
        from dashboard.casual_warnings import consolidate_casual_warnings
        return consolidate_casual_warnings(prediction)
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# Public renderer
# --------------------------------------------------------------------------- #
def render_prediction_card(
    prediction: dict,
    match_meta: dict | None = None,
    *,
    show_warnings: bool = True,
) -> None:
    """Render one prediction card.

    Parameters
    ----------
    prediction
        Output of :func:`soccer_ev_model.ev_workflow.predict_match`.
        Must contain at least ``home_team`` / ``away_team`` / ``date`` /
        ``pi_probs`` (or ``blend_probs``) / ``confidence``.  May also
        carry ``identity_warnings`` which will be passed to the
        ``prediction_why_text`` helper so the Why text can refer to
        identity-resolution issues in plain language.
    match_meta
        Optional match metadata (group, stage, matchday, kickoff_iso,
        …).  Used purely for the human-readable kickoff / group /
        stage caption line; never mutates the prediction dict.
    show_warnings
        If ``True`` (default), translated warnings are rendered as
        Streamlit ``st.info`` bullets below the card.  Set to ``False``
        for callers that prefer to render warnings elsewhere (e.g. the
        custom-matchup expander, which surfaces the identity warning
        alongside the input form).
    """
    meta = match_meta or {}
    home = prediction.get("home_team") or meta.get("home_team_name") or "TBD"
    away = prediction.get("away_team") or meta.get("away_team_name") or "TBD"

    # ---- headline + caption ---- #
    st.markdown(f"### {format_team_matchup(home, away)}")

    date_iso = (
        prediction.get("date")
        or meta.get("kickoff_iso")
        or meta.get("date")
        or ""
    )
    kickoff_str = format_kickoff(date_iso)
    group_str = format_group_label(meta.get("group"))
    stage_str = format_matchday_label(
        meta.get("stage"), meta.get("matchday")
    )
    st.caption(f"{kickoff_str} · {group_str} · {stage_str}")

    # ---- most likely result ---- #
    mlr = _extract_most_likely(prediction)
    p_top = _extract_top_prob(prediction)
    headline = _outcome_headline_text(mlr, prediction)

    st.markdown("**Most Likely Result**")
    # The headline is a one-line emphasis; use ``st.markdown`` with a
    # span rather than ``st.header`` so it sits inline with the body
    # on small screens.
    headline_html = (
        headline
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    st.markdown(
        f"<div style='font-size:1.5em; font-weight:600; line-height:1.2;'>"
        f"{headline_html}</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"{_format_probability(p_top)} model probability")

    # ---- confidence pill ---- #
    label, color_key = _confidence_label(prediction)
    _render_confidence_pill(label, color_key)

    # ---- Why popover (mobile CTA bubble) ---- #
    identity_warnings = list(prediction.get("identity_warnings") or [])
    why_text = _build_why_text(prediction, identity_warnings=identity_warnings)
    with st.popover("❓ Why this pick?", use_container_width=False):
        st.markdown(why_text)

    # ---- consolidated warnings (at most one st.info block) ---- #
    if show_warnings:
        consolidated = _consolidated_warnings(prediction)
        for w in consolidated:
            st.info(f"ℹ️ {w}")


__all__ = ["render_prediction_card", "_confidence_label", "_build_why_text"]
