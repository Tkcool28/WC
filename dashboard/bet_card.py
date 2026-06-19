"""Renders one Bets card in the 💰 Bets view.

A "card" is one matchup's full betting-value surface:

  * Matchup (team names + kickoff + group/stage caption)
  * Three odds inputs (home / draw / away) with labels + helper text
  * ONE large primary "💰 Check Betting Value" button
  * Result block, rendered after the user submits valid odds:
        - **Most Likely Result** (large) — the model's pick, NOT the value.
        - **Best Value Play** (large) — OR "— No Clear Value —" badge.
        - Value confidence pill
        - Edge (decimal)
        - "Why is this value?" popover bubble
        - Optional note when best value differs from most-likely result

The visual emphasis on Most Likely Result vs Best Value Play is
intentionally different: the model pick is a neutral, large label while
the best-value play is a green "💎" block. The "No Clear Value" badge
is a grey, non-icon block so the two never look like the same thing.

Local validation: empty / non-numeric / out-of-range odds produce a
**local** ``st.error`` inside the card and never disrupt the other
cards rendered on the same page. The error stays in the card.

Module boundary
---------------
This module is *pure presentation* — it never recomputes model
probabilities and never modifies the prediction dict. It only calls
:func:`soccer_ev_model.ev_workflow.evaluate_market` when the user taps
the submit button, and only with the (already-validated) odds from the
text inputs. Tests can monkey-patch the lazy import to drive specific
market outcomes.
"""
from __future__ import annotations

from typing import Any, Optional

import streamlit as st

from dashboard.prediction_card import (
    _confidence_label,
    _extract_most_likely,
    _format_probability,
    _outcome_headline_text,
)
from dashboard.text_format import (
    format_group_label,
    format_kickoff,
    format_matchday_label,
    format_team_matchup,
)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _format_odds(o: Any) -> str:
    """Format an American-odds number as ``"+350"`` / ``"-230"`` / ``""``.

    Returns an empty string for ``None`` / empty so the result block
    stays clean. Non-numeric inputs fall through to ``str(o)`` rather
    than raising — the calling site has already validated the inputs
    by the time this is called.
    """
    if o is None or o == "":
        return ""
    try:
        v = int(float(o))
    except (TypeError, ValueError):
        return str(o)
    if v > 0:
        return f"+{v}"
    return str(v)


def _render_no_clear_value() -> None:
    """Render the '— No Clear Value —' badge.

    Visually distinct from a real best-value play:

    * Grey / neutral palette (no green tint, no 💎 icon).
    * Lower contrast than the best-value block.
    * Wrapped in ``<div class="wc-no-value-badge">`` so
      :mod:`dashboard.styles` can target it.
    """
    html = (
        "<div class='wc-no-value-badge' "
        "style='display:inline-block;padding:10px 20px;border-radius:12px;"
        "background:#f1f3f5;color:#495057;font-weight:700;font-size:1.1em;'>"
        "— No Clear Value —</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_best_value(market_label: str, odds: Any) -> None:
    """Render the '💎 Best Value: ...' block.

    Visually distinct from the most-likely-result headline:

    * Green palette (vs neutral grey for no-value / dark text for MLR).
    * Wrapped in ``<div class="wc-best-value">`` so the stylesheet
      can target it.
    * 💎 icon to draw the eye.
    """
    odds_text = _format_odds(odds)
    # Defensive HTML escape so a team name like "AT&T" can't break layout.
    safe_label = (
        market_label
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    odds_suffix = f" at <strong>{odds_text}</strong>" if odds_text else ""
    html = (
        "<div class='wc-best-value' "
        "style='display:block;padding:14px 18px;border-radius:14px;"
        "background:#e6f4ea;color:#0f5132;font-weight:700;font-size:1.2em;"
        "border:1px solid #a3cfbb;margin-top:6px;'>"
        f"💎 Best Value: {safe_label}{odds_suffix}"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def _min_edge_global() -> float:
    """Return the user's min_edge setting (default 0.03) from session state.

    Falls back to 0.03 if the stored value is missing or non-numeric.
    """
    try:
        from dashboard.session_state import KEYS, get

        raw = get(KEYS.BETS_MIN_EDGE, 0.03)
    except Exception:
        return 0.03
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 0.03
    # Sanity clamp — sliders return 0..15 (pct), so 0..0.15 here.
    if v < 0:
        return 0.03
    if v > 1.0:
        return 1.0
    return v


def _validate_odds_text(home: str, draw: str, away: str) -> tuple[Optional[tuple[float, float, float]], Optional[str]]:
    """Validate three American-odds text inputs.

    Returns ``(values, None)`` on success, or ``(None, error_message)``
    on failure. The error message is short and user-facing — it is
    shown as a local ``st.error`` inside the card.

    Validation rules:

    * All three strings must be non-empty after stripping.
    * All three must parse as floats (American format: ``+`` is allowed).
    * All three must be in ``[-10000, -100] ∪ [+100, +10000]`` — sane
      American-odds range, rejects accidental decimal misentries and
      zero / ±100 edge cases.
    """
    h = (home or "").strip()
    d = (draw or "").strip()
    a = (away or "").strip()
    if not h or not d or not a:
        return None, "Enter all three odds (home, draw, away) before checking value."
    # Strip '+' prefix so "+350" parses to 350.0 cleanly.
    def _to_float(s: str) -> float:
        return float(s.strip().lstrip("+"))

    try:
        h_f = _to_float(h)
        d_f = _to_float(d)
        a_f = _to_float(a)
    except ValueError:
        return None, "Odds must be numbers in American format (e.g. -230, +350, +550)."

    for label, v in (("Home", h_f), ("Draw", d_f), ("Away", a_f)):
        if abs(v) < 100 or abs(v) > 10000:
            return None, f"{label} odds ({v:+.0f}) look out of range. Try e.g. -230 or +350."
    return (h_f, d_f, a_f), None


# --------------------------------------------------------------------------- #
# Public renderer
# --------------------------------------------------------------------------- #
def render_bet_card(
    match_meta: dict,
    prediction: dict,
    *,
    key_prefix: str,
) -> None:
    """Render one Bets card.

    The card owns its own odds inputs and submit button — keys are
    namespaced by ``key_prefix`` (typically the match id) so multiple
    cards on the same page don't collide.

    The prediction dict is required (it provides the model's most-likely
    result and the primary_probs that ``evaluate_market`` will
    consume). It is NEVER mutated by this function.

    Parameters
    ----------
    match_meta
        Dict with optional keys: ``home_team_name``, ``away_team_name``,
        ``kickoff_iso``, ``date``, ``group``, ``stage``, ``matchday``.
        Missing keys fall back to the prediction's values or "TBD".
    prediction
        Output of :func:`soccer_ev_model.ev_workflow.predict_match`.
        Must contain at least ``home_team`` / ``away_team`` /
        ``primary_probs`` (or ``blend_probs`` / ``pi_probs``) / ``confidence``.
    key_prefix
        Unique widget-key prefix. Conventionally the match id (cast to
        str). Used to namespace the three ``text_input`` controls and
        the submit button.
    """
    home = prediction.get("home_team") or match_meta.get("home_team_name", "TBD")
    away = prediction.get("away_team") or match_meta.get("away_team_name", "TBD")

    # ---- (1) Matchup header ---- #
    st.markdown(f"### {format_team_matchup(home, away)}")
    kickoff_str = format_kickoff(
        match_meta.get("kickoff_iso") or match_meta.get("date")
    )
    group_str = format_group_label(match_meta.get("group"))
    stage_str = format_matchday_label(
        match_meta.get("stage"), match_meta.get("matchday")
    )
    st.caption(f"{kickoff_str} · {group_str} · {stage_str}")

    # ---- (2) Most Likely Result (model's pick, NOT the value) ---- #
    mlr_key = _extract_most_likely(prediction)
    mlr_text = _outcome_headline_text(mlr_key, prediction)
    # primary_probs is the canonical official prediction (Phase 4+5).
    _probs = prediction.get("primary_probs") or prediction.get("blend_probs") or prediction.get("pi_probs") or {}
    p_top = _probs.get(mlr_key)
    # Surface a subtle note when the goal model was expected but could
    # not be loaded — the user is seeing Elo-only or pi-only predictions.
    if prediction.get("_goal_model_expected") and not prediction.get("_goal_model_used"):
        st.caption("⚠️ Goal model unavailable — using Elo-only blend.")
    st.markdown("**Most Likely Result**")
    headline_html = (
        mlr_text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    st.markdown(
        f"<div class='wc-mlr-headline' "
        f"style='font-size:1.3em; font-weight:600; line-height:1.2; "
        f"color:#1a1a1a;'>"
        f"{headline_html}</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"{_format_probability(p_top)} model probability")

    # ---- (3) Odds inputs (3 columns on desktop, stacked on mobile via CSS) ---- #
    st.markdown("**Enter sportsbook odds** (American format, e.g. -230)")
    c1, c2, c3 = st.columns(3)
    with c1:
        home_odds = st.text_input(
            "Home odds",
            value="",
            key=f"{key_prefix}_home_odds",
            placeholder="e.g. -230",
            help="American odds for the home team (negative = favorite, positive = underdog).",
        )
    with c2:
        draw_odds = st.text_input(
            "Draw odds",
            value="",
            key=f"{key_prefix}_draw_odds",
            placeholder="e.g. +350",
            help="American odds for the draw.",
        )
    with c3:
        away_odds = st.text_input(
            "Away odds",
            value="",
            key=f"{key_prefix}_away_odds",
            placeholder="e.g. +550",
            help="American odds for the away team.",
        )
    st.caption(
        "_Examples are placeholders only — enter the current sportsbook price for this game._"
    )

    # ---- (4) Submit button (one large primary per card) ---- #
    run = st.button(
        "💰 Check Betting Value",
        key=f"{key_prefix}_check_btn",
        type="primary",
        use_container_width=True,
    )

    # ---- (5) Result panel — local to this card ---- #
    if not run:
        return

    values, err = _validate_odds_text(home_odds, draw_odds, away_odds)
    if err is not None:
        st.error(err)
        return
    h_f, d_f, a_f = values  # type: ignore[misc]

    # Call evaluate_market. Import lazily so tests can monkey-patch
    # ``soccer_ev_model.ev_workflow.evaluate_market`` and observe the
    # call (the patch target is the real module, not this card).
    try:
        from soccer_ev_model.ev_workflow import evaluate_market

        market = evaluate_market(
            prediction,
            book_home_odds=h_f,
            book_draw_odds=d_f,
            book_away_odds=a_f,
            min_edge=_min_edge_global(),
        )
    except ValueError as exc:
        # ValueError typically means the odds math rejected the input
        # (e.g. invalid price).  Show the validated message verbatim —
        # the upstream helper produces user-facing copy.
        st.error(str(exc) or "We couldn't evaluate the market for these odds.")
        return
    except Exception:
        # Calm, plain-language error.  No raw exception text / stack
        # trace leaks to the user.  The full traceback stays in the
        # server logs.
        st.error(
            "We couldn't evaluate the market for these odds. "
            "Double-check the prices and try again."
        )
        return

    # Stitch market fields onto the prediction so the existing
    # ux_presenters (which expect a single combined dict) work as-is.
    # This does NOT mutate the original prediction — it builds a
    # shallow-copy locals dict that is consumed by the presenters and
    # then discarded.
    combined = {**prediction, **market}

    # Choose the best-value play via the existing ux_presenter.
    try:
        from dashboard.ux_presenters import value_play
        # The presenter filters by min_edge via plus_ev_flags. We pass
        # the global min_edge so the user's slider choice is honoured.
        best = value_play(combined, min_edge=_min_edge_global())
    except Exception:
        best = {"status": "no_clear_value", "reason": "no value"}

    # ---- Render the result block ---- #
    if best.get("status") == "play":
        # _pretty_market is a 3-line helper; use the same source so the
        # label matches what the manual flow shows.
        from dashboard.ux_presenters import format_odds as _ux_format_odds
        market_label = {
            "home": prediction.get("home_team", "Home"),
            "draw": "Match to End in a Draw",
            "away": prediction.get("away_team", "Away"),
        }.get(best.get("market", ""), "TBD")
        odds_val = best.get("odds")
        _render_best_value(market_label, odds_val)

        # Value confidence — reuse the existing tier-based mapping so
        # the wording stays consistent with the rest of the app.
        try:
            from dashboard.ux_presenters import value_confidence_label
            vcl = value_confidence_label(best, combined)
        except Exception:
            vcl = "Medium"
        st.caption(f"Value confidence: {vcl}")

        # Edge (decimal percentage)
        edge = float(best.get("edge", 0.0))
        st.markdown(
            f"<div class='wc-edge' "
            f"style='font-size:1em; color:#495057;margin-top:4px;'>"
            f"Edge: <strong>{edge * 100:+.1f}%</strong>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Note when best value differs from the model's most-likely result.
        if best.get("market") != mlr_key:
            st.info(
                f"ℹ️ The model expects {mlr_text}, but the best value is "
                f"on {market_label}. The sportsbook may be mispricing the underdog."
            )
    else:
        _render_no_clear_value()
        st.caption("Value confidence: Low (no outcome cleared the edge threshold)")

    # ---- "Why is this value?" popover bubble (always present) ---- #
    try:
        from dashboard.ux_presenters import value_why_text
        why_text = value_why_text(best, combined)
    except Exception:
        why_text = "The sportsbook price implies a different probability than the model."
    with st.popover("❓ Why is this value?", use_container_width=False):
        st.markdown(why_text)


__all__ = ["render_bet_card"]
