"""Renders the Analysis view (technical deep-dive per game).

Phase 5 of the dashboard rearchitecture replaces the Phase 2 stub with a
real, mobile-first, technical Analysis experience:

* Pick a date (driven by :data:`KEYS.SELECTED_DATE`).
* Pick a game (compact matchup picker).
* See 11 sections of technical detail — Prediction Details opens by
  default, the other 10 are collapsed.

The view works with **model-only** predictions (no odds required).  When
the user has run the same game through the Bets view, the
``KEYS.MARKET_BY_MATCH`` cache contains an ``evaluate_market(...)`` result
that the **Market Comparison** section will surface; otherwise the
section shows a calm message — never a hard error — so the rest of the
view remains useful for technical users who only care about the model
output.

Sections (in order):

1.  **Prediction Details** *(default OPEN)* — model's pick, the full
    1X2 probability table, confidence tier, calibrated_p, and the
    confidence warnings list.
2.  **Model Breakdown** — pi-only / elo-only / blend probabilities and
    the model-agreement status (the ``agreement_status`` helper).
3.  **π Pi-Rating** — raw pi-rating diagnostics stored on the prediction.
4.  **📊 Elo Rating** — raw Elo diagnostics stored on the prediction.
5.  **⚖️ Blend** — the blend formula, the per-model weights used, and
    whether the blend actually ran (``blend_was_used``).
6.  **💹 Market Comparison** — *conditional*.  Only renders real data
    when ``market_by_match`` is populated; otherwise shows
    "Enter sportsbook odds in Bets to unlock market comparison."
7.  **🥅 Poisson View** — expected-goals estimate, independent
    Poisson 1X2 from the score matrix, and the Poisson vs Pi
    agreement label.
8.  **👥 Squad Context** — manual squad context from
    :func:`dashboard.context_loader.get_match_context` (Phase 4).
9.  **🏆 Group Context** — matchday label, group context warnings.
10. **📐 Calibration and Data Quality** — confidence assessment + raw
    identity warnings (this is the section the test contract requires
    to surface the tier letter A/B/C/D).
11. **🔧 Raw Diagnostics** — canonical team IDs, the full prediction
    dict, the source-match metadata, and any other raw fields the
    technical user wants to see.

All sections default to **collapsed** except Prediction Details
(``expanded=True``).  The selection is persisted across reruns and
tab changes via :data:`dashboard.session_state.KEYS.ANALYSIS_GAME`.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from dashboard.text_format import (
    format_group_label,
    format_kickoff,
    format_matchday_label,
    format_team_matchup,
)


# --------------------------------------------------------------------------- #
# Public renderer
# --------------------------------------------------------------------------- #
def render_analysis_view(
    matches_for_date: list[dict],
    predictions_by_match: dict[str, dict] | dict[int, dict],
    market_by_match: dict[str, dict] | dict[int, dict] | None = None,
    name_to_id: dict | None = None,
) -> None:
    """Render the full Analysis experience for a single selected date.

    Parameters
    ----------
    matches_for_date
        List of match dicts (raw, ``UnplayedMatch.to_dict()`` shape) for
        the selected date.  The view builds the game picker from this
        list.
    predictions_by_match
        Dict mapping match id (str or int) to a ``predict_match(...)``
        result.  Predictions are required for the per-game detail.
    market_by_match
        Optional dict mapping match id to an ``evaluate_market(...)``
        result.  When populated, the Market Comparison section surfaces
        the real numbers.  When missing/empty, the section shows a
        calm message and the rest of the view is unaffected.
    name_to_id
        Optional team-name -> id mapping; reserved for future
        squad-context lookups.  Not directly used by the renderer.

    Returns
    -------
    None
        The function renders directly to the active Streamlit session.
    """
    # Defensive: tolerate None / falsy args without crashing.
    market_by_match = market_by_match or {}

    # ---- Empty-state copy when no games on the chosen date ---- #
    if not matches_for_date:
        st.info(
            "No matches on this date. Pick another date above, or use "
            "the **Custom matchup** expander in Predictions to predict "
            "any game."
        )
        return

    # ---- Build the (match_id -> label) options for the selectbox ---- #
    options: list[tuple[str, str]] = []
    for m in matches_for_date:
        mid = _match_id_str(m)
        home = (
            m.get("home_team_name")
            or m.get("home_team")
            or "TBD"
        )
        away = (
            m.get("away_team_name")
            or m.get("away_team")
            or "TBD"
        )
        options.append((mid, format_team_matchup(home, away)))
    if not options:
        st.info("No games available.")
        return

    # ---- Persist the user's pick across reruns/tab switches ---- #
    # Imported lazily so importing this module never needs Streamlit's
    # ScriptRunContext (the file is also importable for unit tests).
    from dashboard.session_state import KEYS, get, set_

    valid_ids = [o[0] for o in options]
    current = get(KEYS.ANALYSIS_GAME)
    if current not in valid_ids:
        current = valid_ids[0]
        set_(KEYS.ANALYSIS_GAME, current)

    # ---- Render the selectbox ---- #
    try:
        idx = valid_ids.index(current)
    except ValueError:
        idx = 0
    selected_label = st.selectbox(
        "Select a game",
        options=[o[1] for o in options],
        index=idx,
        key=KEYS.ANALYSIS_GAME + "_label",
    )
    # The user could have changed the selectbox this rerun.  We look up
    # the underlying match_id by label so subsequent sections read the
    # right prediction / market.
    sel_id = next(
        (o[0] for o in options if o[1] == selected_label),
        valid_ids[idx],
    )
    if sel_id != current:
        set_(KEYS.ANALYSIS_GAME, sel_id)

    # ---- Pull prediction + market for the selected game ---- #
    prediction = _lookup(predictions_by_match, sel_id)
    match_meta = next(
        (m for m in matches_for_date if _match_id_str(m) == sel_id),
        {},
    )
    market = _lookup(market_by_match, sel_id) if market_by_match else None

    # ---- Header ---- #
    home_name = (prediction or {}).get("home_team") or match_meta.get("home_team_name") or "TBD"
    away_name = (prediction or {}).get("away_team") or match_meta.get("away_team_name") or "TBD"
    st.markdown(f"## {format_team_matchup(home_name, away_name)}")
    kickoff_str = format_kickoff(
        match_meta.get("kickoff_iso") or match_meta.get("match_date_iso") or match_meta.get("date")
    )
    group_str = format_group_label(match_meta.get("group"))
    stage_str = format_matchday_label(
        match_meta.get("stage"), match_meta.get("matchday")
    )
    st.caption(f"{kickoff_str} · {group_str} · {stage_str}")

    if not prediction:
        st.warning(
            "No prediction available for this game yet. Go to "
            "**Predictions** and tap **Show Predictions** to compute it."
        )
        return

    # ---- 1. Prediction Details (default OPEN) ---- #
    with st.expander("🎯 Prediction Details", expanded=True):
        _render_prediction_details(prediction)

    # ---- 2. Model Breakdown ---- #
    with st.expander("🧮 Model Breakdown", expanded=False):
        _render_model_breakdown(prediction)

    # ---- 3. Pi-Rating ---- #
    with st.expander("π Pi-Rating", expanded=False):
        _render_pi_section(prediction)

    # ---- 4. Elo Rating ---- #
    with st.expander("📊 Elo Rating", expanded=False):
        _render_elo_section(prediction)

    # ---- 5. Blend ---- #
    with st.expander("⚖️ Blend", expanded=False):
        _render_blend_section(prediction)

    # ---- 6. Market Comparison (only if market data exists) ---- #
    with st.expander("💹 Market Comparison", expanded=False):
        if market:
            _render_market_comparison(prediction, market)
        else:
            st.info(
                "Enter sportsbook odds in **Bets** to unlock market "
                "comparison."
            )

    # ---- 7. Poisson View ---- #
    with st.expander("🥅 Poisson View", expanded=False):
        _render_poisson_section(prediction)

    # ---- 8. Squad Context ---- #
    with st.expander("👥 Squad Context", expanded=False):
        _render_squad_context(
            match_meta, prediction=prediction, name_to_id=name_to_id,
        )

    # ---- 9. Group Context ---- #
    with st.expander("🏆 Group Context", expanded=False):
        _render_group_context(prediction, match_meta)

    # ---- 10. Calibration and Data Quality ---- #
    with st.expander("📐 Calibration and Data Quality", expanded=False):
        _render_calibration(prediction)

    # ---- 11. Raw Diagnostics ---- #
    with st.expander("🔧 Raw Diagnostics", expanded=False):
        _render_raw_diagnostics(prediction, match_meta)


# --------------------------------------------------------------------------- #
# Section renderers (one per expander above)
# --------------------------------------------------------------------------- #
def _render_prediction_details(prediction: dict) -> None:
    """Section 1: headline outcome, all three probs, confidence breakdown.

    Pure presentation: imports the same pure helpers
    :func:`dashboard.prediction_card._outcome_headline_text`,
    :func:`dashboard.prediction_card._confidence_label`,
    :func:`dashboard.prediction_card._extract_most_likely`, and
    :func:`dashboard.prediction_card._format_probability`.  No model
    math, no I/O.
    """
    from dashboard.prediction_card import (
        _extract_most_likely,
        _format_probability,
        _outcome_headline_text,
    )

    mlr = _extract_most_likely(prediction)
    headline = _outcome_headline_text(mlr, prediction)
    probs = (
        prediction.get("primary_probs")
        or prediction.get("blend_probs")
        or prediction.get("pi_probs")
        or {}
    )
    p_top = probs.get(mlr) if probs else None

    st.markdown(f"### {headline}")
    st.caption(f"Top-market probability: {_format_probability(p_top)}")

    # ---- All three probabilities ---- #
    if probs:
        st.markdown("**All outcomes:**")
        for k in ("home", "draw", "away"):
            v = probs.get(k)
            if v is not None:
                st.markdown(f"- {k}: {_format_probability(v)}")

    # ---- Confidence breakdown (tier + warnings) ---- #
    confidence = prediction.get("confidence") or {}
    if confidence:
        st.markdown("**Confidence:**")
        for key in (
            "tier",
            "tier_description",
            "calibrated_p",
            "calib_label",
            "label",
        ):
            if key in confidence:
                st.markdown(f"- {key}: `{confidence[key]}`")
        warnings = list(confidence.get("warnings") or [])
        if warnings:
            st.markdown("**Warnings:**")
            for w in warnings:
                st.markdown(f"- {w}")


def _render_model_breakdown(prediction: dict) -> None:
    """Section 2: pi-only, elo-only, blend probabilities + agreement.

    Uses :func:`dashboard.ux_presenters.agreement_status` so the
    agreement label is consistent with the casual Prediction and
    Betting Value tabs.
    """
    pi_only = prediction.get("pi_only_probs") or {}
    elo_only = prediction.get("elo_only_probs") or {}
    blend = (
        prediction.get("primary_probs")
        or prediction.get("blend_probs")
        or prediction.get("pi_probs")
        or {}
    )

    if pi_only:
        st.markdown("**Pi-only:**")
        st.json(pi_only)
    if elo_only:
        st.markdown("**Elo-only:**")
        st.json(elo_only)
    if blend:
        st.markdown("**Blend:**")
        st.json(blend)

    try:
        from dashboard.ux_presenters import agreement_status
        st.markdown(
            f"**Agreement status:** `{agreement_status(prediction)}`"
        )
    except Exception:
        # agreement_status is pure, but be defensive so a future
        # refactor never crashes the Analysis view.
        pass


def _render_pi_section(prediction: dict) -> None:
    """Section 3: raw pi-rating components stored on the prediction."""
    pi_fields = {
        k: v
        for k, v in prediction.items()
        if k.startswith("pi_") and not k.endswith("_probs")
    }
    if pi_fields:
        st.markdown("**Pi-rating components:**")
        st.json(pi_fields)
    else:
        st.caption("No pi-rating diagnostics on this prediction.")


def _render_elo_section(prediction: dict) -> None:
    """Section 4: raw Elo components stored on the prediction."""
    elo_fields = {
        k: v
        for k, v in prediction.items()
        if "elo" in k.lower() and not k.endswith("_probs")
    }
    if elo_fields:
        st.markdown("**Elo components:**")
        st.json(elo_fields)
    else:
        st.caption("No Elo diagnostics on this prediction.")


def _render_blend_section(prediction: dict) -> None:
    """Section 5: the blend formula and the per-model weights used.

    Reads ``blend_w_pi`` / ``blend_w_elo`` if the prediction stores
    them.  Falls back to documented values when the model didn't
    materialise the per-game weights.
    """
    st.markdown("**Blend formula:**")
    st.markdown("`blend = w_pi * pi_probs + w_elo * elo_probs`")
    w_pi = prediction.get("blend_w_pi")
    w_elo = prediction.get("blend_w_elo")
    used = prediction.get("blend_was_used")
    st.markdown(
        f"- w_pi = `{w_pi}`, w_elo = `{w_elo}`, used = `{used}`"
    )


def _render_market_comparison(prediction: dict, market: dict) -> None:
    """Section 6: model vs no-vig book, edges, +EV flags.

    Reads the same fields the legacy per-game ``evaluate_market``
    result exposes — ``book_fair``, ``calibrated_pi``, ``edges``,
    ``plus_ev_flags``, ``best_value_play``.
    """
    st.json({
        "book_fair": market.get("book_fair"),
        "calibrated_pi": market.get("calibrated_pi"),
        "edges": market.get("edges"),
        "plus_ev_flags": market.get("plus_ev_flags"),
        "best_value_play": market.get("best_value_play"),
        "book_odds": market.get("book_odds", prediction.get("book_odds")),
    })


def _render_poisson_section(prediction: dict) -> None:
    """Section 7: independent Poisson xG + 1X2 + agreement label."""
    try:
        from soccer_ev_model.prediction_summary import (
            expected_goals_from_blend,
            poisson_agreement_label,
            poisson_outcome_probs,
        )
    except Exception as exc:
        st.warning(f"Poisson helpers unavailable: {exc}")
        return

    blend = (
        prediction.get("primary_probs")
        or prediction.get("blend_probs")
        or prediction.get("pi_probs")
        or {}
    )
    if not blend:
        st.caption("No blend probabilities available.")
        return

    try:
        xg = expected_goals_from_blend(blend)
    except Exception as exc:
        st.warning(f"Could not derive xG: {exc}")
        return

    home_xg = xg.get("home_xg", "—")
    away_xg = xg.get("away_xg", "—")
    home_name = prediction.get("home_team", "Home")
    away_name = prediction.get("away_team", "Away")
    st.markdown(
        f"**Expected goals:** {home_name} `{home_xg}`, "
        f"{away_name} `{away_xg}`"
    )
    st.markdown("**Poisson outcome probs:**")
    try:
        st.json(poisson_outcome_probs(xg["home_xg"], xg["away_xg"]))
    except Exception as exc:
        st.warning(f"Poisson score matrix failed: {exc}")

    pi = prediction.get("pi_only_probs") or {}
    if pi:
        try:
            st.markdown(
                f"**Poisson vs Pi agreement:** "
                f"`{poisson_agreement_label(blend, pi)}`"
            )
        except Exception:
            pass


def _looks_like_canonical_id(value: Any) -> bool:
    """Return True iff ``value`` looks like a canonical ID (string, not
    a numeric schedule ``fd_id``).

    Canonical IDs are short non-numeric strings like ``"ENG"``, ``"POR"``,
    ``"COD"``.  Numeric schedule IDs (e.g. ``770``, ``"770"``) and
    anything containing only digits are rejected so we never pass them
    to :func:`dashboard.context_loader.get_match_context`.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        # Plain ``int`` schedule fd_ids — reject.
        return False
    s = str(value).strip()
    if not s:
        return False
    # Pure-digit strings (or anything ``str(int)``-able) are schedule ids.
    if s.isdigit():
        return False
    return True


def _render_squad_context(
    match_meta: dict,
    *,
    prediction: dict | None = None,
    name_to_id: dict | None = None,
) -> None:
    """Section 8: manual squad context from Phase 4.

    Reads via :func:`dashboard.context_loader.get_match_context` so the
    Phase 4 manual CSV data is the single source of truth.  We never
    recompute squad strength; we only render what's in the manual
    snapshot.

    Canonical ID routing — priority order:

    1. ``prediction["canonical_home_id"]`` / ``canonical_away_id`` if
       they look like canonical IDs (non-numeric strings).
    2. ``match_meta["canonical_home_id"]`` / ``canonical_away_id`` as
       fallback when the prediction is absent.
    3. Otherwise show a calm unavailable caption.  Schedule ``fd_ids``
       (numeric schedule IDs from football-data.org) are NEVER passed
       to :func:`dashboard.context_loader.get_match_context`.
    """
    _ = name_to_id  # reserved for future Elo lookups
    pred = prediction or {}
    meta = match_meta or {}

    home_canonical = pred.get("canonical_home_id")
    if not _looks_like_canonical_id(home_canonical):
        home_canonical = meta.get("canonical_home_id")

    away_canonical = pred.get("canonical_away_id")
    if not _looks_like_canonical_id(away_canonical):
        away_canonical = meta.get("canonical_away_id")

    if not _looks_like_canonical_id(home_canonical) or not _looks_like_canonical_id(
        away_canonical
    ):
        st.caption(
            "Squad context not available — no canonical team ids on this "
            "match."
        )
        return
    try:
        from dashboard.context_loader import get_match_context
        ctx = get_match_context(
            str(home_canonical).strip(),
            str(away_canonical).strip(),
        )
    except Exception as exc:
        st.warning(f"Squad context unavailable: {exc}")
        return
    if not ctx:
        st.caption("No manual squad context for this game.")
        return
    st.json(ctx)


def _render_group_context(prediction: dict, match_meta: dict) -> None:
    """Section 9: matchday label + group context warnings.

    The full matchday label comes from
    :func:`soccer_ev_model.prediction_summary.matchday_label`; the
    warnings come from
    :func:`soccer_ev_model.prediction_summary.group_context_warnings`.
    Both are pure helpers.
    """
    _ = prediction  # this section only reads match_meta
    stage = match_meta.get("stage")
    matchday = match_meta.get("matchday")
    group = match_meta.get("group")

    if not stage:
        st.caption("No stage / group metadata for this match.")
        return

    try:
        from soccer_ev_model.prediction_summary import (
            group_context_warnings,
            matchday_label,
        )
    except Exception as exc:
        st.warning(f"Group-context helpers unavailable: {exc}")
        return

    try:
        label_info = matchday_label(stage, matchday)
    except Exception as exc:
        st.warning(f"matchday_label failed: {exc}")
        label_info = None

    if label_info:
        # ``matchday_label`` returns a dict; render it readably.
        st.markdown("**Matchday label:**")
        st.json(label_info)

    if group:
        try:
            warnings = group_context_warnings(stage, matchday, group)
        except Exception as exc:
            warnings = []
            st.warning(f"group_context_warnings failed: {exc}")
        if warnings:
            st.markdown("**Group warnings:**")
            for w in warnings:
                # Each warning is a dict with 'text' (string) and
                # 'severity' ('info' | 'warning').  Tag the high-priority
                # ones with the same warning emoji the per-game renderer
                # uses for consistency.
                if isinstance(w, dict):
                    text = w.get("text", str(w))
                    severity = w.get("severity")
                else:
                    text, severity = str(w), None
                prefix = "⚠️ " if severity == "warning" else ""
                st.markdown(f"- {prefix}{text}")


def _render_calibration(prediction: dict) -> None:
    """Section 10: confidence assessment + raw identity warnings.

    The test contract requires this section to surface the tier letter
    (A / B / C / D).  We render the full confidence dict via
    :func:`streamlit.json` so the tier is always present, then add a
    human-readable tier summary line.
    """
    confidence = prediction.get("confidence") or {}
    if not confidence:
        st.caption("No calibration data available.")
        return

    # Human-readable tier line (A / B / C / D) — this is the one the
    # tests check for.
    tier = confidence.get("tier", "")
    tier_desc = confidence.get("tier_description", "")
    if tier:
        st.markdown(f"**Calibration tier:** `{tier}` — {tier_desc}")

    st.json(confidence)

    # Raw identity warnings (NOT translated — this is the technical view).
    identity_warnings = list(prediction.get("identity_warnings") or [])
    if identity_warnings:
        st.markdown("**Raw identity warnings (technical):**")
        for iw in identity_warnings:
            st.markdown(f"- `{iw}`")


def _render_raw_diagnostics(prediction: dict, match_meta: dict) -> None:
    """Section 11: canonical team IDs, full prediction JSON, match meta.

    Canonical team IDs are the test contract: they MUST appear here so
    a technical user can verify the identity resolution worked.
    """
    canonical_home = prediction.get("canonical_home_id", "?")
    canonical_away = prediction.get("canonical_away_id", "?")
    st.markdown("**Canonical IDs:**")
    st.markdown(f"- home: `{canonical_home}`")
    st.markdown(f"- away: `{canonical_away}`")

    # Surface any prediction key that doesn't already have a section.
    handled = {
        "home_team", "away_team", "home_team_id", "away_team_id",
        "date", "primary_probs", "pi_probs", "blend_probs", "pi_only_probs",
        "elo_only_probs", "blend_was_used", "confidence", "banner",
        "canonical_home_id", "canonical_away_id", "identity_warnings",
        "_match_meta",
    }
    extras = {k: v for k, v in prediction.items() if k not in handled}
    if extras:
        st.markdown("**Additional fields:**")
        st.json(extras)

    st.markdown("**Full prediction JSON:**")
    st.json(prediction)
    st.markdown("**Match meta:**")
    st.json(match_meta)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _match_id_str(m: dict) -> str:
    """Return the match_id of a match dict as a string.

    Tolerates the historical ``id`` alias some caches used in early
    phases so the Analysis view remains robust to upstream changes.
    """
    raw = m.get("match_id")
    if raw is None:
        raw = m.get("id")
    if raw is None:
        return ""
    return str(raw)


def _lookup(
    mapping: dict | None,
    sel_id: str,
) -> dict | None:
    """Look up a prediction or market dict by both str and int keys.

    Session state may have int keys (from ``int(mid)``) while the
    selectbox uses string keys, so we try both shapes.
    """
    if not mapping:
        return None
    val = mapping.get(sel_id)
    if val is not None:
        return val
    # Fallback: try int key.
    try:
        return mapping.get(int(sel_id))
    except (TypeError, ValueError):
        return val


__all__ = ["render_analysis_view"]
