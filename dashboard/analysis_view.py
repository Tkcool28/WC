"""Renders the Analysis view (technical deep-dive per game).

Phase 7 of the dashboard rearchitecture reorganizes the Analysis view
into five transparent model sections plus supplementary technical context:

1.  **Primary Model** *(default OPEN)* — the official 60% Elo / 40% Goal
    blend, the selected outcome, the confidence tier, and which fallback
    source was used (or whether the full blend ran).
2.  **Elo** — Elo-only H/D/A probabilities, the rating difference, and
    availability flags for both teams.
3.  **Goal Model** — Goal-only H/D/A, home/away xG, expected total goals,
    most-likely scoreline, low-data flags, and the artifact cutoff/version.
4.  **Pi** — Pi-only H/D/A, diagnostic only, clearly labelled as NOT
    part of the primary blend.
5.  **Disagreement** — whether all three component models agree on the
    top outcome, the largest probability gap between any two models on
    the same market, and a concise warning when disagreement is strong.

After the model sections, the remaining technical sections (Market
Comparison, Poisson View, Squad Context, Group Context, Calibration and
Data Quality, Raw Diagnostics) are preserved as collapsed expanders.

The view works with **model-only** predictions (no odds required).  When
the user has run the same game through the Bets view, the
``KEYS.MARKET_BY_MATCH`` cache contains an ``evaluate_market(...)`` result
that the **Market Comparison** section will surface; otherwise the
section shows a calm message so the rest of the view remains useful for
technical users who only care about the model output.
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

    # ---- 1. Primary Model (default OPEN) ---- #
    with st.expander("🎯 Primary Model (60% Elo / 40% Goal)", expanded=True):
        _render_primary_model(prediction)

    # ---- 2. Elo ---- #
    with st.expander("📊 Elo", expanded=False):
        _render_elo_section(prediction)

    # ---- 3. Goal Model ---- #
    with st.expander("⚽ Goal Model", expanded=False):
        _render_goal_model_section(prediction)

    # ---- 4. Pi ---- #
    with st.expander("π Pi (diagnostic only)", expanded=False):
        _render_pi_section(prediction)

    # ---- 5. Disagreement ---- #
    with st.expander("⚠️ Disagreement", expanded=False):
        _render_disagreement_section(prediction)

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
# Section 1: Primary Model
# --------------------------------------------------------------------------- #
def _render_primary_model(prediction: dict) -> None:
    """Show the official blended prediction: 60% Elo / 40% Goal.

    Surfaces H/D/A probabilities, the selected outcome, the confidence
    tier, and which fallback source was used (or whether the full blend
    ran).  No hidden weights: the 60/40 split is documented in the
    section header itself.
    """
    from dashboard.prediction_card import (
        _extract_most_likely,
        _format_probability,
        _outcome_headline_text,
    )

    primary = (
        prediction.get("primary_probs")
        or prediction.get("blend_probs")
        or prediction.get("pi_probs")
        or {}
    )
    mlr = _extract_most_likely(prediction)
    headline = _outcome_headline_text(mlr, prediction)
    p_top = primary.get(mlr) if primary else None

    st.markdown(f"### {headline}")
    st.caption(f"Top-market probability: {_format_probability(p_top)}")

    # ---- H/D/A probabilities ---- #
    if primary:
        st.markdown("**Home / Draw / Away:**")
        for k in ("home", "draw", "away"):
            v = primary.get(k)
            if v is not None:
                st.markdown(f"- {k}: {_format_probability(v)}")

    # ---- Blend formula + fallback source ---- #
    st.markdown("**Blend formula:**")
    st.markdown("`primary = 0.60 × Elo + 0.40 × Goal Model`")

    _goal_model_used = prediction.get("_goal_model_used", False)
    _goal_model_expected = prediction.get("_goal_model_expected", False)
    _elo_used = prediction.get("blend_was_used", False)

    if _goal_model_used:
        st.caption("Full blend ran: Elo + Goal Model → primary_probs.")
    elif _goal_model_expected and _elo_used:
        st.caption(
            "⚠️ Goal model was expected but failed — fell back to "
            "Pi+Elo blend. primary_probs does NOT include goal model."
        )
    elif _elo_used:
        st.caption(
            "Elo-only fallback: goal model unavailable. "
            "primary_probs = Pi+Elo blend."
        )
    else:
        st.caption(
            "Pi-only fallback: Elo unavailable. "
            "primary_probs = Pi-rating only."
        )

    # ---- Confidence ---- #
    confidence = prediction.get("confidence") or {}
    if confidence:
        tier = confidence.get("tier", "")
        tier_desc = confidence.get("tier_description", "")
        if tier:
            st.markdown(f"**Confidence tier:** `{tier}` — {tier_desc}")
        cal_p = confidence.get("calibrated_p")
        if cal_p is not None:
            st.markdown(f"**Calibrated p:** `{cal_p:.3f}`")
        warnings = list(confidence.get("warnings") or [])
        if warnings:
            st.markdown("**Warnings:**")
            for w in warnings:
                st.markdown(f"- {w}")


# --------------------------------------------------------------------------- #
# Section 2: Elo
# --------------------------------------------------------------------------- #
def _render_elo_section(prediction: dict) -> None:
    """Show Elo-only H/D/A, rating difference, and availability flags."""
    elo_only = prediction.get("elo_only_probs") or {}
    home_elo = prediction.get("home_elo")
    away_elo = prediction.get("away_elo")

    if not elo_only and home_elo is None and away_elo is None:
        st.caption("No Elo diagnostics on this prediction.")
        return

    # ---- Elo H/D/A ---- #
    if elo_only:
        st.markdown("**Elo-only Home / Draw / Away:**")
        for k in ("home", "draw", "away"):
            v = elo_only.get(k)
            if v is not None:
                st.markdown(f"- {k}: {_format_probability(v)}")

    # ---- Rating difference ---- #
    if home_elo is not None and away_elo is not None:
        diff = float(home_elo) - float(away_elo)
        st.markdown(
            f"**Rating difference:** `{diff:+.0f}` "
            f"(home {float(home_elo):.0f} / away {float(away_elo):.0f})"
        )
    elif "blend_w_elo" in prediction:
        # blend_was_used=True but raw ratings not on dict — show weight only
        w_elo = prediction.get("blend_w_elo")
        st.markdown(f"**Elo weight in blend:** `{w_elo}`")

    # ---- Availability flags ---- #
    elo_available = prediction.get("blend_was_used", False) and elo_only
    st.markdown(
        f"**Elo available:** `{'Yes' if elo_available else 'No'}`"
    )
    if not elo_available:
        st.caption(
            "Elo ratings were not used in this prediction's blend. "
            "The primary model fell back to Pi or Pi+Goal."
        )


# --------------------------------------------------------------------------- #
# Section 3: Goal Model
# --------------------------------------------------------------------------- #
def _render_goal_model_section(prediction: dict) -> None:
    """Show Goal-only H/D/A, xG, scoreline, low-data flags, artifact info."""
    goal_used = prediction.get("_goal_model_used", False)
    goal_expected = prediction.get("_goal_model_expected", False)
    goal_probs = prediction.get("goal_model_hda") or {}
    xg = prediction.get("_goal_model_xg") or {}
    mls = prediction.get("_goal_model_most_likely_score")
    total_goals = prediction.get("_goal_model_expected_total_goals")
    low_data = prediction.get("_goal_model_low_data", False)
    low_data_flags = prediction.get("_goal_model_low_data_flags") or []
    model_version = prediction.get("_goal_model_version")
    data_cutoff = prediction.get("_goal_model_data_cutoff")

    # ---- Availability banner ---- #
    if not goal_used and not goal_expected:
        st.caption(
            "Goal model was not loaded for this prediction. "
            "Artifact not available at prediction time."
        )
        return

    if not goal_used and goal_expected:
        st.warning(
            "Goal model was expected but failed at prediction time. "
            "Fell back to Pi+Elo blend."
        )
        return

    # ---- Goal H/D/A ---- #
    if goal_probs:
        st.markdown("**Goal-model Home / Draw / Away:**")
        for k in ("home", "draw", "away"):
            v = goal_probs.get(k)
            if v is not None:
                st.markdown(f"- {k}: {_format_probability(v)}")

    # ---- xG + scoreline ---- #
    if xg:
        home_xg = xg.get("home_xg", "—")
        away_xg = xg.get("away_xg", "—")
        st.markdown(
            f"**Expected goals:** Home `{home_xg}`, Away `{away_xg}`"
        )
    if total_goals is not None:
        st.markdown(f"**Expected total goals:** `{total_goals:.2f}`")
    if mls:
        st.markdown(f"**Most likely scoreline:** `{mls[0]}-{mls[1]}`")

    # ---- Low-data flags ---- #
    if low_data:
        st.markdown(f"**Low-data:** `True`")
    if low_data_flags:
        st.markdown("**Low-data flags:**")
        for flag in low_data_flags:
            st.markdown(f"- `{flag}`")

    # ---- Artifact metadata ---- #
    if model_version or data_cutoff:
        meta_parts = []
        if model_version:
            meta_parts.append(f"model_version: `{model_version}`")
        if data_cutoff:
            meta_parts.append(f"data_cutoff: `{data_cutoff}`")
        if meta_parts:
            st.caption("Artifact: " + " · ".join(meta_parts))


# --------------------------------------------------------------------------- #
# Section 4: Pi (diagnostic only)
# --------------------------------------------------------------------------- #
def _render_pi_section(prediction: dict) -> None:
    """Show Pi-only H/D/A — diagnostic only, NOT part of primary blend."""
    pi_only = prediction.get("pi_only_probs") or {}

    if not pi_only:
        st.caption("No pi-rating diagnostics on this prediction.")
        return

    st.markdown("**Pi-only Home / Draw / Away:**")
    for k in ("home", "draw", "away"):
        v = pi_only.get(k)
        if v is not None:
            st.markdown(f"- {k}: {_format_probability(v)}")

    st.caption(
        "Pi probabilities are diagnostic only. They are NOT used in "
        "the primary blend (primary = 60% Elo + 40% Goal Model). "
        "Pi is shown here for reference and model-development purposes."
    )


# --------------------------------------------------------------------------- #
# Section 5: Disagreement
# --------------------------------------------------------------------------- #
def _render_disagreement_section(prediction: dict) -> None:
    """Concise disagreement analysis across the three component models.

    Answers:
    - Do all three models agree on the top outcome?
    - What is the largest probability gap on any single market?
    - Do Goal and Elo agree?
    - Does Pi disagree with the primary blend?
    - Emit a concise warning when disagreement is strong.
    """
    pi_only = prediction.get("pi_only_probs") or {}
    elo_only = prediction.get("elo_only_probs") or {}
    goal_probs = prediction.get("goal_model_hda") or {}
    primary = (
        prediction.get("primary_probs")
        or prediction.get("blend_probs")
        or prediction.get("pi_probs")
        or {}
    )

    # Determine which models actually ran
    has_pi = bool(pi_only)
    has_elo = bool(elo_only)
    has_goal = bool(goal_probs) and prediction.get("_goal_model_used", False)

    # Build a dict of {model_name: {market: prob}} for comparison
    models: dict[str, dict] = {}
    if has_pi:
        models["Pi"] = pi_only
    if has_elo:
        models["Elo"] = elo_only
    if has_goal:
        models["Goal"] = goal_probs

    if len(models) < 2:
        st.caption(
            "Only one prediction model ran — disagreement analysis "
            "requires at least two."
        )
        return

    # ---- Same top outcome? ---- #
    tops = {}
    for name, probs in models.items():
        if probs:
            tops[name] = max(("home", "draw", "away"), key=lambda m: probs.get(m, 0.0))

    unique_tops = set(tops.values())
    all_agree = len(unique_tops) == 1

    if all_agree:
        top_outcome = next(iter(unique_tops))
        agreeing = ", ".join(tops.keys())
        st.markdown(
            f"**Models agree:** Yes — all ({agreeing}) pick "
            f"`{top_outcome}`."
        )
    else:
        parts = [f"{name} picks `{out}`" for name, out in tops.items()]
        st.markdown(
            f"**Models agree:** No — {' / '.join(parts)}."
        )

    # ---- Largest gap on any single market ---- #
    max_gap = 0.0
    max_gap_market = ""
    for market in ("home", "draw", "away"):
        vals = [p.get(market, 0.0) for p in models.values()]
        gap = max(vals) - min(vals)
        if gap > max_gap:
            max_gap = gap
            max_gap_market = market

    st.markdown(
        f"**Largest gap:** {max_gap * 100:.1f} pts on `{max_gap_market}`."
    )

    # ---- Goal/Elo agreement ---- #
    if has_goal and has_elo:
        goal_top = max(("home", "draw", "away"), key=lambda m: goal_probs.get(m, 0.0))
        elo_top = max(("home", "draw", "away"), key=lambda m: elo_only.get(m, 0.0))
        if goal_top == elo_top:
            st.markdown(
                f"**Goal ↔ Elo:** Agree (both pick `{goal_top}`)."
            )
        else:
            st.markdown(
                f"**Goal ↔ Elo:** Disagree (Goal `{goal_top}` vs "
                f"Elo `{elo_top}`)."
            )
    else:
        st.markdown("**Goal ↔ Elo:** N/A (one or both missing)")

    # ---- Pi vs primary blend ---- #
    if has_pi and primary:
        pi_top = max(("home", "draw", "away"), key=lambda m: pi_only.get(m, 0.0))
        primary_top = max(("home", "draw", "away"), key=lambda m: primary.get(m, 0.0))
        if pi_top == primary_top:
            st.markdown(
                f"**Pi vs Primary:** Agree (both pick `{pi_top}`)."
            )
        else:
            st.markdown(
                f"**Pi vs Primary:** Disagree (Pi `{pi_top}` vs "
                f"Primary `{primary_top}`)."
            )

    # ---- Strong disagreement warning ---- #
    if not all_agree and max_gap >= 0.10:
        st.warning(
            "⚠️ Strong disagreement: models pick different top outcomes "
            f"with a {max_gap * 100:.1f}-pt gap on `{max_gap_market}`. "
            "Treat the primary blend with elevated caution."
        )
    elif not all_agree:
        st.info(
            "Models disagree on the top outcome but the gap is modest. "
            "The primary blend is still the official prediction."
        )


# --------------------------------------------------------------------------- #
# Remaining sections (preserved from Phase 5/6)
# --------------------------------------------------------------------------- #
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
        # Phase 7 model-layer fields (handled in new sections above):
        "_goal_model_used", "_goal_model_xg", "_goal_model_low_data",
        "_goal_model_expected", "_goal_model_most_likely_score",
        "_goal_model_expected_total_goals", "_goal_model_version",
        "_goal_model_data_cutoff", "_goal_model_low_data_flags",
        "goal_model_hda",
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


# --------------------------------------------------------------------------- #
# Pure helper for external use (formatting probabilities)
# --------------------------------------------------------------------------- #
def _format_probability(value: float | None) -> str:
    """Format a probability value as a percentage string.

    Returns a dash for None.  Used across all model sections.
    """
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


__all__ = ["render_analysis_view"]
