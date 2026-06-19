"""
Streamlit dashboard for the +EV soccer workflow (pi-rating + Elo blend).

Top-level information architecture (Phase 2):

* **🎯 Predictions** — model-only outputs, no odds required.
* **💰 Bets** — model outputs + book odds + edge/flag.
* **🔬 Analysis** — full diagnostic breakdown per game.

The legacy two-flow shape (Auto-populate / Manual entry) is preserved
in Phase 2 as stubs reachable from the new nav: Auto-populate lives
under Predictions and Manual lives under Bets. Phases 3 and 4 absorb
these stubs into proper Predictions / Bets renderers.

Both flows share the same renderer helpers and the same
`evaluate_match(...)` call from `soccer_ev_model.ev_workflow`, which
combines pi-rating (50%) with Elo ratings (50%) for a calibrated H/D/A
probability. Weights are hand-tuned from `scripts/blend_backtest.py`.

Mobile-friendly: single column, centered layout, large touch targets.
Caching: @st.cache_data on the training corpus + ratings keyed on the
match date so re-submissions are instant. Elo snapshots are loaded
once at startup from `data/raw/elo_ratings.json`.
"""
from __future__ import annotations

import json
import sys
from datetime import date as _date
from pathlib import Path

import pandas as pd
import streamlit as st

# Make `soccer_ev_model` importable when launched via `streamlit run`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from soccer_ev_model.ev_workflow import (  # noqa: E402
    evaluate_match,
    predict_match,
)
from soccer_ev_model.pi_backtest import load_matches  # noqa: E402
from soccer_ev_model.pi_ratings import compute_pi_ratings  # noqa: E402
from soccer_ev_model.elo_ratings import elo_at, load_elo_ratings  # noqa: E402
from soccer_ev_model.team_identity import (  # noqa: E402
    resolve_team as _resolve_team_identity,
)
from soccer_ev_model.prediction_summary import (  # noqa: E402
    calculate_market_deltas,
    expected_goals_from_blend,
    group_context_warnings,
    largest_market_delta,
    market_divergence_label,
    poisson_agreement_label,
    poisson_outcome_probs,
    resolve_model_probs_for_market,
)

from dashboard.data_loader import (  # noqa: E402
    UnplayedMatch,
    get_unplayed_matches,
    load_matches_cache,
)
from dashboard.context_loader import (  # noqa: E402
    TIER_TO_STYLE as _SQUAD_TIER_TO_STYLE,
    escape_note_text as _escape_note_text,
    format_eur as _format_eur,
    format_gap as _format_gap,
    get_match_context as _get_match_context,
    render_notes_bullets as _render_notes_bullets,
)
from dashboard.ux_presenters import (  # noqa: E402
    analysis_calibration_and_data_quality as _ux_analysis_calibration,
    analysis_market_comparison as _ux_analysis_market,
    analysis_model_breakdown as _ux_analysis_model,
    analysis_poisson_view as _ux_analysis_poisson,
    analysis_prediction_details as _ux_analysis_prediction,
    analysis_raw_diagnostics as _ux_analysis_raw,
    format_odds as _format_odds,
    most_likely_result as _most_likely_result,
    outcome_headline as _outcome_headline,
    prediction_confidence_label as _prediction_confidence_label,
    prediction_why_text as _prediction_why_text,
    translate_and_dedupe_warnings as _translate_and_dedupe_warnings,
    translate_warning as _translate_warning,
    value_confidence_label as _value_confidence_label,
    value_play as _value_play,
    value_why_text as _value_why_text,
)
from dashboard.session_state import (  # noqa: E402
    KEYS,
    get as _ss_get,
    pop as _ss_pop,
    set_ as _ss_set,
)
from dashboard.styles import inject_css as _inject_css  # noqa: E402
from dashboard.prediction_card import render_prediction_card as _render_prediction_card  # noqa: E402
from dashboard.bet_card import render_bet_card as _render_bet_card  # noqa: E402
from dashboard.text_format import (  # noqa: E402
    format_group_label as _format_group_label,
    format_matchday_label as _format_matchday_label,
)
from dashboard.context_cards import (  # noqa: E402
    autoload_context_for_date as _autoload_context_for_date,
    build_tournament_snapshot as _build_tournament_snapshot,
    highest_model_confidence as _highest_model_confidence,
    pick_smart_default_date as _pick_smart_default_date,
    render_highest_confidence as _render_highest_confidence,
    render_tournament_snapshot as _render_tournament_snapshot,
)
from dashboard.data_loader import (  # noqa: E402
    list_dates_with_unplayed as _list_dates_with_unplayed,
)
from dashboard.team_resolution import (  # noqa: E402
    resolve_match_for_prediction as _resolve_match_for_prediction,
)
from soccer_ev_model.goal_model_cached import get_goal_predictor as _get_goal_predictor  # noqa: E402


# --------------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="+EV Soccer Dashboard",
    page_icon="⚽",
    layout="centered",
)


# --------------------------------------------------------------------------- #
# Static assets
# --------------------------------------------------------------------------- #
INTL_PATH = _PROJECT_ROOT / "data" / "processed" / "international_matches.json"
ELO_PATH = _PROJECT_ROOT / "data" / "raw" / "elo_ratings.json"
WC_YEARS = (2010, 2014, 2018, 2022)

TIER_TO_STYLE = {
    "A": ("success", "🟢"),
    "B": ("success", "🟡"),
    "C": ("warning", "🟠"),
    "D": ("error",   "🔴"),
}

MARKET_LABEL = {"home": "Home Win", "draw": "Draw", "away": "Away Win"}

# Anchor "today" for the auto-populate flow. Used as a *fallback* when
# the on-disk schedule is empty or unreadable; the live default in the
# date picker comes from :func:`_smart_default_date` below.
DEFAULT_TODAY = _date(2026, 6, 16)


def _smart_default_date() -> _date:
    """Return the smart default date for the date_input widgets.

    Computed once per Streamlit session (cached on the first call) from
    today's date and the list of dates with unplayed matches on disk.
    Falls back to :data:`DEFAULT_TODAY` if the schedule is unreadable.

    The "compute once" rule keeps the value stable across the seven
    ``st.date_input`` widgets in the dashboard — once the user has
    clicked through to a particular date, every subsequent widget sees
    the same default; user choices are preserved by Streamlit's
    key-based session state.
    """
    global _SMART_DEFAULT_CACHE
    if _SMART_DEFAULT_CACHE is not None:
        return _SMART_DEFAULT_CACHE
    try:
        _SMART_DEFAULT_CACHE = _pick_smart_default_date(
            _date.today(), _list_dates_with_unplayed(),
        )
    except Exception:
        # Defensive: never crash the page over a default.
        _SMART_DEFAULT_CACHE = DEFAULT_TODAY
    return _SMART_DEFAULT_CACHE


# Module-level cache for :func:`_smart_default_date`. ``None`` means
# "not yet computed"; the helper sets it on the first invocation.
_SMART_DEFAULT_CACHE: _date | None = None


# --------------------------------------------------------------------------- #
# Cached data loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading training corpus…")
def load_training_corpus() -> list[dict]:
    """Load 32k intl matches + WC years. Sorted chronologically."""
    intl = json.loads(INTL_PATH.read_text())
    wc: list[dict] = []
    for y in WC_YEARS:
        try:
            wc.extend(load_matches(y))
        except FileNotFoundError:
            # Tolerate a missing WC year without breaking the dashboard.
            pass
    corpus = list(intl) + list(wc)
    corpus.sort(key=lambda m: m.get("date", ""))
    return corpus


@st.cache_data(show_spinner="Building team name → id map…")
def build_name_to_id(corpus: list[dict]) -> dict[str, int]:
    """Build a {team_name: team_id} map from the training corpus.

    The first id we see for a given name wins. The international dataset
    uses names that are consistent across decades, so this is reliable.
    """
    name_to_id: dict[str, int] = {}
    for m in corpus:
        if "home_team" in m and "home_team_id" in m:
            name_to_id.setdefault(m["home_team"], m["home_team_id"])
        if "away_team" in m and "away_team_id" in m:
            name_to_id.setdefault(m["away_team"], m["away_team_id"])
    return name_to_id


@st.cache_data(show_spinner="Computing pi-ratings…")
def get_ratings(cutoff_iso: str, _corpus: list[dict]) -> dict:
    """Compute pi-ratings for matches strictly before `cutoff_iso`.

    The leading underscore on `_corpus` tells Streamlit not to hash the
    (large) list — it just re-runs if the function is invalidated.
    """
    train = [m for m in _corpus if m.get("date", "") < cutoff_iso]
    return compute_pi_ratings(train, cutoff=cutoff_iso)


@st.cache_data(show_spinner="Loading Elo ratings…")
def get_elo_snapshots() -> dict:
    """Load the cached Elo ratings from data/raw/elo_ratings.json.

    Returns an empty dict if the cache file is missing — the dashboard
    then falls back to pure pi-rating (Elo defaults to 1500 for all teams).
    """
    if not ELO_PATH.exists():
        return {}
    return load_elo_ratings(ELO_PATH)


# The auto-populate read is also cached — same file, same content, instant
# on second load. Keyed on (date_iso, file mtime) so it auto-refreshes if
# the user runs `scripts/fetch_live_2026.py` and the snapshot updates.
@st.cache_data(show_spinner=False)
def _get_unplayed_for_date_cached(date_iso: str, cache_path_str: str) -> list[dict]:
    cache_path = Path(cache_path_str)
    if not cache_path.exists():
        return []
    matches = get_unplayed_matches(date_iso, cache_path=cache_path)
    return [m.to_dict() for m in matches]


def _load_unplayed_for_date(date_iso: str) -> list[dict]:
    """Cache-aware wrapper around the loader, used by the UI."""
    from dashboard.data_loader import DEFAULT_CACHE_PATH
    return _get_unplayed_for_date_cached(date_iso, str(DEFAULT_CACHE_PATH))


def _schedule_cache_present() -> bool:
    """Return True when the 2026 schedule cache file exists on disk.

    The dashboard uses this to distinguish two distinct empty states:

    1. **Cache file missing** — schedule data has not been generated yet.
       The user sees a calm "Schedule data isn't loaded yet" pointer and
       a Custom matchup escape hatch.  We never show the on-disk path
       or any recovery CLI (those are internal/operational).

    2. **Cache file present, empty for this date** — schedule data is
       loaded, but no games are scheduled for the chosen date.  The
       user sees a "Pick another date" pointer.

    Both paths use plain language; neither leaks the cache path or any
    internal script reference.
    """
    try:
        from dashboard.data_loader import DEFAULT_CACHE_PATH
        return bool(DEFAULT_CACHE_PATH.exists())
    except Exception:
        return False


def _render_schedule_data_not_loaded() -> None:
    """Render the calm 'schedule data not loaded yet' empty state.

    No script path, no CLI hint — just a clear pointer to the Custom
    matchup expander below.  Used by Predictions, Bets, and Analysis
    when the 2026 schedule cache is missing.
    """
    st.info(
        "Schedule data isn't loaded yet. Check back later, or use the "
        "**Custom matchup** expander below to predict any game."
    )


# --------------------------------------------------------------------------- #
# Renderers (shared by both flows)
# --------------------------------------------------------------------------- #
def _render_confidence_banner(assessment: dict, banner: str) -> None:
    """Render the confidence tier as a colored banner with the full text."""
    style, emoji = TIER_TO_STYLE.get(assessment["tier"], ("info", "❔"))
    fn = getattr(st, style, st.info)
    fn(banner)
    st.caption(
        f"Calibration: top_p={assessment['top_p']:.3f} → "
        f"calibrated {assessment['calibrated_p']:.3f} "
        f"(diff {assessment['calibration_diff']:+.3f}, "
        f"{assessment['calib_label']})"
    )


def _render_warnings(assessment: dict) -> None:
    warnings = assessment.get("warnings") or []
    for w in warnings:
        st.warning(f"⚠️ {w}")


def _render_market_baseline(result: dict) -> None:
    """Render the market baseline section (model vs no-vig book comparison).

    Reads ``result['primary_probs']`` (via ``resolve_model_probs_for_market``)
    and ``result['book_fair']``, then shows the per-market deltas, a
    divergence label, and the outcome with the largest disagreement.
    Pure presentation: no I/O, no model calls.
    """
    model_probs = resolve_model_probs_for_market(result)
    market_probs = result["book_fair"]
    # Pct-pt deltas (output of calculate_market_deltas) for display & largest.
    pts_deltas = calculate_market_deltas(model_probs, market_probs)
    # Raw-probability deltas for the divergence label (its thresholds are
    # documented in the 0-1 scale, not in pts).
    raw_deltas = {m: model_probs[m] - market_probs[m] for m in ("home", "draw", "away")}
    div_label = market_divergence_label(raw_deltas)

    home_name = result.get("home_team", "Home")
    away_name = result.get("away_team", "Away")
    market_labels = {"home": home_name, "draw": "Draw", "away": away_name}
    largest = largest_market_delta(
        pts_deltas,
        market_labels=market_labels,
        model_probs=model_probs,
        market_probs=market_probs,
    )

    st.subheader("Market baseline (model vs book no-vig)")
    st.caption(
        "Manual book odds input. No live odds feed. This is a model-vs-market "
        "comparison, not an action recommendation."
    )

    def _row(market_key: str) -> str:
        return (
            f"{MARKET_LABEL[market_key]:<9} "
            f"{model_probs[market_key] * 100:5.1f}% model  /  "
            f"{market_probs[market_key] * 100:5.1f}% market  /  "
            f"{pts_deltas[market_key]:+.1f} pts"
        )

    lines = [
        "```",
        "Model vs Market",
        _row("home"),
        _row("draw"),
        _row("away"),
        (
            f"Market read: {div_label} — "
            f"{largest['label']} {largest['delta_pts']:+.1f} pts vs market"
        ),
        "```",
    ]
    st.markdown("\n".join(lines))


def _render_poisson_summary(result: dict) -> None:
    """Render the secondary Poisson goal-model block (transparent xG view).

    Computes expected home/away goals from the blend 1X2 distribution via
    ``expected_goals_from_blend``, then runs an independent-Poisson score
    matrix through ``poisson_outcome_probs`` to get a parallel 1X2
    estimate.  This is a *secondary, transparent, non-ML* view shown
    alongside the main blend; it never modifies the blend probabilities
    or any other field of ``result``.

    The block is intentionally compact and phone-friendly: a subheader,
    a one-line xG estimate, a one-line Poisson 1X2, and a one-line
    agreement label.
    """
    model_probs = resolve_model_probs_for_market(result)
    xg = expected_goals_from_blend(model_probs)
    poisson_probs = poisson_outcome_probs(xg["home_xg"], xg["away_xg"])
    agreement = poisson_agreement_label(model_probs, poisson_probs)

    home_name = result.get("home_team", "Home")
    away_name = result.get("away_team", "Away")

    market_labels = {"home": home_name, "draw": "Draw", "away": away_name}
    blend_top = market_labels[agreement["blend_top"]]
    poisson_top = market_labels[agreement["poisson_top"]]

    st.subheader("Poisson goal model (secondary view)")
    st.caption(
        "Transparent expected-goals approximation. Independent of the "
        "main blend. Not from a trained model."
    )

    st.markdown(
        "```\n"
        "Poisson goal model (secondary, transparent)\n"
        f"xG estimate: {home_name} {xg['home_xg']} / {away_name} {xg['away_xg']}\n"
        f"Home {poisson_probs['home'] * 100:.1f}% / "
        f"Draw {poisson_probs['draw'] * 100:.1f}% / "
        f"Away {poisson_probs['away'] * 100:.1f}%\n"
        f"Blend top: {blend_top}\n"
        f"Poisson top: {poisson_top}\n"
        f"Poisson agreement: {agreement['label']}\n"
        "```"
    )


def _render_plus_ev_flags(result: dict, min_edge: float) -> None:
    flags = result.get("plus_ev_flags") or []
    st.subheader(f"+EV flags  (edge ≥ {min_edge:.0%})")
    if not flags:
        st.info("No +EV markets at this threshold.")
        return

    cols = st.columns(min(3, len(flags)))
    for i, f in enumerate(flags):
        market = f["market"]
        label = MARKET_LABEL[market]
        edge = f["edge"]
        col = cols[i % len(cols)]
        col.metric(
            label=label,
            value=f"{edge:+.1%}",
            delta=f"model {f['calibrated_pi']:.1%} vs book {f['book_fair']:.1%}",
            delta_color="normal",
        )


def _render_group_context(
    result: dict,
    *,
    stage: str = "",
    matchday: int | None = None,
    group: str = "",
    finished_matches_in_group: list[dict] | None = None,
) -> None:
    """Render the group-stage context warning block (Phase 3).

    Pure presentation layer.  Reads nothing from ``result`` (the model
    output), calls ``group_context_warnings`` from
    ``soccer_ev_model.prediction_summary``, and emits a compact
    Streamlit block.  The model probabilities are never read, never
    modified, never recomputed.

    Behaviour:

    * Knockout / unknown stage → ``group_context_warnings`` returns
      ``[]`` and this function renders nothing (no subheader, no
      caption).  The rest of the dashboard looks exactly as before.
    * Group stage → a subheader + caption + a bullet list of warning
      texts.  Items with ``severity == 'warning'`` are prefixed with
      a warning emoji so the user can scan for the high-priority
      ones.

    Args:
        result: the ``evaluate_match`` output dict.  Only the
            ``home_team``/``away_team`` names are read (for the
            caption) — no probability fields are touched.
        stage, matchday, group: from the source match metadata.
        finished_matches_in_group: optional list of finished matches
            in the same group; passed through to the helper.
    """
    _ = result  # intentionally unused: this layer does not read model probs
    warnings = group_context_warnings(
        stage=stage,
        matchday=matchday,
        group=group,
        finished_matches_in_group=finished_matches_in_group,
    )
    if not warnings:
        return  # knockout or no group → render nothing

    st.subheader("Group context (warning only)")
    st.caption("Context only — not included in model probability.")

    bullet_lines: list[str] = []
    for w in warnings:
        text = w.get("text", "")
        if w.get("severity") == "warning":
            bullet_lines.append(f"- ⚠️ {text}")
        else:
            bullet_lines.append(f"- {text}")
    st.markdown("\n".join(bullet_lines))


def _render_squad_context(
    result: dict,
    home_canonical_id: str,
    away_canonical_id: str,
) -> None:
    """Render the Phase 4 squad-strength context panel (display-only).

    This block is *purely cosmetic*:

    * It reads the manually curated ``data/manual/*.csv`` files via
      :func:`dashboard.context_loader.get_match_context` — no API,
      no scraping, no model calls.
    * It does NOT call :func:`soccer_ev_model.ev_workflow.evaluate_match`
      or any other probability function.
    * It does NOT modify ``result`` in any way.

    Layout (per the Phase 4 spec):

    1. Squad market value comparison (home vs away)
    2. Value tier badge for each side (success/info/warning)
    3. Gap vs opponent % (signed, with arrow) — "—" if either missing
    4. FIFA ranking fallback when squad value is missing
    5. Notes (injury/absence/rotation/motivation) as a bullet list
    6. Source name + snapshot date at the bottom

    Renders nothing-fancy for missing/empty teams: just "Unknown"
    so the panel never crashes.
    """
    home_name = result.get("home_team", "Home")
    away_name = result.get("away_team", "Away")
    match_ctx = _get_match_context(home_canonical_id or "", away_canonical_id or "")
    home_ctx = match_ctx["home"]
    away_ctx = match_ctx["away"]
    gap = match_ctx["gap"]

    st.subheader("Squad strength context")
    # Spec-mandated exact label. Do NOT change wording.
    st.caption("Context only — not included in the probability model yet.")

    # ---- (a) squad market value comparison ---- #
    home_eur = _format_eur(home_ctx.get("squad_value"))
    away_eur = _format_eur(away_ctx.get("squad_value"))
    st.markdown(
        f"**Squad market value:** {home_eur} &nbsp;vs&nbsp; {away_eur}"
    )

    # ---- (b) value tier per side with coloured badge ---- #
    def _tier_badge(ctx: dict) -> None:
        tier = ctx.get("value_tier") or "unknown"
        style = _SQUAD_TIER_TO_STYLE.get(tier, "info")
        fn = getattr(st, style, st.info)
        fn(f"Value tier: {tier}")

    tcol1, tcol2 = st.columns(2)
    with tcol1:
        st.markdown(f"**{_escape_note_text(home_name)}**")
        _tier_badge(home_ctx)
    with tcol2:
        st.markdown(f"**{_escape_note_text(away_name)}**")
        _tier_badge(away_ctx)

    # ---- (c) gap vs opponent ---- #
    gcol1, gcol2 = st.columns(2)
    gcol1.markdown(
        f"**Gap vs opponent (home):** {_format_gap(gap.get('home_pct'))}"
    )
    gcol2.markdown(
        f"**Gap vs opponent (away):** {_format_gap(gap.get('away_pct'))}"
    )

    # ---- (d) FIFA ranking fallback when squad value missing ---- #
    for side_label, ctx in (("Home", home_ctx), ("Away", away_ctx)):
        if ctx.get("squad_value") is None:
            rank = ctx.get("fifa_rank")
            date = ctx.get("snapshot_date") or ""
            if rank is not None:
                date_part = f" ({date})" if date else ""
                st.caption(f"{side_label} FIFA rank: #{rank}{date_part}")
            else:
                st.caption(f"{side_label} FIFA rank: Unknown")

    # ---- (e) notes (injury/absence/rotation/motivation/other) ---- #
    notes = (home_ctx.get("notes") or []) + (away_ctx.get("notes") or [])
    if notes:
        st.markdown("**Team notes (curated, manual):**")
        bullets = _render_notes_bullets(notes)
        if bullets:
            st.markdown(bullets)

    # ---- (f) source + snapshot date at bottom ---- #
    snap = (home_ctx.get("snapshot_date") or away_ctx.get("snapshot_date") or "").strip()
    source = home_ctx.get("source") or "Transfermarkt-style manual snapshot"
    if snap:
        st.caption(f"Source: {source} — snapshot {snap}")
    else:
        st.caption(f"Source: {source}")


def _finished_matches_in_group_from_cache(group: str) -> list[dict]:
    """Read the WC 2026 cache and return finished matches in `group`.

    Used by the auto-populate renderer to feed the Phase 3 standings
    warning.  Returns an empty list if the cache is missing, the group
    is empty, or there are simply no finished matches yet.

    The function reads from the same on-disk cache the auto-populate
    flow already uses (``data/raw/matches_2026.json``).  It is NOT
    used by the manual flow (which has no group metadata).
    """
    if not group:
        return []
    try:
        payload = load_matches_cache()
    except Exception:
        return []
    raw = payload.get("matches") or []
    out: list[dict] = []
    for m in raw:
        if (m.get("group") or "") != group:
            continue
        status = (m.get("status") or "").upper()
        if status != "FINISHED":
            continue
        # Keep only the fields the standings helper needs (and a few
        # extras for the warning text).  This avoids leaking the entire
        # raw API payload downstream.
        out.append({
            "home_team_id": m.get("home_team_id"),
            "away_team_id": m.get("away_team_id"),
            "home_team_name": m.get("home_team_name"),
            "away_team_name": m.get("away_team_name"),
            "home_goals": m.get("home_goals"),
            "away_goals": m.get("away_goals"),
            "date": m.get("date"),
        })
    return out


def _parse_american_odds(text: str) -> float:
    """Parse American odds from a text input, raising ValueError on bad input."""
    s = (text or "").strip().replace("+", "")
    if not s:
        raise ValueError("empty")
    return float(s)


def _pretty_market(market: str, home_name: str, away_name: str) -> str:
    """Convert a market key ('home'|'draw'|'away') to a human label."""
    return {"home": home_name, "away": away_name, "draw": "Draw"}[market]


# --------------------------------------------------------------------------- #
# Per-game evaluation (pure, testable)
# --------------------------------------------------------------------------- #
def evaluate_one_game(
    home_name: str,
    away_name: str,
    home_team_id: int | None,
    away_team_id: int | None,
    cutoff_iso: str,
    home_odds_txt: str,
    draw_odds_txt: str,
    away_odds_txt: str,
    ratings: dict,
    min_edge: float,
    name_to_id: dict[str, int],
    elo_snapshots: dict | None = None,
) -> dict:
    """Validate inputs for one auto-populated game and call evaluate_match.

    Returns a dict:
        {"ok": True, "result": <evaluate_match output>} on success
        {"ok": False, "error": "<user-facing message>"} on validation failure

    The dashboard uses this so the per-game form has a single source of
    truth for "is the input usable?" and the renderer can show errors
    inline (per game) rather than aborting the whole stacked view.
    """
    h = (home_name or "").strip()
    a = (away_name or "").strip()
    if not h or not a:
        return {"ok": False, "error": "Missing home or away team."}
    if h.lower() == a.lower():
        return {"ok": False, "error": "Home and away teams must differ."}

    # Resolve canonical identities for both teams. The cache's
    # home_team_id / away_team_id are football-data.org integers (small,
    # typically in the 700-10000 range) but the training corpus uses
    # different openfootball integers. We use the identity registry to
    # translate; if the registry already has the id (e.g. it IS a corpus
    # id), we pass it through unchanged. Identity warnings are surfaced
    # in the result so the dashboard can render them.
    h_res = _resolve_team_identity(football_data_id=home_team_id, name=h)
    a_res = _resolve_team_identity(football_data_id=away_team_id, name=a)

    identity_warnings: list[str] = []
    canonical_home_id = h_res.get("canonical_id")
    canonical_away_id = a_res.get("canonical_id")

    # Prefer the team IDs that came from the cache (always populated for
    # 2026 matches); fall back to the corpus map for robustness. If the
    # cache id is a football-data id (small int), translate through the
    # canonical registry. If it's a corpus id, the registry lookup by
    # corpus_id confirms and we pass it through.
    def _resolve_team_id(provided_id, name_hint, res):
        """Translate the cache's id to a corpus id usable by pi-rating.

        Returns (id, warning_or_None). If both the registry AND the
        name_to_id fallback have nothing, returns (None, warning) so the
        caller can show a hard error.

        - If `provided_id` is already a corpus key in `ratings`, return
          it unchanged.
        - If the registry resolved a canonical id and we have a corpus
          id, return the corpus id (translated).
        - If the registry resolved a canonical id but corpus_id is null
          (history_missing), return the provided id and warn (the model
          will see a missing/zero entry in ratings and produce a
          neutral result; we don't want to fail the whole game just
          because CPV/COD/CUW have no history).
        - If the registry returned identity_unresolved, we still return
          the provided_id (so the model produces a neutral prediction)
          and surface the warning, instead of hard-failing the row.
        - Last resort: fall back to the corpus name map.
        """
        # 1) Already present in ratings (e.g. caller passed a corpus id)
        if provided_id is not None and provided_id in (ratings or {}):
            return provided_id, None
        # 2) Registry resolved a corpus id we should use
        if provided_id is not None and res.get("corpus_id") is not None:
            return res["corpus_id"], None
        # 3) Registry resolved a canonical id but no corpus_id
        #    (history_missing). Pass the provided id through (the model
        #    will see neutral / missing ratings for it) and warn.
        if provided_id is not None and res.get("canonical_id") is not None:
            warning = (
                f"Team '{name_hint}' has no training-corpus history "
                f"(canonical={res['canonical_id']}, "
                f"status={res.get('status')}). Using neutral pi-rating."
            )
            return provided_id, warning
        # 4) Registry returned identity_unresolved. Don't hard-fail the
        #    whole row — surface a warning and pass the provided id
        #    through (so the model produces a neutral prediction).
        if provided_id is not None:
            warning = (
                f"Team '{name_hint}' could not be resolved via the "
                f"canonical identity registry "
                f"(canonical_id={res.get('canonical_id')}, "
                f"fd_id={res.get('source_team_id')}). "
                f"Using neutral pi-rating."
            )
            return provided_id, warning
        # 5) No id provided. Try the corpus name map.
        fallback = name_to_id.get(name_hint)
        if fallback is not None:
            return fallback, None
        # 6) Truly unknown: caller will return an error.
        return None, (
            f"Team '{name_hint}' could not be resolved to a training-corpus "
            f"team id (canonical_id={res.get('canonical_id')}, "
            f"fd_id={res.get('source_team_id')})."
        )

    h_id, h_warn = _resolve_team_id(home_team_id, h, h_res)
    a_id, a_warn = _resolve_team_id(away_team_id, a, a_res)
    if h_warn is not None:
        identity_warnings.append(h_warn)
    if a_warn is not None:
        identity_warnings.append(a_warn)

    if h_id is None:
        return {"ok": False, "error": f"Team '{h}' not in training data."}
    if a_id is None:
        return {"ok": False, "error": f"Team '{a}' not in training data."}

    try:
        h_odds = _parse_american_odds(home_odds_txt)
        d_odds = _parse_american_odds(draw_odds_txt)
        a_odds = _parse_american_odds(away_odds_txt)
    except ValueError:
        return {
            "ok": False,
            "error": (
                "Could not parse one of the odds. Use American format "
                "(e.g. -230 or +350)."
            ),
        }

    if not ratings:
        return {"ok": False, "error": f"No ratings for cutoff {cutoff_iso}."}

    # Look up Elo for both teams (if available). The blend uses a 50/50
    # mix of pi-rating and Elo, which scored 0.222 RPS on the 2022 WC
    # walk-forward (vs 0.230 for pi-only).
    home_elo, away_elo = None, None
    if elo_snapshots:
        home_elo, _ = elo_at(elo_snapshots, h, cutoff_iso)
        away_elo, _ = elo_at(elo_snapshots, a, cutoff_iso)

    result = evaluate_match(
        home_team=h,
        away_team=a,
        home_team_id=h_id,
        away_team_id=a_id,
        date=cutoff_iso,
        book_home_odds=h_odds,
        book_draw_odds=d_odds,
        book_away_odds=a_odds,
        ratings=ratings,
        min_edge=min_edge,
        home_elo=home_elo,
        away_elo=away_elo,
        identity_unresolved=(
            h_res.get("status") == "identity_unresolved"
            or a_res.get("status") == "identity_unresolved"
        ),
    )
    # Surface identity resolution results in the same dict the renderer reads.
    # Empty list = both teams fully resolved; non-empty = human-readable
    # warning per team that could not be translated or has no corpus history.
    result["identity_warnings"] = list(identity_warnings)
    result["canonical_home_id"] = canonical_home_id
    result["canonical_away_id"] = canonical_away_id
    return {"ok": True, "result": result}


def _render_section_table(rows: list[tuple[str, str]]) -> None:
    """Render a (label, content) list as a compact table for the Analysis tab."""
    if not rows:
        st.caption("_(no data)_")
        return
    df = pd.DataFrame(
        [{"Item": label, "Value": content} for label, content in rows]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Three-tab per-game renderers (UX-only refactor)
# --------------------------------------------------------------------------- #
def _render_prediction_tab(result: dict, identity_warnings: list[str]) -> None:
    """Render the casual Prediction tab (default landing for advanced users
    drilling into a game result, but always shown first in the per-game block).

    Pure presentation. Reads only from the existing ``result`` dict and the
    precomputed identity warnings. Does NOT touch model probabilities, market
    math, or squad-strength math.

    Layout:
      1. Big "Most Likely Result" card.
      2. Prediction Confidence (High / Medium / Low) badge.
      3. Compact "Why?" expander with a short plain-language reason.
      4. Optional translated warnings (no raw internal codes).
    """
    mlr = _most_likely_result(result)
    pcl = _prediction_confidence_label(result)
    assessment = result.get("confidence", {}) or {}
    raw_warnings = list(assessment.get("warnings") or [])
    style_name, _emoji = TIER_TO_STYLE.get(
        {"High": "A", "Medium": "B", "Low": "C"}.get(pcl, "C"),
        ("info", "❔"),
    )
    fn = getattr(st, style_name, st.info)

    # ---- (1) big result card ---- #
    st.markdown("##### Most Likely Result")
    st.markdown(
        f"<div style='font-size:1.6em; font-weight:600;'>"
        f"{_escape_note_text(_outcome_headline(mlr))}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Model probability: {mlr['probability']:.1%}")

    # ---- (2) confidence badge ---- #
    fn(f"Prediction Confidence: {pcl}")

    # ---- (3) Why expander ---- #
    with st.expander("Why?", expanded=False):
        st.markdown(
            _prediction_why_text(
                result,
                warnings=raw_warnings,
                identity_warnings=identity_warnings,
            )
        )

    # ---- (4) translated warnings (no raw internal codes) ---- #
    all_raw = list(identity_warnings) + raw_warnings
    translated = [_translate_warning(w) for w in all_raw]
    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for t in translated:
        if t and t not in seen:
            seen.add(t)
            deduped.append(t)
    if deduped:
        st.markdown("**Things to know:**")
        for line in deduped:
            st.markdown(f"- {line}")


def _render_betting_value_tab(
    result: dict, min_edge: float, identity_warnings: list[str]
) -> None:
    """Render the Betting Value tab (answers "do the available odds offer
    a worthwhile value opportunity?").

    Pure presentation. Does NOT recompute model probabilities or market
    math. Uses only the existing ``result['plus_ev_flags']``, book odds,
    and confidence assessment.

    Layout:
      1. Entered sportsbook odds (American format).
      2. Best Value Play OR "No Clear Value" card.
      3. Value Confidence (independent of Prediction Confidence).
      4. Compact "Why is this value?" expander.
      5. Expandable "Advanced market details" (implied / no-vig / model /
         edge / EV / divergence / raw odds).
    """
    book_odds = result.get("book_odds") or {}
    vp = _value_play(result, min_edge)
    vcl = _value_confidence_label(vp, result)
    style_name, _emoji = TIER_TO_STYLE.get(
        {"High": "A", "Medium": "B", "Low": "C"}.get(vcl, "C"),
        ("info", "❔"),
    )
    fn = getattr(st, style_name, st.info)

    # ---- (1) entered odds ---- #
    st.markdown("##### Entered sportsbook odds")
    odds_cols = st.columns(3)
    odds_cols[0].markdown(
        f"**Home**: {_format_odds(book_odds.get('home'))}"
    )
    odds_cols[1].markdown(
        f"**Draw**: {_format_odds(book_odds.get('draw'))}"
    )
    odds_cols[2].markdown(
        f"**Away**: {_format_odds(book_odds.get('away'))}"
    )
    st.caption("American format (e.g. -230 favorite, +350 underdog).")

    st.markdown("---")

    # ---- (2) Best Value Play / No Clear Value card ---- #
    if vp["status"] == "play":
        market_key = vp["market"]
        market_label = _pretty_market(
            market_key,
            result.get("home_team", "Home"),
            result.get("away_team", "Away"),
        )
        st.markdown("##### Best Value Play")
        st.markdown(
            f"<div style='font-size:1.4em; font-weight:600;'>"
            f"{_escape_note_text(market_label)} at "
            f"{_format_odds(vp.get('odds'))}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Model: {vp.get('model_p', 0):.1%}  ·  "
            f"Market: {vp.get('market_p', 0):.1%}  ·  "
            f"Edge: {vp.get('edge', 0):+.1%}"
        )
    else:
        st.markdown("##### Best Value Play")
        st.markdown(
            "<div style='font-size:1.4em; font-weight:600;'>"
            "No Clear Value"
            "</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"No outcome offers enough value at the entered odds "
            f"(edge threshold: {min_edge:.0%})."
        )

    # ---- (3) Value Confidence badge (independent of Prediction) ---- #
    fn(f"Value Confidence: {vcl}")

    # ---- (4) Why is this value? expander ---- #
    with st.expander("Why is this value?", expanded=False):
        st.markdown(_value_why_text(vp, result))
        if vp["status"] == "play":
            mlr = _most_likely_result(result)
            if vp["market"] != mlr["market"]:
                st.caption(
                    f"_Note: the most likely result is "
                    f"{mlr['label']} ({mlr['probability']:.1%}). "
                    f"The best value is on a different market._"
                )

    # ---- (5) advanced market details expander ---- #
    with st.expander("Advanced market details", expanded=False):
        _render_market_baseline(result)
        _render_plus_ev_flags(result, min_edge=min_edge)
        st.json({
            "book_odds": result.get("book_odds", {}),
            "book_fair": result.get("book_fair", {}),
            "calibrated_pi": result.get("calibrated_pi", {}),
            "edges": result.get("edges", {}),
            "plus_ev_flags": result.get("plus_ev_flags", []),
        })


def _render_analysis_tab(
    result: dict,
    match_meta: dict | None,
    min_edge: float,
    identity_warnings: list[str],
    home_canonical_id: str,
    away_canonical_id: str,
) -> None:
    """Render the Analysis tab (preserves all existing technical info).

    All existing renderers and the technical expanders from the previous
    layout are still called here.  Nothing is removed; it is reorganized
    into organized expandable sections.  Raw identity warnings are shown
    (NOT translated) in the Calibration / Data Quality section.
    """
    with st.expander("Prediction Details", expanded=False):
        _render_section_table(_ux_analysis_prediction(result))
        # Keep the original confidence banner and warnings visible here
        # (advanced users want to see the exact tier + warnings list).
        _render_confidence_banner(result["confidence"], result["banner"])
        _render_warnings(result["confidence"])

    with st.expander("Model Breakdown", expanded=False):
        _render_section_table(_ux_analysis_model(result))
        # Keep the original Pi / Elo / Blend probability table
        blended = result.get("primary_probs", result.get("blend_probs", result["pi_probs"]))
        pi_only = result.get("pi_only_probs") or blended
        elo_only = result.get("elo_only_probs")
        rows = []
        for market in ("home", "draw", "away"):
            rows.append({
                "Market": _pretty_market(
                    market, result["home_team"], result["away_team"]
                ),
                "Pi only": f"{pi_only[market]:.1%}",
                "Elo only": (
                    f"{elo_only[market]:.1%}" if elo_only is not None else "—"
                ),
                "Blend": f"{blended[market]:.1%}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Market Comparison", expanded=False):
        _render_section_table(_ux_analysis_market(result))
        _render_market_baseline(result)
        _render_plus_ev_flags(result, min_edge=min_edge)

    with st.expander("Poisson Score View", expanded=False):
        _render_section_table(_ux_analysis_poisson(result))
        _render_poisson_summary(result)

    with st.expander("Squad and Team Context", expanded=False):
        # The Phase-4 squad context renderer loads the real data from
        # data/manual/*.csv via the canonical IDs.  It is the single
        # source of truth for squad context — we do NOT also render a
        # duplicate table from result['_squad_context'] (that key is
        # never written, so the duplicate would always read
        # "No squad-strength data available.").
        _render_squad_context(
            result,
            home_canonical_id=home_canonical_id or "",
            away_canonical_id=away_canonical_id or "",
        )

    if match_meta:
        with st.expander("Group Context", expanded=False):
            _render_group_context(
                result,
                stage=match_meta.get("stage", "") or "",
                matchday=match_meta.get("matchday"),
                group=match_meta.get("group", "") or "",
                finished_matches_in_group=match_meta.get(
                    "finished_matches_in_group"
                ),
            )

    with st.expander("Calibration and Data Quality", expanded=False):
        _render_section_table(
            _ux_analysis_calibration(
                result, identity_warnings=identity_warnings
            )
        )
        # Show raw identity warnings (NOT translated) so advanced users
        # can still see canonical codes, history_missing flags, etc.
        if identity_warnings:
            st.markdown("**Raw identity warnings (technical):**")
            for iw in identity_warnings:
                st.markdown(f"- `{iw}`")

    with st.expander("Raw Diagnostics", expanded=False):
        st.json(_ux_analysis_raw(result))


def _render_game_result(
    result: dict,
    min_edge: float,
    *,
    match_meta: dict | None = None,
) -> None:
    """Render the per-game result block (shared by both flows).

    The optional keyword-only ``match_meta`` argument lets the auto-
    populate flow attach source-match metadata (stage, matchday, group,
    finished matches in the group) so the Phase 3 group-context
    warning block can render in the Analysis tab.  When ``match_meta`` is
    None (the manual flow) the dashboard renders exactly as before — no
    group-context block, no behavioural change in the Prediction or
    Betting Value tabs.

    UX-only refactor: the previous stacked body (prob table, edge
    metrics, market baseline, Poisson summary, +EV flags, prediction
    summary, squad context, Pi/Elo expander, input-odds expander) has
    been reorganized into three tabs — Prediction (casual), Betting
    Value (odds-focused), Analysis (preserves all technical depth).
    """
    st.markdown("---")
    st.header(f"{result['home_team']}  vs  {result['away_team']}")
    st.caption(f"Match date: {result['date']}")

    # Identity-resolution line. Shows the canonical 3-letter codes
    # (e.g. ARG vs ALG) so the user can confirm the registry translated
    # correctly. Empty if either side failed to resolve.
    h_can = result.get("canonical_home_id")
    a_can = result.get("canonical_away_id")
    if h_can or a_can:
        st.caption(f"Identity: {h_can or '?'} vs {a_can or '?'}")

    # Identity warnings (one per team that couldn't be resolved or
    # has no corpus history).  Always render above the tabs so the
    # user sees them first.  Pass each one through
    # _translate_and_dedupe_warnings so the casual-facing area above
    # the tabs does not leak the internal canonical / status codes
    # (canonical=CPV, status=history_missing, neutral pi-rating,
    # etc.) and so duplicate raw warnings that translate to the
    # same sentence collapse to a single rendered warning.  The
    # raw version is still available in the Analysis tab under
    # "Calibration and Data Quality" for advanced users.
    identity_warnings = list(result.get("identity_warnings") or [])
    for translated in _translate_and_dedupe_warnings(identity_warnings):
        st.warning(f"🪪 {translated}")

    # Phase 2: the per-game 3-tab layout has been replaced with a flat
    # expander block. The new global Predictions / Bets / Analysis nav
    # (in :func:`main`) means a per-game "view" switcher is no longer
    # needed; everything the user wants to drill into lives in a single
    # expander stack, in priority order:
    #
    #   1. Prediction        (casual, default closed)
    #   2. Betting Value     (odds-focused, default closed)
    #   3. Analysis          (technical, default open)
    #
    # Phases 3-5 will add a global "Analysis" view that wraps this block.
    with st.container():
        with st.expander("🎯 Prediction", expanded=False):
            _render_prediction_tab(result, identity_warnings)
        with st.expander("💰 Betting Value", expanded=False):
            _render_betting_value_tab(
                result, min_edge=min_edge, identity_warnings=identity_warnings
            )
        with st.expander("🔬 Analysis", expanded=True):
            _render_analysis_tab(
                result,
                match_meta=match_meta,
                min_edge=min_edge,
                identity_warnings=identity_warnings,
                home_canonical_id=result.get("canonical_home_id", "") or "",
                away_canonical_id=result.get("canonical_away_id", "") or "",
            )


# --------------------------------------------------------------------------- #
# Auto-populate flow
# --------------------------------------------------------------------------- #
def _render_auto_populate_view(
    corpus: list[dict], name_to_id: dict[str, int], elo_snapshots: dict
) -> None:
    """Date → load games → stack of {matchup, 3 odds inputs, Run button}."""
    st.subheader("📅 Pick a date")
    picked_date = st.date_input(
        "Match date",
        value=_smart_default_date(),
        format="YYYY-MM-DD",
        key="auto_picked_date",
    )
    picked_iso = picked_date.isoformat()

    c1, c2 = st.columns([1, 1])
    load_clicked = c1.button(
        "🔄 Load games for this date",
        use_container_width=True,
        key="auto_load_btn",
    )
    clear_clicked = c2.button(
        "✖ Clear",
        use_container_width=True,
        key="auto_clear_btn",
    )

    if clear_clicked:
        st.session_state.pop("auto_matches", None)
        st.session_state.pop("auto_loaded_date", None)
        st.rerun()

    # Auto-load on first arrival for the default date so the user lands
    # straight on the day's games (no extra click required).
    if (
        "auto_matches" not in st.session_state
        and "auto_loaded_date" not in st.session_state
    ):
        load_clicked = True

    if load_clicked:
        st.session_state["auto_loaded_date"] = picked_iso
        st.session_state["auto_matches"] = _load_unplayed_for_date(picked_iso)

    loaded_date = st.session_state.get("auto_loaded_date")
    matches = st.session_state.get("auto_matches") or []

    if loaded_date is None:
        st.markdown(
            "_Pick a date above and tap **Load games for this date** to "
            "auto-fill the day's matchups._"
        )
        return

    st.markdown(f"**Loaded for {loaded_date}** — {len(matches)} unplayed game(s)")

    if not matches:
        # Plain language: no script path, no recovery instructions.
        # Users get a calm "try another date / use custom matchup" pointer
        # so they aren't sent chasing internal CLI helpers.
        st.info(
            f"No matches on {loaded_date}. "
            "Pick another date above, or use the **Custom matchup** "
            "expander at the bottom of the Predictions view to predict "
            "any game."
        )
        return

    # Compute pi-ratings once for the loaded date and reuse for every game.
    ratings = get_ratings(loaded_date, corpus)

    min_edge_pct = st.slider(
        "Minimum edge for +EV flag (applies to all games below)",
        min_value=0,
        max_value=15,
        value=3,
        step=1,
        key="auto_min_edge",
    )
    min_edge = min_edge_pct / 100.0

    st.caption(
        "Fill in the book odds (American: -230 favorite, +350 underdog) "
        "and tap **Run analysis** on any game. Results render right below it."
    )

    for i, m in enumerate(matches):
        home = m["home_team_name"]
        away = m["away_team_name"]
        kickoff = m.get("kickoff_iso", loaded_date)
        group = m.get("group") or ""
        stage = m.get("stage") or ""
        matchday = m.get("matchday")  # 1/2/3 for group stage, None for knockout

        # ---- per-game card ---- #
        with st.container(border=True):
            hdr = f"**{home}  vs  {away}**"
            meta_bits = [f"🕒 {kickoff}"]
            if group:
                meta_bits.append(group)
            if stage:
                meta_bits.append(stage.replace("_", " ").title())
            st.markdown(hdr)
            st.caption("  ·  ".join(meta_bits))

            o1, o2, o3 = st.columns(3)
            home_odds_txt = o1.text_input(
                "Home", value="", placeholder="-230", key=f"h_odds_{i}",
            )
            draw_odds_txt = o2.text_input(
                "Draw", value="", placeholder="+350", key=f"d_odds_{i}",
            )
            away_odds_txt = o3.text_input(
                "Away", value="", placeholder="+700", key=f"a_odds_{i}",
            )

            run_clicked = st.button(
                "▶ Run analysis",
                key=f"run_{i}",
                use_container_width=True,
            )

            if run_clicked:
                outcome = evaluate_one_game(
                    home_name=home,
                    away_name=away,
                    home_team_id=m.get("home_team_id"),
                    away_team_id=m.get("away_team_id"),
                    cutoff_iso=loaded_date,
                    home_odds_txt=home_odds_txt,
                    draw_odds_txt=draw_odds_txt,
                    away_odds_txt=away_odds_txt,
                    ratings=ratings,
                    min_edge=min_edge,
                    name_to_id=name_to_id,
                    elo_snapshots=elo_snapshots,
                )
                if not outcome["ok"]:
                    st.error(outcome["error"])
                else:
                    # Phase 3 — pass source-match metadata so the
                    # renderer can show the appropriate group-context
                    # warnings.  The finished-matches list comes from
                    # the same on-disk cache the auto-populate flow
                    # already reads.
                    match_meta = {
                        "stage": stage,
                        "matchday": matchday,
                        "group": group,
                        "finished_matches_in_group": (
                            _finished_matches_in_group_from_cache(group)
                            if group
                            else None
                        ),
                    }
                    _render_game_result(
                        outcome["result"],
                        min_edge=min_edge,
                        match_meta=match_meta,
                    )


# --------------------------------------------------------------------------- #
# Manual flow (original form, unchanged in spirit)
# --------------------------------------------------------------------------- #
def _render_manual_view(
    corpus: list[dict], name_to_id: dict[str, int], elo_snapshots: dict
) -> None:
    """The original single-game form. Useful for non-2026 matches."""
    with st.form("match_form", clear_on_submit=False):
        st.subheader("Matchup")
        c1, c2 = st.columns(2)
        home_team = c1.text_input("Home team", value="", placeholder="e.g. Argentina")
        away_team = c2.text_input("Away team", value="", placeholder="e.g. Algeria")

        match_date = st.date_input(
            "Match date",
            value=_smart_default_date(),
            format="YYYY-MM-DD",
        )

        st.subheader("Book odds (American)")
        o1, o2, o3 = st.columns(3)
        home_odds_txt = o1.text_input("Home", value="-230", placeholder="-230")
        draw_odds_txt = o2.text_input("Draw", value="+350", placeholder="+350")
        away_odds_txt = o3.text_input("Away", value="+700", placeholder="+700")

        st.subheader("Filters")
        min_edge_pct = st.slider(
            "Minimum edge for +EV flag",
            min_value=0,
            max_value=15,
            value=3,
            step=1,
            help="Default 3% — markets with pi% − book% above this are flagged.",
        )

        submitted = st.form_submit_button("Run Analysis", use_container_width=True)

    if not submitted:
        st.markdown("---\n")
        st.markdown(
            "Fill in the matchup and book odds, then tap **Run Analysis**.\n\n"
            "**Try the example:** Argentina vs Algeria, 2026-06-16, "
            "odds `-230 / +350 / +700`."
        )
        return

    h_name = (home_team or "").strip()
    a_name = (away_team or "").strip()
    if not h_name or not a_name:
        st.error("Please enter both a home team and an away team.")
        return
    if h_name.lower() == a_name.lower():
        st.error("Home and away teams must be different.")
        return

    h_id = name_to_id.get(h_name)
    a_id = name_to_id.get(a_name)
    if h_id is None:
        st.error(
            f"Team '{h_name}' not found in training data. "
            "Try a different spelling (e.g., 'United States' not 'USA')."
        )
        return
    if a_id is None:
        st.error(
            f"Team '{a_name}' not found in training data. "
            "Try a different spelling (e.g., 'United States' not 'USA')."
        )
        return

    try:
        h_odds = _parse_american_odds(home_odds_txt)
        d_odds = _parse_american_odds(draw_odds_txt)
        a_odds = _parse_american_odds(away_odds_txt)
    except ValueError:
        st.error(
            "Could not parse one of the odds. Use American format: "
            "negative for favorite (-230), positive for underdog (+350)."
        )
        return

    cutoff_iso = match_date.isoformat()
    ratings = get_ratings(cutoff_iso, corpus)

    if not ratings:
        st.error(
            f"No ratings available for cutoff {cutoff_iso}. "
            "Try an earlier date."
        )
        return

    # Look up Elo for both teams (if available) so the model blends Elo
    # with pi-rating (50/50, per scripts/blend_backtest.py).
    home_elo, away_elo = None, None
    if elo_snapshots:
        home_elo, _ = elo_at(elo_snapshots, h_name, cutoff_iso)
        away_elo, _ = elo_at(elo_snapshots, a_name, cutoff_iso)

    result = evaluate_match(
        home_team=h_name,
        away_team=a_name,
        home_team_id=h_id,
        away_team_id=a_id,
        date=cutoff_iso,
        book_home_odds=h_odds,
        book_draw_odds=d_odds,
        book_away_odds=a_odds,
        ratings=ratings,
        min_edge=min_edge_pct / 100.0,
        home_elo=home_elo,
        away_elo=away_elo,
    )
    _render_game_result(result, min_edge=min_edge_pct / 100.0)


# --------------------------------------------------------------------------- #
# Top-level view switcher (Phase 2)
# --------------------------------------------------------------------------- #
# Map between the visible segmented-control label and the short
# deep-linkable slug. The label carries the emoji so the segmented bar
# looks good; the slug is what goes into the URL.
_VIEW_LABEL_TO_SLUG: dict[str, str] = {
    "🎯 Predictions": "predictions",
    "💰 Bets":        "bets",
    "🔬 Analysis":    "analysis",
}
_VIEW_SLUG_TO_LABEL: dict[str, str] = {v: k for k, v in _VIEW_LABEL_TO_SLUG.items()}
_DEFAULT_VIEW_LABEL = "🎯 Predictions"


def _resolve_view_from_query_params() -> str:
    """Map ``st.query_params['view']`` (if any) to a known view label.

    Unknown / missing slugs fall back to :data:`_DEFAULT_VIEW_LABEL`.
    """
    raw = st.query_params.get("view")
    # ``st.query_params.get`` may return a scalar, a list, or None. The
    # segmented-control writer below always emits a scalar string, so we
    # accept the scalar and ignore anything else.
    if isinstance(raw, str):
        return _VIEW_SLUG_TO_LABEL.get(raw, _DEFAULT_VIEW_LABEL)
    return _DEFAULT_VIEW_LABEL


def _render_top_level_nav() -> str:
    """Render the global Predictions / Bets / Analysis switcher.

    Returns the label of the currently selected view. Mutates
    ``st.query_params['view']`` so the URL is deep-linkable.

    Uses :func:`streamlit.segmented_control` (Streamlit ≥ 1.58), which
    renders as a native segmented bar and works well on mobile. The
    selected value is stored under :data:`dashboard.session_state.KEYS.ACTIVE_VIEW`
    so the three section renderers can read it.
    """
    initial = _resolve_view_from_query_params()
    # Honour the previously-selected view if it's still a valid option.
    # This keeps the segmented control sticky across reruns triggered by
    # other widgets (date pickers, sliders, etc.) — without it, every
    # rerun would snap back to the URL-driven value.
    current = st.session_state.get(KEYS.ACTIVE_VIEW, initial)
    if current not in _VIEW_LABEL_TO_SLUG:
        current = initial
    selected = st.segmented_control(
        "View",
        options=list(_VIEW_LABEL_TO_SLUG),
        default=current,
        key=KEYS.ACTIVE_VIEW,
        label_visibility="collapsed",
    ) or current
    # Reflect the choice in the URL so users can share/bookmark it.
    slug = _VIEW_LABEL_TO_SLUG.get(selected, "predictions")
    st.query_params["view"] = slug
    return selected


def _predict_match_cached(
    home_team: str,
    away_team: str,
    home_team_id: int,
    away_team_id: int,
    date_iso: str,
    _ratings_id: int,
    _elo_snapshots_id: int,
    _corpus_id: int,
    goal_predictor=None,
) -> dict:
    """Per-match prediction with a stable cache key (avoids hashing the
    full ``ratings`` dict).

    The original ``get_ratings`` is itself cached, so the heavy lifting
    (pi-rating training) only runs once per cutoff date. This function
    just memoises the per-match ``predict_match`` call so tab switches
    and reruns don't re-evaluate every game.

    The ``_ratings_id`` / ``_elo_snapshots_id`` / ``_corpus_id`` parameters
    are small integer handles produced by the caller.  When the underlying
    data changes, the caller bumps its id and the cache invalidates.

    When ``goal_predictor`` is provided and both teams are recognized by
    the goal model, the H/D/A probabilities from the goal model are
    passed to ``predict_match`` so that ``primary_probs`` reflects the
    Elo60/Goal40 blend.
    """
    # Lazy imports so the module import graph stays flat for callers
    # that only need formatters.
    from dashboard.app import (
        get_elo_snapshots,
        get_ratings,
        load_training_corpus,
    )
    # Re-derive the real objects from the ids. The ids are simply
    # ``id(...)`` of the data structures in :func:`main`'s session, so
    # the lookup is just a small dict walk.
    # In practice, the cache only fires for the lifetime of the
    # Streamlit session and ``id()`` is stable within one process.
    _corpus = _CORPUS_BY_ID.get(_corpus_id) or load_training_corpus()
    _ratings = (
        get_ratings(date_iso, _corpus)
        if not _RATINGS_BY_ID.get((_ratings_id, date_iso))
        else _RATINGS_BY_ID[(_ratings_id, date_iso)]
    )
    _elo = _ELO_BY_ID.get(_elo_snapshots_id) or get_elo_snapshots()
    _CORPUS_BY_ID[_corpus_id] = _corpus
    _RATINGS_BY_ID[(_ratings_id, date_iso)] = _ratings
    _ELO_BY_ID[_elo_snapshots_id] = _elo

    home_elo = away_elo = None
    if _elo:
        home_elo, _ = elo_at(_elo, home_team, date_iso)
        away_elo, _ = elo_at(_elo, away_team, date_iso)

    # Obtain goal model probs when predictor is available
    _goal_probs = None
    _goal_model_xg = None
    _goal_model_low_data = False
    if goal_predictor is not None:
        try:
            _gp = goal_predictor.predict(
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                match_date=date_iso,
            )
            _goal_probs = {
                "home": float(_gp.hda_probs["home"]),
                "draw": float(_gp.hda_probs["draw"]),
                "away": float(_gp.hda_probs["away"]),
            }
            _goal_model_xg = {
                "home_xg": float(_gp.home_xg),
                "away_xg": float(_gp.away_xg),
            }
            _goal_model_low_data = bool(_gp.low_data_flags)
            _goal_model_metadata = {
                "most_likely_score": _gp.most_likely_score,
                "expected_total_goals": float(_gp.expected_total_goals),
                "model_version": _gp.model_version,
                "data_cutoff": _gp.data_cutoff,
                "low_data_flags": _gp.low_data_flags,
            }
        except Exception:
            # Goal model may not have these teams — fall back gracefully
            _goal_probs = None
            _goal_model_xg = None
            _goal_model_low_data = False
            _goal_model_metadata = None

    return predict_match(
        home_team=home_team,
        away_team=away_team,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        date=date_iso,
        ratings=_ratings,
        home_elo=home_elo,
        away_elo=away_elo,
        goal_probs=_goal_probs,
        goal_model_xg=_goal_model_xg,
        goal_model_low_data=_goal_model_low_data,
        _goal_model_expected=goal_predictor is not None,
        goal_model_metadata=_goal_model_metadata,
    )


# Small in-process id -> data registries used by ``_predict_match_cached``.
# Module-level so the same handles are visible to every call within a
# Streamlit session.
_CORPUS_BY_ID: dict[int, list[dict]] = {}
_RATINGS_BY_ID: dict[tuple[int, str], dict] = {}
_ELO_BY_ID: dict[int, dict] = {}


def _render_predictions_view(
    corpus: list[dict], name_to_id: dict[str, int], elo_snapshots: dict, goal_predictor=None
) -> None:
    """Real Phase 3 Predictions renderer (model-only, mobile-first).

    Flow:

    1. Date picker bound to :data:`KEYS.SELECTED_DATE`.
    2. ONE large primary button **"Show Predictions"**.
    3. On click: load the day's unplayed matches, build a single
       pi-ratings snapshot for the date, and render one prediction
       card per game via :func:`dashboard.prediction_card.render_prediction_card`.
    4. Empty-state copy when there are no matches on the chosen date.
    5. A closed-by-default "Custom matchup" expander that lets the
       user compute a single prediction for any (home, away, date)
       triple — NO odds fields, NO min-edge slider.

    Hard constraints (Phase 3 brief):

    * NO odds fields of any kind (no American odds, no no-vig probs,
      no edge, no +EV flag, no ``book_fair``).
    * NO ``min_edge`` slider.
    * NO sportsbook terminology ("edge", "EV", "implied", "no-vig",
      etc.) leaks into the rendered view.
    * NO raw ISO timestamps or raw ``GROUP_X`` codes are rendered.
    * The "Why" control is a :func:`streamlit.popover` styled as a
      mobile CTA bubble (see :mod:`dashboard.styles`).
    """
    # Register the data handles the cache helper uses.
    _corpus_id = id(corpus)
    _elo_id = id(elo_snapshots)
    _CORPUS_BY_ID[_corpus_id] = corpus
    _ELO_BY_ID[_elo_id] = elo_snapshots

    # ---- (1) date picker ---- #
    picked_date = st.date_input(
        "Match date",
        value=_smart_default_date(),
        format="YYYY-MM-DD",
        key=KEYS.SELECTED_DATE,
    )
    picked_iso = picked_date.isoformat() if hasattr(picked_date, "isoformat") else str(picked_date)

    # ---- (2) single primary button ---- #
    show_clicked = st.button(
        "🎯 Show Predictions",
        key="predictions_show_btn",
        type="primary",
        use_container_width=True,
    )

    # ---- determine which (date, matches, predictions) tuple to render ---- #
    # We use a "loaded_date" sentinel in session state so the user can
    # switch dates without losing the previous render — when the picked
    # date changes, the predictions cache is refreshed on the next click.
    loaded_date = _ss_get(KEYS.LOADED_MATCHES + ".date", default=None)
    needs_load = show_clicked or (
        loaded_date != picked_iso
        and _ss_get(KEYS.LOADED_MATCHES) is None
    )

    if needs_load:
        with st.spinner("Loading matches and building predictions…"):
            matches = _load_unplayed_for_date(picked_iso)
            # Build the pi-ratings snapshot once for the whole date. We
            # add a small buffer to the cutoff so the snapshot includes
            # any matches whose UTC date is <= the picked date.
            cutoff_iso = picked_iso + "T23:59:59Z"
            try:
                ratings = get_ratings(cutoff_iso, corpus)
            except Exception:
                # Fallback: cut off at the start of the picked date.
                ratings = get_ratings(picked_iso + "T00:00:00Z", corpus)
            predictions: dict[int, dict] = {}
            # Stable handle for the ratings dict so the per-match cache
            # can invalidate on cutoff change.
            ratings_id = id(ratings)
            for m in matches:
                mid = m.get("match_id")
                if mid is None:
                    continue
                home = m.get("home_team_name") or "Home"
                away = m.get("away_team_name") or "Away"
                home_id = m.get("home_team_id")
                away_id = m.get("away_team_id")
                if home_id is None or away_id is None:
                    # Cache row without an id (manual future match) —
                    # skip silently, the per-game error surface is the
                    # custom matchup expander below.
                    continue
                # Translate the schedule's football-data.org ids into
                # the corpus ids pi-rating expects. Without this, the
                # team_experience lookup misses and the model produces
                # a neutral home-draw-away fallback (PR #9 regression).
                _home_res, _away_res, _id_warnings = _resolve_match_for_prediction(
                    match=m,
                    ratings=ratings,
                    name_to_id=None,  # auto path: rely on the registry
                )
                # Per-match cutoff is the start of the picked date.
                match_cutoff = picked_iso + "T00:00:00Z"
                try:
                    pred = _predict_match_cached(
                        home_team=home,
                        away_team=away,
                        # corpus_id is None only on identity_unresolved
                        # fallback (resolved with status='identity_unresolved'
                        # in extreme edge cases). Use the schedule id
                        # so pi-rating still produces *something*.
                        home_team_id=(
                            int(_home_res.corpus_id)
                            if _home_res.corpus_id is not None
                            else int(home_id)
                        ),
                        away_team_id=(
                            int(_away_res.corpus_id)
                            if _away_res.corpus_id is not None
                            else int(away_id)
                        ),
                        date_iso=match_cutoff,
                        _ratings_id=ratings_id,
                        _elo_snapshots_id=_elo_id,
                        _corpus_id=_corpus_id,
                        goal_predictor=goal_predictor,
                    )
                except Exception as exc:
                    # Don't take down the whole view for one bad row.
                    pred = {
                        "home_team": home,
                        "away_team": away,
                        "home_team_id": int(home_id),
                        "away_team_id": int(away_id),
                        "date": picked_iso,
                        "primary_probs": {"home": 0.4, "draw": 0.3, "away": 0.3},
                        "pi_probs": {"home": 0.4, "draw": 0.3, "away": 0.3},
                        "blend_probs": {"home": 0.4, "draw": 0.3, "away": 0.3},
                        "pi_only_probs": {"home": 0.4, "draw": 0.3, "away": 0.3},
                        "elo_only_probs": None,
                        "blend_was_used": False,
                        "confidence": {
                            "tier": "C",
                            "tier_description": "Limited data",
                            "warnings": [f"prediction error: {exc!s}"],
                        },
                        "banner": "Limited data",
                        "canonical_home_id": (
                            _home_res.canonical_id or ""
                        ),
                        "canonical_away_id": (
                            _away_res.canonical_id or ""
                        ),
                        "identity_warnings": list(_id_warnings),
                    }
                else:
                    # Predict_match() always returns canonical_* keys
                    # (possibly "").  For full path equivalence with
                    # the legacy evaluate_one_game() flow, also
                    # surface identity_warnings when the registry
                    # flagged one or both teams.
                    if _id_warnings:
                        pred["identity_warnings"] = list(_id_warnings)
                    # If predict_match() returned empty canonical IDs
                    # (e.g. because nothing was passed), fill in the
                    # values we resolved here.  Non-empty IDs are left
                    # alone so predict_match's own resolution wins.
                    if not pred.get("canonical_home_id") and _home_res.canonical_id:
                        pred["canonical_home_id"] = _home_res.canonical_id
                    if not pred.get("canonical_away_id") and _away_res.canonical_id:
                        pred["canonical_away_id"] = _away_res.canonical_id
                # Surface the source-match metadata so the card can show
                # the human-readable kickoff / group / stage labels.
                pred["_match_meta"] = {
                    "group": m.get("group", ""),
                    "stage": m.get("stage", ""),
                    "matchday": m.get("matchday"),
                    "kickoff_iso": m.get("kickoff_iso") or picked_iso,
                }
                predictions[int(mid)] = pred
            _ss_set(KEYS.LOADED_MATCHES, matches)
            _ss_set(KEYS.LOADED_MATCHES + ".date", picked_iso)
            _ss_set(KEYS.PREDICTIONS_BY_MATCH, predictions)
            # Also surface the full ratings build in session state so
            # a future "Analysis" view can reuse it without recomputation.
            _ss_set(KEYS.MARKET_BY_MATCH, {})

    matches = _ss_get(KEYS.LOADED_MATCHES, default=[]) or []
    predictions = _ss_get(KEYS.PREDICTIONS_BY_MATCH, default={}) or {}
    loaded_date = _ss_get(KEYS.LOADED_MATCHES + ".date", default=None)

    # ---- (3) heading + empty state ---- #
    st.subheader(f"📅 {picked_iso}")
    if not matches:
        # Distinguish "cache missing" from "no games on this date" so the
        # user gets the right pointer — but NEVER leak the on-disk path
        # or any recovery script reference.
        if not _schedule_cache_present():
            _render_schedule_data_not_loaded()
        else:
            st.markdown(
                f"**No matches on {picked_iso}.**\n\n"
                "_Pick another date above, or use the **Custom matchup** "
                "expander at the bottom to predict any game._"
            )
        _render_custom_matchup_expander(
            corpus=corpus,
            name_to_id=name_to_id,
            elo_snapshots=elo_snapshots,
            goal_predictor=goal_predictor,
        )
        return

    st.caption(
        f"{len(matches)} game{'s' if len(matches) != 1 else ''} on {picked_iso}"
    )

    # ---- (4) render one card per match ---- #
    for m in matches:
        mid = m.get("match_id")
        pred = predictions.get(int(mid)) if mid is not None else None
        if not pred:
            # Stale cache: predictions for this match haven't been
            # computed yet (shouldn't happen because needs_load always
            # recomputes the full set, but be defensive).
            continue
        with st.container(border=False):
            _render_prediction_card(pred, pred.get("_match_meta") or {})

    # ---- (5) custom matchup expander ---- #
    _render_custom_matchup_expander(
        corpus=corpus,
        name_to_id=name_to_id,
        elo_snapshots=elo_snapshots,
        goal_predictor=goal_predictor,
    )


def _render_custom_matchup_expander(
    corpus: list[dict],
    name_to_id: dict[str, int],
    elo_snapshots: dict,
    goal_predictor=None,
) -> None:
    """Custom-matchup expander at the bottom of the Predictions view.

    Lets the user compute a single prediction for any (home, away, date)
    triple. NO odds inputs, NO min-edge slider — this is the casual
    Predictions view's escape hatch for non-2026 fixtures.
    """
    with st.expander("➕ Custom matchup", expanded=False):
        c1, c2 = st.columns(2)
        home_name = c1.text_input(
            "Home team",
            value="",
            placeholder="e.g. Argentina",
            key=KEYS.CUSTOM_HOME,
        )
        away_name = c2.text_input(
            "Away team",
            value="",
            placeholder="e.g. Brazil",
            key=KEYS.CUSTOM_AWAY,
        )
        match_date = st.date_input(
            "Match date",
            value=_smart_default_date(),
            format="YYYY-MM-DD",
            key=KEYS.CUSTOM_DATE,
        )
        run_clicked = st.button(
            "🔮 Predict this matchup",
            key="predictions_custom_run_btn",
            type="primary",
            use_container_width=True,
        )

        if not run_clicked:
            st.caption(
                "_Enter two teams and a date, then tap **Predict this matchup**._"
            )
            return

        h = (home_name or "").strip()
        a = (away_name or "").strip()
        if not h or not a:
            st.error("Please enter both a home team and an away team.")
            return
        if h.lower() == a.lower():
            st.error("Home and away teams must be different.")
            return

        h_id = name_to_id.get(h)
        a_id = name_to_id.get(a)
        if h_id is None:
            st.error(
                f"Team '{h}' not found in training data. "
                "Try a different spelling (e.g. 'United States' not 'USA')."
            )
            return
        if a_id is None:
            st.error(
                f"Team '{a}' not found in training data. "
                "Try a different spelling (e.g. 'United States' not 'USA')."
            )
            return

        cutoff_iso = match_date.isoformat() + "T00:00:00Z"
        try:
            ratings = get_ratings(cutoff_iso, corpus)
        except Exception:
            ratings = {}
        if not ratings:
            st.error(
                f"No ratings available for cutoff {cutoff_iso}. "
                "Try an earlier date."
            )
            return

        home_elo = away_elo = None
        if elo_snapshots:
            home_elo, _ = elo_at(elo_snapshots, h, cutoff_iso)
            away_elo, _ = elo_at(elo_snapshots, a, cutoff_iso)

        try:
            _goal_probs = None
            _goal_model_xg = None
            _goal_model_low_data = False
            if goal_predictor is not None:
                try:
                    _gp = goal_predictor.predict(
                        home_team_id=int(h_id),
                        away_team_id=int(a_id),
                        match_date=match_date.isoformat(),
                    )
                    _goal_probs = {
                        "home": float(_gp.hda_probs["home"]),
                        "draw": float(_gp.hda_probs["draw"]),
                        "away": float(_gp.hda_probs["away"]),
                    }
                    _goal_model_xg = {
                        "home_xg": float(_gp.home_xg),
                        "away_xg": float(_gp.away_xg),
                    }
                    _goal_model_low_data = bool(_gp.low_data_flags)
                    _goal_model_metadata = {
                        "most_likely_score": _gp.most_likely_score,
                        "expected_total_goals": float(_gp.expected_total_goals),
                        "model_version": _gp.model_version,
                        "data_cutoff": _gp.data_cutoff,
                        "low_data_flags": _gp.low_data_flags,
                    }
                except Exception:
                    _goal_probs = None
                    _goal_model_xg = None
                    _goal_model_low_data = False
                    _goal_model_metadata = None

            prediction = predict_match(
                home_team=h,
                away_team=a,
                home_team_id=int(h_id),
                away_team_id=int(a_id),
                date=match_date.isoformat(),
                ratings=ratings,
                home_elo=home_elo,
                away_elo=away_elo,
                goal_probs=_goal_probs,
                goal_model_xg=_goal_model_xg,
                goal_model_low_data=_goal_model_low_data,
                _goal_model_expected=goal_predictor is not None,
                goal_model_metadata=_goal_model_metadata,
            )
        except Exception:
            # Calm, plain-language error — no raw exception text / stack
            # trace leaks to the user.  They get a clear next step
            # (try again, or use a different matchup) without technical
            # internals.  The full traceback stays in the server logs.
            st.error(
                "We couldn't compute a prediction for this game right now. "
                "Try again, or use a different matchup."
            )
            return

        meta = {
            "group": "",
            "stage": "",
            "matchday": None,
            "kickoff_iso": cutoff_iso,
        }
        _render_prediction_card(prediction, meta)


def _render_bets_view(
    corpus: list[dict], name_to_id: dict[str, int], elo_snapshots: dict, goal_predictor=None
) -> None:
    """Real Phase 4 Bets renderer (mobile-first, odds-gated).

    Flow:

      1. Date picker bound to :data:`KEYS.SELECTED_DATE`.
      2. ONE large primary button **"💰 Show Bets"**.
      3. On click: load the day's unplayed matches, build a single
         pi-ratings snapshot for the date, and render one Bets card
         per game via :func:`dashboard.bet_card.render_bet_card`.
      4. Empty-state copy when there are no matches on the chosen date.
      5. A closed-by-default **"Advanced settings"** expander with the
         ``min_edge`` slider (out of casual sight by default).
      6. A closed-by-default **"Custom bet"** expander at the bottom
         for non-2026 fixtures — user types (home, away, date,
         home/draw/away odds) and gets one Bets card with the result.

    Hard constraints (Phase 4 brief):

    * Odds are ONLY in Bets. Predictions view does NOT see odds.
    * ``min_edge`` lives in an **Advanced settings** expander
      (closed by default).
    * One game's invalid odds must NOT disrupt other games — the
      error stays in the offending card.
    * "No Clear Value" must be visually distinct from a real
      best-value pick.
    * "Most Likely Result" and "Best Value Play" are visually
      distinct (different style + size + icon).
    * Draw as best value uses **"Match to End in a Draw"** wording.
    * Empty placeholder examples MUST NOT look like real entered
      data — placeholders only, no default values.

    The Bets view reuses the cached predictions from the Predictions
    view (stored under :data:`KEYS.PREDICTIONS_BY_MATCH`) when a
    shared SELECTED_DATE was loaded by the Predictions view. When the
    user lands directly on Bets, the view re-derives predictions for
    the picked date on click.
    """
    # Register the data handles the cache helper uses (mirrors the
    # Predictions view so the same ``_predict_match_cached`` is used).
    _corpus_id = id(corpus)
    _elo_id = id(elo_snapshots)
    _CORPUS_BY_ID[_corpus_id] = corpus
    _ELO_BY_ID[_elo_id] = elo_snapshots

    # ---- (1) date picker ---- #
    picked_date = st.date_input(
        "Match date",
        value=_smart_default_date(),
        format="YYYY-MM-DD",
        key=KEYS.SELECTED_DATE,
    )
    picked_iso = (
        picked_date.isoformat()
        if hasattr(picked_date, "isoformat")
        else str(picked_date)
    )

    # ---- (2) single primary button ---- #
    show_clicked = st.button(
        "💰 Show Bets",
        key="bets_show_btn",
        type="primary",
        use_container_width=True,
    )

    # ---- (3) advanced settings expander (closed by default) ---- #
    with st.expander("⚙️ Advanced settings", expanded=False):
        st.caption(
            "These settings apply to all games on the date above. "
            "Most users don't need to change them."
        )
        # Slider is in points (0..15) so it reads as an integer, but
        # we store the fractional value in session state under
        # ``KEYS.BETS_MIN_EDGE`` (e.g. 0.03) so the
        # ``evaluate_market(..., min_edge=...)`` call gets a clean
        # 0..1 number.
        min_edge_pct = st.slider(
            "Minimum edge for value plays (%)",
            min_value=0,
            max_value=15,
            value=3,
            step=1,
            key="bets_min_edge_pct",
            help=(
                "Markets where the model's probability exceeds the "
                "no-vig book probability by at least this much are "
                "flagged as value plays. Default 3%."
            ),
        )
        _ss_set(KEYS.BETS_MIN_EDGE, min_edge_pct / 100.0)

    # ---- (4) determine which (date, matches, predictions) to render ---- #
    # We mirror the Predictions view's caching strategy so tab
    # switches don't blow away the user's work: keyed on the picked
    # date, refreshed when the user explicitly clicks the primary
    # button or when the date changes and no cache exists.
    loaded_date = _ss_get(KEYS.LOADED_MATCHES + ".date", default=None)
    needs_load = show_clicked or (
        loaded_date != picked_iso
        and _ss_get(KEYS.LOADED_MATCHES) is None
    )

    if needs_load:
        with st.spinner("Loading matches and building predictions…"):
            matches = _load_unplayed_for_date(picked_iso)
            cutoff_iso = picked_iso + "T23:59:59Z"
            try:
                ratings = get_ratings(cutoff_iso, corpus)
            except Exception:
                ratings = get_ratings(picked_iso + "T00:00:00Z", corpus)
            predictions: dict[int, dict] = {}
            ratings_id = id(ratings)
            for m in matches:
                mid = m.get("match_id")
                if mid is None:
                    continue
                home = m.get("home_team_name") or "Home"
                away = m.get("away_team_name") or "Away"
                home_id = m.get("home_team_id")
                away_id = m.get("away_team_id")
                if home_id is None or away_id is None:
                    continue
                # Translate the schedule's football-data.org ids into
                # the corpus ids pi-rating expects. Without this, the
                # team_experience lookup misses and the model produces
                # a neutral home-draw-away fallback (PR #9 regression).
                _home_res, _away_res, _id_warnings = _resolve_match_for_prediction(
                    match=m,
                    ratings=ratings,
                    name_to_id=None,  # auto path: rely on the registry
                )
                match_cutoff = picked_iso + "T00:00:00Z"
                try:
                    pred = _predict_match_cached(
                        home_team=home,
                        away_team=away,
                        home_team_id=(
                            int(_home_res.corpus_id)
                            if _home_res.corpus_id is not None
                            else int(home_id)
                        ),
                        away_team_id=(
                            int(_away_res.corpus_id)
                            if _away_res.corpus_id is not None
                            else int(away_id)
                        ),
                        date_iso=match_cutoff,
                        _ratings_id=ratings_id,
                        _elo_snapshots_id=_elo_id,
                        _corpus_id=_corpus_id,
                        goal_predictor=goal_predictor,
                    )
                except Exception as exc:
                    # Don't take down the whole view for one bad row.
                    pred = {
                        "home_team": home,
                        "away_team": away,
                        "home_team_id": int(home_id),
                        "away_team_id": int(away_id),
                        "date": picked_iso,
                        "primary_probs": {"home": 0.4, "draw": 0.3, "away": 0.3},
                        "pi_probs": {"home": 0.4, "draw": 0.3, "away": 0.3},
                        "blend_probs": {"home": 0.4, "draw": 0.3, "away": 0.3},
                        "pi_only_probs": {"home": 0.4, "draw": 0.3, "away": 0.3},
                        "elo_only_probs": None,
                        "blend_was_used": False,
                        "confidence": {
                            "tier": "C",
                            "tier_description": "Limited data",
                            "warnings": [f"prediction error: {exc!s}"],
                        },
                        "banner": "Limited data",
                        "canonical_home_id": (
                            _home_res.canonical_id or ""
                        ),
                        "canonical_away_id": (
                            _away_res.canonical_id or ""
                        ),
                        "identity_warnings": list(_id_warnings),
                    }
                else:
                    # Surface identity warnings + fill in canonical IDs
                    # if predict_match() returned empty values.  Mirrors
                    # the legacy evaluate_one_game() contract.
                    if _id_warnings:
                        pred["identity_warnings"] = list(_id_warnings)
                    if not pred.get("canonical_home_id") and _home_res.canonical_id:
                        pred["canonical_home_id"] = _home_res.canonical_id
                    if not pred.get("canonical_away_id") and _away_res.canonical_id:
                        pred["canonical_away_id"] = _away_res.canonical_id
                pred["_match_meta"] = {
                    "group": m.get("group", ""),
                    "stage": m.get("stage", ""),
                    "matchday": m.get("matchday"),
                    "kickoff_iso": m.get("kickoff_iso") or picked_iso,
                }
                predictions[int(mid)] = pred
            _ss_set(KEYS.LOADED_MATCHES, matches)
            _ss_set(KEYS.LOADED_MATCHES + ".date", picked_iso)
            _ss_set(KEYS.PREDICTIONS_BY_MATCH, predictions)
            # Reset the per-match market cache — the user may have
            # been on a different date last time.
            _ss_set(KEYS.MARKET_BY_MATCH, {})

    matches = _ss_get(KEYS.LOADED_MATCHES, default=[]) or []
    predictions = _ss_get(KEYS.PREDICTIONS_BY_MATCH, default={}) or {}

    # ---- (5) heading + empty state ---- #
    st.subheader(f"📅 {picked_iso}")
    if not matches:
        # Distinguish "cache missing" from "no games on this date" so the
        # user gets the right pointer — but NEVER leak the on-disk path
        # or any recovery script reference.
        if not _schedule_cache_present():
            _render_schedule_data_not_loaded()
        else:
            st.markdown(
                f"**No matches on {picked_iso}.**\n\n"
                "_Pick another date above, or use the **Custom bet** expander "
                "at the bottom to evaluate any game._"
            )
        _render_custom_bet_expander(
            corpus=corpus,
            name_to_id=name_to_id,
            elo_snapshots=elo_snapshots,
        )
        return

    st.caption(
        f"{len(matches)} game{'s' if len(matches) != 1 else ''} on {picked_iso} — "
        "enter the sportsbook odds for any game, then tap **Check Betting Value**."
    )

    # ---- (6) render one Bets card per match ---- #
    for m in matches:
        mid = m.get("match_id")
        pred = predictions.get(int(mid)) if mid is not None else None
        if not pred:
            continue
        # mid is guaranteed non-None when pred is truthy (predictions
        # are keyed by int(mid) in the dict above). The cast through
        # str() keeps Pyright happy without changing runtime behaviour.
        with st.container(border=True):
            _render_bet_card(
                match_meta=pred.get("_match_meta") or {},
                prediction=pred,
                key_prefix=f"bets_{int(mid) if mid is not None else 0}",
            )

    # ---- (7) custom bet expander (bottom of the page) ---- #
    _render_custom_bet_expander(
        corpus=corpus,
        name_to_id=name_to_id,
        elo_snapshots=elo_snapshots,
    )


def _render_custom_bet_expander(
    corpus: list[dict],
    name_to_id: dict[str, int],
    elo_snapshots: dict,
) -> None:
    """Custom-bet expander at the bottom of the 💰 Bets view.

    The user types a (home, away, date) triple plus American odds
    for the three markets, taps the primary button, and gets one
    Bets card with the result. Useful for non-2026 fixtures or for
    testing the value engine against an old line.

    Errors stay local — a bad team name / bad odds does not leak
    outside the expander.
    """
    with st.expander("➕ Custom bet", expanded=False):
        c1, c2 = st.columns(2)
        home_name = c1.text_input(
            "Home team",
            value="",
            placeholder="e.g. Argentina",
            key=KEYS.CUSTOM_HOME,
        )
        away_name = c2.text_input(
            "Away team",
            value="",
            placeholder="e.g. Brazil",
            key=KEYS.CUSTOM_AWAY,
        )
        match_date = st.date_input(
            "Match date",
            value=_smart_default_date(),
            format="YYYY-MM-DD",
            key=KEYS.CUSTOM_DATE,
        )
        st.markdown("**Sportsbook odds (American format)**")
        o1, o2, o3 = st.columns(3)
        home_odds_txt = o1.text_input(
            "Home odds",
            value="",
            placeholder="e.g. -230",
            key=KEYS.CUSTOM_HOME_ODDS,
        )
        draw_odds_txt = o2.text_input(
            "Draw odds",
            value="",
            placeholder="e.g. +350",
            key=KEYS.CUSTOM_DRAW_ODDS,
        )
        away_odds_txt = o3.text_input(
            "Away odds",
            value="",
            placeholder="e.g. +550",
            key=KEYS.CUSTOM_AWAY_ODDS,
        )

        run_clicked = st.button(
            "💰 Check Betting Value",
            key="bets_custom_run_btn",
            type="primary",
            use_container_width=True,
        )

        if not run_clicked:
            st.caption(
                "_Enter two teams, a date, and the three book odds, then tap "
                "**Check Betting Value**._"
            )
            return

        h = (home_name or "").strip()
        a = (away_name or "").strip()
        if not h or not a:
            st.error("Please enter both a home team and an away team.")
            return
        if h.lower() == a.lower():
            st.error("Home and away teams must be different.")
            return

        h_id = name_to_id.get(h)
        a_id = name_to_id.get(a)
        if h_id is None:
            st.error(
                f"Team '{h}' not found in training data. "
                "Try a different spelling (e.g., 'United States' not 'USA')."
            )
            return
        if a_id is None:
            st.error(
                f"Team '{a}' not found in training data. "
                "Try a different spelling (e.g., 'United States' not 'USA')."
            )
            return

        # Validate odds locally — same rules as the per-card inputs.
        from dashboard.bet_card import _validate_odds_text

        values, err = _validate_odds_text(
            home_odds_txt, draw_odds_txt, away_odds_txt
        )
        if err is not None:
            st.error(err)
            return
        h_f, d_f, a_f = values  # type: ignore[misc]

        cutoff_iso = match_date.isoformat() + "T00:00:00Z"
        try:
            ratings = get_ratings(cutoff_iso, corpus)
        except Exception:
            ratings = {}
        if not ratings:
            st.error(
                f"No ratings available for cutoff {cutoff_iso}. "
                "Try an earlier date."
            )
            return

        home_elo = away_elo = None
        if elo_snapshots:
            home_elo, _ = elo_at(elo_snapshots, h, cutoff_iso)
            away_elo, _ = elo_at(elo_snapshots, a, cutoff_iso)

        try:
            _goal_probs = None
            _goal_model_xg = None
            _goal_model_low_data = False
            if goal_predictor is not None:
                try:
                    _gp = goal_predictor.predict(
                        home_team_id=int(h_id),
                        away_team_id=int(a_id),
                        match_date=match_date.isoformat(),
                    )
                    _goal_probs = {
                        "home": float(_gp.hda_probs["home"]),
                        "draw": float(_gp.hda_probs["draw"]),
                        "away": float(_gp.hda_probs["away"]),
                    }
                    _goal_model_xg = {
                        "home_xg": float(_gp.home_xg),
                        "away_xg": float(_gp.away_xg),
                    }
                    _goal_model_low_data = bool(_gp.low_data_flags)
                    _goal_model_metadata = {
                        "most_likely_score": _gp.most_likely_score,
                        "expected_total_goals": float(_gp.expected_total_goals),
                        "model_version": _gp.model_version,
                        "data_cutoff": _gp.data_cutoff,
                        "low_data_flags": _gp.low_data_flags,
                    }
                except Exception:
                    _goal_probs = None
                    _goal_model_xg = None
                    _goal_model_low_data = False
                    _goal_model_metadata = None

            prediction = predict_match(
                home_team=h,
                away_team=a,
                home_team_id=int(h_id),
                away_team_id=int(a_id),
                date=match_date.isoformat(),
                ratings=ratings,
                home_elo=home_elo,
                away_elo=away_elo,
                goal_probs=_goal_probs,
                goal_model_xg=_goal_model_xg,
                goal_model_low_data=_goal_model_low_data,
                _goal_model_expected=goal_predictor is not None,
                goal_model_metadata=_goal_model_metadata,
            )
        except Exception:
            # Calm, plain-language error.  No raw exception text /
            # stack trace.  The full traceback stays in the server logs.
            st.error(
                "We couldn't compute a prediction for this game right now. "
                "Try again, or use a different matchup."
            )
            return

        # Build a synthetic market so the card has all three odds on
        # hand. We seed the text-input keys with the user's values
        # so a re-render of the card (rare) keeps them.
        st.session_state["custom_bet_card_home_odds"] = home_odds_txt
        st.session_state["custom_bet_card_draw_odds"] = draw_odds_txt
        st.session_state["custom_bet_card_away_odds"] = away_odds_txt

        # Pre-stash the result so the card renders it on the first
        # pass without requiring a click. This keeps the custom-bet
        # experience "one tap" instead of "two taps".
        try:
            from soccer_ev_model.ev_workflow import evaluate_market

            market = evaluate_market(
                prediction,
                book_home_odds=h_f,
                book_draw_odds=d_f,
                book_away_odds=a_f,
                min_edge=float(
                    _ss_get(KEYS.BETS_MIN_EDGE, 0.03) or 0.03
                ),
            )
            combined = {**prediction, **market}
            from dashboard.ux_presenters import value_play

            best = value_play(
                combined, min_edge=float(
                    _ss_get(KEYS.BETS_MIN_EDGE, 0.03) or 0.03
                )
            )
        except ValueError as exc:
            st.error(str(exc) or "We couldn't evaluate the market for these odds.")
            return
        except Exception:
            # Calm, plain-language error.  No raw exception text / stack
            # trace leaks to the user.
            st.error(
                "We couldn't evaluate the market for these odds. "
                "Double-check the prices and try again."
            )
            return

        # Render the Most Likely Result + Best Value blocks inline so
        # the user doesn't need to scroll back to the card (the card
        # would re-render empty because of the text-input state we
        # set above). The same text blocks the per-game card uses.
        from dashboard.bet_card import (
            _render_best_value,
            _render_no_clear_value,
        )
        from dashboard.prediction_card import (
            _extract_most_likely,
            _format_probability,
            _outcome_headline_text,
        )
        from dashboard.text_format import format_team_matchup

        st.markdown(
            f"### {format_team_matchup(prediction.get('home_team', h), prediction.get('away_team', a))}"
        )
        mlr_key = _extract_most_likely(prediction)
        mlr_text = _outcome_headline_text(mlr_key, prediction)
        _probs = prediction.get("primary_probs") or prediction.get("blend_probs") or prediction.get("pi_probs") or {}
        p_top = _probs.get(mlr_key)
        st.markdown("**Most Likely Result**")
        headline_html = (
            mlr_text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        st.markdown(
            f"<div style='font-size:1.3em; font-weight:600;'>"
            f"{headline_html}</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"{_format_probability(p_top)} model probability")

        if best.get("status") == "play":
            market_label = {
                "home": prediction.get("home_team", "Home"),
                "draw": "Match to End in a Draw",
                "away": prediction.get("away_team", "Away"),
            }.get(best.get("market", ""), "TBD")
            _render_best_value(market_label, best.get("odds"))
            edge = float(best.get("edge", 0.0))
            st.markdown(
                f"<div style='font-size:1em; color:#495057;'>"
                f"Edge: <strong>{edge * 100:+.1f}%</strong>"
                f"</div>",
                unsafe_allow_html=True,
            )
            try:
                from dashboard.ux_presenters import (
                    value_confidence_label,
                    value_why_text,
                )
                st.caption(
                    f"Value confidence: {value_confidence_label(best, combined)}"
                )
            except Exception:
                pass
            if best.get("market") != mlr_key:
                st.info(
                    f"ℹ️ The model expects {mlr_text}, but the best value is "
                    f"on {market_label}."
                )
            try:
                from dashboard.ux_presenters import value_why_text
                why_text = value_why_text(best, combined)
            except Exception:
                why_text = "The sportsbook price implies a different probability than the model."
            with st.popover("❓ Why is this value?", use_container_width=False):
                st.markdown(why_text)
        else:
            _render_no_clear_value()
            st.caption(
                "Value confidence: Low (no outcome cleared the edge threshold)"
            )
            with st.popover("❓ Why no value?", use_container_width=False):
                st.markdown(
                    "No outcome offers enough value at the entered odds. "
                    "Try a different line, or lower the minimum-edge slider "
                    "in Advanced settings."
                )


def _render_analysis_view(
    corpus: list[dict], name_to_id: dict[str, int], elo_snapshots: dict, goal_predictor=None
) -> None:
    """Real Phase 5 Analysis renderer (mobile-first, technical, model-only).

    Flow:

      1. Date picker bound to :data:`KEYS.SELECTED_DATE` (shared with
         Predictions and Bets).  Uses the *same* matcher load as those
         views so the Analysis page always shows the same games the
         user sees in the other tabs.
      2. A single primary button **"🔬 Show Analysis"** that loads
         matches + predictions for the picked date and stores them in
         :data:`KEYS.LOADED_MATCHES` / :data:`KEYS.PREDICTIONS_BY_MATCH`.
      3. A compact matchup selectbox bound to
         :data:`KEYS.ANALYSIS_GAME` so the pick persists across tab
         changes and reruns.
      4. Eleven collapsible sections, one per :mod:`dashboard.analysis_view`
         helper.  **Prediction Details** opens by default; the rest
         are collapsed.  **Market Comparison** is gated on whether the
         user has run a betting-value evaluation through the Bets
         view; if not, it shows a calm message and the other sections
         remain fully usable (the Analysis view works on model-only
         predictions).

    Hard constraints (Phase 5 brief):

      * All sections default to **collapsed** (``expanded=False``)
        **except** Prediction Details (``expanded=True``).
      * Market Comparison shows a calm message when no market data
        exists — it does **not** block any other section.
      * Selected game persists across tab changes via
        :data:`KEYS.ANALYSIS_GAME`.
      * The view works with model-only predictions (no odds required).
      * Raw Diagnostics surfaces canonical team IDs.
      * Calibration section surfaces the tier letter (A / B / C / D).
    """
    # Register the data handles the per-match prediction cache uses
    # (mirrors Predictions / Bets).
    _corpus_id = id(corpus)
    _elo_id = id(elo_snapshots)
    _CORPUS_BY_ID[_corpus_id] = corpus
    _ELO_BY_ID[_elo_id] = elo_snapshots

    # ---- (1) date picker ---- #
    picked_date = st.date_input(
        "Match date",
        value=_smart_default_date(),
        format="YYYY-MM-DD",
        key=KEYS.SELECTED_DATE,
    )
    picked_iso = (
        picked_date.isoformat()
        if hasattr(picked_date, "isoformat")
        else str(picked_date)
    )

    # ---- (2) single primary button (loads matches + predictions) ---- #
    show_clicked = st.button(
        "🔬 Show Analysis",
        key="analysis_show_btn",
        type="primary",
        use_container_width=True,
    )

    # ---- (3) match-prediction cache, refreshed on date / button ---- #
    loaded_date = _ss_get(KEYS.LOADED_MATCHES + ".date", default=None)
    needs_load = show_clicked or (
        loaded_date != picked_iso
        and _ss_get(KEYS.LOADED_MATCHES) is None
    )

    if needs_load:
        with st.spinner("Loading matches and building predictions…"):
            matches = _load_unplayed_for_date(picked_iso)
            cutoff_iso = picked_iso + "T23:59:59Z"
            try:
                ratings = get_ratings(cutoff_iso, corpus)
            except Exception:
                ratings = get_ratings(picked_iso + "T00:00:00Z", corpus)
            predictions: dict[int, dict] = {}
            ratings_id = id(ratings)
            for m in matches:
                mid = m.get("match_id")
                if mid is None:
                    continue
                home = m.get("home_team_name") or "Home"
                away = m.get("away_team_name") or "Away"
                home_id = m.get("home_team_id")
                away_id = m.get("away_team_id")
                if home_id is None or away_id is None:
                    continue
                # Translate the schedule's football-data.org ids into
                # the corpus ids pi-rating expects. Without this, the
                # team_experience lookup misses and the model produces
                # a neutral home-draw-away fallback (PR #9 regression).
                _home_res, _away_res, _id_warnings = _resolve_match_for_prediction(
                    match=m,
                    ratings=ratings,
                    name_to_id=None,  # auto path: rely on the registry
                )
                match_cutoff = picked_iso + "T00:00:00Z"
                try:
                    pred = _predict_match_cached(
                        home_team=home,
                        away_team=away,
                        home_team_id=(
                            int(_home_res.corpus_id)
                            if _home_res.corpus_id is not None
                            else int(home_id)
                        ),
                        away_team_id=(
                            int(_away_res.corpus_id)
                            if _away_res.corpus_id is not None
                            else int(away_id)
                        ),
                        date_iso=match_cutoff,
                        _ratings_id=ratings_id,
                        _elo_snapshots_id=_elo_id,
                        _corpus_id=_corpus_id,
                        goal_predictor=goal_predictor,
                    )
                except Exception as exc:
                    # Don't take down the whole view for one bad row.
                    pred = {
                        "home_team": home,
                        "away_team": away,
                        "home_team_id": int(home_id),
                        "away_team_id": int(away_id),
                        "date": picked_iso,
                        "primary_probs": {"home": 0.4, "draw": 0.3, "away": 0.3},
                        "pi_probs": {"home": 0.4, "draw": 0.3, "away": 0.3},
                        "blend_probs": {"home": 0.4, "draw": 0.3, "away": 0.3},
                        "pi_only_probs": {"home": 0.4, "draw": 0.3, "away": 0.3},
                        "elo_only_probs": None,
                        "blend_was_used": False,
                        "confidence": {
                            "tier": "C",
                            "tier_description": "Limited data",
                            "warnings": [f"prediction error: {exc!s}"],
                        },
                        "banner": "Limited data",
                        "canonical_home_id": (
                            _home_res.canonical_id or ""
                        ),
                        "canonical_away_id": (
                            _away_res.canonical_id or ""
                        ),
                        "identity_warnings": list(_id_warnings),
                    }
                else:
                    # Surface identity warnings + fill in canonical IDs
                    # if predict_match() returned empty values.  Mirrors
                    # the legacy evaluate_one_game() contract.
                    if _id_warnings:
                        pred["identity_warnings"] = list(_id_warnings)
                    if not pred.get("canonical_home_id") and _home_res.canonical_id:
                        pred["canonical_home_id"] = _home_res.canonical_id
                    if not pred.get("canonical_away_id") and _away_res.canonical_id:
                        pred["canonical_away_id"] = _away_res.canonical_id
                pred["_match_meta"] = {
                    "group": m.get("group", ""),
                    "stage": m.get("stage", ""),
                    "matchday": m.get("matchday"),
                    "kickoff_iso": m.get("kickoff_iso") or picked_iso,
                }
                predictions[int(mid)] = pred
            _ss_set(KEYS.LOADED_MATCHES, matches)
            _ss_set(KEYS.LOADED_MATCHES + ".date", picked_iso)
            _ss_set(KEYS.PREDICTIONS_BY_MATCH, predictions)
            # The Analysis view never generates new market data; it
            # only reads whatever the Bets view already cached.  We
            # don't blow away the existing market cache here — if the
            # user has run an evaluation through Bets, those numbers
            # stay visible in the Market Comparison section.

    matches = _ss_get(KEYS.LOADED_MATCHES, default=[]) or []
    predictions = _ss_get(KEYS.PREDICTIONS_BY_MATCH, default={}) or {}
    market_by_match = _ss_get(KEYS.MARKET_BY_MATCH, default={}) or {}

    # ---- (4) heading + empty state ---- #
    st.subheader(f"📅 {picked_iso}")
    if not matches:
        # Distinguish "cache missing" from "no games on this date" so the
        # user gets the right pointer — but NEVER leak the on-disk path
        # or any recovery script reference.
        if not _schedule_cache_present():
            _render_schedule_data_not_loaded()
        else:
            st.markdown(
                f"**No matches on {picked_iso}.**\n\n"
                "_Pick another date above, or use the **Custom matchup** "
                "expander in **Predictions** to predict any game._"
            )
        return

    st.caption(
        f"{len(matches)} game{'s' if len(matches) != 1 else ''} on {picked_iso} "
        f"— pick a game to inspect its full technical breakdown."
    )

    # ---- (5) hand off to the analysis view renderer ---- #
    from dashboard.analysis_view import render_analysis_view as _render_av
    _render_av(
        matches_for_date=matches,
        predictions_by_match=predictions,
        market_by_match=market_by_match,
        name_to_id=name_to_id,
    )


def _render_legacy_autopopulate(
    corpus: list[dict], name_to_id: dict[str, int], elo_snapshots: dict
) -> None:
    """Phase 2 stub: original Auto-populate body, called from Predictions.

    Phases 3 will absorb this into a real Predictions renderer; for
    now it preserves end-to-end behavior.
    """
    _render_auto_populate_view(corpus, name_to_id, elo_snapshots)


def _render_legacy_manual(
    corpus: list[dict], name_to_id: dict[str, int], elo_snapshots: dict
) -> None:
    """Phase 2 stub: original Manual body, called from Bets.

    Phase 4 will absorb this into a real Bets renderer; for now it
    preserves end-to-end behavior.
    """
    _render_manual_view(corpus, name_to_id, elo_snapshots)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    _inject_css()

    st.title("⚽ WC Match Center")
    st.caption("Predictions · Betting Value · Model Analysis")

    corpus = load_training_corpus()
    name_to_id = build_name_to_id(corpus)
    elo_snapshots = get_elo_snapshots()
    goal_predictor, _goal_model_err = _get_goal_predictor()

    selected_view = _render_top_level_nav()

    # ---- Context cards (sit between the top-level nav and the per-view
    # dispatcher; visible BEFORE the user clicks Show Predictions). ---- #
    # Phase 9: autoload matches + predictions on page open so the cards
    # populate without requiring the user to click "Show Predictions".
    # The autoload helper is cheap on every rerun — it short-circuits
    # via KEYS.CONTEXT_AUTOLOAD_DATE when the same date is already
    # cached.  When the user changes the per-view date picker the
    # sentinel mismatch forces a fresh load (the helper registers
    # corpus / elo handles in the same registries the per-view
    # renderers use, so subsequent Show-Predictions clicks reuse the
    # pi-rating snapshot).
    _target_date = _ss_get(KEYS.SELECTED_DATE)
    if hasattr(_target_date, "isoformat") and not isinstance(_target_date, str):
        try:
            _target_date = _target_date.isoformat()
        except Exception:
            _target_date = None
    if not isinstance(_target_date, str) or not _target_date:
        _target_date = _smart_default_date().isoformat()
    _autoload_context_for_date(_target_date, corpus, elo_snapshots, goal_predictor=goal_predictor)

    _loaded_matches_for_cards = _ss_get(KEYS.LOADED_MATCHES, default=[]) or []
    _predictions_for_cards = _ss_get(KEYS.PREDICTIONS_BY_MATCH, default={}) or {}
    _render_tournament_snapshot(
        _build_tournament_snapshot(_loaded_matches_for_cards),
    )
    _render_highest_confidence(
        _highest_model_confidence(
            _loaded_matches_for_cards, _predictions_for_cards,
        ),
    )

    # Route to the active section. Each section is a thin wrapper for
    # now (Phase 2); Phases 3-5 will replace them with proper renderers.
    if selected_view == "🎯 Predictions":
        _render_predictions_view(corpus, name_to_id, elo_snapshots, goal_predictor=goal_predictor)
    elif selected_view == "💰 Bets":
        _render_bets_view(corpus, name_to_id, elo_snapshots, goal_predictor=goal_predictor)
    elif selected_view == "🔬 Analysis":
        _render_analysis_view(corpus, name_to_id, elo_snapshots, goal_predictor=goal_predictor)
    else:
        # Defensive default: unknown label -> Predictions. Should not be
        # reachable because the segmented control emits known labels.
        _render_predictions_view(corpus, name_to_id, elo_snapshots, goal_predictor=goal_predictor)


if __name__ == "__main__":
    main()
