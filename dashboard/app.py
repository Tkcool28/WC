"""
Streamlit dashboard for the +EV soccer workflow (pi-rating + Elo blend).

Two flows:
  1. **Auto-populate** (default) — pick a date, click "Load games", see
     the day's matchups stacked with three odds inputs per game and a
     "Run analysis" button per game. Reads `data/raw/matches_2026.json`
     (cached, read-only).
  2. **Manual** — the original form: type the team names + book odds,
     click "Run Analysis". Useful for friendly / non-2026 matches.

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

from soccer_ev_model.ev_workflow import evaluate_match  # noqa: E402
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
    translate_warning as _translate_warning,
    value_confidence_label as _value_confidence_label,
    value_play as _value_play,
    value_why_text as _value_why_text,
)


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

# Anchor "today" for the auto-populate flow. 2026-06-16 in the brief.
DEFAULT_TODAY = _date(2026, 6, 16)


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

    Reads ``result['blend_probs']`` (fallback ``pi_probs`` via
    ``resolve_model_probs_for_market``) and ``result['book_fair']``, then
    shows the per-market deltas, a divergence label, and the outcome with
    the largest disagreement.  Pure presentation: no I/O, no model calls.
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
        blended = result.get("blend_probs", result["pi_probs"])
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
    # has no corpus history). Always render above the tabs so the
    # user sees them first.
    identity_warnings = list(result.get("identity_warnings") or [])
    for iw in identity_warnings:
        st.warning(f"🪪 {iw}")

    tab_pred, tab_value, tab_analysis = st.tabs(
        ["🎯 Prediction", "💰 Betting Value", "🔬 Analysis"]
    )
    with tab_pred:
        _render_prediction_tab(result, identity_warnings)
    with tab_value:
        _render_betting_value_tab(
            result, min_edge=min_edge, identity_warnings=identity_warnings
        )
    with tab_analysis:
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
        value=DEFAULT_TODAY,
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
        st.warning(
            f"No unplayed matches found for {loaded_date} "
            "— try fetching live data via `scripts/fetch_live_2026.py`."
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
            value=DEFAULT_TODAY,
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
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("⚽ +EV Soccer Dashboard")
    st.caption("pi-rating + Elo blend vs book no-vig — calibrated, tiered, mobile-friendly")

    corpus = load_training_corpus()
    name_to_id = build_name_to_id(corpus)
    elo_snapshots = get_elo_snapshots()

    tab_auto, tab_manual = st.tabs(["📋 Auto-populate (2026 WC)", "✍️ Manual entry"])

    with tab_auto:
        _render_auto_populate_view(corpus, name_to_id, elo_snapshots)

    with tab_manual:
        _render_manual_view(corpus, name_to_id, elo_snapshots)


if __name__ == "__main__":
    main()
