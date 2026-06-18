"""
Pure, deterministic presenter helpers for the 3-tab UX redesign.

These functions consume plain dicts/strings/floats — no I/O, no Streamlit,
no new dependencies.  Called by the dashboard renderer to produce the
Prediction, Betting Value, and Analysis tab content.

All helpers are derived strictly from existing result fields.  They never
recompute model probabilities, never call evaluate_match, and never touch
Elo, Pi, Poisson, or market math.
"""
from __future__ import annotations

import re
from typing import Any

from soccer_ev_model.prediction_summary import (
    confidence_tier,
    model_agreement,
    resolve_model_probs_for_market,
    top_two_outcomes,
)

# --------------------------------------------------------------------------- #
# Market label map (shared)
# --------------------------------------------------------------------------- #
MARKET_LABEL = {"home": "Home Win", "draw": "Draw", "away": "Away Win"}


def _pretty_market(market: str, home_name: str, away_name: str) -> str:
    """Convert a market key ('home'|'draw'|'away') to a human label."""
    return {"home": home_name, "away": away_name, "draw": "Draw"}[market]


# --------------------------------------------------------------------------- #
# outcome headline (Prediction tab result card)
# --------------------------------------------------------------------------- #
def outcome_headline(most_likely: dict) -> str:
    """Return a deterministic, plain-English headline for the Prediction tab.

    Examples:
      * Home win      -> "<Home Team> to Win"
      * Away win      -> "<Away Team> to Win"
      * Draw outcome  -> "Match to End in a Draw"
    """
    market = most_likely.get("market", "")
    label = most_likely.get("label", "")
    if market == "draw":
        return "Match to End in a Draw"
    # Defensive escape: never trust the label to be HTML-safe
    return f"{label} to Win"


# --------------------------------------------------------------------------- #
# most_likely_result
# --------------------------------------------------------------------------- #
def most_likely_result(result: dict) -> dict:
    """Return the outcome with the highest blend probability.

    Uses ``result['primary_probs']`` (the official blended prediction),
    falling back to ``result['blend_probs']`` then ``result['pi_probs']``.
    Does NOT depend on entered odds.

    Returns:
        {"market": str, "label": str, "probability": float}
        where market is 'home'|'draw'|'away' and label is the human name.
    """
    probs = resolve_model_probs_for_market(result)
    order = ("home", "draw", "away")
    best = max(order, key=lambda m: (probs[m], -order.index(m)))
    home_name = result.get("home_team", "Home")
    away_name = result.get("away_team", "Away")
    return {
        "market": best,
        "label": _pretty_market(best, home_name, away_name),
        "probability": probs[best],
    }


# --------------------------------------------------------------------------- #
# prediction_confidence_label
# --------------------------------------------------------------------------- #
def prediction_confidence_label(result: dict) -> str:
    """Return High / Medium / Low from the existing confidence_tier mapping.

    Derives primarily from the existing ``result['confidence']`` assessment
    and the existing ``confidence_tier`` helper.  Does NOT invent tiers.

    Genuine multi-model DISAGREEMENT caps the label below High even when
    the raw tier is "A" — a high calibration / high data score doesn't
    mean anything if Pi and Elo are pointing at different matches.  A
    fragile agreement (same top but >= 10 pts probability gap) also caps
    below High.  A single-model result (missing Elo) is treated the
    same as a multi-model agreement of the same tier: tier A still
    reads as "High" (the existing tier is the best signal we have).
    """
    assessment = result["confidence"]
    tier = assessment.get("tier", "C")
    agreement = agreement_status(result)

    # Genuine multi-model disagreement or fragile agreement can never
    # produce a "High" prediction confidence — the methods are not in
    # alignment.  Cap at the tier below.
    if agreement in ("disagree", "fragile"):
        if tier == "A":
            return "Medium"
        if tier == "B":
            return "Medium"
        return "Low"

    # Tier-only mapping (covers agree, only_pi, only_elo).
    if tier == "A":
        return "High"
    if tier == "B":
        return "Medium"
    return "Low"


# --------------------------------------------------------------------------- #
# agreement_status — distinguish "models agree" from "only one model ran"
# --------------------------------------------------------------------------- #
def agreement_status(result: dict) -> str:
    """Return a deterministic label describing which prediction methods ran.

    Possible values:
      - ``"only_pi"``: pi-rating ran, no Elo (blend_was_used is False, or
        elo_only_probs is missing/None).
      - ``"only_elo"``: Elo ran, no pi (extremely rare; preserved for
        completeness).
      - ``"agree"``: both pi and Elo ran and they agree on the top market
        with < 10 pts probability gap.
      - ``"fragile"``: both ran, same top, but the probability gap is
        >= 10 pts.
      - ``"disagree"``: both ran and they pick different top markets.

    This helper is the single source of truth used by the casual Prediction
    and Betting Value presenters.  It deliberately NEVER compares a model
    against itself: when only one method ran we report "only_pi" /
    "only_elo" rather than fabricating an "agree" label.

    Pure presentation: reads only from the existing ``result`` dict.  No
    recomputation of probabilities, no math, no I/O.
    """
    pi_only = result.get("pi_only_probs")
    elo_only = result.get("elo_only_probs")
    pi_present = bool(pi_only)
    elo_present = bool(elo_only)
    if pi_present and not elo_present:
        return "only_pi"
    if elo_present and not pi_present:
        return "only_elo"
    if not pi_present and not elo_present:
        # Degenerate — fall through to a no-Elo label so the casual
        # wording still makes sense.
        return "only_pi"
    # Both ran: defer to the existing agreement helper.
    # (Reaching here implies pi_only and elo_only are truthy dicts.)
    assert pi_only is not None and elo_only is not None
    return model_agreement(pi_only, elo_only)["label"]


# --------------------------------------------------------------------------- #
# prediction_why_text
# --------------------------------------------------------------------------- #
def prediction_why_text(
    result: dict,
    warnings: list[str] | None = None,
    identity_warnings: list[str] | None = None,
) -> str:
    """Return a SHORT plain-language reason for the prediction.

    Picks the first matching reason from a strict priority order.
    Does NOT include raw tier letters, Elo, Pi, canonical IDs, raw
    counters, or neutral Pi ratings.
    """
    assessment = result.get("confidence", {})
    tier = assessment.get("tier", "")
    warnings = list(warnings or [])
    identity_warnings = list(identity_warnings or [])

    # --- 1. Limited historical data (highest priority) ---
    # Check identity_warnings for history_missing or no training-corpus history
    for iw in identity_words_lower(identity_warnings):
        if "no training-corpus history" in iw or "history_missing" in iw:
            return "Limited historical data is available for this team."
    # Check assessment tier C/D with low_data
    if tier in ("C", "D") and assessment.get("low_data", False):
        return "Limited historical data is available for this team."
    # Also check warnings for data-related signals
    for w in warnings:
        wl = w.lower()
        if "limited data" in wl or "insufficient data" in wl or "coin flip" in wl:
            return "Limited historical data is available for this team."

    # --- 2. Models disagree (BOTH models must have run) ---
    blended = resolve_model_probs_for_market(result)
    agreement = agreement_status(result)
    if agreement == "disagree":
        return "The prediction methods disagree on the most likely outcome."

    # --- 3. Stronger overall team rating (margin >= 15 pts) ---
    top, top_p, second, second_p = top_two_outcomes(blended)
    margin = (top_p - second_p) * 100
    if margin >= 15.0:
        return "One team has a noticeably stronger overall rating."

    # --- 4. Multiple methods agree (BOTH models must have run) ---
    if agreement in ("agree", "fragile"):
        # Genuine multi-model agreement: the model_agreement helper has
        # already confirmed pi_top == elo_top.
        return "Multiple prediction methods agree on the most likely outcome."

    # --- 4b. Only one prediction method was available ---
    if agreement in ("only_pi", "only_elo"):
        return "Only one prediction method was available for this matchup."

    # --- 5. Better squad strength (gap from context_loader) ---
    # This is checked via the match context gap if present in result metadata.
    # The dashboard passes this via the match_meta dict; we check a
    # pre-computed key that the renderer can attach.
    squad_gap = result.get("_squad_gap_pct")
    if squad_gap is not None and abs(squad_gap) >= 20.0:
        return "Squad market value favors one side significantly."

    # --- 6. Closely balanced (margin < 5 pts) ---
    if margin < 5.0:
        return "The match appears closely balanced."

    # --- 7. Methods disagree slightly (fragile) ---
    if agreement == "fragile":
        return "The methods disagree slightly, lowering confidence."

    # Fallback
    return "The match appears closely balanced."


def identity_words_lower(identity_warnings: list[str]) -> list[str]:
    """Lowercase each identity warning for case-insensitive matching."""
    return [w.lower() for w in identity_warnings]


# --------------------------------------------------------------------------- #
# value_play
# --------------------------------------------------------------------------- #
def value_play(result: dict, min_edge: float) -> dict:
    """Return the single best value play or no_clear_value.

    Uses existing ``result['plus_ev_flags']`` (already filtered by min_edge)
    and picks the highest edge.  If no flags, returns no_clear_value.

    Returns:
        {"status": "play", "market": str, "odds": float, "model_p": float,
         "market_p": float, "edge": float}
        or
        {"status": "no_clear_value", "reason": str}
    """
    flags = result.get("plus_ev_flags") or []
    if not flags:
        return {
            "status": "no_clear_value",
            "reason": "No outcome offers enough value at the entered odds",
        }

    # Pick the highest edge (flags are already sorted descending by edge)
    best = flags[0]
    market = best["market"]
    book_odds = result.get("book_odds", {}).get(market)
    return {
        "status": "play",
        "market": market,
        "odds": book_odds,
        "model_p": best["calibrated_pi"],
        "market_p": best["book_fair"],
        "edge": best["edge"],
    }


# --------------------------------------------------------------------------- #
# value_confidence_label
# --------------------------------------------------------------------------- #
def value_confidence_label(value_play_result: dict, result: dict) -> str:
    """Return High / Medium / Low for the value opportunity.

    Computed INDEPENDENTLY from prediction_confidence_label.  Derived
    strictly from edge magnitude, model agreement, and calibration.

    A single-model result (missing Elo) MUST NOT receive an
    agreement-based High label.  The High tier is reserved for plays
    that are supported by genuine multi-model agreement.
    """
    if value_play_result["status"] != "play":
        return "Low"

    edge = value_play_result["edge"]

    # Single source of truth: did BOTH models actually run?
    agreement = agreement_status(result)
    multi_model = agreement in ("agree", "fragile", "disagree")

    # Get calibration label from existing assessment
    assessment = result.get("confidence", {})
    calib_label = assessment.get("calib_label", "medium")

    # High: strong edge (>= 5%), genuine multi-model agreement, and
    # well-calibrated.  Single-model results cannot reach this tier
    # because we cannot confirm two methods support the play.
    if multi_model and edge >= 0.05 and agreement == "agree" and calib_label == "high":
        return "High"

    # Low: weak edge (< 2%), genuine multi-model disagreement, or
    # low calibration.  Single-model plays are NOT auto-Low here:
    # the brief says they must not be High, and a strong single-model
    # edge is still a meaningful signal (it just can't be corroborated).
    if (
        edge < 0.02
        or agreement == "disagree"
        or calib_label == "low"
    ):
        return "Low"

    # Medium: everything else — including single-model plays with a
    # strong edge but no second model to corroborate.
    return "Medium"


# --------------------------------------------------------------------------- #
# value_why_text
# --------------------------------------------------------------------------- #
def value_why_text(value_play_result: dict, result: dict) -> str:
    """Return a short plain-language reason for the value assessment.

    Plain-language rules:
      * The "Multiple prediction methods support this opportunity" line
        is only used when BOTH models actually ran (Pi + Elo).  A
        single-model result gets a separate single-model explanation.
      * The "model disagreement" line is only used when both models ran
        and either disagree on the top market or agree fragilely.
    """
    if value_play_result["status"] != "play":
        return "No outcome offers enough value at the entered odds"

    edge = value_play_result["edge"]
    market = value_play_result["market"]

    # Single source of truth: did BOTH models actually run?
    agreement = agreement_status(result)
    multi_model = agreement in ("agree", "fragile", "disagree")

    # Check if selected market is the predicted favorite
    blended = resolve_model_probs_for_market(result)
    top, _top_p, _second, _second_p = top_two_outcomes(blended)
    is_favorite = market == top

    if not is_favorite and edge > 0:
        # Neutral wording that works whether the predicted favorite is a
        # team (home/away) or a draw.  The previous copy ("The favorite
        # is most likely to win") was misleading when the model's top
        # outcome was a draw, because a draw does not "win" — see PR #8
        # review thread 3430230185.
        if top == "draw":
            return (
                "A draw is the most likely result, but this outcome "
                "offers better value at the current odds"
            )
        return (
            "The predicted winner is most likely, but its price is too "
            "expensive"
        )

    if multi_model and agreement == "agree" and edge > 0:
        return "Multiple prediction methods support this opportunity"

    if multi_model and agreement in ("disagree", "fragile") and edge > 0:
        return "The value exists, but model disagreement lowers confidence"

    if not multi_model and edge > 0:
        return (
            "Only one prediction method is available, so this value is "
            "based on a single signal"
        )

    if edge > 0:
        return "The sportsbook price suggests a lower chance than the model estimates"

    return "No outcome offers enough value at the entered odds"


# --------------------------------------------------------------------------- #
# translate_warning
# --------------------------------------------------------------------------- #
# Patterns that indicate internal/technical warnings needing translation.
# Each pattern is intentionally un-anchored (no leading ``^``) so it
# matches both leading-key forms ("canonical=CPV", "history_missing",
# "neutral pi-rating") and embedded-key forms
# ("status=history_missing", " ... (canonical=CPV, status=history_missing) ... ",
# etc.).  This guarantees that the casual-facing area above the
# Prediction / Betting Value / Analysis tabs never leaks raw
# canonical= / status= / pi-rating codes, regardless of which form the
# identity-warning pipeline produces.
_INTERNAL_WARNING_PATTERNS = [
    re.compile(r"canonical=", re.IGNORECASE),
    re.compile(r"\bhistory_missing\b", re.IGNORECASE),
    re.compile(r"home:\d+\s+away:\d+", re.IGNORECASE),
    re.compile(r"neutral\s+pi-rating", re.IGNORECASE),
    # Full identity warning sentence containing canonical= and status=
    re.compile(r"has no training-corpus history", re.IGNORECASE),
    re.compile(r"could not be resolved", re.IGNORECASE),
    re.compile(r"identity_unresolved", re.IGNORECASE),
]

_USER_FACING_SENTENCE_RE = re.compile(r"^[A-Z][a-z]+.*\s.*[a-z]\.?$")


def translate_warning(raw: str) -> str:
    """Translate internal warnings into user-facing text.

    Passes through any text that already looks like a user-facing sentence
    (contains spaces and doesn't match internal patterns).
    """
    if not raw:
        return raw

    # Check if it matches any internal pattern
    for pattern in _INTERNAL_WARNING_PATTERNS:
        if pattern.search(raw):
            return "Limited historical data is available for this team."

    # Check if it already looks like a user-facing sentence
    # (starts with capital, has spaces, ends with period)
    if " " in raw and raw[0].isupper() and raw.endswith("."):
        return raw

    # Check for other user-facing patterns
    if _USER_FACING_SENTENCE_RE.match(raw):
        return raw

    # Default: pass through
    return raw


def translate_and_dedupe_warnings(warnings: list[str]) -> list[str]:
    """Translate a list of raw warnings and deduplicate the translated output.

    Pure helper used by the casual-facing area above the Prediction / Betting
    Value / Analysis tabs.  Each entry is run through :func:`translate_warning`;
    entries that translate to the same sentence (or that translate to an empty
    string) are collapsed to a single entry, preserving first-seen order.

    The "no internal codes above the casual tabs" contract is enforced by
    :func:`translate_warning`; this helper additionally guarantees that
    identical translated sentences are not rendered twice.  The raw
    identity-warnings list is still rendered in the Analysis tab under
    "Calibration and Data Quality" (advanced view), unchanged.
    """
    if not warnings:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for raw in warnings:
        translated = translate_warning(raw)
        if not translated:
            continue
        if translated in seen:
            continue
        seen.add(translated)
        out.append(translated)
    return out


# --------------------------------------------------------------------------- #
# format_odds
# --------------------------------------------------------------------------- #
def format_odds(american_int: int | float | None) -> str:
    """Format American odds the way the existing app shows them.

    e.g. -230 stays "-230", "+350" stays "+350".
    """
    if american_int is None:
        return "—"
    val = int(american_int)
    if val > 0:
        return f"+{val}"
    return str(val)


# --------------------------------------------------------------------------- #
# Analysis-tab section helpers
# --------------------------------------------------------------------------- #
def analysis_prediction_details(result: dict) -> list[tuple[str, str]]:
    """Return (label, content) tuples for the Prediction Details section."""
    assessment = result.get("confidence", {})
    tier = assessment.get("tier", "?")
    tier_desc = assessment.get("tier_description", "")
    top_p = assessment.get("top_p", 0.0)
    cal_p = assessment.get("calibrated_p", 0.0)
    cal_diff = assessment.get("calibration_diff", 0.0)
    calib_label = assessment.get("calib_label", "")

    blended = resolve_model_probs_for_market(result)
    top, top_p_val, second, second_p_val = top_two_outcomes(blended)
    margin = (top_p_val - second_p_val) * 100

    home_name = result.get("home_team", "Home")
    away_name = result.get("away_team", "Away")

    lines = [
        ("Confidence tier", f"Tier {tier} — {tier_desc}"),
        ("Top probability (raw)", f"{top_p:.3f}"),
        ("Calibrated probability", f"{cal_p:.3f} (diff {cal_diff:+.3f}, {calib_label})"),
        ("Prediction margin", f"+{margin:.1f} pts"),
        ("Top outcome", f"{_pretty_market(top, home_name, away_name)} ({top_p_val:.1%})"),
        ("Second outcome", f"{_pretty_market(second, home_name, away_name)} ({second_p_val:.1%})"),
    ]
    return lines


def analysis_model_breakdown(result: dict) -> list[tuple[str, str]]:
    """Return (label, content) tuples for the Model Breakdown section.

    Pure presentation.  Uses ``agreement_status`` (the single source of
    truth) so Pi is never compared against itself when Elo is missing.
    """
    blended = resolve_model_probs_for_market(result)
    pi_only = result.get("pi_only_probs") or blended
    elo_only = result.get("elo_only_probs")
    blend_was_used = result.get("blend_was_used", False)

    home_name = result.get("home_team", "Home")
    away_name = result.get("away_team", "Away")

    lines = []
    for market in ("home", "draw", "away"):
        label = _pretty_market(market, home_name, away_name)
        pi_val = pi_only.get(market, 0.0)
        elo_val = elo_only.get(market, 0.0) if elo_only else None
        blend_val = blended.get(market, 0.0)

        elo_str = f"{elo_val:.1%}" if elo_val is not None else "—"
        lines.append((
            label,
            f"Pi: {pi_val:.1%} | Elo: {elo_str} | Blend: {blend_val:.1%}",
        ))

    lines.append(("Blend used", "Yes (Pi + Elo)" if blend_was_used else "No (Pi only)"))

    # Model agreement — use agreement_status to avoid Pi-vs-self.
    agreement = agreement_status(result)
    agreement_text = {
        "only_pi":     "Only Pi ran (no Elo available)",
        "only_elo":    "Only Elo ran (no Pi available)",
        "agree":       "Pi and Elo agree",
        "fragile":     "Pi and Elo agree (fragile, ≥10pt probability gap)",
        "disagree":    "Pi and Elo disagree",
    }[agreement]
    lines.append(("Model agreement", agreement_text))
    if agreement == "disagree":
        # Show the two top picks so the user can see exactly where
        # the methods diverge.  Both pi_only and elo_only are present
        # at this point (we're in the multi-model branch).
        assert pi_only is not None and elo_only is not None
        pi_top = _top_market_safe(pi_only)
        elo_top = _top_market_safe(elo_only)
        lines.append(("Pi top", _pretty_market(pi_top, home_name, away_name)))
        lines.append(("Elo top", _pretty_market(elo_top, home_name, away_name)))

    return lines


def _top_market_safe(probs: dict[str, float]) -> str:
    """Return the top market key from a prob dict.  Defensive — never raises."""
    if not probs:
        return "home"
    return max(("home", "draw", "away"), key=lambda m: probs.get(m, 0.0))


def analysis_market_comparison(result: dict) -> list[tuple[str, str]]:
    """Return (label, content) tuples for the Market Comparison section."""
    from soccer_ev_model.prediction_summary import (
        calculate_market_deltas,
        largest_market_delta,
        market_divergence_label,
        prediction_margin_pct,
    )

    model_probs = resolve_model_probs_for_market(result)
    market_probs = result["book_fair"]
    pts_deltas = calculate_market_deltas(model_probs, market_probs)
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

    lines = [
        ("Divergence label", div_label),
        ("Largest delta", f"{largest['label']} {largest['delta_pts']:+.1f} pts"),
    ]

    for market in ("home", "draw", "away"):
        label = _pretty_market(market, home_name, away_name)
        lines.append((
            label,
            f"Model: {model_probs[market]:.1%} | Market: {market_probs[market]:.1%} | Delta: {pts_deltas[market]:+.1f} pts",
        ))

    return lines


def analysis_poisson_view(result: dict) -> list[tuple[str, str]]:
    """Return (label, content) tuples for the Poisson Score View section."""
    from soccer_ev_model.prediction_summary import (
        expected_goals_from_blend,
        poisson_agreement_label,
        poisson_outcome_probs,
    )

    model_probs = resolve_model_probs_for_market(result)
    xg = expected_goals_from_blend(model_probs)
    poisson_probs = poisson_outcome_probs(xg["home_xg"], xg["away_xg"])
    agreement = poisson_agreement_label(model_probs, poisson_probs)

    home_name = result.get("home_team", "Home")
    away_name = result.get("away_team", "Away")
    market_labels = {"home": home_name, "draw": "Draw", "away": away_name}

    lines = [
        ("xG estimate", f"{home_name} {xg['home_xg']} / {away_name} {xg['away_xg']}"),
        ("Poisson home", f"{poisson_probs['home']:.1%}"),
        ("Poisson draw", f"{poisson_probs['draw']:.1%}"),
        ("Poisson away", f"{poisson_probs['away']:.1%}"),
        ("Blend top", market_labels.get(agreement["blend_top"], agreement["blend_top"])),
        ("Poisson top", market_labels.get(agreement["poisson_top"], agreement["poisson_top"])),
        ("Agreement", agreement["label"]),
    ]
    return lines


def analysis_squad_context(
    result: dict,
    home_canonical_id: str = "",
    away_canonical_id: str = "",
) -> list[tuple[str, str]]:
    """Return (label, content) tuples for the Squad and Team Context section.

    Reads from the result dict and optional canonical IDs.  Does NOT call
    the context_loader directly (that's the renderer's job).  Instead it
    reads pre-attached squad context from result['_squad_context'] if present,
    or returns a minimal placeholder.
    """
    squad_ctx = result.get("_squad_context")
    if not squad_ctx:
        return [("Squad context", "No squad-strength data available.")]

    home_ctx = squad_ctx.get("home", {})
    away_ctx = squad_ctx.get("away", {})
    gap = squad_ctx.get("gap", {})

    home_name = result.get("home_team", "Home")
    away_name = result.get("away_team", "Away")

    lines = []

    # Squad values
    hv = home_ctx.get("squad_value")
    av = away_ctx.get("squad_value")
    lines.append((f"{home_name} squad value", _format_eur_short(hv)))
    lines.append((f"{away_name} squad value", _format_eur_short(av)))

    # Tiers
    lines.append((f"{home_name} value tier", home_ctx.get("value_tier", "unknown")))
    lines.append((f"{away_name} value tier", away_ctx.get("value_tier", "unknown")))

    # Gap
    home_gap = gap.get("home_pct")
    away_gap = gap.get("away_pct")
    lines.append((f"{home_name} gap vs opponent", _format_gap_short(home_gap)))
    lines.append((f"{away_name} gap vs opponent", _format_gap_short(away_gap)))

    # FIFA ranks
    home_rank = home_ctx.get("fifa_rank")
    away_rank = away_ctx.get("fifa_rank")
    if home_rank is not None:
        lines.append((f"{home_name} FIFA rank", f"#{home_rank}"))
    if away_rank is not None:
        lines.append((f"{away_name} FIFA rank", f"#{away_rank}"))

    # Notes
    notes = (home_ctx.get("notes") or []) + (away_ctx.get("notes") or [])
    if notes:
        lines.append(("Team notes", f"{len(notes)} note(s) attached"))

    return lines


def _format_eur_short(value: int | None) -> str:
    """Short EUR formatter (mirrors dashboard.context_loader.format_eur)."""
    if value is None:
        return "Unknown"
    v = float(value)
    if v >= 100_000_000:
        return f"€{v / 1_000_000_000:.2f}B"
    if v >= 1_000_000:
        return f"€{v / 1_000_000:.0f}M"
    return f"€{int(v):,}"


def _format_gap_short(pct: float | None) -> str:
    """Short gap formatter (mirrors dashboard.context_loader.format_gap)."""
    if pct is None:
        return "—"
    if pct > 0:
        return f"▲ +{pct:.1f}%"
    if pct < 0:
        return f"▼ {pct:.1f}%"
    return "± 0.0%"


def analysis_calibration_and_data_quality(
    result: dict,
    identity_warnings: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Return (label, content) tuples for the Calibration and Data Quality section."""
    assessment = result.get("confidence", {})
    identity_warnings = identity_warnings or []

    lines = [
        ("Tier", assessment.get("tier", "?")),
        ("Tier description", assessment.get("tier_description", "")),
        ("Top_p (raw)", f"{assessment.get('top_p', 0.0):.3f}"),
        ("Calibrated_p", f"{assessment.get('calibrated_p', 0.0):.3f}"),
        ("Calibration diff", f"{assessment.get('calibration_diff', 0.0):+.3f}"),
        ("Calibration label", assessment.get("calib_label", "")),
        ("Data label", assessment.get("data_label", "")),
        ("Home matches played", str(assessment.get("home_matches_played", "?"))),
        ("Away matches played", str(assessment.get("away_matches_played", "?"))),
        ("Edge warning", str(assessment.get("edge_warning", False))),
        ("Identity unresolved", str(assessment.get("identity_unresolved", False))),
    ]

    # Warnings
    warnings = assessment.get("warnings") or []
    if warnings:
        for i, w in enumerate(warnings):
            lines.append((f"Warning {i + 1}", w))

    # Identity warnings
    if identity_warnings:
        for i, iw in enumerate(identity_warnings):
            lines.append((f"Identity warning {i + 1}", iw))

    return lines


def analysis_raw_diagnostics(result: dict) -> dict:
    """Return a dict of raw diagnostic values for the Raw Diagnostics section."""
    return {
        "book_odds": result.get("book_odds", {}),
        "book_fair": result.get("book_fair", {}),
        "primary_probs": result.get("primary_probs", {}),
        "pi_probs": result.get("pi_probs", {}),
        "blend_probs": result.get("blend_probs", result.get("pi_probs", {})),
        "pi_only_probs": result.get("pi_only_probs", {}),
        "elo_only_probs": result.get("elo_only_probs"),
        "blend_was_used": result.get("blend_was_used", False),
        "calibrated_pi": result.get("calibrated_pi", {}),
        "edges": result.get("edges", {}),
        "plus_ev_flags": result.get("plus_ev_flags", []),
        "canonical_home_id": result.get("canonical_home_id", ""),
        "canonical_away_id": result.get("canonical_away_id", ""),
        "identity_warnings": result.get("identity_warnings", []),
    }
