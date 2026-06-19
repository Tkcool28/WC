"""
Pure pi-rating + no-vig +EV workflow.

For a given match, compute pi-rating fair odds, then compare to book odds.
Returns +EV candidates where pi-rating % > no-vig market %.

This is the "manual +EV" approach: pi-rating IS the model.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from .no_vig import implied_probs, remove_vig
from .pi_ratings import (
    compute_pi_ratings,
    compute_pi_ratings_walk_forward,
    get_team_experience,
    pi_diff_features,
)
from .confidence import assess_match_confidence, render_warning_banner

# Base rates for the logistic mapping. Used by both the pi-only and the
# pi+Elo blend paths. Centralised so any future recalibration is one edit.
_BASE_H, _BASE_D, _BASE_A = 0.40, 0.27, 0.33
# Scale on the combined matchup signal. 0.6 was chosen so a 1.0 pi edge
# gives ~65% win prob; the same scale is applied to the blended signal so
# the two systems share a common unit (Elo is divided by 400 by convention).
_LOGIT_SCALE = 0.6
# Draw shrinks as |matchup| grows.
_DRAW_SCALE = 0.3


def _logistic_matchup(matchup: float) -> dict[str, float]:
    """Map a single combined-matchup value to H/D/A probabilities.

    `matchup` is expected to be on the same scale as pi_matchup (i.e. an
    additive offence+defence difference for the home side). The Elo blend
    caller is responsible for normalising Elo to that same scale (we use
    /400 by convention: 400 Elo ≈ 10x win ratio).

    Positive matchup => home is stronger => shifts mass to p_h.
    Negative matchup => away is stronger => shifts mass to p_a.
    |matchup| large  => draw probability shrinks.
    """
    import math

    logit_shift = _LOGIT_SCALE * matchup
    p_h = _BASE_H * math.exp(logit_shift)
    p_a = _BASE_A * math.exp(-logit_shift)
    p_d = _BASE_D * math.exp(-_DRAW_SCALE * abs(matchup))

    total = p_h + p_d + p_a
    return {"home": p_h / total, "draw": p_d / total, "away": p_a / total}


def _probs_from_ratings(match: dict, ratings: dict) -> dict[str, float]:
    """Map a pi-rating snapshot to H/D/A probabilities for a single match.

    Pure pi-rating path. The pi+Elo blend path is `_probs_from_ratings_blend`.
    """
    feats = pi_diff_features(
        home_id=match["home_team_id"],
        away_id=match["away_team_id"],
        ratings=ratings,
    )

    pi_off_diff = feats["pi_off_diff"]      # home_attack - away_attack
    pi_def_diff = feats["pi_def_diff"]      # home_defense - away_defense (higher = better def)

    # Combined: home better off AND better def -> positive
    pi_matchup = pi_off_diff + pi_def_diff
    return _logistic_matchup(pi_matchup)


def _probs_from_ratings_blend(
    match: dict,
    ratings: dict,
    home_elo: float,
    away_elo: float,
    w_pi: float = 1.0,
    w_elo: float = 0.0,
) -> dict[str, float]:
    """Map (pi-rating, Elo) to H/D/A probabilities via a hand-tuned linear blend.

    The blend is a single additive combination of the two matchup signals
    on a common scale:

        elo_diff_normalized = (home_elo - away_elo) / 400
        combined_matchup    = w_pi * pi_matchup + w_elo * elo_diff_normalized

    Elo is divided by 400 by convention (400 Elo ≈ 10x win ratio). With
    the same _LOGIT_SCALE = 0.6 applied to the combined signal, the two
    sources contribute on roughly the same scale as a 1.0 pi edge.

    Constraints:
        w_pi + w_elo == 1.0  (callers are responsible; we don't renormalise)
        w_elo == 0.0 is the pure pi-rating case (identical to _probs_from_ratings)
        w_pi  == 0.0 is the pure Elo case (smoke test)
    """
    feats = pi_diff_features(
        home_id=match["home_team_id"],
        away_id=match["away_team_id"],
        ratings=ratings,
    )
    pi_matchup = feats["pi_off_diff"] + feats["pi_def_diff"]
    elo_diff_normalized = (home_elo - away_elo) / 400.0
    combined = w_pi * pi_matchup + w_elo * elo_diff_normalized
    return _logistic_matchup(combined)


def pi_rating_match_probs(
    matches: list[dict],
    match: dict,
) -> dict[str, float]:
    """
    Compute H/D/A probabilities for `match` using pi-ratings trained on `matches`.

    Uses a logistic mapping of pi-matchup to W/D/A probs.
    Coefficients chosen to roughly match historical WC base rates:
    ~40% home, ~27% draw, ~33% away (for neutral venue).

    The pi_diff drives how these base rates shift.

    For a single prediction this uses the cutoff-based path; for backtests
    that call this many times, prefer `pi_rating_match_probs_batch` which
    is dramatically faster.
    """
    cutoff = match["date"]
    ratings = compute_pi_ratings(matches, cutoff=cutoff)
    return _probs_from_ratings(match, ratings)


def pi_rating_match_probs_batch(
    train_matches: list[dict],
    test_matches: list[dict],
) -> list[dict[str, float]]:
    """Compute H/D/A probs for many test matches in a single walk-forward pass.

    For each test match, ratings are computed from all training matches
    AND any prior test matches (walk-forward). Test matches MUST be in
    chronological order.

    This is much faster than calling `pi_rating_match_probs` once per
    test match: O(N) instead of O(N * M) for N training matches and
    M test matches.
    """
    snapshots = compute_pi_ratings_walk_forward(
        train_matches, test_matches, consume_test_results=True,
    )
    return [_probs_from_ratings(m, ratings) for m, ratings in snapshots]


def _calibrate_probs(pi_probs: dict[str, float], calibrated_top_p: float) -> dict[str, float]:
    """Build a calibrated probability distribution.

    The calibration table (confidence.py) maps raw top_p -> empirically-observed
    hit rate. To turn that into a full 3-way distribution, we:
      1. Find the market with the highest raw probability (the "top" market).
      2. Scale the top market's probability DOWN to `calibrated_top_p`.
      3. Redistribute the difference (raw_top - calibrated_top) proportionally
         to the other two markets, in the ratio of their raw probabilities.

    This preserves the ordering of the three markets and the relative
    underdog-vs-underdog split, while toning down the overconfident top.
    The result always sums to 1.0.
    """
    markets = list(pi_probs.keys())  # ["home", "draw", "away"]
    raw_top_market = max(markets, key=lambda m: pi_probs[m])
    raw_top_p = pi_probs[raw_top_market]

    # If top is already at or below the calibrated value, return the raw probs
    # (the calibration is only meant to REDUCE overconfidence, not invent
    # confidence where the model is already honest).
    if raw_top_p <= calibrated_top_p:
        return dict(pi_probs)

    shrink = raw_top_p - calibrated_top_p
    other_markets = [m for m in markets if m != raw_top_market]
    other_sum = sum(pi_probs[m] for m in other_markets)

    calibrated = dict(pi_probs)
    calibrated[raw_top_market] = calibrated_top_p

    if other_sum <= 0:
        # Pathological case (shouldn't happen with valid probs): split evenly
        for m in other_markets:
            calibrated[m] = shrink / len(other_markets)
    else:
        for m in other_markets:
            calibrated[m] = pi_probs[m] + shrink * (pi_probs[m] / other_sum)

    # Renormalize for floating-point safety
    total = sum(calibrated.values())
    if total > 0:
        calibrated = {k: v / total for k, v in calibrated.items()}
    return calibrated


def predict_match(
    home_team: str,
    away_team: str,
    home_team_id: int,
    away_team_id: int,
    date: str,
    ratings: dict,
    *,
    home_elo: float | None = None,
    away_elo: float | None = None,
    blend_w_pi: float = 0.5,
    blend_w_elo: float = 0.5,
    goal_probs: dict[str, float] | None = None,
    goal_model_xg: dict[str, float] | None = None,
    goal_model_low_data: bool = False,
    _goal_model_expected: bool = False,
    identity_unresolved: bool = False,
    canonical_home_id: str | None = None,
    canonical_away_id: str | None = None,
    goal_model_metadata: dict | None = None,
) -> dict:
    """Pure model-only prediction. NO odds required.

    Computes the raw pi-rating (or pi+Elo blend) probabilities, the
    full confidence assessment, and the warning banner for a single
    matchup. Returns a dict shaped to be consumable by the mobile
    dashboard and by `evaluate_market` (the market layer).

    The returned dict contains ONLY model-derived keys. It MUST NOT
    contain `book_odds`, `book_fair`, `calibrated_pi`, `edges`, or
    `plus_ev_flags` — those are the responsibility of `evaluate_market`.

    Args:
        home_team, away_team: human-readable team names.
        home_team_id, away_team_id: integer team ids (must match `ratings` keys).
        date: ISO date string (e.g. "2026-06-16").
        ratings: output of compute_pi_ratings() at the cutoff date.
        home_elo, away_elo: optional Elo ratings. If BOTH are provided,
            the model uses a hand-tuned blend of pi-rating and Elo; if
            either is None, the model is pure pi-rating. Hand-tuned
            weights from scripts/blend_backtest.py: w_pi=0.5, w_elo=0.5
            was the best of 3 candidates on the 2022 WC walk-forward
            (RPS 0.222 vs 0.230 for pi-only, n=64).
        blend_w_pi, blend_w_elo: blend weights (default 0.5/0.5).
            Ignored if home_elo or away_elo is None.
        goal_probs: optional goal model H/D/A probabilities. When provided
            along with Elo ratings, primary_probs uses the Elo60/Goal40
            blend (60% Elo + 40% Goal model). If None or Elo is missing,
            falls back to the pi+Elo blend (or pure pi-rating).
        identity_unresolved: keyword-only. Set by the dashboard when
            team identity could not be resolved through the canonical
            registry. Propagated into the returned `confidence` dict
            as a separate flag so the renderer can show a distinct
            warning without conflating it with the "low data" tier.
        canonical_home_id, canonical_away_id: optional 3-letter
            canonical team ids (e.g. "ARG", "BRA"). Surfaced in the
            result so downstream renderers don't have to re-look-up
            identities. Empty string if unresolved.

    Returns:
        dict with at least these keys:
          - primary_probs      (3-way dict, the official blended prediction)
          - pi_probs           (3-way dict, alias of primary_probs for backward compat)
          - blend_probs        (3-way dict, alias of primary_probs for backward compat)
          - pi_only_probs      (3-way dict, pure pi-rating)
          - elo_only_probs     (3-way dict or None)
          - home_team, away_team
          - home_team_id, away_team_id
          - date
          - blend_was_used     (bool)
          - blend_w_pi         (float, the weight used)
          - blend_w_elo        (float, the weight used)
          - confidence         (full assess_match_confidence dict)
          - banner             (str)
          - canonical_home_id  (str, best-effort; "" if unresolved)
          - canonical_away_id  (str, best-effort; "" if unresolved)
    """
    # Build a synthetic match dict for the existing probs helper
    match = {
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "date": date,
    }
    if home_elo is not None and away_elo is not None:
        pi = _probs_from_ratings_blend(
            match, ratings,
            home_elo=home_elo, away_elo=away_elo,
            w_pi=blend_w_pi, w_elo=blend_w_elo,
        )
        pi_only = _probs_from_ratings(match, ratings)
        elo_only = _probs_from_ratings_blend(
            match, ratings,
            home_elo=home_elo, away_elo=away_elo,
            w_pi=0.0, w_elo=1.0,
        )
        blend_was_used = True
    else:
        pi = _probs_from_ratings(match, ratings)
        pi_only = pi
        elo_only = None
        blend_was_used = False

    # Compute primary_probs: Elo60/Goal40 blend when goal_probs and Elo
    # are both available, otherwise fall back to pi+Elo blend (or pure pi).
    # This is the official prediction source for all downstream consumers.
    _goal_model_used = False
    _goal_model_xg = None
    _goal_model_low_data = False
    _goal_model_most_likely_score = None
    _goal_model_expected_total_goals = None
    _goal_model_version = None
    _goal_model_data_cutoff = None
    _goal_model_low_data_flags = None
    if goal_probs is not None and elo_only is not None:
        primary_raw = {
            "home": 0.6 * elo_only["home"] + 0.4 * goal_probs["home"],
            "draw": 0.6 * elo_only["draw"] + 0.4 * goal_probs["draw"],
            "away": 0.6 * elo_only["away"] + 0.4 * goal_probs["away"],
        }
        primary_total = sum(primary_raw.values())
        if primary_total > 0:
            primary_probs = {k: v / primary_total for k, v in primary_raw.items()}
        else:
            primary_probs = dict(pi)
        _goal_model_used = True
        _goal_model_xg = goal_model_xg
        _goal_model_low_data = goal_model_low_data
        _goal_model_most_likely_score = (goal_model_metadata or {}).get("most_likely_score")
        _goal_model_expected_total_goals = (goal_model_metadata or {}).get("expected_total_goals")
        _goal_model_version = (goal_model_metadata or {}).get("model_version")
        _goal_model_data_cutoff = (goal_model_metadata or {}).get("data_cutoff")
        _goal_model_low_data_flags = (goal_model_metadata or {}).get("low_data_flags")
    else:
        # Fall back to pi+Elo blend (or pure pi)
        primary_probs = dict(pi)

    # Confidence assessment
    home_exp = get_team_experience(ratings, home_team_id)
    away_exp = get_team_experience(ratings, away_team_id)
    confidence = assess_match_confidence(
        home_matches_played=home_exp["matches_played"],
        away_matches_played=away_exp["matches_played"],
        pi_probs=pi,
    )
    # Identity-resolution flag is a *separate* signal from the tier logic
    # in `assess_match_confidence`. The dashboard uses it to show a
    # "Team identity could not be resolved" warning INSTEAD of (or in
    # addition to) the low-data warning. We attach it here so the
    # renderer's existing _render_warnings / _render_confidence_banner
    # functions continue to work unchanged.
    confidence["identity_unresolved"] = bool(identity_unresolved)

    banner = render_warning_banner(confidence)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "date": date,
        # primary_probs: the official prediction source (Elo60/Goal40 blend
        # when goal_probs is available + Elo is present, pi+Elo or pure pi
        # otherwise).  All consumers MUST use this field.  pi_probs and
        # blend_probs are backward-compat aliases.
        "primary_probs": {k: round(v, 4) for k, v in primary_probs.items()},
        "pi_probs": {k: round(v, 4) for k, v in pi.items()},
        "blend_probs": {k: round(v, 4) for k, v in pi.items()},
        "pi_only_probs": {k: round(v, 4) for k, v in pi_only.items()},
        "elo_only_probs": {k: round(v, 4) for k, v in elo_only.items()} if elo_only is not None else None,
        "blend_was_used": blend_was_used,
        "blend_w_pi": blend_w_pi,
        "blend_w_elo": blend_w_elo,
        "_goal_model_used": _goal_model_used,
        "_goal_model_xg": _goal_model_xg,
        "_goal_model_low_data": _goal_model_low_data,
        "_goal_model_expected": _goal_model_expected,
        "_goal_model_most_likely_score": _goal_model_most_likely_score,
        "_goal_model_expected_total_goals": _goal_model_expected_total_goals,
        "_goal_model_version": _goal_model_version,
        "_goal_model_data_cutoff": _goal_model_data_cutoff,
        "_goal_model_low_data_flags": _goal_model_low_data_flags,
        "goal_model_hda": {k: round(v, 4) for k, v in goal_probs.items()} if goal_probs is not None and _goal_model_used else None,
        "confidence": confidence,
        "banner": banner,
        "canonical_home_id": canonical_home_id or "",
        "canonical_away_id": canonical_away_id or "",
        "home_elo": home_elo,
        "away_elo": away_elo,
    }


def evaluate_market(
    prediction: dict,
    book_home_odds: float,
    book_draw_odds: float,
    book_away_odds: float,
    min_edge: float = 0.03,
    *,
    identity_unresolved: bool = False,
) -> dict:
    """Market evaluation. REQUIRES valid odds.

    Takes a prediction dict (output of `predict_match`) and book odds,
    and returns the market-derived keys: book no-vig probs, calibration,
    edges, +EV flags, plus summary signals (divergence label, largest
    delta). This is the layer the betting-value UI consumes.

    Args:
        prediction: dict produced by `predict_match`. Must contain at
            minimum: `primary_probs` (3-way dict), `confidence` (with key
            `calibrated_p`), `home_team`, `away_team`.  `primary_probs` is
            the official blended prediction field.  For backward compat,
            `pi_probs` is accepted if `primary_probs` is missing.
        book_home_odds, book_draw_odds, book_away_odds: American odds.
            Must be valid (non-zero, in range). Inherits the same
            `ValueError` semantics as `no_vig.remove_vig`.
        min_edge: minimum edge to flag as +EV (default 0.03 = 3%).
        identity_unresolved: keyword-only. Surfaced on the
            `confidence["identity_unresolved"]` flag for backward
            compatibility with `evaluate_match`'s contract. The
            predict_match output already carries the canonical flag;
            this only changes the output of `evaluate_market` when
            called with a prediction dict that does NOT include the
            identity flag (rare).

    Returns:
        dict with at least these keys:
          - book_odds            (dict of label -> float)
          - book_fair            (no-vig 3-way dict, rounded to 4 dp)
          - calibrated_pi        (3-way dict, rounded to 4 dp)
          - edges                (3-way dict, rounded to 4 dp)
          - plus_ev_flags        (list of {market, edge, ...})
          - plus_ev_count        (int, len(plus_ev_flags))
          - market_divergence    (str or None — label from prediction_summary)
          - largest_market_delta (dict or None — from prediction_summary)

    Raises:
        ValueError: if any odds is zero or out-of-range (propagated
            from `no_vig.remove_vig` via `implied_probs`).
        ValueError: if `prediction` is missing required keys.
    """
    # ---- validate prediction dict ----
    required = ("primary_probs",)
    for k in required:
        if k not in prediction:
            # Backward compat: accept pi_probs if primary_probs is missing
            if "pi_probs" in prediction:
                prediction["primary_probs"] = prediction["pi_probs"]
            else:
                raise ValueError(
                    f"evaluate_market: prediction dict missing required key 'primary_probs'"
                )
    primary_probs = prediction["primary_probs"]
    confidence = prediction.get("confidence") or {}
    if "calibrated_p" not in confidence:
        raise ValueError(
            "evaluate_market: prediction['confidence'] missing 'calibrated_p'"
        )

    # ---- book no-vig probabilities (may raise ValueError) ----
    imp = implied_probs(book_home_odds, book_draw_odds, book_away_odds)
    book_fair = imp["fair"]

    # ---- calibrated primary probs. NOTE: consumed ONLY by the +EV flag
    # pipeline below. The dashboard prediction summary intentionally
    # displays the raw primary_probs, not these calibrated values —
    # calibration is an EV-layer concern. ----
    calibrated_pi = _calibrate_probs(primary_probs, confidence["calibrated_p"])

    # ---- edges (raw primary minus book_fair). This is the +EV signal. ----
    edges = {
        m: round(primary_probs[m] - book_fair[m], 4)
        for m in ("home", "draw", "away")
    }

    # ---- +EV flags (pre-filtered by min_edge) ----
    plus_ev_flags = []
    for market in ("home", "draw", "away"):
        if edges[market] >= min_edge:
            plus_ev_flags.append({
                "market": market,
                "edge": edges[market],
                "calibrated_pi": round(calibrated_pi[market], 4),
                "book_fair": round(book_fair[market], 4),
            })
    plus_ev_flags.sort(key=lambda f: -f["edge"])

    # ---- market-divergence summary signals (optional, additive) ----
    # Imported lazily here to keep this module's import graph flat for
    # callers that only need the core market math.
    try:
        from .prediction_summary import (
            largest_market_delta,
            market_divergence_label,
        )
        # Build a pct-scale deltas dict (matches the contract in
        # `prediction_summary.largest_market_delta`: model - market,
        # in percentage points).
        deltas_pct = {
            m: round((primary_probs[m] - book_fair[m]) * 100, 1)
            for m in ("home", "draw", "away")
        }
        market_divergence = market_divergence_label(deltas_pct)
        market_labels = {
            "home": prediction.get("home_team", "home"),
            "away": prediction.get("away_team", "away"),
            "draw": "Draw",
        }
        largest_market_delta_val = largest_market_delta(
            deltas_pct,
            market_labels=market_labels,
            model_probs=primary_probs,
            market_probs=book_fair,
        )
    except Exception:
        # Never let the summary signals break the core market math.
        market_divergence = None
        largest_market_delta_val = None

    return {
        "book_odds": {
            "home": book_home_odds,
            "draw": book_draw_odds,
            "away": book_away_odds,
        },
        "book_fair": {k: round(v, 4) for k, v in book_fair.items()},
        "calibrated_pi": {k: round(v, 4) for k, v in calibrated_pi.items()},
        "edges": edges,
        "plus_ev_flags": plus_ev_flags,
        "plus_ev_count": len(plus_ev_flags),
        "market_divergence": market_divergence,
        "largest_market_delta": largest_market_delta_val,
    }


def evaluate_match(
    home_team: str,
    away_team: str,
    home_team_id: int,
    away_team_id: int,
    date: str,
    book_home_odds: float,
    book_draw_odds: float,
    book_away_odds: float,
    ratings: dict,
    min_edge: float = 0.03,
    home_elo: float | None = None,
    away_elo: float | None = None,
    blend_w_pi: float = 0.5,
    blend_w_elo: float = 0.5,
    *,
    identity_unresolved: bool = False,
    canonical_home_id: str | None = None,
    canonical_away_id: str | None = None,
) -> dict:
    """Backward-compat wrapper: prediction + market, merged.

    This is the function the dashboard has historically called. It is
    now a thin wrapper around `predict_match` (model-only) and
    `evaluate_market` (market layer), returning the union of both
    dicts so the legacy return shape — home_team / away_team / date /
    book_odds / book_fair / pi_probs / pi_only_probs / elo_only_probs
    / blend_probs / blend_was_used / calibrated_pi / edges / confidence
    / plus_ev_flags / banner — is preserved verbatim.

    If any of `book_home_odds`, `book_draw_odds`, `book_away_odds` is
    `None`, the market layer is skipped and the result is the
    prediction dict only (the dashboard does this when no bookmaker
    prices are available). This branch is additive and does not
    affect existing callers, all of which pass numeric odds.

    Args:
        (see `predict_match` and `evaluate_market` for the per-arg docs)

    Returns:
        dict — union of prediction and (optional) market dict. Strict
        superset of the legacy `evaluate_match` return shape.
    """
    prediction = predict_match(
        home_team=home_team,
        away_team=away_team,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        date=date,
        ratings=ratings,
        home_elo=home_elo,
        away_elo=away_elo,
        blend_w_pi=blend_w_pi,
        blend_w_elo=blend_w_elo,
        identity_unresolved=identity_unresolved,
        canonical_home_id=canonical_home_id,
        canonical_away_id=canonical_away_id,
    )

    # Graceful handling: if any odds is missing, return prediction only.
    all_odds = (book_home_odds, book_draw_odds, book_away_odds)
    if any(o is None for o in all_odds):
        return prediction

    market = evaluate_market(
        prediction,
        book_home_odds=book_home_odds,
        book_draw_odds=book_draw_odds,
        book_away_odds=book_away_odds,
        min_edge=min_edge,
        identity_unresolved=identity_unresolved,
    )
    return {**prediction, **market}


def find_value_bets(
    matches: list[dict],
    upcoming: list[dict],
    book_odds: dict[int, dict[str, tuple[int, int]]],
    min_edge: float = 0.03,
    include_confidence: bool = False,
) -> list[dict]:
    """
    Compare pi-rating fair probs to book no-vig probs for each upcoming match.

    Args:
        matches: historical matches (for pi-rating training)
        upcoming: list of matches to evaluate
        book_odds: {match_id: {"home": (num, den), "draw": (num, den), "away": (num, den)}}
                   American odds as (positive, negative) tuple, e.g. (+150, -150)
        min_edge: minimum edge (pi% - book%) to flag
        include_confidence: if True, augment each result with a 'confidence' key
                            holding the full assess_match_confidence dict for
                            that match. Off by default for backward compat.

    Returns: list of {match_id, market, pi_prob, book_prob, edge} for +EV plays.
             If include_confidence=True, each row also has a 'confidence' dict.
    """
    results = []
    for match in upcoming:
        mid = match.get("match_id") or match.get("id")
        if mid not in book_odds:
            continue

        # Pi-rating fair probs
        pi = pi_rating_match_probs(matches, match)

        # Book no-vig probs
        odds = book_odds[mid]
        book = implied_probs(odds["home"], odds["draw"], odds["away"])["fair"]

        # Optional confidence assessment (compute once per match, not per market)
        confidence = None
        if include_confidence:
            # We need ratings to look up matches_played
            cutoff = match["date"]
            ratings = compute_pi_ratings(matches, cutoff=cutoff)
            from .confidence import assess_match_confidence as _assess
            from .pi_ratings import get_team_experience as _gte
            home_exp = _gte(ratings, match["home_team_id"])
            away_exp = _gte(ratings, match["away_team_id"])
            confidence = _assess(
                home_matches_played=home_exp["matches_played"],
                away_matches_played=away_exp["matches_played"],
                pi_probs=pi,
            )

        for market in ("home", "draw", "away"):
            edge = pi[market] - book[market]
            row = {
                "match_id": mid,
                "match": f"{match.get('home_team', '?')} vs {match.get('away_team', '?')}",
                "date": match.get("date"),
                "market": market,
                "pi_prob": round(pi[market], 4),
                "book_prob": round(book[market], 4),
                "edge": round(edge, 4),
            }
            if confidence is not None:
                row["confidence"] = confidence
            if edge >= min_edge:
                results.append(row)
    return sorted(results, key=lambda x: -x["edge"])
