"""
Pure, testable helpers for per-match prediction summaries.

These functions consume plain dicts/strings/floats — no I/O, no Streamlit,
no new dependencies.  Called by the dashboard renderer to produce the
"Prediction summary" block shown below the existing confidence banner.

NOTE: these helpers operate on RAW model probabilities (the blend).
Calibration is an EV-layer concern (see `ev_workflow.evaluate_match` ->
`calibrated_pi` -> `plus_ev_flags`) and is NOT applied to the summary.
"""
from __future__ import annotations

import math


def top_two_outcomes(
    blended_probs: dict[str, float],
) -> tuple[str, float, str, float]:
    """Return (top_market, top_p, second_market, second_p).

    Tiebreak order: home > draw > away (i.e. if two markets have the same
    probability, the one earlier in ``("home", "draw", "away")`` wins).
    """
    order = ("home", "draw", "away")
    sorted_markets = sorted(
        order, key=lambda m: (-blended_probs[m], order.index(m))
    )
    top = sorted_markets[0]
    second = sorted_markets[1]
    return top, blended_probs[top], second, blended_probs[second]


def prediction_margin_pct(blended_probs: dict[str, float]) -> float:
    """(top − second) in percentage points, e.g. 18.4 for +18.4 pts."""
    top, top_p, _second, second_p = top_two_outcomes(blended_probs)
    return round((top_p - second_p) * 100, 1)


def draw_risk_label(draw_p: float) -> tuple[str, float]:
    """Classify draw probability into a risk bucket.

    Rules (boundary-inclusive on the low end):
      < 0.22  → "Low"
      < 0.29  → "Normal"
      >= 0.29 → "High"

    Returns ``(label, draw_p)``.
    """
    if draw_p < 0.22:
        return "Low", draw_p
    if draw_p < 0.29:
        return "Normal", draw_p
    return "High", draw_p


def _top_market(probs: dict[str, float]) -> str:
    """Return the market key with the highest probability (home > draw > away tiebreak)."""
    order = ("home", "draw", "away")
    return max(order, key=lambda m: (probs[m], -order.index(m)))


def model_agreement(
    pi_probs: dict[str, float],
    elo_probs: dict[str, float],
) -> dict:
    """Compare Pi-rating and Elo top picks.

    Returns a dict with:
      pi_top:         str  — home / draw / away
      elo_top:        str
      same_top:       bool — True if both models pick the same market
      fragile:        bool — same top but probability gap >= 10pp
      pi_p_at_top:    float — pi_probs[pi_top]
      elo_p_at_top:   float — elo_probs[elo_top]
      label:          "agree" | "disagree" | "fragile"
    """
    pi_top = _top_market(pi_probs)
    elo_top = _top_market(elo_probs)

    pi_p = pi_probs[pi_top]
    elo_p = elo_probs[elo_top]

    same_top = pi_top == elo_top
    prob_gap = abs(pi_p - elo_p) * 100  # in percentage points
    fragile = same_top and prob_gap >= 10.0

    if not same_top:
        label = "disagree"
    elif fragile:
        label = "fragile"
    else:
        label = "agree"

    return {
        "pi_top": pi_top,
        "elo_top": elo_top,
        "same_top": same_top,
        "fragile": fragile,
        "pi_p_at_top": pi_p,
        "elo_p_at_top": elo_p,
        "label": label,
    }


def confidence_tier(
    blended_probs: dict[str, float],
    prediction_margin_pts: float,
    draw_p: float,
    agreement_label: str,
    *,
    low_data: bool,
    identity_unresolved: bool = False,
    blend_is_pure_pi: bool = False,
) -> str:
    """Apply tier rules in order and return a human label.

    One of:
      "Identity unresolved" | "Low-data warning" | "Model disagreement"
      | "Toss-up" | "Strong favorite" | "Lean favorite" | "Draw lean"

    Rules are applied in the exact order specified — first match wins.
    """
    top_market, top_p, _second_market, _second_p = top_two_outcomes(blended_probs)

    # 1. Identity unresolved wins first — the team couldn't even be
    #    resolved through the canonical registry, so the prediction is
    #    unsafe regardless of how clean the pi-rating tier looks.
    if identity_unresolved:
        return "Identity unresolved"

    # 2. Low-data always wins second
    if low_data:
        return "Low-data warning"

    # 3. Model disagreement
    if agreement_label == "disagree":
        return "Model disagreement"

    # 4. Toss-up by margin
    if prediction_margin_pts < 8:
        return "Toss-up"

    # 5. Strong favorite
    if (
        top_market in ("home", "away")
        and top_p >= 0.60
        and prediction_margin_pts >= 20
        and draw_p < 0.29
        and agreement_label == "agree"
    ):
        return "Strong favorite"

    # 6. Lean favorite
    if (
        top_market in ("home", "away")
        and prediction_margin_pts >= 8
    ):
        return "Lean favorite"

    # 7. Draw lean
    if top_market == "draw" and prediction_margin_pts >= 8:
        return "Draw lean"

    # 8. Catch-all
    return "Toss-up"


# --------------------------------------------------------------------------- #
# Market baseline helpers (Phase 1: model vs book no-vig comparison)
# --------------------------------------------------------------------------- #
# These helpers expose the no-vig book fair probabilities as a visible
# "Market baseline" alongside the model blend. They are pure (no I/O, no
# Streamlit) and operate on plain dicts/strings/floats only.

_REQUIRED_MARKETS = ("home", "draw", "away")


def calculate_market_deltas(
    model_probs: dict[str, float],
    market_probs: dict[str, float],
) -> dict[str, float]:
    """Return per-market deltas in PERCENTAGE POINTS.

    Each value is ``(model_prob - market_prob) * 100`` rounded to 1 decimal.
    A negative value means the model is LESS confident than the market on
    that outcome.

    Asserts both inputs have exactly the keys ``{"home","draw","away"}``.
    Raises ``ValueError`` otherwise.
    """
    expected = set(_REQUIRED_MARKETS)
    for label, probs in (("model_probs", model_probs), ("market_probs", market_probs)):
        if set(probs.keys()) != expected:
            raise ValueError(
                f"{label} must have exactly keys {sorted(expected)}, "
                f"got {sorted(probs.keys())}"
            )
    return {
        m: round((model_probs[m] - market_probs[m]) * 100, 1)
        for m in _REQUIRED_MARKETS
    }


def market_divergence_label(deltas: dict[str, float]) -> str:
    """Classify the maximum absolute delta into a human-readable label.

    Boundaries (lower-bound inclusive):
        max(|delta|) <  0.03   → 'Strong market agreement'
        max(|delta|) <  0.07   → 'Moderate market agreement'
        max(|delta|) <  0.12   → 'Model divergence'
        max(|delta|) >= 0.12   → 'Major model divergence'
    """
    max_abs = max(abs(deltas[m]) for m in _REQUIRED_MARKETS)
    if max_abs < 0.03:
        return "Strong market agreement"
    if max_abs < 0.07:
        return "Moderate market agreement"
    if max_abs < 0.12:
        return "Model divergence"
    return "Major model divergence"


def largest_market_delta(
    deltas: dict[str, float],
    market_labels: dict[str, str] | None = None,
    *,
    model_probs: dict[str, float] | None = None,
    market_probs: dict[str, float] | None = None,
) -> dict:
    """Return the outcome with the largest absolute delta (strongest disagreement).

    Returns a dict with:
        market:     'home' | 'draw' | 'away'
        delta_pts:  float — signed, model - market, in percentage points (1 dp)
        model_pct:  float — model probability at the chosen market, 0-100 (1 dp);
                    ``None`` if ``model_probs`` was not supplied
        market_pct: float — market probability at the chosen market, 0-100 (1 dp);
                    ``None`` if ``market_probs`` was not supplied
        label:      str  — included only when ``market_labels`` is provided

    Tiebreak: home > draw > away (per ``_REQUIRED_MARKETS``).
    """
    # Tiebreak order: home > draw > away.  We sort by (-abs_delta, order)
    # so the first element is the chosen market.
    order = _REQUIRED_MARKETS
    chosen = min(
        order,
        key=lambda m: (-abs(deltas[m]), order.index(m)),
    )
    result: dict = {
        "market": chosen,
        "delta_pts": round(deltas[chosen], 1),
        "model_pct": round(model_probs[chosen] * 100, 1) if model_probs is not None else None,
        "market_pct": round(market_probs[chosen] * 100, 1) if market_probs is not None else None,
    }
    if market_labels is not None:
        result["label"] = market_labels.get(chosen, chosen)
    return result


def resolve_model_probs_for_market(evaluate_match_result: dict) -> dict[str, float]:
    """Pick the model probability dict to compare against the market.

    Prefers ``result['blend_probs']`` (the explicit Pi+Elo blend alias),
    falls back to ``result['pi_probs']`` for backward compatibility with
    results produced by older workflow versions.  Raises ``KeyError`` if
    neither key is present so callers fail loudly rather than guessing.
    """
    if "blend_probs" in evaluate_match_result and evaluate_match_result["blend_probs"] is not None:
        return evaluate_match_result["blend_probs"]
    if "pi_probs" in evaluate_match_result and evaluate_match_result["pi_probs"] is not None:
        return evaluate_match_result["pi_probs"]
    raise KeyError(
        "evaluate_match result must contain 'blend_probs' or 'pi_probs'"
    )


# --------------------------------------------------------------------------- #
# Poisson goal-model layer (Phase 2: transparent expected-goals approximation)
# --------------------------------------------------------------------------- #
# These helpers expose a *secondary*, *non-ML* probability estimate derived
# from independent Poisson home/away goal counts.  The math is fully
# transparent: P(k | λ) = e^(-λ) * λ^k / k!, computed via log-space using
# only stdlib `math.exp`, `math.lgamma`, and `math.fsum`.  The output is
# shown ALONGSIDE the existing Pi+Elo blend; it does NOT modify the blend
# or any other evaluate_match output.  The intent is to surface a second,
# interpretable signal so the user can see when the two views agree or
# disagree.
#
# Inputs are validated for negativity and NaN; everything else (clamping,
# normalization) is the caller's responsibility per the helper docstrings.


def _poisson_log_pmf(k: int, lam: float) -> float:
    """Return log P(k | λ) = -λ + k*log(λ) - lgamma(k+1).

    Uses ``math.lgamma(k+1)`` for the log-factorial so we can keep the
    computation in log-space and avoid overflow for large k.  Caller is
    responsible for ensuring ``lam > 0`` and ``k >= 0``.
    """
    # Note: math.lgamma(0+1) == lgamma(1) == log(0!) == 0
    return -lam + k * math.log(lam) - math.lgamma(k + 1)


def _validate_xg(name: str, value: float) -> None:
    """Raise ``ValueError`` if ``value`` is negative or NaN.

    Used by ``poisson_score_matrix`` and ``poisson_outcome_probs`` to
    fail loudly on inputs that would silently produce nonsense (e.g. a
    negative rate makes the PMF undefined; NaN propagates everywhere).
    """
    if not math.isfinite(value):
        raise ValueError(
            f"{name} must be a finite number, got {value!r}"
        )
    if value < 0:
        raise ValueError(
            f"{name} must be non-negative, got {value!r}"
        )


def poisson_score_matrix(
    home_xg: float,
    away_xg: float,
    max_goals: int = 8,
) -> list[list[float]]:
    """Return a (max_goals+1) x (max_goals+1) matrix of P(home=i, away=j).

    Each cell ``M[i][j]`` is ``P(home=i) * P(away=j)`` under independent
    Poisson distributions with rates ``home_xg`` and ``away_xg``.  The
    sum of the matrix is **strictly less than 1.0** because we truncate
    at ``max_goals``; the caller (``poisson_outcome_probs``) is expected
    to renormalize.

    Math: ``P(k | λ) = exp(-λ + k*log(λ) - lgamma(k+1))`` (log-PMF,
    exponentiated).  No new dependencies: only ``math.exp``, ``math.log``,
    ``math.lgamma``.

    Parameters
    ----------
    home_xg, away_xg:
        Expected goals for each side.  Must be non-negative and finite.
    max_goals:
        Inclusive upper bound for the score grid.  Default 8 (matches
        the spec; loss of mass beyond is negligible at typical xG).
    """
    _validate_xg("home_xg", home_xg)
    _validate_xg("away_xg", away_xg)
    if max_goals < 0:
        raise ValueError(
            f"max_goals must be non-negative, got {max_goals!r}"
        )

    # Pre-compute the per-side PMFs once and reuse them across the grid.
    # Building them as a list of length max_goals+1 keeps the inner loop
    # to a single multiplication per cell.
    n = max_goals + 1
    home_pmf = [math.exp(_poisson_log_pmf(k, home_xg)) for k in range(n)]
    away_pmf = [math.exp(_poisson_log_pmf(k, away_xg)) for k in range(n)]

    return [
        [home_pmf[i] * away_pmf[j] for j in range(n)]
        for i in range(n)
    ]


def poisson_outcome_probs(
    home_xg: float,
    away_xg: float,
    max_goals: int = 8,
) -> dict:
    """Return ``{'home': p, 'draw': p, 'away': p}`` derived from the score matrix.

    Aggregates the independent-Poisson score matrix:
      - ``home`` = sum over all cells where i > j (home goals > away)
      - ``draw`` = sum over the diagonal (i == j)
      - ``away`` = sum over all cells where i < j

    The three values are non-negative and sum to **<= 1.0** (the
    remainder is the truncation tail beyond ``max_goals``).  If the sum
    is < 1.0, the three values are renormalized so they sum to 1.0.
    Renormalization is the standard practice for score-matrix models;
    the lost mass at typical xG values (e.g. 1.5 vs 0.8) is on the
    order of 1e-4 or smaller.

    Clamp negative/NaN inputs: if either ``home_xg`` or ``away_xg`` is
    negative or NaN, raise ``ValueError`` (see ``_validate_xg``).

    Parameters
    ----------
    home_xg, away_xg:
        Expected goals for each side.  Must be non-negative and finite.
    max_goals:
        Inclusive upper bound for the score grid.  Default 8.
    """
    matrix = poisson_score_matrix(home_xg, away_xg, max_goals=max_goals)
    n = max_goals + 1

    home_p = 0.0
    draw_p = 0.0
    away_p = 0.0
    for i in range(n):
        for j in range(n):
            cell = matrix[i][j]
            if i > j:
                home_p += cell
            elif i == j:
                draw_p += cell
            else:
                away_p += cell

    # Use math.fsum-style normalization so floating-point error doesn't
    # leave a residual when the matrix is nearly exhaustive.
    total = home_p + draw_p + away_p
    if total > 0 and total < 1.0:
        # Renormalize so the three values sum to exactly 1.0.  Scale
        # factor is 1/total; we keep this branch explicit so the
        # behaviour is obvious in code review.
        inv = 1.0 / total
        home_p *= inv
        draw_p *= inv
        away_p *= inv
    elif total > 1.0:
        # Numerical over-shoot (should not happen given a valid PMF,
        # but guard against it so we never return > 1.0 total).
        inv = 1.0 / total
        home_p *= inv
        draw_p *= inv
        away_p *= inv
    # else: total == 0 (both lambdas are exactly 0) — leave all three
    # at 0.0; the caller can decide what to display.

    return {"home": home_p, "draw": draw_p, "away": away_p}


def expected_goals_from_blend(
    blend_probs: dict,
    base_total: float = 2.55,
    edge_scale: float = 2.2,
    max_edge: float = 2.2,
    min_xg: float = 0.2,
    max_xg: float = 4.0,
) -> dict:
    """Translate a blend 1X2 distribution into expected home/away goals.

    The mapping is intentionally simple and *transparent*: we treat the
    strength edge (``blend['home'] - blend['away']``) as a linear
    indicator of the goal differential, then split the base total
    between the two sides.  This is the standard "Poisson from 1X2"
    back-of-the-envelope technique used in football modelling for
    decades; it lets us derive xG without re-fitting the model and
    keeps the Poisson layer interpretable.

    Parameters
    ----------
    blend_probs:
        ``{'home': float, 'draw': float, 'away': float}``.  Sum should
        be ~1.0; we do not renormalize here.
    base_total:
        Baseline total goals.  Default 2.55 per the spec (roughly the
        long-run average for senior international football).
    edge_scale:
        Multiplier on the strength edge.  Default 2.2 per the spec.
    max_edge:
        Clamp on ``|goal_diff|``.  Default 2.2 per the spec — keeps
        extreme blends from producing absurd xG values.
    min_xg, max_xg:
        Per-side clamps on the resulting xG values.  Defaults 0.2 / 4.0.

    Returns
    -------
    dict with keys:
      - ``home_xg`` (float, 3 dp)
      - ``away_xg`` (float, 3 dp)
      - ``strength_edge`` (float, 3 dp) — ``blend['home'] - blend['away']``
      - ``goal_diff`` (float, 3 dp) — clamped ``strength_edge * edge_scale``

    All output floats are rounded to 3 decimals for display stability.
    """
    strength_edge = blend_probs["home"] - blend_probs["away"]
    raw_diff = strength_edge * edge_scale
    if raw_diff > max_edge:
        goal_diff = max_edge
    elif raw_diff < -max_edge:
        goal_diff = -max_edge
    else:
        goal_diff = raw_diff

    raw_home = (base_total + goal_diff) / 2.0
    raw_away = (base_total - goal_diff) / 2.0
    if raw_home > max_xg:
        home_xg = max_xg
    elif raw_home < min_xg:
        home_xg = min_xg
    else:
        home_xg = raw_home
    if raw_away > max_xg:
        away_xg = max_xg
    elif raw_away < min_xg:
        away_xg = min_xg
    else:
        away_xg = raw_away

    return {
        "home_xg": round(home_xg, 3),
        "away_xg": round(away_xg, 3),
        "strength_edge": round(strength_edge, 3),
        "goal_diff": round(goal_diff, 3),
    }


def poisson_agreement_label(
    blend_probs: dict,
    poisson_probs: dict,
) -> dict:
    """Return ``{'blend_top', 'poisson_top', 'agrees', 'label'}`` for the two views.

    ``blend_top`` is the top market under the Pi+Elo blend; ``poisson_top``
    is the top market under the independent-Poisson expected-goals model.
    ``agrees`` is True iff they match.  ``label`` is the human-readable
    ``'agrees'`` or ``'disagrees'`` string.  Reuses ``_top_market`` from
    this module so the tiebreak order stays consistent with the rest of
    the helpers (home > draw > away).
    """
    blend_top = _top_market(blend_probs)
    poisson_top = _top_market(poisson_probs)
    agrees = blend_top == poisson_top
    return {
        "blend_top": blend_top,
        "poisson_top": poisson_top,
        "agrees": agrees,
        "label": "agrees" if agrees else "disagrees",
    }


# --------------------------------------------------------------------------- #
# Phase 3 — group-stage context warnings
# --------------------------------------------------------------------------- #
#
# Pure, read-only helpers.  The dashboard renderer passes the source-match
# metadata (stage, matchday, group, finished matches in the group) and gets
# back an ordered list of warning dicts.  The model probabilities are
# never read or modified by anything in this block — these helpers are a
# display-only layer above the existing pi+Elo blend, market baseline, and
# Poisson views.
#
# The output schema for every helper in this section is documented inline.
# Forbidden vocabulary (per project policy): bet, wager, play, lock, stake,
# bankroll.  Allowed phrasing: "context warning", "warning only", "context
# only", "rotation", "qualification", "draw risk", "draw-sensitive".


_KNOWN_GROUP_STAGES = frozenset({"GROUP_STAGE"})
_KNOWN_KNOCKOUT_STAGES = frozenset(
    {
        "LAST_32",
        "LAST_16",
        "QUARTER_FINALS",
        "SEMI_FINALS",
        "THIRD_PLACE",
        "FINAL",
    }
)


def matchday_label(stage: str, matchday: int | None) -> dict:
    """Return ``{'label', 'severity', 'is_group_stage', 'matchday', 'is_final_group_match'}``.

    ``stage`` is one of the FIFA-style stage codes (``'GROUP_STAGE'``,
    ``'LAST_16'``, ``'QUARTER_FINALS'``, ``'SEMI_FINALS'``, ``'THIRD_PLACE'``,
    ``'FINAL'``, ``'LAST_32'``) or empty/unknown.  ``matchday`` is ``1``,
    ``2``, or ``3`` for the group stage and ``None`` for knockout.

    Severity buckets:

    * ``'info'`` — no special action; the model output is fine to read.
    * ``'warning'`` — model probabilities should be read with extra
      caution (final group match, where rotation/qualification
      incentives can dominate team strength).

    The function never raises: it always returns a valid dict with the
    documented keys.  Unknown stages fall through to the
    ``'Unknown stage'`` label, and an out-of-range matchday (e.g. ``0``
    or ``4``) inside a known group stage falls through to the generic
    ``'Group-stage match'`` label.
    """
    is_group_stage = stage in _KNOWN_GROUP_STAGES
    is_knockout = stage in _KNOWN_KNOCKOUT_STAGES

    if is_group_stage:
        if matchday == 1:
            return {
                "label": "Opening group match",
                "severity": "info",
                "is_group_stage": True,
                "matchday": 1,
                "is_final_group_match": False,
            }
        if matchday == 2:
            return {
                "label": "Second group match",
                "severity": "info",
                "is_group_stage": True,
                "matchday": 2,
                "is_final_group_match": False,
            }
        if matchday == 3:
            return {
                "label": "Final group match",
                "severity": "warning",
                "is_group_stage": True,
                "matchday": 3,
                "is_final_group_match": True,
            }
        # Group stage but matchday unknown / out of [1, 2, 3].
        return {
            "label": "Group-stage match",
            "severity": "info",
            "is_group_stage": True,
            "matchday": matchday,
            "is_final_group_match": False,
        }

    if is_knockout:
        return {
            "label": "Knockout stage",
            "severity": "info",
            "is_group_stage": False,
            "matchday": None,
            "is_final_group_match": False,
        }

    # Empty string, None, or any other unknown stage value.
    return {
        "label": "Unknown stage",
        "severity": "info",
        "is_group_stage": False,
        "matchday": None,
        "is_final_group_match": False,
    }


def compute_group_standings(
    finished_matches: list[dict],
    team_ids_in_group: list[int] | None = None,
) -> dict[int, dict]:
    """Compute a points table from a list of finished match dicts.

    Each match dict must contain at least ``home_team_id``,
    ``away_team_id``, ``home_goals``, ``away_goals`` (ints).  An
    optional ``home_team_name`` / ``away_team_name`` is preserved into
    the per-team record so callers can render a one-line summary
    without a separate name lookup.  The order of ``finished_matches``
    is the canonical order (no date-based reordering).

    Returns a dict mapping ``team_id`` to::

        {
            'team_id': int,
            'name': str | None,
            'played': int,
            'wins': int,
            'draws': int,
            'losses': int,
            'gf': int,
            'ga': int,
            'gd': int,
            'points': int,
        }

    Iteration order is insertion order (Python 3.7+); callers that
    need a ranking should sort by ``(points desc, gd desc, gf desc,
    team_id asc)``.  The function never raises on empty input — it
    returns an empty dict.
    """
    if not finished_matches:
        return {}

    # Build a fresh table that doesn't alias any caller-owned object.
    table: dict[int, dict] = {}

    def _row(tid: int, name: str | None) -> dict:
        if tid not in table:
            table[tid] = {
                "team_id": tid,
                "name": name,
                "played": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "gf": 0,
                "ga": 0,
                "gd": 0,
                "points": 0,
            }
        elif name and not table[tid].get("name"):
            # First non-empty name wins, so we don't overwrite a known
            # name with a later None.
            table[tid]["name"] = name
        return table[tid]

    for m in finished_matches:
        h = m.get("home_team_id")
        a = m.get("away_team_id")
        hg = m.get("home_goals")
        ag = m.get("away_goals")
        # Defensive: skip rows that don't have the four ints we need.
        if not isinstance(h, int) or not isinstance(a, int):
            continue
        if not isinstance(hg, int) or not isinstance(ag, int):
            continue

        h_name = m.get("home_team_name") if isinstance(m.get("home_team_name"), str) else None
        a_name = m.get("away_team_name") if isinstance(m.get("away_team_name"), str) else None

        home = _row(h, h_name)
        away = _row(a, a_name)

        home["played"] += 1
        away["played"] += 1
        home["gf"] += hg
        home["ga"] += ag
        away["gf"] += ag
        away["ga"] += hg

        if hg > ag:
            home["wins"] += 1
            home["points"] += 3
            away["losses"] += 1
        elif hg < ag:
            away["wins"] += 1
            away["points"] += 3
            home["losses"] += 1
        else:
            home["draws"] += 1
            away["draws"] += 1
            home["points"] += 1
            away["points"] += 1

    for row in table.values():
        row["gd"] = row["gf"] - row["ga"]

    # If the caller passed an explicit team roster, ensure those teams
    # appear in the output (with zeros) even if they haven't played yet.
    if team_ids_in_group:
        for tid in team_ids_in_group:
            if not isinstance(tid, int):
                continue
            _row(tid, None)

    return table


def _format_standings_text(
    standings: dict[int, dict],
    group: str | None,
) -> str:
    """Build the one-line standings summary text for a ``standings`` warning.

    Renders a compact "Team X 3pts (+2), Team Y 1pt (0), Team Z 0pt (-2)"
    line for the team_ids present in ``standings`` (already filtered by
    the caller to "teams that have played").  The exact ordering matches
    the ranking rule documented in ``compute_group_standings``.
    """
    if not standings:
        return "Current group standings (context only): no matches played."

    ranked = sorted(
        standings.values(),
        key=lambda r: (-r["points"], -r["gd"], -r["gf"], r["team_id"]),
    )
    parts: list[str] = []
    for r in ranked:
        name = r.get("name") or f"Team {r['team_id']}"
        pts = r["points"]
        pts_word = "pt" if pts == 1 else "pts"
        gd = r["gd"]
        gd_sign = "+" if gd > 0 else ""  # negative already has its sign
        parts.append(f"{name} {pts}{pts_word} ({gd_sign}{gd})")
    prefix = f"Current group standings (context only) [{group}]:" if group else "Current group standings (context only):"
    return f"{prefix} " + ", ".join(parts) + "."


def group_context_warnings(
    stage: str,
    matchday: int | None,
    group: str | None = None,
    *,
    finished_matches_in_group: list[dict] | None = None,
) -> list[dict]:
    """Return an ordered list of group-context warnings for a match.

    Each warning is a dict ``{'text': str, 'severity': 'info'|'warning', 'tag': str}``.

    Tags used: ``'group_stage'``, ``'opening'``, ``'second'``, ``'final'``,
    ``'rotation'``, ``'qualification'``, ``'draw_sensitive'``, ``'standings'``,
    ``'no_data'``.  The list is ordered most-actionable first.

    Rules:

    * If ``stage`` is not ``'GROUP_STAGE'`` (or is empty/unknown), the
      function returns an empty list — there is no group warning to give.
    * ``matchday == 1`` → one info warning about normal opening
      incentives.
    * ``matchday == 2`` → one info warning about pressure depending on
      the first result.
    * ``matchday == 3`` → three warnings: a high-priority
      severity=warning about final-match rotation, a follow-up warning
      reminding the user the model doesn't account for rotation /
      qualification / draw-is-enough scenarios, and an info note that
      draw risk is context-sensitive.
    * ``matchday is None`` or out of ``[1, 2, 3]`` for a group stage → at
      minimum a ``'group_stage'`` info warning.
    * If ``finished_matches_in_group`` is a non-empty list, compute
      standings with ``compute_group_standings`` and append an info
      ``'standings'`` warning with a one-line summary.
    * If a ``group`` is supplied but ``finished_matches_in_group`` is
      empty / None, append a ``'no_data'`` info warning.

    The function is pure: no I/O, no Streamlit, no ``Date.now``, no
    random, no calls to ``evaluate_match``.  Two calls with the same
    arguments return identical output.
    """
    if stage not in _KNOWN_GROUP_STAGES:
        return []

    warnings: list[dict] = []

    if matchday == 1:
        warnings.append({
            "text": "Opening group match — normal incentives",
            "severity": "info",
            "tag": "opening",
        })
    elif matchday == 2:
        warnings.append({
            "text": "Second group match — pressure depends on first result",
            "severity": "info",
            "tag": "second",
        })
    elif matchday == 3:
        warnings.append({
            "text": "Final group match — motivation/rotation risk can be high",
            "severity": "warning",
            "tag": "final",
        })
        warnings.append({
            "text": (
                "Model probabilities do not account for rotation, "
                "qualification incentives, or draw-is-enough scenarios."
            ),
            "severity": "warning",
            "tag": "rotation",
        })
        warnings.append({
            "text": "Draw risk may be context-sensitive",
            "severity": "info",
            "tag": "draw_sensitive",
        })
    else:
        # Group stage but no usable matchday.
        warnings.append({
            "text": "Group-stage match (matchday unknown) — group context only",
            "severity": "info",
            "tag": "group_stage",
        })

    if finished_matches_in_group:
        # Only include teams that have actually played (played > 0); a
        # team that's in the table because the caller pre-populated
        # `team_ids_in_group` but hasn't played yet would just clutter
        # the one-line summary.
        standings = compute_group_standings(finished_matches_in_group)
        played = {tid: row for tid, row in standings.items() if row.get("played", 0) > 0}
        warnings.append({
            "text": _format_standings_text(played, group),
            "severity": "info",
            "tag": "standings",
        })
    elif group:
        # Group is set but we have no data — surface that explicitly so
        # the user knows the standings are unavailable, not "all zeros".
        warnings.append({
            "text": "No group matches yet played (or standings not loaded).",
            "severity": "info",
            "tag": "no_data",
        })

    return warnings
