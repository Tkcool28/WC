"""
Pure, testable helpers for per-match prediction summaries.

These functions consume plain dicts/strings/floats — no I/O, no Streamlit,
no new dependencies.  Called by the dashboard renderer to produce the
"Prediction summary" block shown below the existing confidence banner.
"""
from __future__ import annotations


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
