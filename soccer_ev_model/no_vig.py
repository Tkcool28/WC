"""3-way no-vig probability utilities for soccer.

We work in two formats:
- American odds (e.g. -200, +550, +340) — most US-facing books
- Decimal odds (e.g. 1.50, 6.50, 4.40) — European format

For a 3-way (home / draw / away) market, the bookmaker's implied probabilities
sum to > 1.0 (the overround / vig). The no-vig probability removes the overround
proportionally across all three outcomes, so the fair probabilities sum to 1.0.

This is the standard "proportional" or "multiplicative" method. It is NOT the
only method (logarithmic / Shin / power methods exist), but it is the most
widely used in soccer and matches what the public soccer-modeling papers do.

The "model" is the user. We just provide the math. The user pastes in book odds
and gets back a clean fair-probability table to compare against the model's
own output.
"""

from __future__ import annotations


# ---- conversions ----


def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds (multiplier, including stake).

    Negative American odds (e.g. -200) -> decimal = 1 + 100/|odds|
    Positive American odds (e.g. +550) -> decimal = 1 + odds/100
    """
    if odds >= 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def american_to_implied(odds: float) -> float:
    """Convert American odds to the raw bookmaker-implied probability.

    This is the no-margin version. The book's 3-way market will sum these
    to > 1.0; the overage is the vig.
    """
    return 1.0 / american_to_decimal(odds)


def decimal_to_implied(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if decimal_odds <= 1.0:
        raise ValueError(f"Decimal odds must be > 1.0, got {decimal_odds}")
    return 1.0 / decimal_odds


# ---- 3-way no-vig ----


def remove_vig(home_odds: float, draw_odds: float, away_odds: float) -> dict:
    """Return fair (no-vig) probabilities for a 3-way market.

    Args:
        home_odds, draw_odds, away_odds: American odds for each outcome.

    Returns:
        dict with keys 'home', 'draw', 'away' mapping to fair probabilities
        in (0, 1) that sum to 1.0.
    """
    raw = {
        "home": american_to_implied(home_odds),
        "draw": american_to_implied(draw_odds),
        "away": american_to_implied(away_odds),
    }

    for label, odds in [
        ("home", home_odds),
        ("draw", draw_odds),
        ("away", away_odds),
    ]:
        if odds == 0:
            raise ValueError(f"{label} odds cannot be zero")
        if odds < -10000 or odds > 10000:
            raise ValueError(f"{label} odds out of sane range: {odds}")

    # All three negative is a degenerate market (no real book would post that).
    if all(o < 0 for o in [home_odds, draw_odds, away_odds]):
        raise ValueError(
            "All three 3-way outcomes cannot be negative American odds. "
            "At least one (the underdog) must be positive."
        )

    overround = sum(raw.values())
    if overround <= 1.0:
        # Already a no-vig market or has negative margin. Return raw, capped to (0, 1).
        return {k: max(0.0, min(1.0, v)) for k, v in raw.items()}

    return {k: v / overround for k, v in raw.items()}


def implied_probs(
    home_odds: float, draw_odds: float, away_odds: float
) -> dict:
    """Return both the raw book %s and the no-vig %s, plus the overround.

    Convenience wrapper around remove_vig() that also reports the raw
    (vig-included) implied probabilities and the total overround as a
    percentage.

    Returns:
        dict with keys:
          - 'raw':   raw implied probs (sum > 1.0)
          - 'fair':  no-vig probs (sum = 1.0)
          - 'vig_pct': total overround as a decimal (e.g. 0.048 = 4.8%)
    """
    raw = {
        "home": american_to_implied(home_odds),
        "draw": american_to_implied(draw_odds),
        "away": american_to_implied(away_odds),
    }
    fair = remove_vig(home_odds, draw_odds, away_odds)
    vig_pct = sum(raw.values()) - 1.0
    return {"raw": raw, "fair": fair, "vig_pct": vig_pct}
