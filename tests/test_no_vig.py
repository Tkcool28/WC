"""Tests for the no-vig (margin-removed) 3-way odds utility.

We support 3-way (home / draw / away) odds in two formats:
- American odds (e.g. -200, +550, +340) — most US-facing books
- Decimal odds (e.g. 1.50, 6.50, 4.40) — European format

The no-vig probability is the implied probability with the bookmaker's margin
(overround) removed proportionally. We use the proportional (multiplicative)
method, which is the standard for 3-way markets.
"""

import math
import pytest
from soccer_ev_model.no_vig import (
    american_to_decimal,
    american_to_implied,
    decimal_to_implied,
    remove_vig,
    implied_probs,
)


# ---- american_to_decimal ----


def test_american_to_decimal_negative():
    """Negative American odds: -200 means bet 200 to win 100, so decimal is 1.50."""
    assert math.isclose(american_to_decimal(-200), 1.50, abs_tol=1e-9)


def test_american_to_decimal_positive():
    """Positive American odds: +550 means bet 100 to win 550, so decimal is 6.50."""
    assert math.isclose(american_to_decimal(550), 6.50, abs_tol=1e-9)


# ---- american_to_implied ----


def test_american_to_implied_negative():
    """-200 -> 200/300 = 0.6667 implied probability."""
    assert math.isclose(american_to_implied(-200), 200 / 300, abs_tol=1e-9)


def test_american_to_implied_positive():
    """+550 -> 100/650 = 0.1538 implied probability."""
    assert math.isclose(american_to_implied(550), 100 / 650, abs_tol=1e-9)


def test_american_to_implied_inverts_decimal_minus_one():
    """For any odds, implied == 1 / decimal."""
    for odds in [-500, -200, -110, 100, 150, 550, 1500]:
        assert math.isclose(
            american_to_implied(odds),
            1 / american_to_decimal(odds),
            abs_tol=1e-9,
        )


# ---- decimal_to_implied ----


def test_decimal_to_implied():
    """Decimal 2.00 means even money -> 50% implied."""
    assert math.isclose(decimal_to_implied(2.00), 0.5, abs_tol=1e-9)


def test_decimal_to_implied_longshot():
    """Decimal 10.00 -> 10% implied."""
    assert math.isclose(decimal_to_implied(10.00), 0.1, abs_tol=1e-9)


# ---- remove_vig (3-way) ----


def test_remove_vig_balanced_market():
    """If a market is perfectly balanced, removing vig gives back the raw fractions.

    Home -110, Draw +300, Away +300: implied 0.524, 0.250, 0.250 = 1.024.
    Removing 2.4% vig evenly yields (0.512, 0.244, 0.244).
    """
    probs = remove_vig(-110, 300, 300)
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-9)
    # Home is still the favorite, but no longer by the overround.
    assert probs["home"] > probs["draw"]
    assert probs["home"] > probs["away"]


def test_remove_vig_heavy_favorite():
    """France vs Senegal from June 16: -200, +340, +550.

    Implied: 0.667, 0.227, 0.154. Sum = 1.048 (4.8% vig).
    No-vig: 0.637, 0.217, 0.147. Sum = 1.0.
    """
    probs = remove_vig(-200, 340, 550)
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-9)
    assert math.isclose(probs["home"], 0.667 / 1.048, abs_tol=1e-3)
    assert math.isclose(probs["draw"], 0.227 / 1.048, abs_tol=1e-3)
    assert math.isclose(probs["away"], 0.154 / 1.048, abs_tol=1e-3)


def test_remove_vig_does_not_invert_favorite_status():
    """Removing vig must not flip who is the favorite. France stays the pick."""
    probs = remove_vig(-200, 340, 550)
    assert probs["home"] > probs["draw"] > probs["away"]


# ---- implied_probs (the easy wrapper) ----


def test_implied_probs_returns_raw_and_fair():
    """implied_probs should return both the raw book %s and the no-vig %s."""
    result = implied_probs(-200, 340, 550)
    # Raw implied %s (with vig)
    assert "raw" in result
    assert math.isclose(result["raw"]["home"], 0.667, abs_tol=1e-3)
    # No-vig %s (sum to 1.0)
    assert "fair" in result
    assert math.isclose(sum(result["fair"].values()), 1.0, abs_tol=1e-9)
    # Vig (overround) is reported as a positive percentage
    assert "vig_pct" in result
    assert 0 < result["vig_pct"] < 0.20  # sane: 0-20% overround


def test_implied_probs_vig_is_positive_when_realistic():
    """A realistic 3-way market has 2-8% overround. Verify we get that range."""
    # Realistic WC opener line
    result = implied_probs(-200, 340, 550)
    assert 0.02 < result["vig_pct"] < 0.10, (
        f"Expected 2-10% vig, got {result['vig_pct']*100:.2f}%"
    )


# ---- validation ----


def test_remove_vig_rejects_zero_odds():
    """Zero or near-zero odds are nonsense. Must raise clearly."""
    with pytest.raises(ValueError):
        remove_vig(0, 300, 300)
    with pytest.raises(ValueError):
        remove_vig(-200, 0, 550)


def test_remove_vig_rejects_all_negative_odds():
    """In a 3-way market at most 2 can be negative (favorites). All 3 negative is bogus."""
    with pytest.raises(ValueError):
        remove_vig(-110, -110, -110)


# ---- common-sense safety ----


def test_remove_vig_output_in_unit_interval():
    """Each no-vig probability must be strictly between 0 and 1."""
    probs = remove_vig(-150, 250, 400)
    for k, v in probs.items():
        assert 0 < v < 1, f"{k} = {v} not in (0, 1)"
