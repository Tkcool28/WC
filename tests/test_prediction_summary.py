"""Tests for the prediction_summary helpers.

All functions are pure — no I/O, no Streamlit — so these are straightforward
unit tests with parametrized and hand-crafted cases.
"""
import pytest

from soccer_ev_model.prediction_summary import (
    confidence_tier,
    draw_risk_label,
    model_agreement,
    prediction_margin_pct,
    top_two_outcomes,
)


# --------------------------------------------------------------------------- #
# top_two_outcomes
# --------------------------------------------------------------------------- #

def test_top_two_basic_home_top():
    top, top_p, second, second_p = top_two_outcomes(
        {"home": 0.55, "draw": 0.25, "away": 0.20}
    )
    assert top == "home" and top_p == 0.55
    assert second == "draw" and second_p == 0.25


def test_top_two_basic_away_top():
    top, top_p, second, second_p = top_two_outcomes(
        {"home": 0.20, "draw": 0.25, "away": 0.55}
    )
    assert top == "away" and top_p == 0.55
    assert second == "draw" and second_p == 0.25


def test_top_two_draw_is_second():
    """When draw is the second-highest, it should appear as second."""
    top, top_p, second, second_p = top_two_outcomes(
        {"home": 0.50, "draw": 0.32, "away": 0.18}
    )
    assert top == "home"
    assert second == "draw"
    assert second_p == 0.32


def test_top_two_tiebreak_all_equal():
    """home == draw == away → home wins (first in tiebreak order)."""
    top, _top_p, second, _second_p = top_two_outcomes(
        {"home": 0.33, "draw": 0.33, "away": 0.33}
    )
    assert top == "home"
    assert second == "draw"


def test_top_two_tiebreak_draw_vs_away():
    """draw == away and both > home → draw wins (draw before away)."""
    top, _top_p, second, _second_p = top_two_outcomes(
        {"home": 0.20, "draw": 0.40, "away": 0.40}
    )
    assert top == "draw"
    assert second == "away"


# --------------------------------------------------------------------------- #
# prediction_margin_pct
# --------------------------------------------------------------------------- #

def test_margin_basic():
    margin = prediction_margin_pct({"home": 0.55, "draw": 0.25, "away": 0.20})
    assert margin == pytest.approx(30.0, abs=0.1)


def test_margin_draw_second():
    """Margin where draw is second: {home:0.50, draw:0.32, away:0.18} → 18.0."""
    margin = prediction_margin_pct({"home": 0.50, "draw": 0.32, "away": 0.18})
    assert margin == pytest.approx(18.0, abs=0.1)


def test_margin_small():
    margin = prediction_margin_pct({"home": 0.35, "draw": 0.33, "away": 0.32})
    assert margin == pytest.approx(2.0, abs=0.1)


# --------------------------------------------------------------------------- #
# draw_risk_label
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("draw_p,expected", [
    (0.219, "Low"),       # just under 0.22
    (0.10, "Low"),
    (0.0, "Low"),
    (0.22, "Normal"),     # boundary: exactly 0.22
    (0.25, "Normal"),
    (0.289, "Normal"),    # just under 0.29
    (0.29, "High"),       # boundary: exactly 0.29
    (0.35, "High"),
    (0.50, "High"),
])
def test_draw_risk_boundaries(draw_p, expected):
    label, returned_p = draw_risk_label(draw_p)
    assert label == expected
    assert returned_p == draw_p


# --------------------------------------------------------------------------- #
# model_agreement
# --------------------------------------------------------------------------- #

def test_agree_clean():
    """Both pick home, gap < 10pp → agree."""
    result = model_agreement(
        {"home": 0.55, "draw": 0.25, "away": 0.20},
        {"home": 0.52, "draw": 0.28, "away": 0.20},
    )
    assert result["label"] == "agree"
    assert result["same_top"] is True
    assert result["fragile"] is False
    assert result["pi_top"] == "home"
    assert result["elo_top"] == "home"


def test_fragile_agreement():
    """Both pick home, gap >= 10pp → fragile."""
    result = model_agreement(
        {"home": 0.62, "draw": 0.20, "away": 0.18},
        {"home": 0.50, "draw": 0.28, "away": 0.22},
    )
    assert result["label"] == "fragile"
    assert result["same_top"] is True
    assert result["fragile"] is True
    assert result["pi_p_at_top"] == 0.62
    assert result["elo_p_at_top"] == 0.50


def test_disagree():
    """Pi picks home, Elo picks away → disagree."""
    result = model_agreement(
        {"home": 0.55, "draw": 0.25, "away": 0.20},
        {"home": 0.20, "draw": 0.25, "away": 0.55},
    )
    assert result["label"] == "disagree"
    assert result["same_top"] is False
    assert result["fragile"] is False
    assert result["pi_top"] == "home"
    assert result["elo_top"] == "away"


def test_agree_returns_probs():
    result = model_agreement(
        {"home": 0.60, "draw": 0.25, "away": 0.15},
        {"home": 0.58, "draw": 0.27, "away": 0.15},
    )
    assert result["pi_p_at_top"] == 0.60
    assert result["elo_p_at_top"] == 0.58


# --------------------------------------------------------------------------- #
# confidence_tier
# --------------------------------------------------------------------------- #

def test_strong_favorite():
    """Home top, 0.65, margin 22, draw 0.20, agree → Strong favorite."""
    tier = confidence_tier(
        {"home": 0.65, "draw": 0.20, "away": 0.15},
        prediction_margin_pts=22.0,
        draw_p=0.20,
        agreement_label="agree",
        low_data=False,
    )
    assert tier == "Strong favorite"


def test_lean_favorite():
    """Home top, 0.55, margin 12, draw 0.25, agree → Lean favorite."""
    tier = confidence_tier(
        {"home": 0.55, "draw": 0.25, "away": 0.20},
        prediction_margin_pts=12.0,
        draw_p=0.25,
        agreement_label="agree",
        low_data=False,
    )
    assert tier == "Lean favorite"


def test_toss_up_small_margin():
    """Margin < 8 → Toss-up."""
    tier = confidence_tier(
        {"home": 0.36, "draw": 0.33, "away": 0.31},
        prediction_margin_pts=5.0,
        draw_p=0.33,
        agreement_label="agree",
        low_data=False,
    )
    assert tier == "Toss-up"


def test_toss_up_catch_all():
    """With agree and margin exactly at boundary (7.9), falls through to Toss-up."""
    tier = confidence_tier(
        {"home": 0.40, "draw": 0.31, "away": 0.29},
        prediction_margin_pts=7.9,
        draw_p=0.31,
        agreement_label="agree",
        low_data=False,
    )
    assert tier == "Toss-up"


def test_draw_lean_with_margin_12():
    """Draw top at 0.36, margin 12 → Draw lean (rule 6)."""
    tier = confidence_tier(
        {"home": 0.30, "draw": 0.36, "away": 0.34},
        prediction_margin_pts=12.0,
        draw_p=0.36,
        agreement_label="agree",
        low_data=False,
    )
    assert tier == "Draw lean"


def test_model_disagreement():
    """Pi picks home, Elo picks away → Model disagreement."""
    tier = confidence_tier(
        {"home": 0.55, "draw": 0.25, "away": 0.20},
        prediction_margin_pts=12.0,
        draw_p=0.25,
        agreement_label="disagree",
        low_data=False,
    )
    assert tier == "Model disagreement"


def test_fragile_agreement_tier():
    """Fragile agreement is NOT disagree — falls through to margin/toss-up rules."""
    tier = confidence_tier(
        {"home": 0.55, "draw": 0.25, "away": 0.20},
        prediction_margin_pts=12.0,
        draw_p=0.25,
        agreement_label="fragile",
        low_data=False,
    )
    # fragile != "disagree", margin >= 8, top in (home,away) → Lean favorite
    assert tier == "Lean favorite"


def test_draw_lean():
    """Draw top, 0.42, margin 10 → Draw lean."""
    tier = confidence_tier(
        {"home": 0.30, "draw": 0.42, "away": 0.28},
        prediction_margin_pts=10.0,
        draw_p=0.42,
        agreement_label="agree",
        low_data=False,
    )
    assert tier == "Draw lean"


def test_low_data_warning_overrides_strong_favorite():
    """Low-data fires first even when all strong-favorite conditions are met."""
    tier = confidence_tier(
        {"home": 0.65, "draw": 0.20, "away": 0.15},
        prediction_margin_pts=22.0,
        draw_p=0.20,
        agreement_label="agree",
        low_data=True,
    )
    assert tier == "Low-data warning"


def test_model_disagreement_before_toss_up():
    """Disagree fires before toss-up even when margin < 8."""
    tier = confidence_tier(
        {"home": 0.35, "draw": 0.33, "away": 0.32},
        prediction_margin_pts=3.0,
        draw_p=0.33,
        agreement_label="disagree",
        low_data=False,
    )
    assert tier == "Model disagreement"


def test_strong_favorite_requires_agree():
    """Strong favorite requires agree — fragile should NOT qualify."""
    tier = confidence_tier(
        {"home": 0.65, "draw": 0.20, "away": 0.15},
        prediction_margin_pts=22.0,
        draw_p=0.20,
        agreement_label="fragile",
        low_data=False,
    )
    # fragile != "agree" so rule 4 doesn't match; rule 5 (lean) does
    assert tier == "Lean favorite"


def test_strong_favorite_requires_draw_under_29():
    """draw_p >= 0.29 disqualifies Strong favorite."""
    tier = confidence_tier(
        {"home": 0.65, "draw": 0.29, "away": 0.06},
        prediction_margin_pts=36.0,
        draw_p=0.29,
        agreement_label="agree",
        low_data=False,
    )
    # draw_p is NOT < 0.29, so rule 4 fails → falls to rule 5 (lean)
    assert tier == "Lean favorite"
