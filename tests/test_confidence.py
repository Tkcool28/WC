"""Tests for the confidence assessment module.

This module wraps a pi-rating prediction with:
- A calibration lookup (raw pi% -> actual hit rate)
- A data-volume label (high/medium/low/insufficient)
- A combined tier (A/B/C/D) and human-readable warnings
- A rendered banner for the dashboard

The tier classification is what the dashboard will use to gate +EV signals.
"""
import pytest

from soccer_ev_model.confidence import (
    assess_match_confidence,
    calibration_lookup,
    calibration_confidence_label,
    matches_played_confidence_label,
    render_warning_banner,
    TIER_DESCRIPTIONS,
    CALIBRATION_TABLE,
    MIN_MATCHES_TRUSTED,
    MIN_MATCHES_HIGH_CONF,
)


# ---- calibration_lookup ----


def test_calibration_lookup_0p45_returns_high_confidence_band():
    """A raw pi of 0.45 (between 0.4 and 0.5) should map to ~0.458 hit rate."""
    assert calibration_lookup(0.45) == pytest.approx(0.458, abs=1e-6)


def test_calibration_lookup_0p85_corrects_overconfidence():
    """A raw pi of 0.85 (between 0.8 and 0.9) is heavily overconfident.
    Empirical hit rate is ~0.721."""
    assert calibration_lookup(0.85) == pytest.approx(0.721, abs=1e-6)


def test_calibration_lookup_0p7_maps_to_overconfident_band():
    """A raw pi of 0.70 (just at the boundary) should map to the 0.7-0.8 band."""
    assert calibration_lookup(0.70) == pytest.approx(0.646, abs=1e-6)


def test_calibration_lookup_returns_in_unit_interval():
    """All calibration values must be valid probabilities in (0, 1)."""
    for p in [0.05, 0.20, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.99]:
        v = calibration_lookup(p)
        assert 0.0 < v < 1.0, f"calibration_lookup({p}) = {v} out of range"


# ---- matches_played_confidence_label ----


def test_matches_played_label_150_is_high():
    """150 matches played should be 'high' (well above MIN_MATCHES_HIGH_CONF=100)."""
    assert matches_played_confidence_label(150) == "high"


def test_matches_played_label_50_is_medium():
    """50 matches played should be 'medium' (between 30 and 100)."""
    assert matches_played_confidence_label(50) == "medium"


def test_matches_played_label_15_is_low():
    """15 matches played should be 'low' (between 5 and 30)."""
    assert matches_played_confidence_label(15) == "low"


def test_matches_played_label_2_is_insufficient():
    """2 matches played should be 'insufficient' (< 5)."""
    assert matches_played_confidence_label(2) == "insufficient"


def test_matches_played_label_5_is_low_boundary():
    """5 matches played is the boundary of 'insufficient'; should be 'low'."""
    assert matches_played_confidence_label(5) == "low"


def test_matches_played_label_30_is_medium_boundary():
    """30 matches = MIN_MATCHES_TRUSTED boundary; should be 'medium'."""
    assert matches_played_confidence_label(MIN_MATCHES_TRUSTED) == "medium"


def test_matches_played_label_100_is_high_boundary():
    """100 matches = MIN_MATCHES_HIGH_CONF boundary; should be 'high'."""
    assert matches_played_confidence_label(MIN_MATCHES_HIGH_CONF) == "high"


# ---- assess_match_confidence ----


def test_assess_tier_a_for_high_data_and_even_matchup():
    """Both teams with lots of data AND balanced pi (top_p ~ 0.5):
    -> tier A (no warnings, fully trustable)."""
    assessment = assess_match_confidence(
        home_matches_played=150,
        away_matches_played=150,
        pi_probs={"home": 0.5, "draw": 0.25, "away": 0.25},
    )
    assert assessment["tier"] == "A"
    assert assessment["warnings"] == []
    assert assessment["edge_warning"] is False
    assert assessment["data_label"] == "high"
    assert assessment["calib_label"] == "high"
    assert assessment["top_p"] == 0.5
    assert assessment["calibrated_p"] == pytest.approx(0.532, abs=1e-6)


def test_assess_tier_d_for_insufficient_data():
    """Both teams with < 5 matches: tier D, insufficient data warning."""
    assessment = assess_match_confidence(
        home_matches_played=2,
        away_matches_played=3,
        pi_probs={"home": 0.5, "draw": 0.25, "away": 0.25},
    )
    assert assessment["tier"] == "D"
    assert assessment["data_label"] == "insufficient"
    assert assessment["edge_warning"] is True
    # Must mention insufficient / <5 in the warning
    assert any("<5" in w or "insufficient" in w.lower() for w in assessment["warnings"])
    # The tier description should match TIER_DESCRIPTIONS
    assert assessment["tier_description"] == TIER_DESCRIPTIONS["D"]


def test_assess_tier_c_for_overconfident_pi_and_medium_data():
    """Medium data + top_p=0.75 (overconfidence band):
    -> tier C with both data and overconfidence warnings."""
    assessment = assess_match_confidence(
        home_matches_played=50,
        away_matches_played=80,
        pi_probs={"home": 0.75, "draw": 0.15, "away": 0.10},
    )
    assert assessment["tier"] == "C"
    assert assessment["edge_warning"] is True
    # Must have BOTH a data warning and a calibration warning
    assert any("data" in w.lower() or "matches" in w.lower() for w in assessment["warnings"])
    assert any("overconfident" in w.lower() for w in assessment["warnings"])


def test_assess_tier_b_for_just_medium_data():
    """Medium data (50 matches) but pi is in the well-calibrated band (top_p = 0.55):
    -> tier B (medium confidence)."""
    assessment = assess_match_confidence(
        home_matches_played=50,
        away_matches_played=80,
        pi_probs={"home": 0.55, "draw": 0.25, "away": 0.20},
    )
    assert assessment["tier"] == "B"
    assert assessment["edge_warning"] is False


def test_assess_returns_well_known_keys():
    """The assessment dict must have all the keys the dashboard expects."""
    assessment = assess_match_confidence(
        home_matches_played=10, away_matches_played=20,
        pi_probs={"home": 0.5, "draw": 0.3, "away": 0.2},
    )
    expected_keys = {
        "tier", "tier_description", "top_p", "calibrated_p",
        "calibration_diff", "calib_label", "data_label",
        "home_matches_played", "away_matches_played", "min_matches_played",
        "warnings", "edge_warning",
    }
    assert expected_keys <= set(assessment.keys())


def test_assess_calibration_diff_is_negative_for_overconfident_pick():
    """calibration_diff = calibrated_p - top_p. If pi is overconfident, diff is negative."""
    assessment = assess_match_confidence(
        home_matches_played=150, away_matches_played=150,
        pi_probs={"home": 0.80, "draw": 0.10, "away": 0.10},
    )
    # 0.80 -> calibrated 0.721, so diff = -0.079
    assert assessment["calibration_diff"] < 0


def test_assess_min_matches_picks_weaker_team():
    """data_label should be the WORST of the two teams' data labels."""
    # Home has 150, away has 15. Worst is "low" (15 matches).
    assessment = assess_match_confidence(
        home_matches_played=150, away_matches_played=15,
        pi_probs={"home": 0.55, "draw": 0.25, "away": 0.20},
    )
    assert assessment["data_label"] == "low"
    assert assessment["min_matches_played"] == 15


# ---- render_warning_banner ----


def test_banner_for_tier_a_has_no_warning_lines():
    """Tier A: the banner should exist but contain no warning text
    (the dashboard only shows the tier line)."""
    assessment = assess_match_confidence(
        home_matches_played=150, away_matches_played=150,
        pi_probs={"home": 0.5, "draw": 0.25, "away": 0.25},
    )
    banner = render_warning_banner(assessment)
    assert isinstance(banner, str)
    assert len(banner) > 0
    # No warning emoji (⚠️) when there are no warnings
    assert "⚠️" not in banner
    # The tier line itself should be present
    assert "Tier A" in banner


def test_banner_for_tier_c_has_warning_lines_and_emoji():
    """Tier C: banner must contain warning lines with the warning emoji."""
    assessment = assess_match_confidence(
        home_matches_played=50, away_matches_played=80,
        pi_probs={"home": 0.75, "draw": 0.15, "away": 0.10},
    )
    banner = render_warning_banner(assessment)
    assert "Tier C" in banner
    assert "⚠️" in banner
    # The tier emoji (orange circle) is in the tier line
    assert "🟠" in banner


def test_banner_for_tier_d_has_warning_lines_and_red_emoji():
    """Tier D: banner has the red emoji + warning lines."""
    assessment = assess_match_confidence(
        home_matches_played=2, away_matches_played=3,
        pi_probs={"home": 0.5, "draw": 0.25, "away": 0.25},
    )
    banner = render_warning_banner(assessment)
    assert "Tier D" in banner
    assert "🔴" in banner
    assert "⚠️" in banner


def test_banner_uses_correct_emoji_per_tier():
    """The banner should use a distinct emoji for each tier."""
    pi = {"home": 0.5, "draw": 0.25, "away": 0.25}
    a = assess_match_confidence(150, 150, pi)  # calib=high, data=high -> A
    b = assess_match_confidence(150, 150, {"home": 0.65, "draw": 0.20, "away": 0.15})  # calib=medium, data=high -> B
    c = assess_match_confidence(50, 80, {"home": 0.75, "draw": 0.15, "away": 0.10})  # calib=low, data=medium -> C
    d = assess_match_confidence(2, 3, pi)  # data=insufficient -> D
    assert "🟢" in render_warning_banner(a)
    assert "🟡" in render_warning_banner(b)
    assert "🟠" in render_warning_banner(c)
    assert "🔴" in render_warning_banner(d)
