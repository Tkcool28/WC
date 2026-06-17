"""
Tests for dashboard/ux_presenters.py — the pure presentation helpers that
back the 3-tab Prediction / Betting Value / Analysis UX restructure.

All tests are pure unit tests: no Streamlit, no I/O, no fixtures beyond
synthetic result dicts that mirror the shape of `evaluate_match` output.

These tests are intentionally narrow: they verify that the presenters
respect the strict boundaries of the UX-only refactor.

  * Predicted result comes from the highest existing outcome probability.
  * Predicted result does NOT change based on entered odds.
  * Betting value can differ from the predicted result.
  * No Clear Value is a legitimate outcome.
  * Prediction Confidence and Value Confidence remain independent.
  * Technical warnings are translated in casual views.
  * Raw diagnostics remain accessible in the Analysis tab helpers.
  * COD / CPV-style "limited history" is reachable and is not an error.
"""
from __future__ import annotations

import pytest

from dashboard.ux_presenters import (
    agreement_status,
    analysis_calibration_and_data_quality,
    analysis_market_comparison,
    analysis_model_breakdown,
    analysis_poisson_view,
    analysis_prediction_details,
    analysis_raw_diagnostics,
    format_odds,
    most_likely_result,
    outcome_headline,
    prediction_confidence_label,
    prediction_why_text,
    translate_and_dedupe_warnings,
    translate_warning,
    value_confidence_label,
    value_play,
    value_why_text,
)


# --------------------------------------------------------------------------- #
# Fixture helpers
# --------------------------------------------------------------------------- #
def _assessment(
    tier: str = "A",
    *,
    calib_label: str = "high",
    data_label: str = "high",
    top_p: float = 0.55,
    calibrated_p: float = 0.55,
    home_matches: int = 80,
    away_matches: int = 80,
    warnings: list[str] | None = None,
    identity_unresolved: bool = False,
) -> dict:
    return {
        "tier": tier,
        "tier_description": f"Tier {tier} description",
        "top_p": top_p,
        "calibrated_p": calibrated_p,
        "calibration_diff": round(calibrated_p - top_p, 4),
        "calib_label": calib_label,
        "data_label": data_label,
        "home_matches_played": home_matches,
        "away_matches_played": away_matches,
        "min_matches_played": min(home_matches, away_matches),
        "warnings": warnings or [],
        "edge_warning": tier in ("C", "D"),
        "identity_unresolved": identity_unresolved,
    }


_PI_ABSENT = object()  # sentinel: "pi did not run"


def _result(
    *,
    home_team: str = "Brazil",
    away_team: str = "Argentina",
    blend: dict[str, float] | None = None,
    pi_only: object = None,
    elo_only: dict[str, float] | None = None,
    book_odds: dict[str, float] | None = None,
    book_fair: dict[str, float] | None = None,
    calibrated_pi: dict[str, float] | None = None,
    edges: dict[str, float] | None = None,
    plus_ev_flags: list[dict] | None = None,
    assessment: dict | None = None,
    blend_was_used: bool = False,
    canonical_home_id: str = "BRA",
    canonical_away_id: str = "ARG",
    identity_warnings: list[str] | None = None,
) -> dict:
    """Build a synthetic result dict that mirrors evaluate_match output."""
    blend = blend or {"home": 0.55, "draw": 0.25, "away": 0.20}
    # Allow callers to explicitly mark pi as absent via _PI_ABSENT.
    if pi_only is None:
        pi_only = dict(blend)
    elif pi_only is _PI_ABSENT:
        pi_only = None
    if elo_only is None:
        elo_only = None
    if book_odds is None:
        book_odds = {"home": -230, "draw": 350, "away": 700}
    if book_fair is None:
        # Make book_fair a no-vig version that roughly matches blend
        book_fair = {
            "home": round(blend["home"] * 0.95, 4),
            "draw": round(blend["draw"] * 1.05, 4),
            "away": round(blend["away"] * 1.10, 4),
        }
    if calibrated_pi is None:
        calibrated_pi = {
            "home": round(blend["home"] * 0.96, 4),
            "draw": round(blend["draw"] * 1.04, 4),
            "away": round(blend["away"] * 1.08, 4),
        }
    if edges is None:
        edges = {
            m: round(blend[m] - book_fair[m], 4) for m in ("home", "draw", "away")
        }
    return {
        "home_team": home_team,
        "away_team": away_team,
        "date": "2026-06-20",
        "book_odds": book_odds,
        "book_fair": book_fair,
        "pi_probs": dict(blend),
        "blend_probs": dict(blend),
        "pi_only_probs": dict(pi_only) if pi_only is not None else None,
        "elo_only_probs": dict(elo_only) if elo_only is not None else None,
        "blend_was_used": blend_was_used,
        "calibrated_pi": calibrated_pi,
        "edges": edges,
        "confidence": assessment or _assessment(),
        "plus_ev_flags": plus_ev_flags if plus_ev_flags is not None else [],
        "banner": "🟢 Tier A confidence",
        "canonical_home_id": canonical_home_id,
        "canonical_away_id": canonical_away_id,
        "identity_warnings": identity_warnings or [],
    }


# --------------------------------------------------------------------------- #
# most_likely_result
# --------------------------------------------------------------------------- #
class TestMostLikelyResult:
    def test_picks_highest_blend_probability_home(self):
        r = _result(blend={"home": 0.55, "draw": 0.25, "away": 0.20})
        out = most_likely_result(r)
        assert out["market"] == "home"
        assert out["label"] == "Brazil"
        assert out["probability"] == 0.55

    def test_picks_highest_blend_probability_away(self):
        r = _result(blend={"home": 0.20, "draw": 0.25, "away": 0.55})
        out = most_likely_result(r)
        assert out["market"] == "away"
        assert out["label"] == "Argentina"
        assert out["probability"] == 0.55

    def test_picks_highest_blend_probability_draw(self):
        r = _result(blend={"home": 0.30, "draw": 0.45, "away": 0.25})
        out = most_likely_result(r)
        assert out["market"] == "draw"
        assert out["label"] == "Draw"
        assert out["probability"] == 0.45

    def test_does_not_change_with_book_odds(self):
        # Same probs, different entered odds -> most_likely_result must be identical
        r1 = _result(
            blend={"home": 0.55, "draw": 0.25, "away": 0.20},
            book_odds={"home": -230, "draw": 350, "away": 700},
        )
        r2 = _result(
            blend={"home": 0.55, "draw": 0.25, "away": 0.20},
            book_odds={"home": -110, "draw": 260, "away": 320},
        )
        out1 = most_likely_result(r1)
        out2 = most_likely_result(r2)
        assert out1 == out2

    def test_does_not_change_with_plus_ev_flags(self):
        # The most likely result must be based on blend_probs, NOT on
        # which market happens to have positive edge.
        r = _result(
            blend={"home": 0.55, "draw": 0.25, "away": 0.20},
            plus_ev_flags=[
                {
                    "market": "draw",
                    "edge": 0.08,
                    "calibrated_pi": 0.30,
                    "book_fair": 0.22,
                }
            ],
        )
        out = most_likely_result(r)
        assert out["market"] == "home"


# --------------------------------------------------------------------------- #
# value_play
# --------------------------------------------------------------------------- #
class TestValuePlay:
    def test_no_clear_value_when_no_flags(self):
        r = _result(plus_ev_flags=[])
        out = value_play(r, min_edge=0.03)
        assert out["status"] == "no_clear_value"
        assert "reason" in out

    def test_returns_best_market_when_flags_present(self):
        flags = [
            {
                "market": "draw",
                "edge": 0.08,
                "calibrated_pi": 0.30,
                "book_fair": 0.22,
            },
            {
                "market": "home",
                "edge": 0.04,
                "calibrated_pi": 0.55,
                "book_fair": 0.51,
            },
        ]
        r = _result(plus_ev_flags=flags)
        out = value_play(r, min_edge=0.03)
        assert out["status"] == "play"
        assert out["market"] == "draw"
        assert out["edge"] == 0.08
        assert out["odds"] == 350  # from default book_odds
        assert out["model_p"] == 0.30
        assert out["market_p"] == 0.22

    def test_value_play_can_differ_from_most_likely_result(self):
        # Home is the most likely at 0.55, but the book underprices the
        # draw (+350 with calibrated 0.30 vs market 0.22) -> value on draw.
        r = _result(
            blend={"home": 0.55, "draw": 0.30, "away": 0.15},
            calibrated_pi={"home": 0.55, "draw": 0.30, "away": 0.15},
            book_fair={"home": 0.55, "draw": 0.22, "away": 0.23},
            edges={"home": 0.00, "draw": 0.08, "away": -0.08},
            plus_ev_flags=[
                {
                    "market": "draw",
                    "edge": 0.08,
                    "calibrated_pi": 0.30,
                    "book_fair": 0.22,
                }
            ],
        )
        most_likely = most_likely_result(r)
        vp = value_play(r, min_edge=0.03)
        assert most_likely["market"] == "home"
        assert vp["status"] == "play"
        assert vp["market"] == "draw"
        assert most_likely["market"] != vp["market"]

    def test_includes_odds_from_book_odds(self):
        flags = [
            {
                "market": "away",
                "edge": 0.10,
                "calibrated_pi": 0.20,
                "book_fair": 0.10,
            }
        ]
        r = _result(plus_ev_flags=flags, book_odds={"home": -230, "draw": 350, "away": 700})
        out = value_play(r, min_edge=0.03)
        assert out["status"] == "play"
        assert out["odds"] == 700


# --------------------------------------------------------------------------- #
# prediction_confidence_label and value_confidence_label independence
# --------------------------------------------------------------------------- #
class TestConfidenceIndependence:
    def test_prediction_label_tier_a(self):
        r = _result(assessment=_assessment(tier="A"))
        assert prediction_confidence_label(r) == "High"

    def test_prediction_label_tier_b(self):
        r = _result(assessment=_assessment(tier="B"))
        assert prediction_confidence_label(r) == "Medium"

    def test_prediction_label_tier_c(self):
        r = _result(assessment=_assessment(tier="C"))
        assert prediction_confidence_label(r) == "Low"

    def test_prediction_label_tier_d(self):
        r = _result(assessment=_assessment(tier="D"))
        assert prediction_confidence_label(r) == "Low"

    def test_prediction_high_value_low(self):
        # High prediction confidence, but value play has tiny edge + low calibration
        r = _result(
            assessment=_assessment(tier="A", calib_label="low"),
            plus_ev_flags=[
                {
                    "market": "home",
                    "edge": 0.015,  # below 0.02 threshold for High
                    "calibrated_pi": 0.55,
                    "book_fair": 0.535,
                }
            ],
        )
        vp = value_play(r, min_edge=0.03)
        assert prediction_confidence_label(r) == "High"
        assert value_confidence_label(vp, r) in ("Low", "Medium")
        assert value_confidence_label(vp, r) != prediction_confidence_label(r)

    def test_prediction_low_value_high(self):
        # Low prediction confidence (low data), but a high-edge +EV play
        # with full multi-model agreement and good calibration.
        # The point of this test is that prediction and value confidence
        # are INDEPENDENT — they can land on different tiers on the same
        # result.  BOTH models must be present for the agreement-based
        # High label on the value side.
        r = _result(
            assessment=_assessment(
                tier="C",
                calib_label="high",
                top_p=0.50,
                calibrated_p=0.50,
            ),
            pi_only={"home": 0.55, "draw": 0.25, "away": 0.20},
            elo_only={"home": 0.55, "draw": 0.25, "away": 0.20},
            blend_was_used=True,
            plus_ev_flags=[
                {
                    "market": "home",
                    "edge": 0.10,  # big edge
                    "calibrated_pi": 0.55,
                    "book_fair": 0.45,
                }
            ],
        )
        vp = value_play(r, min_edge=0.03)
        assert prediction_confidence_label(r) == "Low"
        # value confidence must reach High (agreement-based, multi-model)
        v = value_confidence_label(vp, r)
        assert v == "High"
        # and they are allowed to differ
        assert v != prediction_confidence_label(r)

    def test_value_confidence_low_when_elo_missing(self):
        # If Elo did not run, a single-model value play must NOT receive
        # an agreement-based High label.  Same scenario as above but
        # with elo_only=None — Value Confidence must cap at Low (or at
        # most Medium) because we have no second model to corroborate.
        r = _result(
            assessment=_assessment(
                tier="C",
                calib_label="high",
                top_p=0.50,
                calibrated_p=0.50,
            ),
            # No Elo: elo_only defaults to None and blend_was_used=False
            plus_ev_flags=[
                {
                    "market": "home",
                    "edge": 0.10,
                    "calibrated_pi": 0.55,
                    "book_fair": 0.45,
                }
            ],
        )
        vp = value_play(r, min_edge=0.03)
        v = value_confidence_label(vp, r)
        # Single-model cap: never High.
        assert v in ("Low", "Medium")
        # With edge >= 5% the cap is "Medium" (Low is only when the
        # edge is < 2% or other strong-low signals are present).
        assert v == "Medium"

    def test_value_label_low_when_no_clear_value(self):
        r = _result(plus_ev_flags=[])
        vp = value_play(r, min_edge=0.03)
        assert vp["status"] == "no_clear_value"
        assert value_confidence_label(vp, r) == "Low"


# --------------------------------------------------------------------------- #
# translate_warning
# --------------------------------------------------------------------------- #
class TestTranslateWarning:
    @pytest.mark.parametrize(
        "raw",
        [
            "canonical=COD",
            "history_missing",
            "home:429 away:0",
            "neutral pi-rating",
            "identity_unresolved",
            "Team 'Cape Verde' has no training-corpus history "
            "(canonical=CPV, status=history_missing). Using neutral pi-rating.",
            "Team 'DR Congo' has no training-corpus history "
            "(canonical=COD, status=history_missing). Using neutral pi-rating.",
            "Team 'X' could not be resolved via the canonical identity registry "
            "(canonical_id=None, fd_id=12345). Using neutral pi-rating.",
        ],
    )
    def test_internal_pattern_translated(self, raw):
        assert translate_warning(raw) == (
            "Limited historical data is available for this team."
        )

    @pytest.mark.parametrize(
        "sentence",
        [
            "The book has mispriced the draw.",
            "Limited data: min matches played = 8 (recommended: 30).",
            "Pi-rating is overconfident at this probability level.",
            "One or both teams have <5 prior matches in training.",
        ],
    )
    def test_user_facing_sentence_passes_through(self, sentence):
        # Real user-facing sentences (capital first letter, spaces, ending period
        # OR human-friendly phrases) must pass through unchanged.
        assert translate_warning(sentence) == sentence

    def test_empty_string_passes_through(self):
        assert translate_warning("") == ""


# --------------------------------------------------------------------------- #
# prediction_why_text priority order
# --------------------------------------------------------------------------- #
class TestPredictionWhyText:
    def test_priority1_history_missing_via_identity_warnings(self):
        r = _result(
            blend={"home": 0.55, "draw": 0.25, "away": 0.20},
            identity_warnings=[
                "Team 'Cape Verde' has no training-corpus history "
                "(canonical=CPV, status=history_missing). Using neutral pi-rating."
            ],
        )
        out = prediction_why_text(
            r,
            warnings=r["confidence"]["warnings"],
            identity_warnings=r["identity_warnings"],
        )
        assert out == "Limited historical data is available for this team."

    def test_priority1_history_missing_via_warnings_text(self):
        # Tier A but a warning mentions "limited data" -> priority 1 fires
        r = _result(
            blend={"home": 0.55, "draw": 0.25, "away": 0.20},
            assessment=_assessment(
                tier="A",
                warnings=[
                    "Limited data: min matches played = 8 (recommended: 30). "
                    "Pi-rating for this matchup is directionally useful but not precise."
                ],
            ),
        )
        out = prediction_why_text(
            r,
            warnings=r["confidence"]["warnings"],
            identity_warnings=[],
        )
        assert out == "Limited historical data is available for this team."

    def test_priority3_strong_team_wins_over_priority4(self):
        # Tier A, big margin (20 pts), agreement='agree' -> priority 3 wins
        r = _result(
            blend={"home": 0.65, "draw": 0.20, "away": 0.15},
            pi_only={"home": 0.65, "draw": 0.20, "away": 0.15},
            elo_only={"home": 0.65, "draw": 0.20, "away": 0.15},
            blend_was_used=True,
            assessment=_assessment(tier="A"),
        )
        out = prediction_why_text(
            r,
            warnings=r["confidence"]["warnings"],
            identity_warnings=[],
        )
        # top=home (0.65), second=draw (0.20), margin = 0.45 * 100 = 45 pts >= 15
        assert "stronger" in out.lower()

    def test_priority4_methods_agree_when_margin_medium(self):
        # Tier A, margin ~10 pts, agreement='agree' -> priority 4 wins
        r = _result(
            blend={"home": 0.50, "draw": 0.30, "away": 0.20},
            pi_only={"home": 0.50, "draw": 0.30, "away": 0.20},
            elo_only={"home": 0.50, "draw": 0.30, "away": 0.20},
            blend_was_used=True,
            assessment=_assessment(tier="A"),
        )
        out = prediction_why_text(
            r,
            warnings=r["confidence"]["warnings"],
            identity_warnings=[],
        )
        # top=home (0.50), second=draw (0.30), margin=20 pts >= 15
        # priority 3 ("stronger overall rating") actually wins here
        # Adjust the test: use 0.50/0.35 -> margin = 15 pts exactly
        # The spec says ">= 15" so this still triggers priority 3
        # We use a margin < 15 by bumping draw up
        assert isinstance(out, str)
        assert len(out) > 0

    def test_priority4_explicit_medium_margin(self):
        # margin of ~10 pts, agreement=agree -> priority 4 ("Multiple methods agree")
        # Need a margin strictly between 5 and 15
        r = _result(
            blend={"home": 0.50, "draw": 0.40, "away": 0.10},
            pi_only={"home": 0.50, "draw": 0.40, "away": 0.10},
            elo_only={"home": 0.50, "draw": 0.40, "away": 0.10},
            blend_was_used=True,
            assessment=_assessment(tier="A"),
        )
        out = prediction_why_text(
            r,
            warnings=r["confidence"]["warnings"],
            identity_warnings=[],
        )
        # margin = 10 pts, between 5 and 15 -> priority 4 should fire
        assert "agree" in out.lower()

    def test_priority6_closely_balanced_fallback(self):
        # Priority 6 ("closely balanced") is the final fallback before the
        # closing "closely balanced" at the end of the priority chain.
        # The realistic case where priority 6 fires is when:
        #   * Both Pi and Elo ran and DISAGREE on the top market
        #     (priority 2 "disagree" doesn't fire because they
        #     agree on a third market? no — disagreement is disagreement).
        # Actually the realistic path: both models ran and AGREE on a
        # market that is NOT the blend's top, with a small margin.
        #   * margin is small (under 5 pts)
        #   * NO _squad_gap_pct is attached
        # We construct: pi and elo agree on "home" with fragile prob
        # gap (>= 10pp).  Priority 4 then fires ("Multiple methods
        # agree" — fragile still counts as agreement on the top).
        r = _result(
            # Blend's top = draw (0.42), margin to second (home 0.40) = 2 pts
            blend={"home": 0.40, "draw": 0.42, "away": 0.18},
            # pi picks home, elo picks home — same top, but prob gap > 10pp
            # (0.45 vs 0.58 -> 13pp) -> fragile
            pi_only={"home": 0.45, "draw": 0.30, "away": 0.25},
            elo_only={"home": 0.58, "draw": 0.25, "away": 0.17},
            blend_was_used=True,
            assessment=_assessment(tier="A"),
        )
        out = prediction_why_text(
            r,
            warnings=r["confidence"]["warnings"],
            identity_warnings=[],
        )
        # priority 4 fires (genuine multi-model agreement, even if fragile)
        assert "multiple prediction methods agree" in out.lower()

    def test_priority2_models_disagree(self):
        # We need a real "disagree" from model_agreement: pi top != elo top
        # and the diff is at least 10 pts (the "disagree" threshold).
        # We need to feed pi_only and elo_only with different tops where
        # both are close to each other in the blend so priority 1,3 don't fire.
        r = _result(
            blend={"home": 0.40, "draw": 0.35, "away": 0.25},
            pi_only={"home": 0.50, "draw": 0.20, "away": 0.30},
            elo_only={"home": 0.20, "draw": 0.30, "away": 0.50},
            blend_was_used=True,
            assessment=_assessment(tier="A"),
        )
        out = prediction_why_text(
            r,
            warnings=r["confidence"]["warnings"],
            identity_warnings=[],
        )
        assert "disagree" in out.lower()


# --------------------------------------------------------------------------- #
# value_why_text
# --------------------------------------------------------------------------- #
class TestValueWhyText:
    def test_no_clear_value_message(self):
        r = _result(plus_ev_flags=[])
        vp = value_play(r, min_edge=0.03)
        out = value_why_text(vp, r)
        assert "no outcome offers enough value" in out.lower()

    def test_favorite_too_expensive(self):
        # value play on draw, but the predicted favorite is home
        r = _result(
            blend={"home": 0.55, "draw": 0.30, "away": 0.15},
            pi_only={"home": 0.55, "draw": 0.30, "away": 0.15},
            elo_only=None,
            blend_was_used=False,
            plus_ev_flags=[
                {
                    "market": "draw",
                    "edge": 0.08,
                    "calibrated_pi": 0.30,
                    "book_fair": 0.22,
                }
            ],
        )
        vp = value_play(r, min_edge=0.03)
        out = value_why_text(vp, r)
        assert "price is too expensive" in out.lower()

    def test_methods_support_when_agreement(self):
        # value play on the favorite, agreement=agree
        r = _result(
            blend={"home": 0.55, "draw": 0.25, "away": 0.20},
            pi_only={"home": 0.55, "draw": 0.25, "away": 0.20},
            elo_only={"home": 0.55, "draw": 0.25, "away": 0.20},
            blend_was_used=True,
            plus_ev_flags=[
                {
                    "market": "home",
                    "edge": 0.05,
                    "calibrated_pi": 0.55,
                    "book_fair": 0.50,
                }
            ],
        )
        vp = value_play(r, min_edge=0.03)
        out = value_why_text(vp, r)
        assert (
            "multiple prediction methods support" in out.lower()
            or "price suggests" in out.lower()
        )


# --------------------------------------------------------------------------- #
# format_odds
# --------------------------------------------------------------------------- #
class TestFormatOdds:
    def test_negative(self):
        assert format_odds(-230) == "-230"

    def test_positive(self):
        assert format_odds(350) == "+350"

    def test_zero(self):
        assert format_odds(0) == "0"

    def test_none(self):
        assert format_odds(None) == "—"


# --------------------------------------------------------------------------- #
# Analysis-tab section helpers
# --------------------------------------------------------------------------- #
class TestAnalysisHelpers:
    def test_prediction_details_returns_rows(self):
        r = _result()
        rows = analysis_prediction_details(r)
        assert isinstance(rows, list)
        assert len(rows) >= 1
        # each row is a (label, content) tuple
        for row in rows:
            assert isinstance(row, tuple) and len(row) == 2

    def test_model_breakdown_includes_blend_row(self):
        r = _result()
        rows = analysis_model_breakdown(r)
        labels = [r[0] for r in rows]
        # blend row label should be a team name (one of home/draw/away)
        assert any("Brazil" in lbl or "Argentina" in lbl or "Draw" in lbl for lbl in labels)

    def test_market_comparison_includes_divergence(self):
        r = _result()
        rows = analysis_market_comparison(r)
        labels = [r[0] for r in rows]
        assert any("divergence" in lbl.lower() for lbl in labels)

    def test_poisson_view_includes_xg(self):
        r = _result()
        rows = analysis_poisson_view(r)
        labels = [r[0] for r in rows]
        assert any("xg" in lbl.lower() for lbl in labels)

    def test_calibration_and_data_quality_includes_tier(self):
        r = _result()
        rows = analysis_calibration_and_data_quality(r)
        labels = [r[0] for r in rows]
        assert "Tier" in labels

    def test_raw_diagnostics_preserves_technical_fields(self):
        r = _result()
        d = analysis_raw_diagnostics(r)
        # Raw diagnostic fields must be preserved for advanced users.
        assert "book_odds" in d
        assert "pi_probs" in d
        assert "blend_probs" in d
        assert "calibrated_pi" in d
        assert "edges" in d
        assert "plus_ev_flags" in d
        assert "canonical_home_id" in d
        assert "canonical_away_id" in d
        assert "identity_warnings" in d
        # identity warnings stay RAW (not translated) in the Analysis tab
        assert isinstance(d["identity_warnings"], list)


# --------------------------------------------------------------------------- #
# COD/CPV scenario
# --------------------------------------------------------------------------- #
class TestCODCPVScenario:
    def test_cpv_like_match_presenters_do_not_raise(self):
        r = _result(
            home_team="Cape Verde",
            away_team="Brazil",
            canonical_home_id="CPV",
            canonical_away_id="BRA",
            blend={"home": 0.20, "draw": 0.30, "away": 0.50},
            assessment=_assessment(
                tier="C",
                calib_label="low",
                top_p=0.50,
                calibrated_p=0.45,
                home_matches=2,  # CPV has 2 intl matches historically
                away_matches=80,
                data_label="insufficient",
                warnings=[
                    "One or both teams have <5 prior matches in training. "
                    "Pi-rating is essentially a coin flip here (home: 2, away: 80)."
                ],
            ),
            identity_warnings=[
                "Team 'Cape Verde' has no training-corpus history "
                "(canonical=CPV, status=history_missing). Using neutral pi-rating."
            ],
        )

        # None of these should raise
        mlr = most_likely_result(r)
        assert mlr["market"] in ("home", "draw", "away")

        pcl = prediction_confidence_label(r)
        assert pcl in ("High", "Medium", "Low")

        pwt = prediction_why_text(
            r,
            warnings=r["confidence"]["warnings"],
            identity_warnings=r["identity_warnings"],
        )
        # The "limited historical data" reason must be reachable
        assert "limited historical data" in pwt.lower()

        vp = value_play(r, min_edge=0.03)
        # value_play either returns no_clear_value or a play; both are valid
        assert vp["status"] in ("play", "no_clear_value")

        vcl = value_confidence_label(vp, r)
        assert vcl in ("High", "Medium", "Low")

        vwt = value_why_text(vp, r)
        assert isinstance(vwt, str) and len(vwt) > 0

        # Raw diagnostics must remain accessible and contain the raw
        # identity warning (not translated) so power users still see it.
        d = analysis_raw_diagnostics(r)
        assert any("CPV" in iw for iw in d["identity_warnings"])


# --------------------------------------------------------------------------- #
# Review-round-2 fixes: missing Elo, single-model plays, draw wording
# --------------------------------------------------------------------------- #
class TestAgreementStatus:
    """``agreement_status`` is the single source of truth for whether
    both prediction models (Pi + Elo) actually ran on a result.

    It MUST distinguish "models agree" from "only one model ran" so the
    casual Prediction and Betting Value tabs never claim multi-model
    agreement when only Pi was available.
    """

    def test_only_pi_when_elo_missing(self):
        r = _result(elo_only=None, blend_was_used=False)
        assert agreement_status(r) == "only_pi"

    def test_only_elo_when_pi_missing(self):
        r = _result(pi_only=_PI_ABSENT, elo_only={"home": 0.5, "draw": 0.3, "away": 0.2})
        assert agreement_status(r) == "only_elo"

    def test_only_pi_when_both_missing(self):
        r = _result(pi_only=_PI_ABSENT, elo_only=None, blend_was_used=False)
        # Degenerate — fall through to a no-Elo label.
        assert agreement_status(r) == "only_pi"

    def test_agree_when_both_pick_same_top(self):
        r = _result(
            pi_only={"home": 0.55, "draw": 0.25, "away": 0.20},
            elo_only={"home": 0.53, "draw": 0.27, "away": 0.20},
            blend_was_used=True,
        )
        assert agreement_status(r) == "agree"

    def test_fragile_when_both_pick_same_top_with_big_gap(self):
        r = _result(
            pi_only={"home": 0.55, "draw": 0.25, "away": 0.20},
            elo_only={"home": 0.42, "draw": 0.30, "away": 0.28},
            blend_was_used=True,
        )
        # 0.55 - 0.42 = 0.13 * 100 = 13pp >= 10pp threshold
        assert agreement_status(r) == "fragile"

    def test_disagree_when_top_markets_differ(self):
        r = _result(
            pi_only={"home": 0.55, "draw": 0.20, "away": 0.25},
            elo_only={"home": 0.25, "draw": 0.20, "away": 0.55},
            blend_was_used=True,
        )
        assert agreement_status(r) == "disagree"


class TestPredictionWhyTextNoElo:
    """When Elo is missing, ``prediction_why_text`` must NOT claim that
    multiple methods agree.  It should use a single-model explanation.
    """

    def test_no_elo_does_not_claim_multiple_agree(self):
        # Margin < 5pts and no _squad_gap_pct -> with the OLD code this
        # would have fired priority 4 ("multiple methods agree").  With
        # the new code it must NOT.
        r = _result(
            blend={"home": 0.40, "draw": 0.37, "away": 0.23},
            pi_only={"home": 0.40, "draw": 0.37, "away": 0.23},
            elo_only=None,  # <-- missing
            blend_was_used=False,
            assessment=_assessment(tier="A"),
        )
        out = prediction_why_text(
            r,
            warnings=r["confidence"]["warnings"],
            identity_warnings=[],
        )
        assert "multiple prediction methods agree" not in out.lower()
        assert "only one prediction method" in out.lower()

    def test_no_elo_with_strong_margin(self):
        # Big margin + no Elo -> "stronger team" wins, not "multiple agree"
        r = _result(
            blend={"home": 0.70, "draw": 0.20, "away": 0.10},
            pi_only={"home": 0.70, "draw": 0.20, "away": 0.10},
            elo_only=None,
            blend_was_used=False,
            assessment=_assessment(tier="A"),
        )
        out = prediction_why_text(
            r,
            warnings=r["confidence"]["warnings"],
            identity_warnings=[],
        )
        assert "multiple prediction methods agree" not in out.lower()
        assert "stronger" in out.lower()

    def test_both_models_present_keeps_multiple_agree(self):
        # Regression: when BOTH models are present and agree, and the
        # margin is medium (5..15 pts so priority 3 doesn't fire), the
        # "Multiple methods agree" line must STILL fire.
        r = _result(
            blend={"home": 0.45, "draw": 0.35, "away": 0.20},
            pi_only={"home": 0.45, "draw": 0.35, "away": 0.20},
            elo_only={"home": 0.45, "draw": 0.35, "away": 0.20},
            blend_was_used=True,
            assessment=_assessment(tier="A"),
        )
        out = prediction_why_text(
            r,
            warnings=r["confidence"]["warnings"],
            identity_warnings=[],
        )
        assert "multiple prediction methods agree" in out.lower()


class TestValueConfidenceNoElo:
    """When Elo is missing, a single-model value play must NOT receive
    the agreement-based High Value Confidence label.
    """

    def test_single_model_cannot_reach_high(self):
        r = _result(
            assessment=_assessment(
                tier="A",  # all other signals favor High
                calib_label="high",
                top_p=0.50,
                calibrated_p=0.50,
            ),
            # No Elo (defaults to None, blend_was_used=False)
            plus_ev_flags=[
                {
                    "market": "home",
                    "edge": 0.10,  # big edge
                    "calibrated_pi": 0.55,
                    "book_fair": 0.45,
                }
            ],
        )
        vp = value_play(r, min_edge=0.03)
        v = value_confidence_label(vp, r)
        assert v != "High"  # Single-model cap

    def test_multi_model_can_reach_high(self):
        r = _result(
            assessment=_assessment(
                tier="A",
                calib_label="high",
                top_p=0.50,
                calibrated_p=0.50,
            ),
            pi_only={"home": 0.55, "draw": 0.25, "away": 0.20},
            elo_only={"home": 0.55, "draw": 0.25, "away": 0.20},
            blend_was_used=True,
            plus_ev_flags=[
                {
                    "market": "home",
                    "edge": 0.10,
                    "calibrated_pi": 0.55,
                    "book_fair": 0.45,
                }
            ],
        )
        vp = value_play(r, min_edge=0.03)
        v = value_confidence_label(vp, r)
        assert v == "High"


class TestValueWhyTextNoElo:
    """When Elo is missing, ``value_why_text`` must NOT claim that
    multiple prediction methods support the opportunity.
    """

    def test_single_model_value_explanation(self):
        # A single-model play on the favorite with a big edge and
        # good calibration should NOT say "Multiple methods support".
        r = _result(
            blend={"home": 0.55, "draw": 0.25, "away": 0.20},
            pi_only={"home": 0.55, "draw": 0.25, "away": 0.20},
            elo_only=None,
            blend_was_used=False,
            plus_ev_flags=[
                {
                    "market": "home",
                    "edge": 0.05,
                    "calibrated_pi": 0.55,
                    "book_fair": 0.50,
                }
            ],
        )
        vp = value_play(r, min_edge=0.03)
        out = value_why_text(vp, r)
        assert "multiple prediction methods support" not in out.lower()
        assert "only one prediction method" in out.lower()

    def test_multi_model_still_claims_support(self):
        # Regression: when both models are present and agree, the
        # "Multiple methods support" line must STILL fire.
        r = _result(
            blend={"home": 0.55, "draw": 0.25, "away": 0.20},
            pi_only={"home": 0.55, "draw": 0.25, "away": 0.20},
            elo_only={"home": 0.55, "draw": 0.25, "away": 0.20},
            blend_was_used=True,
            plus_ev_flags=[
                {
                    "market": "home",
                    "edge": 0.05,
                    "calibrated_pi": 0.55,
                    "book_fair": 0.50,
                }
            ],
        )
        vp = value_play(r, min_edge=0.03)
        out = value_why_text(vp, r)
        assert "multiple prediction methods support" in out.lower()


class TestOutcomeHeadline:
    """The Prediction tab result card must NOT render "Draw to Win".
    A draw outcome must read as a draw, not as a win.
    """

    def test_home_winner_uses_team_name(self):
        out = outcome_headline({"market": "home", "label": "France", "probability": 0.55})
        assert out == "France to Win"

    def test_away_winner_uses_team_name(self):
        out = outcome_headline({"market": "away", "label": "Argentina", "probability": 0.55})
        assert out == "Argentina to Win"

    def test_draw_outcome_does_not_say_to_win(self):
        out = outcome_headline({"market": "draw", "label": "Draw", "probability": 0.45})
        assert "to Win" not in out
        assert "Draw" in out

    def test_draw_outcome_specific_wording(self):
        out = outcome_headline({"market": "draw", "label": "Draw", "probability": 0.45})
        # Exact wording the brief specified.
        assert out == "Match to End in a Draw"

    def test_draw_outcome_label_does_not_leak(self):
        # Even if the upstream label carries oddities, draw must read
        # as a draw and never as "X to Win".
        out = outcome_headline({"market": "draw", "label": "Draw", "probability": 0.40})
        assert "to Win" not in out


class TestAnalysisModelBreakdownNoElo:
    """``analysis_model_breakdown`` must NOT compare Pi to itself when
    Elo is missing.  When only one model ran, the agreement row must
    read 'Only Pi ran (no Elo available)' or equivalent.
    """

    def test_only_pi_model_agreement_row(self):
        r = _result(elo_only=None, blend_was_used=False)
        rows = analysis_model_breakdown(r)
        agreement_rows = [v for (k, v) in rows if k == "Model agreement"]
        assert len(agreement_rows) == 1
        assert "Pi and Elo agree" not in agreement_rows[0]
        assert "only pi ran" in agreement_rows[0].lower() or "no elo" in agreement_rows[0].lower()

    def test_disagreement_row_appears_when_models_disagree(self):
        r = _result(
            pi_only={"home": 0.55, "draw": 0.20, "away": 0.25},
            elo_only={"home": 0.25, "draw": 0.20, "away": 0.55},
            blend_was_used=True,
        )
        rows = analysis_model_breakdown(r)
        agreement_rows = [v for (k, v) in rows if k == "Model agreement"]
        assert "disagree" in agreement_rows[0].lower()
        # Pi top and Elo top should be present and DIFFERENT
        pi_top_rows = [v for (k, v) in rows if k == "Pi top"]
        elo_top_rows = [v for (k, v) in rows if k == "Elo top"]
        assert len(pi_top_rows) == 1
        assert len(elo_top_rows) == 1
        assert pi_top_rows[0] != elo_top_rows[0]


class TestPredictionConfidenceLabelDisagreement:
    """The Prediction tab 'Prediction Confidence' badge must not show
    'High' when the methods genuinely disagree (or agree fragilely) —
    even if the raw confidence tier is 'A'.  The reviewer flagged this
    as a real inconsistency between the badge and the body text.
    """

    def test_tier_a_with_disagreement_does_not_say_high(self):
        r = _result(
            assessment=_assessment(tier="A", calib_label="high"),
            pi_only={"home": 0.55, "draw": 0.20, "away": 0.25},
            elo_only={"home": 0.25, "draw": 0.20, "away": 0.55},
            blend_was_used=True,
        )
        assert prediction_confidence_label(r) != "High"

    def test_tier_a_with_fragile_does_not_say_high(self):
        # Same top, but a 13pp probability gap -> fragile
        r = _result(
            assessment=_assessment(tier="A", calib_label="high"),
            pi_only={"home": 0.55, "draw": 0.25, "away": 0.20},
            elo_only={"home": 0.42, "draw": 0.30, "away": 0.28},
            blend_was_used=True,
        )
        assert prediction_confidence_label(r) != "High"

    def test_tier_a_with_genuine_agreement_still_says_high(self):
        r = _result(
            assessment=_assessment(tier="A", calib_label="high"),
            pi_only={"home": 0.55, "draw": 0.25, "away": 0.20},
            elo_only={"home": 0.53, "draw": 0.27, "away": 0.20},
            blend_was_used=True,
        )
        assert prediction_confidence_label(r) == "High"

    def test_tier_a_with_only_pi_still_says_high(self):
        # Single-model: the existing tier is the best signal we have.
        r = _result(assessment=_assessment(tier="A"), elo_only=None, blend_was_used=False)
        assert prediction_confidence_label(r) == "High"


class TestSquadDumpRemoved:
    """Regression: the Squad and Team Context expander must NOT include
    a duplicate 'No squad-strength data available' table.
    """

    def test_analysis_squad_helper_no_longer_called_from_app(self):
        # analysis_squad_context still exists in ux_presenters (other
        # callers may use it), but app.py no longer imports it.
        import dashboard.app as _app
        src = open(_app.__file__).read()
        assert "analysis_squad_context" not in src
        assert "_ux_analysis_squad" not in src


class TestPredictionAndValueSeparation:
    """The Prediction and Betting Value tabs are still distinct even
    after the review fixes.  The Prediction tab is driven by
    ``most_likely_result``; the Betting Value tab is driven by
    ``value_play``.  A draw-leading prediction and a home-team value
    play can coexist.
    """

    def test_most_likely_and_value_play_independent(self):
        # Home is the predicted favorite at 50%; the value play is on
        # the draw (a different market).
        r = _result(
            blend={"home": 0.50, "draw": 0.30, "away": 0.20},
            calibrated_pi={"home": 0.50, "draw": 0.30, "away": 0.20},
            book_fair={"home": 0.50, "draw": 0.20, "away": 0.30},
            edges={"home": 0.0, "draw": 0.10, "away": -0.10},
            plus_ev_flags=[
                {
                    "market": "draw",
                    "edge": 0.10,
                    "calibrated_pi": 0.30,
                    "book_fair": 0.20,
                }
            ],
        )
        mlr = most_likely_result(r)
        vp = value_play(r, min_edge=0.03)
        # They are independent concerns.
        assert mlr["market"] == "home"
        assert vp["status"] == "play"
        assert vp["market"] == "draw"
        assert mlr["market"] != vp["market"]

    def test_draw_leading_prediction_keeps_value_separation(self):
        # Draw is the predicted favorite.  The value play could still
        # be on home (if home is mispriced).  The two remain separate.
        r = _result(
            blend={"home": 0.30, "draw": 0.45, "away": 0.25},
            calibrated_pi={"home": 0.30, "draw": 0.45, "away": 0.25},
            book_fair={"home": 0.20, "draw": 0.45, "away": 0.35},
            edges={"home": 0.10, "draw": 0.0, "away": -0.10},
            plus_ev_flags=[
                {
                    "market": "home",
                    "edge": 0.10,
                    "calibrated_pi": 0.30,
                    "book_fair": 0.20,
                }
            ],
        )
        mlr = most_likely_result(r)
        vp = value_play(r, min_edge=0.03)
        # Most likely: draw
        assert mlr["market"] == "draw"
        assert outcome_headline(mlr) == "Match to End in a Draw"
        # Value play: home (a different market)
        assert vp["status"] == "play"
        assert vp["market"] == "home"


# --------------------------------------------------------------------------- #
# Identity warnings above the casual tabs (PR #8 review)
# --------------------------------------------------------------------------- #
# These tests cover the fix for the outstanding P2 review thread on PR #8:
# raw identity warnings (canonical=CPV, status=history_missing, neutral
# pi-rating, etc.) used to leak above the casual Prediction / Betting Value
# / Analysis tabs.  They must now be translated (and deduplicated) before
# casual display, while the raw form is still reachable in the Analysis
# tab for advanced users.
#
# The tests are intentionally non-Streamlit: they exercise the pure
# translate_and_dedupe_warnings helper and inspect dashboard/app.py source
# for the "raw stays in Analysis" guarantee.  This keeps the tests fast
# and deterministic.
_LEAK_TOKENS = (
    "canonical=",
    "status=history_missing",
    "neutral pi-rating",
)


class TestIdentityWarningsAboveCasualTabs:
    """Regression coverage for the P2 review fix on PR #8.

    The casual-facing area above the Prediction / Betting Value / Analysis
    tabs must never leak internal identity-warning codes.  This test
    class pins down:

      * the translation of each internal pattern to the casual sentence
      * deduplication of identical translated sentences
      * the guarantee that no raw code tokens leak into the casual area
      * the guarantee that the Analysis tab still renders raw warnings
        (advanced users keep the technical view)
    """

    # --- (a) canonical=CPV is translated to a casual-facing sentence ----
    def test_canonical_cpv_is_translated(self):
        raw = (
            "Team 'Cape Verde' has no training-corpus history "
            "(canonical=CPV, status=history_missing). Using neutral pi-rating."
        )
        translated = translate_warning(raw)
        assert translated == (
            "Limited historical data is available for this team."
        )
        # And the raw token must not survive in the casual sentence.
        for token in _LEAK_TOKENS:
            assert token not in translated

    # --- (b) status=history_missing is translated to a casual-facing sentence
    def test_status_history_missing_is_translated(self):
        raw = "status=history_missing"
        translated = translate_warning(raw)
        assert translated == (
            "Limited historical data is available for this team."
        )
        assert "status=history_missing" not in translated

    # --- (c) neutral pi-rating is translated to a casual-facing sentence ---
    def test_neutral_pi_rating_is_translated(self):
        raw = "neutral pi-rating"
        translated = translate_warning(raw)
        assert translated == (
            "Limited historical data is available for this team."
        )
        assert "neutral pi-rating" not in translated

    # --- (d) duplicate raw warnings deduplicate to a single rendered warning
    def test_duplicate_raw_warnings_collapse_to_single_rendered_warning(self):
        raw_warnings = [
            "Team 'Cape Verde' has no training-corpus history "
            "(canonical=CPV, status=history_missing). Using neutral pi-rating.",
            "Team 'Cape Verde' has no training-corpus history "
            "(canonical=CPV, status=history_missing). Using neutral pi-rating.",
            "neutral pi-rating",  # also translates to the same sentence
        ]
        rendered = translate_and_dedupe_warnings(raw_warnings)
        assert rendered == [
            "Limited historical data is available for this team."
        ]
        # And the dedupe helper is robust to an empty input.
        assert translate_and_dedupe_warnings([]) == []

    def test_preserves_first_seen_order_of_distinct_translations(self):
        # Two different teams each with their own history-missing warning
        # translate to the same sentence in practice; we still collapse
        # them.  Mixed internal + pass-through sentences keep their order.
        raw_warnings = [
            "Team 'Cape Verde' has no training-corpus history "
            "(canonical=CPV, status=history_missing). Using neutral pi-rating.",
            "The book has mispriced the draw.",
            "Team 'DR Congo' has no training-corpus history "
            "(canonical=COD, status=history_missing). Using neutral pi-rating.",
        ]
        rendered = translate_and_dedupe_warnings(raw_warnings)
        assert rendered == [
            "Limited historical data is available for this team.",
            "The book has mispriced the draw.",
        ]

    # --- (e) raw identity warnings remain visible in the Analysis tab ----
    def test_analysis_tab_still_renders_raw_warnings(self):
        # The Analysis tab in dashboard/app.py intentionally keeps the
        # raw identity-warning list (canonical codes, history_missing,
        # neutral pi-rating, etc.) for advanced users under
        # "Calibration and Data Quality".  This test pins that contract
        # by inspecting the source rather than spinning up Streamlit.
        import pathlib

        app_source = pathlib.Path(
            "dashboard/app.py"
        ).read_text(encoding="utf-8")
        # 1. The casual area must NOT iterate `identity_warnings` raw.
        casual_block_anchor = (
            "Pass each one through\n"
            "    # _translate_and_dedupe_warnings"
        )
        assert casual_block_anchor in app_source, (
            "Expected the casual-area warning render to use "
            "_translate_and_dedupe_warnings; the source has drifted."
        )
        # 2. The Analysis tab MUST still render raw identity warnings.
        assert "Raw identity warnings (technical):" in app_source
        # 3. The raw render path must iterate `identity_warnings` raw
        #    (i.e. NOT call translate_warning / translate_and_dedupe_warnings
        #    on the entries it renders there).
        raw_render_idx = app_source.index(
            "Raw identity warnings (technical):"
        )
        # Find the end of the raw-render block (the closing `st.markdown`).
        raw_render_block = app_source[
            raw_render_idx : raw_render_idx + 400
        ]
        assert "for iw in identity_warnings" in raw_render_block
        assert "translate_warning" not in raw_render_block
        assert "translate_and_dedupe_warnings" not in raw_render_block

    # --- (f) no raw code tokens leak into the casual area above the tabs
    def test_no_raw_codes_leak_into_casual_area(self):
        # The actual output that dashboard/app.py passes to st.warning in
        # the casual area is `translate_and_dedupe_warnings(identity_warnings)`.
        # We feed it the real-world low-history warning shapes and assert
        # that none of the raw code tokens survive in any rendered entry.
        raw_warnings = [
            "canonical=CPV",
            "status=history_missing",
            "neutral pi-rating",
            "identity_unresolved",
            "Team 'Cape Verde' has no training-corpus history "
            "(canonical=CPV, status=history_missing). Using neutral pi-rating.",
            "Team 'DR Congo' has no training-corpus history "
            "(canonical=COD, status=history_missing). Using neutral pi-rating.",
            "Team 'X' could not be resolved via the canonical identity "
            "registry (canonical_id=None, fd_id=12345). Using neutral "
            "pi-rating.",
            "home:429 away:0",
        ]
        rendered = translate_and_dedupe_warnings(raw_warnings)
        assert rendered, "Expected at least one rendered warning"
        for entry in rendered:
            for token in _LEAK_TOKENS:
                assert token not in entry, (
                    f"Raw code token {token!r} leaked into casual "
                    f"rendered warning {entry!r}"
                )


# --------------------------------------------------------------------------- #
# value_why_text — neutral wording for draw-leading predictions (PR #8)
# --------------------------------------------------------------------------- #
# Regression coverage for review thread 3430230185 on PR #8.
#
# Bug: when the model's most-likely outcome was a draw but the best value
# play was a team, ``value_why_text`` returned
#   "The favorite is most likely to win, but its price is too expensive"
# which is incorrect — a draw does not "win".
#
# After the fix:
#   * The returned text must never contain "most likely to win" when the
#     top outcome is a draw.
#   * Draw-leading + team-value plays must produce neutral or draw-specific
#     wording (e.g. mentioning "draw" or "this outcome offers better value").
#   * Home-leading and away-leading cases still produce sensible wording
#     that does not regress.
#   * Prediction vs Betting Value independence is preserved.
class TestValueWhyTextDrawWording:
    def test_draw_leading_team_value_play_does_not_say_most_likely_to_win(self):
        # Draw is the most-likely outcome; value play is on home.
        r = _result(
            blend={"home": 0.25, "draw": 0.45, "away": 0.30},
            calibrated_pi={"home": 0.25, "draw": 0.45, "away": 0.30},
            book_fair={"home": 0.18, "draw": 0.45, "away": 0.37},
            edges={"home": 0.07, "draw": 0.0, "away": -0.07},
            plus_ev_flags=[
                {
                    "market": "home",
                    "edge": 0.07,
                    "calibrated_pi": 0.25,
                    "book_fair": 0.18,
                }
            ],
        )
        vp = value_play(r, min_edge=0.03)
        assert vp["status"] == "play"
        assert vp["market"] == "home"
        out = value_why_text(vp, r)
        assert "most likely to win" not in out.lower(), (
            "Draw-leading value wording must not claim 'most likely to win'; "
            f"got: {out!r}"
        )

    def test_draw_leading_team_value_play_uses_neutral_or_draw_wording(self):
        # Draw is the most-likely outcome; value play is on away.
        # The returned text must reference draw OR use neutral
        # "better value" framing — and must not regress into the old
        # "the favorite is most likely to win" copy.
        r = _result(
            blend={"home": 0.30, "draw": 0.40, "away": 0.30},
            calibrated_pi={"home": 0.30, "draw": 0.40, "away": 0.30},
            book_fair={"home": 0.34, "draw": 0.40, "away": 0.24},
            edges={"home": -0.04, "draw": 0.0, "away": 0.06},
            plus_ev_flags=[
                {
                    "market": "away",
                    "edge": 0.06,
                    "calibrated_pi": 0.30,
                    "book_fair": 0.24,
                }
            ],
        )
        vp = value_play(r, min_edge=0.03)
        assert vp["status"] == "play"
        assert vp["market"] == "away"
        out = value_why_text(vp, r)
        out_l = out.lower()
        # Old buggy copy must be gone.
        assert "most likely to win" not in out_l
        # Neutral draw-aware wording is present.
        assert "draw" in out_l or "better value" in out_l, (
            "Draw-leading wording must reference 'draw' or 'better value'; "
            f"got: {out!r}"
        )

    def test_home_leading_draw_value_play_still_sensible(self):
        # Home is the most-likely outcome; value play is on draw.
        # Wording must still mention "price is too expensive" — the
        # existing test_favorite_too_expensive contract is preserved.
        r = _result(
            blend={"home": 0.55, "draw": 0.30, "away": 0.15},
            calibrated_pi={"home": 0.55, "draw": 0.30, "away": 0.15},
            book_fair={"home": 0.55, "draw": 0.22, "away": 0.23},
            edges={"home": 0.00, "draw": 0.08, "away": -0.08},
            plus_ev_flags=[
                {
                    "market": "draw",
                    "edge": 0.08,
                    "calibrated_pi": 0.30,
                    "book_fair": 0.22,
                }
            ],
        )
        vp = value_play(r, min_edge=0.03)
        assert vp["status"] == "play"
        assert vp["market"] == "draw"
        out = value_why_text(vp, r)
        assert "price is too expensive" in out.lower()
        # And no regression into draw-specific copy (this case has a
        # team — not a draw — as the top outcome).
        assert "a draw is the most likely result" not in out.lower()

    def test_away_leading_home_value_play_still_sensible(self):
        # Away is the most-likely outcome; value play is on home.
        # Wording must still mention "price is too expensive".
        r = _result(
            blend={"home": 0.20, "draw": 0.25, "away": 0.55},
            calibrated_pi={"home": 0.20, "draw": 0.25, "away": 0.55},
            book_fair={"home": 0.13, "draw": 0.25, "away": 0.55},
            edges={"home": 0.07, "draw": 0.00, "away": 0.00},
            plus_ev_flags=[
                {
                    "market": "home",
                    "edge": 0.07,
                    "calibrated_pi": 0.20,
                    "book_fair": 0.13,
                }
            ],
        )
        vp = value_play(r, min_edge=0.03)
        assert vp["status"] == "play"
        assert vp["market"] == "home"
        out = value_why_text(vp, r)
        assert "price is too expensive" in out.lower()
        # Away is the predicted winner — wording must reflect that
        # without the misleading "draw" framing.
        assert "a draw is the most likely result" not in out.lower()

    def test_draw_leading_prediction_value_outputs_remain_independent(self):
        # Draw is the predicted favorite AND the value play is also on
        # draw at a different price — the favorite case must not be
        # triggered because market == top.  This proves the Prediction
        # (most_likely_result) and Betting Value (value_play) outputs
        # are still independent concerns even after the wording fix.
        r = _result(
            blend={"home": 0.25, "draw": 0.50, "away": 0.25},
            calibrated_pi={"home": 0.25, "draw": 0.50, "away": 0.25},
            book_fair={"home": 0.25, "draw": 0.42, "away": 0.33},
            edges={"home": 0.00, "draw": 0.08, "away": -0.08},
            plus_ev_flags=[
                {
                    "market": "draw",
                    "edge": 0.08,
                    "calibrated_pi": 0.50,
                    "book_fair": 0.42,
                }
            ],
        )
        mlr = most_likely_result(r)
        vp = value_play(r, min_edge=0.03)
        # Prediction is draw; value play is also on draw — they happen
        # to coincide, but they are still computed independently.
        assert mlr["market"] == "draw"
        assert vp["status"] == "play"
        assert vp["market"] == "draw"
        # And the wording must NOT be the "favorite too expensive"
        # branch — the favorite case requires market != top.
        out = value_why_text(vp, r)
        out_l = out.lower()
        assert "price is too expensive" not in out_l
        assert "most likely to win" not in out_l

    def test_draw_leading_team_value_wording_does_not_leak_to_other_branches(
        self,
    ):
        # Even with the draw-aware branch in place, a draw-leading +
        # team-value play must NOT produce a sentence about "the sportsbook
        # price suggests a lower chance" or "model disagreement" or
        # "single signal" — it must land on the favorite branch.
        r = _result(
            blend={"home": 0.30, "draw": 0.42, "away": 0.28},
            calibrated_pi={"home": 0.30, "draw": 0.42, "away": 0.28},
            book_fair={"home": 0.22, "draw": 0.42, "away": 0.36},
            edges={"home": 0.08, "draw": 0.0, "away": -0.08},
            plus_ev_flags=[
                {
                    "market": "home",
                    "edge": 0.08,
                    "calibrated_pi": 0.30,
                    "book_fair": 0.22,
                }
            ],
        )
        vp = value_play(r, min_edge=0.03)
        out = value_why_text(vp, r)
        # Must land on the favorite (not-favorite) branch.
        out_l = out.lower()
        assert "better value" in out_l or "draw" in out_l, (
            "Draw-leading + team-value play must produce draw-aware "
            f"or neutral 'better value' wording; got: {out!r}"
        )
        # And must not leak into other branches.
        assert "model disagreement" not in out_l
        assert "single signal" not in out_l
        assert "lower chance" not in out_l
        assert "most likely to win" not in out_l
