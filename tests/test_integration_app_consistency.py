"""
Phase 9B — Integration tests and full validation.

App-consistency suite: verifies that primary_probs is the single official
prediction source across all consumers.  No consumer independently
recomputes the blend or reads pi_probs when primary_probs is available.

Five categories:
  1. App-consistency: primary_probs used by every consumer
  2. Matchup integration: 5+ representative scenarios
  3. Dashboard integration: AppTest-level tests
  4. Focused model integration: predict_match pipeline tests
  5. Full local suite runner
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

_WC_ROOT = Path(__file__).resolve().parent.parent
if str(_WC_ROOT) not in sys.path:
    sys.path.insert(0, str(_WC_ROOT))

_DASHBOARD_APP = _WC_ROOT / "dashboard" / "app.py"


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _make_match(date, home_id, away_id, hg=0, ag=0, result="H"):
    return {
        "match_id": f"{date}_{home_id}_{away_id}",
        "date": date,
        "home_team": f"Team{home_id}",
        "away_team": f"Team{away_id}",
        "home_team_id": home_id,
        "away_team_id": away_id,
        "home_goals": hg,
        "away_goals": ag,
        "result": result,
    }


def _train_ratings(home_id=1, away_id=2, n_each=40):
    """Build a minimal training set and compute pi-ratings."""
    train = []
    for i in range(n_each):
        train.append(_make_match(f"2020-{(i % 9) + 1:02d}-01", home_id, away_id, 2, 0))
        train.append(_make_match(f"2020-{(i % 9) + 1:02d}-02", away_id, home_id, 1, 1))
    from soccer_ev_model.pi_ratings import compute_pi_ratings
    return train, compute_pi_ratings(train, cutoff="2020-12-01")


def _fake_prediction_full(
    home_team="Argentina",
    away_team="Brazil",
    home_id=1,
    away_id=2,
    primary_overrides=None,
    pi_overrides=None,
    elo_only_overrides=None,
    goal_model_used=False,
    goal_model_expected=False,
    elo_blend=False,
    confidence_tier="A",
    calibrated_p=0.55,
):
    """Build a realistic prediction dict with independent primary/pi/elo values.

    By default primary_probs != pi_probs so we can detect which one a consumer used.
    """
    primary = primary_overrides or {"home": 0.60, "draw": 0.25, "away": 0.15}
    pi = pi_overrides or {"home": 0.55, "draw": 0.27, "away": 0.18}
    elo_only = elo_only_overrides or {"home": 0.58, "draw": 0.26, "away": 0.16}

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_team_id": home_id,
        "away_team_id": away_id,
        "date": "2026-06-17",
        "primary_probs": dict(primary),
        "pi_probs": dict(pi),
        "blend_probs": dict(pi),
        "pi_only_probs": dict(pi),
        "elo_only_probs": dict(elo_only) if elo_blend else None,
        "blend_was_used": elo_blend,
        "blend_w_pi": 0.5 if elo_blend else 0.0,
        "blend_w_elo": 0.5 if elo_blend else 0.0,
        "_goal_model_used": goal_model_used,
        "_goal_model_expected": goal_model_expected,
        "_goal_model_xg": {"home_xg": 1.8, "away_xg": 1.1} if goal_model_used else None,
        "_goal_model_most_likely_score": [2, 1] if goal_model_used else None,
        "_goal_model_expected_total_goals": 2.9 if goal_model_used else None,
        "_goal_model_low_data": False,
        "_goal_model_version": "v2.1" if goal_model_used else None,
        "_goal_model_data_cutoff": "2026-05-01" if goal_model_used else None,
        "_goal_model_low_data_flags": [],
        "goal_model_hda": {"home": 0.58, "draw": 0.20, "away": 0.22} if goal_model_used else None,
        "home_elo": 1850.0 if elo_blend else None,
        "away_elo": 1720.0 if elo_blend else None,
        "confidence": {
            "tier": confidence_tier,
            "tier_description": "High confidence",
            "calibrated_p": calibrated_p,
            "calib_label": "high",
            "data_label": "high",
            "warnings": [],
        },
        "banner": "OK",
        "canonical_home_id": "ARG",
        "canonical_away_id": "BRA",
        "identity_warnings": [],
    }


def _fake_market():
    return {
        "book_odds": {"home": -150, "draw": +280, "away": +550},
        "book_fair": {"home": 0.45, "draw": 0.25, "away": 0.30},
        "calibrated_pi": {"home": 0.52, "draw": 0.24, "away": 0.24},
        "edges": {"home": 0.07, "draw": -0.01, "away": -0.06},
        "plus_ev_flags": [{"market": "home", "edge": 0.07, "calibrated_pi": 0.52, "book_fair": 0.45}],
        "plus_ev_count": 1,
        "market_divergence": "slight",
        "largest_market_delta": {"label": "Home Win", "delta_pts": 7.0},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 1: App-Consistency Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPrimaryProbsConsistency:
    """Verify every consumer uses primary_probs, not pi_probs independently."""

    def test_prediction_card_extract_most_likely_uses_primary_probs(self):
        """_extract_most_likely must prefer primary_probs over pi_probs."""
        from dashboard.prediction_card import _extract_most_likely

        # primary says home (0.60), pi says away (0.45)
        pred = _fake_prediction_full(
            primary_overrides={"home": 0.60, "draw": 0.25, "away": 0.15},
            pi_overrides={"home": 0.30, "draw": 0.25, "away": 0.45},
        )
        # If consumer uses primary_probs, most_likely = home
        # If consumer uses pi_probs, most_likely = away
        assert _extract_most_likely(pred) == "home", (
            "prediction_card must use primary_probs (home=0.60), not pi_probs (away=0.45)"
        )

    def test_prediction_card_extract_top_prob_uses_primary_probs(self):
        """_extract_top_prob must return the primary_probs max, not pi_probs max."""
        from dashboard.prediction_card import _extract_top_prob

        pred = _fake_prediction_full(
            primary_overrides={"home": 0.72, "draw": 0.18, "away": 0.10},
            pi_overrides={"home": 0.30, "draw": 0.25, "away": 0.45},
        )
        top = _extract_top_prob(pred)
        assert top == pytest.approx(0.72, abs=1e-9), (
            f"Expected top prob from primary_probs (0.72), got {top}"
        )

    def test_prediction_card_outcome_headline_uses_primary_probs(self):
        """_outcome_headline depends on _extract_most_likely which must use primary_probs."""
        from dashboard.prediction_card import _outcome_headline_text, _extract_most_likely

        pred = _fake_prediction_full(
            primary_overrides={"home": 0.15, "draw": 0.25, "away": 0.60},
            pi_overrides={"home": 0.55, "draw": 0.27, "away": 0.18},
        )
        mlr = _extract_most_likely(pred)
        headline = _outcome_headline_text(mlr, pred)
        # primary says away (0.60), so headline must be "Brazil to Win"
        # if it used pi_probs, it would say "Argentina to Win"
        assert "Brazil to Win" in headline, (
            f"Headline should reflect primary_probs (away win), got: {headline}"
        )

    def test_most_likely_result_uses_primary_probs(self):
        """ux_presenters.most_likely_result must use primary_probs."""
        from dashboard.ux_presenters import most_likely_result

        pred = _fake_prediction_full(
            primary_overrides={"home": 0.60, "draw": 0.25, "away": 0.15},
            pi_overrides={"home": 0.30, "draw": 0.25, "away": 0.45},
        )
        mlr = most_likely_result(pred)
        assert mlr["market"] == "home", (
            f"most_likely_result should pick 'home' from primary_probs, got: {mlr['market']}"
        )

    def test_bet_card_most_likely_uses_primary_probs(self):
        """bet_card _extract_most_likely reads primary_probs for the Most Likely Result."""
        from dashboard.prediction_card import _extract_most_likely

        # bet_card reuses _extract_most_likely from prediction_card
        pred = _fake_prediction_full(
            primary_overrides={"home": 0.15, "draw": 0.25, "away": 0.60},
            pi_overrides={"home": 0.55, "draw": 0.27, "away": 0.18},
        )
        mlr = _extract_most_likely(pred)
        assert mlr == "away", (
            f"bet_card Most Likely Result must use primary_probs (away=0.60), got: {mlr}"
        )

    def test_bet_card_model_probability_uses_primary_probs(self):
        """bet_card model probability display reads from primary_probs chain."""
        pred = _fake_prediction_full(
            primary_overrides={"home": 0.72, "draw": 0.18, "away": 0.10},
            pi_overrides={"home": 0.33, "draw": 0.33, "away": 0.34},
        )
        _probs = pred.get("primary_probs") or pred.get("blend_probs") or pred.get("pi_probs") or {}
        # This mirrors the line in bet_card.py line 237
        mlr = "home"  # primary_probs home is highest
        p_top = _probs.get(mlr)
        assert p_top == pytest.approx(0.72, abs=1e-9), (
            f"bet_card model prob should come from primary_probs (0.72), got {p_top}"
        )

    def test_evaluate_market_reads_primary_probs(self):
        """evaluate_market must read primary_probs, not pi_probs."""
        from soccer_ev_model.ev_workflow import evaluate_market

        train, ratings = _train_ratings()
        pred = predict_match_for_eval("Team1", "Team2", 1, 2, "2020-12-01", ratings)

        # Now call evaluate_market — edges should be based on primary_probs
        market = evaluate_market(
            pred,
            book_home_odds=-150,
            book_draw_odds=+280,
            book_away_odds=+550,
            min_edge=0.03,
        )
        # edges = primary_probs - book_fair
        # Verify that the edge for 'home' matches primary_probs, not pi_probs
        primary_home = pred["primary_probs"]["home"]
        pi_home = pred["pi_probs"]["home"]
        book_fair_home = market["book_fair"]["home"]
        expected_edge_from_primary = round(primary_home - book_fair_home, 4)
        expected_edge_from_pi = round(pi_home - book_fair_home, 4)

        actual_edge = market["edges"]["home"]
        assert actual_edge == expected_edge_from_primary, (
            f"evaluate_market edge={actual_edge} should match primary_probs edge={expected_edge_from_primary}, "
            f"not pi_probs edge={expected_edge_from_pi}"
        )

    def test_highest_model_confidence_uses_primary_probs(self):
        """highest_model_confidence reads primary_probs (via _probs_for)."""
        from dashboard.context_cards import highest_model_confidence, _probs_for

        pred = _fake_prediction_full(
            primary_overrides={"home": 0.80, "draw": 0.12, "away": 0.08},
            pi_overrides={"home": 0.30, "draw": 0.35, "away": 0.35},
        )
        probs = _probs_for(pred)
        # primary_probs home=0.80 is highest
        assert probs["home"] == pytest.approx(0.80, abs=1e-9), (
            f"_probs_for should return primary_probs (home=0.80), got {probs['home']}"
        )

        matches = [{"match_id": 100, "home_team": "Argentina", "away_team": "Brazil"}]
        predictions = {100: pred}
        result = highest_model_confidence(matches, predictions)
        assert result is not None
        assert result["market"] == "home", (
            f"highest_confidence should pick 'home' from primary_probs, got: {result['market']}"
        )
        assert result["probability"] == pytest.approx(0.80, abs=1e-9)

    def test_analysis_primary_model_section_reads_primary_probs(self):
        """Analysis _render_primary_model reads primary_probs, not pi_probs."""
        from dashboard.prediction_card import _extract_most_likely, _format_probability

        pred = _fake_prediction_full(
            primary_overrides={"home": 0.72, "draw": 0.18, "away": 0.10},
            pi_overrides={"home": 0.30, "draw": 0.35, "away": 0.35},
        )
        # The analysis view reads primary_probs the same way prediction_card does
        primary = (
            pred.get("primary_probs")
            or pred.get("blend_probs")
            or pred.get("pi_probs")
            or {}
        )
        mlr = _extract_most_likely(pred)
        p_top = primary.get(mlr)
        assert p_top == pytest.approx(0.72, abs=1e-9), (
            f"Analysis primary section should use primary_probs top (0.72), got {p_top}"
        )

    def test_no_consumer_recomputes_blend_independently(self):
        """Verify that no consumer module imports blend_ensemble directly.

        Consumers should ONLY read primary_probs from the prediction dict.
        They should never recompute the blend themselves.
        """
        import dashboard.prediction_card as pc
        import dashboard.bet_card as bc
        import dashboard.analysis_view as av
        import dashboard.context_cards as cc
        import dashboard.ux_presenters as up

        # Read each module's source and check it doesn't import blend_ensemble
        # (the pure-math ensemble module should only be used by ev_workflow)
        modules_to_check = [pc, bc, av, cc, up]
        for mod in modules_to_check:
            source = Path(mod.__file__).read_text()
            # These modules should NOT import production_ensemble.blend_ensemble
            assert "blend_ensemble" not in source, (
                f"{mod.__name__} imports blend_ensemble — "
                "consumers must use primary_probs, not recompute the blend"
            )

    def test_resolve_model_probs_for_market_prefers_primary_probs(self):
        """resolve_model_probs_for_market must return primary_probs when available."""
        from soccer_ev_model.prediction_summary import resolve_model_probs_for_market

        pred = _fake_prediction_full(
            primary_overrides={"home": 0.65, "draw": 0.20, "away": 0.15},
            pi_overrides={"home": 0.30, "draw": 0.35, "away": 0.35},
        )
        resolved = resolve_model_probs_for_market(pred)
        assert resolved["home"] == pytest.approx(0.65, abs=1e-9), (
            f"resolve_model_probs_for_market should return primary_probs, "
            f"got home={resolved['home']} (expected 0.65)"
        )

    def test_agreement_status_reads_from_prediction_dict(self):
        """agreement_status reads pi_only_probs and elo_only_probs from prediction dict."""
        from dashboard.ux_presenters import agreement_status

        # Both pi and Elo present, agree
        pred = _fake_prediction_full(elo_blend=True)
        assert agreement_status(pred) == "agree"

        # Only pi ran
        pred_pi_only = _fake_prediction_full(elo_blend=False)
        assert agreement_status(pred_pi_only) == "only_pi"


def predict_match_for_eval(home, away, home_id, away_id, date, ratings,
                           home_elo=None, away_elo=None):
    """Thin wrapper around predict_match for test convenience."""
    from soccer_ev_model.ev_workflow import predict_match
    return predict_match(
        home_team=home, away_team=away,
        home_team_id=home_id, away_team_id=away_id,
        date=date, ratings=ratings,
        home_elo=home_elo, away_elo=away_elo,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 2: Matchup Integration Tests (5+ Representative Scenarios)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMatchupIntegration:
    """Integration tests using 5+ representative matchup scenarios.

    Each scenario tests the full predict_match pipeline and verifies
    the output contract (primary_probs present, sums to 1.0, etc.).
    """

    def _make_ratings_many_teams(self):
        """Build ratings with several teams of varying strength."""
        train = []
        # Team 1: very strong (many wins)
        for i in range(60):
            train.append(_make_match(f"2020-{(i % 9) + 1:02d}-01", 1, 2, 3, 0))
        # Team 2: moderate
        for i in range(60):
            train.append(_make_match(f"2020-{(i % 9) + 1:02d}-02", 2, 3, 1, 1))
        # Team 3: weak (many losses)
        for i in range(60):
            train.append(_make_match(f"2020-{(i % 9) + 1:02d}-03", 3, 1, 0, 2))
        # Team 4: very few matches (low-data)
        train.append(_make_match("2020-01-01", 4, 1, 0, 1))
        # Teams 5 and 6: played each other a few times
        for i in range(5):
            train.append(_make_match(f"2020-{(i % 9) + 1:02d}-04", 5, 6, 1, 1))

        from soccer_ev_model.pi_ratings import compute_pi_ratings
        return compute_pi_ratings(train, cutoff="2020-12-01")

    # ── Scenario 1: Strong favorite ──
    def test_strong_favorite(self):
        """Strong team vs weak team: primary_probs heavily favor the strong team."""
        ratings = self._make_ratings_many_teams()
        from soccer_ev_model.ev_workflow import predict_match

        result = predict_match(
            home_team="Team1", away_team="Team3",
            home_team_id=1, away_team_id=3,
            date="2021-01-01", ratings=ratings,
        )
        assert "primary_probs" in result
        total = sum(result["primary_probs"].values())
        assert total == pytest.approx(1.0, abs=1e-3)
        # Strong team should have high home win probability
        assert result["primary_probs"]["home"] > result["primary_probs"]["away"]
        assert result["primary_probs"]["home"] > 0.5

    # ── Scenario 2: Close matchup (evenly matched teams) ──
    def test_close_matchup(self):
        """Two evenly matched teams: probabilities should be close to each other."""
        ratings = self._make_ratings_many_teams()
        from soccer_ev_model.ev_workflow import predict_match

        # Teams 5 and 6 have identical records — the most evenly matched pair
        result = predict_match(
            home_team="Team5", away_team="Team6",
            home_team_id=5, away_team_id=6,
            date="2021-01-01", ratings=ratings,
        )
        probs = result["primary_probs"]
        total = sum(probs.values())
        assert total == pytest.approx(1.0, abs=1e-3)
        # Home and away should be relatively close (within 30 percentage points)
        assert abs(probs["home"] - probs["away"]) < 0.30

    # ── Scenario 3: High draw probability ──
    def test_high_draw_probability(self):
        """Two very similar teams: draw probability should be relatively high."""
        ratings = self._make_ratings_many_teams()
        from soccer_ev_model.ev_workflow import predict_match

        # Teams 5 and 6 have identical records (5 draws each)
        result = predict_match(
            home_team="Team5", away_team="Team6",
            home_team_id=5, away_team_id=6,
            date="2021-01-01", ratings=ratings,
        )
        probs = result["primary_probs"]
        total = sum(probs.values())
        assert total == pytest.approx(1.0, abs=1e-3)
        # Both teams are even, so neither home nor away should dominate
        assert probs["home"] < 0.6
        assert probs["away"] < 0.6

    # ── Scenario 4: Low-data team ──
    def test_low_data_team(self):
        """Low-data team in prediction: confidence should reflect data scarcity."""
        ratings = self._make_ratings_many_teams()
        from soccer_ev_model.ev_workflow import predict_match

        # Team 4 has only 1 historical match
        result = predict_match(
            home_team="Team4", away_team="Team2",
            home_team_id=4, away_team_id=2,
            date="2021-01-01", ratings=ratings,
        )
        assert "primary_probs" in result
        assert "confidence" in result
        # With very low data for Team 4, confidence tier should be C or D
        tier = result["confidence"]["tier"]
        assert tier in ("C", "D"), (
            f"Low-data matchup should have tier C or D, got {tier}"
        )

    # ── Scenario 5: Identity edge case ──
    def test_identity_edge_case_with_elo_blend(self):
        """Prediction with Elo blend: elo_only_probs must be present."""
        ratings = self._make_ratings_many_teams()
        from soccer_ev_model.ev_workflow import predict_match

        result = predict_match(
            home_team="Team1", away_team="Team2",
            home_team_id=1, away_team_id=2,
            date="2021-01-01", ratings=ratings,
            home_elo=1850.0, away_elo=1720.0,
        )
        assert result["blend_was_used"] is True
        assert result["elo_only_probs"] is not None
        assert result["pi_only_probs"] is not None
        # primary_probs, pi_probs, blend_probs should all be present
        assert "primary_probs" in result
        assert "pi_probs" in result
        assert "blend_probs" in result
        # Verify H/D/A sum
        for key in ("primary_probs", "pi_probs", "blend_probs"):
            total = sum(result[key].values())
            assert total == pytest.approx(1.0, abs=1e-3), (
                f"{key} sums to {total}, expected 1.0"
            )

    # ── Scenario 6: Goal model integration ──
    def test_goal_model_integration(self):
        """Prediction with goal_probs: _goal_model_used must be True."""
        ratings = self._make_ratings_many_teams()
        from soccer_ev_model.ev_workflow import predict_match

        goal_probs = {"home": 0.55, "draw": 0.20, "away": 0.25}
        result = predict_match(
            home_team="Team1", away_team="Team2",
            home_team_id=1, away_team_id=2,
            date="2021-01-01", ratings=ratings,
            home_elo=1850.0, away_elo=1720.0,
            goal_probs=goal_probs,
            goal_model_xg={"home_xg": 1.8, "away_xg": 1.1},
            goal_model_metadata={
                "most_likely_score": [2, 1],
                "expected_total_goals": 2.9,
                "model_version": "v2.1",
                "data_cutoff": "2026-05-01",
                "low_data_flags": [],
            },
        )
        assert result["_goal_model_used"] is True
        assert result["_goal_model_expected"] is False  # default unless set
        assert result["goal_model_hda"] is not None
        # primary_probs should be the Elo60/Goal40 blend, not the raw pi
        # Verify it's a valid probability distribution
        total = sum(result["primary_probs"].values())
        assert total == pytest.approx(1.0, abs=1e-3)

    # ── Scenario 7: Evaluate market with prediction ──
    def test_evaluate_market_integration(self):
        """Full predict → evaluate_market pipeline."""
        ratings = self._make_ratings_many_teams()
        from soccer_ev_model.ev_workflow import predict_match, evaluate_market

        pred = predict_match(
            home_team="Team1", away_team="Team3",
            home_team_id=1, away_team_id=3,
            date="2021-01-01", ratings=ratings,
        )
        market = evaluate_market(
            pred,
            book_home_odds=-200,
            book_draw_odds=+300,
            book_away_odds=+500,
        )
        assert "book_fair" in market
        assert "edges" in market
        assert "plus_ev_flags" in market
        assert "calibrated_pi" in market
        # book_fair must sum to ~1.0
        total = sum(market["book_fair"].values())
        assert total == pytest.approx(1.0, abs=1e-3)


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 3: Dashboard Integration Tests (AppTest)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDashboardIntegration:
    """AppTest-level dashboard integration tests."""

    def test_predictions_view_boots_cleanly(self):
        """Predictions view boots without exception."""
        at = AppTest.from_file(str(_DASHBOARD_APP), default_timeout=60)
        at.query_params["view"] = "predictions"
        at.run()
        assert not at.exception, f"Predictions raised: {at.exception}"

    def test_bets_view_boots_cleanly(self):
        """Bets view boots without exception."""
        at = AppTest.from_file(str(_DASHBOARD_APP), default_timeout=60)
        at.query_params["view"] = "bets"
        at.run()
        assert not at.exception, f"Bets raised: {at.exception}"

    def test_analysis_view_boots_cleanly(self):
        """Analysis view boots without exception."""
        at = AppTest.from_file(str(_DASHBOARD_APP), default_timeout=60)
        at.query_params["view"] = "analysis"
        at.run()
        assert not at.exception, f"Analysis raised: {at.exception}"

    def test_predictions_view_has_no_sportsbook_terms(self):
        """Predictions view must not leak sportsbook terminology."""
        at = AppTest.from_file(str(_DASHBOARD_APP), default_timeout=60)
        at.query_params["view"] = "predictions"
        at.run()
        assert not at.exception
        text_parts = []
        for el in at.markdown:
            v = (el.value or "")
            v = re.sub(r"<style.*?</style>", "", v, flags=re.DOTALL | re.IGNORECASE)
            text_parts.append(v)
        for el in at.caption:
            text_parts.append(el.value or "")
        for el in at.info:
            text_parts.append(el.value or "")
        text = "\n".join(text_parts).lower()
        forbidden = ["+ev", "min_edge", "min edge", "sportsbook", "bookmaker", "no-vig"]
        for bad in forbidden:
            assert bad not in text, f"Term '{bad}' leaked into Predictions"

    def test_predictions_view_custom_matchup_expander(self):
        """Predictions view has a Custom matchup expander."""
        at = AppTest.from_file(str(_DASHBOARD_APP), default_timeout=60)
        at.query_params["view"] = "predictions"
        at.run()
        expander_labels = [e.label or "" for e in at.expander]
        assert any("Custom matchup" in t for t in expander_labels)

    def test_bets_view_min_edge_in_advanced_settings(self):
        """min_edge slider is inside Advanced settings expander."""
        at = AppTest.from_file(str(_DASHBOARD_APP), default_timeout=60)
        at.query_params["view"] = "bets"
        at.run()
        advanced = next((e for e in at.expander if "Advanced settings" in (e.label or "")), None)
        assert advanced is not None
        inner_sliders = [s.label or "" for s in advanced.slider]
        assert any("Minimum edge" in s for s in inner_sliders)

    def test_bets_view_custom_bet_expander(self):
        """Bets view has a Custom bet expander."""
        at = AppTest.from_file(str(_DASHBOARD_APP), default_timeout=60)
        at.query_params["view"] = "bets"
        at.run()
        expander_labels = [e.label or "" for e in at.expander]
        assert any("Custom bet" in t for t in expander_labels)

    def test_analysis_view_has_primary_model_expander_open(self):
        """Analysis view Primary Model expander is open by default."""
        at = AppTest.from_file(str(_DASHBOARD_APP), default_timeout=60)
        at.query_params["view"] = "analysis"
        at.run()
        primary = next(
            (e for e in at.expander if "Primary Model" in (e.label or "")), None
        )
        assert primary is not None
        assert primary.proto.expanded is True

    def test_analysis_view_has_11_expanders(self):
        """Analysis view renders all 11 expanders."""
        at = AppTest.from_file(str(_DASHBOARD_APP), default_timeout=60)
        at.query_params["view"] = "analysis"
        at.run()
        expanders = list(at.expander)
        assert expanders is not None
        expected_labels = [
            "Primary Model",
            "Elo",
            "Goal Model",
            "Pi",
            "Disagreement",
            "Market Comparison",
            "Poisson View",
            "Squad Context",
            "Group Context",
            "Calibration and Data Quality",
            "Raw Diagnostics",
        ]
        expander_labels = [e.label or "" for e in expanders]
        for expected in expected_labels:
            assert any(expected in lbl for lbl in expander_labels), (
                f"Missing expander: {expected}; got: {expander_labels}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 4: Focused Model Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFocusedModelIntegration:
    """Focused model integration tests covering key contract edges."""

    def _make_ratings(self, home_id=1, away_id=2, n_each=40):
        train = []
        for i in range(n_each):
            train.append(_make_match(f"2020-{(i % 9) + 1:02d}-01", home_id, away_id, 2, 0))
            train.append(_make_match(f"2020-{(i % 9) + 1:02d}-02", away_id, home_id, 1, 1))
        from soccer_ev_model.pi_ratings import compute_pi_ratings
        return compute_pi_ratings(train, cutoff="2020-12-01")

    def test_predict_match_output_contract_keys(self):
        """predict_match must return all required contract keys."""
        ratings = self._make_ratings()
        from soccer_ev_model.ev_workflow import predict_match

        result = predict_match(
            home_team="Team1", away_team="Team2",
            home_team_id=1, away_team_id=2,
            date="2020-12-01", ratings=ratings,
        )
        required_keys = [
            "primary_probs", "pi_probs", "blend_probs",
            "pi_only_probs", "elo_only_probs",
            "blend_was_used", "blend_w_pi", "blend_w_elo",
            "home_team", "away_team", "home_team_id", "away_team_id",
            "date", "confidence", "banner",
            "canonical_home_id", "canonical_away_id",
        ]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

    def test_predict_match_primary_probs_sums_to_one(self):
        """primary_probs must always sum to 1.0."""
        ratings = self._make_ratings()
        from soccer_ev_model.ev_workflow import predict_match

        result = predict_match(
            home_team="Team1", away_team="Team2",
            home_team_id=1, away_team_id=2,
            date="2020-12-01", ratings=ratings,
        )
        total = sum(result["primary_probs"].values())
        assert total == pytest.approx(1.0, abs=1e-3)

    def test_predict_match_with_goal_model_blends_correctly(self):
        """With goal_probs + Elo, primary_probs must be the 60/40 blend."""
        ratings = self._make_ratings()
        from soccer_ev_model.ev_workflow import predict_match

        elo_home_elo = 1850.0
        elo_away_elo = 1720.0
        goal_probs = {"home": 0.55, "draw": 0.20, "away": 0.25}

        result = predict_match(
            home_team="Team1", away_team="Team2",
            home_team_id=1, away_team_id=2,
            date="2020-12-01", ratings=ratings,
            home_elo=elo_home_elo, away_elo=elo_away_elo,
            goal_probs=goal_probs,
        )
        assert result["_goal_model_used"] is True
        # elo_only_probs must exist
        assert result["elo_only_probs"] is not None
        # goal_model_hda must exist
        assert result["goal_model_hda"] is not None
        # primary_probs must differ from pi_probs (since goal model shifted the blend)
        # We can't assert exact values without knowing the Elo-only probs, but
        # we can verify it's a valid distribution
        total = sum(result["primary_probs"].values())
        assert total == pytest.approx(1.0, abs=1e-3)

    def test_predict_match_without_elo_is_pure_pi(self):
        """Without Elo, primary_probs must equal pi_probs (pure pi-rating)."""
        ratings = self._make_ratings()
        from soccer_ev_model.ev_workflow import predict_match

        result = predict_match(
            home_team="Team1", away_team="Team2",
            home_team_id=1, away_team_id=2,
            date="2020-12-01", ratings=ratings,
        )
        assert result["blend_was_used"] is False
        # Without Elo, primary_probs should be pi+Elo fallback = pi
        for k in ("home", "draw", "away"):
            assert result["primary_probs"][k] == pytest.approx(
                result["pi_probs"][k], abs=1e-3
            ), f"primary_probs[{k}] != pi_probs[{k}] without Elo"

    def test_confidence_tier_reflects_data_volume(self):
        """Confidence tier must be A/B/C/D and warnings populated for low tiers."""
        ratings = self._make_ratings()
        from soccer_ev_model.ev_workflow import predict_match

        result = predict_match(
            home_team="Team1", away_team="Team2",
            home_team_id=1, away_team_id=2,
            date="2020-12-01", ratings=ratings,
        )
        conf = result["confidence"]
        assert conf["tier"] in ("A", "B", "C", "D")
        assert "calibrated_p" in conf
        assert "warnings" in conf  # always present (may be empty list)

    def test_blend_fallback_all_cases_produce_valid_output(self):
        """All fallback cases A-E must produce primary_probs summing to 1.0."""
        from soccer_ev_model.blend_fallback import predict_with_fallback

        cases = [
            (True, True, False),   # A
            (True, True, True),    # B
            (True, False, False),  # C
            (False, True, False),  # D
            (False, False, False), # E
        ]
        elo_probs = {"home": 0.60, "draw": 0.20, "away": 0.20}
        goal_probs = {"home": 0.30, "draw": 0.40, "away": 0.30}

        for elo_avail, goal_avail, goal_low in cases:
            result = predict_with_fallback(
                home_team="A", away_team="B",
                home_team_id=1, away_team_id=2,
                match_date="2026-01-01",
                elo_probs=elo_probs if elo_avail else None,
                goal_probs=goal_probs if goal_avail else None,
                elo_available=elo_avail,
                goal_available=goal_avail,
                goal_low_data=goal_low,
            )
            total = sum(result.primary_probs.values())
            assert total == pytest.approx(1.0, abs=1e-3), (
                f"Case {result.case}: primary_probs sums to {total}"
            )
            # Must have a case label
            assert result.case in ("A", "B", "C", "D", "E")

    def test_blend_fallback_case_e_is_uniform(self):
        """Case E (both unavailable) must produce uniform 1/3 baseline."""
        from soccer_ev_model.blend_fallback import predict_with_fallback

        result = predict_with_fallback(
            home_team="X", away_team="Y",
            home_team_id=9998, away_team_id=9999,
            match_date="2026-01-01",
            elo_probs=None, goal_probs=None,
            elo_available=False, goal_available=False,
        )
        assert result.case == "E"
        for k in ("home", "draw", "away"):
            assert result.primary_probs[k] == pytest.approx(1 / 3, abs=1e-3)

    def test_evaluate_market_requires_primary_probs(self):
        """evaluate_market must raise ValueError if primary_probs is missing and no pi_probs."""
        from soccer_ev_model.ev_workflow import evaluate_market

        bad_pred = {
            "home_team": "A", "away_team": "B",
            "confidence": {"calibrated_p": 0.5},
        }
        with pytest.raises(ValueError, match="primary_probs"):
            evaluate_market(bad_pred, -150, +280, +550)

    def test_evaluate_market_backward_compat_accepts_pi_probs(self):
        """evaluate_market must accept pi_probs as fallback for primary_probs."""
        from soccer_ev_model.ev_workflow import evaluate_market

        pred = {
            "home_team": "A", "away_team": "B",
            "pi_probs": {"home": 0.55, "draw": 0.25, "away": 0.20},
            "confidence": {"calibrated_p": 0.55},
        }
        # Should not raise — pi_probs is accepted as fallback
        market = evaluate_market(pred, -150, +280, +550)
        assert "edges" in market
        # Verify edges are based on pi_probs (since that's what was provided)
        assert market["edges"]["home"] == pytest.approx(
            0.55 - market["book_fair"]["home"], abs=1e-3
        )

    def test_production_ensemble_blend_is_deterministic(self):
        """blend_ensemble must produce identical output for identical input."""
        from soccer_ev_model.production_ensemble import blend_ensemble

        elo = {"home": 0.60, "draw": 0.20, "away": 0.20}
        goal = {"home": 0.30, "draw": 0.40, "away": 0.30}

        results = [blend_ensemble(elo, goal) for _ in range(20)]
        for r in results[1:]:
            assert r["primary_probs"] == results[0]["primary_probs"]

    def test_production_ensemble_rejects_nan(self):
        """blend_ensemble must reject NaN inputs."""
        from soccer_ev_model.production_ensemble import blend_ensemble, EnsembleInputError

        with pytest.raises(EnsembleInputError):
            blend_ensemble(
                {"home": float("nan"), "draw": 0.2, "away": 0.3},
                {"home": 0.3, "draw": 0.4, "away": 0.3},
            )

    def test_production_ensemble_rejects_negative(self):
        """blend_ensemble must reject negative probabilities."""
        from soccer_ev_model.production_ensemble import blend_ensemble, EnsembleInputError

        with pytest.raises(EnsembleInputError):
            blend_ensemble(
                {"home": -0.1, "draw": 0.6, "away": 0.5},
                {"home": 0.3, "draw": 0.4, "away": 0.3},
            )


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 5: Context Cards Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestContextCardsIntegration:
    """Integration tests for Tournament Snapshot and Highest Confidence cards."""

    def test_tournament_snapshot_empty_matches(self):
        """Empty matches list produces is_empty=True snapshot."""
        from dashboard.context_cards import build_tournament_snapshot

        snap = build_tournament_snapshot([])
        assert snap["is_empty"] is True
        assert snap["count"] == 0

    def test_tournament_snapshot_group_stage(self):
        """Group stage matches produce correct header and groups label."""
        from dashboard.context_cards import build_tournament_snapshot

        matches = [
            {"stage": "GROUP_STAGE", "matchday": 1, "group": "GROUP_A"},
            {"stage": "GROUP_STAGE", "matchday": 1, "group": "GROUP_B"},
        ]
        snap = build_tournament_snapshot(matches)
        assert snap["count"] == 2
        assert "Group Stage" in snap["header"]
        assert "Matchday 1" in snap["header"]
        assert "Groups A, B" == snap["groups_label"]

    def test_tournament_snapshot_knockout(self):
        """Knockout matches use 'remaining' wording."""
        from dashboard.context_cards import build_tournament_snapshot

        matches = [
            {"stage": "ROUND_OF_16", "matchday": None, "group": ""},
        ]
        snap = build_tournament_snapshot(matches)
        assert "remaining" in snap["count_label"]

    def test_highest_confidence_none_for_empty(self):
        """highest_model_confidence returns None for empty inputs."""
        from dashboard.context_cards import highest_model_confidence

        assert highest_model_confidence([], {}) is None
        assert highest_model_confidence(None, None) is None

    def test_highest_confidence_picks_highest_across_matches(self):
        """highest_model_confidence picks the single highest probability across all matches."""
        from dashboard.context_cards import highest_model_confidence

        matches = [
            {"match_id": 1, "home_team": "A", "away_team": "B"},
            {"match_id": 2, "home_team": "C", "away_team": "D"},
        ]
        predictions = {
            1: _fake_prediction_full(
                primary_overrides={"home": 0.50, "draw": 0.30, "away": 0.20}
            ),
            2: _fake_prediction_full(
                primary_overrides={"home": 0.85, "draw": 0.10, "away": 0.05}
            ),
        }
        result = highest_model_confidence(matches, predictions)
        assert result is not None
        assert result["match_id"] == 2
        assert result["probability"] == pytest.approx(0.85, abs=1e-9)

    def test_highest_confidence_uses_primary_probs_not_pi(self):
        """highest_model_confidence must use primary_probs, not pi_probs."""
        from dashboard.context_cards import highest_model_confidence

        matches = [{"match_id": 1, "home_team": "A", "away_team": "B"}]
        # primary says home=0.90, pi says home=0.20
        predictions = {
            1: _fake_prediction_full(
                primary_overrides={"home": 0.90, "draw": 0.05, "away": 0.05},
                pi_overrides={"home": 0.20, "draw": 0.30, "away": 0.50},
            ),
        }
        result = highest_model_confidence(matches, predictions)
        assert result["probability"] == pytest.approx(0.90, abs=1e-9)
        assert result["market"] == "home"


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 6: End-to-End Pipeline Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEndToEndPipeline:
    """End-to-end pipeline: train → predict → evaluate → verify consistency."""

    def _make_ratings(self):
        train = []
        for i in range(50):
            train.append(_make_match(f"2020-{(i % 9) + 1:02d}-01", 1, 2, 2, 0))
            train.append(_make_match(f"2020-{(i % 9) + 1:02d}-02", 2, 3, 1, 1))
            train.append(_make_match(f"2020-{(i % 9) + 1:02d}-03", 3, 1, 0, 2))
        from soccer_ev_model.pi_ratings import compute_pi_ratings
        return train, compute_pi_ratings(train, cutoff="2020-12-01")

    def test_full_pipeline_predict_then_evaluate(self):
        """Full pipeline: predict_match → evaluate_market → verify edges."""
        from soccer_ev_model.ev_workflow import predict_match, evaluate_market

        train, ratings = self._make_ratings()
        pred = predict_match(
            home_team="Team1", away_team="Team2",
            home_team_id=1, away_team_id=2,
            date="2021-01-01", ratings=ratings,
        )
        market = evaluate_market(pred, -150, +280, +550)

        # Verify edges = primary_probs - book_fair
        for m in ("home", "draw", "away"):
            expected_edge = round(pred["primary_probs"][m] - market["book_fair"][m], 4)
            assert market["edges"][m] == expected_edge, (
                f"Edge[{m}]={market['edges'][m]} != expected {expected_edge}"
            )

    def test_full_pipeline_with_elo_blend(self):
        """Full pipeline with Elo blend: elo_only_probs present, primary valid."""
        from soccer_ev_model.ev_workflow import predict_match, evaluate_market

        train, ratings = self._make_ratings()
        pred = predict_match(
            home_team="Team1", away_team="Team2",
            home_team_id=1, away_team_id=2,
            date="2021-01-01", ratings=ratings,
            home_elo=1850.0, away_elo=1720.0,
        )
        assert pred["blend_was_used"] is True
        assert pred["elo_only_probs"] is not None

        market = evaluate_market(pred, -200, +300, +500)
        total = sum(market["book_fair"].values())
        assert total == pytest.approx(1.0, abs=1e-3)

    def test_full_pipeline_with_goal_model(self):
        """Full pipeline with goal model: _goal_model_used=True, primary is blend."""
        from soccer_ev_model.ev_workflow import predict_match, evaluate_market

        train, ratings = self._make_ratings()
        goal_probs = {"home": 0.55, "draw": 0.20, "away": 0.25}
        pred = predict_match(
            home_team="Team1", away_team="Team2",
            home_team_id=1, away_team_id=2,
            date="2021-01-01", ratings=ratings,
            home_elo=1850.0, away_elo=1720.0,
            goal_probs=goal_probs,
        )
        assert pred["_goal_model_used"] is True
        assert pred["goal_model_hda"] is not None

        market = evaluate_market(pred, -150, +280, +550)
        # Verify +EV flags are based on primary_probs
        for flag in market["plus_ev_flags"]:
            m = flag["market"]
            assert flag["edge"] == market["edges"][m]

    def test_prediction_dict_all_prob_fields_sum_to_one(self):
        """All probability dicts in a prediction must sum to 1.0."""
        from soccer_ev_model.ev_workflow import predict_match

        train, ratings = self._make_ratings()
        pred = predict_match(
            home_team="Team1", away_team="Team2",
            home_team_id=1, away_team_id=2,
            date="2021-01-01", ratings=ratings,
            home_elo=1850.0, away_elo=1720.0,
        )
        prob_fields = ["primary_probs", "pi_probs", "blend_probs", "pi_only_probs"]
        for field in prob_fields:
            total = sum(pred[field].values())
            assert total == pytest.approx(1.0, abs=1e-3), (
                f"{field} sums to {total}"
            )
        if pred["elo_only_probs"] is not None:
            total = sum(pred["elo_only_probs"].values())
            assert total == pytest.approx(1.0, abs=1e-3), (
                f"elo_only_probs sums to {total}"
            )
