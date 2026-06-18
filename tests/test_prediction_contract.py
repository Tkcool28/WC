"""Contract tests for the Phase 4 primary_probs migration.

Verifies that:
1. predict_match outputs primary_probs as the canonical official field
2. primary_probs, pi_probs, and blend_probs are identical aliases
3. All consumer resolution paths prefer primary_probs
4. evaluate_market reads primary_probs (not pi_probs)
5. resolve_model_probs_for_market prefers primary_probs over blend_probs/pi_probs
6. No consumer independently recomputes the blend
"""
import pytest

from soccer_ev_model.ev_workflow import (
    evaluate_market,
    evaluate_match,
    predict_match,
)
from soccer_ev_model.prediction_summary import resolve_model_probs_for_market


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _train_ratings():
    train = []
    for i in range(40):
        train.append(_make_match(f"2020-{(i % 9) + 1:02d}-01", 1, 2, 2, 0))
        train.append(_make_match(f"2020-{(i % 9) + 1:02d}-02", 2, 3, 1, 1))
    from soccer_ev_model.pi_ratings import compute_pi_ratings
    ratings = compute_pi_ratings(train, cutoff="2020-12-01")
    return train, ratings


# ---------------------------------------------------------------------------
# 1. predict_match outputs primary_probs
# ---------------------------------------------------------------------------

class TestPredictMatchPrimaryProbs:
    def test_primary_probs_key_present(self):
        """predict_match must include primary_probs in its output."""
        _, ratings = _train_ratings()
        result = predict_match(
            home_team="Team1", away_team="Team2",
            home_team_id=1, away_team_id=2,
            date="2020-12-01", ratings=ratings,
        )
        assert "primary_probs" in result

    def test_primary_probs_is_valid_prob_dict(self):
        """primary_probs must be a valid 3-outcome probability dict."""
        _, ratings = _train_ratings()
        result = predict_match(
            home_team="Team1", away_team="Team2",
            home_team_id=1, away_team_id=2,
            date="2020-12-01", ratings=ratings,
        )
        pp = result["primary_probs"]
        assert set(pp.keys()) == {"home", "draw", "away"}
        assert all(isinstance(v, float) for v in pp.values())
        assert all(0.0 <= v <= 1.0 for v in pp.values())
        assert abs(sum(pp.values()) - 1.0) < 1e-6

    def test_primary_probs_equals_pi_probs(self):
        """primary_probs and pi_probs must be identical (alias contract)."""
        _, ratings = _train_ratings()
        result = predict_match(
            home_team="Team1", away_team="Team2",
            home_team_id=1, away_team_id=2,
            date="2020-12-01", ratings=ratings,
        )
        assert result["primary_probs"] == result["pi_probs"]

    def test_primary_probs_equals_blend_probs(self):
        """primary_probs and blend_probs must be identical (alias contract)."""
        _, ratings = _train_ratings()
        result = predict_match(
            home_team="Team1", away_team="Team2",
            home_team_id=1, away_team_id=2,
            date="2020-12-01", ratings=ratings,
        )
        assert result["primary_probs"] == result["blend_probs"]

    def test_primary_probs_with_elo(self):
        """When Elo is provided, primary_probs reflects the pi+Elo blend."""
        _, ratings = _train_ratings()
        result = predict_match(
            home_team="Team1", away_team="Team2",
            home_team_id=1, away_team_id=2,
            date="2020-12-01", ratings=ratings,
            home_elo=1500, away_elo=1400,
        )
        assert "primary_probs" in result
        assert result["primary_probs"] == result["pi_probs"]
        assert result["primary_probs"] == result["blend_probs"]
        assert result["blend_was_used"] is True

    def test_primary_probs_without_elo(self):
        """Without Elo, primary_probs is pure pi-rating."""
        _, ratings = _train_ratings()
        result = predict_match(
            home_team="Team1", away_team="Team2",
            home_team_id=1, away_team_id=2,
            date="2020-12-01", ratings=ratings,
        )
        assert "primary_probs" in result
        assert result["blend_was_used"] is False


# ---------------------------------------------------------------------------
# 2. resolve_model_probs_for_market prefers primary_probs
# ---------------------------------------------------------------------------

class TestResolveModelProbs:
    def test_prefers_primary_probs(self):
        """resolve_model_probs_for_market must return primary_probs when present."""
        result = {
            "primary_probs": {"home": 0.5, "draw": 0.3, "away": 0.2},
            "blend_probs": {"home": 0.4, "draw": 0.3, "away": 0.3},
            "pi_probs": {"home": 0.3, "draw": 0.4, "away": 0.3},
        }
        probs = resolve_model_probs_for_market(result)
        assert probs == {"home": 0.5, "draw": 0.3, "away": 0.2}

    def test_falls_back_to_blend_probs(self):
        """Without primary_probs, falls back to blend_probs."""
        result = {
            "blend_probs": {"home": 0.4, "draw": 0.3, "away": 0.3},
            "pi_probs": {"home": 0.3, "draw": 0.4, "away": 0.3},
        }
        probs = resolve_model_probs_for_market(result)
        assert probs == {"home": 0.4, "draw": 0.3, "away": 0.3}

    def test_falls_back_to_pi_probs(self):
        """With only pi_probs, falls back to pi_probs."""
        result = {
            "pi_probs": {"home": 0.3, "draw": 0.4, "away": 0.3},
        }
        probs = resolve_model_probs_for_market(result)
        assert probs == {"home": 0.3, "draw": 0.4, "away": 0.3}

    def test_raises_when_none_present(self):
        """Raises KeyError when no prob dict is present."""
        with pytest.raises(KeyError):
            resolve_model_probs_for_market({})

    def test_ignores_none_values(self):
        """None values in prob keys are skipped, not returned."""
        result = {
            "primary_probs": None,
            "blend_probs": {"home": 0.4, "draw": 0.3, "away": 0.3},
        }
        probs = resolve_model_probs_for_market(result)
        assert probs == {"home": 0.4, "draw": 0.3, "away": 0.3}


# ---------------------------------------------------------------------------
# 3. evaluate_market uses primary_probs
# ---------------------------------------------------------------------------

class TestEvaluateMarketPrimaryProbs:
    def test_evaluate_market_uses_primary_probs(self):
        """evaluate_market must read primary_probs for edge calculations."""
        _, ratings = _train_ratings()
        pred = predict_match(
            home_team="Team1", away_team="Team2",
            home_team_id=1, away_team_id=2,
            date="2020-12-01", ratings=ratings,
        )
        market = evaluate_market(
            pred,
            book_home_odds=+200,
            book_draw_odds=+300,
            book_away_odds=+250,
        )
        # Edges must be computed against primary_probs
        primary = pred["primary_probs"]
        book_fair = market["book_fair"]
        for m in ("home", "draw", "away"):
            expected_edge = round(primary[m] - book_fair[m], 4)
            assert market["edges"][m] == expected_edge, (
                f"Edge for {m} should be computed from primary_probs, "
                f"got {market['edges'][m]}, expected {expected_edge}"
            )

    def test_evaluate_market_backward_compat_pi_probs(self):
        """evaluate_market accepts pi_probs when primary_probs is missing."""
        fake_pi = {"home": 0.5, "draw": 0.3, "away": 0.2}
        pred = {
            "pi_probs": fake_pi,
            "confidence": {"calibrated_p": 0.45},
        }
        market = evaluate_market(
            pred,
            book_home_odds=+200,
            book_draw_odds=+300,
            book_away_odds=+250,
        )
        # Should not raise; primary_probs was synthesized from pi_probs
        assert "edges" in market

    def test_evaluate_market_raises_without_any_probs(self):
        """evaluate_market raises when neither primary_probs nor pi_probs present."""
        pred = {"confidence": {"calibrated_p": 0.45}}
        with pytest.raises(ValueError, match="primary_probs"):
            evaluate_market(
                pred,
                book_home_odds=+200,
                book_draw_odds=+300,
                book_away_odds=+250,
            )


# ---------------------------------------------------------------------------
# 4. evaluate_match wrapper preserves primary_probs
# ---------------------------------------------------------------------------

class TestEvaluateMatchPrimaryProbs:
    def test_evaluate_match_includes_primary_probs(self):
        """evaluate_match wrapper must include primary_probs in output."""
        _, ratings = _train_ratings()
        result = evaluate_match(
            home_team="Team1", away_team="Team2",
            home_team_id=1, away_team_id=2,
            date="2020-12-01", ratings=ratings,
            book_home_odds=+200,
            book_draw_odds=+300,
            book_away_odds=+250,
        )
        assert "primary_probs" in result
        assert result["primary_probs"] == result["pi_probs"]
        assert result["primary_probs"] == result["blend_probs"]

    def test_evaluate_match_no_odds_preserves_primary_probs(self):
        """evaluate_match without odds still includes primary_probs."""
        _, ratings = _train_ratings()
        result = evaluate_match(
            home_team="Team1", away_team="Team2",
            home_team_id=1, away_team_id=2,
            date="2020-12-01", ratings=ratings,
            book_home_odds=None,
            book_draw_odds=None,
            book_away_odds=None,
        )
        assert "primary_probs" in result


# ---------------------------------------------------------------------------
# 5. Consumer consistency: all consumers see the same probs
# ---------------------------------------------------------------------------

class TestConsumerConsistency:
    def test_all_consumers_see_same_probs(self):
        """All consumer resolution paths must return the same prob dict."""
        _, ratings = _train_ratings()
        result = predict_match(
            home_team="Team1", away_team="Team2",
            home_team_id=1, away_team_id=2,
            date="2020-12-01", ratings=ratings,
            home_elo=1500, away_elo=1400,
        )

        # resolve_model_probs_for_market
        resolved = resolve_model_probs_for_market(result)
        assert resolved == result["primary_probs"]

        # Direct access patterns used by consumers
        primary = result["primary_probs"]
        assert result["pi_probs"] == primary
        assert result["blend_probs"] == primary

    def test_consumer_with_only_legacy_keys(self):
        """Consumers receiving legacy-only dicts still work via fallback."""
        legacy_result = {
            "pi_probs": {"home": 0.5, "draw": 0.3, "away": 0.2},
            "blend_probs": {"home": 0.5, "draw": 0.3, "away": 0.2},
        }
        resolved = resolve_model_probs_for_market(legacy_result)
        # primary_probs is absent, so blend_probs wins
        assert resolved == {"home": 0.5, "draw": 0.3, "away": 0.2}

    def test_primary_probs_takes_priority_over_all_aliases(self):
        """When primary_probs is present, it wins even if aliases differ."""
        result = {
            "primary_probs": {"home": 0.6, "draw": 0.25, "away": 0.15},
            "blend_probs": {"home": 0.5, "draw": 0.3, "away": 0.2},
            "pi_probs": {"home": 0.4, "draw": 0.35, "away": 0.25},
        }
        resolved = resolve_model_probs_for_market(result)
        assert resolved == {"home": 0.6, "draw": 0.25, "away": 0.15}
