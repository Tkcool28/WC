"""Tests for the Phase-1 decoupling: predict_match() vs evaluate_market().

Phase 1 of the mobile-app rearchitecture splits evaluate_match() into two
functions:
  - predict_match(): pure model-only prediction, NO odds required
  - evaluate_market(): market layer over a prediction dict, REQUIRES odds

evaluate_match() remains as a backward-compat wrapper that returns the
union of both. These tests pin that contract.
"""
import pytest

from soccer_ev_model.ev_workflow import (
    evaluate_market,
    evaluate_match,
    predict_match,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

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
    """Build a small training set + ratings snapshot. Returns the ratings dict."""
    train = []
    for i in range(40):
        train.append(_make_match(f"2020-{(i % 9) + 1:02d}-01", 1, 2, 2, 0))
        train.append(_make_match(f"2020-{(i % 9) + 1:02d}-02", 2, 3, 1, 1))
    from soccer_ev_model.pi_ratings import compute_pi_ratings
    ratings = compute_pi_ratings(train, cutoff="2020-12-01")
    return train, ratings


# --------------------------------------------------------------------------- #
# predict_match: model-only, no odds required
# --------------------------------------------------------------------------- #

def test_predict_match_no_odds_required():
    """predict_match must work end-to-end without any bookmaker odds.

    It must produce a fully-populated result containing the four core
    model-output keys: pi_probs, blend_probs, confidence, banner.
    """
    _train, ratings = _train_ratings()
    result = predict_match(
        home_team="Team1",
        away_team="Team2",
        home_team_id=1,
        away_team_id=2,
        date="2020-12-01",
        ratings=ratings,
    )
    # Core model-output keys must all be present
    assert "pi_probs" in result
    assert "blend_probs" in result
    assert "confidence" in result
    assert "banner" in result
    # pi_probs / blend_probs must be valid 3-way prob dicts
    for k in ("pi_probs", "blend_probs"):
        assert set(result[k].keys()) == {"home", "draw", "away"}
        assert abs(sum(result[k].values()) - 1.0) < 1e-3
    # Confidence is a non-empty dict; banner is a non-empty string
    assert isinstance(result["confidence"], dict)
    assert isinstance(result["banner"], str) and len(result["banner"]) > 0


def test_predict_match_probabilities_match_evaluate_match_model_portion():
    """The model portion of predict_match must equal that of evaluate_match.

    With dummy valid odds, the model-derived keys (pi_probs, blend_probs,
    pi_only_probs, elo_only_probs, blend_was_used, confidence, banner)
    must be identical between predict_match and evaluate_match.
    """
    _train, ratings = _train_ratings()
    pred = predict_match(
        home_team="Team1",
        away_team="Team2",
        home_team_id=1,
        away_team_id=2,
        date="2020-12-01",
        ratings=ratings,
    )
    wrapper = evaluate_match(
        home_team="Team1",
        away_team="Team2",
        home_team_id=1,
        away_team_id=2,
        date="2020-12-01",
        ratings=ratings,
        book_home_odds=-150,
        book_draw_odds=300,
        book_away_odds=400,
    )
    for k in ("pi_probs", "blend_probs", "pi_only_probs", "elo_only_probs",
              "blend_was_used", "confidence", "banner"):
        assert pred[k] == wrapper[k], f"mismatch on key {k!r}"


# --------------------------------------------------------------------------- #
# evaluate_market: requires valid odds, uses passed prediction
# --------------------------------------------------------------------------- #

def test_evaluate_market_requires_valid_odds():
    """evaluate_market must raise ValueError on invalid odds."""
    _train, ratings = _train_ratings()
    pred = predict_match(
        home_team="Team1", away_team="Team2",
        home_team_id=1, away_team_id=2,
        date="2020-12-01", ratings=ratings,
    )
    # Zero odds -> ValueError
    with pytest.raises(ValueError):
        evaluate_market(pred, book_home_odds=0, book_draw_odds=300, book_away_odds=400)
    # Out-of-range odds -> ValueError (per no_vig.remove_vig rules)
    with pytest.raises(ValueError):
        evaluate_market(pred, book_home_odds=-150, book_draw_odds=300,
                        book_away_odds=20000)


def test_evaluate_market_uses_passed_prediction():
    """evaluate_market must use the prediction dict's pi_probs directly.

    It must NOT re-derive model probabilities from ratings. We pass a
    fake prediction with a known extreme pi_probs and verify the
    resulting edges and plus_ev_flags reflect THOSE values, not
    anything derived from the ratings.
    """
    fake_pi_probs = {"home": 0.90, "draw": 0.05, "away": 0.05}
    fake_confidence = {"calibrated_p": 0.55, "warnings": [], "tier": "A"}
    prediction = {
        "pi_probs": fake_pi_probs,
        "confidence": fake_confidence,
        "home_team": "Team1",
        "away_team": "Team2",
    }
    # With these obviously-juiced pi_probs and book odds that say
    # the home team is only a 60% favourite, the home market should
    # be a +EV play.
    result = evaluate_market(
        prediction,
        book_home_odds=-150,    # implied fair ~ 0.6
        book_draw_odds=400,
        book_away_odds=500,
        min_edge=0.05,
    )
    # Edges must be exactly pi - book_fair (no rounding) and
    # therefore anchored to the fake pi_probs.
    book_fair = result["book_fair"]
    for m in ("home", "draw", "away"):
        expected_edge = round(fake_pi_probs[m] - book_fair[m], 4)
        assert result["edges"][m] == expected_edge
    # The home market has edge ~ 0.30, well above min_edge=0.05, so
    # plus_ev_flags must contain a 'home' entry.
    flagged = {f["market"] for f in result["plus_ev_flags"]}
    assert "home" in flagged


# --------------------------------------------------------------------------- #
# Decoupling boundary
# --------------------------------------------------------------------------- #

def test_predict_match_does_not_contain_market_keys():
    """predict_match must NOT return any market-layer keys."""
    _train, ratings = _train_ratings()
    result = predict_match(
        home_team="Team1", away_team="Team2",
        home_team_id=1, away_team_id=2,
        date="2020-12-01", ratings=ratings,
    )
    for forbidden in ("book_odds", "book_fair", "edges", "plus_ev_flags", "calibrated_pi"):
        assert forbidden not in result, (
            f"predict_match should not contain market key {forbidden!r}"
        )


def test_predict_match_with_no_elo():
    """With no Elo, predict_match returns the pure-pi path.

    `elo_only_probs` must be None and `blend_was_used` must be False,
    and `pi_only_probs` must equal `pi_probs` (no blend).
    """
    _train, ratings = _train_ratings()
    result = predict_match(
        home_team="Team1", away_team="Team2",
        home_team_id=1, away_team_id=2,
        date="2020-12-01", ratings=ratings,
        home_elo=None, away_elo=None,
    )
    assert result["elo_only_probs"] is None
    assert result["blend_was_used"] is False
    assert result["pi_only_probs"] == result["pi_probs"]


# --------------------------------------------------------------------------- #
# evaluate_match: backward-compat wrapper preserves legacy shape
# --------------------------------------------------------------------------- #

def test_evaluate_match_wrapper_returns_legacy_shape():
    """The wrapper must return all 14 legacy keys + canonical ids + identity flag.

    Legacy 14-key shape:
      pi_probs, pi_only_probs, elo_only_probs, blend_probs, blend_was_used,
      blend_w_pi, blend_w_elo, book_odds, book_fair, calibrated_pi,
      edges, plus_ev_flags, plus_ev_count, confidence, banner
    Plus home/away/date/canonical ids.
    """
    _train, ratings = _train_ratings()
    result = evaluate_match(
        home_team="Team1", away_team="Team2",
        home_team_id=1, away_team_id=2,
        date="2020-12-01",
        book_home_odds=-150, book_draw_odds=300, book_away_odds=400,
        ratings=ratings,
        canonical_home_id="T1", canonical_away_id="T2",
    )
    expected = {
        # Identity / provenance
        "home_team", "away_team", "date",
        "canonical_home_id", "canonical_away_id",
        # Model layer
        "pi_probs", "pi_only_probs", "elo_only_probs", "blend_probs",
        "blend_was_used", "blend_w_pi", "blend_w_elo",
        "confidence", "banner",
        # Market layer
        "book_odds", "book_fair", "calibrated_pi",
        "edges", "plus_ev_flags", "plus_ev_count",
    }
    missing = expected - set(result.keys())
    assert not missing, f"evaluate_match wrapper missing legacy keys: {missing}"


def test_evaluate_match_missing_odds_returns_prediction_only():
    """When any odds is None, the wrapper must return the prediction only.

    No market keys (book_odds, edges, plus_ev_flags, calibrated_pi)
    may appear, and no exception should be raised.
    """
    _train, ratings = _train_ratings()
    result = evaluate_match(
        home_team="Team1", away_team="Team2",
        home_team_id=1, away_team_id=2,
        date="2020-12-01",
        book_home_odds=None,        # <-- no bookmaker prices
        book_draw_odds=300,
        book_away_odds=400,
        ratings=ratings,
    )
    # Prediction keys must be present
    for k in ("pi_probs", "blend_probs", "pi_only_probs", "elo_only_probs",
              "blend_was_used", "confidence", "banner"):
        assert k in result, f"missing prediction key {k!r}"
    # Market keys must be absent
    for k in ("book_odds", "book_fair", "edges", "plus_ev_flags", "calibrated_pi"):
        assert k not in result, f"unexpected market key {k!r} when odds missing"
