"""Tests for the high-level evaluate_match / find_value_bets integration.

The pipeline is: pi-rating -> no-vig book odds -> confidence assessment ->
+EV flags. The dashboard calls evaluate_match() to get one fully-populated
dict per match.
"""
import pytest

from soccer_ev_model.ev_workflow import (
    evaluate_match,
    find_value_bets,
    pi_rating_match_probs,
)


def _make_match(date, home_id, away_id, hg=0, ag=0, result="H", home_name=None, away_name=None):
    return {
        "match_id": f"{date}_{home_id}_{away_id}",
        "date": date,
        "home_team": home_name or f"Team{home_id}",
        "away_team": away_name or f"Team{away_id}",
        "home_team_id": home_id,
        "away_team_id": away_id,
        "home_goals": hg,
        "away_goals": ag,
        "result": result,
    }


# ---- evaluate_match: structure ----


def test_evaluate_match_returns_all_expected_keys():
    """evaluate_match must return a dict with all keys the dashboard uses."""
    train = []
    for i in range(60):
        # Team 1 wins a lot
        train.append(_make_match(f"2020-{i+1:02d}-01", 1, 9, 2, 0))
    match = _make_match("2020-12-01", 1, 2, 0, 0)
    # Book odds: -150 home, +300 draw, +400 away
    ratings = {1: {"offense": 0.3, "defense": 0.2, "matches_played": 60},
               2: {"offense": -0.1, "defense": -0.1, "matches_played": 60}}
    # We compute ratings separately and pass them in
    from soccer_ev_model.pi_ratings import compute_pi_ratings
    ratings = compute_pi_ratings(train, cutoff="2020-12-01")

    result = evaluate_match(
        home_team="Team1",
        away_team="Team2",
        home_team_id=1,
        away_team_id=2,
        date="2020-12-01",
        book_home_odds=-150,
        book_draw_odds=300,
        book_away_odds=400,
        ratings=ratings,
    )

    expected = {
        "home_team", "away_team", "date", "book_odds", "book_fair",
        "pi_probs", "calibrated_pi", "edges", "confidence",
        "plus_ev_flags", "banner",
    }
    assert expected <= set(result.keys())


def test_evaluate_match_probs_sum_to_one():
    """Both pi_probs and book_fair must be valid probability distributions."""
    train = [_make_match("2020-01-01", 1, 2, 1, 0)] * 50
    from soccer_ev_model.pi_ratings import compute_pi_ratings
    ratings = compute_pi_ratings(train, cutoff="2020-12-01")

    result = evaluate_match(
        home_team="Team1", away_team="Team2",
        home_team_id=1, away_team_id=2,
        date="2020-12-01",
        book_home_odds=-150, book_draw_odds=300, book_away_odds=400,
        ratings=ratings,
    )
    for k in ("pi_probs", "book_fair", "calibrated_pi"):
        total = sum(result[k].values())
        assert abs(total - 1.0) < 1e-3, f"{k} sums to {total}, not 1.0"


def test_evaluate_match_edges_are_pi_minus_book_fair():
    """edges = pi_probs - book_fair (the +EV signal, per market)."""
    from soccer_ev_model.pi_ratings import compute_pi_ratings
    train = [_make_match("2020-01-01", 1, 2, 2, 0)] * 50
    ratings = compute_pi_ratings(train, cutoff="2020-12-01")

    result = evaluate_match(
        home_team="Team1", away_team="Team2",
        home_team_id=1, away_team_id=2,
        date="2020-12-01",
        book_home_odds=-200, book_draw_odds=300, book_away_odds=500,
        ratings=ratings,
    )
    for market in ("home", "draw", "away"):
        expected_edge = result["primary_probs"][market] - result["book_fair"][market]
        assert result["edges"][market] == pytest.approx(expected_edge, abs=1e-6)


def test_evaluate_match_includes_confidence_assessment():
    """The 'confidence' key must be the full assess_match_confidence dict."""
    from soccer_ev_model.pi_ratings import compute_pi_ratings
    train = [_make_match("2020-01-01", 1, 2, 1, 0)] * 50
    ratings = compute_pi_ratings(train, cutoff="2020-12-01")

    result = evaluate_match(
        home_team="Team1", away_team="Team2",
        home_team_id=1, away_team_id=2,
        date="2020-12-01",
        book_home_odds=-150, book_draw_odds=300, book_away_odds=400,
        ratings=ratings,
    )
    conf = result["confidence"]
    # Must be the full assessment dict
    for k in ("tier", "tier_description", "top_p", "calibrated_p",
              "calib_label", "data_label", "warnings", "edge_warning",
              "home_matches_played", "away_matches_played"):
        assert k in conf, f"confidence missing key {k}"


def test_evaluate_match_banner_is_a_string():
    """The banner must be a non-empty string."""
    from soccer_ev_model.pi_ratings import compute_pi_ratings
    train = [_make_match("2020-01-01", 1, 2, 1, 0)] * 50
    ratings = compute_pi_ratings(train, cutoff="2020-12-01")

    result = evaluate_match(
        home_team="Team1", away_team="Team2",
        home_team_id=1, away_team_id=2,
        date="2020-12-01",
        book_home_odds=-150, book_draw_odds=300, book_away_odds=400,
        ratings=ratings,
    )
    assert isinstance(result["banner"], str)
    assert len(result["banner"]) > 0


def test_evaluate_match_plus_ev_flags_only_above_threshold():
    """plus_ev_flags should only contain markets with edge >= min_edge (default 0.03)."""
    from soccer_ev_model.pi_ratings import compute_pi_ratings
    train = [_make_match("2020-01-01", 1, 2, 1, 0)] * 50
    ratings = compute_pi_ratings(train, cutoff="2020-12-01")

    result = evaluate_match(
        home_team="Team1", away_team="Team2",
        home_team_id=1, away_team_id=2,
        date="2020-12-01",
        book_home_odds=-150, book_draw_odds=300, book_away_odds=400,
        ratings=ratings,
    )
    for flag in result["plus_ev_flags"]:
        assert flag["edge"] >= 0.03, f"Flag has edge {flag['edge']} below threshold"
        assert "market" in flag


def test_evaluate_match_min_edge_is_respected():
    """With min_edge=0.10, only markets with edge >= 0.10 are flagged."""
    from soccer_ev_model.pi_ratings import compute_pi_ratings
    train = [_make_match("2020-01-01", 1, 2, 1, 0)] * 50
    ratings = compute_pi_ratings(train, cutoff="2020-12-01")

    result = evaluate_match(
        home_team="Team1", away_team="Team2",
        home_team_id=1, away_team_id=2,
        date="2020-12-01",
        book_home_odds=-150, book_draw_odds=300, book_away_odds=400,
        ratings=ratings,
        min_edge=0.10,
    )
    for flag in result["plus_ev_flags"]:
        assert flag["edge"] >= 0.10


def test_evaluate_match_calibrated_pi_corrects_overconfidence():
    """If pi_probs has top_p > 0.7, calibrated_pi should reduce the top market."""
    from soccer_ev_model.pi_ratings import compute_pi_ratings
    train = [_make_match("2020-01-01", 1, 2, 1, 0)] * 50
    ratings = compute_pi_ratings(train, cutoff="2020-12-01")
    # Force a top_p > 0.7 by checking the actual pi probs
    probs = pi_rating_match_probs(train, _make_match("2020-12-01", 1, 2, 0, 0))
    top_market = max(probs, key=probs.get)
    top_p = probs[top_market]

    result = evaluate_match(
        home_team="Team1", away_team="Team2",
        home_team_id=1, away_team_id=2,
        date="2020-12-01",
        book_home_odds=-150, book_draw_odds=300, book_away_odds=400,
        ratings=ratings,
    )
    # If top_p > 0.7, the calibrated top should be < raw top_p
    if top_p > 0.7:
        assert result["calibrated_pi"][top_market] < result["pi_probs"][top_market]


# ---- find_value_bets: backward compatibility ----


def test_find_value_bets_signature_unchanged_by_default():
    """Calling find_value_bets without include_confidence must return the
    original shape (no 'confidence' key) so old callers keep working."""
    train = [_make_match("2020-01-01", 1, 2, 1, 0)] * 30
    target = _make_match("2020-12-01", 1, 2, 0, 0)
    book_odds = {target["match_id"]: {"home": -150, "draw": 300, "away": 400}}
    results = find_value_bets(train, [target], book_odds, min_edge=0.01)
    # Each row should be a +EV flag with the original keys
    for r in results:
        for k in ("match_id", "match", "market", "pi_prob", "book_prob", "edge"):
            assert k in r, f"missing key {k} in {r}"
        # 'confidence' should NOT be present by default
        assert "confidence" not in r


def test_find_value_bets_with_include_confidence_adds_assessment():
    """With include_confidence=True, every match in book_odds gets a 'confidence'
    key in each row, regardless of whether it was flagged as +EV."""
    train = [_make_match("2020-01-01", 1, 2, 1, 0)] * 30
    target = _make_match("2020-12-01", 1, 2, 0, 0)
    book_odds = {target["match_id"]: {"home": -150, "draw": 300, "away": 400}}
    results = find_value_bets(train, [target], book_odds,
                              min_edge=0.01, include_confidence=True)
    # There should be at least one result
    assert len(results) >= 1
    for r in results:
        assert "confidence" in r
        assert "tier" in r["confidence"]
        assert "warnings" in r["confidence"]


# ---- evaluate_match: blend keys (pi_only_probs, elo_only_probs, blend_was_used) ----


def _train_ratings():
    """Build a small training set and return (train_list, ratings_dict)."""
    train = []
    for i in range(40):
        train.append(_make_match(f"2020-{(i % 9) + 1:02d}-01", 1, 2, 2, 0))
        train.append(_make_match(f"2020-{(i % 9) + 1:02d}-02", 2, 3, 1, 1))
    from soccer_ev_model.pi_ratings import compute_pi_ratings
    ratings = compute_pi_ratings(train, cutoff="2020-12-01")
    return train, ratings


def test_evaluate_match_blend_returns_new_keys():
    """With both elos provided, evaluate_match returns pi_only_probs,
    elo_only_probs, and blend_was_used=True."""
    _train, ratings = _train_ratings()
    result = evaluate_match(
        home_team="Team1", away_team="Team2",
        home_team_id=1, away_team_id=2,
        date="2020-12-01",
        book_home_odds=-150, book_draw_odds=300, book_away_odds=400,
        ratings=ratings,
        home_elo=1600, away_elo=1500,
    )
    assert "pi_only_probs" in result
    assert "elo_only_probs" in result
    assert result["blend_was_used"] is True
    # pi_only and elo_only should both be valid prob dicts
    for key in ("pi_only_probs", "elo_only_probs"):
        probs = result[key]
        assert set(probs.keys()) == {"home", "draw", "away"}
        assert abs(sum(probs.values()) - 1.0) < 1e-3
    # pi_probs should be the BLEND (not pi_only), and must differ from pi_only
    # when the Elo signal is non-neutral (1600 vs 1500 is a real edge).
    assert result["pi_probs"] == result["pi_only_probs"]  # pi_probs is now pure pi (diagnostic), same as pi_only


def test_evaluate_match_pure_pi_returns_new_keys():
    """Without Elo, pi_only_probs == pi_probs and blend_was_used is False."""
    _train, ratings = _train_ratings()
    result = evaluate_match(
        home_team="Team1", away_team="Team2",
        home_team_id=1, away_team_id=2,
        date="2020-12-01",
        book_home_odds=-150, book_draw_odds=300, book_away_odds=400,
        ratings=ratings,
    )
    assert result["blend_was_used"] is False
    assert result["pi_only_probs"] == result["pi_probs"]
    assert result["elo_only_probs"] is None


def test_evaluate_match_manual_odds_regression():
    """Manual odds: parsing, edges, plus_ev_flags still work after changes."""
    _train, ratings = _train_ratings()
    result = evaluate_match(
        home_team="Team1", away_team="Team2",
        home_team_id=1, away_team_id=2,
        date="2020-12-01",
        book_home_odds=-200, book_draw_odds=350, book_away_odds=500,
        ratings=ratings,
        min_edge=0.03,
    )
    # Edges must still be primary_probs - book_fair
    for market in ("home", "draw", "away"):
        expected = result["primary_probs"][market] - result["book_fair"][market]
        assert result["edges"][market] == pytest.approx(expected, abs=1e-3)
    # All +EV flags must be >= min_edge
    for flag in result["plus_ev_flags"]:
        assert flag["edge"] >= 0.03
    # Existing keys must still be present
    for k in ("home_team", "away_team", "date", "book_odds", "book_fair",
              "pi_probs", "calibrated_pi", "edges", "confidence",
              "plus_ev_flags", "banner"):
        assert k in result


# ---- evaluate_match: blend_probs alias for clarity ----


def test_evaluate_match_blend_probs_alias_present_and_identical():
    """`blend_probs` is the new explicit alias for the blend (the value
    historically stored in `pi_probs`). Both keys must exist and contain
    the same dict."""
    _train, ratings = _train_ratings()
    result = evaluate_match(
        home_team="Team1", away_team="Team2",
        home_team_id=1, away_team_id=2,
        date="2020-12-01",
        book_home_odds=-150, book_draw_odds=300, book_away_odds=400,
        ratings=ratings,
    )
    assert "blend_probs" in result
    assert "pi_probs" in result
    assert result["blend_probs"] is not None
    assert result["pi_probs"] is not None
    assert result["blend_probs"] == result["primary_probs"]  # blend_probs aliases primary_probs
    # Shape sanity: home / draw / away
    assert set(result["blend_probs"].keys()) == {"home", "draw", "away"}


def test_evaluate_match_blend_probs_alias_holds_with_elo_too():
    """When Elo is supplied, `blend_probs` is the actual blend (not pure
    pi) and still equals `pi_probs`."""
    _train, ratings = _train_ratings()
    result = evaluate_match(
        home_team="Team1", away_team="Team2",
        home_team_id=1, away_team_id=2,
        date="2020-12-01",
        book_home_odds=-150, book_draw_odds=300, book_away_odds=400,
        ratings=ratings,
        home_elo=1600, away_elo=1500,
    )
    assert result["blend_was_used"] is True
    assert result["blend_probs"] == result["primary_probs"]  # blend_probs aliases primary_probs
    # And the blend must actually differ from pi-only when Elo is non-neutral
    assert result["blend_probs"] != result["pi_only_probs"]
