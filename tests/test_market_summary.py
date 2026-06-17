"""Tests for the market baseline helpers (Phase 1: model vs no-vig market).

The helpers under test are pure (no I/O, no Streamlit, no new dependencies).
We import them from `soccer_ev_model.prediction_summary`.  The test for
``resolve_model_probs_for_market`` constructs a mock ``evaluate_match``
result dict; the integration test (``test_evaluate_match_unchanged_structure``)
runs the real workflow and pins the *structure* of the result (not exact
numerics) so the helpers can be exercised end-to-end.
"""
import pytest

from soccer_ev_model.ev_workflow import evaluate_match
from soccer_ev_model.prediction_summary import (
    calculate_market_deltas,
    largest_market_delta,
    market_divergence_label,
    resolve_model_probs_for_market,
)


# --------------------------------------------------------------------------- #
# calculate_market_deltas
# --------------------------------------------------------------------------- #

def test_calculate_market_deltas_basic():
    """home 0.50 vs market 0.45 -> home delta = +5.0 pts; all three
    markets covered; values rounded to 1 decimal."""
    model = {"home": 0.50, "draw": 0.25, "away": 0.25}
    market = {"home": 0.45, "draw": 0.30, "away": 0.25}
    deltas = calculate_market_deltas(model, market)
    assert deltas == {"home": 5.0, "draw": -5.0, "away": 0.0}
    # All values rounded to 1 decimal place (no trailing integers from rounding)
    for v in deltas.values():
        # round(x, 1) -> at most 1 decimal place
        assert round(v, 1) == v


def test_calculate_market_deltas_negative_delta():
    """Model LESS than market -> negative pts (model is less confident)."""
    model = {"home": 0.40, "draw": 0.30, "away": 0.30}
    market = {"home": 0.55, "draw": 0.25, "away": 0.20}
    deltas = calculate_market_deltas(model, market)
    assert deltas["home"] < 0
    assert deltas["draw"] > 0
    assert deltas["away"] > 0
    # Exact values: (0.40 - 0.55)*100 = -15.0
    assert deltas["home"] == -15.0


def test_calculate_market_deltas_keys_must_match():
    """Missing/extra keys raise ValueError."""
    bad_model = {"home": 0.5, "draw": 0.5}  # missing 'away'
    good_market = {"home": 0.4, "draw": 0.3, "away": 0.3}
    with pytest.raises(ValueError):
        calculate_market_deltas(bad_model, good_market)

    bad_market = {"home": 0.4, "draw": 0.3, "away": 0.3, "extra": 0.0}
    with pytest.raises(ValueError):
        calculate_market_deltas(
            {"home": 0.5, "draw": 0.25, "away": 0.25}, bad_market
        )


# --------------------------------------------------------------------------- #
# market_divergence_label
# --------------------------------------------------------------------------- #

def test_market_divergence_label_all_four_boundaries():
    """Six boundary cases covering the four labels."""
    # max=0.029 -> "Strong market agreement"
    assert market_divergence_label(
        {"home": 0.029, "draw": 0.0, "away": 0.0}
    ) == "Strong market agreement"

    # max=0.03 -> "Moderate market agreement" (boundary, lower end)
    assert market_divergence_label(
        {"home": 0.03, "draw": 0.0, "away": 0.0}
    ) == "Moderate market agreement"

    # max=0.069 -> "Moderate market agreement" (just under 0.07)
    assert market_divergence_label(
        {"home": 0.069, "draw": 0.0, "away": 0.0}
    ) == "Moderate market agreement"

    # max=0.07 -> "Model divergence" (boundary)
    assert market_divergence_label(
        {"home": 0.07, "draw": 0.0, "away": 0.0}
    ) == "Model divergence"

    # max=0.119 -> "Model divergence"
    assert market_divergence_label(
        {"home": 0.119, "draw": 0.0, "away": 0.0}
    ) == "Model divergence"

    # max=0.12 -> "Major model divergence" (boundary)
    assert market_divergence_label(
        {"home": 0.12, "draw": 0.0, "away": 0.0}
    ) == "Major model divergence"


def test_market_divergence_label_uses_max_absolute_delta():
    """Label is determined by the LARGEST |delta|, not the average or sum."""
    deltas = {"home": 0.01, "draw": 0.025, "away": -0.028}
    # max |delta| = 0.028 < 0.03 -> Strong agreement
    assert market_divergence_label(deltas) == "Strong market agreement"

    deltas = {"home": 0.01, "draw": -0.029, "away": 0.0}
    # max |delta| = 0.029 < 0.03 -> Strong agreement
    assert market_divergence_label(deltas) == "Strong market agreement"


# --------------------------------------------------------------------------- #
# largest_market_delta
# --------------------------------------------------------------------------- #

def test_largest_market_delta_picks_max_abs():
    """{home:-1pt, draw:-8pts, away:+5pts} -> largest is 'draw'."""
    deltas = {"home": -1.0, "draw": -8.0, "away": 5.0}
    result = largest_market_delta(deltas)
    assert result["market"] == "draw"
    # delta_pts is the signed delta, rounded to 1 dp
    assert result["delta_pts"] == -8.0


def test_largest_market_delta_with_labels():
    """With market_labels provided, the result includes a 'label' key
    carrying the team/outcome name."""
    deltas = {"home": 1.0, "draw": -8.0, "away": 5.0}
    labels = {"home": "Brazil", "draw": "Draw", "away": "Haiti"}
    result = largest_market_delta(deltas, market_labels=labels)
    assert result["market"] == "draw"
    assert result["label"] == "Draw"
    # delta_pts rounded to 1 dp
    assert result["delta_pts"] == -8.0


def test_largest_market_delta_tiebreak_home_beats_away():
    """Tiebreak: home > draw > away.  {home:+5pts, away:-5pts} -> 'home' wins."""
    deltas = {"home": 5.0, "draw": 0.0, "away": -5.0}
    result = largest_market_delta(deltas)
    assert result["market"] == "home"
    assert result["delta_pts"] == 5.0


def test_largest_market_delta_draw_wins_over_away_on_tie():
    """Tiebreak: home > draw > away.  When draw and away tie on |delta|,
    'draw' wins because it comes before 'away' in the order."""
    deltas = {"home": 0.0, "draw": -5.0, "away": 5.0}
    result = largest_market_delta(deltas)
    assert result["market"] == "draw"


def test_largest_market_delta_includes_pct_when_probs_supplied():
    """When the caller passes model_probs and market_probs, the returned
    dict includes model_pct and market_pct (rounded to 1 dp)."""
    deltas = {"home": 5.0, "draw": -1.0, "away": -4.0}
    model = {"home": 0.55, "draw": 0.24, "away": 0.21}
    market = {"home": 0.50, "draw": 0.25, "away": 0.25}
    result = largest_market_delta(
        deltas, model_probs=model, market_probs=market,
    )
    assert result["market"] == "home"
    assert result["model_pct"] == 55.0
    assert result["market_pct"] == 50.0
    assert result["delta_pts"] == 5.0


# --------------------------------------------------------------------------- #
# resolve_model_probs_for_market
# --------------------------------------------------------------------------- #

def test_resolve_model_probs_prefers_blend():
    """When both 'blend_probs' and 'pi_probs' are present, blend_probs wins."""
    result = {
        "blend_probs": {"home": 0.6, "draw": 0.25, "away": 0.15},
        "pi_probs": {"home": 0.5, "draw": 0.3, "away": 0.2},
    }
    chosen = resolve_model_probs_for_market(result)
    assert chosen == {"home": 0.6, "draw": 0.25, "away": 0.15}


def test_resolve_model_probs_falls_back_to_pi():
    """When only 'pi_probs' is present, return it."""
    result = {
        "pi_probs": {"home": 0.5, "draw": 0.3, "away": 0.2},
    }
    chosen = resolve_model_probs_for_market(result)
    assert chosen == {"home": 0.5, "draw": 0.3, "away": 0.2}


def test_resolve_model_probs_raises_when_both_missing():
    """If neither key is present, raise KeyError."""
    with pytest.raises(KeyError):
        resolve_model_probs_for_market({})

    with pytest.raises(KeyError):
        resolve_model_probs_for_market({"book_fair": {"home": 0.5, "draw": 0.3, "away": 0.2}})


def test_resolve_model_probs_raises_when_both_set_to_none():
    """Explicit None values count as 'missing' and fall through to the
    raise path."""
    with pytest.raises(KeyError):
        resolve_model_probs_for_market(
            {"blend_probs": None, "pi_probs": None}
        )


# --------------------------------------------------------------------------- #
# End-to-end: evaluate_match + new helpers
# --------------------------------------------------------------------------- #

def _make_train():
    """Build a tiny training set so pi-rating has something to fit."""
    train = []
    for i in range(40):
        train.append({
            "match_id": f"2020-{i:02d}-1",
            "date": f"2020-{(i % 9) + 1:02d}-01",
            "home_team": "Team1",
            "away_team": "Team2",
            "home_team_id": 1,
            "away_team_id": 2,
            "home_goals": 2,
            "away_goals": 0,
            "result": "H",
        })
    return train


def test_evaluate_match_unchanged_structure():
    """evaluate_match returns blend_probs + book_fair; the new helpers
    produce a non-empty label and the deltas sum to ~0 for a realistic
    call.  Don't pin exact numerics — pin the *structure*."""
    from soccer_ev_model.pi_ratings import compute_pi_ratings

    train = _make_train()
    ratings = compute_pi_ratings(train, cutoff="2020-12-01")

    result = evaluate_match(
        home_team="Team1", away_team="Team2",
        home_team_id=1, away_team_id=2,
        date="2020-12-01",
        book_home_odds=-150, book_draw_odds=300, book_away_odds=400,
        ratings=ratings,
    )

    # Both keys the new helpers depend on are present.
    assert "blend_probs" in result
    assert "book_fair" in result
    assert set(result["blend_probs"].keys()) == {"home", "draw", "away"}
    assert set(result["book_fair"].keys()) == {"home", "draw", "away"}

    # Helpers produce a valid label (one of the 4 strings).
    # market_divergence_label takes raw probability deltas; calculate_market_deltas
    # returns pts (×100).  We pass raw deltas to the label so its thresholds
    # (0.03, 0.07, 0.12) apply at face value.
    model = resolve_model_probs_for_market(result)
    market = result["book_fair"]
    raw_deltas = {m: model[m] - market[m] for m in ("home", "draw", "away")}
    pts_deltas = calculate_market_deltas(model, market)
    label = market_divergence_label(raw_deltas)
    assert label in {
        "Strong market agreement",
        "Moderate market agreement",
        "Model divergence",
        "Major model divergence",
    }, f"unexpected label: {label}"

    # Deltas (model - market) * 100 must sum to ~0 (modulo rounding).
    assert abs(sum(pts_deltas.values())) < 0.5
    # And so must the raw deltas.
    assert abs(sum(raw_deltas.values())) < 0.005

    # largest_market_delta works on the realistic deltas (in pts).
    largest = largest_market_delta(
        pts_deltas,
        market_labels={"home": "Team1", "draw": "Draw", "away": "Team2"},
        model_probs=model, market_probs=market,
    )
    assert largest["market"] in {"home", "draw", "away"}
    assert "label" in largest


def test_manual_odds_do_not_change_blend_probs():
    """Two different manual odds triples must produce IDENTICAL blend_probs
    and pi_probs — manual odds only feed book_fair and edges."""
    from soccer_ev_model.pi_ratings import compute_pi_ratings

    train = _make_train()
    ratings = compute_pi_ratings(train, cutoff="2020-12-01")

    # First odds triple
    r1 = evaluate_match(
        home_team="Team1", away_team="Team2",
        home_team_id=1, away_team_id=2,
        date="2020-12-01",
        book_home_odds=-150, book_draw_odds=300, book_away_odds=400,
        ratings=ratings,
    )
    # Second odds triple (different prices) — model output must be identical
    r2 = evaluate_match(
        home_team="Team1", away_team="Team2",
        home_team_id=1, away_team_id=2,
        date="2020-12-01",
        book_home_odds=-200, book_draw_odds=350, book_away_odds=500,
        ratings=ratings,
    )

    # Model probabilities are unchanged by manual odds
    assert r1["blend_probs"] == r2["blend_probs"]
    assert r1["pi_probs"] == r2["pi_probs"]
    # Book fair probs reflect the different prices
    assert r1["book_fair"] != r2["book_fair"]
    # Edges reflect both the model (constant) and book_fair (changed)
    assert r1["edges"] != r2["edges"]
