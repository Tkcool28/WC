"""Tests for the pi-rating + Elo blend in ev_workflow."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soccer_ev_model.ev_workflow import (  # noqa: E402
    _probs_from_ratings,
    _probs_from_ratings_blend,
)
from soccer_ev_model.pi_ratings import compute_pi_ratings  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _make_match(date, home_id, away_id, hg, ag, result="H"):
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


def _trained_ratings():
    """Build a tiny history where team 1 is clearly stronger than team 99."""
    train = []
    for i in range(8):
        train.append(_make_match(f"2020-0{i+1}-01", 1, 99, 3, 0, result="H"))
    return compute_pi_ratings(train, cutoff="2020-09-01")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_blend_with_w_elo_zero_matches_pi_only():
    """Regression guard: w_elo=0 (and w_pi=1) must yield identical probs
    to the pure pi-rating function for the same match and ratings."""
    ratings = _trained_ratings()
    match = _make_match("2020-12-01", 1, 99, 0, 0)

    pi_only = _probs_from_ratings(match, ratings)
    blend = _probs_from_ratings_blend(
        match, ratings,
        home_elo=1900, away_elo=1900,  # Elo says nothing
        w_pi=1.0, w_elo=0.0,
    )

    for k in ("home", "draw", "away"):
        assert pi_only[k] == pytest.approx(blend[k], abs=1e-12), (
            f"pi_only[{k}]={pi_only[k]} != blend[{k}]={blend[k]}"
        )


def test_blend_with_w_pi_zero_is_elo_only():
    """Smoke test: w_pi=0 (pure Elo) should still return a valid 3-way
    distribution, and a positive home-Elo edge should push p_home > base."""
    ratings = _trained_ratings()  # not used by Elo path
    match = _make_match("2020-12-01", 1, 99, 0, 0)

    # Symmetric Elo: combined_matchup = 0, expect base rates
    sym = _probs_from_ratings_blend(
        match, ratings,
        home_elo=1700, away_elo=1700,
        w_pi=0.0, w_elo=1.0,
    )
    assert set(sym.keys()) == {"home", "draw", "away"}
    assert abs(sum(sym.values()) - 1.0) < 1e-6
    # When both Elos are equal, home/away/away should match the base rates
    # (no asymmetry in the formula when matchup = 0).
    assert sym["home"] == pytest.approx(0.40, abs=1e-6)
    assert sym["away"] == pytest.approx(0.33, abs=1e-6)
    assert sym["draw"] == pytest.approx(0.27, abs=1e-6)

    # Strong home Elo: home_elo - away_elo = 400 => elo_diff_normalized = 1.0
    strong = _probs_from_ratings_blend(
        match, ratings,
        home_elo=1900, away_elo=1500,
        w_pi=0.0, w_elo=1.0,
    )
    assert strong["home"] > sym["home"], (
        "Elo edge should push p_home up vs the symmetric case"
    )
    assert strong["away"] < sym["away"], (
        "Elo edge should push p_away down vs the symmetric case"
    )


def test_blend_handles_default_elo():
    """Elo default (1500/1500) and equal Elos must not crash and must
    produce a sensible, normalised distribution.

    Note: even with equal Elos (so elo_diff_normalized = 0), a non-zero
    w_elo still attenuates the pi signal because the linear combination
    scales pi_matchup by w_pi. So the blend with w_pi < 1 and equal Elos
    should be CLOSER to the uniform base rates (40/27/33) than pi_only.
    """
    ratings = _trained_ratings()
    match = _make_match("2020-12-01", 1, 99, 0, 0)
    pi_only = _probs_from_ratings(match, ratings)

    # w_elo = 0 must reproduce pi_only exactly (sanity)
    probs_pi = _probs_from_ratings_blend(
        match, ratings,
        home_elo=1500, away_elo=1500,
        w_pi=1.0, w_elo=0.0,
    )
    assert abs(sum(probs_pi.values()) - 1.0) < 1e-6
    for k in ("home", "draw", "away"):
        assert probs_pi[k] == pytest.approx(pi_only[k], abs=1e-12)

    # Equal Elos + w_pi < 1 should pull the blend toward the symmetric
    # base rates (pi signal is attenuated by w_pi).
    for wpi in (0.5, 0.7, 0.3):
        welo = 1.0 - wpi
        p = _probs_from_ratings_blend(
            match, ratings,
            home_elo=1500, away_elo=1500,
            w_pi=wpi, w_elo=welo,
        )
        assert abs(sum(p.values()) - 1.0) < 1e-6
        # pi_only has p_home pulled UP by the team-1-favourable signal.
        # The blend with w_pi < 1 should be CLOSER to base 0.40 than pi_only.
        base_h = 0.40
        assert abs(p["home"] - base_h) < abs(pi_only["home"] - base_h), (
            f"w_pi={wpi}: blend home {p['home']} should be closer to "
            f"base {base_h} than pi_only home {pi_only['home']}"
        )
        # And in particular should sit between base and pi_only.
        lo, hi = sorted([base_h, pi_only["home"]])
        assert lo <= p["home"] <= hi, (
            f"w_pi={wpi}: p_home {p['home']} not in [{lo}, {hi}]"
        )


def test_blend_shifts_probs_when_elo_disagrees_with_pi():
    """Synthetic: pi-rating is neutral, Elo strongly favours home. The blend
    should pull probs toward home relative to pi-only."""
    # Build a history that makes team 1 ≈ team 99 (alternating wins).
    train = []
    for i in range(6):
        if i % 2 == 0:
            train.append(_make_match(f"2020-0{i+1}-01", 1, 99, 2, 1, result="H"))
        else:
            train.append(_make_match(f"2020-0{i+1}-01", 99, 1, 2, 1, result="A"))
    ratings = compute_pi_ratings(train, cutoff="2020-07-01")

    match = _make_match("2020-12-01", 1, 99, 0, 0)

    # Pi-only baseline (w_elo = 0): we expect roughly base-rate probs
    pi_only = _probs_from_ratings(match, ratings)

    # pi_heavy (0.7 / 0.3) with Elo strongly favouring home
    # elo_diff_normalized = (1900 - 1500) / 400 = 1.0
    blend = _probs_from_ratings_blend(
        match, ratings,
        home_elo=1900, away_elo=1500,
        w_pi=0.7, w_elo=0.3,
    )

    # The blend must shift p_home UP and p_away DOWN relative to pi-only
    assert blend["home"] > pi_only["home"], (
        f"Expected blend home {blend['home']} > pi_only home {pi_only['home']}"
    )
    assert blend["away"] < pi_only["away"], (
        f"Expected blend away {blend['away']} < pi_only away {pi_only['away']}"
    )
    # And the distribution must still be valid
    assert abs(sum(blend.values()) - 1.0) < 1e-6
