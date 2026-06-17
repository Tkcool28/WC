"""Tests for the Poisson goal-model helpers (Phase 2).

The helpers under test are pure (no I/O, no Streamlit, no new
dependencies).  We import them from
``soccer_ev_model.prediction_summary``.  The integration test
(``test_poisson_does_not_alter_blend``) calls the real ``evaluate_match``
workflow and pins the structure of the result so the helpers can be
exercised end-to-end without depending on exact model numerics.

Forbidden-words check: no new string literal in the production code or
in these tests contains any of the six words the project explicitly
forbids (gambling terminology).  Phrases used in the production code
are the spec-mandated: "Poisson goal model", "expected goals", "xG
estimate", "secondary view", "transparent".
"""
import math

import pytest

from soccer_ev_model.ev_workflow import evaluate_match
from soccer_ev_model.prediction_summary import (
    expected_goals_from_blend,
    poisson_agreement_label,
    poisson_outcome_probs,
    poisson_score_matrix,
)


# --------------------------------------------------------------------------- #
# poisson_score_matrix
# --------------------------------------------------------------------------- #

def test_poisson_score_matrix_shape():
    """Default max_goals=8 → 9x9 grid; all cells non-negative; sum in (0, 1]."""
    m = poisson_score_matrix(1.5, 0.8)
    assert len(m) == 9
    for row in m:
        assert len(row) == 9
        for cell in row:
            assert cell >= 0.0
    total = sum(sum(row) for row in m)
    assert 0.0 < total <= 1.0


def test_poisson_score_matrix_independent():
    """M[i][j] = P(home=i) * P(away=j) within 1e-9 (independent Poisson)."""
    for home_lam, away_lam in [(1.5, 0.8), (0.5, 2.5)]:
        m = poisson_score_matrix(home_lam, away_lam)
        # Reference: hand-compute the marginal PMFs via the same log-PMF
        # formula.  We replicate it locally (not via a call to a private
        # helper) to keep the test self-contained.
        def pmf(k, lam):
            return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))
        for i in range(9):
            for j in range(9):
                expected = pmf(i, home_lam) * pmf(j, away_lam)
                assert abs(m[i][j] - expected) < 1e-9, (
                    f"mismatch at (i={i}, j={j}, home_lam={home_lam}, "
                    f"away_lam={away_lam}): got {m[i][j]}, expected {expected}"
                )


# --------------------------------------------------------------------------- #
# poisson_outcome_probs
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("home_lam, away_lam", [(1.5, 0.8), (2.0, 0.3), (0.8, 0.8)])
def test_poisson_outcome_probs_sum_to_one(home_lam, away_lam):
    """Post-normalization, the three probs sum to 1.0 within 1e-9."""
    p = poisson_outcome_probs(home_lam, away_lam)
    total = p["home"] + p["draw"] + p["away"]
    assert abs(total - 1.0) < 1e-9


@pytest.mark.parametrize("home_lam, away_lam", [(1.5, 0.8), (2.0, 0.3), (0.8, 0.8)])
def test_poisson_outcome_probs_no_negative(home_lam, away_lam):
    """All three probs are non-negative."""
    p = poisson_outcome_probs(home_lam, away_lam)
    assert p["home"] >= 0.0
    assert p["draw"] >= 0.0
    assert p["away"] >= 0.0


def test_poisson_outcome_probs_known_distribution():
    """At (3.0, 0.5) (strong home favorite) home > draw > away."""
    p = poisson_outcome_probs(3.0, 0.5)
    assert p["home"] > p["draw"] > p["away"]


def test_poisson_outcome_probs_tossup_has_higher_draw():
    """Draw awareness scales with matchup parity.

    At near-equal rates draw_p is in [0.20, 0.35]; at a strong mismatch
    draw_p is below 0.18.
    """
    p_tossup = poisson_outcome_probs(1.27, 1.27)
    p_mismatch = poisson_outcome_probs(3.0, 0.5)
    assert 0.20 <= p_tossup["draw"] <= 0.35, (
        f"tossup draw_p={p_tossup['draw']:.4f} outside [0.20, 0.35]"
    )
    assert p_mismatch["draw"] < 0.18, (
        f"mismatch draw_p={p_mismatch['draw']:.4f} not below 0.18"
    )


def test_poisson_outcome_probs_rejects_negative():
    """Negative home_xg and NaN away_xg both raise ValueError."""
    with pytest.raises(ValueError):
        poisson_outcome_probs(-0.1, 1.0)
    with pytest.raises(ValueError):
        poisson_outcome_probs(1.0, math.nan)


# --------------------------------------------------------------------------- #
# expected_goals_from_blend
# --------------------------------------------------------------------------- #

def test_expected_goals_from_blend_strong_favorite():
    """Strong home favorite (0.70/0.18/0.12) → home_xg > away_xg, total ~2.55."""
    blend = {"home": 0.70, "draw": 0.18, "away": 0.12}
    g = expected_goals_from_blend(blend)
    assert g["home_xg"] > g["away_xg"]
    assert g["home_xg"] > 1.5
    assert g["away_xg"] < 1.0
    assert abs(g["home_xg"] + g["away_xg"] - 2.55) < 0.05


def test_expected_goals_from_blend_clamps_extreme_edge():
    """Extreme blend (1.0/0.0/0.0): goal_diff hits max_edge, away_xg hits min_xg.

    With the spec constants (base_total=2.55, edge_scale=2.2, max_edge=2.2)
    the formula's natural max for home_xg at the strongest possible
    strength_edge is (2.55 + 2.2) / 2 = 2.375 — the upper max_xg=4.0
    clamp is dead code with these constants.  The meaningful "clamp"
    property here is that the formula behaves predictably at extremes:
    goal_diff is bounded to ±max_edge, and the away-side hits the
    min_xg floor of 0.2.
    """
    blend = {"home": 1.0, "draw": 0.0, "away": 0.0}
    g = expected_goals_from_blend(blend)
    # Spec-formula actual output: home_xg=2.375, away_xg=0.2.
    assert g["home_xg"] == 2.375
    assert g["away_xg"] == 0.2
    # goal_diff is clamped to ±max_edge (2.2).
    assert abs(g["goal_diff"]) <= 2.2 + 1e-9
    # away_xg is clamped to the min_xg floor (0.2).
    assert g["away_xg"] == 0.2
    # home_xg sits at the formula's natural max given the spec constants.
    # The upper max_xg=4.0 clamp never fires here, so home_xg < 4.0.
    assert g["home_xg"] < 4.0


def test_expected_goals_from_blend_tight_matchup_is_balanced():
    """Near-equal blend (0.40/0.27/0.33) → home_xg ≈ 1.352, away_xg ≈ 1.198.

    The spec formula splits base_total (2.55) across the two sides
    according to the strength edge.  At blend {0.40, 0.27, 0.33} the
    strength_edge is +0.07, so home_xg gets the slight bump and
    away_xg gets the slight trim, but the total stays at the
    base_total of 2.55 (this is a property of the (b±d)/2 split —
    it conserves the base total exactly).  The "tossup" property
    here is that the two sides stay within ~0.20 of each other.
    """
    blend = {"home": 0.40, "draw": 0.27, "away": 0.33}
    g = expected_goals_from_blend(blend)
    # Spec-formula actual output: home_xg=1.352, away_xg=1.198.
    assert g["home_xg"] == 1.352
    assert g["away_xg"] == 1.198
    # The two sides are close (within 0.20) — the matchup is balanced.
    assert abs(g["home_xg"] - g["away_xg"]) < 0.20
    # (b+d)/2 split conserves base_total exactly: home_xg + away_xg = 2.55.
    assert abs(g["home_xg"] + g["away_xg"] - 2.55) < 1e-6
    # home has a slight edge under this blend.
    assert g["home_xg"] > g["away_xg"]


def test_expected_goals_from_blend_total_near_baseline():
    """For moderate favorites, total_xg stays in [2.4, 2.7]."""
    for blend in [
        {"home": 0.55, "draw": 0.25, "away": 0.20},
        {"home": 0.60, "draw": 0.22, "away": 0.18},
        {"home": 0.50, "draw": 0.28, "away": 0.22},
    ]:
        g = expected_goals_from_blend(blend)
        total = g["home_xg"] + g["away_xg"]
        assert 2.4 <= total <= 2.7, (
            f"blend {blend} gave total {total} outside [2.4, 2.7]"
        )


# --------------------------------------------------------------------------- #
# Integration: Poisson must not mutate the blend
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


def test_poisson_does_not_alter_blend():
    """Poisson helpers must not mutate ``result['blend_probs']``.

    We snapshot a 4-dp rounded list of the blend probs, run the Poisson
    pipeline, and re-snapshot.  They must be byte-identical.
    """
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

    # Pre-snapshot: rounded to 4 decimal places (a stable comparison).
    before = [round(result["blend_probs"][k], 4) for k in ("home", "draw", "away")]

    # Run the full Poisson pipeline (this should be a read-only view).
    model_probs = result["blend_probs"]
    xg = expected_goals_from_blend(model_probs)
    poisson_probs = poisson_outcome_probs(xg["home_xg"], xg["away_xg"])
    _ = poisson_agreement_label(model_probs, poisson_probs)

    # Post-snapshot: must be byte-identical.
    after = [round(result["blend_probs"][k], 4) for k in ("home", "draw", "away")]
    assert before == after, (
        f"Poisson mutated blend_probs: before={before}, after={after}"
    )

    # And pi_probs (the alias) must be unchanged too.
    assert result["blend_probs"] == result["pi_probs"]


# --------------------------------------------------------------------------- #
# poisson_agreement_label
# --------------------------------------------------------------------------- #

def test_poisson_agreement_label_match():
    """When both top picks match, label='agrees' and agrees=True."""
    blend = {"home": 0.60, "draw": 0.25, "away": 0.15}
    poisson = {"home": 0.55, "draw": 0.27, "away": 0.18}
    out = poisson_agreement_label(blend, poisson)
    assert out["blend_top"] == "home"
    assert out["poisson_top"] == "home"
    assert out["agrees"] is True
    assert out["label"] == "agrees"


def test_poisson_agreement_label_mismatch():
    """When top picks differ, label='disagrees' and agrees=False."""
    blend = {"home": 0.60, "draw": 0.25, "away": 0.15}
    poisson = {"home": 0.20, "draw": 0.30, "away": 0.50}
    out = poisson_agreement_label(blend, poisson)
    assert out["blend_top"] == "home"
    assert out["poisson_top"] == "away"
    assert out["agrees"] is False
    assert out["label"] == "disagrees"


# --------------------------------------------------------------------------- #
# Realistic case: Brazil vs Haiti (illustrative blend)
# --------------------------------------------------------------------------- #

def test_poisson_realistic_brazil_haiti():
    """Illustrative Brazil-vs-Haiti blend: home_xg ≈ 1.8965, away_xg ≈ 0.6535.

    Pins the *shape* of the response (home strong favorite, away weak,
    Poisson probs monotonically ordered, conservation holds) without
    depending on the model returning the same numbers as another
    revision.  Loose bounds bracket the spec formula's actual output.
    """
    blend = {"home": 0.682, "draw": 0.201, "away": 0.117}
    g = expected_goals_from_blend(blend)
    # Spec-formula actual output: home_xg=1.8965, away_xg=0.6535.
    # Loose bounds bracket the actual values.
    assert 1.85 <= g["home_xg"] <= 1.95, (
        f"home_xg={g['home_xg']} outside [1.85, 1.95]"
    )
    assert 0.60 <= g["away_xg"] <= 0.70, (
        f"away_xg={g['away_xg']} outside [0.60, 0.70]"
    )
    # goal_diff is positive and within [1.0, 1.5] under this blend.
    assert 1.0 <= g["goal_diff"] <= 1.5, (
        f"goal_diff={g['goal_diff']} outside [1.0, 1.5]"
    )
    # Poisson probs at these xG values: home strong, away weak, draw
    # in the middle, and the three sum to 1.0 within 1e-9.
    p = poisson_outcome_probs(g["home_xg"], g["away_xg"])
    assert p["home"] > p["draw"] > p["away"], (
        f"Poisson probs not monotonically ordered: {p}"
    )
    assert abs(p["home"] + p["draw"] + p["away"] - 1.0) < 1e-9
