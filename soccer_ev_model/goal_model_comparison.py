"""Comparison and blending framework for the independent goal model.

Compares:
  - Pi-only model (pure pi-rating logistic mapping)
  - Elo-only model (Elo-driven Poisson)
  - Current Pi/Elo blend (w_pi=1.0, w_elo=0.0 by default in production)
  - Independent goal model (regularized team Poisson)

All models evaluated on identical common sample.  No model sees data the others don't.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np

from soccer_ev_model.goal_model import (
    RegularizedTeamPoissonModel,
    GlobalPoissonModel,
    EloPoissonModel,
    summarize_prediction,
    _EPS,
)
from soccer_ev_model.goal_model_data import GoalMatch
from soccer_ev_model.goal_model_backtest import (
    BacktestMetrics,
    compute_metrics,
    HoldoutPeriod,
)
from soccer_ev_model.elo_ratings import elo_at, load_elo_ratings, DEFAULT_ELO
from soccer_ev_model.pi_ratings import compute_pi_ratings, _parse_date
from soccer_ev_model.ev_workflow import (
    _logistic_matchup,
    _probs_from_ratings,
    _probs_from_ratings_blend,
    _BASE_H,
    _BASE_D,
    _BASE_A,
    _LOGIT_SCALE,
    _DRAW_SCALE,
)


# ── Pi-only model adapter ───────────────────────────────────────────────────

def pi_only_probs(
    match: GoalMatch,
    pi_ratings: dict,
) -> dict[str, float]:
    """Compute H/D/A probabilities from pi-ratings for a single match."""
    h = pi_ratings.get(match.home_team_id)
    a = pi_ratings.get(match.away_team_id)
    if h is None or a is None:
        return {"home": _BASE_H, "draw": _BASE_D, "away": _BASE_A}
    pi_matchup = (h["offense"] - a["offense"]) + (h["defense"] - a["defense"])
    return _logistic_matchup(pi_matchup)


# ── Elo-only model adapter ──────────────────────────────────────────────────

def elo_only_probs(
    home_elo: float,
    away_elo: float,
) -> dict[str, float]:
    """Compute H/D/A probabilities from Elo difference only.

    Uses the same logistic mapping as the pi/Elo blend, but with
    Elo as the sole signal (no pi component).
    """
    elo_diff = (home_elo - away_elo) / 400.0
    return _logistic_matchup(elo_diff)


# ── Blend adapter ───────────────────────────────────────────────────────────

def blend_probs(
    pi_probs: dict[str, float],
    goal_probs: dict[str, float],
    w_pi: float,
    w_goal: float,
) -> dict[str, float]:
    """Blend pi-rating and goal model H/D/A probabilities.

    Args:
        pi_probs: {home, draw, away} from pi-ratings
        goal_probs: {home, draw, away} from goal model
        w_pi: weight for pi-rating probs
        w_goal: weight for goal model probs

    Returns:
        Blended {home, draw, away} that sums to 1.0.
    """
    total_w = w_pi + w_goal
    if total_w <= 0:
        return {"home": 1 / 3, "draw": 1 / 3, "away": 1 / 3}

    nw_pi = w_pi / total_w
    nw_goal = w_goal / total_w

    blended = {
        "home": nw_pi * pi_probs["home"] + nw_goal * goal_probs["home"],
        "draw": nw_pi * pi_probs["draw"] + nw_goal * goal_probs["draw"],
        "away": nw_pi * pi_probs["away"] + nw_goal * goal_probs["away"],
    }
    # Normalize
    s = sum(blended.values())
    if s > 0:
        blended = {k: v / s for k, v in blended.items()}
    return blended


# ── Confirmation / disagreement signal ──────────────────────────────────────

@dataclass
class ConfirmationSignal:
    """Result of comparing goal model to a 1X2 reference (e.g. pi/Elo blend)."""
    same_top: bool
    top_agreement: str          # "home", "draw", "away"
    prob_delta_home: float
    prob_delta_draw: float
    prob_delta_away: float
    expected_goal_support: str  # "home", "draw", "away"
    scoreline_agreement: bool   # same most-likely score outcome
    disagreement_level: str     # "none", "mild", "strong"
    warning: str                # human-readable warning if strong disagreement


def confirmation_signal(
    ref_probs: dict[str, float],
    goal_pred: dict,
) -> ConfirmationSignal:
    """Compare goal model prediction to a reference 1X2 model.

    Args:
        ref_probs: {home, draw, away} from reference model
        goal_pred: full prediction dict from goal model (with hda_probs, most_likely_score)

    Returns:
        ConfirmationSignal with agreement metrics.
    """
    goal_hda = goal_pred["hda_probs"]
    ref_top = max(ref_probs, key=ref_probs.get)
    goal_top = max(goal_hda, key=goal_hda.get)
    same_top = ref_top == goal_top

    # Expected goal support
    hx = goal_pred["home_xg"]
    ax = goal_pred["away_xg"]
    if hx > ax + 0.3:
        eg_support = "home"
    elif ax > hx + 0.3:
        eg_support = "away"
    else:
        eg_support = "draw"

    # Scoreline agreement
    mls = goal_pred.get("most_likely_score", [0, 0])
    if mls[0] > mls[1]:
        scoreline_outcome = "home"
    elif mls[0] < mls[1]:
        scoreline_outcome = "away"
    else:
        scoreline_outcome = "draw"
    scoreline_agree = scoreline_outcome == ref_top

    # Disagreement level
    max_delta = max(
        abs(goal_hda["home"] - ref_probs["home"]),
        abs(goal_hda["draw"] - ref_probs["draw"]),
        abs(goal_hda["away"] - ref_probs["away"]),
    )
    if not same_top:
        disagreement_level = "strong"
    elif max_delta > 0.10:
        disagreement_level = "mild"
    else:
        disagreement_level = "none"

    warning = ""
    if disagreement_level == "strong":
        warning = (
            f"Strong disagreement: reference picks {ref_top}, "
            f"goal model picks {goal_top} "
            f"(ref: {ref_probs[ref_top]:.2f}, goal: {goal_hda[goal_top]:.2f})"
        )

    return ConfirmationSignal(
        same_top=same_top,
        top_agreement=goal_top,
        prob_delta_home=round(goal_hda["home"] - ref_probs["home"], 4),
        prob_delta_draw=round(goal_hda["draw"] - ref_probs["draw"], 4),
        prob_delta_away=round(goal_hda["away"] - ref_probs["away"], 4),
        expected_goal_support=eg_support,
        scoreline_agreement=scoreline_agree,
        disagreement_level=disagreement_level,
        warning=warning,
    )


# ── Unified comparison row ─────────────────────────────────────────────────

@dataclass
class ComparisonRow:
    """Single-match prediction from all models."""
    match_date: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    neutral: bool
    tournament: str

    # Pi-only
    pi_home_p: float = 0.0
    pi_draw_p: float = 0.0
    pi_away_p: float = 0.0

    # Elo-only
    elo_home_p: float = 0.0
    elo_draw_p: float = 0.0
    elo_away_p: float = 0.0

    # Current blend (pi + elo)
    blend_home_p: float = 0.0
    blend_draw_p: float = 0.0
    blend_away_p: float = 0.0

    # Goal model
    goal_home_p: float = 0.0
    goal_draw_p: float = 0.0
    goal_away_p: float = 0.0
    goal_home_xg: float = 0.0
    goal_away_xg: float = 0.0

    # Flags
    pi_missing: bool = False
    elo_missing: bool = False
    goal_missing: bool = False


# ── Multi-model backtest engine ─────────────────────────────────────────────

@dataclass
class MultiModelBacktestResult:
    """Results from comparing multiple models on the same holdout."""
    holdout_name: str
    n_common: int
    n_total: int
    pi_metrics: Optional[BacktestMetrics] = None
    elo_metrics: Optional[BacktestMetrics] = None
    blend_metrics: Optional[BacktestMetrics] = None
    goal_metrics: Optional[BacktestMetrics] = None
    fit_time_seconds: float = 0.0


def run_multi_model_backtest(
    matches: list[GoalMatch],
    holdout: HoldoutPeriod,
    elo_snapshots: dict,
    pi_learning_rate: float = 0.005,
    goal_shrinkage: float = 5.0,
    max_train_years: float = 20.0,
    min_train_matches: int = 50,
    verbose: bool = False,
) -> MultiModelBacktestResult:
    """Run all four models on the same holdout with identical training data.

    For each date in the holdout:
      1. Train pi-ratings on all matches before date
      2. Train goal model on all matches before date
      3. Look up Elo for both teams before date
      4. Predict all four models for each match on that date

    All models see the same training cutoff.  Elo uses strict-less-than date.
    Pi-ratings use cutoff-based filtering.  Goal model uses chronological window.

    Returns:
        MultiModelBacktestResult with per-model metrics on common sample.
    """
    import time as _time

    holdout_matches = sorted(
        [m for m in matches if holdout.start_date <= m.match_date <= holdout.end_date],
        key=lambda m: m.match_date,
    )

    if not holdout_matches:
        return MultiModelBacktestResult(holdout_name=holdout.name, n_common=0, n_total=0)

    # Group by date
    by_date: dict[date, list[GoalMatch]] = defaultdict(list)
    for m in holdout_matches:
        by_date[m.match_date].append(m)

    # Collect predictions per model
    rows: list[ComparisonRow] = []
    fit_time = 0.0

    for d in sorted(by_date.keys()):
        day_matches = by_date[d]
        max_train_date = d - timedelta(days=int(max_train_years * 365.25))
        train = [m for m in matches if max_train_date <= m.match_date < d]

        if len(train) < min_train_matches:
            continue

        t0 = _time.time()

        # Pi-ratings — convert GoalMatch to dicts for compute_pi_ratings
        train_dicts = [
            {
                "date": m.match_date.isoformat(),
                "home_team_id": m.home_team_id,
                "away_team_id": m.away_team_id,
                "home_goals": m.home_goals,
                "away_goals": m.away_goals,
                "result": m.result,
            }
            for m in train
        ]
        pi_ratings = compute_pi_ratings(train_dicts, cutoff=d.isoformat(), learning_rate=pi_learning_rate)

        # Goal model
        try:
            goal_model = RegularizedTeamPoissonModel.fit(train, shrinkage=goal_shrinkage, iterations=30)
        except (ValueError, RuntimeError):
            goal_model = None

        fit_time += _time.time() - t0

        for m in day_matches:
            row = ComparisonRow(
                match_date=m.match_date.isoformat(),
                home_team=m.home_team,
                away_team=m.away_team,
                home_goals=m.home_goals,
                away_goals=m.away_goals,
                neutral=m.neutral,
                tournament=m.tournament,
            )

            # Pi-only
            try:
                p = pi_only_probs(m, pi_ratings)
                row.pi_home_p = p["home"]
                row.pi_draw_p = p["draw"]
                row.pi_away_p = p["away"]
            except Exception:
                row.pi_missing = True

            # Elo-only
            try:
                he, hm = elo_at(elo_snapshots, m.home_team, m.match_date)
                ae, am = elo_at(elo_snapshots, m.away_team, m.match_date)
                p = elo_only_probs(he, ae)
                row.elo_home_p = p["home"]
                row.elo_draw_p = p["draw"]
                row.elo_away_p = p["away"]
                row.elo_missing = hm or am
            except Exception:
                row.elo_missing = True

            # Current blend (pi + elo, w_pi=1.0, w_elo=0.0 = pure pi in production)
            try:
                if not row.pi_missing:
                    # Production default: w_pi=1.0, w_elo=0.0
                    row.blend_home_p = row.pi_home_p
                    row.blend_draw_p = row.pi_draw_p
                    row.blend_away_p = row.pi_away_p
            except Exception:
                pass

            # Goal model
            try:
                if goal_model is not None:
                    pred = goal_model.predict(
                        home_team_id=m.home_team_id,
                        away_team_id=m.away_team_id,
                        neutral=m.neutral,
                    )
                    hda = pred["hda_probs"]
                    row.goal_home_p = hda["home"]
                    row.goal_draw_p = hda["draw"]
                    row.goal_away_p = hda["away"]
                    row.goal_home_xg = pred["home_xg"]
                    row.goal_away_xg = pred["away_xg"]
                else:
                    row.goal_missing = True
            except Exception:
                row.goal_missing = True

            rows.append(row)

    # Compute metrics on common sample (all four models present)
    common = [r for r in rows if not (r.pi_missing or r.elo_missing or r.goal_missing)]

    def _probs_and_actuals(model: str):
        """Return (predictions_list, actuals_hg, actuals_ag) for a model column group."""
        preds = []
        ahg = []
        aag = []
        for r in common:
            if model == "pi":
                p = {"home": r.pi_home_p, "draw": r.pi_draw_p, "away": r.pi_away_p}
            elif model == "elo":
                p = {"home": r.elo_home_p, "draw": r.elo_draw_p, "away": r.elo_away_p}
            elif model == "blend":
                p = {"home": r.blend_home_p, "draw": r.blend_draw_p, "away": r.blend_away_p}
            elif model == "goal":
                p = {"home": r.goal_home_p, "draw": r.goal_draw_p, "away": r.goal_away_p}
            else:
                raise ValueError(f"unknown model: {model}")
            preds.append({
                "hda_probs": p,
                "home_xg": r.goal_home_xg,
                "away_xg": r.goal_away_xg,
                "most_likely_score": [0, 0],
                "low_data_flags": [],
            })
            ahg.append(r.home_goals)
            aag.append(r.away_goals)
        return preds, ahg, aag

    result = MultiModelBacktestResult(
        holdout_name=holdout.name,
        n_common=len(common),
        n_total=len(rows),
        fit_time_seconds=round(fit_time, 2),
    )

    for model_name in ("pi", "elo", "blend", "goal"):
        preds, ahg, aag = _probs_and_actuals(model_name)
        if not preds:
            continue
        metrics = compute_metrics(preds, ahg, aag, holdout_name=holdout.name)
        setattr(result, f"{model_name}_metrics", metrics)

    return result
