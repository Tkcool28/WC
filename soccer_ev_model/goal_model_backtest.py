"""Chronological backtest harness for the independent goal model.

Implements a strict expanding-window backtest with no leakage:
  - Training data: all matches strictly before the prediction date
  - Same-date matches: all predicted from the same cutoff
  - After prediction: results consumed before moving to next date

Metrics computed:
  - Outcome: multiclass log loss, RPS, Brier, top-pick accuracy, H/D/A calibration
  - Goal: MAE home/away/total, Poisson NLL, exact-score accuracy, within-one accuracy
  - Diagnostics: predicted vs actual H/D/A frequencies, total goals, low-history counts
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from soccer_ev_model.goal_model import (
    EloPoissonModel,
    GlobalPoissonModel,
    RegularizedTeamPoissonModel,
    scoreline_matrix,
    summarize_prediction,
)
from soccer_ev_model.goal_model_data import GoalMatch, build_goal_matches, load_raw_matches

# Safe epsilon for log computations
_EPS = 1e-10


# ===========================================================================
# Backtest configuration
# ===========================================================================


@dataclass(frozen=True)
class HoldoutPeriod:
    """Defines a holdout period for backtesting."""
    name: str
    start_date: date
    end_date: date
    description: str = ""


# Standard holdout periods
HOLDOUT_2014_WC = HoldoutPeriod(
    "2014_WC", date(2014, 6, 12), date(2014, 7, 13),
    "2014 FIFA World Cup (Brazil)",
)
HOLDOUT_2018_WC = HoldoutPeriod(
    "2018_WC", date(2018, 6, 14), date(2018, 7, 15),
    "2018 FIFA World Cup (Russia)",
)
HOLDOUT_2022_WC = HoldoutPeriod(
    "2022_WC", date(2022, 11, 20), date(2022, 12, 18),
    "2022 FIFA World Cup (Qatar)",
)
HOLDOUT_2023_ONWARD = HoldoutPeriod(
    "2023_onward", date(2023, 1, 1), date(2026, 12, 31),
    "Recent international matches from 2023 onward",
)

DEFAULT_HOLDOUTS = [HOLDOUT_2014_WC, HOLDOUT_2018_WC, HOLDOUT_2022_WC, HOLDOUT_2023_ONWARD]


# ===========================================================================
# Metrics computation
# ===========================================================================


@dataclass
class BacktestMetrics:
    """Container for all backtest metrics."""
    # Sample info
    n_matches: int = 0
    n_excluded: int = 0
    holdout_name: str = ""

    # Outcome metrics
    log_loss: float = 0.0
    ranked_probability_score: float = 0.0
    brier_score: float = 0.0
    top_pick_accuracy: float = 0.0
    home_calibration: float = 0.0  # actual home win rate
    draw_calibration: float = 0.0
    away_calibration: float = 0.0

    # Goal metrics
    mae_home_goals: float = 0.0
    mae_away_goals: float = 0.0
    mae_total_goals: float = 0.0
    poisson_nll: float = 0.0
    exact_score_accuracy: float = 0.0
    within_one_goal_accuracy: float = 0.0

    # Diagnostics
    avg_pred_home_prob: float = 0.0
    avg_pred_draw_prob: float = 0.0
    avg_pred_away_prob: float = 0.0
    actual_home_rate: float = 0.0
    actual_draw_rate: float = 0.0
    actual_away_rate: float = 0.0
    avg_pred_total_goals: float = 0.0
    avg_actual_total_goals: float = 0.0
    n_low_history: int = 0
    n_unseen: int = 0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def compute_rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Compute Ranked Probability Score for ordinal 3-class outcomes.

    Classes ordered: home < draw < away.
    RPS = sum over classes of (cumulative_pred - cumulative_actual)^2
    """
    n = len(outcomes)
    if n == 0:
        return 0.0

    # Cumulative probabilities and outcomes
    cum_pred = np.cumsum(probs, axis=1)  # (n, 3)
    cum_actual = np.zeros_like(probs)
    for i in range(n):
        if outcomes[i] == 0:  # home
            cum_actual[i] = [1, 1, 1]
        elif outcomes[i] == 1:  # draw
            cum_actual[i] = [0, 1, 1]
        else:  # away
            cum_actual[i] = [0, 0, 1]

    # RPS = (1/(K-1)) * sum_k (cum_pred_k - cum_actual_k)^2
    rps = np.mean(np.sum((cum_pred - cum_actual) ** 2, axis=1) / 2.0)
    return float(rps)


def compute_metrics(
    predictions: list[dict],
    actuals_hg: list[int],
    actuals_ag: list[int],
    holdout_name: str = "",
) -> BacktestMetrics:
    """Compute full metrics from predictions and actuals.

    Each prediction dict must have:
        - hda_probs: {home: p, draw: p, away: p}
        - home_xg, away_xg: expected goals
        - score_probs: full matrix (optional, for exact score)
        - low_data_flags: list of flags (optional)
    """
    n = len(predictions)
    m = BacktestMetrics(n_matches=n, holdout_name=holdout_name)

    if n == 0:
        return m

    # Build arrays
    probs = np.array([[p["hda_probs"]["home"], p["hda_probs"]["draw"], p["hda_probs"]["away"]] for p in predictions])
    home_xg = np.array([p["home_xg"] for p in predictions])
    away_xg = np.array([p["away_xg"] for p in predictions])
    actual_hg = np.array(actuals_hg, dtype=float)
    actual_ag = np.array(actuals_ag, dtype=float)

    # Outcome codes: 0=home, 1=draw, 2=away
    outcomes = np.array([
        0 if h > a else (1 if h == a else 2)
        for h, a in zip(actuals_hg, actuals_ag)
    ])

    # --- Outcome metrics ---
    # Log loss
    log_probs = np.log(np.clip(probs[np.arange(n), outcomes], _EPS, 1.0))
    m.log_loss = float(-np.mean(log_probs))

    # RPS
    m.ranked_probability_score = compute_rps(probs, outcomes)

    # Brier score (multiclass)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), outcomes] = 1.0
    m.brier_score = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    # Top-pick accuracy
    top_picks = np.argmax(probs, axis=1)
    m.top_pick_accuracy = float(np.mean(top_picks == outcomes))

    # Calibration
    actual_home = np.mean(outcomes == 0)
    actual_draw = np.mean(outcomes == 1)
    actual_away = np.mean(outcomes == 2)
    m.home_calibration = float(actual_home)
    m.draw_calibration = float(actual_draw)
    m.away_calibration = float(actual_away)

    # --- Goal metrics ---
    m.mae_home_goals = float(np.mean(np.abs(home_xg - actual_hg)))
    m.mae_away_goals = float(np.mean(np.abs(away_xg - actual_ag)))
    m.mae_total_goals = float(np.mean(np.abs((home_xg + away_xg) - (actual_hg + actual_ag))))

    # Poisson NLL
    poisson_nll = 0.0
    for i in range(n):
        lam_h = max(home_xg[i], _EPS)
        lam_a = max(away_xg[i], _EPS)
        # log P(k; lambda) = -lambda + k*log(lambda) - log(k!)
        def log_poisson(k, lam):
            return -lam + k * math.log(lam) - math.lgamma(k + 1)
        poisson_nll -= log_poisson(actuals_hg[i], lam_h) + log_poisson(actuals_ag[i], lam_a)
    m.poisson_nll = float(poisson_nll / n)

    # Exact score accuracy
    exact = sum(
        1 for i in range(n)
        if predictions[i].get("most_likely_score") == [actuals_hg[i], actuals_ag[i]]
    )
    m.exact_score_accuracy = exact / n

    # Within-one-goal accuracy
    within_one = sum(
        1 for i in range(n)
        if abs(home_xg[i] - actual_hg[i]) <= 1 and abs(away_xg[i] - actual_ag[i]) <= 1
    )
    m.within_one_goal_accuracy = within_one / n

    # --- Diagnostics ---
    m.avg_pred_home_prob = float(np.mean(probs[:, 0]))
    m.avg_pred_draw_prob = float(np.mean(probs[:, 1]))
    m.avg_pred_away_prob = float(np.mean(probs[:, 2]))
    m.actual_home_rate = float(actual_home)
    m.actual_draw_rate = float(actual_draw)
    m.actual_away_rate = float(actual_away)
    m.avg_pred_total_goals = float(np.mean(home_xg + away_xg))
    m.avg_actual_total_goals = float(np.mean(actual_hg + actual_ag))

    # Low history / unseen counts
    for p in predictions:
        flags = p.get("low_data_flags", [])
        if any("low_history" in f for f in flags):
            m.n_low_history += 1
        if any("unseen" in f for f in flags):
            m.n_unseen += 1

    return m


# ===========================================================================
# Backtest engine
# ===========================================================================


@dataclass
class BacktestResult:
    """Results from a single model backtest."""
    model_name: str
    holdout_name: str
    metrics: BacktestMetrics
    predictions: list[dict] = field(default_factory=list)
    actuals_hg: list[int] = field(default_factory=list)
    actuals_ag: list[int] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    fit_time_seconds: float = 0.0
    predict_time_seconds: float = 0.0


def run_backtest(
    model_name: str,
    matches: list[GoalMatch],
    holdout: HoldoutPeriod,
    elo_snapshots: dict | None = None,
    min_train_matches: int = 50,
    max_train_years: float = 20.0,
    verbose: bool = False,
) -> BacktestResult:
    """Run a chronological backtest for a single model and holdout period.

    Args:
        model_name: one of "global_poisson", "regularized_team"
        matches: full match list (will be filtered to holdout)
        holdout: holdout period definition
        elo_snapshots: optional Elo rating snapshots for Elo models
        min_train_matches: minimum training matches required
        max_train_years: maximum years of training data (to limit computation)
        verbose: print progress

    Returns:
        BacktestResult with all metrics.
    """
    import time

    # Filter to holdout period
    holdout_matches = [
        m for m in matches
        if holdout.start_date <= m.match_date <= holdout.end_date
    ]
    holdout_matches.sort(key=lambda m: m.match_date)

    if not holdout_matches:
        return BacktestResult(
            model_name=model_name,
            holdout_name=holdout.name,
            metrics=BacktestMetrics(holdout_name=holdout.name),
        )

    predictions = []
    actuals_hg = []
    actuals_ag = []
    dates = []

    # Group by date for same-date prediction
    by_date: dict[date, list[GoalMatch]] = defaultdict(list)
    for m in holdout_matches:
        by_date[m.match_date].append(m)

    fit_time = 0.0
    predict_time = 0.0

    sorted_dates = sorted(by_date.keys())
    n_dates = len(sorted_dates)

    for idx, d in enumerate(sorted_dates):
        day_matches = by_date[d]

        # Training: matches strictly before date d, within max window
        max_train_date = d - timedelta(days=int(max_train_years * 365.25))
        train = [m for m in matches if max_train_date <= m.match_date < d]

        if len(train) < min_train_matches:
            continue

        if verbose and idx % 50 == 0:
            print(f"  [{idx}/{n_dates}] {d}: {len(train)} train, {len(day_matches)} test")

        # Fit model
        t0 = time.time()
        try:
            if model_name == "global_poisson":
                model = GlobalPoissonModel.fit(train)
            elif model_name == "regularized_team":
                model = RegularizedTeamPoissonModel.fit(train, shrinkage=20, iterations=30)
            else:
                raise ValueError(f"unknown model: {model_name}")
        except (ValueError, RuntimeError):
            continue
        fit_time += time.time() - t0

        # Predict all matches on this date
        t0 = time.time()
        for m in day_matches:
            try:
                if model_name == "global_poisson":
                    pred = model.predict(neutral=m.neutral)
                elif model_name == "regularized_team":
                    pred = model.predict(
                        home_team_id=m.home_team_id,
                        away_team_id=m.away_team_id,
                        neutral=m.neutral,
                    )
                else:
                    continue
                predictions.append(pred)
                actuals_hg.append(m.home_goals)
                actuals_ag.append(m.away_goals)
                dates.append(m.match_date.isoformat())
            except (ValueError, RuntimeError):
                continue
        predict_time += time.time() - t0

    metrics = compute_metrics(predictions, actuals_hg, actuals_ag, holdout_name=holdout.name)
    return BacktestResult(
        model_name=model_name,
        holdout_name=holdout.name,
        metrics=metrics,
        predictions=predictions,
        actuals_hg=actuals_hg,
        actuals_ag=actuals_ag,
        dates=dates,
        fit_time_seconds=round(fit_time, 2),
        predict_time_seconds=round(predict_time, 2),
    )


def run_all_backtests(
    matches: list[GoalMatch],
    holdouts: Sequence[HoldoutPeriod] = DEFAULT_HOLDOUTS,
    models: Sequence[str] = ("global_poisson", "regularized_team"),
    elo_snapshots: dict | None = None,
) -> list[BacktestResult]:
    """Run backtests for all model/holdout combinations."""
    results = []
    for holdout in holdouts:
        for model_name in models:
            result = run_backtest(model_name, matches, holdout, elo_snapshots)
            results.append(result)
    return results


# ===========================================================================
# Report generation
# ===========================================================================


def format_metrics_table(results: list[BacktestResult]) -> str:
    """Format results as a Markdown table."""
    lines = [
        "| Model | Holdout | N | Log Loss | RPS | Brier | Top-1 | Home Cal | Draw Cal | Away Cal | MAE Home | MAE Away | MAE Total | Exact% | Within-1% |",
        "|-------|---------|---|----------|-----|-------|-------|----------|----------|----------|----------|----------|-----------|--------|-----------|",
    ]
    for r in results:
        m = r.metrics
        lines.append(
            f"| {r.model_name} | {r.holdout_name} | {m.n_matches} "
            f"| {m.log_loss:.4f} | {m.ranked_probability_score:.4f} | {m.brier_score:.4f} "
            f"| {m.top_pick_accuracy:.3f} "
            f"| {m.home_calibration:.3f} | {m.draw_calibration:.3f} | {m.away_calibration:.3f} "
            f"| {m.mae_home_goals:.3f} | {m.mae_away_goals:.3f} | {m.mae_total_goals:.3f} "
            f"| {m.exact_score_accuracy:.3f} | {m.within_one_goal_accuracy:.3f} |"
        )
    return "\n".join(lines)


def write_backtest_report(
    results: list[BacktestResult],
    output_dir: str | Path = "reports",
) -> Path:
    """Write backtest results to JSON and Markdown."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_data = {
        "model_version": "goal-model-research-v0.2",
        "generated_at": str(np.datetime64("now")),
        "results": [
            {
                "model": r.model_name,
                "holdout": r.holdout_name,
                "fit_time_s": r.fit_time_seconds,
                "predict_time_s": r.predict_time_seconds,
                "metrics": r.metrics.to_dict(),
            }
            for r in results
        ],
    }
    json_path = output_dir / "backtest_results.json"
    import json
    json_path.write_text(json.dumps(json_data, indent=2, sort_keys=True), encoding="utf-8")

    # Markdown
    md_lines = [
        "# Goal Model Backtest Results",
        "",
        f"**Generated:** {json_data['generated_at']}",
        "",
        "## Metrics Summary",
        "",
        format_metrics_table(results),
        "",
        "## Diagnostics",
        "",
        "| Model | Holdout | Avg Pred H | Avg Pred D | Avg Pred A | Actual H | Actual D | Actual A | Avg Pred TG | Avg Actual TG | Low Hist | Unseen |",
        "|-------|---------|------------|------------|------------|----------|----------|----------|-------------|---------------|----------|--------|",
    ]
    for r in results:
        m = r.metrics
        md_lines.append(
            f"| {r.model_name} | {r.holdout_name} "
            f"| {m.avg_pred_home_prob:.3f} | {m.avg_pred_draw_prob:.3f} | {m.avg_pred_away_prob:.3f} "
            f"| {m.actual_home_rate:.3f} | {m.actual_draw_rate:.3f} | {m.actual_away_rate:.3f} "
            f"| {m.avg_pred_total_goals:.3f} | {m.avg_actual_total_goals:.3f} "
            f"| {m.n_low_history} | {m.n_unseen} |"
        )

    md_path = output_dir / "backtest_results.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return json_path
