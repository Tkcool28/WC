"""Run Phase 3, 4, and 5 experiments: recency, importance, Dixon-Coles, hierarchical.

This script tests:
  Phase 3A: Recency weighting (half-life grid)
  Phase 3B: Match-importance weighting (3 schemes)
  Phase 3C: Combined recency + importance
  Phase 4:   Dixon-Coles low-score correction (rho grid)
  Phase 5:   Hierarchical/shrinkage for low-history teams

Output: reports/experiment_results.json, reports/experiment_results.md
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np

from soccer_ev_model.goal_model import (
    GlobalPoissonModel,
    RegularizedTeamPoissonModel,
    dixon_coles_correction,
    importance_weights,
    recency_weights,
    scoreline_matrix,
    summarize_prediction,
)
from soccer_ev_model.goal_model_data import (
    GoalMatch,
    MATCH_IMPORTANCE_WEIGHTS,
    MATCH_IMPORTANCE_WEIGHTS_STRONG,
    build_goal_matches,
    classify_tournament,
    load_raw_matches,
)


# ===========================================================================
# Experiment configuration
# ===========================================================================

RECENCY_HALF_LIVES = [None, 365, 730, 1460, 2920]  # None = no decay

IMPORTANCE_SCHEMES = {
    "none": None,  # uniform weights
    "mild": MATCH_IMPORTANCE_WEIGHTS,
    "strong": MATCH_IMPORTANCE_WEIGHTS_STRONG,
}

DC_RHO_GRID = [-0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10]

SHRINKAGE_VALUES = [5, 10, 20, 40, 80]

# Holdouts
HOLDOUTS = [
    ("2014_WC", date(2014, 6, 12), date(2014, 7, 13)),
    ("2018_WC", date(2018, 6, 14), date(2018, 7, 15)),
    ("2022_WC", date(2022, 11, 20), date(2022, 12, 18)),
    ("2023_onward", date(2023, 1, 1), date(2026, 12, 31)),
]


# ===========================================================================
# Weighted model wrappers
# ===========================================================================

@dataclass
class WeightedRegularizedTeamModel:
    """Wrapper that fits a regularized team model with recency + importance weights.

    For recency: weights are applied by duplicating matches proportionally to their weight.
    For importance: tournament class weights multiply the recency weights.
    """
    base_model: RegularizedTeamPoissonModel
    shrinkage: int = 20

    @classmethod
    def fit(
        cls,
        matches: list[GoalMatch],
        cutoff_date: date,
        half_life_days: float | None = None,
        importance_scheme: dict | None = None,
        shrinkage: int = 20,
    ) -> "WeightedRegularizedTeamModel":
        """Fit with combined recency + importance weights.

        We implement weighting by effective sample size: each match contributes
        proportionally to its weight in the parameter estimation.
        """
        # Compute combined weights
        w_rec = recency_weights(matches, cutoff_date, half_life_days)
        if importance_scheme is not None:
            w_imp = importance_weights(matches, importance_scheme)
            weights = w_rec * w_imp
        else:
            weights = w_rec

        # Normalize weights to sum to n (preserves effective sample size)
        weights = weights / weights.mean()

        # Fit weighted model: use weighted match list
        # For the EM algorithm, we scale each match's contribution by its weight
        # This is done by repeating matches proportionally to their weight
        # For efficiency, we use fractional weights in the accumulation step
        weighted_matches = []
        for m, w in zip(matches, weights):
            # Store weight alongside match
            weighted_matches.append((m, w))

        base_model = _fit_weighted_team_poisson(weighted_matches, shrinkage=shrinkage)
        return cls(base_model=base_model, shrinkage=shrinkage)


def _fit_weighted_team_poisson(
    weighted_matches: list[tuple[GoalMatch, float]],
    shrinkage: float = 20.0,
    iterations: int = 30,
    tolerance: float = 1e-4,
) -> RegularizedTeamPoissonModel:
    """Fit a team Poisson model with per-match weights."""
    if not weighted_matches:
        raise ValueError("empty match set")

    rows = [m for m, _ in weighted_matches]
    n = len(rows)

    total_goals = sum(m.home_goals + m.away_goals for m in rows)
    global_rate = max(total_goals / (2 * n), 1e-6)

    nn = [m for m in rows if not m.neutral]
    if nn:
        home_adv = math.log(max(sum(m.home_goals for m in nn) / len(nn), 1e-6) / global_rate)
    else:
        home_adv = 0.0

    teams = sorted({m.home_team_id for m in rows} | {m.away_team_id for m in rows})
    attacks = {t: 0.0 for t in teams}
    defenses = {t: 0.0 for t in teams}
    counts = {t: 0 for t in teams}
    for m, w in zip(rows, [w for _, w in weighted_matches]):
        counts[m.home_team_id] += w
        counts[m.away_team_id] += w

    converged = False
    iterations_run = 0

    for it in range(iterations):
        iterations_run = it + 1
        scored = {t: 0.0 for t in teams}
        exp_scored = {t: 0.0 for t in teams}
        conceded = {t: 0.0 for t in teams}
        exp_conceded = {t: 0.0 for t in teams}

        for (m, w) in weighted_matches:
            ha = 0.0 if m.neutral else home_adv
            eh = global_rate * math.exp(ha + attacks[m.home_team_id] - defenses[m.away_team_id])
            ea = global_rate * math.exp(attacks[m.away_team_id] - defenses[m.home_team_id])
            scored[m.home_team_id] += w * m.home_goals
            exp_scored[m.home_team_id] += w * eh
            scored[m.away_team_id] += w * m.away_goals
            exp_scored[m.away_team_id] += w * ea
            conceded[m.home_team_id] += w * m.away_goals
            exp_conceded[m.home_team_id] += w * ea
            conceded[m.away_team_id] += w * m.home_goals
            exp_conceded[m.away_team_id] += w * eh

        max_change = 0.0
        new_attacks = {}
        new_defenses = {}
        for t in teams:
            denom_a = exp_scored[t] + shrinkage * global_rate
            denom_d = exp_conceded[t] + shrinkage * global_rate
            if denom_a > 0 and denom_d > 0:
                raw_a = math.log((scored[t] + shrinkage * global_rate) / denom_a)
                raw_d = -math.log((conceded[t] + shrinkage * global_rate) / denom_d)
            else:
                raw_a = 0.0
                raw_d = 0.0
            new_a = 0.5 * attacks[t] + 0.5 * raw_a
            new_d = 0.5 * defenses[t] + 0.5 * raw_d
            max_change = max(max_change, abs(new_a - attacks[t]), abs(new_d - defenses[t]))
            new_attacks[t] = new_a
            new_defenses[t] = new_d

        attacks = new_attacks
        defenses = new_defenses

        mean_a = float(np.mean(list(attacks.values())))
        mean_d = float(np.mean(list(defenses.values())))
        attacks = {t: v - mean_a for t, v in attacks.items()}
        defenses = {t: v - mean_d for t, v in defenses.items()}

        if max_change < tolerance:
            converged = True
            break

    return RegularizedTeamPoissonModel(
        global_rate=global_rate,
        home_advantage=home_adv,
        attacks=attacks,
        defenses=defenses,
        counts={t: int(round(c)) for t, c in counts.items()},
        shrinkage=shrinkage,
        data_cutoff=max(m.match_date for m in rows).isoformat(),
        iterations_run=iterations_run,
        converged=converged,
    )


# ===========================================================================
# Experiment runner
# ===========================================================================

@dataclass
class ExperimentResult:
    """Results from a single experiment configuration."""
    phase: str
    config_name: str
    holdout_name: str
    n_matches: int = 0
    log_loss: float = 0.0
    rps: float = 0.0
    brier: float = 0.0
    top1: float = 0.0
    poisson_nll: float = 0.0
    exact_score: float = 0.0
    mae_total: float = 0.0
    elapsed_seconds: float = 0.0
    params: dict = field(default_factory=dict)


def compute_simple_metrics(predictions, actuals_hg, actuals_ag):
    """Compute core metrics efficiently."""
    n = len(predictions)
    if n == 0:
        return {}

    probs = np.array([[p["hda_probs"]["home"], p["hda_probs"]["draw"], p["hda_probs"]["away"]] for p in predictions])
    outcomes = np.array([0 if h > a else (1 if h == a else 2) for h, a in zip(actuals_hg, actuals_ag)])
    home_xg = np.array([p["home_xg"] for p in predictions])
    away_xg = np.array([p["away_xg"] for p in predictions])
    actual_hg = np.array(actuals_hg, dtype=float)
    actual_ag = np.array(actuals_ag, dtype=float)

    # Log loss
    log_probs = np.log(np.clip(probs[np.arange(n), outcomes], 1e-10, 1.0))
    log_loss = float(-np.mean(log_probs))

    # RPS
    cum_pred = np.cumsum(probs, axis=1)
    cum_actual = np.zeros_like(probs)
    for i in range(n):
        if outcomes[i] == 0:
            cum_actual[i] = [1, 1, 1]
        elif outcomes[i] == 1:
            cum_actual[i] = [0, 1, 1]
        else:
            cum_actual[i] = [0, 0, 1]
    rps = float(np.mean(np.sum((cum_pred - cum_actual) ** 2, axis=1) / 2.0))

    # Brier
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), outcomes] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    # Top-1
    top1 = float(np.mean(np.argmax(probs, axis=1) == outcomes))

    # Poisson NLL
    EPS = 1e-10
    poisson_nll = 0.0
    for i in range(n):
        lam_h = max(home_xg[i], EPS)
        lam_a = max(away_xg[i], EPS)
        poisson_nll += (lam_h - actual_hg[i] * math.log(lam_h) + math.lgamma(actual_hg[i] + 1))
        poisson_nll += (lam_a - actual_ag[i] * math.log(lam_a) + math.lgamma(actual_ag[i] + 1))
    poisson_nll /= n

    # Exact score accuracy
    exact = sum(1 for i in range(n) if predictions[i].get("most_likely_score") == [actuals_hg[i], actuals_ag[i]])

    # MAE total
    mae_total = float(np.mean(np.abs((home_xg + away_xg) - (actual_hg + actual_ag))))

    return {
        "n_matches": n,
        "log_loss": log_loss,
        "rps": rps,
        "brier": brier,
        "top1": top1,
        "poisson_nll": poisson_nll,
        "exact_score": exact / n,
        "mae_total": mae_total,
    }


def run_weighted_backtest(
    model_name: str,
    matches: list[GoalMatch],
    holdout_name: str,
    holdout_start: date,
    holdout_end: date,
    half_life_days: float | None = None,
    importance_scheme: dict | None = None,
    shrinkage: int = 20,
    max_train_years: float = 10.0,
    dc_rho: float | None = None,
) -> dict:
    """Run a backtest with optional recency, importance, and Dixon-Coles."""
    holdout_matches = [m for m in matches if holdout_start <= m.match_date <= holdout_end]
    holdout_matches.sort(key=lambda m: m.match_date)

    if not holdout_matches:
        return {"n_matches": 0}

    by_date: dict[date, list[GoalMatch]] = defaultdict(list)
    for m in holdout_matches:
        by_date[m.match_date].append(m)

    predictions = []
    actuals_hg = []
    actuals_ag = []

    t0 = time.time()

    for d in sorted(by_date.keys()):
        day_matches = by_date[d]
        max_train_date = d - timedelta(days=int(max_train_years * 365.25))
        train = [m for m in matches if max_train_date <= m.match_date < d]

        if len(train) < 50:
            continue

        try:
            if model_name == "global_poisson":
                model = GlobalPoissonModel.fit(train)
                for m in day_matches:
                    pred = model.predict(neutral=m.neutral)
                    # Apply Dixon-Coles if specified
                    if dc_rho is not None:
                        pred = _apply_dc_to_prediction(pred, dc_rho)
                    predictions.append(pred)
                    actuals_hg.append(m.home_goals)
                    actuals_ag.append(m.away_goals)
            elif model_name == "weighted_team":
                wmodel = WeightedRegularizedTeamModel.fit(
                    train, d, half_life_days, importance_scheme, shrinkage
                )
                for m in day_matches:
                    pred = wmodel.base_model.predict(
                        home_team_id=m.home_team_id,
                        away_team_id=m.away_team_id,
                        neutral=m.neutral,
                    )
                    if dc_rho is not None:
                        pred = _apply_dc_to_prediction(pred, dc_rho)
                    predictions.append(pred)
                    actuals_hg.append(m.home_goals)
                    actuals_ag.append(m.away_goals)
        except (ValueError, RuntimeError):
            continue

    elapsed = time.time() - t0
    metrics = compute_simple_metrics(predictions, actuals_hg, actuals_ag)
    metrics["elapsed_seconds"] = round(elapsed, 1)
    return metrics


def _apply_dc_to_prediction(pred: dict, rho: float) -> dict:
    """Apply Dixon-Coles correction to a prediction."""
    home_xg = pred["home_xg"]
    away_xg = pred["away_xg"]
    max_g = 15

    # Build unnormalized matrix
    hp = np.array([math.exp(-home_xg) * home_xg ** i / math.factorial(i) for i in range(max_g + 1)])
    ap = np.array([math.exp(-away_xg) * away_xg ** i / math.factorial(i) for i in range(max_g + 1)])
    unnorm = np.outer(hp, ap)

    corrected = dixon_coles_correction(unnorm, home_xg, away_xg, rho, max_goals=max_g)

    # Rebuild prediction from corrected matrix
    h = float(np.tril(corrected, -1).sum())
    d = float(np.trace(corrected))
    a = float(np.triu(corrected, 1).sum())
    idx = np.unravel_index(int(np.argmax(corrected)), corrected.shape)

    pred = dict(pred)  # copy
    pred["score_probs"] = corrected.tolist()
    pred["hda_probs"] = {"home": h, "draw": d, "away": a}
    pred["most_likely_score"] = [int(idx[0]), int(idx[1])]
    pred["dc_rho"] = rho
    return pred


def run_experiments(matches: list[GoalMatch]) -> list[ExperimentResult]:
    """Run all Phase 3, 4, 5 experiments."""
    results = []

    # ============================================================
    # Phase 3A: Recency weighting
    # ============================================================
    print("\n=== Phase 3A: Recency Weighting ===")
    for half_life in RECENCY_HALF_LIVES:
        hl_name = f"{half_life}d" if half_life else "none"
        for holdout_name, ho_start, ho_end in HOLDOUTS:
            m = run_weighted_backtest(
                "weighted_team", matches, holdout_name, ho_start, ho_end,
                half_life_days=half_life, shrinkage=20, max_train_years=5,
            )
            if m["n_matches"] > 0:
                results.append(ExperimentResult(
                    phase="3A_recency", config_name=f"hl={hl_name}",
                    holdout_name=holdout_name, **m,
                    params={"half_life_days": half_life},
                ))
                print(f"  hl={hl_name:6s} | {holdout_name:12s} | N={m['n_matches']:4d} | "
                      f"LogL={m['log_loss']:.4f} | RPS={m['rps']:.4f} | {m['elapsed_seconds']:.0f}s")

    # ============================================================
    # Phase 3B: Match-importance weighting
    # ============================================================
    print("\n=== Phase 3B: Match Importance ===")
    for scheme_name, scheme in IMPORTANCE_SCHEMES.items():
        for holdout_name, ho_start, ho_end in HOLDOUTS:
            m = run_weighted_backtest(
                "weighted_team", matches, holdout_name, ho_start, ho_end,
                importance_scheme=scheme, shrinkage=20, max_train_years=5,
            )
            if m["n_matches"] > 0:
                results.append(ExperimentResult(
                    phase="3B_importance", config_name=f"scheme={scheme_name}",
                    holdout_name=holdout_name, **m,
                    params={"scheme": scheme_name},
                ))
                print(f"  scheme={scheme_name:8s} | {holdout_name:12s} | N={m['n_matches']:4d} | "
                      f"LogL={m['log_loss']:.4f} | RPS={m['rps']:.4f} | {m['elapsed_seconds']:.0f}s")

    # ============================================================
    # Phase 3C: Combined recency + importance
    # ============================================================
    print("\n=== Phase 3C: Combined Recency + Importance ===")
    best_half_life = 736  # ~2 years
    for scheme_name, scheme in [("mild", MATCH_IMPORTANCE_WEIGHTS), ("strong", MATCH_IMPORTANCE_WEIGHTS_STRONG)]:
        for holdout_name, ho_start, ho_end in HOLDOUTS:
            m = run_weighted_backtest(
                "weighted_team", matches, holdout_name, ho_start, ho_end,
                half_life_days=best_half_life, importance_scheme=scheme,
                shrinkage=20, max_train_years=5,
            )
            if m["n_matches"] > 0:
                results.append(ExperimentResult(
                    phase="3C_combined", config_name=f"hl=736d+scheme={scheme_name}",
                    holdout_name=holdout_name, **m,
                    params={"half_life_days": 736, "scheme": scheme_name},
                ))
                print(f"  hl=736+{scheme_name:8s} | {holdout_name:12s} | N={m['n_matches']:4d} | "
                      f"LogL={m['log_loss']:.4f} | RPS={m['rps']:.4f} | {m['elapsed_seconds']:.0f}s")

    # ============================================================
    # Phase 4: Dixon–Coles
    # ============================================================
    print("\n=== Phase 4: Dixon-Coles ===")
    for rho in DC_RHO_GRID:
        for holdout_name, ho_start, ho_end in HOLDOUTS:
            m = run_weighted_backtest(
                "weighted_team", matches, holdout_name, ho_start, ho_end,
                half_life_days=best_half_life, importance_scheme=MATCH_IMPORTANCE_WEIGHTS,
                shrinkage=20, max_train_years=5, dc_rho=rho,
            )
            if m["n_matches"] > 0:
                results.append(ExperimentResult(
                    phase="4_dc", config_name=f"rho={rho:.2f}",
                    holdout_name=holdout_name, **m,
                    params={"dc_rho": rho},
                ))
                print(f"  rho={rho:+.2f}     | {holdout_name:12s} | N={m['n_matches']:4d} | "
                      f"LogL={m['log_loss']:.4f} | RPS={m['rps']:.4f} | {m['elapsed_seconds']:.0f}s")

    # ============================================================
    # Phase 5: Shrinkage grid
    # ============================================================
    print("\n=== Phase 5: Shrinkage Grid ===")
    for shrinkage in SHRINKAGE_VALUES:
        for holdout_name, ho_start, ho_end in HOLDOUTS:
            m = run_weighted_backtest(
                "weighted_team", matches, holdout_name, ho_start, ho_end,
                shrinkage=shrinkage, max_train_years=5,
            )
            if m["n_matches"] > 0:
                results.append(ExperimentResult(
                    phase="5_shrinkage", config_name=f"shrink={shrinkage}",
                    holdout_name=holdout_name, **m,
                    params={"shrinkage": shrinkage},
                ))
                print(f"  shrink={shrinkage:3d}   | {holdout_name:12s} | N={m['n_matches']:4d} | "
                      f"LogL={m['log_loss']:.4f} | RPS={m['rps']:.4f} | {m['elapsed_seconds']:.0f}s")

    return results


def write_experiment_report(results: list[ExperimentResult], output_dir: str | Path = "reports"):
    """Write experiment results to JSON and Markdown."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / "experiment_results.json"
    data = {
        "model_version": "goal-model-research-v0.2",
        "phases": ["3A_recency", "3B_importance", "3C_combined", "4_dc", "5_shrinkage"],
        "results": [
            {
                "phase": r.phase,
                "config": r.config_name,
                "holdout": r.holdout_name,
                **{k: v for k, v in r.__dict__.items() if k not in ("phase", "config_name", "holdout_name", "params")},
                "params": r.params,
            }
            for r in results
        ],
    }
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    # Markdown summary
    md_lines = ["# Experiment Results: Phases 3-5", ""]

    # Aggregate by phase and config
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        agg[r.phase][r.config_name].append(r)

    for phase in sorted(agg.keys()):
        md_lines += [f"## {phase}", ""]
        md_lines.append("| Config | Holdout | N | Log Loss | RPS | Brier | Top-1 | Poisson NLL | Exact% | MAE Total | Time(s) |")
        md_lines.append("|--------|---------|---|----------|-----|-------|-------|-------------|--------|-----------|---------|")

        # Find best config by average log loss across holdouts
        config_avg = {}
        for config, res_list in agg[phase].items():
            avg_ll = np.mean([r.log_loss for r in res_list if r.n_matches > 0])
            config_avg[config] = avg_ll

        best_config = min(config_avg, key=config_avg.get) if config_avg else ""

        for config in sorted(agg[phase].keys()):
            for r in sorted(agg[phase][config], key=lambda x: x.holdout_name):
                marker = " ★" if config == best_config else ""
                md_lines.append(
                    f"| {r.config_name}{marker} | {r.holdout_name} | {r.n_matches} "
                    f"| {r.log_loss:.4f} | {r.rps:.4f} | {r.brier:.4f} | {r.top1:.3f} "
                    f"| {r.poisson_nll:.4f} | {r.exact_score:.3f} | {r.mae_total:.3f} | {r.elapsed_seconds:.0f} |"
                )

        # Summary for best config
        md_lines.append("")
        bc_results = agg[phase][best_config]
        avg_ll = np.mean([r.log_loss for r in bc_results])
        avg_rps = np.mean([r.rps for r in bc_results])
        total_n = sum(r.n_matches for r in bc_results)
        md_lines.append(f"**Best config:** `{best_config}` | Avg LogL: {avg_ll:.4f} | Avg RPS: {avg_rps:.4f} | Total N: {total_n}")
        md_lines.append("")

    md_path = output_dir / "experiment_results.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return json_path


def main():
    print("Loading matches...")
    raw = load_raw_matches()
    matches, excluded = build_goal_matches(raw)
    print(f"  {len(matches)} usable matches")

    results = run_experiments(matches)

    json_path = write_experiment_report(results)
    print(f"\nWrote {json_path}")
    print(f"Wrote reports/experiment_results.md")


if __name__ == "__main__":
    main()
