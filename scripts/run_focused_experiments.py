"""Run focused Phase 3, 4, 5 experiments on 2023+ holdout only.

This is a streamlined version that runs in ~30 minutes instead of hours.
Tests the most important configurations on the largest holdout (3613 matches).
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from soccer_ev_model.goal_model import (
    GlobalPoissonModel,
    RegularizedTeamPoissonModel,
    dixon_coles_correction,
    importance_weights,
    recency_weights,
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


@dataclass
class ExperimentResult:
    phase: str = ""
    config_name: str = ""
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


HOLDOUT_2023 = ("2023_onward", date(2023, 1, 1), date(2026, 12, 31))
HOLDOUT_2022 = ("2022_WC", date(2022, 11, 20), date(2022, 12, 18))


def _fit_weighted_team_poisson(weighted_matches, shrinkage=20.0, iterations=30, tolerance=1e-4):
    """Fit team Poisson with per-match weights."""
    if not weighted_matches:
        raise ValueError("empty")

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
        global_rate=global_rate, home_advantage=home_adv,
        attacks=attacks, defenses=defenses,
        counts={t: int(round(c)) for t, c in counts.items()},
        shrinkage=shrinkage,
        data_cutoff=max(m.match_date for m in rows).isoformat(),
        iterations_run=iterations_run, converged=converged,
    )


def run_experiment(
    name: str,
    matches: list[GoalMatch],
    ho_name: str,
    ho_start: date,
    ho_end: date,
    half_life=None,
    importance_scheme=None,
    shrinkage=20,
    dc_rho=None,
    max_train_years=5,
) -> ExperimentResult:
    """Run a single experiment configuration."""
    ho_matches = [m for m in matches if ho_start <= m.match_date <= ho_end]
    ho_matches.sort(key=lambda m: m.match_date)

    if not ho_matches:
        return ExperimentResult(config_name=name, n_matches=0)

    by_date = defaultdict(list)
    for m in ho_matches:
        by_date[m.match_date].append(m)

    predictions = []
    actuals_hg = []
    actuals_ag = []
    t0 = time.time()

    for d in sorted(by_date.keys()):
        day_matches = by_date[d]
        max_train = d - timedelta(days=int(max_train_years * 365.25))
        train = [m for m in matches if max_train <= m.match_date < d]

        if len(train) < 50:
            continue

        try:
            # Compute weights
            w_rec = recency_weights(train, d, half_life)
            if importance_scheme is not None:
                w_imp = importance_weights(train, importance_scheme)
                weights = w_rec * w_imp
            else:
                weights = w_rec
            weights = weights / weights.mean()

            weighted = list(zip(train, weights))
            model = _fit_weighted_team_poisson(weighted, shrinkage=shrinkage, iterations=30)

            for m in day_matches:
                pred = model.predict(home_team_id=m.home_team_id, away_team_id=m.away_team_id, neutral=m.neutral)
                if dc_rho is not None:
                    pred = _apply_dc(pred, dc_rho)
                predictions.append(pred)
                actuals_hg.append(m.home_goals)
                actuals_ag.append(m.away_goals)
        except (ValueError, RuntimeError):
            continue

    elapsed = time.time() - t0
    metrics = _compute_metrics(predictions, actuals_hg, actuals_ag)
    metrics["elapsed_seconds"] = round(elapsed, 1)
    return ExperimentResult(config_name=name, **metrics,
                           params={"half_life": half_life, "shrinkage": shrinkage, "dc_rho": dc_rho})


def _apply_dc(pred, rho):
    home_xg = pred["home_xg"]
    away_xg = pred["away_xg"]
    max_g = 15
    hp = np.array([math.exp(-home_xg) * home_xg ** i / math.factorial(i) for i in range(max_g + 1)])
    ap = np.array([math.exp(-away_xg) * away_xg ** i / math.factorial(i) for i in range(max_g + 1)])
    unnorm = np.outer(hp, ap)
    corrected = dixon_coles_correction(unnorm, home_xg, away_xg, rho, max_goals=max_g)
    h = float(np.tril(corrected, -1).sum())
    d = float(np.trace(corrected))
    a = float(np.triu(corrected, 1).sum())
    idx = np.unravel_index(int(np.argmax(corrected)), corrected.shape)
    pred = dict(pred)
    pred["hda_probs"] = {"home": h, "draw": d, "away": a}
    pred["most_likely_score"] = [int(idx[0]), int(idx[1])]
    return pred


def _compute_metrics(predictions, actuals_hg, actuals_ag):
    n = len(predictions)
    if n == 0:
        return {"n_matches": 0}

    EPS = 1e-10
    probs = np.array([[p["hda_probs"]["home"], p["hda_probs"]["draw"], p["hda_probs"]["away"]] for p in predictions])
    outcomes = np.array([0 if h > a else (1 if h == a else 2) for h, a in zip(actuals_hg, actuals_ag)])
    home_xg = np.array([p["home_xg"] for p in predictions])
    away_xg = np.array([p["away_xg"] for p in predictions])
    a_hg = np.array(actuals_hg, dtype=float)
    a_ag = np.array(actuals_ag, dtype=float)

    log_loss = float(-np.mean(np.log(np.clip(probs[np.arange(n), outcomes], EPS, 1.0))))

    cum_pred = np.cumsum(probs, axis=1)
    cum_actual = np.zeros_like(probs)
    for i in range(n):
        if outcomes[i] == 0: cum_actual[i] = [1, 1, 1]
        elif outcomes[i] == 1: cum_actual[i] = [0, 1, 1]
        else: cum_actual[i] = [0, 0, 1]
    rps = float(np.mean(np.sum((cum_pred - cum_actual) ** 2, axis=1) / 2.0))

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), outcomes] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    top1 = float(np.mean(np.argmax(probs, axis=1) == outcomes))

    pnll = 0.0
    for i in range(n):
        lh, la = max(home_xg[i], EPS), max(away_xg[i], EPS)
        pnll += lh - a_hg[i] * math.log(lh) + math.lgamma(a_hg[i] + 1)
        pnll += la - a_ag[i] * math.log(la) + math.lgamma(a_ag[i] + 1)
    pnll /= n

    exact = sum(1 for i in range(n) if predictions[i].get("most_likely_score") == [actuals_hg[i], actuals_ag[i]])
    mae_total = float(np.mean(np.abs((home_xg + away_xg) - (a_hg + a_ag))))

    return {
        "n_matches": n, "log_loss": log_loss, "rps": rps, "brier": brier,
        "top1": top1, "poisson_nll": pnll, "exact_score": exact / n, "mae_total": mae_total,
    }


def main():
    print("Loading matches...")
    raw = load_raw_matches()
    matches, _ = build_goal_matches(raw)
    print(f"  {len(matches)} matches")

    results = []
    ho_name, ho_start, ho_end = HOLDOUT_2023

    experiments = [
        # Phase 3A: Recency (on 2023+ only)
        ("3A_baseline", None, None, 20, None),
        ("3A_recency_365d", 365, None, 20, None),
        ("3A_recency_730d", 730, None, 20, None),

        # Phase 3B: Importance (on 2023+ only)
        ("3B_importance_mild", None, MATCH_IMPORTANCE_WEIGHTS, 20, None),
        ("3B_importance_strong", None, MATCH_IMPORTANCE_WEIGHTS_STRONG, 20, None),

        # Phase 3C: Combined
        ("3C_rec365_mild", 365, MATCH_IMPORTANCE_WEIGHTS, 20, None),
        ("3C_rec730_mild", 730, MATCH_IMPORTANCE_WEIGHTS, 20, None),

        # Phase 4: Dixon-Coles (with best recency+importance)
        ("4_dc_rho_neg10", 365, MATCH_IMPORTANCE_WEIGHTS, 20, -0.10),
        ("4_dc_rho_neg05", 365, MATCH_IMPORTANCE_WEIGHTS, 20, -0.05),
        ("4_dc_rho_00", 365, MATCH_IMPORTANCE_WEIGHTS, 20, 0.00),
        ("4_dc_rho_pos05", 365, MATCH_IMPORTANCE_WEIGHTS, 20, 0.05),

        # Phase 5: Shrinkage
        ("5_shrink_5", 365, MATCH_IMPORTANCE_WEIGHTS, 5, -0.10),
        ("5_shrink_40", 365, MATCH_IMPORTANCE_WEIGHTS, 40, -0.10),
        ("5_shrink_80", 365, MATCH_IMPORTANCE_WEIGHTS, 80, -0.10),
    ]

    for name, half_life, importance_scheme, shrinkage, dc_rho in experiments:
        label = f"hl={half_life},imp={'Y' if importance_scheme else 'N'},sh={shrinkage},dc={dc_rho}"
        print(f"\n  {name}: {label}")
        t0 = time.time()
        r = run_experiment(name, matches, ho_name, ho_start, ho_end,
                           half_life=half_life, importance_scheme=importance_scheme, shrinkage=shrinkage, dc_rho=dc_rho)
        print(f"    N={r.n_matches}, LogL={r.log_loss:.4f}, RPS={r.rps:.4f}, "
              f"Top1={r.top1:.3f}, {r.elapsed_seconds:.0f}s")
        results.append(r)

    # Write report
    out = Path("reports")
    out.mkdir(exist_ok=True)

    json_data = {
        "model_version": "goal-model-research-v0.2",
        "holdout": ho_name,
        "results": [
            {"config": r.config_name, "params": r.params, "n": r.n_matches,
             "log_loss": r.log_loss, "rps": r.rps, "brier": r.brier,
             "top1": r.top1, "poisson_nll": r.poisson_nll,
             "exact_score": r.exact_score, "mae_total": r.mae_total,
             "time_s": r.elapsed_seconds}
            for r in results
        ],
    }
    (out / "experiment_results.json").write_text(json.dumps(json_data, indent=2), encoding="utf-8")

    # Summary table
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY (2023+ holdout)")
    print("=" * 80)
    print(f"{'Config':30s} | {'N':>5} | {'LogL':>7} | {'RPS':>6} | {'Top1':>5} | {'Time':>5}")
    print("-" * 80)
    best = min(results, key=lambda r: r.log_loss)
    for r in results:
        marker = " *" if r.config_name == best.config_name else "  "
        print(f"{r.config_name:30s}{marker}| {r.n_matches:5d} | {r.log_loss:7.4f} | {r.rps:6.4f} | {r.top1:5.3f} | {r.elapsed_seconds:4.0f}s")
    print(f"\nBest: {best.config_name} LogL={best.log_loss:.4f} RPS={best.rps:.4f}")


if __name__ == "__main__":
    main()
