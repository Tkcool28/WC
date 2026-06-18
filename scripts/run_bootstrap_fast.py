#!/usr/bin/env python3
"""Fast bootstrap with reduced iterations.

Usage:
    cd /root/WC-goal-model && PYTHONPATH=/root/WC-goal-model \
        /usr/local/lib/hermes-agent/venv/bin/python scripts/run_bootstrap_fast.py
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from soccer_ev_model.goal_model import RegularizedTeamPoissonModel
from soccer_ev_model.goal_model_data import build_goal_matches, load_raw_matches, classify_tournament
from soccer_ev_model.goal_model_backtest import compute_metrics, HOLDOUT_2022_WC

REPORTS = Path("reports")
ITERS = 5  # reduced for speed — convergence is typically achieved by 5 on 28k+ rows
N_BOOTSTRAPS = 50  # 50 is sufficient for CI estimation


def main():
    t_start = time.time()
    print(f"BOOTSTRAP (fast, ITERS={ITERS}, N={N_BOOTSTRAPS})", flush=True)

    raw = load_raw_matches()
    matches, _ = build_goal_matches(raw)

    holdout_matches = sorted(
        [m for m in matches
         if HOLDOUT_2022_WC.start_date <= m.match_date <= HOLDOUT_2022_WC.end_date],
        key=lambda m: m.match_date,
    )

    by_tournament = defaultdict(list)
    for m in holdout_matches:
        by_tournament[classify_tournament(m.tournament)].append(m)

    tournaments = list(by_tournament.keys())
    print(f"  Tournaments: {tournaments}", flush=True)

    lls = []
    rps_list = []
    brier_list = []
    rng = np.random.RandomState(42)

    for b in range(N_BOOTSTRAPS):
        boot_tournaments = list(rng.choice(tournaments, size=len(tournaments), replace=True))
        boot_by_date = defaultdict(list)
        for t in boot_tournaments:
            for m in by_tournament[t]:
                boot_by_date[m.match_date].append(m)

        preds = []
        ahg = []
        aag = []
        for d in sorted(boot_by_date.keys()):
            day = boot_by_date[d]
            train = [m for m in matches if m.match_date < d]
            if len(train) < 50:
                continue
            try:
                model = RegularizedTeamPoissonModel.fit(train, shrinkage=5, iterations=ITERS)
            except (ValueError, RuntimeError):
                continue
            for m in day:
                try:
                    p = model.predict(
                        home_team_id=m.home_team_id,
                        away_team_id=m.away_team_id,
                        neutral=m.neutral,
                    )
                    preds.append(p)
                    ahg.append(m.home_goals)
                    aag.append(m.away_goals)
                except Exception:
                    continue

        if preds:
            m = compute_metrics(preds, ahg, aag)
            lls.append(m.log_loss)
            rps_list.append(m.ranked_probability_score)
            brier_list.append(m.brier_score)

        if (b + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  {b+1}/{N_BOOTSTRAPS} done ({elapsed:.0f}s)", flush=True)

    lls.sort()
    rps_list.sort()
    brier_list.sort()
    n = len(lls)

    result = {
        "n_bootstrap": N_BOOTSTRAPS,
        "n_valid": n,
        "iterations_per_fit": ITERS,
        "log_loss": {
            "mean": round(float(np.mean(lls)), 4),
            "std": round(float(np.std(lls)), 4),
            "median": round(float(lls[n // 2]), 4),
            "ci_2_5": round(float(lls[int(n * 0.025)]), 4),
            "ci_97_5": round(float(lls[int(n * 0.975)]), 4),
        },
        "rps": {
            "mean": round(float(np.mean(rps_list)), 4),
            "std": round(float(np.std(rps_list)), 4),
            "median": round(float(rps_list[n // 2]), 4),
            "ci_2_5": round(float(rps_list[int(n * 0.025)]), 4),
            "ci_97_5": round(float(rps_list[int(n * 0.975)]), 4),
        },
        "brier": {
            "mean": round(float(np.mean(brier_list)), 4),
            "std": round(float(np.std(brier_list)), 4),
            "median": round(float(brier_list[n // 2]), 4),
            "ci_2_5": round(float(brier_list[int(n * 0.025)]), 4),
            "ci_97_5": round(float(brier_list[int(n * 0.975)]), 4),
        },
    }

    print(f"\n  log_loss: mean={result['log_loss']['mean']} "
          f"std={result['log_loss']['std']} "
          f"CI=[{result['log_loss']['ci_2_5']}, {result['log_loss']['ci_97_5']}]", flush=True)
    print(f"  rps:      mean={result['rps']['mean']} "
          f"CI=[{result['rps']['ci_2_5']}, {result['rps']['ci_97_5']}]", flush=True)
    print(f"  brier:    mean={result['brier']['mean']} "
          f"CI=[{result['brier']['ci_2_5']}, {result['brier']['ci_97_5']}]", flush=True)
    print(f"\n  Time: {time.time() - t_start:.1f}s", flush=True)

    with open(REPORTS / "bootstrap.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
