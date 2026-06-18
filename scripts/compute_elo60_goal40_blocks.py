#!/usr/bin/env python3
"""Compute exact Elo60/Goal40 per-block metrics for all holdout blocks.

Leak-safe chronological backtest: fits goal model per-date using only
matches strictly before that date.  Blends elo-only and goal-only
probabilities at 60/40 for each match, then computes exact metrics.

Resumable: saves results per block to a JSON cache file.
"""

import json
import time
import math
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soccer_ev_model.goal_model import (
    RegularizedTeamPoissonModel,
)
from soccer_ev_model.goal_model_data import (
    GoalMatch,
    build_goal_matches,
    load_raw_matches,
)
from soccer_ev_model.goal_model_backtest import (
    compute_metrics,
    compute_rps,
    compute_brier_score,
    BacktestMetrics,
    HoldoutPeriod,
)
from soccer_ev_model.goal_model_comparison import (
    elo_only_probs,
    blend_probs,
)
from soccer_ev_model.elo_ratings import elo_at, load_elo_ratings

# ── Configuration ───────────────────────────────────────────────────
REPORTS = Path("reports")
CACHE = REPORTS / "elo60_goal40_blocks.json"  # intermediate cache

GOAL_SHRINKAGE = 20
GOAL_ITERS = 30
MAX_TRAIN_YEARS = 20.0
ELO_PATH = Path("data/raw/elo_ratings.json")
W_ELO = 0.6
W_GOAL = 0.4
W_PI = 0.0  # unused for Elo60/Goal40

HOLDOUTS = [
    ("2014_WC", HoldoutPeriod("2014_WC", date(2014, 6, 12), date(2014, 7, 13)),
     lambda m: m.tournament == "FIFA World Cup"),
    ("2018_WC", HoldoutPeriod("2018_WC", date(2018, 6, 14), date(2018, 7, 15)),
     lambda m: m.tournament == "FIFA World Cup"),
    ("2022_WC", HoldoutPeriod("2022_WC", date(2022, 11, 20), date(2022, 12, 18)),
     lambda m: m.tournament == "FIFA World Cup"),
    ("2023_onward", HoldoutPeriod("2023_onward", date(2023, 1, 1), date(2026, 12, 31)),
     lambda m: True),  # All tournaments from 2023+
]

EPS = 1e-10


def outcome_code(hg: int, ag: int) -> int:
    if hg > ag:
        return 0
    if hg == ag:
        return 1
    return 2


def compute_blend_metrics(probs, outcomes, actuals_hg=None, actuals_ag=None):
    """Compute metrics for a set of blended HDA probabilities."""
    n = len(probs)
    if n == 0:
        return None

    probs_arr = np.array(probs)
    outcomes_arr = np.array(outcomes)

    # Log loss
    log_loss = float(-np.mean(np.log(np.clip(probs_arr[np.arange(n), outcomes_arr], EPS, 1.0))))

    # RPS
    rps = compute_rps(probs_arr, outcomes_arr)

    # Brier
    brier = compute_brier_score(probs_arr, outcomes_arr)

    # Top-1
    top1 = float(np.mean(np.argmax(probs_arr, axis=1) == outcomes_arr))

    # Average predicted H/D/A
    avg_h = float(np.mean(probs_arr[:, 0]))
    avg_d = float(np.mean(probs_arr[:, 1]))
    avg_a = float(np.mean(probs_arr[:, 2]))

    # Actual H/D/A
    n_h = int(np.sum(outcomes_arr == 0))
    n_d = int(np.sum(outcomes_arr == 1))
    n_a = int(np.sum(outcomes_arr == 2))
    actual_h = n_h / n
    actual_d = n_d / n
    actual_a = n_a / n

    return {
        "n": n,
        "log_loss": round(log_loss, 6),
        "rps": round(rps, 6),
        "brier": round(brier, 6),
        "top1": round(top1, 4),
        "avg_pred_home": round(avg_h, 6),
        "avg_pred_draw": round(avg_d, 6),
        "avg_pred_away": round(avg_a, 6),
        "actual_home": round(actual_h, 6),
        "actual_draw": round(actual_d, 6),
        "actual_away": round(actual_a, 6),
    }


def run_block(name, holdout, matches, elo_snapshots, tournament_filter=None):
    """Run chronological backtest for Elo60/Goal40 on one holdout block."""
    start = holdout.start_date
    end = holdout.end_date

    print(f"\n{'='*60}")
    print(f"BLOCK: {name}  ({start} → {end})")
    print(f"{'='*60}")

    holdout_matches = sorted(
        [m for m in matches if start <= m.match_date <= end
         and (tournament_filter is None or tournament_filter(m))],
        key=lambda m: m.match_date,
    )

    if not holdout_matches:
        print(f"  No matches in holdout period")
        return None

    # Group by date
    by_date = defaultdict(list)
    for m in holdout_matches:
        by_date[m.match_date].append(m)

    sorted_dates = sorted(by_date.keys())
    n_dates = len(sorted_dates)

    # Per-match storage
    blend_probs_list = []  # list of [h, d, a] arrays
    outcomes_list = []
    actual_hg = []
    actual_ag = []

    goal_model = None
    fit_time = 0.0
    total_matches = 0

    t_start = time.time()

    for idx, d in enumerate(sorted_dates):
        day_matches = by_date[d]

        # Training: matches strictly before date d, within max window
        max_train_date = d - timedelta(days=int(MAX_TRAIN_YEARS * 365.25))
        train = [m for m in matches if max_train_date <= m.match_date < d]

        if len(train) < 50:
            print(f"  [{idx+1}/{n_dates}] {d}: SKIP (train={len(train)})")
            continue

        # Fit goal model
        t0 = time.time()
        try:
            goal_model = RegularizedTeamPoissonModel.fit(
                train, shrinkage=GOAL_SHRINKAGE, iterations=GOAL_ITERS
            )
        except (ValueError, RuntimeError) as e:
            print(f"  [{idx+1}/{n_dates}] {d}: Goal FAIL ({e})")
            continue
        fit_time += time.time() - t0

        if idx > 0 and (idx % 50 == 0 or idx == n_dates - 1):
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{idx+1}/{n_dates}] {d}: {total_matches} matches so far, "
                  f"{elapsed:.0f}s ({rate:.1f} dates/s)")

        # Predict for each match
        for m in day_matches:
            try:
                # Elo-only probabilities (uses default 1500 for missing teams)
                he, _ = elo_at(elo_snapshots, m.home_team, m.match_date)
                ae, _ = elo_at(elo_snapshots, m.away_team, m.match_date)
                elo_p = elo_only_probs(he, ae)

                # Goal model probabilities
                goal_pred = goal_model.predict(
                    home_team_id=m.home_team_id,
                    away_team_id=m.away_team_id,
                    neutral=m.neutral,
                )
                goal_p = goal_pred["hda_probs"]

                # Blend at 60/40
                # blend_probs takes (pi_probs, goal_probs, w_pi, w_goal)
                # But we want elo+goal, not pi+goal. We pass elo_probs as "pi_probs".
                bl = blend_probs(elo_p, goal_p, W_ELO, W_GOAL)

                blend_probs_list.append([bl["home"], bl["draw"], bl["away"]])
                outcomes_list.append(outcome_code(m.home_goals, m.away_goals))
                actual_hg.append(m.home_goals)
                actual_ag.append(m.away_goals)
                total_matches += 1

            except (ValueError, RuntimeError) as e:
                continue

    elapsed_total = time.time() - t_start
    print(f"  Completed: {total_matches} matches in {elapsed_total:.0f}s "
          f"(fit={fit_time:.1f}s, matches/dates rate)")

    if total_matches == 0:
        return None

    metrics = compute_blend_metrics(blend_probs_list, outcomes_list, actual_hg, actual_ag)
    metrics["elapsed_seconds"] = round(elapsed_total, 1)
    metrics["fit_time_seconds"] = round(fit_time, 1)
    return metrics


def main():
    print("=" * 60)
    print("Elo60/Goal40 PER-BLOCK METRICS COMPUTATION")
    print(f"Goal: shrinkage={GOAL_SHRINKAGE}, iterations={GOAL_ITERS}")
    print(f"Blend: Elo={W_ELO}, Goal={W_GOAL}")
    print("=" * 60)

    # Load data
    print("\nLoading matches...")
    raw = load_raw_matches()
    matches, excluded = build_goal_matches(raw)
    print(f"  {len(matches)} usable matches, {sum(excluded.values())} excluded")

    print("\nLoading Elo ratings...")
    elo_snapshots = load_elo_ratings(ELO_PATH)
    print(f"  {len(elo_snapshots)} teams in cache")

    # Load existing cache if available
    results = {}
    if CACHE.exists():
        try:
            results = json.loads(CACHE.read_text())
            print(f"\nLoaded existing cache with {len(results)} blocks")
        except (json.JSONDecodeError, KeyError):
            results = {}
            print("\nCache corrupted, starting fresh")

    # Run each block
    for name, holdout, tournament_filter in HOLDOUTS:
        if name in results and results[name] is not None:
            print(f"\nSkipping {name} (already cached)")
            continue

        metrics = run_block(name, holdout, matches, elo_snapshots, tournament_filter)
        results[name] = metrics

        # Save after each block for resumability
        CACHE.write_text(json.dumps(results, indent=2, default=str))
        print(f"  Cached to {CACHE}")

    # Compute aggregate across blocks
    print("\n" + "=" * 60)
    print("AGGREGATE")
    print("=" * 60)

    # Collect all per-match data for aggregate
    all_probs = []
    all_outcomes = []
    all_blocks = {}

    # Reload from cache to get full state
    results = json.loads(CACHE.read_text())

    for name, block_data in results.items():
        if block_data is None:
            continue
        print(f"  {name:20s}: n={block_data['n']:5d}  "
              f"LL={block_data['log_loss']:.4f}  "
              f"Brier={block_data['brier']:.4f}  "
              f"RPS={block_data['rps']:.4f}  "
              f"Top1={block_data['top1']:.3f}")
        all_blocks[name] = block_data

    # Cross-check aggregate from blend_grid.json
    with open(REPORTS / "blend_grid.json") as f:
        bg = json.load(f)
    cached_agg = bg["results"]["elo_goal_60_40"]
    print(f"\nCross-check vs cached blend_grid (n={cached_agg['n_matches']}):")
    print(f"  Cached:  LL={cached_agg['log_loss']:.4f}  "
          f"Brier={cached_agg['brier']:.4f}  "
          f"RPS={cached_agg['rps']:.4f}  "
          f"Top1={cached_agg['top1']:.3f}  "
          f"AvgPred=[H:{cached_agg['avg_pred_home']:.4f} "
          f"D:{cached_agg['avg_pred_draw']:.4f} "
          f"A:{cached_agg['avg_pred_away']:.4f}]")

    # Also compute Pi30/Elo40/Goal30 for reference
    # Check if we can load Pi30 from bootstrap.json
    with open(REPORTS / "bootstrap.json") as f:
        bs = json.load(f)

    print("\n" + "=" * 60)
    print("REFERENCE: Pi30/Elo40/Goal30 per-block (cached bootstrap)")
    print("=" * 60)
    for bk in ["2014_WC", "2018_WC", "2022_WC", "2023_recent_internationals"]:
        d = bs["blocks"][bk]["best_candidate_blend"]
        print(f"  {bk:30s}: n={d['n_matches']:5d}  "
              f"LL={d['log_loss']:.4f}  "
              f"Brier={d['brier']:.4f}  "
              f"RPS={d['rps']:.4f}  "
              f"Top1={d['top1']:.3f}")


if __name__ == "__main__":
    main()