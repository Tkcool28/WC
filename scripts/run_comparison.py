#!/usr/bin/env python3
"""Generate cached predictions for all 4 models on 2022 WC, then run blend/calibration/robustness.

Usage:
    cd /root/WC-goal-model && PYTHONPATH=/root/WC-goal-model \
        /usr/local/lib/hermes-agent/venv/bin/python scripts/run_comparison.py

Outputs:
    reports/common_sample_predictions.csv
    reports/model_comparison.json
    reports/blend_grid.json
    reports/calibration.json
    reports/robustness.json
"""
import csv
import json
import math
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from soccer_ev_model.goal_model import RegularizedTeamPoissonModel, summarize_prediction
from soccer_ev_model.goal_model_data import build_goal_matches, load_raw_matches, classify_tournament
from soccer_ev_model.goal_model_backtest import (
    BacktestMetrics,
    compute_metrics,
    HoldoutPeriod,
    HOLDOUT_2022_WC,
)
from soccer_ev_model.goal_model_comparison import (
    pi_only_probs,
    elo_only_probs,
    blend_probs,
    confirmation_signal,
)
from soccer_ev_model.elo_ratings import elo_at, load_elo_ratings
from soccer_ev_model.pi_ratings import compute_pi_ratings

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

HOLDOUT = HOLDOUT_2022_WC
ITERS = 20  # model fitting iterations — enough for convergence on 28k+ training rows


def outcome_code(hg: int, ag: int) -> int:
    if hg > ag: return 0
    if hg == ag: return 1
    return 2


def train_dicts(matches):
    return [
        {
            "date": m.match_date.isoformat(),
            "home_team_id": m.home_team_id,
            "away_team_id": m.away_team_id,
            "home_goals": m.home_goals,
            "away_goals": m.away_goals,
            "result": m.result,
        }
        for m in matches
    ]


def generate_predictions(matches, elo_snapshots, holdout):
    """Fit all 4 models once per date, return list of prediction dicts."""
    holdout_matches = sorted(
        [m for m in matches if holdout.start_date <= m.match_date <= holdout.end_date],
        key=lambda m: m.match_date,
    )
    if not holdout_matches:
        return []

    by_date: dict[date, list] = defaultdict(list)
    for m in holdout_matches:
        by_date[m.match_date].append(m)

    rows = []
    t0 = time.time()
    dates = sorted(by_date.keys())

    for i, d in enumerate(dates):
        day_matches = by_date[d]
        train = [m for m in matches if m.match_date < d]
        if len(train) < 50:
            print(f"  [{i+1}/{len(dates)}] {d}: SKIP (train={len(train)})", flush=True)
            continue

        # Fit models
        try:
            pi_ratings = compute_pi_ratings(train_dicts(train), cutoff=d.isoformat())
        except Exception as e:
            print(f"  [{i+1}/{len(dates)}] {d}: Pi FAIL ({e})", flush=True)
            continue

        try:
            goal_model = RegularizedTeamPoissonModel.fit(train, shrinkage=5, iterations=ITERS)
        except Exception as e:
            print(f"  [{i+1}/{len(dates)}] {d}: Goal FAIL ({e})", flush=True)
            continue

        for m in day_matches:
            row = {
                "date": m.match_date.isoformat(),
                "home_team": m.home_team,
                "away_team": m.away_team,
                "home_team_id": m.home_team_id,
                "away_team_id": m.away_team_id,
                "home_goals": m.home_goals,
                "away_goals": m.away_goals,
                "neutral": m.neutral,
                "tournament": m.tournament,
                "pi_ok": False,
                "elo_ok": False,
                "goal_ok": False,
            }

            # Pi-only
            try:
                p = pi_only_probs(m, pi_ratings)
                row["pi_home"] = p["home"]
                row["pi_draw"] = p["draw"]
                row["pi_away"] = p["away"]
                row["pi_ok"] = True
            except Exception:
                pass

            # Elo-only
            try:
                he, _ = elo_at(elo_snapshots, m.home_team, m.match_date)
                ae, _ = elo_at(elo_snapshots, m.away_team, m.match_date)
                p = elo_only_probs(he, ae)
                row["elo_home"] = p["home"]
                row["elo_draw"] = p["draw"]
                row["elo_away"] = p["away"]
                row["elo_ok"] = True
            except Exception:
                pass

            # Goal model
            try:
                g = goal_model.predict(
                    home_team_id=m.home_team_id,
                    away_team_id=m.away_team_id,
                    neutral=m.neutral,
                )
                row["goal_home"] = g["hda_probs"]["home"]
                row["goal_draw"] = g["hda_probs"]["draw"]
                row["goal_away"] = g["hda_probs"]["away"]
                row["goal_home_xg"] = g["home_xg"]
                row["goal_away_xg"] = g["away_xg"]
                row["goal_ok"] = True
            except Exception:
                pass

            rows.append(row)

        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(dates)}] {d}: {len(day_matches)} matches, cumulative {elapsed:.1f}s", flush=True)

    total = time.time() - t0
    print(f"  Predictions generated: {len(rows)} rows in {total:.1f}s", flush=True)
    return rows


def save_predictions_csv(rows, path):
    """Save prediction rows to CSV."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows)} rows to {path}", flush=True)


def load_predictions_csv(path):
    """Load prediction rows from CSV."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert types
            row["home_goals"] = int(row["home_goals"])
            row["away_goals"] = int(row["away_goals"])
            row["neutral"] = row["neutral"] == "True"
            row["pi_ok"] = row["pi_ok"] == "True"
            row["elo_ok"] = row["elo_ok"] == "True"
            row["goal_ok"] = row["goal_ok"] == "True"
            for k in ("pi_home", "pi_draw", "pi_away",
                      "elo_home", "elo_draw", "elo_away",
                      "goal_home", "goal_draw", "goal_away",
                      "goal_home_xg", "goal_away_xg"):
                if k in row:
                    row[k] = float(row[k])
            rows.append(row)
    return rows


def common_sample(rows):
    """Rows where all 4 models produced predictions."""
    return [r for r in rows if r["pi_ok"] and r["elo_ok"] and r["goal_ok"]]


def row_to_pred_dict(r, model):
    """Convert a row to the pred dict format expected by compute_metrics."""
    if model == "pi":
        p = {"home": r["pi_home"], "draw": r["pi_draw"], "away": r["pi_away"]}
    elif model == "elo":
        p = {"home": r["elo_home"], "draw": r["elo_draw"], "away": r["elo_away"]}
    elif model == "blend":
        # Current production blend = pure pi (w_pi=1.0, w_elo=0.0)
        p = {"home": r["pi_home"], "draw": r["pi_draw"], "away": r["pi_away"]}
    elif model == "goal":
        p = {"home": r["goal_home"], "draw": r["goal_draw"], "away": r["goal_away"]}
    else:
        raise ValueError(f"unknown model: {model}")
    return {
        "hda_probs": p,
        "home_xg": r.get("goal_home_xg", 0.0),
        "away_xg": r.get("goal_away_xg", 0.0),
        "most_likely_score": [0, 0],
        "low_data_flags": [],
    }


def compute_model_metrics(rows, model):
    """Compute BacktestMetrics for a model on the common sample."""
    preds = [row_to_pred_dict(r, model) for r in rows]
    ahg = [r["home_goals"] for r in rows]
    aag = [r["away_goals"] for r in rows]
    return compute_metrics(preds, ahg, aag)


def metrics_to_dict(m: BacktestMetrics, include_goals=False):
    d = {
        "n_matches": m.n_matches,
        "log_loss": round(m.log_loss, 4),
        "rps": round(m.ranked_probability_score, 4),
        "brier": round(m.brier_score, 4),
        "top1": round(m.top_pick_accuracy, 3),
        "home_cal": round(m.home_calibration, 3),
        "draw_cal": round(m.draw_calibration, 3),
        "away_cal": round(m.away_calibration, 3),
    }
    if include_goals:
        d.update({
            "mae_home_goals": round(m.mae_home_goals, 3),
            "mae_away_goals": round(m.mae_away_goals, 3),
            "mae_total_goals": round(m.mae_total_goals, 3),
            "poisson_nll": round(m.poisson_nll, 4),
            "exact_score": round(m.exact_score_accuracy, 3),
        })
    return d


def run_comparison(rows):
    """Compare all 4 models on common sample."""
    common = common_sample(rows)
    print(f"\n=== MODEL COMPARISON (common sample n={len(common)}) ===", flush=True)

    result = {
        "holdout": "2022_WC",
        "n_common": len(common),
        "n_total": len(rows),
        "exclusions": {
            "pi_missing": sum(1 for r in rows if not r["pi_ok"]),
            "elo_missing": sum(1 for r in rows if not r["elo_ok"]),
            "goal_missing": sum(1 for r in rows if not r["goal_ok"]),
        },
    }

    for model_name in ("pi", "elo", "blend", "goal"):
        if common:
            m = compute_model_metrics(common, model_name)
            result[model_name] = metrics_to_dict(m, include_goals=(model_name == "goal"))
            print(f"  {model_name:12s}: LL={m.log_loss:.4f} RPS={m.ranked_probability_score:.4f} "
                  f"Top1={m.top_pick_accuracy:.3f} Brier={m.brier_score:.4f}", flush=True)
        else:
            result[model_name] = None
            print(f"  {model_name:12s}: NO DATA", flush=True)

    return result


def run_blend_grid(rows):
    """Test blend weight grid on cached predictions."""
    common = common_sample(rows)
    print(f"\n=== BLEND GRID (n={len(common)}) ===", flush=True)

    blend_weights = [
        (1.0, 0.0, "pi_only"),
        (0.0, 1.0, "goal_only"),
        (0.9, 0.1, "blend_90_10"),
        (0.8, 0.2, "blend_80_20"),
        (0.7, 0.3, "blend_70_30"),
        (0.6, 0.4, "blend_60_40"),
        (0.5, 0.5, "blend_50_50"),
        (0.4, 0.6, "blend_40_60"),
        (0.3, 0.7, "blend_30_70"),
    ]

    results = {}
    best_ll = float("inf")
    best_label = None

    for w_pi, w_goal, label in blend_weights:
        preds = []
        ahg = []
        aag = []
        for r in common:
            pi_p = {"home": r["pi_home"], "draw": r["pi_draw"], "away": r["pi_away"]}
            goal_p = {"home": r["goal_home"], "draw": r["goal_draw"], "away": r["goal_away"]}
            bl_p = blend_probs(pi_p, goal_p, w_pi, w_goal)
            preds.append({"hda_probs": bl_p, "home_xg": 0, "away_xg": 0,
                          "most_likely_score": [0, 0], "low_data_flags": []})
            ahg.append(r["home_goals"])
            aag.append(r["away_goals"])

        m = compute_metrics(preds, ahg, aag)
        results[label] = {
            "log_loss": round(m.log_loss, 4),
            "rps": round(m.ranked_probability_score, 4),
            "brier": round(m.brier_score, 4),
            "top1": round(m.top_pick_accuracy, 3),
            "n": m.n_matches,
        }
        if m.log_loss < best_ll:
            best_ll = m.log_loss
            best_label = label
        print(f"  {label:14s}: LL={m.log_loss:.4f} RPS={m.ranked_probability_score:.4f} "
              f"Top1={m.top_pick_accuracy:.3f}", flush=True)

    print(f"  Best: {best_label} (LL={best_ll:.4f})", flush=True)
    return {"holdout": "2022_WC", "results": results, "best": best_label}


def run_calibration(rows):
    """Calibration tables for goal model."""
    common = common_sample(rows)
    print(f"\n=== CALIBRATION (n={len(common)}) ===", flush=True)

    predictions_hda = []
    outcomes = []
    for r in common:
        predictions_hda.append({"home": r["goal_home"], "draw": r["goal_draw"], "away": r["goal_away"]})
        outcomes.append(outcome_code(r["home_goals"], r["away_goals"]))

    labels = ["home", "draw", "away"]
    bins = defaultdict(lambda: {"count": 0, "sum_pred": 0.0, "sum_actual": 0.0})

    for p, o in zip(predictions_hda, outcomes):
        for label in labels:
            prob = p[label]
            bin_idx = min(int(prob * 10), 9)
            key = (label, bin_idx)
            bins[key]["count"] += 1
            bins[key]["sum_pred"] += prob
            bins[key]["sum_actual"] += 1.0 if labels[o] == label else 0.0

    cal_table = {}
    for (label, bin_idx), data in sorted(bins.items()):
        if data["count"] < 2:
            continue
        lo = bin_idx / 10
        hi = (bin_idx + 1) / 10
        avg_pred = data["sum_pred"] / data["count"]
        avg_actual = data["sum_actual"] / data["count"]
        cal_table[f"{label}_{lo:.1f}-{hi:.1f}"] = {
            "count": data["count"],
            "avg_pred": round(avg_pred, 4),
            "actual_freq": round(avg_actual, 4),
            "cal_error": round(abs(avg_pred - avg_actual), 4),
        }

    # Reliability by confidence
    conf_buckets = defaultdict(lambda: {"preds": [], "outcomes": []})
    for pred, outcome in zip(predictions_hda, outcomes):
        top_prob = max(pred["home"], pred["draw"], pred["away"])
        if top_prob < 0.40:
            bucket = "below_0.40"
        elif top_prob < 0.50:
            bucket = "0.40_0.50"
        elif top_prob < 0.60:
            bucket = "0.50_0.60"
        elif top_prob < 0.70:
            bucket = "0.60_0.70"
        else:
            bucket = "above_0.70"
        conf_buckets[bucket]["preds"].append(pred)
        conf_buckets[bucket]["outcomes"].append(outcome)

    reliability = {}
    for bucket_name in sorted(conf_buckets.keys()):
        data = conf_buckets[bucket_name]
        if not data["preds"]:
            continue
        n = len(data["preds"])
        avg_top = sum(max(p["home"], p["draw"], p["away"]) for p in data["preds"]) / n
        correct = sum(
            1 for p, o in zip(
                [max(p, key=p.get) for p in data["preds"]],
                data["outcomes"],
            )
            if ["home", "draw", "away"][o] == p
        )
        reliability[bucket_name] = {
            "count": n,
            "avg_top_prob": round(avg_top, 4),
            "top_pick_accuracy": round(correct / n, 4),
        }

    print(f"  Calibration bins: {len(cal_table)} entries", flush=True)
    print(f"  Confidence buckets: {list(reliability.keys())}", flush=True)

    return {
        "holdout": "2022_WC",
        "n": len(predictions_hda),
        "calibration_table": cal_table,
        "reliability_by_confidence": reliability,
    }


def run_confirmation(rows):
    """Confirmation/disagreement analysis."""
    common = common_sample(rows)
    print(f"\n=== CONFIRMATION/DISAGREEMENT (n={len(common)}) ===", flush=True)

    agree = 0
    disagree_mild = 0
    disagree_strong = 0
    total = 0

    for r in common:
        ref = {"home": r["pi_home"], "draw": r["pi_draw"], "away": r["pi_away"]}
        goal_pred = {
            "hda_probs": {"home": r["goal_home"], "draw": r["goal_draw"], "away": r["goal_away"]},
            "home_xg": r.get("goal_home_xg", 0),
            "away_xg": r.get("goal_away_xg", 0),
            "most_likely_score": [0, 0],
        }
        sig = confirmation_signal(ref, goal_pred)
        if sig.same_top:
            agree += 1
        if sig.disagreement_level == "mild":
            disagree_mild += 1
        elif sig.disagreement_level == "strong":
            disagree_strong += 1
        total += 1

    result = {
        "holdout": "2022_WC",
        "n": total,
        "same_top_count": agree,
        "same_top_rate": round(agree / total, 4) if total else 0,
        "mild_disagreement": disagree_mild,
        "strong_disagreement": disagree_strong,
    }
    print(f"  Same top: {agree}/{total} ({result['same_top_rate']:.1%})", flush=True)
    print(f"  Mild disagreement: {disagree_mild}", flush=True)
    print(f"  Strong disagreement: {disagree_strong}", flush=True)
    return result


def run_subgroup_analysis(rows):
    """Subgroup metrics for goal model."""
    common = common_sample(rows)
    print(f"\n=== SUBGROUP ANALYSIS ===", flush=True)

    def subgroup_mask(rows, key_func):
        return [r for r in common if key_func(r)]

    subgroups = {
        "neutral": lambda r: r["neutral"],
        "non_neutral": lambda r: not r["neutral"],
        "group_stage": lambda r: "Group" in r.get("tournament", "") or "group" in r.get("tournament", "").lower(),
        "knockout": lambda r: "Round" in r.get("tournament", "") or "Final" in r.get("tournament", ""),
    }

    results = {}
    for sg_name, mask_fn in subgroups.items():
        sg_rows = subgroup_mask(common, mask_fn)
        if len(sg_rows) < 5:
            print(f"  {sg_name}: n={len(sg_rows)} (too few)", flush=True)
            continue
        m = compute_model_metrics(sg_rows, "goal")
        results[sg_name] = {
            "n": len(sg_rows),
            "log_loss": round(m.log_loss, 4),
            "rps": round(m.ranked_probability_score, 4),
            "top1": round(m.top_pick_accuracy, 3),
        }
        print(f"  {sg_name:16s}: n={len(sg_rows):3d} LL={m.log_loss:.4f} RPS={m.ranked_probability_score:.4f} "
              f"Top1={m.top_pick_accuracy:.3f}", flush=True)

    return results


def main():
    t_start = time.time()
    print("=" * 60, flush=True)
    print("COMPARISON EXPERIMENT: Phases 8-10", flush=True)
    print("=" * 60, flush=True)

    # Load data
    print("\n[1/6] Loading data...", flush=True)
    raw = load_raw_matches()
    matches, excluded = build_goal_matches(raw)
    elo_snapshots = load_elo_ratings("data/raw/elo_ratings.json")
    print(f"  {len(matches)} matches, {sum(excluded.values())} excluded", flush=True)
    print(f"  Elo: {len(elo_snapshots)} teams", flush=True)

    # Generate predictions
    print(f"\n[2/6] Generating predictions for {HOLDOUT.name}...", flush=True)
    rows = generate_predictions(matches, elo_snapshots, HOLDOUT)
    save_predictions_csv(rows, REPORTS / "common_sample_predictions.csv")

    # Comparison
    print(f"\n[3/6] Model comparison...", flush=True)
    comparison = run_comparison(rows)
    with open(REPORTS / "model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # Blend grid
    print(f"\n[4/6] Blend grid...", flush=True)
    blend_results = run_blend_grid(rows)
    with open(REPORTS / "blend_grid.json", "w") as f:
        json.dump(blend_results, f, indent=2)

    # Calibration
    print(f"\n[5/6] Calibration...", flush=True)
    cal_results = run_calibration(rows)
    with open(REPORTS / "calibration.json", "w") as f:
        json.dump(cal_results, f, indent=2)

    # Confirmation + subgroups
    print(f"\n[6/6] Confirmation and subgroups...", flush=True)
    conf_results = run_confirmation(rows)
    sg_results = run_subgroup_analysis(rows)

    total_time = time.time() - t_start
    print(f"\n{'=' * 60}", flush=True)
    print(f"DONE in {total_time:.1f}s", flush=True)
    print(f"  Common sample: {comparison['n_common']} matches", flush=True)
    print(f"  Reports written to {REPORTS}/", flush=True)

    # Summary
    summary = {
        "runtime_seconds": round(total_time, 1),
        "n_common": comparison["n_common"],
        "n_total": comparison["n_total"],
        "comparison": comparison,
        "blend_best": blend_results["best"],
        "confirmation": conf_results,
        "subgroups": sg_results,
    }
    with open(REPORTS / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: reports/experiment_summary.json", flush=True)


if __name__ == "__main__":
    main()
