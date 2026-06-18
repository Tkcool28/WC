"""Run chronological backtests for the independent goal model.

Usage:
    python3 scripts/run_backtest.py
    python3 scripts/run_backtest.py --output-dir reports
    python3 scripts/run_backtest.py --models global_poisson regularized_team
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from soccer_ev_model.goal_model_backtest import (
    DEFAULT_HOLDOUTS,
    BacktestResult,
    run_all_backtests,
    write_backtest_report,
)
from soccer_ev_model.goal_model_data import build_goal_matches, load_raw_matches


def main() -> None:
    parser = argparse.ArgumentParser(description="Run goal model backtests")
    parser.add_argument("--output-dir", default="reports", help="Output directory for reports")
    parser.add_argument("--models", nargs="+", default=["global_poisson", "regularized_team"],
                       help="Models to test")
    args = parser.parse_args()

    print("Loading matches...")
    raw = load_raw_matches()
    matches, excluded = build_goal_matches(raw)
    print(f"  {len(matches)} usable matches, {sum(excluded.values())} excluded")

    print(f"\nRunning backtests: {args.models}")
    print(f"Holdouts: {[h.name for h in DEFAULT_HOLDOUTS]}")
    print()

    results = run_all_backtests(matches, holdouts=DEFAULT_HOLDOUTS, models=args.models)

    # Print summary
    for r in results:
        m = r.metrics
        print(f"{r.model_name:25s} | {r.holdout_name:15s} | N={m.n_matches:4d} | "
              f"LogL={m.log_loss:.4f} | RPS={m.ranked_probability_score:.4f} | "
              f"Brier={m.brier_score:.4f} | Top1={m.top_pick_accuracy:.3f} | "
              f"fit={r.fit_time_seconds:.1f}s pred={r.predict_time_seconds:.1f}s")

    # Write reports
    json_path = write_backtest_report(results, args.output_dir)
    md_path = Path(args.output_dir) / "backtest_results.md"
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
