"""Run walk-forward backtest on REAL World Cup + international data.

Two flavors of backtest:
  1. WC-only walk-forward (the original): train on WC data, predict WC data.
     Small training set (192-256 matches), CatBoost tends to overfit.
  2. International + WC walk-forward: train on the full 32k+ international
     dataset (with Elo features) + WC years, predict on a held-out window.
     This is the comparison that shows whether trained models beat the
     pi-rating baseline.

Both flavors report naive_uniform, logreg, catboost RPS, and a per-window
breakdown. RPS is the metric to watch (lower is better; 0.167 ≈ uniform).

Usage:
    python scripts/walk_forward.py
    python scripts/walk_forward.py --wc-only     # skip the intl+WC flavor
    python scripts/walk_forward.py --no-elo      # disable Elo features
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soccer_ev_model.backtest import compare_to_naive  # noqa: E402
from soccer_ev_model.elo_ratings import load_elo_ratings  # noqa: E402
from soccer_ev_model.features import build_feature_matrix  # noqa: E402
from soccer_ev_model.train import report_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
ELO_PATH = RAW / "elo_ratings.json"


def load_openfootball_year(year: int) -> list[dict]:
    p = RAW / f"matches_{year}_openfootball.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())["matches"]


def load_international() -> list[dict]:
    p = PROCESSED / "international_matches.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wc-only", action="store_true",
                        help="Skip the international+WC walk-forward")
    parser.add_argument("--no-elo", action="store_true",
                        help="Disable Elo features (use 1500 default for all)")
    args = parser.parse_args()

    print("=" * 75)
    print("WC MODEL WALK-FORWARD BACKTEST")
    print("=" * 75)
    print()

    # ---- Load ----
    wc_matches = []
    for y in (2010, 2014, 2018, 2022, 2026):
        wc_matches.extend(load_openfootball_year(y))
    intl_matches = load_international()
    elo_snapshots = None if args.no_elo else (load_elo_ratings(ELO_PATH) if ELO_PATH.exists() else None)
    print(f"Loaded {len(wc_matches)} WC matches (2010-2026)")
    print(f"Loaded {len(intl_matches)} international matches (1990-2026)")
    if elo_snapshots is not None:
        print(f"Loaded {len(elo_snapshots)} teams of Elo data")
    else:
        print("Elo: disabled (no cache or --no-elo)")
    print()

    # ====================================================================== #
    # Backtest A: WC-only walk-forward (the original test)
    # ====================================================================== #
    print("=" * 75)
    print("Backtest A: WC-only, train through 2018, predict 2022 (n=64)")
    print("=" * 75)
    res = compare_to_naive(
        wc_matches,
        train_end_date="2018-12-31",
        model_types=("logreg", "catboost"),
    )
    for name, m in res.items():
        print(f"  {report_metrics(m, label=name)}")
    print()

    # ====================================================================== #
    # Backtest B: intl+WC walk-forward — the meaningful comparison
    # ====================================================================== #
    if not args.wc_only:
        print("=" * 75)
        print("Backtest B: intl+WC, train through 2018, predict 2022 (n=64)")
        print("=" * 75)
        # Re-define the comparison for B because compare_to_naive calls
        # build_feature_matrix without the elo_snapshots kwarg. We
        # monkey-patch by importing the module's build_feature_matrix and
        # wrapping it.
        from soccer_ev_model import backtest as _bt

        _orig_build = _bt.build_feature_matrix
        if elo_snapshots is not None:
            def _build_with_elo(matches):
                return _orig_build(matches, elo_snapshots=elo_snapshots)
            _bt.build_feature_matrix = _build_with_elo
        try:
            res_b = _bt.compare_to_naive(
                intl_matches + wc_matches,
                train_end_date="2018-12-31",
                model_types=("logreg", "catboost"),
            )
        finally:
            _bt.build_feature_matrix = _orig_build
        for name, m in res_b.items():
            print(f"  {report_metrics(m, label=name)}")
        print()

    # ====================================================================== #
    # Interpretation
    # ====================================================================== #
    print("=" * 75)
    print("INTERPRETATION")
    print("=" * 75)
    print()
    print("RPS is the metric to focus on (lower = better):")
    print("  - naive_uniform  ≈ 0.167  (random guessing)")
    print("  - trained model < 0.167  has signal")
    print("  - trained model ≈ 0.167  has no signal")
    print("  - trained model > 0.167  is actively wrong (overfit)")
    print()
    print("Compare Backtest A vs Backtest B at the same cutoff (2018→2022):")
    print("  - If B is much lower than A, the extra intl data + Elo features help.")
    print("  - If they're similar, the trained model isn't using the data well,")
    print("    and pi-rating alone is the better bet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
