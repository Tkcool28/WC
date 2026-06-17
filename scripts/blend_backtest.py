"""Walk-forward backtest comparing pi-only vs pi+Elo blend on WC data.

Same data window as scripts/walk_forward.py:
  - Train: WC matches before 2018-12-31 (2010 + 2014 + 2018)
  - Test:  WC matches 2018-12-31 through 2022-12-31 (2018 + 2022)

For each test match we compute (a) the pi-rating snapshot and (b) the
Elo values for both teams, both using strict-less-than date filtering
(leak-safe). We then evaluate three hand-tuned blend candidates:

  - pi_only   : w_pi=1.0, w_elo=0.0   (current behavior, baseline)
  - equal     : w_pi=0.5, w_elo=0.5   (treat both signals equally)
  - pi_heavy  : w_pi=0.7, w_elo=0.3   (trust pi more; it has 32k matches
                                        of context, Elo is per-team with
                                        no margin-of-victory weighting)

The score is RPS (ranked probability score) using the existing
`evaluate` from `soccer_ev_model.train`. Lower is better; 0.167 is
naive uniform.

Output:
  - RPS table with all 3 candidates + naive_uniform
  - Per-candidate verdict (better than pi_only? by how much?)

We do NOT touch the dashboard here. The parent decides wiring after
seeing these numbers.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soccer_ev_model.elo_ratings import elo_at, load_elo_ratings  # noqa: E402
from soccer_ev_model.ev_workflow import _probs_from_ratings_blend  # noqa: E402
from soccer_ev_model.pi_ratings import (  # noqa: E402
    _parse_date,
    compute_pi_ratings_walk_forward,
)
from soccer_ev_model.train import CLASS_LABELS, evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
ELO_PATH = RAW / "elo_ratings.json"

TRAIN_END = "2018-12-31"
TEST_END = "2022-12-31"

# The 3 blend candidates. Hand-tuned, no ML, no grid search.
CANDIDATES = [
    ("pi_only", 1.0, 0.0),    # baseline
    ("equal", 0.5, 0.5),      # 50/50
    ("pi_heavy", 0.7, 0.3),   # trust pi more
]


def load_openfootball_year(year: int) -> list[dict]:
    p = RAW / f"matches_{year}_openfootball.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())["matches"]


def split_train_test() -> tuple[list[dict], list[dict]]:
    """Train: matches before 2018-12-31. Test: 2018-12-31 → 2022-12-31.

    The 2018 WC (June 2018) ends up in the train set because every 2018
    match date is strictly less than the cutoff. The test set is
    therefore just the 2022 WC (n=64). This mirrors the existing
    walk_forward.py behaviour: same cutoff, same comparison.
    """
    train, test = [], []
    for y in (2010, 2014, 2018, 2022):
        for m in load_openfootball_year(y):
            d = _parse_date(m["date"])
            if d <= _parse_date(TRAIN_END):
                train.append(m)
            elif d <= _parse_date(TEST_END):
                test.append(m)
    train.sort(key=lambda m: m["date"])
    test.sort(key=lambda m: m["date"])
    return train, test


def build_elo_table(test_matches: list[dict], elo_snapshots: dict) -> list[tuple[int, bool]]:
    """Pre-compute Elo lookups for all (home, away) of test matches.

    Returns: list of (home_elo, away_elo, home_missing, away_missing) per
    test match, in the same order as test_matches. We use the test match's
    date as the lookup cutoff, which makes the lookup leak-safe: a
    team can never see its own result from the same match.
    """
    out = []
    for m in test_matches:
        h_elo, h_miss = elo_at(elo_snapshots, m["home_team_name"], m["date"])
        a_elo, a_miss = elo_at(elo_snapshots, m["away_team_name"], m["date"])
        out.append((h_elo, a_elo, h_miss, a_miss))
    return out


def main() -> int:
    print("=" * 75)
    print("PI + ELO BLEND WALK-FORWARD BACKTEST")
    print("=" * 75)
    print()

    # ---- Load ----
    # Note: with the spec's TRAIN_END=2018-12-31, the 2018 WC (June 2018)
    # falls in the train set (matches < "2018-12-31" → train). The test
    # set is therefore just the 2022 WC (n=64). This matches the existing
    # scripts/walk_forward.py behaviour exactly: same cutoff, same split
    # function, same n. The spec's "n=80" was an off-by-WC estimate.
    train, test = split_train_test()
    elo_snapshots = load_elo_ratings(ELO_PATH) if ELO_PATH.exists() else None
    if elo_snapshots is None:
        print("ERROR: no Elo cache at", ELO_PATH)
        return 1

    print(f"Train: {len(train)} WC matches before {TRAIN_END} (2010/2014/2018)")
    print(f"Test:  {len(test)} WC matches {TRAIN_END} → {TEST_END} (2022 WC)")
    print(f"Elo:   {len(elo_snapshots)} teams in cache")
    print()

    # ---- Pre-compute pi-rating snapshots for every test match (walk-forward) ----
    # compute_pi_ratings_walk_forward is leak-safe: each test match's
    # snapshot uses only matches with strictly earlier dates.
    snapshots = compute_pi_ratings_walk_forward(
        train, test, consume_test_results=True,
    )

    # ---- Pre-compute Elo for every test match ----
    elo_table = build_elo_table(test, elo_snapshots)
    n_missing_elo = sum(1 for h, a, hm, am in elo_table if hm or am)
    print(f"Elo coverage: {len(test) - n_missing_elo}/{len(test)} test matches have Elo for BOTH teams")
    print(f"  ({n_missing_elo} test matches use the 1500 default for at least one team)")
    print()

    # ---- Build the per-candidate RPS table ----
    y_true = np.array([m["result"] for m in test])

    print("=" * 75)
    print("RESULTS")
    print("=" * 75)
    print()
    header = f"{'candidate':<14s} {'w_pi':>5s} {'w_elo':>5s}   {'rps':>7s}   {'acc':>6s}   {'log_loss':>9s}   {'n':>4s}"
    print(header)
    print("-" * len(header))

    results = {}

    # Naive uniform baseline first (1/3, 1/3, 1/3)
    n = len(test)
    naive_probs = np.tile([1/3, 1/3, 1/3], (n, 1))
    naive_m = evaluate(pd.Series(y_true), naive_probs)
    results["naive_uniform"] = naive_m
    print(f"{'naive_uniform':<14s} {'-':>5s} {'-':>5s}   {naive_m['rps']:7.4f}   {naive_m['accuracy']:6.3f}   {naive_m['log_loss']:9.4f}   {naive_m['n']:4d}")

    for name, wpi, welo in CANDIDATES:
        probs_rows = []
        for (m, ratings), (h_elo, a_elo, _h_miss, _a_miss) in zip(snapshots, elo_table):
            p = _probs_from_ratings_blend(
                m, ratings,
                home_elo=h_elo, away_elo=a_elo,
                w_pi=wpi, w_elo=welo,
            )
            # In (H, D, A) order matching CLASS_LABELS = ('H', 'D', 'A')
            probs_rows.append([p["home"], p["draw"], p["away"]])
        probs_arr = np.array(probs_rows)
        m = evaluate(pd.Series(y_true), probs_arr)
        results[name] = m
        print(f"{name:<14s} {wpi:5.2f} {welo:5.2f}   {m['rps']:7.4f}   {m['accuracy']:6.3f}   {m['log_loss']:9.4f}   {m['n']:4d}")

    print()
    print("=" * 75)
    print("VERDICTS")
    print("=" * 75)
    print()
    base_rps = results["pi_only"]["rps"]
    for name, _wpi, _welo in CANDIDATES:
        r = results[name]["rps"]
        delta = base_rps - r  # positive = better than pi_only
        if name == "pi_only":
            print(f"  {name:<12s}  baseline (rps={r:.4f})")
        else:
            sign = "better" if delta > 0 else "worse" if delta < 0 else "tied"
            print(f"  {name:<12s}  rps={r:.4f}  delta_vs_pi_only={delta:+.4f}  ({sign})")
    print()
    print(f"  naive_uniform  rps={results['naive_uniform']['rps']:.4f}  (random guess baseline)")
    print()
    print("Lower RPS is better. 0.167 = uniform. Anything below is signal.")
    return 0


# pandas is only needed for the evaluate() signature; import lazily so the
# script doesn't fail at import time if pandas is in a weird state.
import pandas as pd  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
