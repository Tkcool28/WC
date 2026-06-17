"""
Pi-rating backtest using international match data.

Trains pi-ratings on the full pool of international matches (1990-2018)
plus the 2010, 2014, and 2018 World Cups (re-stated for emphasis), and
then evaluates on:

  - 2022 World Cup (64 matches, all completed)
  - 2026 World Cup group stage (16 completed matches as of 2026-06-15)

The optimization is that we use the walk-forward batch path
(`pi_rating_match_probs_batch`) so the full backtest runs in <1s of
compute instead of the naive ~60s of repeated full-history retrains.

Reports:
  - RPS for pi-rating vs uniform baseline
  - Top-pick accuracy
  - Hit rate at >50%/>60%/>70% confidence thresholds
  - 5-bucket calibration table

Usage:
    python scripts/pi_backtest_with_intl.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Allow running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soccer_ev_model.ev_workflow import pi_rating_match_probs_batch
from soccer_ev_model.pi_backtest import rps


RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
INTL_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "international_matches.json"


def load_wc_openfootball(year: int) -> list[dict]:
    """Load a WC year from the openfootball cache, normalized to our schema.

    These files use the same team_id hashing as the intl file, so
    pi-ratings flow across sources without remapping.
    """
    raw_path = RAW_DIR / f"matches_{year}_openfootball.json"
    if not raw_path.exists():
        return []
    raw = json.loads(raw_path.read_text())["matches"]
    out = []
    for m in raw:
        if m.get("result") not in ("H", "D", "A"):
            continue
        if m.get("home_goals") is None or m.get("away_goals") is None:
            continue
        out.append({
            "match_id": f"{m['date'][:10]}_{m['home_team_id']}_{m['away_team_id']}",
            "date": m["date"][:10],
            "home_team": m["home_team_name"],
            "away_team": m["away_team_name"],
            "home_team_id": m["home_team_id"],
            "away_team_id": m["away_team_id"],
            "home_goals": m["home_goals"],
            "away_goals": m["away_goals"],
            "result": {"H": "home", "D": "draw", "A": "away"}[m["result"]],
        })
    return out


def load_intl() -> list[dict]:
    """Load all parsed international matches from the processed JSON."""
    return json.loads(INTL_PATH.read_text())


def build_training(intl: list[dict], wc_years: list[int], through_year: int) -> list[dict]:
    """Compose the training set: intl (1990..through_year) + WC years.

    The WC years are re-included even if they overlap with intl coverage
    because the intl data has looser date cuts and the WC years emphasize
    the highest-quality matches.
    """
    cutoff = f"{through_year + 1}-01-01"
    intl_train = [m for m in intl if "1990-01-01" <= m["date"] < cutoff]
    wc_train = []
    for y in wc_years:
        wc_train.extend(load_wc_openfootball(y))
    # Dedupe by (date, home_id, away_id) — intl + WC may overlap for WC matches
    seen = set()
    out = []
    for m in intl_train + wc_train:
        key = (m["date"], m["home_team_id"], m["away_team_id"])
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    out.sort(key=lambda m: m["date"])
    return out


def load_test(year: int) -> list[dict]:
    """Load a WC year as the test set (chronological order)."""
    matches = load_wc_openfootball(year)
    matches.sort(key=lambda m: m["date"])
    return matches


def evaluate(test_matches: list[dict], probs: list[dict]) -> dict:
    """Compute the standard set of backtest metrics for a test set."""
    outcomes = [m["result"] for m in test_matches]

    # RPS for pi-rating
    model_rps = rps(probs, outcomes)

    # Uniform baseline RPS (the floor we want to beat)
    uniform = [{"home": 1/3, "draw": 1/3, "away": 1/3} for _ in outcomes]
    uniform_rps = rps(uniform, outcomes)

    # Top-pick accuracy
    correct = sum(
        1 for p, o in zip(probs, outcomes) if max(p, key=p.get) == o
    )
    top_pick_acc = correct / len(outcomes) if outcomes else 0.0

    # Hit rate at confidence thresholds (top-p market is the chosen one,
    # and the prediction "hit" if that market actually won)
    def hit_rate(thresh: float) -> tuple[float, int]:
        n = 0
        hits = 0
        for p, o in zip(probs, outcomes):
            top_m = max(p, key=p.get)
            top_p = p[top_m]
            if top_p > thresh:
                n += 1
                if top_m == o:
                    hits += 1
        return (hits / n if n else 0.0, n)

    hr_50 = hit_rate(0.50)
    hr_60 = hit_rate(0.60)
    hr_70 = hit_rate(0.70)

    # 5-bucket calibration
    buckets = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.01)]
    calib = {b: {"pred": [], "actual": []} for b in buckets}
    for p, o in zip(probs, outcomes):
        top_p = max(p.values())
        top_m = max(p, key=p.get)
        for lo, hi in buckets:
            if lo <= top_p < hi:
                calib[(lo, hi)]["pred"].append(top_p)
                calib[(lo, hi)]["actual"].append(1.0 if top_m == o else 0.0)
                break

    calib_summary = []
    for (lo, hi), v in calib.items():
        if v["pred"]:
            mean_pred = sum(v["pred"]) / len(v["pred"])
            mean_actual = sum(v["actual"]) / len(v["actual"])
            calib_summary.append({
                "bucket": f"{lo:.1f}-{hi:.1f}",
                "n": len(v["pred"]),
                "pred": mean_pred,
                "actual": mean_actual,
                "diff": mean_actual - mean_pred,
            })

    return {
        "n_matches": len(outcomes),
        "model_rps": model_rps,
        "uniform_rps": uniform_rps,
        "rps_improvement": uniform_rps - model_rps,
        "top_pick_acc": top_pick_acc,
        "hr_50": hr_50,
        "hr_60": hr_60,
        "hr_70": hr_70,
        "calibration": calib_summary,
    }


def format_report(label: str, r: dict) -> str:
    """Format an evaluation result block for human reading."""
    lines = [f"\n{label}", "-" * 60]
    lines.append(f"  Matches: {r['n_matches']}")
    lines.append(f"  RPS (pi-rating): {r['model_rps']:.4f}")
    lines.append(f"  RPS (uniform):   {r['uniform_rps']:.4f}")
    lines.append(f"  RPS improvement: {r['rps_improvement']:+.4f} "
                 f"({'better' if r['rps_improvement'] > 0 else 'WORSE'})")
    lines.append(f"  Top-pick accuracy: {r['top_pick_acc']:.1%}")
    h50, n50 = r["hr_50"]
    h60, n60 = r["hr_60"]
    h70, n70 = r["hr_70"]
    lines.append(f"  Hit rate @ >50% conf: {h50:.1%} (n={n50})")
    lines.append(f"  Hit rate @ >60% conf: {h60:.1%} (n={n60})")
    lines.append(f"  Hit rate @ >70% conf: {h70:.1%} (n={n70})")
    lines.append("  Calibration:")
    for c in r["calibration"]:
        if c["n"] > 0:
            tag = "ok" if abs(c["diff"]) < 0.1 else "X "
            lines.append(
                f"    {c['bucket']}  n={c['n']:3d}  "
                f"pred={c['pred']:.2f}  actual={c['actual']:.2f}  "
                f"diff={c['diff']:+.2f} {tag}"
            )
    return "\n".join(lines)


def main() -> int:
    intl = load_intl()
    print(f"Loaded {len(intl)} international matches (1990-2026)")

    # ---- 2022 backtest: train 1990-2018 intl + 2010+2014+2018 WC ----
    train_2022 = build_training(intl, wc_years=[2010, 2014, 2018], through_year=2018)
    test_2022 = load_test(2022)
    print(f"Training set: {len(train_2022)} matches")
    print(f"2022 test set: {len(test_2022)} matches")

    t0 = time.time()
    probs_2022 = pi_rating_match_probs_batch(train_2022, test_2022)
    elapsed_2022 = time.time() - t0
    print(f"  pi_rating_match_probs_batch for 64 matches on "
          f"{len(train_2022)} training: {elapsed_2022:.2f}s")
    res_2022 = evaluate(test_2022, probs_2022)

    # ---- 2026 backtest: train 1990-2022 intl + 2010+2014+2018+2022 WC ----
    train_2026 = build_training(intl, wc_years=[2010, 2014, 2018, 2022], through_year=2022)
    test_2026 = load_test(2026)
    # Some 2026 openfootball rows may be missing results for unplayed games;
    # filter to played ones for the test set.
    test_2026 = [m for m in test_2026 if m.get("result") in ("home", "draw", "away")]
    print(f"\n2026 training set: {len(train_2026)} matches")
    print(f"2026 test set: {len(test_2026)} matches (played only)")

    t0 = time.time()
    probs_2026 = pi_rating_match_probs_batch(train_2026, test_2026)
    elapsed_2026 = time.time() - t0
    print(f"  pi_rating_match_probs_batch for {len(test_2026)} matches on "
          f"{len(train_2026)} training: {elapsed_2026:.2f}s")
    res_2026 = evaluate(test_2026, probs_2026)

    # ---- Report ----
    print("\n" + "=" * 70)
    print("PI-RATING BACKTEST (international data, walk-forward)")
    print("=" * 70)
    print(format_report(
        "2022 WC  (train: 1990-2018 intl + 2010+2014+2018 WC)",
        res_2022,
    ))
    print(format_report(
        "2026 WC  (train: 1990-2022 intl + 2010+2014+2018+2022 WC)",
        res_2026,
    ))

    # ---- Comparison vs the old (WC-only) backtest ----
    # Old reference: 2022 with WC-only training was +0.028 RPS improvement.
    print("\n" + "=" * 70)
    print("COMPARISON vs OLD (WC-only training)")
    print("=" * 70)
    old_2022_imp = 0.028
    new_2022_imp = res_2022["rps_improvement"]
    print(f"  2022 RPS improvement: old={old_2022_imp:+.4f}, "
          f"new={new_2022_imp:+.4f} "
          f"({'better' if new_2022_imp > old_2022_imp else 'worse' if new_2022_imp < old_2022_imp else 'same'})")
    print(f"  2022 top-pick:  old=60% (WC-only), new={res_2022['top_pick_acc']:.1%} (intl + WC)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
