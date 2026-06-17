"""
Backtest the pure pi-rating workflow on historical WC data.

For each tournament, train pi-ratings on prior tournaments,
predict each match in the target tournament, then check:
1. Calibration: when we say X%, does X% actually hit?
2. RPS vs uniform
3. Hit rate on top-probability picks
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .ev_workflow import pi_rating_match_probs
from .no_vig import remove_vig

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def load_matches(year: int) -> list[dict]:
    """Load matches for a given WC year, normalising to our schema."""
    for fname in (f"matches_{year}_openfootball.json", f"matches_{year}.json"):
        path = DATA_DIR / fname
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        raw_matches = data.get("matches", [])
        if not raw_matches:
            continue
        # Detect schema: international-pulled format has flat fields
        first = raw_matches[0]
        if "home_team_name" in first:
            return [_normalise_intl(m) for m in raw_matches if m.get("result") in ("H", "D", "A")]
        else:
            return [_normalise_openfootball(m) for m in raw_matches if _has_result(m)]
    return []


def _normalise_openfootball(m: dict) -> dict:
    score = m.get("score", {}).get("ft", [0, 0])
    return {
        "match_id": m.get("key", f"{m.get('date')}_{m.get('team1_key')}_{m.get('team2_key')}"),
        "date": m.get("date", "")[:10],
        "home_team": m.get("team1", {}).get("name", "?"),
        "away_team": m.get("team2", {}).get("name", "?"),
        "home_team_id": m.get("team1_key", ""),
        "away_team_id": m.get("team2_key", ""),
        "home_goals": score[0],
        "away_goals": score[1],
        "result": m.get("winner"),
    }


def _normalise_intl(m: dict) -> dict:
    return {
        "match_id": f"{m.get('date','')}_{m.get('home_team_id')}_{m.get('away_team_id')}",
        "date": m.get("date", "")[:10],
        "home_team": m.get("home_team_name", "?"),
        "away_team": m.get("away_team_name", "?"),
        "home_team_id": m.get("home_team_id"),
        "away_team_id": m.get("away_team_id"),
        "home_goals": m.get("home_goals", 0),
        "away_goals": m.get("away_goals", 0),
        "result": {"H": "home", "D": "draw", "A": "away"}.get(m.get("result")),
    }


def _has_result(m: dict) -> bool:
    if m.get("winner") not in ("home", "away", "draw"):
        return False
    score = m.get("score", {}).get("ft", [0, 0])
    return score[0] is not None and score[1] is not None


def rps(probs: list[dict], outcomes: list[str]) -> float:
    """Ranked Probability Score for multi-class predictions."""
    total = 0.0
    n = len(probs)
    markets = ("home", "draw", "away")
    for p, o in zip(probs, outcomes):
        cum_pred = 0.0
        cum_actual = 0.0
        for m in markets:
            cum_pred += p[m]
            cum_actual += (1.0 if o == m else 0.0)
            total += (cum_pred - cum_actual) ** 2
    return total / n


def backtest_tournament(train_years: list[int], test_year: int) -> dict:
    """Train on prior years, predict test_year matches."""
    train_matches = []
    for y in train_years:
        train_matches.extend(load_matches(y))

    test_matches = load_matches(test_year)
    if not test_matches:
        return {"error": f"no matches for {test_year}"}

    probs = []
    outcomes = []
    for m in test_matches:
        try:
            p = pi_rating_match_probs(train_matches, m)
            probs.append(p)
            outcomes.append(m["result"])
        except Exception as e:
            print(f"  skip {m.get('home_team')} vs {m.get('away_team')}: {e}", file=sys.stderr)
            continue
    correct = 0
    for p, o in zip(probs, outcomes):
        pick = max(p, key=p.get)
        if pick == o:
            correct += 1
    top_pick_acc = correct / len(probs)

    # RPS
    model_rps = rps(probs, outcomes)

    # Uniform baseline RPS
    uniform = [{"home": 1/3, "draw": 1/3, "away": 1/3} for _ in outcomes]
    uniform_rps = rps(uniform, outcomes)

    # Calibration: bucket predictions, check actual hit rate
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
                "mean_pred": round(mean_pred, 3),
                "mean_actual": round(mean_actual, 3),
                "diff": round(mean_actual - mean_pred, 3),
            })

    return {
        "test_year": test_year,
        "n_matches": len(probs),
        "top_pick_acc": round(top_pick_acc, 3),
        "model_rps": round(model_rps, 4),
        "uniform_rps": round(uniform_rps, 4),
        "rps_improvement": round(uniform_rps - model_rps, 4),
        "calibration": calib_summary,
    }


if __name__ == "__main__":
    # Walk-forward backtest
    print("=" * 60)
    print("PI-RATING BACKTEST (no model, just pi-rating + logistic)")
    print("=" * 60)

    configs = [
        ([2010, 2014], 2018, "Train 2010+2014 -> Test 2018"),
        ([2010, 2014, 2018], 2022, "Train 2010+2014+2018 -> Test 2022"),
        ([2010, 2014, 2018, 2022], 2026, "Train 2010-2022 -> Test 2026 (group stage done so far)"),
    ]

    for train, test, label in configs:
        print(f"\n{label}")
        print("-" * 60)
        result = backtest_tournament(train, test)
        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue
        print(f"  Matches: {result['n_matches']}")
        print(f"  Top-pick accuracy: {result['top_pick_acc']:.1%}")
        print(f"  RPS (pi-rating): {result['model_rps']:.4f}")
        print(f"  RPS (uniform):   {result['uniform_rps']:.4f}")
        print(f"  RPS improvement: {result['rps_improvement']:+.4f} ({'better' if result['rps_improvement'] > 0 else 'WORSE'})")
        print(f"  Calibration:")
        for c in result["calibration"]:
            if c["n"] > 0:
                tag = "✓" if abs(c["diff"]) < 0.1 else "✗"
                print(f"    {c['bucket']}  n={c['n']:3d}  pred={c['mean_pred']:.2f}  actual={c['mean_actual']:.2f}  diff={c['diff']:+.2f} {tag}")
