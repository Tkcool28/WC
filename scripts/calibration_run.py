"""Run pi-rating on all intl matches 2015-2024 and tabulate hit rate by confidence bucket.

This is the calibration check. If pi-rating says 70% and actual hit rate is 65%, the
model is well-calibrated. If actual is 50%, the model is overconfident.
"""
import json
import time
from collections import defaultdict

from soccer_ev_model.pi_ratings import compute_pi_ratings
from soccer_ev_model.pi_backtest import load_matches
from soccer_ev_model.ev_workflow import _probs_from_ratings

OUT_LOG = "/tmp/calibration_run.log"


def main():
    with open('/root/soccer-model-lab/data/processed/international_matches.json') as f:
        intl = json.load(f)

    test_matches = [m for m in intl if '2015-01-01' <= m['date'] < '2025-01-01' and m.get('result')]
    print(f"Total intl matches 2015-2024: {len(test_matches)}", flush=True)

    all_wc = []
    for y in [2010, 2014, 2018, 2022]:
        all_wc.extend(load_matches(y))

    buckets = defaultdict(lambda: {'pred_sum': 0.0, 'actual': 0, 'n': 0})
    ratings_cache = {}
    processed = 0
    start = time.time()

    for m in test_matches:
        cache_key = m['date'][:7] + '-01'
        if cache_key not in ratings_cache:
            intl_train = [x for x in intl if x['date'] < cache_key]
            wc_train = [w for w in all_wc if w['date'][:4] < cache_key[:4]]
            train = intl_train + wc_train
            train.sort(key=lambda x: x['date'])
            ratings_cache[cache_key] = compute_pi_ratings(train, cutoff=cache_key, learning_rate=0.005)
        ratings = ratings_cache[cache_key]

        target = {
            'date': m['date'],
            'home_team_id': m['home_team_id'],
            'away_team_id': m['away_team_id'],
            'home_team': m['home_team'],
            'away_team': m['away_team'],
        }
        try:
            probs = _probs_from_ratings(target, ratings)
        except Exception:
            continue

        top_p = max(probs.values())
        top_m = max(probs, key=probs.get)
        actual = m['result']
        hit = 1 if top_m == actual else 0

        for lo, hi in [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7),
                       (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]:
            if lo <= top_p < hi:
                buckets[(lo, hi)]['pred_sum'] += top_p
                buckets[(lo, hi)]['actual'] += hit
                buckets[(lo, hi)]['n'] += 1
                break

        processed += 1
        if processed % 2000 == 0:
            print(f"  {processed} matches ({time.time() - start:.0f}s)", flush=True)

    print(f"Done: {processed} matches in {time.time() - start:.0f}s", flush=True)
    print()
    print("Hit rate by pi-rating confidence bucket (intl matches 2015-2024):")
    print(f"{'Bucket':<10} {'N':>6} {'Mean pred':>10} {'Hit rate':>10} {'Cal diff':>10}")
    print("-" * 50)
    for (lo, hi), v in sorted(buckets.items()):
        if v['n'] > 0:
            mean_pred = v['pred_sum'] / v['n']
            hit_rate = v['actual'] / v['n']
            diff = hit_rate - mean_pred
            marker = 'OK' if abs(diff) < 0.05 else ('WARN' if abs(diff) > 0.10 else 'meh')
            print(f"{lo:.1f}-{hi:.1f}    {v['n']:>6} {mean_pred:>9.1%} {hit_rate:>9.1%} {diff:>+9.1%}  {marker}")


if __name__ == "__main__":
    main()
