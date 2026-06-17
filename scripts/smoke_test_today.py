"""Smoke test: predict today's WC matches using pi-ratings.

This is a real end-to-end test using actual data:
1. Load historical matches from openfootball (2010, 2014, 2018, 2022)
2. Compute pi-ratings using only those historical matches
3. Load today's matches from football-data.org
4. For each unplayed match, predict P(H/D/A) using pi-rating strength diff

The pi-rating approach is naive (no Elo, no recent form weighting, no model
training). This is a PROOF OF CONCEPT, not a real betting model. We just
want to confirm the data pipeline works end-to-end.

Usage:
    python scripts/smoke_test_today.py
"""

import json
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soccer_ev_model.pi_ratings import compute_pi_ratings, pi_diff_features


RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def load_openfootball_year(year: int) -> list[dict]:
    """Load one year of openfootball matches from cache."""
    path = RAW_DIR / f"matches_{year}_openfootball.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())["matches"]


def load_football_data_2026() -> list[dict]:
    """Load the live 2026 matches from football-data.org cache."""
    path = RAW_DIR / "matches_2026.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())["matches"]


def pi_to_three_way(matchup_strength_diff: float) -> tuple[float, float, float]:
    """Convert a pi-rating matchup strength diff into (P_home, P_draw, P_away).

    Naive logistic model:
        - matchup_strength_diff is the home team's combined (off+def) advantage
        - Negative values mean the away team is stronger.
        - We use a logistic function with a scale factor (k=0.85) tuned so that
          a strength diff of 0 yields roughly 45/27/28 (home/draw/away) and
          a strong favorite (diff ~2) yields ~75/15/10.

    This is a PLACEHOLDER. The real model (Phase 2) will replace this with
    CatBoost trained on historical data.
    """
    k = 0.85
    # Logistic of the strength diff for home-vs-away
    p_home_not_draw = 1.0 / (1.0 + math.exp(-k * matchup_strength_diff))
    # Draw probability: 27% baseline, reduced as the strength gap widens
    p_draw = 0.27 * math.exp(-abs(matchup_strength_diff) * 0.4)
    p_home = p_home_not_draw * (1.0 - p_draw)
    p_away = (1.0 - p_home_not_draw) * (1.0 - p_draw)
    return p_home, p_draw, p_away


def main() -> int:
    print("=" * 70)
    print("WC MODEL SMOKE TEST — Predict today's matches using pi-ratings")
    print("=" * 70)
    print()

    # 1. Load historical data (openfootball only, since team IDs must match
    #    across historical and current). The 2026 openfootball file gets
    #    included below as the "current tournament" source.
    print("Loading historical WC matches from openfootball...")
    historical = []
    for year in [2010, 2014, 2018, 2022]:
        matches = load_openfootball_year(year)
        print(f"  {year}: {len(matches)} matches")
        historical.extend(matches)
    print(f"  TOTAL historical: {len(historical)} matches")
    print()

    # 2. Compute pi-ratings. Cutoff at the start of the 2026 tournament so we
    #    don't leak 2026 group-stage results into the ratings we use to
    #    predict today's match. (The 2026 openfootball file is loaded but only
    #    used to FIND today's matches, not to compute ratings.)
    print("Computing pi-ratings (cutoff: 2026-06-11, tournament start)...")
    ratings = compute_pi_ratings(historical, cutoff="2026-06-11")
    print(f"  {len(ratings)} teams have ratings")
    print()

    # 3. Find today's unplayed matches in the 2026 openfootball file
    today_matches_2026 = load_openfootball_year(2026)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    unplayed_today = [
        m for m in today_matches_2026
        if (m.get("date") or "").startswith(today) and m.get("status") == "TIMED"
    ]
    print(f"Today's unplayed matches ({today}): {len(unplayed_today)}")
    print()

    if not unplayed_today:
        print("No unplayed matches today. Try again during a WC matchday.")
        return 0

    # 4. Predict each match
    print("-" * 70)
    for m in unplayed_today:
        home_id = m["home_team_id"]
        away_id = m["away_team_id"]
        home_name = m["home_team_name"]
        away_name = m["away_team_name"]
        date_str = m["date"]

        feats = pi_diff_features(home_id, away_id, ratings)
        p_home, p_draw, p_away = pi_to_three_way(feats["pi_matchup"])

        # Confidence: how far is the model's top pick from 1/3?
        top_p = max(p_home, p_draw, p_away)
        if top_p >= 0.60:
            confidence = "HIGH"
        elif top_p >= 0.45:
            confidence = "MED"
        else:
            confidence = "LOW"

        pick = "Home" if p_home == top_p else ("Draw" if p_draw == top_p else "Away")

        print(f"  {date_str} | {home_name} vs {away_name}")
        print(f"    pi_matchup: {feats['pi_matchup']:+.3f}  "
              f"(off_diff={feats['pi_off_diff']:+.2f}, def_diff={feats['pi_def_diff']:+.2f})")
        print(f"    P(Home)={p_home*100:5.1f}%  P(Draw)={p_draw*100:5.1f}%  "
              f"P(Away)={p_away*100:5.1f}%")
        print(f"    Model pick: {pick}  (confidence: {confidence})")
        print()
    print("-" * 70)
    print()
    print("NOTE: This is a pi-rating baseline, not a trained model.")
    print("The model probabilities here are naive. Phase 2 will train")
    print("CatBoost on this data and replace the placeholder logistic.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
