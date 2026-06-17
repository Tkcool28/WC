"""Run evaluate_match on the 3 known 2026 WC group-stage matches.

These are real upcoming matches from data/raw/matches_2026.json:
  - Iraq vs Norway         (2026-06-16)
  - Argentina vs Algeria   (2026-06-17)
  - Austria vs Jordan      (2026-06-17)

For each, we compute pi-ratings from the intl + WC history (1990-2022),
then compare to representative book odds (American) for a +EV analysis.

The book odds below are representative of typical WC opener lines for each
matchup. They are NOT pulled from a live feed (per the safety rules).
"""
import json
from collections import defaultdict

from soccer_ev_model.confidence import render_warning_banner
from soccer_ev_model.ev_workflow import evaluate_match
from soccer_ev_model.pi_ratings import compute_pi_ratings
from soccer_ev_model.pi_backtest import load_matches


# ---- book odds (representative, for +EV pipeline demo) ----
# Iraq vs Norway: roughly even matchup, slight favorite either side
# Argentina vs Algeria: heavy favorite Argentina
# Austria vs Jordan: moderate favorite Austria
BOOK_ODDS = {
    "iraq_norway":        {"home": +240, "draw": +220, "away": +120},  # Norway fav
    "argentina_algeria":  {"home": -450, "draw": +550, "away": +1500}, # Arg HEAVY fav
    "austria_jordan":     {"home": -200, "draw": +320, "away": +550},  # Austria fav
}


def main():
    # Load the training corpus
    with open('/root/soccer-model-lab/data/processed/international_matches.json') as f:
        intl = json.load(f)

    # Add WC matches for richer ratings
    all_wc = []
    for y in [2010, 2014, 2018, 2022]:
        all_wc.extend(load_matches(y))

    # Map team names -> intl IDs (consistent across intl + wc)
    name_to_id = {}
    for m in intl:
        name_to_id.setdefault(m["home_team"], m["home_team_id"])
        name_to_id.setdefault(m["away_team"], m["away_team_id"])

    # Cutoff: today is June 16, 2026 (per the system date)
    # Use 2026-06-16 as the cutoff (matches ON the cutoff are INCLUDED)
    cutoff = "2026-06-16"
    intl_train = [x for x in intl if x["date"] < cutoff]
    wc_train = [w for w in all_wc if w["date"][:4] < cutoff[:4]]
    train = intl_train + wc_train
    train.sort(key=lambda x: x["date"])
    print(f"Training set: {len(train)} matches (intl: {len(intl_train)}, WC: {len(wc_train)})")

    ratings = compute_pi_ratings(train, cutoff=cutoff)
    print(f"Ratings computed for {len(ratings)} teams\n")

    # ---- 3 matches ----
    matches = [
        ("iraq_norway", "Iraq", "Norway", "2026-06-16", BOOK_ODDS["iraq_norway"]),
        ("argentina_algeria", "Argentina", "Algeria", "2026-06-17", BOOK_ODDS["argentina_algeria"]),
        ("austria_jordan", "Austria", "Jordan", "2026-06-17", BOOK_ODDS["austria_jordan"]),
    ]

    results = []
    for slug, home_name, away_name, date, odds in matches:
        h_id = name_to_id.get(home_name)
        a_id = name_to_id.get(away_name)
        if h_id is None or a_id is None:
            print(f"!! Could not find ids for {home_name} / {away_name}")
            continue

        result = evaluate_match(
            home_team=home_name,
            away_team=away_name,
            home_team_id=h_id,
            away_team_id=a_id,
            date=date,
            book_home_odds=odds["home"],
            book_draw_odds=odds["draw"],
            book_away_odds=odds["away"],
            ratings=ratings,
        )
        results.append((slug, result))

        print("=" * 60)
        print(f"  {home_name} vs {away_name}  ({date})")
        print("=" * 60)
        c = result["confidence"]
        print(f"  Tier: {c['tier']}  ({c['tier_description']})")
        print(f"  Data: home={c['home_matches_played']}, away={c['away_matches_played']} matches")
        print(f"  Calibration: top_p={c['top_p']:.3f} -> calibrated {c['calibrated_p']:.3f} "
              f"(diff={c['calibration_diff']:+.3f}, label={c['calib_label']})")
        if c["warnings"]:
            print(f"  Warnings:")
            for w in c["warnings"]:
                print(f"    - {w}")
        bf = result["book_fair"]
        pp = result["pi_probs"]
        cp = result["calibrated_pi"]
        ed = result["edges"]
        print(f"  Book fair:    H={bf['home']:.1%}  D={bf['draw']:.1%}  A={bf['away']:.1%}")
        print(f"  Pi (raw):     H={pp['home']:.1%}  D={pp['draw']:.1%}  A={pp['away']:.1%}")
        print(f"  Pi (calib):   H={cp['home']:.1%}  D={cp['draw']:.1%}  A={cp['away']:.1%}")
        print(f"  Edges:        H={ed['home']:+.3f}  D={ed['draw']:+.3f}  A={ed['away']:+.3f}")
        if result["plus_ev_flags"]:
            print(f"  +EV flags:")
            for f in result["plus_ev_flags"]:
                print(f"    - {f['market']}: edge={f['edge']:+.3f}")
        else:
            print(f"  +EV flags:    (none at min_edge=0.03)")
        print()
        print("  Banner:")
        for line in result["banner"].split("\n"):
            print(f"    {line}")
        print()

    # ---- summary stats ----
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    tier_counts = defaultdict(int)
    for slug, r in results:
        tier_counts[r["confidence"]["tier"]] += 1
    for t in "ABCD":
        n = tier_counts.get(t, 0)
        print(f"  Tier {t}: {n} match{'es' if n != 1 else ''}")
    print()

    # ---- by-tier expected hit rate (from calibration backtest) ----
    # Per CALIBRATION_TABLE / 9,678-match backtest
    # Tier A ~ top_p <= 0.60 (well-calibrated buckets):  ~50% hit rate
    # Tier C/D: overconfident or insufficient data      : signal unreliable
    print("Expected hit rates by tier (from 9,678-match calibration):")
    print("  Tier A: well-calibrated, 50-53% hit rate (model is honest)")
    print("  Tier B: decent, ~46-56% hit rate (moderate caveats)")
    print("  Tier C: overconfident pi or limited data, signal is unreliable")
    print("  Tier D: insufficient data, pi is a coin flip")


if __name__ == "__main__":
    main()
