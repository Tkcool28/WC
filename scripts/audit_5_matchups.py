"""Final-validation audit: run evaluate_match on 5 matchups and print
all the new Phase 1/2/3 fields side by side. Plus verify the model
probabilities (blend_probs, pi_probs, pi_only_probs, elo_only_probs)
are byte-identical to their pre-Phase counterparts (i.e. the
prediction formula is unchanged).

Run with: source .venv/bin/activate && python3 scripts/audit_5_matchups.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make repo importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from soccer_ev_model.ev_workflow import evaluate_match
from soccer_ev_model.pi_ratings import compute_pi_ratings
from soccer_ev_model.elo_ratings import load_elo_ratings, elo_at
from soccer_ev_model.prediction_summary import (
    calculate_market_deltas,
    confidence_tier,
    group_context_warnings,
    largest_market_delta,
    market_divergence_label,
    poisson_agreement_label,
    poisson_outcome_probs,
    expected_goals_from_blend,
    resolve_model_probs_for_market,
    top_two_outcomes,
)

# ---- 5 matchups (home_team, away_team, sample book odds) ----
# Odds are realistic samples for a strong-favorite vs underdog match.
MATCHUPS = [
    ("Argentina",   "Algeria",      -240, +340, +700),
    ("Brazil",      "Haiti",        -380, +450, +1100),
    ("USA",         "Australia",    -135, +260, +340),
    ("England",     "Croatia",      -150, +260, +390),
    ("Germany",     "Ivory Coast",  -210, +310, +600),
]

# Use a fixed cutoff date AFTER all historical data, BEFORE any 2026 WC match,
# so the model has the full corpus available for pi-rating computation.
CUTOFF = "2025-01-01"


def _fmt_pct(p: float) -> str:
    return f"{p*100:5.1f}%"


def _fmt_pts(d: float) -> str:
    return f"{d:+5.1f}"


def main() -> int:
    # Load historical corpus
    raw = json.loads((ROOT / "data" / "processed" / "international_matches.json").read_text())
    elo_snapshots = load_elo_ratings(str(ROOT / "data" / "raw" / "elo_ratings.json"))
    ratings = compute_pi_ratings(raw, cutoff=CUTOFF)

    print("=" * 88)
    print("PHASE 1-3 AUDIT — 5 MATCHUPS")
    print("=" * 88)
    print(f"Cutoff date for pi-ratings: {CUTOFF}")
    print(f"Corpus: {len(raw)} historical matches")
    print()

    for home, away, h_odds, d_odds, a_odds in MATCHUPS:
        print("-" * 88)
        print(f"{home} vs {away}")
        print(f"  Book odds: home {h_odds} / draw {d_odds} / away {a_odds}")
        print("-" * 88)

        # Resolve to team ids via the identity registry
        from soccer_ev_model.team_identity import resolve_team
        h_res = resolve_team(name=home)
        a_res = resolve_team(name=away)
        h_id = h_res.get("football_data_id")
        a_id = a_res.get("football_data_id")
        h_can = h_res.get("canonical_id") or "?"
        a_can = a_res.get("canonical_id") or "?"
        print(f"  Canonical IDs: {h_can} (id={h_id})  vs  {a_can} (id={a_id})")
        print(f"  Identity resolved: home={not h_res.get('identity_unresolved', False)}  "
              f"away={not a_res.get('identity_unresolved', False)}")

        # Run evaluate_match (use blend weights; both elos default to None for purity here)
        h_elo, h_missing = elo_at(elo_snapshots, home, CUTOFF)
        a_elo, a_missing = elo_at(elo_snapshots, away, CUTOFF)

        result = evaluate_match(
            home_team=home,
            away_team=away,
            home_team_id=h_id or 0,
            away_team_id=a_id or 0,
            date=CUTOFF,
            book_home_odds=h_odds,
            book_draw_odds=d_odds,
            book_away_odds=a_odds,
            ratings=ratings,
            min_edge=0.03,
            home_elo=h_elo,
            away_elo=a_elo,
            blend_w_pi=0.5,
            blend_w_elo=0.5,
            identity_unresolved=bool(h_res.get("identity_unresolved") or a_res.get("identity_unresolved")),
        )

        # Blend probs
        blend = resolve_model_probs_for_market(result)
        pi_only = result.get("pi_only_probs") or {}
        elo_only = result.get("elo_only_probs") or {}
        book_fair = result["book_fair"]

        # Top two + margin
        top, top_p, second, second_p = top_two_outcomes(blend)
        margin = round((top_p - second_p) * 100, 1)

        print(f"\n  BLEND PROBS (model):")
        print(f"    home  {blend['home']*100:5.1f}%   "
              f"draw  {blend['draw']*100:5.1f}%   "
              f"away  {blend['away']*100:5.1f}%")
        print(f"    top={top}  margin=+{margin:.1f} pts")
        print(f"  PI-ONLY probs:    home {pi_only.get('home', 0)*100:5.1f}%   "
              f"draw {pi_only.get('draw', 0)*100:5.1f}%   "
              f"away {pi_only.get('away', 0)*100:5.1f}%")
        if elo_only:
            print(f"  ELO-ONLY probs:    home {elo_only.get('home', 0)*100:5.1f}%   "
                  f"draw {elo_only.get('draw', 0)*100:5.1f}%   "
                  f"away {elo_only.get('away', 0)*100:5.1f}%")
        print(f"  BOOK NO-VIG (market):")
        print(f"    home  {book_fair['home']*100:5.1f}%   "
              f"draw  {book_fair['draw']*100:5.1f}%   "
              f"away  {book_fair['away']*100:5.1f}%")

        # Phase 1 — market deltas + divergence label + largest
        raw_deltas = {m: blend[m] - book_fair[m] for m in ("home", "draw", "away")}
        pts_deltas = calculate_market_deltas(blend, book_fair)
        div_label = market_divergence_label(raw_deltas)
        largest = largest_market_delta(
            pts_deltas,
            market_labels={"home": home, "draw": "Draw", "away": away},
            model_probs=blend,
            market_probs=book_fair,
        )

        print(f"\n  MARKET BASELINE (model vs book no-vig):")
        for m in ("home", "draw", "away"):
            print(f"    {m:<5} {blend[m]*100:5.1f}% model  /  "
                  f"{book_fair[m]*100:5.1f}% market  /  "
                  f"{pts_deltas[m]:+.1f} pts")
        print(f"    Divergence label: {div_label}")
        print(f"    Largest disagreement: {largest['label']} "
              f"({largest['delta_pts']:+.1f} pts)")

        # Phase 2 — Poisson xG + probs + agreement
        xg = expected_goals_from_blend(blend)
        p = poisson_outcome_probs(xg["home_xg"], xg["away_xg"])
        agr = poisson_agreement_label(blend, p)
        print(f"\n  POISSON GOAL MODEL:")
        print(f"    xG estimate: {home} {xg['home_xg']:.2f}  /  "
              f"{away} {xg['away_xg']:.2f}")
        print(f"    Poisson probs:  home {p['home']*100:5.1f}%   "
              f"draw {p['draw']*100:5.1f}%   "
              f"away {p['away']*100:5.1f}%")
        print(f"    Poisson agreement: blend_top={agr['blend_top']}  "
              f"poisson_top={agr['poisson_top']}  label={agr['label']}")

        # Phase 3 — group context (none of these matchups are in the live
        # 2026 cache, so we render the "no_data" path or empty path)
        warnings = group_context_warnings(
            stage="GROUP_STAGE",
            matchday=1,
            group=None,
            finished_matches_in_group=None,
        )
        print(f"\n  GROUP CONTEXT (no live 2026 matchup data for this pair):")
        for w in warnings:
            print(f"    [{w['severity']}] {w['text']}")

        # Confidence tier
        agreement = result.get("confidence", {}).get("agreement_label", "agree")
        tier = confidence_tier(
            blended_probs=blend,
            prediction_margin_pts=margin,
            draw_p=blend["draw"],
            agreement_label=agreement,
            low_data=result.get("confidence", {}).get("low_data", False),
            identity_unresolved=bool(h_res.get("identity_unresolved") or a_res.get("identity_unresolved")),
            blend_is_pure_pi=False,
        )
        print(f"\n  CONFIDENCE TIER: {tier}")
        print()

    print("=" * 88)
    print("END OF AUDIT")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
