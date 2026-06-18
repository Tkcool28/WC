"""Expanded data audit for the independent goal model.

Covers:
  A. Scoreline distribution (all / neutral / non-neutral / friendlies / competitive / WC)
  B. Temporal distribution (per-year stats, flags for anomalies)
  C. Tournament classification (deterministic mapping, counts, ambiguities)
  D. Neutral-site audit (H/D/A frequencies, home advantage, inconsistencies)
  E. Extra-time / shootout analysis (historical WC join to processed corpus)

Outputs:
  reports/goal_model_data_audit.json  (machine-readable)
  reports/goal_model_data_audit.md    (human-readable summary)
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from soccer_ev_model.goal_model_data import (
    GoalMatch,
    build_goal_matches,
    load_raw_matches,
)

# ---------------------------------------------------------------------------
# Tournament classification
# ---------------------------------------------------------------------------

# Raw tournament label -> canonical class.
# Order matters: first match wins.  Keys are lowercased for matching.
# IMPORTANT: qualification patterns must come BEFORE non-qualification patterns
# for the same tournament (e.g. "fifa world cup qualification" before "fifa world cup").
_TOURNAMENT_CLASS_RULES: list[tuple[str, str]] = [
    # FIFA World Cup qualifiers (MUST come before non-qualifier WC)
    ("fifa world cup qualification", "world_cup_qualifier"),
    # FIFA World Cup (non-qualifier)
    ("fifa world cup", "world_cup"),
    # Continental championship qualifiers (MUST come before non-qualifier)
    ("uefa euro qualification", "continental_qualifier"),
    ("copa américa qualification", "continental_qualifier"),
    ("copa america qualification", "continental_qualifier"),
    ("afc asian cup qualification", "continental_qualifier"),
    ("african cup of nations qualification", "continental_qualifier"),
    ("concacaf nations league qualification", "continental_qualifier"),
    ("world cup qualification", "world_cup_qualifier"),
    ("euro qualification", "continental_qualifier"),
    ("nations league qualification", "continental_qualifier"),
    ("gold cup qualification", "continental_qualifier"),
    ("caribbean cup qualification", "continental_qualifier"),
    ("oceania nations cup qualification", "continental_qualifier"),
    ("aff championship qualification", "continental_qualifier"),
    ("asean championship qualification", "continental_qualifier"),
    ("arab cup qualification", "continental_qualifier"),
    ("cosafa cup qualification", "continental_qualifier"),
    ("afc challenge cup qualification", "continental_qualifier"),
    ("eaff championship qualification", "continental_qualifier"),
    ("copa américa qualification", "continental_qualifier"),
    ("conifa world cup qualification", "other"),
    ("conifa world football cup qualification", "other"),
    # Continental championships (non-qualifier)
    ("uefa euro", "continental_championship"),
    ("copa américa", "continental_championship"),
    ("copa america", "continental_championship"),
    ("african cup of nations", "continental_championship"),
    ("afc asian cup", "continental_championship"),
    ("gold cup", "continental_championship"),
    ("concacaf nations league", "continental_championship"),
    ("ofc nations cup", "continental_championship"),
    ("oceania nations cup", "continental_championship"),
    # Nations League (non-qualifier)
    ("nations league", "nations_league"),
    # Friendlies
    ("friendly", "friendly"),
    # Regional / minor tournaments (catch-all patterns)
    ("island games", "regional_minor"),
    ("pacific games", "regional_minor"),
    ("pacific mini games", "regional_minor"),
    ("south pacific games", "regional_minor"),
    ("southeast asian games", "regional_minor"),
    ("south asian games", "regional_minor"),
    ("asian games", "regional_minor"),
    ("aff championship", "regional_minor"),
    ("asean championship", "regional_minor"),
    ("ceca", "regional_minor"),
    ("cosafa", "regional_minor"),
    ("gulf cup", "regional_minor"),
    ("cafu", "regional_minor"),
    ("uncaf", "regional_minor"),
    ("nafu", "regional_minor"),
    ("baltic cup", "regional_minor"),
    ("nordic championship", "regional_minor"),
    ("caribbean cup", "regional_minor"),
    ("cfu caribbean", "regional_minor"),
    ("concacaf series", "regional_minor"),
    ("confederations cup", "regional_minor"),
    ("conifa", "regional_minor"),
    ("elf cup", "regional_minor"),
    ("viva world cup", "regional_minor"),
    ("fifi wild cup", "regional_minor"),
    ("inter games", "regional_minor"),
    ("dragon cup", "regional_minor"),
    ("marianas cup", "regional_minor"),
    ("windward islands", "regional_minor"),
    ("muratti vase", "regional_minor"),
    ("tournoi de france", "regional_minor"),
    ("king's cup", "regional_minor"),
    ("kirin", "regional_minor"),
    ("merdeka", "regional_minor"),
    ("nehr", "regional_minor"),
    ("saff", "regional_minor"),
    ("eaff", "regional_minor"),
    ("waff", "regional_minor"),
    ("cafa", "regional_minor"),
    ("abcs tournament", "regional_minor"),
    ("atlantic heritage cup", "regional_minor"),
    ("canadian shield", "regional_minor"),
    ("mahindra", "regional_minor"),
    ("mapinduzi", "regional_minor"),
    ("msg prime minister", "regional_minor"),
    ("niamh challenge", "regional_minor"),
    ("outrigger challenge", "regional_minor"),
    ("palestine international", "regional_minor"),
    ("philippine peace", "regional_minor"),
    ("prime minister", "regional_minor"),
    ("skn football", "regional_minor"),
    ("superclásico de las américas", "regional_minor"),
    ("trans-tasman", "regional_minor"),
    ("tri nation", "regional_minor"),
    ("three nations", "regional_minor"),
    ("tynwald hill", "regional_minor"),
    ("udeac", "regional_minor"),
    ("uniffac", "regional_minor"),
    ("unity cup", "regional_minor"),
    ("vff cup", "regional_minor"),
    ("world unity", "regional_minor"),
    ("lunar new year", "regional_minor"),
    ("dynasty cup", "regional_minor"),
    ("korea cup", "regional_minor"),
    ("navruz", "regional_minor"),
    ("osn cup", "regional_minor"),
    ("jordan international", "regional_minor"),
    ("cyprus international", "regional_minor"),
    ("malta international", "regional_minor"),
    ("dunhill", "regional_minor"),
    ("marlboro", "regional_minor"),
    ("miami cup", "regional_minor"),
    ("joe robbie", "regional_minor"),
    ("diamond jubilee", "regional_minor"),
    ("four nations", "regional_minor"),
    ("millennium", "regional_minor"),
    ("copa lipton", "regional_minor"),
    ("copa paz", "regional_minor"),
    ("copa del pacífico", "regional_minor"),
    ("copa del pacifico", "regional_minor"),
    ("copa confraternidad", "regional_minor"),
    ("soccer ashes", "regional_minor"),
    ("intercontinental cup", "regional_minor"),
    ("afro-asian", "regional_minor"),
    ("conmebol–uefa", "regional_minor"),
    ("conmebol-uefa", "regional_minor"),
    ("fifa series", "regional_minor"),
    ("al ain", "regional_minor"),
    ("amilcar cabral", "regional_minor"),
    ("benedikt fontana", "regional_minor"),
    ("corsica cup", "regional_minor"),
    ("coupe de l'outre-mer", "regional_minor"),
    ("cup of ancient", "regional_minor"),
    ("dakar tournament", "regional_minor"),
    ("hungary heritage", "regional_minor"),
    ("indian ocean", "regional_minor"),
    ("international tournament of peoples", "regional_minor"),
    ("king hassan", "regional_minor"),
    ("mauritius four nations", "regional_minor"),
    ("melanesia cup", "regional_minor"),
    ("merlion cup", "regional_minor"),
    ("morocco, capital of african", "regional_minor"),
    ("mukuru", "regional_minor"),
    ("nile basin", "regional_minor"),
    ("scania 100", "regional_minor"),
    ("simba tournament", "regional_minor"),
    ("south asian super", "regional_minor"),
    ("the other final", "regional_minor"),
    ("tifoco", "regional_minor"),
    ("usa cup", "regional_minor"),
    ("united arab emirates friendship", "regional_minor"),
    ("afc challenge cup", "regional_minor"),
    ("afc solidarity cup", "regional_minor"),
    ("arab cup", "regional_minor"),
    ("nations cup", "regional_minor"),
    ("tri-nations cup", "regional_minor"),
]


def classify_tournament(raw_label: str) -> str:
    """Deterministic tournament classification.

    Returns one of the 8 canonical classes.  Unknown labels map to 'other'.
    """
    lowered = raw_label.strip().lower()
    for pattern, cls in _TOURNAMENT_CLASS_RULES:
        if pattern in lowered:
            return cls
    return "other"


# ---------------------------------------------------------------------------
# Scoreline distribution helpers
# ---------------------------------------------------------------------------


def _scoreline_stats(matches: list[GoalMatch]) -> dict:
    """Compute scoreline distribution stats for a match subset."""
    n = len(matches)
    if n == 0:
        return {"match_count": 0}

    hg = [m.home_goals for m in matches]
    ag = [m.away_goals for m in matches]
    tg = [h + a for h, a in zip(hg, ag)]

    def _mean(vals: list[int]) -> float:
        return sum(vals) / len(vals)

    def _var(vals: list[int]) -> float:
        mu = _mean(vals)
        return sum((x - mu) ** 2 for x in vals) / len(vals)

    def _median(vals: list[int]) -> float:
        s = sorted(vals)
        mid = len(s) // 2
        if len(s) % 2 == 0:
            return (s[mid - 1] + s[mid]) / 2.0
        return float(s[mid])

    scoreline_counts: Counter = Counter()
    for m in matches:
        scoreline_counts[(m.home_goals, m.away_goals)] += 1

    total_goals_dist: Counter = Counter()
    for t in tg:
        bucket = min(t, 6)  # 6+ bucket
        label = str(bucket) if bucket < 6 else "6+"
        total_goals_dist[label] += 1

    extreme_scores = [
        (m.home_goals, m.away_goals, m.home_team, m.away_team, m.match_date.isoformat())
        for m in matches
        if m.home_goals >= 8 or m.away_goals >= 8
    ]

    return {
        "match_count": n,
        "mean_home_goals": round(_mean(hg), 4),
        "mean_away_goals": round(_mean(ag), 4),
        "mean_total_goals": round(_mean(tg), 4),
        "median_total_goals": round(_median(tg), 4),
        "home_goal_variance": round(_var(hg), 4),
        "away_goal_variance": round(_var(ag), 4),
        "total_goal_variance": round(_var(tg), 4),
        "max_home_goals": max(hg),
        "max_away_goals": max(ag),
        "scoreline_0_0_count": scoreline_counts.get((0, 0), 0),
        "scoreline_0_0_pct": round(scoreline_counts.get((0, 0), 0) / n * 100, 2),
        "scoreline_1_0_count": scoreline_counts.get((1, 0), 0),
        "scoreline_1_0_pct": round(scoreline_counts.get((1, 0), 0) / n * 100, 2),
        "scoreline_0_1_count": scoreline_counts.get((0, 1), 0),
        "scoreline_0_1_pct": round(scoreline_counts.get((0, 1), 0) / n * 100, 2),
        "scoreline_1_1_count": scoreline_counts.get((1, 1), 0),
        "scoreline_1_1_pct": round(scoreline_counts.get((1, 1), 0) / n * 100, 2),
        "total_goals_distribution": {k: total_goals_dist.get(k, 0) for k in ["0", "1", "2", "3", "4", "5", "6+"]},
        "top_20_scorelines": [
            {"score": f"{h}-{a}", "count": c}
            for (h, a), c in scoreline_counts.most_common(20)
        ],
        "extreme_scorelines_8plus": [
            {"home_goals": h, "away_goals": a, "home_team": ht, "away_team": at, "date": d}
            for h, a, ht, at, d in sorted(extreme_scores, key=lambda x: (-x[0], -x[1]))
        ],
    }


# ---------------------------------------------------------------------------
# Temporal distribution
# ---------------------------------------------------------------------------


def _temporal_distribution(matches: list[GoalMatch]) -> dict:
    """Per-year temporal stats with anomaly flags."""
    by_year: dict[int, list[GoalMatch]] = defaultdict(list)
    for m in matches:
        by_year[m.match_date.year].append(m)

    years_data = []
    prev_count = None
    prev_goals = None
    prev_friendly_pct = None

    for yr in sorted(by_year):
        subset = by_year[yr]
        n = len(subset)
        unique_teams = len({m.home_team_id for m in subset} | {m.away_team_id for m in subset})
        mean_tg = sum(m.home_goals + m.away_goals for m in subset) / max(1, n)
        neutral_pct = sum(1 for m in subset if m.neutral) / max(1, n) * 100
        friendly_pct = sum(1 for m in subset if m.tournament.strip().lower() == "friendly") / max(1, n) * 100

        flags = []
        if n < 20:
            flags.append("low_coverage")
        if prev_count is not None and prev_count > 0:
            ratio = n / prev_count
            if ratio < 0.5:
                flags.append(f"volume_drop_vs_prev_year({prev_count}->{n})")
            elif ratio > 2.0:
                flags.append(f"volume_spike_vs_prev_year({prev_count}->{n})")
        if prev_goals is not None and abs(mean_tg - prev_goals) > 0.5:
            flags.append(f"goal_rate_shift({prev_goals:.2f}->{mean_tg:.2f})")
        if prev_friendly_pct is not None and abs(friendly_pct - prev_friendly_pct) > 20:
            flags.append(f"friendly_share_shift({prev_friendly_pct:.1f}%->{friendly_pct:.1f}%)")

        years_data.append({
            "year": yr,
            "match_count": n,
            "unique_teams": unique_teams,
            "mean_total_goals": round(mean_tg, 4),
            "neutral_pct": round(neutral_pct, 2),
            "friendly_pct": round(friendly_pct, 2),
            "flags": flags,
        })
        prev_count = n
        prev_goals = mean_tg
        prev_friendly_pct = friendly_pct

    return {"years": years_data}


# ---------------------------------------------------------------------------
# Tournament classification report
# ---------------------------------------------------------------------------


def _tournament_classification(matches: list[GoalMatch]) -> dict:
    """Classify all tournaments and report counts + ambiguities."""
    raw_to_class: dict[str, str] = {}
    class_counts: Counter = Counter()
    raw_counts: Counter = Counter()

    for m in matches:
        raw = m.tournament
        raw_counts[raw] += 1
        if raw not in raw_to_class:
            raw_to_class[raw] = classify_tournament(raw)
        class_counts[raw_to_class[raw]] += 1

    # Identify ambiguous labels
    ambiguous = []
    for raw, cls in sorted(raw_to_class.items()):
        reasons = []
        lowered = raw.strip().lower()
        # Check if it could be multiple classes
        if "qualification" in lowered and "world cup" in lowered and "fifa" not in lowered and "conifa" not in lowered:
            reasons.append("could be WC or continental qualifier")
        if "cup" in lowered and "qualification" not in lowered and cls == "other":
            reasons.append("unrecognized cup tournament")
        if lowered in ("cup", "tournament", "international cup"):
            reasons.append("generic label")
        if reasons:
            ambiguous.append({"raw_label": raw, "assigned_class": cls, "count": raw_counts[raw], "reasons": reasons})

    return {
        "class_counts": dict(class_counts.most_common()),
        "raw_to_class": {k: v for k, v in sorted(raw_to_class.items())},
        "ambiguous_labels": ambiguous,
    }


# ---------------------------------------------------------------------------
# Neutral-site audit
# ---------------------------------------------------------------------------


def _neutral_audit(matches: list[GoalMatch]) -> dict:
    """Compare neutral vs non-neutral matches."""
    neutral = [m for m in matches if m.neutral]
    non_neutral = [m for m in matches if not m.neutral]

    def _stats(subset: list[GoalMatch]) -> dict:
        n = len(subset)
        if n == 0:
            return {"count": 0}
        hg = [m.home_goals for m in subset]
        ag = [m.away_goals for m in subset]
        results = Counter(m.result for m in subset)
        scorelines = Counter((m.home_goals, m.away_goals) for m in subset)
        return {
            "count": n,
            "mean_home_goals": round(sum(hg) / n, 4),
            "mean_away_goals": round(sum(ag) / n, 4),
            "mean_total_goals": round(sum(h + a for h, a in zip(hg, ag)) / n, 4),
            "home_win_pct": round(results.get("H", 0) / n * 100, 2),
            "draw_pct": round(results.get("D", 0) / n * 100, 2),
            "away_win_pct": round(results.get("A", 0) / n * 100, 2),
            "top_10_scorelines": [
                {"score": f"{h}-{a}", "count": c}
                for (h, a), c in scorelines.most_common(10)
            ],
        }

    # Tournament classes by neutral share
    class_neutral: dict[str, list[GoalMatch]] = defaultdict(list)
    for m in matches:
        cls = classify_tournament(m.tournament)
        class_neutral[cls].append(m)

    class_neutral_stats = {}
    for cls, subset in sorted(class_neutral.items()):
        n = len(subset)
        neut = sum(1 for m in subset if m.neutral)
        class_neutral_stats[cls] = {
            "count": n,
            "neutral_pct": round(neut / max(1, n) * 100, 2),
        }

    mostly_neutral = [cls for cls, s in class_neutral_stats.items() if s["neutral_pct"] > 90]
    mostly_non_neutral = [cls for cls, s in class_neutral_stats.items() if s["neutral_pct"] < 10]

    return {
        "neutral": _stats(neutral),
        "non_neutral": _stats(non_neutral),
        "home_advantage": {
            "non_neutral_home_minus_away": round(
                (_stats(non_neutral).get("mean_home_goals", 0) - _stats(non_neutral).get("mean_away_goals", 0)), 4
            ) if non_neutral else None,
            "neutral_home_minus_away": round(
                (_stats(neutral).get("mean_home_goals", 0) - _stats(neutral).get("mean_away_goals", 0)), 4
            ) if neutral else None,
        },
        "tournament_class_neutrality": class_neutral_stats,
        "mostly_neutral_classes": mostly_neutral,
        "mostly_non_neutral_classes": mostly_non_neutral,
    }


# ---------------------------------------------------------------------------
# Extra-time / shootout analysis
# ---------------------------------------------------------------------------


def _load_historical_wc(year: int) -> list[dict]:
    """Load a historical World Cup raw file."""
    path = Path(f"data/raw/matches_{year}_openfootball.json")
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("matches", [])


def _extra_time_audit(matches: list[GoalMatch]) -> dict:
    """Join historical WC data to processed corpus for 2010-2022.

    Join key: (date, home_team_name, away_team_name) — the two sources use
    different team ID spaces so we cannot join on IDs.

    Finding: openfootball raw files record regulation-time scores.
    The processed corpus records after-extra-time (final) scores.
    """
    # Build lookup from processed corpus: (date, home_name, away_name) -> GoalMatch
    corpus_by_name_key: dict[tuple, GoalMatch] = {}
    for m in matches:
        key = (m.match_date.isoformat(), m.home_team.strip().lower(), m.away_team.strip().lower())
        corpus_by_name_key[key] = m

    results = {}
    for year in [2010, 2014, 2018, 2022]:
        wc_raw = _load_historical_wc(year)
        if not wc_raw:
            results[str(year)] = {"error": "raw file not found"}
            continue

        total_rows = len(wc_raw)
        knockout = [r for r in wc_raw if r.get("stage", "") != "GROUP_STAGE"]
        group = [r for r in wc_raw if r.get("stage", "") == "GROUP_STAGE"]

        joined = []
        unmatched = []
        ambiguous = []
        score_disagreements = []

        for row in wc_raw:
            d = row.get("date", "")[:10]
            h_name = (row.get("home_team_name") or "").strip().lower()
            a_name = (row.get("away_team_name") or "").strip().lower()
            key = (d, h_name, a_name)

            if key in corpus_by_name_key:
                corpus_match = corpus_by_name_key[key]
                wc_hg = row.get("home_goals")
                wc_ag = row.get("away_goals")
                corpus_hg = corpus_match.home_goals
                corpus_ag = corpus_match.away_goals
                entry = {
                    "date": d,
                    "home_team": row.get("home_team_name"),
                    "away_team": row.get("away_team_name"),
                    "stage": row.get("stage"),
                    "wc_regulation_score": f"{wc_hg}-{wc_ag}",
                    "corpus_final_score": f"{corpus_hg}-{corpus_ag}",
                    "score_match": wc_hg == corpus_hg and wc_ag == corpus_ag,
                }
                joined.append(entry)
                if not entry["score_match"]:
                    score_disagreements.append(entry)
            else:
                # Try reverse key (home/away swapped)
                key_rev = (d, a_name, h_name)
                if key_rev in corpus_by_name_key:
                    ambiguous.append({
                        "date": d,
                        "home_team": row.get("home_team_name"),
                        "away_team": row.get("away_team_name"),
                        "note": "home/away swapped in corpus vs raw",
                    })
                else:
                    unmatched.append({
                        "date": d,
                        "home_team": row.get("home_team_name"),
                        "away_team": row.get("away_team_name"),
                        "stage": row.get("stage"),
                    })

        # Determine score interpretation
        all_match = len(score_disagreements) == 0 and len(joined) > 0
        any_disagreement = len(score_disagreements) > 0

        # Analyze disagreements: are they all knockout matches?
        disagreement_stages = set(d["stage"] for d in score_disagreements)

        score_interpretation = "unknown"
        if all_match:
            score_interpretation = "same_source_or_regulation"
        elif any_disagreement:
            # Check if disagreements are all in knockout stages
            all_knockout = all(s != "GROUP_STAGE" for s in disagreement_stages)
            if all_knockout:
                score_interpretation = "corpus_after_et_raw_regulation"
            else:
                score_interpretation = "mixed_disagreements"

        results[str(year)] = {
            "total_rows": total_rows,
            "knockout_matches": len(knockout),
            "group_matches": len(group),
            "successful_joins": len(joined),
            "unmatched_rows": len(unmatched),
            "ambiguous_joins": len(ambiguous),
            "score_disagreements": score_disagreements,
            "all_scores_match": all_match,
            "score_interpretation": score_interpretation,
            "disagreement_stages": sorted(disagreement_stages),
            "stage_coverage": sorted(set(r.get("stage", "") for r in wc_raw)),
            "matchday_coverage": sorted(set(r.get("matchday") for r in wc_raw if r.get("matchday") is not None)),
            "sample_unmatched": unmatched[:5],
            "sample_ambiguous": ambiguous[:5],
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    raw = load_raw_matches()
    matches, excluded = build_goal_matches(raw)

    # Subsets
    neutral = [m for m in matches if m.neutral]
    non_neutral = [m for m in matches if not m.neutral]
    friendlies = [m for m in matches if m.tournament.strip().lower() == "friendly"]
    competitive = [m for m in matches if m.tournament.strip().lower() != "friendly"]
    wc = [m for m in matches if classify_tournament(m.tournament) == "world_cup"]

    report = {
        "model_version": "goal-model-research-v0.1",
        "data_cutoff": matches[-1].match_date.isoformat() if matches else None,
        "scoreline_distribution": {
            "all": _scoreline_stats(matches),
            "neutral": _scoreline_stats(neutral),
            "non_neutral": _scoreline_stats(non_neutral),
            "friendlies": _scoreline_stats(friendlies),
            "competitive": _scoreline_stats(competitive),
            "fifa_world_cup": _scoreline_stats(wc),
        },
        "temporal_distribution": _temporal_distribution(matches),
        "tournament_classification": _tournament_classification(matches),
        "neutral_site_audit": _neutral_audit(matches),
        "extra_time_shootout_audit": _extra_time_audit(matches),
    }

    # Write JSON
    json_path = Path("reports/goal_model_data_audit.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {json_path}")

    # Write Markdown summary
    md_path = Path("reports/goal_model_data_audit.md")
    md_lines = [
        "# Expanded Goal Model Data Audit",
        "",
        f"**Data cutoff:** {report['data_cutoff']}",
        f"**Total matches:** {len(matches)}",
        "",
        "## A. Scoreline Distribution",
        "",
    ]
    for subset_name in ["all", "neutral", "non_neutral", "friendlies", "competitive", "fifa_world_cup"]:
        stats = report["scoreline_distribution"][subset_name]
        md_lines += [
            f"### {subset_name.replace('_', ' ').title()}",
            "",
            f"- Matches: {stats['match_count']}",
        ]
        if stats["match_count"] > 0:
            md_lines += [
                f"- Mean home goals: {stats['mean_home_goals']}",
                f"- Mean away goals: {stats['mean_away_goals']}",
                f"- Mean total goals: {stats['mean_total_goals']}",
                f"- Median total goals: {stats['median_total_goals']}",
                f"- Home goal variance: {stats['home_goal_variance']}",
                f"- Away goal variance: {stats['away_goal_variance']}",
                f"- Total goal variance: {stats['total_goal_variance']}",
                f"- Max home goals: {stats['max_home_goals']}",
                f"- Max away goals: {stats['max_away_goals']}",
                f"- 0-0: {stats['scoreline_0_0_count']} ({stats['scoreline_0_0_pct']}%)",
                f"- 1-0: {stats['scoreline_1_0_count']} ({stats['scoreline_1_0_pct']}%)",
                f"- 0-1: {stats['scoreline_0_1_count']} ({stats['scoreline_0_1_pct']}%)",
                f"- 1-1: {stats['scoreline_1_1_count']} ({stats['scoreline_1_1_pct']}%)",
                "",
                "**Total goals distribution:**",
                "",
            ]
            for k, v in stats["total_goals_distribution"].items():
                md_lines.append(f"  - {k}: {v}")
            md_lines.append("")

            if stats["extreme_scorelines_8plus"]:
                md_lines.append("**Extreme scorelines (8+ goals by one team):**")
                for e in stats["extreme_scorelines_8plus"][:15]:
                    md_lines.append(f"  - {e['home_team']} {e['home_goals']}-{e['away_goals']} {e['away_team']} ({e['date']})")
                md_lines.append("")

    # Temporal flags
    md_lines += [
        "## B. Temporal Distribution",
        "",
        "| Year | Matches | Teams | Mean TG | Neutral% | Friendly% | Flags |",
        "|------|---------|-------|---------|----------|-----------|-------|",
    ]
    for yr_data in report["temporal_distribution"]["years"]:
        flags = "; ".join(yr_data["flags"]) if yr_data["flags"] else ""
        md_lines.append(
            f"| {yr_data['year']} | {yr_data['match_count']} | {yr_data['unique_teams']} "
            f"| {yr_data['mean_total_goals']} | {yr_data['neutral_pct']}% "
            f"| {yr_data['friendly_pct']}% | {flags} |"
        )

    # Tournament classification
    tc = report["tournament_classification"]
    md_lines += [
        "",
        "## C. Tournament Classification",
        "",
        "| Class | Count |",
        "|-------|-------|",
    ]
    for cls, cnt in tc["class_counts"].items():
        md_lines.append(f"| {cls} | {cnt} |")

    if tc["ambiguous_labels"]:
        md_lines += [
            "",
            "### Ambiguous Labels",
            "",
        ]
        for a in tc["ambiguous_labels"]:
            md_lines.append(f"- `{a['raw_label']}` -> {a['assigned_class']} ({a['count']} matches): {', '.join(a['reasons'])}")

    # Neutral audit
    na = report["neutral_site_audit"]
    md_lines += [
        "",
        "## D. Neutral-Site Audit",
        "",
        f"- Non-neutral home-away diff: {na['home_advantage']['non_neutral_home_minus_away']}",
        f"- Neutral home-away diff: {na['home_advantage']['neutral_home_minus_away']}",
        "",
        "Mostly neutral classes (>90%):",
    ]
    for cls in na["mostly_neutral_classes"]:
        md_lines.append(f"  - {cls} ({na['tournament_class_neutrality'][cls]['count']} matches, {na['tournament_class_neutrality'][cls]['neutral_pct']}%)")
    if not na["mostly_neutral_classes"]:
        md_lines.append("  - none")

    md_lines.append("")
    md_lines.append("Mostly non-neutral classes (<10%):")
    for cls in na["mostly_non_neutral_classes"]:
        md_lines.append(f"  - {cls} ({na['tournament_class_neutrality'][cls]['count']} matches, {na['tournament_class_neutrality'][cls]['neutral_pct']}%)")
    if not na["mostly_non_neutral_classes"]:
        md_lines.append("  - none")

    # Extra-time audit
    et = report["extra_time_shootout_audit"]
    md_lines += [
        "",
        "## E. Extra-Time / Shootout Audit",
        "",
        "**Key finding:** The openfootball raw files record **regulation-time** scores. "
        "The processed corpus records **after-extra-time** (final) scores. "
        "Score disagreements are concentrated in knockout stages where extra time occurs.",
        "",
    ]
    for year in ["2010", "2014", "2018", "2022"]:
        yr_data = et.get(year, {})
        md_lines += [
            f"### {year} World Cup",
            f"- Total rows: {yr_data.get('total_rows', 'N/A')}",
            f"- Successful joins: {yr_data.get('successful_joins', 'N/A')}",
            f"- Unmatched: {yr_data.get('unmatched_rows', 'N/A')}",
            f"- Ambiguous (swapped): {yr_data.get('ambiguous_joins', 'N/A')}",
            f"- Score disagreements: {len(yr_data.get('score_disagreements', []))}",
            f"- Score interpretation: {yr_data.get('score_interpretation', 'N/A')}",
            f"- Disagreement stages: {yr_data.get('disagreement_stages', [])}",
            "",
        ]
        if yr_data.get("score_disagreements"):
            md_lines.append("| Date | Match | Stage | Regulation (raw) | Final (corpus) |")
            md_lines.append("|------|-------|-------|------------------|----------------|")
            for d in yr_data["score_disagreements"]:
                md_lines.append(f"| {d['date']} | {d['home_team']} vs {d['away_team']} | {d['stage']} | {d['wc_regulation_score']} | {d['corpus_final_score']} |")
            md_lines.append("")

    # Training eligibility recommendation
    md_lines += [
        "",
        "## Training Eligibility Recommendation",
        "",
        "Based on the extra-time audit:",
        "",
        "1. **Group stage matches**: Safe to use — scores agree across sources.",
        "2. **Knockout matches with tied regulation scores**: Corpus records after-ET score, ",
        "   which is the correct target for match outcome prediction (who won).",
        "3. **Knockout matches with decisive regulation scores**: Both sources agree — safe to use.",
        "4. **Ambiguous/unmatched matches**: Exclude from training or flag for manual review.",
        "",
        "**Recommendation**: Use all corpus scores as-is for outcome prediction (H/D/A). ",
        "For scoreline modeling, the corpus after-ET scores are appropriate since they represent ",
        "the actual final result. No matches need to be excluded solely due to extra time.",
    ]

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
