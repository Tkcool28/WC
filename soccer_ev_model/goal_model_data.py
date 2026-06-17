"""Data loading and audit helpers for the independent international goal model.

This module never mutates raw inputs.  It normalizes the tracked processed corpus
into a deterministic, scoreline-safe table and reports data limitations explicitly.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import date
import json
from pathlib import Path
from typing import Iterable

DEFAULT_MATCH_PATH = Path("data/processed/international_matches.json")


@dataclass(frozen=True)
class GoalMatch:
    match_date: date
    home_team: str
    away_team: str
    home_team_id: int
    away_team_id: int
    home_goals: int
    away_goals: int
    tournament: str
    neutral: bool

    @property
    def result(self) -> str:
        if self.home_goals > self.away_goals:
            return "H"
        if self.home_goals < self.away_goals:
            return "A"
        return "D"


def _parse_date(value: object) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def load_raw_matches(path: str | Path = DEFAULT_MATCH_PATH) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("international match corpus must be a JSON list")
    return payload


def build_goal_matches(rows: Iterable[dict]) -> tuple[list[GoalMatch], Counter]:
    matches: list[GoalMatch] = []
    excluded: Counter = Counter()
    seen: set[tuple] = set()
    for row in rows:
        d = _parse_date(row.get("date"))
        if d is None:
            excluded["invalid_date"] += 1
            continue
        required = ("home_team", "away_team", "home_team_id", "away_team_id", "home_goals", "away_goals")
        if any(row.get(k) is None for k in required):
            excluded["missing_required_field"] += 1
            continue
        try:
            hg, ag = int(row["home_goals"]), int(row["away_goals"])
            hid, aid = int(row["home_team_id"]), int(row["away_team_id"])
        except (TypeError, ValueError):
            excluded["invalid_numeric_field"] += 1
            continue
        if hg < 0 or ag < 0 or hid == aid:
            excluded["invalid_score_or_identity"] += 1
            continue
        key = (d.isoformat(), hid, aid, hg, ag, str(row.get("tournament", "")), bool(row.get("neutral", False)))
        if key in seen:
            excluded["exact_duplicate"] += 1
            continue
        seen.add(key)
        matches.append(GoalMatch(
            match_date=d,
            home_team=str(row["home_team"]).strip(),
            away_team=str(row["away_team"]).strip(),
            home_team_id=hid,
            away_team_id=aid,
            home_goals=hg,
            away_goals=ag,
            tournament=str(row.get("tournament") or "Unknown").strip(),
            neutral=bool(row.get("neutral", False)),
        ))
    matches.sort(key=lambda m: (m.match_date, m.home_team_id, m.away_team_id))
    return matches, excluded


def audit_goal_corpus(path: str | Path = DEFAULT_MATCH_PATH) -> dict:
    raw = load_raw_matches(path)
    clean, excluded = build_goal_matches(raw)
    ids_to_names: dict[int, set[str]] = defaultdict(set)
    names_to_ids: dict[str, set[int]] = defaultdict(set)
    tournament_counts: Counter = Counter()
    result_counts: Counter = Counter()
    neutral_count = 0
    same_fixture_day: Counter = Counter()
    for m in clean:
        ids_to_names[m.home_team_id].add(m.home_team)
        ids_to_names[m.away_team_id].add(m.away_team)
        names_to_ids[m.home_team].add(m.home_team_id)
        names_to_ids[m.away_team].add(m.away_team_id)
        tournament_counts[m.tournament] += 1
        result_counts[m.result] += 1
        neutral_count += int(m.neutral)
        same_fixture_day[(m.match_date, m.home_team_id, m.away_team_id)] += 1
    ambiguous_ids = {str(k): sorted(v) for k, v in ids_to_names.items() if len(v) > 1}
    ambiguous_names = {k: sorted(v) for k, v in names_to_ids.items() if len(v) > 1}
    duplicate_fixture_rows = sum(v - 1 for v in same_fixture_day.values() if v > 1)
    return {
        "raw_rows": len(raw),
        "usable_scorelines": len(clean),
        "date_min": clean[0].match_date.isoformat() if clean else None,
        "date_max": clean[-1].match_date.isoformat() if clean else None,
        "unique_team_ids": len(ids_to_names),
        "unique_team_names": len(names_to_ids),
        "neutral_matches": neutral_count,
        "non_neutral_matches": len(clean) - neutral_count,
        "result_counts": dict(result_counts),
        "tournament_count": len(tournament_counts),
        "top_tournaments": tournament_counts.most_common(25),
        "excluded": dict(excluded),
        "duplicate_fixture_day_rows": duplicate_fixture_rows,
        "team_ids_with_multiple_names": ambiguous_ids,
        "team_names_with_multiple_ids": ambiguous_names,
        "has_stage_field": any("stage" in r and r.get("stage") not in (None, "") for r in raw),
        "has_matchday_field": any("matchday" in r and r.get("matchday") not in (None, "") for r in raw),
        "has_extra_time_field": any(any(k in r for k in ("extra_time", "after_extra_time", "aet")) for r in raw),
        "has_shootout_field": any(any(k in r for k in ("penalties", "shootout", "penalty_score")) for r in raw),
        "limitations": [
            "The processed corpus has tournament labels but no reliable stage/matchday state fields unless reported above.",
            "Regulation, extra-time and shootout scores cannot be separated when explicit fields are absent.",
            "Knockout matches with tied listed scores must not be interpreted as advancement outcomes.",
        ],
    }


def write_audit_report(output: str | Path, path: str | Path = DEFAULT_MATCH_PATH) -> dict:
    report = audit_goal_corpus(path)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
