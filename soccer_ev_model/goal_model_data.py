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


# ---------------------------------------------------------------------------
# Tournament classification
# ---------------------------------------------------------------------------

# Ordered rules: first match wins.  Keys are lowercased for matching.
# Qualification patterns MUST precede non-qualification patterns for the same tournament.
_TOURNAMENT_CLASS_RULES: list[tuple[str, str]] = [
    # WC qualifiers (before non-qualifier WC)
    ("fifa world cup qualification", "world_cup_qualifier"),
    ("fifa world cup", "world_cup"),
    # Continental championship qualifiers (before non-qualifier)
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
    ("conifa world cup qualification", "other"),
    ("conifa world football cup qualification", "other"),
    # Continental championships (non-qualifier)
    ("uefa euro", "continental_championship"),
    ("copa américa", "continental_championship"),
    ("copa america", "continental_championship"),
    ("african cup of nations", "continental_championship"),
    ("afc asian cup", "continental_championship"),
    ("gold cup", "continental_championship"),
    ("concacaf nations league", "nations_league"),
    ("ofc nations cup", "continental_championship"),
    ("oceania nations cup", "continental_championship"),
    # Nations League (non-qualifier)
    ("nations league", "nations_league"),
    # Friendlies
    ("friendly", "friendly"),
    # Regional / minor tournaments
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

TOURNAMENT_CLASSES = [
    "world_cup",
    "continental_championship",
    "world_cup_qualifier",
    "continental_qualifier",
    "nations_league",
    "friendly",
    "regional_minor",
    "other",
]

# Default match-importance weights (scheme 1: mild)
MATCH_IMPORTANCE_WEIGHTS: dict[str, float] = {
    "world_cup": 1.20,
    "continental_championship": 1.15,
    "world_cup_qualifier": 1.10,
    "continental_qualifier": 1.05,
    "nations_league": 1.00,
    "friendly": 0.75,
    "regional_minor": 0.90,
    "other": 0.90,
}

# Stronger friendly discount (scheme 2)
MATCH_IMPORTANCE_WEIGHTS_STRONG: dict[str, float] = {
    "world_cup": 1.25,
    "continental_championship": 1.20,
    "world_cup_qualifier": 1.10,
    "continental_qualifier": 1.05,
    "nations_league": 1.00,
    "friendly": 0.50,
    "regional_minor": 0.80,
    "other": 0.80,
}


def classify_tournament(raw_label: str) -> str:
    """Deterministic tournament classification.

    Returns one of the 8 canonical classes.  Unknown labels map to 'other'.
    """
    lowered = raw_label.strip().lower()
    for pattern, cls in _TOURNAMENT_CLASS_RULES:
        if pattern in lowered:
            return cls
    return "other"


def tournament_class_counts(matches: Iterable[GoalMatch]) -> dict[str, int]:
    """Return match count per tournament class."""
    counts: Counter = Counter()
    for m in matches:
        counts[classify_tournament(m.tournament)] += 1
    return dict(counts)
