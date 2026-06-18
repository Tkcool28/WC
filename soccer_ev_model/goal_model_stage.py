"""Tournament stage and state enrichment for the independent goal model.

Reconstructs safely-available tournament context from raw match files:
  - tournament (FIFA World Cup, etc.)
  - year
  - stage (group_stage, round_of_16, quarter_final, semi_final, third_place, final)
  - group (Group A, ...)
  - matchday (1, 2, 3 for group; 4+ for knockout)
  - knockout flag
  - final-group-match flag (matchday 3 in group stage)
  - neutral flag (all World Cup matches at finals are neutral)

Enrichment is stored separately — it never mutates the processed corpus.
Join is by (match_date, home_team_id, away_team_id) with uniqueness
validation.

Only uses information available before kickoff.  No future standings.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional


# ── Stage labels ─────────────────────────────────────────────────────────────

STAGE_GROUP = "group_stage"
STAGE_ROUND_OF_16 = "round_of_16"
STAGE_QUARTER_FINAL = "quarter_final"
STAGE_SEMI_FINAL = "semi_final"
STAGE_THIRD_PLACE = "third_place"
STAGE_FINAL = "final"

KNOCKOUT_STAGES = {
    STAGE_ROUND_OF_16,
    STAGE_QUARTER_FINAL,
    STAGE_SEMI_FINAL,
    STAGE_THIRD_PLACE,
    STAGE_FINAL,
}


def _normalise_stage(raw: str) -> str:
    """Map raw stage strings to canonical labels."""
    s = raw.strip().upper()
    if "GROUP" in s:
        return STAGE_GROUP
    if "LAST 16" in s or "ROUND OF 16" in s or "LAST_16" in s:
        return STAGE_ROUND_OF_16
    if "QUARTER" in s:
        return STAGE_QUARTER_FINAL
    if "SEMI" in s:
        return STAGE_SEMI_FINAL
    if "THIRD" in s or "3RD" in s or "MATCH FOR THIRD" in s:
        return STAGE_THIRD_PLACE
    if "FINAL" in s and "THIRD" not in s and "QUARTER" not in s and "SEMI" not in s:
        return STAGE_FINAL
    return "unknown"


def _extract_year_from_path(path: Path) -> int:
    """Extract World Cup year from file path like matches_2022_openfootball.json."""
    m = re.search(r"(\d{4})", path.stem)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot extract year from {path}")


# ── Stage enrichment entry ─────────────────────────────────────────────────

@dataclass(frozen=True)
class StageEnrichment:
    """Tournament-stage context for a single match, all pre-match available."""
    tournament: str
    year: int
    stage: str          # canonical stage label
    group: str          # "" if not applicable
    matchday: int       # 1,2,3 for group; 4+ for knockout rounds
    is_knockout: bool
    is_final_group_match: bool  # group-stage matchday == 3
    is_neutral: bool            # all finals matches are neutral
    source_file: str


@dataclass
class EnrichmentResult:
    """Result of stage enrichment build."""
    entries: dict[tuple[str, int, int], StageEnrichment]
    unmatched_keys: list[tuple[str, int, int]]
    duplicate_keys: list[tuple[str, int, int]]
    ambiguous_keys: list[tuple[str, list[StageEnrichment]]]
    total_raw: int
    total_enriched: int


def build_stage_enrichment(
    raw_dir: str | Path = "data/raw",
    wc_years: Optional[list[int]] = None,
) -> EnrichmentResult:
    """Build stage enrichment from raw World Cup files.

    Args:
        raw_dir: directory containing matches_YYYY_openfootball.json files
        wc_years: list of WC years to process.  None = [2014, 2018, 2022].

    Returns:
        EnrichmentResult with all entries and diagnostics.
    """
    if wc_years is None:
        wc_years = [2014, 2018, 2022]

    raw_dir = Path(raw_dir)
    entries: dict[tuple[str, int, int], StageEnrichment] = {}
    duplicate_keys: list[tuple[str, int, int]] = []
    ambiguous_keys: list[tuple[str, list[StageEnrichment]]] = []
    total_raw = 0

    for year in wc_years:
        # Find the raw file
        candidates = [
            raw_dir / f"matches_{year}_openfootball.json",
            raw_dir / f"matches_{year}.json",
        ]
        path = None
        for c in candidates:
            if c.exists():
                path = c
                break
        if path is None:
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        raw_matches = data.get("matches", data) if isinstance(data, dict) else data

        for rm in raw_matches:
            total_raw += 1
            d = str(rm.get("date", ""))[:10]
            hid = rm.get("home_team_id")
            aid = rm.get("away_team_id")

            if not d or hid is None or aid is None:
                continue

            stage_raw = rm.get("stage", "")
            stage = _normalise_stage(str(stage_raw))
            group = rm.get("group", "") or ""
            matchday = rm.get("matchday")
            if matchday is None:
                matchday = 0
            try:
                matchday = int(matchday)
            except (ValueError, TypeError):
                matchday = 0

            is_knockout = stage in KNOCKOUT_STAGES
            is_final_group = (stage == STAGE_GROUP and matchday == 3)

            entry = StageEnrichment(
                tournament="FIFA World Cup",
                year=year,
                stage=stage,
                group=str(group),
                matchday=matchday,
                is_knockout=is_knockout,
                is_final_group_match=is_final_group,
                is_neutral=True,  # all WC finals are neutral-venue
                source_file=str(path.name),
            )

            key = (d, int(hid), int(aid))
            if key in entries:
                existing = entries[key]
                if existing.stage != entry.stage:
                    # Ambiguous: two different stage labels for same key
                    ambiguous_keys.append((key, [existing, entry]))
                duplicate_keys.append(key)
            else:
                entries[key] = entry

    return EnrichmentResult(
        entries=entries,
        unmatched_keys=[],
        duplicate_keys=duplicate_keys,
        ambiguous_keys=ambiguous_keys,
        total_raw=total_raw,
        total_enriched=len(entries),
    )


def join_stage_to_matches(
    matches: list,  # list of GoalMatch
    enrichment: dict[tuple[str, int, int], StageEnrichment],
) -> tuple[list[StageEnrichment], list]:
    """Join stage enrichment to processed matches.

    Args:
        matches: list of GoalMatch (processed corpus)
        enrichment: dict from build_stage_enrichment().entries

    Returns:
        (joined, unmatched): list of StageEnrichment for matched matches,
            and list of GoalMatch that had no enrichment entry.
    """
    joined = []
    unmatched = []
    seen_keys: set[tuple[str, int, int]] = set()

    for m in matches:
        key = (m.match_date.isoformat(), m.home_team_id, m.away_team_id)
        if key in enrichment:
            if key in seen_keys:
                # Duplicate — skip but don't fail
                continue
            seen_keys.add(key)
            joined.append(enrichment[key])
        else:
            unmatched.append(m)

    return joined, unmatched


@dataclass(frozen=True)
class StageContext:
    """Minimal pre-match stage features for model input.

    All fields are deterministic and available before kickoff.
    No future standings used.
    """
    is_knockout: bool
    is_final_group_match: bool
    is_qualifier: bool
    is_friendly: bool
    is_world_cup: bool
    is_neutral: bool
    stage: str
    is_group_stage: bool
    is_early_group: bool         # matchday 1 or 2
    is_final_group: bool         # matchday 3


def classify_stage_context(
    match_tournament: str,
    match_neutral: bool,
    stage_enrichment: Optional[StageEnrichment] = None,
) -> StageContext:
    """Build StageContext for a single match.

    Args:
        match_tournament: raw tournament label from processed corpus
        match_neutral: neutral flag from processed corpus
        stage_enrichment: optional StageEnrichment if available (WC finals only)

    Returns:
        StageContext with pre-match features.
    """
    lowered = match_tournament.strip().lower()
    is_wc = "fifa world cup" in lowered and "qualification" not in lowered
    is_qualifier = "qualification" in lowered or "qualifier" in lowered
    is_friendly = "friendly" in lowered

    if stage_enrichment is not None and is_wc:
        return StageContext(
            is_knockout=stage_enrichment.is_knockout,
            is_final_group_match=stage_enrichment.is_final_group_match,
            is_qualifier=False,
            is_friendly=False,
            is_world_cup=True,
            is_neutral=True,  # WC finals always neutral
            stage=stage_enrichment.stage,
            is_group_stage=(stage_enrichment.stage == STAGE_GROUP),
            is_early_group=(stage_enrichment.stage == STAGE_GROUP and stage_enrichment.matchday <= 2),
            is_final_group=stage_enrichment.is_final_group_match,
        )

    return StageContext(
        is_knockout=False,
        is_final_group_match=False,
        is_qualifier=is_qualifier,
        is_friendly=is_friendly,
        is_world_cup=is_wc,
        is_neutral=match_neutral,
        stage="",
        is_group_stage=False,
        is_early_group=False,
        is_final_group=False,
    )
