"""Prior injection for the independent goal model.

Provides leak-safe, transparent prior interfaces for:
  - FIFA ranking priors (historical snapshots)
  - Squad-strength priors (historical snapshots)
  - Elo-based priors (already handled via EloPoissonModel)

Eligibility rule: a prior may enter a historical prediction only if its
timestamp is strictly earlier than the match date, the team identity maps
unambiguously, and the source value represents information available at that
date. Missing values always have an explicit fallback.

All transformations are deterministic and inspectable.  No black-box ML.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

_EPS = 1e-10

# ── Source inventory ────────────────────────────────────────────────────────

@dataclass
class SourceInventory:
    """Describes a single external data source for priors."""
    name: str
    path: str
    schema: list[str]
    date_coverage: tuple[str, str]
    team_count: int
    update_frequency: str
    has_historical_snapshots: bool
    pre_match_safe: bool
    notes: list[str]


def inventory_sources(
    data_root: str | Path = "data",
) -> list[SourceInventory]:
    """Inventory all available prio sources.

    Returns a list of SourceInventory describing what exists,
    what's usable, and what's leak-safe.
    """
    root = Path(data_root)
    sources = []

    # FIFA ranking snapshot
    fifa_path = root / "manual" / "fifa_ranking_snapshot.csv"
    if fifa_path.exists():
        rows = list(csv.DictReader(fifa_path.open()))
        dates = sorted(set(r["snapshot_date"] for r in rows))
        teams = set(r["canonical_team_id"] for r in rows)
        sources.append(SourceInventory(
            name="fifa_ranking_snapshot",
            path=str(fifa_path),
            schema=list(rows[0].keys()) if rows else [],
            date_coverage=(dates[0], dates[-1]) if dates else ("", ""),
            team_count=len(teams),
            update_frequency="single_snapshot" if len(dates) <= 1 else "periodic",
            has_historical_snapshots=len(dates) > 1,
            pre_match_safe=False,  # single current snapshot — not leak-safe for history
            notes=[
                f"Single snapshot date: {dates[0] if dates else 'none'}",
                f"Teams: {len(teams)}",
                "Only current values exist — NOT usable in historical backtests.",
                "May be used as production-only context feature.",
            ],
        ))

    # Squad-strength snapshot
    squad_path = root / "manual" / "squad_strength_snapshot.csv"
    if squad_path.exists():
        rows = list(csv.DictReader(squad_path.open()))
        dates = sorted(set(r["snapshot_date"] for r in rows))
        teams = set(r["canonical_team_id"] for r in rows)
        sources.append(SourceInventory(
            name="squad_strength_snapshot",
            path=str(squad_path),
            schema=list(rows[0].keys()) if rows else [],
            date_coverage=(dates[0], dates[-1]) if dates else ("", ""),
            team_count=len(teams),
            update_frequency="single_snapshot" if len(dates) <= 1 else "periodic",
            has_historical_snapshots=len(dates) > 1,
            pre_match_safe=False,  # single current snapshot
            notes=[
                f"Single snapshot date: {dates[0] if dates else 'none'}",
                f"Teams: {len(teams)}",
                "Only current values exist — NOT usable in historical backtests.",
                "May be used as production-only context feature.",
            ],
        ))

    # Team context notes
    notes_path = root / "manual" / "team_context_notes.csv"
    if notes_path.exists():
        rows = list(csv.DictReader(notes_path.open()))
        dates = sorted(set(r["snapshot_date"] for r in rows))
        sources.append(SourceInventory(
            name="team_context_notes",
            path=str(notes_path),
            schema=list(rows[0].keys()) if rows else [],
            date_coverage=(dates[0], dates[-1]) if dates else ("", ""),
            team_count=len(set(r["canonical_team_id"] for r in rows)),
            update_frequency="manual",
            has_historical_snapshots=False,
            pre_match_safe=False,
            notes=[
                "Free-text notes — not convertible to safe numerical adjustments.",
                "Production-only warnings/flags only.",
            ],
        ))

    # Elo ratings — historical snapshots
    elo_path = root / "raw" / "elo_ratings.json"
    if elo_path.exists():
        elo = json.loads(elo_path.read_text())
        years = elo.get("years_covered", [])
        sources.append(SourceInventory(
            name="elo_ratings",
            path=str(elo_path),
            schema=["source", "fetched_at", "years_covered", "teams"],
            date_coverage=(str(years[0]), str(years[-1])) if years else ("", ""),
            team_count=len(elo.get("teams", {})),
            update_frequency="annual",
            has_historical_snapshots=True,
            pre_match_safe=True,  # elo_at() uses strict-less-than date filter
            notes=[
                f"Years: {years[0]}-{years[-1]}" if years else "no years",
                "Pre-match values via elo_at() with strict date filtering.",
                "Safe for historical backtesting.",
            ],
        ))

    return sources


# ── Team identity bridge ────────────────────────────────────────────────────

def load_team_code_to_id(
    identity_path: str | Path = "data/team_identity.json",
) -> dict[str, int]:
    """Load the 3-letter code -> corpus_id mapping from team_identity.json."""
    data = json.loads(Path(identity_path).read_text(encoding="utf-8"))
    return {
        code: v["corpus_id"]
        for code, v in data.items()
        if code != "_meta" and "corpus_id" in v
    }


# ── FIFA ranking prior ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class FifaRankingPrior:
    """Loadable FIFA ranking data with leak-safe date filtering.

    For historical backtesting: requires multi-date snapshots.
    For production: single current snapshot is fine.
    """

    # code -> [(date, rank, points)]
    snapshots: dict[str, list[tuple[date, int, float]]]
    available_dates: list[str]
    team_code_to_corpus_id: dict[str, int]

    @classmethod
    def load(
        cls,
        path: str | Path = "data/manual/fifa_ranking_snapshot.csv",
        identity_path: str | Path = "data/team_identity.json",
    ) -> "FifaRankingPrior":
        code_to_id = load_team_code_to_id(identity_path)
        snaps: dict[str, list[tuple[date, int, float]]] = {}
        dates_set: set[str] = set()

        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                code = row["canonical_team_id"].strip()
                d = row["snapshot_date"].strip()
                dates_set.add(d)
                rank = int(row["fifa_rank"])
                points = float(row["fifa_points"])
                snaps.setdefault(code, []).append((date.fromisoformat(d), rank, points))

        return cls(
            snapshots=snaps,
            available_dates=sorted(dates_set),
            team_code_to_corpus_id=code_to_id,
        )

    def lookup(
        self,
        team_code: str,
        match_date: date,
    ) -> tuple[Optional[float], Optional[int], bool]:
        """Return (points, rank, missing) for a team prior to match_date.

        Returns (None, None, True) if no valid prior exists.
        """
        entries = self.snapshots.get(team_code)
        if not entries:
            return None, None, True

        best = None
        for d, rank, points in entries:
            if d < match_date:
                best = (points, rank)
            else:
                break  # entries are sorted ascending

        if best is None:
            return None, None, True
        return best[0], best[1], False

    @property
    def has_historical_snapshots(self) -> bool:
        return len(self.available_dates) > 1

    @property
    def is_backtestable(self) -> bool:
        return self.has_historical_snapshots


# ── Squad-strength prior ────────────────────────────────────────────────────

@dataclass(frozen=True)
class SquadStrengthPrior:
    """Loadable squad-strength data with leak-safe date filtering.

    For historical backtesting: requires multi-date snapshots.
    For production: single current snapshot is fine as optional context.
    """

    # code -> [(date, market_value_eur)]
    snapshots: dict[str, list[tuple[date, float]]]
    available_dates: list[str]
    team_code_to_corpus_id: dict[str, int]

    @classmethod
    def load(
        cls,
        path: str | Path = "data/manual/squad_strength_snapshot.csv",
        identity_path: str | Path = "data/team_identity.json",
    ) -> "SquadStrengthPrior":
        code_to_id = load_team_code_to_id(identity_path)
        snaps: dict[str, list[tuple[date, float]]] = {}
        dates_set: set[str] = set()

        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                code = row["canonical_team_id"].strip()
                d = row["snapshot_date"].strip()
                dates_set.add(d)
                value = float(row["squad_market_value_eur"])
                snaps.setdefault(code, []).append((date.fromisoformat(d), value))

        return cls(
            snapshots=snaps,
            available_dates=sorted(dates_set),
            team_code_to_corpus_id=code_to_id,
        )

    def lookup(
        self,
        team_code: str,
        match_date: date,
    ) -> tuple[Optional[float], bool]:
        """Return (market_value, missing) for a team prior to match_date."""
        entries = self.snapshots.get(team_code)
        if not entries:
            return None, True

        best = None
        for d, value in entries:
            if d < match_date:
                best = value
            else:
                break

        if best is None:
            return None, True
        return best, False

    @property
    def has_historical_snapshots(self) -> bool:
        return len(self.available_dates) > 1

    @property
    def is_backtestable(self) -> bool:
        return self.has_historical_snapshots


# ── Prior application helpers ───────────────────────────────────────────────

def fifa_points_to_attack_shift(
    home_points: float,
    away_points: float,
    weight: float,
) -> float:
    """Convert FIFA ranking point difference into an additive log-rate shift.

    shift = weight * log((home_points + eps) / (away_points + eps))

    The shift is added to home attack and subtracted from away attack
    (symmetric treatment).  Weight is small (0.0–0.4 range tested).

    At weight == 0.0 the shift is always 0.0 (no prior effect).
    """
    if weight == 0.0:
        return 0.0
    return weight * math.log((home_points + _EPS) / (away_points + _EPS))


def squad_value_to_attack_shift(
    home_value: float,
    away_value: float,
    weight: float,
) -> float:
    """Convert squad market-value ratio into an additive log-rate shift.

    shift = weight * log((home_value + eps) / (away_value + eps))

    The shift is added to home attack and subtracted from away attack
    (symmetric treatment).
    """
    if weight == 0.0:
        return 0.0
    return weight * math.log((home_value + _EPS) / (away_value + _EPS))


# ── Production-only context flags ───────────────────────────────────────────

@dataclass
class ContextFlags:
    """Non-numeric context flags for production predictions.

    These flags do not alter model parameters.  They are metadata
    attached to a prediction for human review.
    """
    home_injury_warning: bool = False
    away_injury_warning: bool = False
    home_rotation_warning: bool = False
    away_rotation_warning: bool = False
    home_absence_warning: bool = False
    away_absence_warning: bool = False
    home_motivation_flag: str = ""
    away_motivation_flag: str = ""
    notes: list[str] = field(default_factory=list)


def load_context_flags(
    path: str | Path = "data/manual/team_context_notes.csv",
    snapshot_date: Optional[date] = None,
) -> dict[str, ContextFlags]:
    """Load structured context notes into per-team flags.

    Args:
        path: path to team_context_notes.csv
        snapshot_date: only include notes on or before this date.
            None means include all (production use with current snapshot).

    Returns:
        {team_code: ContextFlags}
    """
    flags: dict[str, ContextFlags] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = row["canonical_team_id"].strip()
            d = date.fromisoformat(row["snapshot_date"])
            if snapshot_date is not None and d > snapshot_date:
                continue
            cat = row["note_category"].strip()
            text = row["note_text"].strip()

            if code not in flags:
                flags[code] = ContextFlags()

            if cat == "injury":
                flags[code].home_injury_warning = True
                flags[code].notes.append(f"Injury: {text}")
            elif cat == "rotation":
                flags[code].home_rotation_warning = True
                flags[code].notes.append(f"Rotation: {text}")
            elif cat == "absence":
                flags[code].home_absence_warning = True
                flags[code].notes.append(f"Absence: {text}")
            elif cat == "motivation":
                flags[code].home_motivation_flag = text
                flags[code].notes.append(f"Motivation: {text}")
            else:
                flags[code].notes.append(f"{cat}: {text}")

    return flags
