"""Read-only loader for the cached 2026 World Cup fixture list.

The dashboard auto-fills today's matchups from `data/raw/matches_2026.json`
(generated out-of-band by `scripts/fetch_live_2026.py`). This module exposes
pure functions that filter the cache by date / status / team-name presence
so the Streamlit layer can stay thin and the filtering logic is unit-tested.

Filtering rules (verified against the real cache):
  - Status is anything other than FINISHED (TIMED, IN_PLAY, PAUSED, …)
  - `home_team_name` AND `away_team_name` are non-null and non-empty
    (drops TBD knockout placeholders)
  - `result` is null/empty (a finished game is filtered by status, but
    this is a belt-and-braces guard in case a future cache has weird data)
  - Match is on the requested calendar date in Mountain Time (America/Denver).
    The cache stores kickoffs in UTC; we convert to MT and compare against
    the date picker value. This means a game at 2026-06-17T01:00:00Z
    (7 PM MDT on 2026-06-16) is correctly listed under "2026-06-16".

The cache file is treated as a *snapshot*. If it's missing or malformed
the loader returns an empty list rather than raising — the dashboard
renders a "no games found" message in that case.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date as _date
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

# dashboard/ → ../data/raw/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_PATH = _PROJECT_ROOT / "data" / "raw" / "matches_2026.json"
CACHE_FILENAME = "matches_2026.json"

# All "calendar day" comparisons are done in Mountain Time because Todd (the
# sole user) lives in America/Denver. If we ever have an East Coast user
# we'd want a per-user tz — for now a hard-coded MT is honest and simple.
# Note: ZoneInfo resolves DST automatically (MDT = UTC-6, MST = UTC-7), so a
# match at 2026-11-02T01:00:00Z after the DST end (Sun 2026-11-01 02:00)
# correctly lands on 2026-11-01 18:00 MT (MST, not MDT).
USER_TZ = ZoneInfo("America/Denver")

# Statuses that mean "the game hasn't produced a final result yet".
# FINISHED is the only status that implies a final result in this dataset;
# IN_PLAY / PAUSED can also have no result, but listing them all as
# "unplayed" here would be wrong. We keep the rule narrow and explicit:
# anything that is NOT FINISHED counts as unplayed for filtering purposes.
FINISHED_STATUSES = frozenset({"FINISHED"})


# --------------------------------------------------------------------------- #
# Output type
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class UnplayedMatch:
    """A single fixture in the shape the dashboard wants to render.

    The fields are deliberately small and pre-flattened so the Streamlit
    layer doesn't have to deal with the raw API shape (which has
    home_team_id, status, group, … that the auto-populate flow doesn't need).
    """
    match_id: int
    kickoff_iso: str        # full ISO (UTC) with time, e.g. "2026-06-16T19:00:00Z"
    match_date_iso: str     # date only, e.g. "2026-06-16" — *Mountain Time* date
    home_team_name: str
    away_team_name: str
    home_team_id: int | None = None
    away_team_id: int | None = None
    group: str = ""
    stage: str = ""
    # 1, 2, or 3 for group-stage matches; None for knockout / unknown.
    # Plumbed through the dashboard so the renderer can attach the
    # appropriate group-context warnings (Phase 3).
    matchday: int | None = None
    status: str = "TIMED"
    kickoff_mt_iso: str = ""  # full ISO with MT offset, e.g. "2026-06-16T13:00:00-06:00"
    # Canonical team identity (3-letter code like "ARG", "ALG", "USA") used to
    # route internal joins through a single stable key. Populated from
    # `soccer_ev_model.team_identity`; None means the loader could not
    # resolve a canonical id for this team.
    canonical_home_id: str | None = None
    canonical_away_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for st.session_state / json.dumps round-trips."""
        return {
            "match_id": self.match_id,
            "kickoff_iso": self.kickoff_iso,
            "match_date_iso": self.match_date_iso,
            "home_team_name": self.home_team_name,
            "away_team_name": self.away_team_name,
            "home_team_id": self.home_team_id,
            "away_team_id": self.away_team_id,
            "group": self.group,
            "stage": self.stage,
            "matchday": self.matchday,
            "status": self.status,
            "kickoff_mt_iso": self.kickoff_mt_iso,
            "canonical_home_id": self.canonical_home_id,
            "canonical_away_id": self.canonical_away_id,
        }


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def _normalize_date(date_input: str | _date) -> str:
    """Accept a 'YYYY-MM-DD' string, a full ISO timestamp, or a date object.

    Returns just the 'YYYY-MM-DD' portion so we can match against the MT-date
    we compute from each match's kickoff.
    """
    if isinstance(date_input, _date):
        return date_input.isoformat()
    s = str(date_input).strip()
    # "2026-06-16T19:00:00Z" -> "2026-06-16"
    return s[:10]


def _parse_utc_iso(raw_iso: str) -> datetime | None:
    """Parse a UTC ISO-8601 string like '2026-06-17T01:00:00Z' (or '+00:00').

    Returns None for empty / unparseable input so callers can skip the row.
    """
    if not raw_iso:
        return None
    s = raw_iso.strip()
    if not s:
        return None
    # Trailing 'Z' is RFC 3339 shorthand for +00:00; Python's fromisoformat
    # only learned about 'Z' in 3.11, but the project targets 3.9+ in CI, so
    # normalize defensively.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    # If naive, assume UTC (shouldn't happen with our cache, but be safe).
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(ZoneInfo("UTC"))


def _kickoff_mt_date(raw_iso: str) -> str:
    """Return the YYYY-MM-DD in Mountain Time for a UTC ISO kickoff string.

    Empty / unparseable input returns '' so the caller can skip the match.
    DST is handled automatically by ZoneInfo: 2026-06-17T01:00:00Z is
    2026-06-16 in MDT (UTC-6), and after the fall-back on 2026-11-01 a
    2026-11-02T01:00:00Z kickoff is 2026-11-01 in MST (UTC-7).
    """
    dt = _parse_utc_iso(raw_iso)
    if dt is None:
        return ""
    return dt.astimezone(USER_TZ).date().isoformat()


def _kickoff_mt_iso(raw_iso: str) -> str:
    """Return the full ISO timestamp in Mountain Time (with offset).

    Example: '2026-06-17T01:00:00Z' -> '2026-06-16T19:00:00-06:00'.
    """
    dt = _parse_utc_iso(raw_iso)
    if dt is None:
        return ""
    return dt.astimezone(USER_TZ).isoformat()


def _is_blank(value: Any) -> bool:
    """True for None or empty/whitespace string."""
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def load_matches_cache(cache_path: Path | None = None) -> dict[str, Any]:
    """Read the matches_2026.json file, returning a safe empty dict on error.

    Never raises — the dashboard needs to survive a missing/malformed cache
    and just show "no games found" instead of crashing.
    """
    path = cache_path or DEFAULT_CACHE_PATH
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        log.warning("2026 WC cache not found at %s", path)
        return {"matches": [], "year": 2026}
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("2026 WC cache at %s is unreadable: %s", path, exc)
        return {"matches": [], "year": 2026}


def get_unplayed_matches(
    date_input: str | _date,
    cache_path: Path | None = None,
) -> list[UnplayedMatch]:
    """Return all unplayed games on `date_input` (Mountain Time), with both teams known.

    The returned list is sorted by MT kickoff time (earliest first) so the
    dashboard can render them in chronological order from the user's POV.
    """
    target_date = _normalize_date(date_input)
    payload = load_matches_cache(cache_path)
    raw_matches = payload.get("matches") or []

    out: list[UnplayedMatch] = []
    for m in raw_matches:
        # 1) Skip finished games (they have a result, even if on the same date)
        status = (m.get("status") or "").upper()
        if status in FINISHED_STATUSES:
            continue

        # 2) Skip anything where result is already set (belt-and-braces)
        if not _is_blank(m.get("result")):
            continue

        # 3) Match the date in Mountain Time. A kickoff at 2026-06-17T01:00:00Z
        #    is 7 PM MDT on 2026-06-16, so it belongs to the 16th in the user's
        #    calendar even though the UTC date prefix reads 17th.
        kickoff = m.get("date") or ""
        mt_date = _kickoff_mt_date(kickoff)
        if mt_date != target_date:
            continue

        # 4) Both team names must be present (drops TBD knockouts)
        home_name = m.get("home_team_name")
        away_name = m.get("away_team_name")
        if _is_blank(home_name) or _is_blank(away_name):
            continue

        # 5) Resolve canonical team ids (3-letter codes) via the identity
        #    layer. Resolved lazily so the loader can be used in tests
        #    without the registry file present (the import is cheap and
        #    the JSON read happens on first call).
        from soccer_ev_model.team_identity import resolve_team
        home_res = resolve_team(
            football_data_id=m.get("home_team_id"),
            name=str(home_name).strip(),
        )
        away_res = resolve_team(
            football_data_id=m.get("away_team_id"),
            name=str(away_name).strip(),
        )

        out.append(UnplayedMatch(
            match_id=m.get("id", 0),
            kickoff_iso=kickoff,
            match_date_iso=target_date,
            home_team_name=str(home_name).strip(),
            away_team_name=str(away_name).strip(),
            home_team_id=m.get("home_team_id"),
            away_team_id=m.get("away_team_id"),
            group=m.get("group", "") or "",
            stage=m.get("stage", "") or "",
            # 1, 2, or 3 for group-stage matches; None for knockout
            # rows (where the field is null in the cache).  The
            # dashboard uses this for the Phase 3 group-context
            # warnings — see prediction_summary.matchday_label.
            matchday=m.get("matchday") if isinstance(m.get("matchday"), int) else None,
            status=status or "TIMED",
            kickoff_mt_iso=_kickoff_mt_iso(kickoff),
            canonical_home_id=home_res.get("canonical_id"),
            canonical_away_id=away_res.get("canonical_id"),
        ))

    # Sort by MT kickoff so the user's eye sees them chronologically. Falls
    # back to the raw UTC string for any row whose MT conversion failed.
    out.sort(key=lambda u: u.kickoff_mt_iso or u.kickoff_iso)
    return out


def list_dates_with_unplayed(
    cache_path: Path | None = None,
) -> list[str]:
    """Return a sorted list of Mountain Time dates that have at least one unplayed game.

    Useful for the dashboard to populate a date-picker dropdown of "days
    with fixtures" — the user can then click any one to auto-load. Each
    date is the MT date (so a game at 2026-06-17T01:00:00Z appears under
    2026-06-16, not 2026-06-17).
    """
    payload = load_matches_cache(cache_path)
    raw_matches = payload.get("matches") or []

    dates: set[str] = set()
    for m in raw_matches:
        status = (m.get("status") or "").upper()
        if status in FINISHED_STATUSES:
            continue
        if not _is_blank(m.get("result")):
            continue
        if _is_blank(m.get("home_team_name")) or _is_blank(m.get("away_team_name")):
            continue
        d = _kickoff_mt_date(m.get("date") or "")
        if d:
            dates.add(d)
    return sorted(dates)
