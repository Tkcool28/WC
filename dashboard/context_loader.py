"""
Manual squad-strength context loader (Phase 4 — display-only).

This module is a *pure presentation layer* — it reads three manually
curated CSV files from ``data/manual/`` and exposes them as plain
dicts/lists. It never touches the probability model, never modifies
``evaluate_match()`` output, and never reads from any network source.

Hard rules followed here:

* No scraping, no API calls — all data is in CSV files committed
  alongside the repo.
* No probability changes — the loader has no opinion on the model.
* All loaders are *tolerant*: a missing or malformed CSV must return
  ``{}`` / ``[]`` / a fully-populated empty context dict, never raise.
* Stdlib + pandas only — no new heavy deps.

Value-tier thresholds (in EUR, squad market value):

* ``elite``  : >= 800,000,000
* ``high``   : >= 400,000,000
* ``mid``    : >= 150,000,000
* ``low``    : <  150,000,000
* ``unknown``: missing

These bands are deliberately stable across re-snapshots so the
dashboard does not flicker on every CSV update.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

# dashboard/context_loader.py -> repo root
_DASHBOARD_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _DASHBOARD_DIR.parent

SQUAD_STRENGTH_PATH = _PROJECT_ROOT / "data" / "manual" / "squad_strength_snapshot.csv"
FIFA_RANKING_PATH = _PROJECT_ROOT / "data" / "manual" / "fifa_ranking_snapshot.csv"
TEAM_NOTES_PATH = _PROJECT_ROOT / "data" / "manual" / "team_context_notes.csv"

# --------------------------------------------------------------------------- #
# Public constants
# --------------------------------------------------------------------------- #

# value_tier thresholds (EUR). Documented in module docstring.
VALUE_TIER_THRESHOLDS = (
    ("elite", 800_000_000),
    ("high", 400_000_000),
    ("mid", 150_000_000),
)
TIER_TO_STYLE = {
    "elite": "success",
    "high": "success",
    "mid": "info",
    "low": "warning",
    "unknown": "info",
}
VALID_NOTE_CATEGORIES = {"injury", "absence", "rotation", "motivation", "other"}

SOURCE_NAME = "Transfermarkt-style manual snapshot (data/manual/*.csv)"


# --------------------------------------------------------------------------- #
# Low-level loaders — tolerant, never raise
# --------------------------------------------------------------------------- #

def _log_missing(path: Path) -> None:
    """One-line stderr warning when a manual CSV is missing.

    Per spec: log to stderr, never raise. We do NOT use streamlit's
    logging here because the loader is also called from non-UI tests.
    """
    print(
        f"[squad_context] manual CSV missing: {path} — returning empty context",
        file=sys.stderr,
    )


def _coerce_int(value: Any) -> int | None:
    """Coerce a pandas/numpy value to a clean int, or None if invalid."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_str(value: Any) -> str:
    """Coerce a pandas/numpy value to a stripped str, or '' if NaN."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def load_squad_strength(path: Path | None = None) -> dict[str, dict]:
    """Return ``{canonical_team_id: {squad_market_value_eur, ...}, ...}``.

    Tolerant: returns ``{}`` if the file is missing or malformed. Logs
    a one-line warning to stderr on a missing file (NOT on malformed
    rows, which are silently skipped to keep the dashboard robust).

    The ``path`` arg is optional; when None, falls back to the
    module-level ``SQUAD_STRENGTH_PATH`` (resolved at *call* time so
    tests can ``monkeypatch.setattr(ctx_mod, 'SQUAD_STRENGTH_PATH', ...)``
    and have the change take effect on the next call).
    """
    if path is None:
        path = SQUAD_STRENGTH_PATH
    if not path.exists():
        _log_missing(path)
        return {}
    try:
        df = pd.read_csv(path)
    except Exception as e:  # malformed CSV
        print(
            f"[squad_context] failed to read {path}: {e!r} — returning empty",
            file=sys.stderr,
        )
        return {}

    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        cid = _coerce_str(row.get("canonical_team_id"))
        if not cid:
            continue
        out[cid] = {
            "squad_market_value_eur": _coerce_int(row.get("squad_market_value_eur")),
            "avg_player_value_eur": _coerce_int(row.get("avg_player_value_eur")),
            "top_5_player_value_eur": _coerce_int(row.get("top_5_player_value_eur")),
            "most_valuable_player": _coerce_str(row.get("most_valuable_player")),
            "source_url": _coerce_str(row.get("source_url")),
            "snapshot_date": _coerce_str(row.get("snapshot_date")),
        }
    return out


def load_fifa_ranking(path: Path | None = None) -> dict[str, dict]:
    """Return ``{canonical_team_id: {fifa_rank, fifa_points, snapshot_date, source_url}, ...}``.

    Tolerant: returns ``{}`` on missing/malformed file. ``path`` is
    optional; when None, falls back to the module-level
    ``FIFA_RANKING_PATH`` (resolved at call time so tests can patch).
    """
    if path is None:
        path = FIFA_RANKING_PATH
    if not path.exists():
        _log_missing(path)
        return {}
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(
            f"[squad_context] failed to read {path}: {e!r} — returning empty",
            file=sys.stderr,
        )
        return {}

    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        cid = _coerce_str(row.get("canonical_team_id"))
        if not cid:
            continue
        out[cid] = {
            "fifa_rank": _coerce_int(row.get("fifa_rank")),
            "fifa_points": _coerce_int(row.get("fifa_points")),
            "snapshot_date": _coerce_str(row.get("snapshot_date")),
            "source_url": _coerce_str(row.get("source_url")),
        }
    return out


def load_team_notes(path: Path | None = None) -> dict[str, list[dict]]:
    """Return ``{canonical_team_id: [ {snapshot_date, note_category, note_text}, ... ], ...}``.

    Tolerant: returns ``{}`` on missing/malformed file. Malformed
    ``note_category`` values are kept as-is (the dashboard renders
    whatever the curator wrote) rather than dropped, to avoid silent
    data loss in production. ``path`` is optional; when None, falls
    back to the module-level ``TEAM_NOTES_PATH`` (resolved at call
    time so tests can patch).
    """
    if path is None:
        path = TEAM_NOTES_PATH
    if not path.exists():
        _log_missing(path)
        return {}
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(
            f"[squad_context] failed to read {path}: {e!r} — returning empty",
            file=sys.stderr,
        )
        return {}

    out: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        cid = _coerce_str(row.get("canonical_team_id"))
        if not cid:
            continue
        out.setdefault(cid, []).append({
            "snapshot_date": _coerce_str(row.get("snapshot_date")),
            "note_category": _coerce_str(row.get("note_category")),
            "note_text": _coerce_str(row.get("note_text")),
        })
    return out


# --------------------------------------------------------------------------- #
# Derived helpers
# --------------------------------------------------------------------------- #

def value_tier(squad_value_eur) -> str:
    """Map a squad market value (EUR) to ``elite/high/mid/low/unknown``.

    ``unknown`` covers None / non-numeric / zero / negative inputs — we treat
    those as "missing" rather than as a real low-tier value, so the dashboard
    never accidentally labels a malformed input as a budget squad.
    """
    if squad_value_eur is None:
        return "unknown"
    # bool is a subclass of int in Python, so True/False would otherwise
    # sneak through int() — reject them explicitly.
    if isinstance(squad_value_eur, bool):
        return "unknown"
    try:
        v = float(squad_value_eur)
    except (TypeError, ValueError):
        return "unknown"
    if v <= 0:
        return "unknown"
    for name, threshold in VALUE_TIER_THRESHOLDS:
        if v >= threshold:
            return name
    return "low"


def gap_vs_opponent_pct(
    team_value,
    opp_value,
) -> float | None:
    """Return ``(team - opp) / opp * 100`` if both are positive, else ``None``.

    Missing inputs (None) and non-positive inputs (zero or negative) on
    *either* side are treated as missing, so we never divide by zero or
    emit a meaningful percentage from garbage data.
    """
    if team_value is None or opp_value is None:
        return None
    if float(team_value) <= 0 or float(opp_value) <= 0:
        return None
    return (float(team_value) - float(opp_value)) / float(opp_value) * 100.0


def get_team_context(canonical_id: str) -> dict[str, Any]:
    """Return a fully-populated context dict for one canonical team id.

    Always returns the same key set, even when the team is unknown or
    every source CSV is missing. This is the *only* function the
    dashboard should call for per-team context.

    Keys (every one always present):
        squad_value        : int | None
        avg_value          : int | None
        top5_value         : int | None
        mvp                : str
        value_tier         : 'elite' | 'high' | 'mid' | 'low' | 'unknown'
        fifa_rank          : int | None
        fifa_points        : int | None
        notes              : list[dict]  (may be empty)
        source             : str
        snapshot_date      : str
        gap_vs_opponent_pct: float | None

    Special case — if every source CSV is missing we return ``{}``
    so callers can detect "no manual data is configured" and
    render their own fallback copy.
    """
    canonical_id = (canonical_id or "").strip()
    if not canonical_id:
        # Return a fully-populated empty context (no team to look up)
        return _empty_context()

    squad_map = load_squad_strength()
    rank_map = load_fifa_ranking()
    notes_map = load_team_notes()

    # No manual data at all (e.g. CSVs deleted) → return an empty
    # dict so the caller can show a single "no manual context" line
    # instead of a panel full of "Unknown" badges.
    if not squad_map and not rank_map and not notes_map:
        return {}

    squad_row = squad_map.get(canonical_id) or {}
    rank_row = rank_map.get(canonical_id) or {}
    notes = list(notes_map.get(canonical_id) or [])

    squad_value = squad_row.get("squad_market_value_eur")
    return {
        "squad_value": squad_value,
        "avg_value": squad_row.get("avg_player_value_eur"),
        "top5_value": squad_row.get("top_5_player_value_eur"),
        "mvp": squad_row.get("most_valuable_player") or "",
        "value_tier": value_tier(squad_value),
        "fifa_rank": rank_row.get("fifa_rank"),
        "fifa_points": rank_row.get("fifa_points"),
        "notes": notes,
        "source": SOURCE_NAME,
        "snapshot_date": (
            squad_row.get("snapshot_date")
            or rank_row.get("snapshot_date")
            or ""
        ),
        "gap_vs_opponent_pct": None,  # filled in by get_match_context
    }


def _empty_context() -> dict[str, Any]:
    return {
        "squad_value": None,
        "avg_value": None,
        "top5_value": None,
        "mvp": "",
        "value_tier": "unknown",
        "fifa_rank": None,
        "fifa_points": None,
        "notes": [],
        "source": SOURCE_NAME,
        "snapshot_date": "",
        "gap_vs_opponent_pct": None,
    }


# --------------------------------------------------------------------------- #
# Match-level aggregator (both teams at once + gap)
# --------------------------------------------------------------------------- #

def get_match_context(home_canonical_id: str, away_canonical_id: str) -> dict[str, Any]:
    """Return ``{'home': context, 'away': context, 'gap': {home_pct, away_pct}}``.

    The two gap values are symmetric: ``away_pct == -home_pct`` when
    both sides have a value. Either side is ``None`` when one of the
    teams has no squad value. This is the function the dashboard
    panel calls.
    """
    home = get_team_context(home_canonical_id)
    away = get_team_context(away_canonical_id)
    hv, av = home.get("squad_value"), away.get("squad_value")
    home_gap = gap_vs_opponent_pct(hv, av)  # home vs away
    away_gap = gap_vs_opponent_pct(av, hv)  # away vs home
    return {
        "home": home,
        "away": away,
        "gap": {
            "home_pct": home_gap,
            "away_pct": away_gap,
        },
    }


# --------------------------------------------------------------------------- #
# Pure-render helpers (testable in isolation, no Streamlit import)
# --------------------------------------------------------------------------- #

def format_eur(value: int | None) -> str:
    """Format a EUR value as ``€1.2B`` / ``€850M`` / ``Unknown``.

    Display-only; never raises. ``None`` -> ``"Unknown"``.

    Threshold policy (chosen so typical squad market values read as
    a single human-friendly number):

    * ``>= 1_000_000_000`` → ``€X.YZB`` (one decimal)
    * ``>= 100_000_000``   → ``€X.YZB`` (one decimal — so 850M reads as €0.85B)
    * ``>= 1_000_000``     → ``€X M`` (no decimals)
    * otherwise           → ``€X,XXX`` (with thousands separators)
    """
    if value is None:
        return "Unknown"
    v = float(value)
    if v >= 100_000_000:
        return f"€{v / 1_000_000_000:.2f}B"
    if v >= 1_000_000:
        return f"€{v / 1_000_000:.0f}M"
    return f"€{int(v):,}"


def format_gap(pct: float | None) -> str:
    """Format a gap as ``▲ +100.0%`` / ``▼ -33.3%`` / ``—``.

    Used for the arrow+sign display in the dashboard panel.
    """
    if pct is None:
        return "—"
    if pct > 0:
        return f"▲ +{pct:.1f}%"
    if pct < 0:
        return f"▼ {pct:.1f}%"
    return "± 0.0%"


def escape_note_text(text: str) -> str:
    """HTML-escape a free-text note so it is safe to drop in markdown.

    The dashboard always calls this on the raw note text before
    rendering — never trust a CSV-supplied string.
    """
    return html.escape(text or "")


def render_notes_bullets(notes: list[dict]) -> str:
    """Build a markdown bullet list of notes, with the category as a badge.

    The text is HTML-escaped. Returns ``""`` when the list is empty.
    """
    if not notes:
        return ""
    lines: list[str] = []
    for n in notes:
        cat = escape_note_text(n.get("note_category") or "other").upper()
        text = escape_note_text(n.get("note_text") or "")
        lines.append(f"- **[{cat}]** {text}")
    return "\n".join(lines)


__all__ = [
    "SOURCE_NAME",
    "TIER_TO_STYLE",
    "VALUE_TIER_THRESHOLDS",
    "VALID_NOTE_CATEGORIES",
    "SQUAD_STRENGTH_PATH",
    "FIFA_RANKING_PATH",
    "TEAM_NOTES_PATH",
    "escape_note_text",
    "format_eur",
    "format_gap",
    "gap_vs_opponent_pct",
    "get_match_context",
    "get_team_context",
    "load_fifa_ranking",
    "load_squad_strength",
    "load_team_notes",
    "render_notes_bullets",
    "value_tier",
]
