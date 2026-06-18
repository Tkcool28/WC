"""Human-readable text formatters for the Predictions view.

Converts internal codes / ISO timestamps into strings a casual user can read.
The functions in this module are pure: no I/O, no Streamlit, no model calls.
They are designed to be unit-tested in isolation and to be safe to call
from any renderer (mobile Predictions view, per-game Analysis view, custom
matchup expander, etc.).

Phase 3 of the dashboard rearchitecture introduces this module. The Phase 6
styles pass will reuse the same string formatters; for now they are wired
into ``dashboard.prediction_card`` (the single-card renderer) and the
``_render_predictions_view`` flow in ``dashboard.app``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def format_kickoff(iso_dt: Optional[str]) -> str:
    """Format an ISO timestamp as a short, human-readable kickoff string.

    Examples
    --------
    >>> format_kickoff("2026-06-17T17:00:00Z")
    'Jun 17 · 5:00 PM UTC'
    >>> format_kickoff("2026-06-17T22:30:00+00:00")
    'Jun 17 · 10:30 PM UTC'

    Returns
    -------
    str
        A short kickoff string, or the literal ``"TBD"`` if the input is
        ``None`` / empty / unparseable.
    """
    if not iso_dt:
        return "TBD"
    try:
        # Tolerate trailing Z (RFC 3339) and naive ISO strings.
        s = str(iso_dt).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        # ``%-I`` is a Linux strftime extension that strips the leading
        # zero from the 12-hour hour.  The project tests run on Linux, so
        # this is fine; if Windows compat ever matters, fall back to
        # ``dt.strftime("%I:%M %p").lstrip("0")``.
        return dt.strftime("%b %-d · %-I:%M %p UTC")
    except (ValueError, TypeError, AttributeError):
        return "TBD"


def format_group_label(group_code: Optional[str]) -> str:
    """Convert a raw group code into a human-readable label.

    Examples
    --------
    >>> format_group_label("GROUP_K")
    'Group K'
    >>> format_group_label("K")
    'Group K'
    >>> format_group_label(None)
    'TBD'
    >>> format_group_label("")
    'TBD'
    """
    if not group_code:
        return "TBD"
    s = str(group_code).strip().upper()
    if s.startswith("GROUP_"):
        s = s[len("GROUP_"):]
    if not s:
        return "TBD"
    return f"Group {s.title()}"


def format_matchday_label(
    stage: Optional[str], matchday: Optional[int] = None
) -> str:
    """Convert a stage / matchday pair into a human-readable label.

    Examples
    --------
    >>> format_matchday_label("GROUP_STAGE", 1)
    'Group Stage · Matchday 1'
    >>> format_matchday_label("KNOCKOUT", None)
    'Knockout'
    >>> format_matchday_label(None, 1)
    'TBD'
    """
    if not stage:
        return "TBD"
    s = str(stage).strip().replace("_", " ").title()
    if matchday is not None:
        return f"{s} · Matchday {matchday}"
    return s


def format_team_matchup(home: Optional[str], away: Optional[str]) -> str:
    """Format a team matchup as a short human-readable string.

    Examples
    --------
    >>> format_team_matchup("England", "Croatia")
    'England vs Croatia'
    >>> format_team_matchup("", "Croatia")
    'TBD vs Croatia'
    """
    h = (home or "TBD").strip() or "TBD"
    a = (away or "TBD").strip() or "TBD"
    return f"{h} vs {a}"
