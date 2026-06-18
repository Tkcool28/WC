"""Tests for the dashboard context-cards helpers.

These tests cover the three dashboard-polish features:

* Tournament Snapshot card (build_tournament_snapshot)
* Smart Date Default (pick_smart_default_date)
* Highest Model Confidence card (highest_model_confidence)

The helpers live in :mod:`dashboard.context_cards` and are pure: no
Streamlit, no disk, no model calls.  The tests therefore run without a
Streamlit ScriptRun and don't need ``streamlit.testing.v1.AppTest``.

Renderer smoke tests use the small ``_capture`` helper at the bottom of
this file, which monkey-patches ``streamlit`` with a fake that records
all ``st.markdown`` / ``st.caption`` / ``st.markdown(... unsafe_allow_html=True)``
calls.  This keeps the renderers thin (they ARE just markdown) while
still verifying the user-visible strings the brief calls out.
"""
from __future__ import annotations

import sys
import types
from datetime import date
from typing import Any

import pytest


# --------------------------------------------------------------------------- #
# Module import (skip if context_cards not yet implemented)
# --------------------------------------------------------------------------- #
def _import_context_cards():
    """Import the module under test lazily so individual tests can
    use ``pytest.importorskip`` semantics without duplicating boilerplate."""
    from dashboard import context_cards  # noqa: WPS433 — intentional runtime import
    return context_cards


# --------------------------------------------------------------------------- #
# Fake Streamlit for renderer smoke tests
# --------------------------------------------------------------------------- #
class _FakeStreamlit:
    """Minimal streamlit stand-in that records every render call.

    The context-card renderers only use a small surface area
    (``st.markdown`` and ``st.caption``), so the fake is correspondingly
    small.  Recorded calls are exposed via ``self.calls`` for the test
    assertions.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))

    def markdown(self, *args: Any, **kwargs: Any) -> None:
        self._record("markdown", *args, **kwargs)

    def caption(self, *args: Any, **kwargs: Any) -> None:
        self._record("caption", *args, **kwargs)

    def subheader(self, *args: Any, **kwargs: Any) -> None:
        self._record("subheader", *args, **kwargs)


def _capture_renderer(renderer, *args, **kwargs):
    """Run a renderer with a fake streamlit module patched in.

    Returns the ``_FakeStreamlit`` so tests can inspect what was emitted.
    The patch is scoped to the call via a ``try/finally`` so it cannot
    leak between tests.

    Implementation note: the renderers bind ``streamlit as st`` at import
    time, so swapping ``sys.modules['streamlit']`` is not enough — we
    also have to re-bind the ``st`` attribute on the *already-imported*
    context_cards module to point at our fake.
    """
    fake = _FakeStreamlit()
    from dashboard import context_cards as _cc  # already imported by tests
    saved_st = _cc.st
    saved_mod = sys.modules.get("streamlit")
    _cc.st = fake
    sys.modules["streamlit"] = fake
    try:
        renderer(*args, **kwargs)
    finally:
        _cc.st = saved_st
        if saved_mod is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = saved_mod
    return fake


def _rendered_text(fake: _FakeStreamlit) -> str:
    """Concatenate every string written by the fake streamlit."""
    parts: list[str] = []
    for name, args, _kwargs in fake.calls:
        for a in args:
            if isinstance(a, str):
                parts.append(a)
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Helpers for building match dicts in tests
# --------------------------------------------------------------------------- #
def _match(mid: int, *, stage: str | None = "GROUP_STAGE",
           group: str | None = "GROUP_A", matchday: int | None = 1,
           home: str = "Home", away: str = "Away") -> dict:
    """Return a minimal match dict matching the loader's to_dict() shape."""
    return {
        "match_id": mid,
        "stage": stage,
        "group": group,
        "matchday": matchday,
        "home_team_name": home,
        "away_team_name": away,
    }


def _pred(mid: int, *, home: float, draw: float, away: float,
          home_team: str = "Home", away_team: str = "Away",
          has_blend: bool = True) -> dict:
    """Return a minimal prediction dict with blend_probs (and/or pi_probs)."""
    pred: dict[str, Any] = {
        "home_team": home_team,
        "away_team": away_team,
    }
    if has_blend:
        pred["blend_probs"] = {"home": home, "draw": draw, "away": away}
    else:
        pred["pi_probs"] = {"home": home, "draw": draw, "away": away}
    return pred


# =========================================================================== #
# Tournament Snapshot
# =========================================================================== #
class TestTournamentSnapshot:
    """build_tournament_snapshot + render_tournament_snapshot."""

    def test_group_stage_day_with_matchday(self) -> None:
        """5 group-stage matches on matchday 2 spread across groups C/D/E.

        The snapshot must report:
          * header = "Group Stage · Matchday 2"
          * match count line = "5 matches scheduled today"
          * groups line = "Groups C, D, E"
        """
        cc = _import_context_cards()
        matches = [
            _match(1, matchday=2, group="GROUP_C"),
            _match(2, matchday=2, group="GROUP_C"),
            _match(3, matchday=2, group="GROUP_D"),
            _match(4, matchday=2, group="GROUP_E"),
            _match(5, matchday=2, group="GROUP_E"),
        ]
        snap = cc.build_tournament_snapshot(matches)
        assert snap["header"] == "Group Stage · Matchday 2"
        assert snap["count"] == 5
        assert snap["count_label"] == "5 matches scheduled today"
        # "Groups C, D, E" — first 3 groups, no "and N more" suffix.
        assert snap["groups_label"] == "Groups C, D, E"

    def test_group_stage_multiple_matchdays_same_day(self) -> None:
        """Rule: header uses the *earliest* matchday present; sub-line
        lists every matchday seen on the day.

        The spec says the dashboard may either group or pick a dominant
        matchday; we choose the documented rule: header shows the
        earliest matchday (so users see when the day *starts*), and
        ``matchdays`` carries the full set so the renderer can list
        them in a sub-line.
        """
        cc = _import_context_cards()
        matches = [
            _match(1, matchday=2, group="GROUP_A"),
            _match(2, matchday=1, group="GROUP_B"),
            _match(3, matchday=2, group="GROUP_C"),
        ]
        snap = cc.build_tournament_snapshot(matches)
        # Earliest matchday is 1.
        assert snap["matchday"] == 1
        # Header uses earliest matchday; groups all listed (3 total).
        assert "Matchday 1" in snap["header"]
        # All matchdays surfaced in the sub-line.
        assert sorted(snap["matchdays"]) == [1, 2]

    def test_knockout_round_of_16(self) -> None:
        """8 matches at ROUND_OF_16 → round label + "8 matches remaining"."""
        cc = _import_context_cards()
        matches = [
            _match(i, stage="ROUND_OF_16", group=None, matchday=None)
            for i in range(1, 9)
        ]
        snap = cc.build_tournament_snapshot(matches)
        # The header should use format_matchday_label(stage, None) which
        # renders "Round Of 16" (title-cased, underscores → spaces).
        assert "Round" in snap["header"]
        assert snap["count"] == 8
        assert snap["count_label"] == "8 matches remaining"
        # No groups in knockout.
        assert snap["groups_label"] == ""

    def test_final_one_match(self) -> None:
        """1 match at FINAL → "Final" + "1 match remaining"."""
        cc = _import_context_cards()
        matches = [_match(1, stage="FINAL", group=None, matchday=None)]
        snap = cc.build_tournament_snapshot(matches)
        assert snap["header"] == "Final"
        assert snap["count"] == 1
        assert snap["count_label"] == "1 match remaining"

    def test_missing_stage_metadata(self) -> None:
        """No stage at all → fallback "Tournament snapshot" + count only."""
        cc = _import_context_cards()
        matches = [_match(1, stage=None, group=None, matchday=None)] * 3
        snap = cc.build_tournament_snapshot(matches)
        assert snap["header"] == "Tournament snapshot"
        assert snap["count"] == 3
        assert snap["count_label"] == "3 matches scheduled today"
        # No groups, no matchday.
        assert snap["groups_label"] == ""
        assert snap["matchday"] is None

    def test_empty_matches(self) -> None:
        """Empty list → snapshot signals "empty" without crashing the renderer."""
        cc = _import_context_cards()
        snap = cc.build_tournament_snapshot([])
        assert snap["count"] == 0
        # Renderer should render the placeholder copy, not crash.
        fake = _capture_renderer(cc.render_tournament_snapshot, snap)
        rendered = _rendered_text(fake)
        assert "Tournament snapshot unavailable" in rendered

    def test_single_group(self) -> None:
        """All matches in one group → "Groups A" (singular label is OK)."""
        cc = _import_context_cards()
        matches = [_match(i, group="GROUP_A") for i in range(1, 4)]
        snap = cc.build_tournament_snapshot(matches)
        # The spec allows singular "Group A" or plural "Groups A".
        # We use "Group A" (singular) for a single-group day.
        assert "Group A" in snap["groups_label"]
        assert snap["count"] == 3

    def test_groups_truncation_with_and_n_more(self) -> None:
        """When >3 groups, show first 3 + "and N more" suffix.

        6 matches across groups A..F → first 3 are A/B/C and 3 more.
        """
        cc = _import_context_cards()
        groups = ["GROUP_A", "GROUP_B", "GROUP_C", "GROUP_D",
                  "GROUP_E", "GROUP_F"]
        matches = [_match(i, group=g) for i, g in enumerate(groups, start=1)]
        snap = cc.build_tournament_snapshot(matches)
        assert "A, B, C" in snap["groups_label"]
        assert "3 more" in snap["groups_label"]

    def test_renderer_emits_group_data(self) -> None:
        """Renderer smoke test: snapshot dict → markdown contains the strings."""
        cc = _import_context_cards()
        matches = [
            _match(1, matchday=2, group="GROUP_C"),
            _match(2, matchday=2, group="GROUP_D"),
            _match(3, matchday=2, group="GROUP_E"),
            _match(4, matchday=2, group="GROUP_E"),
            _match(5, matchday=2, group="GROUP_E"),
        ]
        snap = cc.build_tournament_snapshot(matches)
        fake = _capture_renderer(cc.render_tournament_snapshot, snap)
        rendered = _rendered_text(fake)
        assert "Group Stage · Matchday 2" in rendered
        assert "5 matches scheduled today" in rendered
        assert "Groups C, D, E" in rendered


# =========================================================================== #
# Smart Date Default
# =========================================================================== #
class TestSmartDateDefault:
    """pick_smart_default_date — pure date-arithmetic helper."""

    def test_today_in_available(self) -> None:
        """If today is in the list, return today."""
        from dashboard.context_cards import pick_smart_default_date
        d = pick_smart_default_date(
            today=date(2026, 6, 16),
            available_dates=["2026-06-15", "2026-06-16", "2026-06-17"],
        )
        assert d == date(2026, 6, 16)

    def test_no_today_nearest_future(self) -> None:
        """Today not in list → nearest future date."""
        from dashboard.context_cards import pick_smart_default_date
        d = pick_smart_default_date(
            today=date(2026, 6, 16),
            available_dates=["2026-06-15", "2026-06-18"],
        )
        assert d == date(2026, 6, 18)

    def test_no_today_nearest_future_skips_today(self) -> None:
        """Today not in list, only one future date → return it."""
        from dashboard.context_cards import pick_smart_default_date
        d = pick_smart_default_date(
            today=date(2026, 6, 16),
            available_dates=["2026-06-19"],
        )
        assert d == date(2026, 6, 19)

    def test_no_future_only_past(self) -> None:
        """No future dates → most recent past date <= today."""
        from dashboard.context_cards import pick_smart_default_date
        d = pick_smart_default_date(
            today=date(2026, 6, 30),
            available_dates=["2026-06-15", "2026-06-16"],
        )
        assert d == date(2026, 6, 16)

    def test_empty_available(self) -> None:
        """No dates at all → defensive: return today."""
        from dashboard.context_cards import pick_smart_default_date
        d = pick_smart_default_date(today=date(2026, 6, 16), available_dates=[])
        assert d == date(2026, 6, 16)

    def test_exact_today_no_other_dates(self) -> None:
        """List contains only today → return today."""
        from dashboard.context_cards import pick_smart_default_date
        d = pick_smart_default_date(
            today=date(2026, 6, 16), available_dates=["2026-06-16"],
        )
        assert d == date(2026, 6, 16)

    def test_picks_nearest_when_multiple_future(self) -> None:
        """Out-of-order list with multiple future dates → nearest future wins."""
        from dashboard.context_cards import pick_smart_default_date
        d = pick_smart_default_date(
            today=date(2026, 6, 16),
            available_dates=["2026-06-20", "2026-06-17", "2026-06-19"],
        )
        assert d == date(2026, 6, 17)


# =========================================================================== #
# Highest Model Confidence
# =========================================================================== #
class TestHighestModelConfidence:
    """highest_model_confidence — picks the single highest market prob."""

    def test_picks_highest_home_win(self) -> None:
        """Two matches, both home-win favourite; the larger wins."""
        cc = _import_context_cards()
        matches = [
            _match(1, home="England", away="Croatia"),
            _match(2, home="Portugal", away="Spain"),
        ]
        predictions = {
            1: _pred(1, home=0.50, draw=0.25, away=0.25,
                     home_team="England", away_team="Croatia"),
            2: _pred(2, home=0.60, draw=0.20, away=0.20,
                     home_team="Portugal", away_team="Spain"),
        }
        result = cc.highest_model_confidence(matches, predictions)
        assert result is not None
        assert result["match_id"] == 2
        assert result["market"] == "home"
        assert result["home_team"] == "Portugal"
        assert result["away_team"] == "Spain"
        assert result["probability"] == pytest.approx(0.60)

    def test_picks_draw_when_highest(self) -> None:
        """Draw is the top market → renderer surfaces 'Match to End in a Draw'."""
        cc = _import_context_cards()
        matches = [_match(1, home="England", away="Croatia")]
        # draw (0.40) is the largest of the three.
        predictions = {
            1: _pred(1, home=0.30, draw=0.40, away=0.30,
                     home_team="England", away_team="Croatia"),
        }
        result = cc.highest_model_confidence(matches, predictions)
        assert result is not None
        assert result["market"] == "draw"
        assert result["probability"] == pytest.approx(0.40)
        # Renderer copy for draw is "Match to End in a Draw".
        fake = _capture_renderer(cc.render_highest_confidence, result)
        rendered = _rendered_text(fake)
        assert "Match to End in a Draw" in rendered

    def test_ignores_odds_and_edge(self) -> None:
        """Odds / edge / book_fair must not influence the pick — only model probs.

        Match A: tiny model prob + huge book edge (book thinks it's a lock).
        Match B: moderate model prob + no book edge.
        Highest MODEL confidence should be B, even though A is "value".
        """
        cc = _import_context_cards()
        matches = [
            _match(1, home="Iceland", away="Argentina"),
            _match(2, home="France", away="Brazil"),
        ]
        predictions = {
            1: {
                "home_team": "Iceland", "away_team": "Argentina",
                "blend_probs": {"home": 0.10, "draw": 0.10, "away": 0.80},
                "book_fair": {"home": 0.95},       # book says Iceland
                "edge": 0.85,                       # giant value
                "odds": {"home": "+800", "away": "-2000"},
                "value": True,
            },
            2: {
                "home_team": "France", "away_team": "Brazil",
                "blend_probs": {"home": 0.50, "draw": 0.25, "away": 0.25},
                "book_fair": {"home": 0.50},
                "edge": 0.0,
                "odds": {"home": "+100"},
                "value": False,
            },
        }
        result = cc.highest_model_confidence(matches, predictions)
        assert result is not None
        # Argentina is the higher *model* probability (0.80), not Iceland.
        assert result["match_id"] == 1
        assert result["market"] == "away"
        assert result["home_team"] == "Iceland"
        assert result["away_team"] == "Argentina"
        assert result["probability"] == pytest.approx(0.80)

    def test_placeholder_when_no_predictions(self) -> None:
        """Matches present, predictions dict empty → returns None."""
        cc = _import_context_cards()
        matches = [_match(1), _match(2)]
        result = cc.highest_model_confidence(matches, predictions={})
        assert result is None

    def test_placeholder_when_no_matches(self) -> None:
        """No matches at all → returns None."""
        cc = _import_context_cards()
        result = cc.highest_model_confidence(
            [],
            predictions={1: _pred(1, home=0.5, draw=0.3, away=0.2)},
        )
        assert result is None

    def test_falls_back_to_pi_probs_when_blend_missing(self) -> None:
        """If blend_probs is absent, use pi_probs."""
        cc = _import_context_cards()
        matches = [_match(1)]
        predictions = {1: _pred(1, home=0.40, draw=0.30, away=0.30, has_blend=False)}
        result = cc.highest_model_confidence(matches, predictions)
        assert result is not None
        assert result["market"] == "home"
        assert result["probability"] == pytest.approx(0.40)

    def test_uses_blend_probs_when_present(self) -> None:
        """If both blend_probs and pi_probs are present, blend_probs wins.

        We seed pi_probs with deliberately wrong values to prove the
        helper picked blend_probs and not pi_probs.
        """
        cc = _import_context_cards()
        matches = [_match(1)]
        pred = _pred(1, home=0.55, draw=0.25, away=0.20, has_blend=True)
        pred["pi_probs"] = {"home": 0.10, "draw": 0.10, "away": 0.10}
        predictions = {1: pred}
        result = cc.highest_model_confidence(matches, predictions)
        assert result is not None
        assert result["market"] == "home"
        assert result["probability"] == pytest.approx(0.55)

    def test_returns_dict_shape(self) -> None:
        """Returned dict must have exactly the documented keys."""
        cc = _import_context_cards()
        matches = [_match(1)]
        predictions = {1: _pred(1, home=0.5, draw=0.3, away=0.2)}
        result = cc.highest_model_confidence(matches, predictions)
        assert result is not None
        assert set(result.keys()) == {
            "match_id", "market", "probability", "home_team", "away_team",
        }

    def test_tie_break_prefers_earlier_match(self) -> None:
        """Two matches, identical top probs → earlier match in the list wins."""
        cc = _import_context_cards()
        matches = [
            _match(1, home="England", away="Croatia"),
            _match(2, home="Portugal", away="Spain"),
        ]
        predictions = {
            1: _pred(1, home=0.50, draw=0.25, away=0.25,
                     home_team="England", away_team="Croatia"),
            2: _pred(2, home=0.50, draw=0.25, away=0.25,
                     home_team="Portugal", away_team="Spain"),
        }
        result = cc.highest_model_confidence(matches, predictions)
        assert result is not None
        assert result["match_id"] == 1
        assert result["home_team"] == "England"

    def test_renderer_with_prediction(self) -> None:
        """Confidence dict → renderer emits '{team} to Win' + percent."""
        cc = _import_context_cards()
        result = {
            "match_id": 1,
            "market": "home",
            "probability": 0.612,
            "home_team": "France",
            "away_team": "Brazil",
        }
        fake = _capture_renderer(cc.render_highest_confidence, result)
        rendered = _rendered_text(fake)
        assert "France to Win" in rendered
        assert "61.2%" in rendered
        # The caption MUST be present (per the brief).
        assert "model probability only" in rendered.lower()

    def test_renderer_placeholder_when_none(self) -> None:
        """confidence=None → renderer emits the placeholder copy."""
        cc = _import_context_cards()
        fake = _capture_renderer(cc.render_highest_confidence, None)
        rendered = _rendered_text(fake)
        assert "Run predictions" in rendered
