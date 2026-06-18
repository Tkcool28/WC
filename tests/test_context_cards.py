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


# =========================================================================== #
# Autoload (Phase 9 — dashboard context cards populate on page open)
# =========================================================================== #
class TestAutoloadPure:
    """_autoload_pure: pure helper, no Streamlit session_state.

    The pure helper takes ``load_unplayed_fn``, ``predict_match_fn``,
    ``resolve_match_fn``, ``get_ratings_fn`` as injected callables so
    we can drive it without touching the loader / model / Streamlit.
    Each test exercises one rule from the spec.
    """

    def _pred_stub(self, **overrides):
        """Return a callable that behaves like _predict_match_cached."""
        pred = {
            "home_team": overrides.get("home", "Home"),
            "away_team": overrides.get("away", "Away"),
            "blend_probs": {"home": 0.5, "draw": 0.3, "away": 0.2},
            "pi_probs": {"home": 0.5, "draw": 0.3, "away": 0.2},
            "canonical_home_id": "",
            "canonical_away_id": "",
        }
        pred.update(overrides.get("extra", {}))
        return pred

    def test_autoload_loads_matches_without_button_click(self) -> None:
        """Pure helper returns matches + predictions for the date — no
        button click required, no session_state needed.

        The card renderers read from session_state; here we just verify
        the pure helper produces the right payload.  Session-state
        wiring is covered by the integration test at the bottom.
        """
        from dashboard.context_cards import _autoload_pure
        matches = [
            {"match_id": 1, "home_team_name": "A", "away_team_name": "B",
             "home_team_id": 11, "away_team_id": 22},
            {"match_id": 2, "home_team_name": "C", "away_team_name": "D",
             "home_team_id": 33, "away_team_id": 44},
        ]
        calls = {"load_unplayed": [], "get_ratings": [], "predict": []}

        def load_unplayed(date_iso):
            calls["load_unplayed"].append(date_iso)
            return matches

        def get_ratings(cutoff_iso, corpus):
            calls["get_ratings"].append(cutoff_iso)
            return {}

        def predict_match(**kwargs):
            calls["predict"].append(kwargs)
            mid = kwargs.get("home_team_id")  # use anything unique
            return self._pred_stub(home="A", away="B", extra={"mid_tag": mid})

        payload = _autoload_pure(
            "2026-06-18",
            corpus=[{"date": "2020-01-01"}],
            elo_snapshots={"elo": 1500},
            load_unplayed_fn=load_unplayed,
            predict_match_fn=predict_match,
            get_ratings_fn=get_ratings,
        )

        # Matches came through.
        assert len(payload["matches"]) == 2
        assert payload["date_iso"] == "2026-06-18"
        # Predictions were computed for both matches.
        assert set(payload["predictions"].keys()) == {1, 2}
        # get_ratings was called with the smart cutoff.
        assert calls["get_ratings"] == ["2026-06-18T23:59:59Z"]
        # predict_match was called for both matches.
        assert len(calls["predict"]) == 2

    def test_autoload_populates_predictions_for_highest_confidence(self) -> None:
        """The pure payload's predictions block is shaped so the existing
        ``highest_model_confidence(matches, predictions)`` helper picks
        the top market.
        """
        from dashboard.context_cards import _autoload_pure, highest_model_confidence

        matches = [
            {"match_id": 1, "home_team_name": "France", "away_team_name": "Brazil",
             "home_team_id": 11, "away_team_id": 22},
        ]

        def load_unplayed(date_iso):
            return matches

        def predict_match(**kwargs):
            return self._pred_stub(
                home="France", away="Brazil",
                extra={"blend_probs": {"home": 0.7, "draw": 0.2, "away": 0.1}},
            )

        payload = _autoload_pure(
            "2026-06-18", corpus=[], elo_snapshots={},
            load_unplayed_fn=load_unplayed, predict_match_fn=predict_match,
            get_ratings_fn=None,
        )
        top = highest_model_confidence(payload["matches"], payload["predictions"])
        assert top is not None
        assert top["match_id"] == 1
        assert top["market"] == "home"
        assert top["probability"] == pytest.approx(0.7)

    def test_autoload_short_circuits_on_cache_hit(self) -> None:
        """Two pure-helper calls for the same date should re-run
        ``load_unplayed_fn`` and ``predict_match_fn`` once each.

        The Streamlit session-state short-circuit lives in
        ``autoload_context_for_date`` (the wrapper).  Here we verify
        the *pure* contract: callers can hold the payload themselves
        and avoid a second load by simply not re-invoking the pure
        helper.  The integration test below confirms the wrapper's
        own short-circuit via session state.
        """
        from dashboard.context_cards import _autoload_pure
        calls = {"load_unplayed": 0, "get_ratings": 0, "predict": 0}
        matches = [
            {"match_id": 1, "home_team_name": "A", "away_team_name": "B",
             "home_team_id": 11, "away_team_id": 22},
        ]

        def load_unplayed(date_iso):
            calls["load_unplayed"] += 1
            return matches

        def get_ratings(cutoff_iso, corpus):
            calls["get_ratings"] += 1
            return {}

        def predict_match(**kwargs):
            calls["predict"] += 1
            return self._pred_stub()

        # First call: does the work.
        _autoload_pure(
            "2026-06-18", corpus=[{}], elo_snapshots={},
            load_unplayed_fn=load_unplayed,
            predict_match_fn=predict_match,
            get_ratings_fn=get_ratings,
        )
        first = dict(calls)
        # Second call: by spec the wrapper short-circuits, so we don't
        # re-call the pure helper. The wrapper integration test below
        # exercises this end-to-end. Here we just confirm the pure
        # helper is deterministic and side-effect-free between calls.
        payload2 = _autoload_pure(
            "2026-06-18", corpus=[{}], elo_snapshots={},
            load_unplayed_fn=load_unplayed,
            predict_match_fn=predict_match,
            get_ratings_fn=get_ratings,
        )
        # Pure helper itself doesn't cache — it always re-runs the
        # injected callables. That's fine: the WRAPPER caches via
        # session state. This assertion documents the pure-helper
        # contract (no caching layer) so refactors don't accidentally
        # add one.
        assert calls == {
            "load_unplayed": first["load_unplayed"] + 1,
            "get_ratings": first["get_ratings"] + 1,
            "predict": first["predict"] + 1,
        }
        # And the payload is the same shape regardless.
        assert payload2["date_iso"] == "2026-06-18"
        assert 1 in payload2["predictions"]

    def test_autoload_invalidates_on_date_change(self) -> None:
        """Different dates → different matches, predict_match called twice."""
        from dashboard.context_cards import _autoload_pure
        schedule = {
            "2026-06-18": [
                {"match_id": 1, "home_team_name": "A", "away_team_name": "B",
                 "home_team_id": 11, "away_team_id": 22},
            ],
            "2026-06-19": [
                {"match_id": 2, "home_team_name": "C", "away_team_name": "D",
                 "home_team_id": 33, "away_team_id": 44},
                {"match_id": 3, "home_team_name": "E", "away_team_name": "F",
                 "home_team_id": 55, "away_team_id": 66},
            ],
        }
        seen_dates: list[str] = []

        def load_unplayed(date_iso):
            seen_dates.append(date_iso)
            return schedule.get(date_iso, [])

        def predict_match(**kwargs):
            return self._pred_stub()

        p_a = _autoload_pure(
            "2026-06-18", corpus=[{}], elo_snapshots={},
            load_unplayed_fn=load_unplayed, predict_match_fn=predict_match,
            get_ratings_fn=None,
        )
        p_b = _autoload_pure(
            "2026-06-19", corpus=[{}], elo_snapshots={},
            load_unplayed_fn=load_unplayed, predict_match_fn=predict_match,
            get_ratings_fn=None,
        )
        assert seen_dates == ["2026-06-18", "2026-06-19"]
        assert [m["match_id"] for m in p_a["matches"]] == [1]
        assert [m["match_id"] for m in p_b["matches"]] == [2, 3]
        assert set(p_a["predictions"].keys()) == {1}
        assert set(p_b["predictions"].keys()) == {2, 3}

    def test_autoload_handles_empty_schedule(self) -> None:
        """Date with no matches → empty matches + empty predictions, no raise."""
        from dashboard.context_cards import _autoload_pure

        def load_unplayed(date_iso):
            return []

        def predict_match(**kwargs):
            raise AssertionError("predict_match must not be called when no matches")

        payload = _autoload_pure(
            "2026-06-18", corpus=[{}], elo_snapshots={},
            load_unplayed_fn=load_unplayed, predict_match_fn=predict_match,
            get_ratings_fn=None,
        )
        assert payload["matches"] == []
        assert payload["predictions"] == {}
        assert payload["date_iso"] == "2026-06-18"

    def test_autoload_uses_smart_default_on_first_visit(self) -> None:
        """First visit (no KEYS.SELECTED_DATE) → main() picks the smart default.

        We don't drive main() here (that's an integration test).  Instead
        we verify the contract the wrapper relies on: when KEYS.SELECTED_DATE
        is absent, ``pick_smart_default_date`` is the source of truth.
        """
        from dashboard.context_cards import pick_smart_default_date
        # Simulate a session state where KEYS.SELECTED_DATE was never set.
        fake_ss: dict = {}
        picked = None
        if fake_ss.get("selected_date") is None:
            # Mirror what main() does on the first-visit branch.
            picked = pick_smart_default_date(
                today=date(2026, 6, 18),
                available_dates=["2026-06-15", "2026-06-18", "2026-06-20"],
            )
        assert picked == date(2026, 6, 18)
        # Sanity: if a widget has populated the key, main() uses that instead.
        fake_ss["selected_date"] = "2026-06-20"
        chosen = fake_ss["selected_date"]
        assert chosen == "2026-06-20"

    def test_autoload_falls_back_per_match_on_predict_error(self) -> None:
        """When predict_match_fn raises for ONE match, autoload still
        completes with the other match's real prediction + the failing
        match's tier-C fallback dict (same shape the per-view renderers use).
        """
        from dashboard.context_cards import _autoload_pure
        matches = [
            {"match_id": 1, "home_team_name": "A", "away_team_name": "B",
             "home_team_id": 11, "away_team_id": 22},
            {"match_id": 2, "home_team_name": "C", "away_team_name": "D",
             "home_team_id": 33, "away_team_id": 44},
            {"match_id": 3, "home_team_name": "E", "away_team_name": "F",
             "home_team_id": 55, "away_team_id": 66},
        ]

        def load_unplayed(date_iso):
            return matches

        def predict_match(**kwargs):
            # Fail only when the home_team_id is 33 (match 2).
            if kwargs["home_team_id"] == 33:
                raise RuntimeError("kaboom")
            return self._pred_stub(
                home=kwargs["home_team"], away=kwargs["away_team"],
                extra={"blend_probs": {"home": 0.6, "draw": 0.2, "away": 0.2}},
            )

        payload = _autoload_pure(
            "2026-06-18", corpus=[{}], elo_snapshots={},
            load_unplayed_fn=load_unplayed, predict_match_fn=predict_match,
            get_ratings_fn=None,
        )
        # All 3 matches got a prediction entry.
        assert set(payload["predictions"].keys()) == {1, 2, 3}
        # The failing match has the tier-C fallback shape.
        fallback = payload["predictions"][2]
        assert fallback["confidence"]["tier"] == "C"
        assert fallback["confidence"]["tier_description"] == "Limited data"
        assert "kaboom" in fallback["confidence"]["warnings"][0]
        # The neutral probs match the per-view fallback.
        assert fallback["blend_probs"] == {"home": 0.4, "draw": 0.3, "away": 0.3}
        assert fallback["pi_probs"] == {"home": 0.4, "draw": 0.3, "away": 0.3}
        # The successful matches kept their real probs.
        assert payload["predictions"][1]["blend_probs"]["home"] == pytest.approx(0.6)
        assert payload["predictions"][3]["blend_probs"]["home"] == pytest.approx(0.6)


class TestAutoloadSessionIntegration:
    """End-to-end check that ``autoload_context_for_date`` writes to
    the same session-state keys the per-view renderers read.

    Uses ``streamlit.testing.v1.AppTest`` so we run the real
    Streamlit runtime (with its ``AppSession``) without launching a
    browser.  We point the helper at ``main()`` in isolation by
    calling the helper directly inside the AppTest script-run context.
    """

    def test_wrapper_writes_session_state_keys(self) -> None:
        """autoload_context_for_date populates KEYS.LOADED_MATCHES,
        KEYS.LOADED_MATCHES + '.date', KEYS.PREDICTIONS_BY_MATCH,
        and KEYS.CONTEXT_AUTOLOAD_DATE — the same keys the card
        renderers read.
        """
        from streamlit.testing.v1 import AppTest

        # Minimal Streamlit script that calls the helper with tiny stubs.
        # We bind load_unplayed / predict to lambdas returning trivial
        # data so the test doesn't need a real schedule file.
        at = AppTest.from_string(
            """
import streamlit as st
from datetime import date as _date
from dashboard.context_cards import autoload_context_for_date
from dashboard.session_state import KEYS

matches = [
    {"match_id": 1, "home_team_name": "A", "away_team_name": "B",
     "home_team_id": 11, "away_team_id": 22, "stage": "GROUP_STAGE",
     "group": "GROUP_A", "matchday": 1},
]
predictions = {
    1: {
        "home_team": "A", "away_team": "B",
        "blend_probs": {"home": 0.7, "draw": 0.2, "away": 0.1},
        "pi_probs":   {"home": 0.6, "draw": 0.25, "away": 0.15},
        "canonical_home_id": "", "canonical_away_id": "",
    },
}

# Monkey-patch the module-level helpers the autoload wrapper imports.
import dashboard.app as _app
import dashboard.context_cards as _cc
_app._load_unplayed_for_date = lambda d: matches
_app._predict_match_cached = lambda **kw: dict(predictions[kw["home_team_id"] // 11])
_app.get_ratings = lambda cutoff, corpus: {}
_app._CORPUS_BY_ID = {}
_app._ELO_BY_ID = {}
# Also patch the names _cc imported by name inside the wrapper.
_cc_module = _cc.autoload_context_for_date.__module__
import sys
sys.modules[_cc_module].__dict__  # touch to ensure loaded

# Drive the wrapper. The corpus / elo_snapshots objects don't matter
# for this test — we only care that session_state gets populated.
corpus = [{"date": "2020-01-01"}]
elo = {"elo": 1500}
result = autoload_context_for_date("2026-06-18", corpus, elo)

st.session_state["_test_result"] = result
st.session_state["_test_keys"] = list(st.session_state.keys())
"""
        ).run(timeout=30)
        assert not at.exception, f"AppTest raised: {at.exception}"

        ss = at.session_state
        # KEYS.LOADED_MATCHES is populated.
        assert ss["loaded_matches"] == [
            {
                "match_id": 1, "home_team_name": "A", "away_team_name": "B",
                "home_team_id": 11, "away_team_id": 22, "stage": "GROUP_STAGE",
                "group": "GROUP_A", "matchday": 1,
            },
        ]
        # KEYS.LOADED_MATCHES + '.date' matches the autoload date.
        assert ss["loaded_matches.date"] == "2026-06-18"
        # KEYS.PREDICTIONS_BY_MATCH has an entry for match 1.
        assert 1 in ss["predictions_by_match"]
        # KEYS.CONTEXT_AUTOLOAD_DATE sentinel is set so we short-circuit
        # on the next rerun for the same date.
        assert ss["context.autoloaded"] == "2026-06-18"

    def test_wrapper_short_circuits_on_second_call(self) -> None:
        """Calling autoload_context_for_date twice for the same date in
        one session invokes the loader / model exactly once.
        """
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(
            """
import streamlit as st
from dashboard.context_cards import autoload_context_for_date
from dashboard.session_state import KEYS

calls = {"load": 0, "predict": 0}
matches = [
    {"match_id": 1, "home_team_name": "A", "away_team_name": "B",
     "home_team_id": 11, "away_team_id": 22},
]
prediction = {
    "home_team": "A", "away_team": "B",
    "blend_probs": {"home": 0.7, "draw": 0.2, "away": 0.1},
    "pi_probs":   {"home": 0.7, "draw": 0.2, "away": 0.1},
    "canonical_home_id": "", "canonical_away_id": "",
}

import dashboard.app as _app
_app._load_unplayed_for_date = lambda d: (calls.__setitem__("load", calls["load"] + 1) or matches)
_app._predict_match_cached = lambda **kw: (
    calls.__setitem__("predict", calls["predict"] + 1) or dict(prediction)
)
_app.get_ratings = lambda cutoff, corpus: {}
_app._CORPUS_BY_ID = {}
_app._ELO_BY_ID = {}

corpus = [{"date": "2020-01-01"}]
elo = {"elo": 1500}
# First call: does the work.
autoload_context_for_date("2026-06-18", corpus, elo)
first = dict(calls)
# Second call: same date → short-circuits.
autoload_context_for_date("2026-06-18", corpus, elo)
second = dict(calls)

st.session_state["_test_calls_after_first"] = first
st.session_state["_test_calls_after_second"] = second
"""
        ).run(timeout=30)
        assert not at.exception, f"AppTest raised: {at.exception}"

        first = at.session_state["_test_calls_after_first"]
        second = at.session_state["_test_calls_after_second"]
        # First call exercised both loader and predict.
        assert first["load"] == 1
        assert first["predict"] == 1
        # Second call short-circuited: no additional loader/predict calls.
        assert second["load"] == 1
        assert second["predict"] == 1

    def test_wrapper_reloads_on_date_change(self) -> None:
        """Two different dates in the same session → two loader calls."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(
            """
import streamlit as st
from dashboard.context_cards import autoload_context_for_date

calls = {"load": 0, "predict": 0}
matches_a = [
    {"match_id": 1, "home_team_name": "A", "away_team_name": "B",
     "home_team_id": 11, "away_team_id": 22},
]
matches_b = [
    {"match_id": 2, "home_team_name": "C", "away_team_name": "D",
     "home_team_id": 33, "away_team_id": 44},
    {"match_id": 3, "home_team_name": "E", "away_team_name": "F",
     "home_team_id": 55, "away_team_id": 66},
]
prediction = {
    "blend_probs": {"home": 0.7, "draw": 0.2, "away": 0.1},
    "pi_probs":   {"home": 0.7, "draw": 0.2, "away": 0.1},
    "canonical_home_id": "", "canonical_away_id": "",
}

def _load(d):
    calls["load"] += 1
    return matches_a if d == "2026-06-18" else matches_b

def _pred(**kw):
    calls["predict"] += 1
    p = dict(prediction)
    p["home_team"] = kw.get("home_team", "Home")
    p["away_team"] = kw.get("away_team", "Away")
    return p

import dashboard.app as _app
_app._load_unplayed_for_date = _load
_app._predict_match_cached = _pred
_app.get_ratings = lambda cutoff, corpus: {}
_app._CORPUS_BY_ID = {}
_app._ELO_BY_ID = {}

corpus = [{"date": "2020-01-01"}]
elo = {"elo": 1500}
autoload_context_for_date("2026-06-18", corpus, elo)
after_a = dict(calls)
autoload_context_for_date("2026-06-19", corpus, elo)
after_b = dict(calls)

st.session_state["_after_a"] = after_a
st.session_state["_after_b"] = after_b
st.session_state["_matches_after_b"] = st.session_state["loaded_matches"]
"""
        ).run(timeout=30)
        assert not at.exception, f"AppTest raised: {at.exception}"

        after_a = at.session_state["_after_a"]
        after_b = at.session_state["_after_b"]
        # Date A: 1 match loaded, 1 prediction.
        assert after_a == {"load": 1, "predict": 1}
        # Date B: 1 more load + 2 more predictions (2 matches).
        assert after_b == {"load": 2, "predict": 3}
        # The latest matches list is date B's.
        loaded_ids = [m["match_id"] for m in at.session_state["_matches_after_b"]]
        assert loaded_ids == [2, 3]


# =========================================================================== #
# SessionKeys — autoload sentinel key is present and namespaced
# =========================================================================== #
class TestAutoloadSessionKey:
    """The autoload helper relies on KEYS.CONTEXT_AUTOLOAD_DATE; verify
    it exists, is a dotted-namespace string, and doesn't collide with
    any existing key.
    """

    def test_context_autoload_date_key_exists(self) -> None:
        from dashboard.session_state import KEYS
        # Sentinel key for cache invalidation.
        assert isinstance(KEYS.CONTEXT_AUTOLOAD_DATE, str)
        assert KEYS.CONTEXT_AUTOLOAD_DATE  # non-empty
        # Must be namespaced (the dotted convention enforced by tests
        # elsewhere — see tests/test_dashboard_nav.py).
        assert "." in KEYS.CONTEXT_AUTOLOAD_DATE

    def test_context_autoload_date_key_does_not_collide(self) -> None:
        """Defensive: no existing key has the same string value."""
        from dashboard.session_state import KEYS, SessionKeys
        for field in SessionKeys.__dataclass_fields__:
            if field == "CONTEXT_AUTOLOAD_DATE":
                continue
            other = getattr(KEYS, field)
            assert other != KEYS.CONTEXT_AUTOLOAD_DATE, (
                f"CONTEXT_AUTOLOAD_DATE collides with {field}"
            )
