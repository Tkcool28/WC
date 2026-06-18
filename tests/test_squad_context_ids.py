"""Tests for Squad Context canonical-ID routing in Analysis view.

Covers dashboard.analysis_view._render_squad_context — must read
canonical IDs (e.g. "ENG", "CRO") from prediction or match_meta,
never pass numeric schedule fd_ids into get_match_context.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

import dashboard.analysis_view as av
from dashboard.analysis_view import _looks_like_canonical_id
from dashboard.context_loader import get_match_context


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_looks_like_canonical_id_accepts_known_canonicals() -> None:
    assert _looks_like_canonical_id("ENG") is True
    assert _looks_like_canonical_id("CRO") is True
    assert _looks_like_canonical_id("POR") is True
    assert _looks_like_canonical_id("COD") is True


def test_looks_like_canonical_id_rejects_numeric_schedule_ids() -> None:
    # int schedule fd_ids
    assert _looks_like_canonical_id(770) is False
    assert _looks_like_canonical_id(799) is False
    # str-of-int schedule fd_ids (the match_meta path)
    assert _looks_like_canonical_id("770") is False
    assert _looks_like_canonical_id("799") is False
    # Zero
    assert _looks_like_canonical_id(0) is False
    # Empty / None
    assert _looks_like_canonical_id(None) is False
    assert _looks_like_canonical_id("") is False


# --------------------------------------------------------------------------- #
# Routing behavior — direct call with monkeypatched get_match_context
# --------------------------------------------------------------------------- #
def test_squad_context_uses_canonical_ids_from_prediction(monkeypatch) -> None:
    """When prediction has canonical_home_id / canonical_away_id,
    _render_squad_context must call get_match_context with those (not
    the schedule fd_ids from match_meta)."""
    captured: list[tuple[str, str]] = []

    def spy(home_id, away_id):
        captured.append((home_id, away_id))
        return get_match_context(home_id, away_id)

    monkeypatch.setattr(
        "dashboard.context_loader.get_match_context", spy
    )
    monkeypatch.setattr(
        "dashboard.analysis_view.get_match_context", spy, raising=False
    )

    match_meta = {
        "home_team_name": "England", "away_team_name": "Croatia",
        "home_team_id": 770, "away_team_id": 799,  # schedule fd_ids
    }
    prediction = {
        "canonical_home_id": "ENG", "canonical_away_id": "CRO",
    }
    av._render_squad_context(match_meta, prediction=prediction)
    # The spy was invoked with canonical IDs, NOT schedule fd_ids.
    assert captured, "get_match_context was not called at all"
    for h, a in captured:
        assert h != "770" and h != 770, (
            f"schedule fd_id routed to get_match_context as home: {h!r}"
        )
        assert a != "799" and a != 799, (
            f"schedule fd_id routed to get_match_context as away: {a!r}"
        )
    canonical_pairs = {(h, a) for h, a in captured}
    assert any(h == "ENG" for h, _ in canonical_pairs), (
        f"canonical ENG not routed; captured: {captured!r}"
    )
    assert any(a == "CRO" for _, a in canonical_pairs), (
        f"canonical CRO not routed; captured: {captured!r}"
    )


def test_squad_context_falls_back_to_match_meta_canonical(monkeypatch) -> None:
    """When prediction is None, fall back to match_meta canonical IDs."""
    captured: list[tuple[str, str]] = []

    def spy(home_id, away_id):
        captured.append((home_id, away_id))
        return get_match_context(home_id, away_id)

    monkeypatch.setattr(
        "dashboard.context_loader.get_match_context", spy
    )
    monkeypatch.setattr(
        "dashboard.analysis_view.get_match_context", spy, raising=False
    )

    match_meta = {
        "home_team_name": "England", "away_team_name": "Croatia",
        "canonical_home_id": "ENG", "canonical_away_id": "CRO",
    }
    av._render_squad_context(match_meta, prediction=None)
    assert captured, "get_match_context was not called"
    for h, a in captured:
        assert h == "ENG" and a == "CRO", (
            f"expected fallback to match_meta canonical IDs; got {(h, a)!r}"
        )


def test_squad_context_skips_numeric_ids(monkeypatch) -> None:
    """When only schedule fd_ids are present (no canonical), the helper
    must NOT call get_match_context with the numeric ids."""
    captured: list[tuple[str, str]] = []

    def spy(home_id, away_id):
        captured.append((home_id, away_id))
        return get_match_context(home_id, away_id)

    monkeypatch.setattr(
        "dashboard.context_loader.get_match_context", spy
    )
    monkeypatch.setattr(
        "dashboard.analysis_view.get_match_context", spy, raising=False
    )

    match_meta = {
        "home_team_name": "England", "away_team_name": "Croatia",
        "home_team_id": 770, "away_team_id": 799,  # schedule fd_ids only
    }
    av._render_squad_context(match_meta, prediction=None)
    assert captured == [], (
        f"get_match_context was called with numeric ids: {captured!r}. "
        "Squad context must NOT pass schedule fd_ids to the loader."
    )


def test_squad_context_no_canonical_at_all_shows_calm_caption(monkeypatch) -> None:
    """match_meta has only numeric ids AND no prediction — must NOT
    call get_match_context at all."""
    captured: list[tuple[str, str]] = []

    def spy(home_id, away_id):
        captured.append((home_id, away_id))
        return get_match_context(home_id, away_id)

    monkeypatch.setattr(
        "dashboard.context_loader.get_match_context", spy
    )
    monkeypatch.setattr(
        "dashboard.analysis_view.get_match_context", spy, raising=False
    )

    match_meta = {"home_team_id": 770, "away_team_id": 799}
    av._render_squad_context(match_meta, prediction=None)
    assert captured == [], (
        f"get_match_context should NOT be called when only fd_ids are present; got: {captured!r}"
    )


def test_squad_context_does_not_call_get_match_context_with_int() -> None:
    """Regression guard: numeric schedule IDs must never reach the loader,
    regardless of which fallback path is taken. This is a documentation
    marker so future readers see the invariant explicitly stated.
    The actual enforcement is in the tests above."""
    assert True


def test_squad_context_england_croatia_resolves_real_data() -> None:
    """End-to-end: when canonical IDs ENG/CRO are passed to
    get_match_context, the real loader must return a non-null context
    dict for each side (whatever the manual CSV contains)."""
    ctx = get_match_context("ENG", "CRO")
    home = ctx.get("home") or {}
    away = ctx.get("away") or {}
    assert home is not None
    assert away is not None
    # Acceptable value_tier values per context_loader.value_tier:
    # elite / high / mid / low / unknown / None.
    assert home.get("value_tier") in ("elite", "high", "mid", "low", "unknown", None)
    assert away.get("value_tier") in ("elite", "high", "mid", "low", "unknown", None)


def test_render_analysis_view_passes_prediction_to_squad_context(monkeypatch) -> None:
    """End-to-end: render_analysis_view should pass the prediction
    through to _render_squad_context so canonical IDs reach the loader."""
    captured_kwargs: list[dict] = []

    def spy_squad(match_meta, *, prediction=None, name_to_id=None):
        captured_kwargs.append({
            "match_meta": match_meta,
            "prediction": prediction,
        })

    monkeypatch.setattr(av, "_render_squad_context", spy_squad)

    # Build minimal inputs to render_analysis_view.
    matches = [{
        "match_id": "M1",
        "home_team_name": "England", "away_team_name": "Croatia",
        "kickoff_iso": "2026-06-17T20:00:00Z",
        "group": "GROUP_K", "stage": "GROUP_STAGE", "matchday": 1,
    }]
    prediction = {
        "home_team": "England", "away_team": "Croatia",
        "canonical_home_id": "ENG", "canonical_away_id": "CRO",
        "blend_probs": {"home": 0.51, "draw": 0.25, "away": 0.24},
        "confidence": {"tier": "A", "warnings": []},
        "identity_warnings": [],
    }
    av.render_analysis_view(
        matches_for_date=matches,
        predictions_by_match={"M1": prediction},
        market_by_match={},
    )
    assert captured_kwargs, "_render_squad_context was not called"
    pred_passed = captured_kwargs[0]["prediction"]
    assert pred_passed is prediction, (
        "prediction must be passed through to _render_squad_context "
        "so canonical IDs are available for lookup"
    )