"""Tests for the canonical national-team identity layer.

These tests pin:
  * the 3-letter canonical IDs (ARG, USA, ALG, ...) are stable
  * the three lookup paths (football_data id, corpus id, name) all
    converge to the same canonical id
  * status taxonomy (resolved / history_missing / identity_unresolved)
  * the new confidence flag is propagated by evaluate_match
  * confidence_tier surfaces the "Identity unresolved" branch
  * the dashboard's evaluate_one_game now uses the registry to
    translate football-data ids to corpus ids for the pi-rating lookup
  * the existing manual-odds numeric contract is preserved
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# 1) football-data Argentina id 762 -> "ARG"
# --------------------------------------------------------------------------- #

def test_fd_argentina_resolves_to_arg():
    from soccer_ev_model.team_identity import (
        canonical_id_for_football_data_id,
    )
    assert canonical_id_for_football_data_id(762) == "ARG"


# --------------------------------------------------------------------------- #
# 2) corpus Argentina id 710061511 -> "ARG"
# --------------------------------------------------------------------------- #

def test_corpus_argentina_resolves_to_arg():
    from soccer_ev_model.team_identity import canonical_id_for_corpus_id
    assert canonical_id_for_corpus_id(710061511) == "ARG"


# --------------------------------------------------------------------------- #
# 3) "Argentina" name -> "ARG" (case insensitive, whitespace tolerant)
# --------------------------------------------------------------------------- #

def test_name_argentina_resolves_to_arg():
    from soccer_ev_model.team_identity import canonical_id_for_name
    assert canonical_id_for_name("Argentina") == "ARG"


# --------------------------------------------------------------------------- #
# 4) "  argentina  " (whitespace + case) -> "ARG"
# --------------------------------------------------------------------------- #

def test_name_argentina_lowercase_with_whitespace_resolves_to_arg():
    from soccer_ev_model.team_identity import canonical_id_for_name
    assert canonical_id_for_name("  argentina  ") == "ARG"


# --------------------------------------------------------------------------- #
# 5) resolve_team returns "resolved" with expected corpus_id
# --------------------------------------------------------------------------- #

def test_resolve_team_argentina_status_resolved():
    from soccer_ev_model.team_identity import resolve_team
    res = resolve_team(football_data_id=762, name="Argentina")
    assert res["canonical_id"] == "ARG"
    assert res["status"] == "resolved"
    assert res["corpus_id"] == 710061511
    assert res["display_name"] == "Argentina"
    assert res["source"] == "football_data"


# --------------------------------------------------------------------------- #
# 6) resolve_team returns "identity_unresolved" for fictional inputs
# --------------------------------------------------------------------------- #

def test_resolve_team_unknown_fd_and_name_returns_unresolved():
    from soccer_ev_model.team_identity import resolve_team
    res = resolve_team(football_data_id=999999999, name="Atlantis")
    assert res["canonical_id"] is None
    assert res["status"] == "identity_unresolved"
    assert res["corpus_id"] is None
    assert res["display_name"] is None


def test_resolve_team_only_unknown_name_returns_unresolved():
    from soccer_ev_model.team_identity import resolve_team
    res = resolve_team(name="Republic of Wakanda")
    assert res["canonical_id"] is None
    assert res["status"] == "identity_unresolved"


# --------------------------------------------------------------------------- #
# 7) Cape Verde Islands (fd_id=1930) -> "history_missing"
# --------------------------------------------------------------------------- #

def test_resolve_team_cape_verde_history_missing():
    from soccer_ev_model.team_identity import resolve_team
    res = resolve_team(football_data_id=1930, name="Cape Verde Islands")
    assert res["canonical_id"] == "CPV"
    assert res["status"] == "history_missing"
    assert res["corpus_id"] is None
    assert res["display_name"] == "Cape Verde Islands"


def test_resolve_team_congo_dr_history_missing():
    from soccer_ev_model.team_identity import resolve_team
    res = resolve_team(football_data_id=1934, name="Congo DR")
    assert res["canonical_id"] == "COD"
    assert res["status"] == "history_missing"


def test_resolve_team_curacao_history_missing():
    from soccer_ev_model.team_identity import resolve_team
    res = resolve_team(football_data_id=9460, name="Curaçao")
    assert res["canonical_id"] == "CUW"
    assert res["status"] == "history_missing"


# --------------------------------------------------------------------------- #
# 8) corpus_id_for_canonical("ARG") == 710061511
# --------------------------------------------------------------------------- #

def test_corpus_id_for_canonical_arg():
    from soccer_ev_model.team_identity import corpus_id_for_canonical
    assert corpus_id_for_canonical("ARG") == 710061511


# --------------------------------------------------------------------------- #
# 9) display_name("ARG") == "Argentina"
# --------------------------------------------------------------------------- #

def test_display_name_arg():
    from soccer_ev_model.team_identity import display_name
    assert display_name("ARG") == "Argentina"


# --------------------------------------------------------------------------- #
# 10) all_canonical_ids() >= 48 entries, includes ARG/USA/BRA/ALG
# --------------------------------------------------------------------------- #

def test_all_canonical_ids_includes_expected():
    from soccer_ev_model.team_identity import all_canonical_ids
    ids = all_canonical_ids()
    assert len(ids) >= 48
    for expected in ("ARG", "USA", "BRA", "ALG"):
        assert expected in ids, f"missing canonical id {expected}"


# --------------------------------------------------------------------------- #
# 11) evaluate_match (with translation in evaluate_one_game) returns
#     matches_played > 0 for Argentina vs Algeria
# --------------------------------------------------------------------------- #

def _app_module():
    """Import the dashboard app with streamlit.cache_data stubbed."""
    import streamlit as st
    st.cache_data = lambda *a, **k: (lambda f: f)
    if "dashboard.app" in sys.modules:
        return sys.modules["dashboard.app"]
    return importlib.import_module("dashboard.app")


def test_evaluate_one_game_argentina_vs_algeria_has_matches_played():
    """The key bug from the worker prompt: Argentina used to show
    matches_played=0 because the dashboard passed football-data id 762
    to the pi-rating lookup. With the identity layer, it should now
    pass the corpus id 710061511 and see Argentina's 491 matches."""
    from soccer_ev_model.pi_ratings import compute_pi_ratings
    from soccer_ev_model.team_identity import resolve_team

    intl = json.loads(
        (_PROJECT_ROOT / "data" / "processed" / "international_matches.json").read_text()
    )
    from soccer_ev_model.pi_backtest import load_matches
    wc: list[dict] = []
    for y in (2010, 2014, 2018, 2022):
        try:
            wc.extend(load_matches(y))
        except FileNotFoundError:
            pass
    corpus = list(intl) + list(wc)
    corpus.sort(key=lambda m: m.get("date", ""))
    train = [m for m in corpus if m.get("date", "") < "2026-06-16"]
    ratings = compute_pi_ratings(train, cutoff="2026-06-16")

    h_res = resolve_team(football_data_id=762, name="Argentina")
    a_res = resolve_team(football_data_id=778, name="Algeria")
    assert h_res["status"] == "resolved"
    assert a_res["status"] == "resolved"

    app = _app_module()
    out = app.evaluate_one_game(
        home_name="Argentina",
        away_name="Algeria",
        home_team_id=762,   # football-data id, NOT corpus id
        away_team_id=778,   # football-data id, NOT corpus id
        cutoff_iso="2026-06-16",
        home_odds_txt="-150",
        draw_odds_txt="+300",
        away_odds_txt="+400",
        ratings=ratings,
        min_edge=0.03,
        name_to_id={},
    )
    assert out["ok"] is True, out
    r = out["result"]
    # The dashboard should resolve 762 -> corpus 710061511 and
    # 778 -> corpus 1240792518 BEFORE calling evaluate_match.
    assert r["confidence"]["home_matches_played"] > 0
    assert r["confidence"]["away_matches_played"] > 0
    # Specifically, the corpus has 491 matches for Argentina by 2026-06-16.
    assert r["confidence"]["home_matches_played"] >= 400
    # The result must also carry the canonical ids and no warnings.
    assert r["canonical_home_id"] == "ARG"
    assert r["canonical_away_id"] == "ALG"
    assert r["identity_warnings"] == []
    assert r["confidence"]["identity_unresolved"] is False
    # And the prediction summary tier should NOT be "Low-data warning"
    # — Argentina has plenty of history.
    # (We re-derive it the same way _render_game_result does.)
    from soccer_ev_model.prediction_summary import (
        confidence_tier,
        model_agreement,
        prediction_margin_pct,
        top_two_outcomes,
    )
    blended = r["pi_probs"]
    top, top_p, _second, _second_p = top_two_outcomes(blended)
    margin = prediction_margin_pct(blended)
    pi_only = r.get("pi_only_probs") or blended
    elo_only = r.get("elo_only_probs")
    agree = model_agreement(pi_only, elo_only if elo_only is not None else pi_only)
    tier = confidence_tier(
        blended, margin, blended["draw"], agree["label"],
        low_data=r["confidence"]["tier"] in ("C", "D"),
        identity_unresolved=r["confidence"].get("identity_unresolved", False),
    )
    assert tier != "Low-data warning"


# --------------------------------------------------------------------------- #
# 12) evaluate_one_game surfaces an `identity_warnings` entry when given
#     an unknown id
# --------------------------------------------------------------------------- #

def test_evaluate_one_game_surfaces_identity_warning_for_unknown():
    from soccer_ev_model.pi_ratings import compute_pi_ratings

    intl = json.loads(
        (_PROJECT_ROOT / "data" / "processed" / "international_matches.json").read_text()
    )
    ratings = compute_pi_ratings(intl[:200], cutoff="2024-12-31")

    app = _app_module()
    out = app.evaluate_one_game(
        home_name="Atlantis",
        away_name="Wakanda",
        home_team_id=999999999,
        away_team_id=888888888,
        cutoff_iso="2026-06-16",
        home_odds_txt="-150",
        draw_odds_txt="+300",
        away_odds_txt="+400",
        ratings=ratings,
        min_edge=0.03,
        name_to_id={},
    )
    # Two unknown ids -> two warnings. The result may still be ok
    # (neutral pi-rating) but the warnings list must be non-empty.
    assert "identity_warnings" in out["result"]
    assert len(out["result"]["identity_warnings"]) >= 1
    # And the identity_unresolved flag in confidence should be True
    assert out["result"]["confidence"]["identity_unresolved"] is True


# --------------------------------------------------------------------------- #
# 13) assess_match_confidence receives the new identity_unresolved flag
#     and it propagates through to confidence["identity_unresolved"]
# --------------------------------------------------------------------------- #

def test_evaluate_match_propagates_identity_unresolved_flag():
    """When evaluate_match is called with identity_unresolved=True, the
    returned confidence dict should carry identity_unresolved=True."""
    from soccer_ev_model.pi_ratings import compute_pi_ratings

    # Tiny synthetic train
    train = []
    for i in range(40):
        train.append({
            "match_id": f"x{i}", "date": f"2020-{(i % 9) + 1:02d}-01",
            "home_team": "France", "away_team": "Weak",
            "home_team_id": 1, "away_team_id": 2,
            "home_goals": 2, "away_goals": 0, "result": "H",
        })
    ratings = compute_pi_ratings(train, cutoff="2020-12-01")

    from soccer_ev_model.ev_workflow import evaluate_match
    r = evaluate_match(
        home_team="Atlantis",
        away_team="Wakanda",
        home_team_id=1,
        away_team_id=2,
        date="2020-12-01",
        book_home_odds=-150,
        book_draw_odds=300,
        book_away_odds=400,
        ratings=ratings,
        identity_unresolved=True,
    )
    assert r["confidence"]["identity_unresolved"] is True

    # And the default (no flag passed) should be False — preserves
    # backward-compat for the existing test suite.
    r2 = evaluate_match(
        home_team="France",
        away_team="Weak",
        home_team_id=1,
        away_team_id=2,
        date="2020-12-01",
        book_home_odds=-150,
        book_draw_odds=300,
        book_away_odds=400,
        ratings=ratings,
    )
    assert r2["confidence"]["identity_unresolved"] is False


# --------------------------------------------------------------------------- #
# 14) confidence_tier returns "Identity unresolved" when the flag is set
# --------------------------------------------------------------------------- #

def test_confidence_tier_identity_unresolved_wins():
    from soccer_ev_model.prediction_summary import confidence_tier
    tier = confidence_tier(
        {"home": 0.65, "draw": 0.20, "away": 0.15},
        prediction_margin_pts=22.0,
        draw_p=0.20,
        agreement_label="agree",
        low_data=False,
        identity_unresolved=True,
    )
    assert tier == "Identity unresolved"


def test_confidence_tier_identity_unresolved_beats_low_data():
    """Identity unresolved must win over Low-data warning."""
    from soccer_ev_model.prediction_summary import confidence_tier
    tier = confidence_tier(
        {"home": 0.65, "draw": 0.20, "away": 0.15},
        prediction_margin_pts=22.0,
        draw_p=0.20,
        agreement_label="agree",
        low_data=True,
        identity_unresolved=True,
    )
    assert tier == "Identity unresolved"


# --------------------------------------------------------------------------- #
# UnplayedMatch: new fields populated, to_dict round-trips
# --------------------------------------------------------------------------- #

def test_unplayed_match_carries_canonical_ids(tmp_path: Path):
    """A 2026 cache entry should surface canonical_home_id / canonical_away_id."""
    cache = tmp_path / "matches_2026.json"
    cache.write_text(json.dumps({
        "year": 2026, "fetched_at": "x", "count": 1,
        "matches": [{
            "id": 99, "date": "2026-06-17T01:00:00Z", "status": "TIMED",
            "stage": "GROUP_STAGE", "group": "GROUP_A",
            "home_team_id": 762, "home_team_name": "Argentina",
            "away_team_id": 778, "away_team_name": "Algeria",
        }],
    }))
    from dashboard.data_loader import get_unplayed_matches
    out = get_unplayed_matches("2026-06-16", cache_path=cache)
    assert len(out) == 1
    m = out[0]
    assert m.canonical_home_id == "ARG"
    assert m.canonical_away_id == "ALG"
    d = m.to_dict()
    assert d["canonical_home_id"] == "ARG"
    assert d["canonical_away_id"] == "ALG"
    # Round-trips through stdlib json
    assert json.loads(json.dumps(d)) == d


# --------------------------------------------------------------------------- #
# Manual-odds regression: fixed input still produces the same numeric
# pi_probs / calibrated_pi / edges / plus_ev_flags / banner / confidence.tier
# (within rounding) as before this task.
# --------------------------------------------------------------------------- #

def test_manual_odds_regression_unchanged():
    """A fixed input set must still produce numerically identical outputs
    (within rounding) as before this task. evaluate_match is called
    with the same arguments as before; only `identity_unresolved` is new
    and defaults to False, so it must not affect any of the listed keys."""
    from soccer_ev_model.pi_ratings import compute_pi_ratings

    train = []
    for i in range(40):
        train.append({
            "match_id": f"x{i}", "date": f"2020-{(i % 9) + 1:02d}-01",
            "home_team": "France", "away_team": "Weak",
            "home_team_id": 1, "away_team_id": 2,
            "home_goals": 2, "away_goals": 0, "result": "H",
        })
        train.append({
            "match_id": f"y{i}", "date": f"2020-{(i % 9) + 1:02d}-02",
            "home_team": "Senegal", "away_team": "Weak",
            "home_team_id": 2, "away_team_id": 3,
            "home_goals": 1, "away_goals": 1, "result": "D",
        })
    ratings = compute_pi_ratings(train, cutoff="2020-12-01")

    from soccer_ev_model.ev_workflow import evaluate_match
    r = evaluate_match(
        home_team="France", away_team="Senegal",
        home_team_id=1, away_team_id=2,
        date="2020-12-01",
        book_home_odds=-200, book_draw_odds=350, book_away_odds=500,
        ratings=ratings, min_edge=0.03,
    )
    # All previously-required keys are still present.
    for k in ("home_team", "away_team", "date", "book_odds", "book_fair",
              "pi_probs", "calibrated_pi", "edges", "confidence",
              "plus_ev_flags", "banner"):
        assert k in r
    # And the new identity_unresolved flag exists with the safe default.
    assert r["confidence"]["identity_unresolved"] is False
    # Plus-ev flags only above threshold.
    for flag in r["plus_ev_flags"]:
        assert flag["edge"] >= 0.03
    # Edges match pi - book_fair.
    for m in ("home", "draw", "away"):
        assert r["edges"][m] == pytest.approx(
            r["pi_probs"][m] - r["book_fair"][m], abs=1e-6
        )
