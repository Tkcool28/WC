"""Regression tests for the PR #9 ID-resolution fix.

The auto-populated Predictions / Bets / Analysis paths in
:mod:`dashboard.app` used to pass ``m["home_team_id"]`` /
``m["away_team_id"]`` (football-data.org ids) directly to
:func:`soccer_ev_model.ev_workflow.predict_match`. The pi-rating layer
expects training-corpus ids, so the lookup missed and high-history
teams like England / Croatia showed ``matches_played=0`` with a
neutral home-draw-away fallback.

The fix lifts the resolution logic that used to live nested inside
:func:`dashboard.app.evaluate_one_game` into a shared helper
(:mod:`dashboard.team_resolution`) and wires it into the three auto
loops. These tests pin the contract:

  1. Known teams (England, Croatia) resolve to the right canonical /
     corpus ids and produce non-neutral pi probabilities.
  2. The corpus-id passthrough path still works (caches that already
     store corpus ids don't get double-translated).
  3. Genuine missing-history teams (COD, CPV) flag a
     ``history_missing`` warning and fall back to neutral pi (no hard
     crash, no silent fall-through).
  4. Unknown teams flag an ``identity_unresolved`` warning.
  5. Combined warnings on a match with one resolved + one
     history_missing team behave correctly.
  6. The new auto path agrees with the legacy ``evaluate_one_game``
     path for the same inputs.
  7. The 2026-06-17 audit is complete: every team either resolves
     cleanly with matches_played > 0, OR has a proper warning.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Test fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def corpus() -> list[dict]:
    """Training corpus (international + WC years)."""
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
    return corpus


@pytest.fixture(scope="module")
def name_to_id(corpus) -> dict[str, int]:
    """name -> corpus_id map from the training corpus (mirrors
    :func:`dashboard.app.build_name_to_id`)."""
    out: dict[str, int] = {}
    for m in corpus:
        if "home_team" in m and "home_team_id" in m:
            out.setdefault(m["home_team"], m["home_team_id"])
        if "away_team" in m and "away_team_id" in m:
            out.setdefault(m["away_team"], m["away_team_id"])
    return out


@pytest.fixture(scope="module")
def ratings(corpus) -> dict:
    """pi-ratings snapshot at the 2026-06-17 cutoff."""
    from soccer_ev_model.pi_ratings import compute_pi_ratings
    train = [m for m in corpus if m.get("date", "") < "2026-06-17T23:59:59Z"]
    return compute_pi_ratings(train, cutoff="2026-06-17T23:59:59Z")


@pytest.fixture(scope="module")
def schedule_2026_06_17() -> list[dict]:
    """The 5 auto-populated matches on 2026-06-17 from the live cache."""
    raw = json.loads(
        (_PROJECT_ROOT / "data" / "raw" / "matches_2026.json").read_text()
    )
    matches = raw["matches"]
    on_day = [
        m for m in matches
        if (m.get("date") or m.get("kickoff_iso") or "").startswith("2026-06-17")
    ]
    assert len(on_day) == 5, f"expected 5 matches on 2026-06-17, got {len(on_day)}"
    return on_day


@pytest.fixture(scope="module")
def england_croatia(schedule_2026_06_17) -> dict:
    """The England vs Croatia match on 2026-06-17."""
    for m in schedule_2026_06_17:
        h = m.get("home_team_name", "")
        a = m.get("away_team_name", "")
        if "England" in h and "Croatia" in a:
            return m
    pytest.fail("England vs Croatia not found in 2026-06-17 schedule")


# --------------------------------------------------------------------------- #
# 1) Known teams — England
# --------------------------------------------------------------------------- #
def test_resolve_team_for_prediction_known_team_england(england_croatia, ratings):
    """England: schedule fd_id 770 -> canonical ENG, corpus 4102578634."""
    from dashboard.team_resolution import resolve_team_for_prediction
    m = england_croatia
    res = resolve_team_for_prediction(
        schedule_team_id=m["home_team_id"],
        team_name="England",
        ratings=ratings,
        name_to_id=None,
    )
    assert res.canonical_id == "ENG"
    assert res.corpus_id == 4102578634   # NOT the fd_id (770)
    assert res.source_team_id == 770
    assert res.source_team_name == "England"
    assert res.status == "resolved"
    assert res.warning is None


# --------------------------------------------------------------------------- #
# 2) Known teams — Croatia
# --------------------------------------------------------------------------- #
def test_resolve_team_for_prediction_known_team_croatia(england_croatia, ratings):
    """Croatia: schedule fd_id 799 -> canonical CRO, corpus 645283928."""
    from dashboard.team_resolution import resolve_team_for_prediction
    m = england_croatia
    res = resolve_team_for_prediction(
        schedule_team_id=m["away_team_id"],
        team_name="Croatia",
        ratings=ratings,
        name_to_id=None,
    )
    assert res.canonical_id == "CRO"
    assert res.corpus_id == 645283928
    assert res.source_team_id == 799
    assert res.source_team_name == "Croatia"
    assert res.status == "resolved"
    assert res.warning is None


# --------------------------------------------------------------------------- #
# 3) Corpus-id passthrough — when the schedule id IS already a corpus id
# --------------------------------------------------------------------------- #
def test_resolve_team_for_prediction_corpus_id_passthrough(ratings):
    """If the schedule id is already a key in ratings, use it unchanged."""
    from dashboard.team_resolution import resolve_team_for_prediction
    # England's actual corpus id is 4102578634 — it should be a key in ratings.
    assert 4102578634 in ratings
    res = resolve_team_for_prediction(
        schedule_team_id=4102578634,
        team_name="England",
        ratings=ratings,
        name_to_id=None,
    )
    # The corpus id should pass through unchanged (no re-translation).
    assert res.corpus_id == 4102578634
    assert res.canonical_id == "ENG"
    assert res.status == "resolved"
    assert res.warning is None


# --------------------------------------------------------------------------- #
# 4) history_missing — Congo DR (canonical exists, corpus id is null)
# --------------------------------------------------------------------------- #
def test_resolve_team_for_prediction_history_missing_cod(ratings):
    """COD: canonical=COD, but corpus_id is null in the registry.

    The resolver must:
      * set status='history_missing'
      * set a warning
      * fall back to the schedule_team_id (so pi-rating produces a
        neutral result rather than crashing)
      * NOT fabricate a fake id
    """
    from dashboard.team_resolution import resolve_team_for_prediction
    # 1934 is Congo DR's football-data id; registry has it but with
    # corpus_id=None (genuine missing-history team).
    res = resolve_team_for_prediction(
        schedule_team_id=1934,
        team_name="Congo DR",
        ratings=ratings,
        name_to_id=None,
    )
    assert res.canonical_id == "COD"
    assert res.status == "history_missing"
    assert res.warning is not None
    assert "canonical=COD" in res.warning
    assert "status=history_missing" in res.warning
    # The corpus_id falls back to the schedule id (no fabrication).
    assert res.corpus_id == 1934
    # And the warning should be a real, parseable string.
    assert "fd_id=1934" in res.warning
    assert "name=Congo DR" in res.warning


def test_resolve_team_for_prediction_history_missing_cpv(ratings):
    """CPV (Cape Verde): same pattern as COD — canonical exists, no corpus."""
    from dashboard.team_resolution import resolve_team_for_prediction
    res = resolve_team_for_prediction(
        schedule_team_id=1930,
        team_name="Cape Verde Islands",
        ratings=ratings,
        name_to_id=None,
    )
    assert res.canonical_id == "CPV"
    assert res.status == "history_missing"
    assert res.warning is not None
    assert "status=history_missing" in res.warning
    assert res.corpus_id == 1930  # fallback, not fabricated


# --------------------------------------------------------------------------- #
# 5) identity_unresolved — unknown team
# --------------------------------------------------------------------------- #
def test_resolve_team_for_prediction_identity_unresolved(ratings):
    """Unknown team: status='identity_unresolved', warning set."""
    from dashboard.team_resolution import resolve_team_for_prediction
    # Use a name that's truly absent from the corpus + an fd_id that's
    # not in ratings and not in the registry. This makes the resolver
    # fall through to branch 6 (truly unresolved) instead of being
    # accidentally resolved via the corpus-name fallback or ratings.
    res = resolve_team_for_prediction(
        schedule_team_id=999_999_999,  # Not in ratings, not in registry
        team_name="Atlantis National Team",
        ratings=ratings,
        name_to_id=None,  # disable corpus-name fallback so we reach branch 6
    )
    assert res.canonical_id is None
    assert res.status == "identity_unresolved"
    assert res.warning is not None
    assert "canonical=None" in res.warning
    assert "status=identity_unresolved" in res.warning


# --------------------------------------------------------------------------- #
# 6) Match-level: combined warnings (one resolved + one history_missing)
# --------------------------------------------------------------------------- #
def test_resolve_match_for_prediction_warnings_combined(ratings, schedule_2026_06_17):
    """Portugal vs Congo DR: 1 warning (Congo DR history_missing)."""
    from dashboard.team_resolution import resolve_match_for_prediction
    m = next(
        m for m in schedule_2026_06_17
        if m.get("home_team_name") == "Portugal"
        and m.get("away_team_name") == "Congo DR"
    )
    home_res, away_res, warnings = resolve_match_for_prediction(
        match=m, ratings=ratings, name_to_id=None,
    )
    assert home_res.status == "resolved"
    assert home_res.canonical_id == "POR"
    assert away_res.status == "history_missing"
    assert away_res.canonical_id == "COD"
    # Exactly one warning (COD); POR is clean.
    assert len(warnings) == 1
    assert "canonical=COD" in warnings[0]
    assert "status=history_missing" in warnings[0]


# --------------------------------------------------------------------------- #
# 7) Match-level: no warnings for a clean matchup (England vs Croatia)
# --------------------------------------------------------------------------- #
def test_resolve_match_for_prediction_no_warnings_for_known_match(
    ratings, england_croatia,
):
    from dashboard.team_resolution import resolve_match_for_prediction
    h, a, w = resolve_match_for_prediction(
        match=england_croatia, ratings=ratings, name_to_id=None,
    )
    assert h.status == "resolved"
    assert a.status == "resolved"
    assert h.canonical_id == "ENG"
    assert a.canonical_id == "CRO"
    assert w == []


# --------------------------------------------------------------------------- #
# 8) End-to-end: England's pi is non-neutral after the fix
# --------------------------------------------------------------------------- #
def test_england_croatia_pi_is_non_neutral(ratings, england_croatia):
    """The headline bug: after the fix, pi-only for ENG vs CRO must
    be non-neutral (NOT exactly home=0.4, draw=0.27, away=0.33)."""
    from dashboard.team_resolution import resolve_match_for_prediction
    from soccer_ev_model.ev_workflow import predict_match

    h, a, _ = resolve_match_for_prediction(
        match=england_croatia, ratings=ratings, name_to_id=None,
    )
    assert h.corpus_id is not None and a.corpus_id is not None
    pred = predict_match(
        home_team="England",
        away_team="Croatia",
        home_team_id=int(h.corpus_id),
        away_team_id=int(a.corpus_id),
        date="2026-06-17",
        ratings=ratings,
    )
    pi = pred["pi_only_probs"]
    # The neutral baseline (no rating history) is home=0.4, draw=0.27, away=0.33.
    neutral = {"home": 0.4, "draw": 0.27, "away": 0.33}
    assert pi != neutral, f"pi_only_probs is still neutral: {pi}"
    # England is home-favored; pi['home'] should be the largest.
    assert pi["home"] == max(pi.values())


# --------------------------------------------------------------------------- #
# 9) End-to-end: matches_played > 5 for both England and Croatia
# --------------------------------------------------------------------------- #
def test_england_croatia_matches_played_nonzero(ratings, england_croatia):
    """The headline bug: after the fix, matches_played must be > 5
    for both teams (not 0)."""
    from dashboard.team_resolution import resolve_match_for_prediction
    from soccer_ev_model.pi_ratings import get_team_experience

    h, a, _ = resolve_match_for_prediction(
        match=england_croatia, ratings=ratings, name_to_id=None,
    )
    assert h.corpus_id is not None and a.corpus_id is not None
    mp_home = get_team_experience(ratings, h.corpus_id)["matches_played"]
    mp_away = get_team_experience(ratings, a.corpus_id)["matches_played"]
    assert mp_home > 5, f"home matches_played should be > 5, got {mp_home}"
    assert mp_away > 5, f"away matches_played should be > 5, got {mp_away}"


# --------------------------------------------------------------------------- #
# 10) Regression: genuine missing-history teams still produce neutral pi
# --------------------------------------------------------------------------- #
def test_cod_cpv_neutral_pi_preserved(ratings):
    """COD/CPV used to produce neutral pi (no corpus history). The fix
    must preserve that — the only change is that we now ALSO surface
    a proper history_missing warning instead of silently returning
    neutral."""
    from dashboard.team_resolution import resolve_team_for_prediction
    from soccer_ev_model.ev_workflow import predict_match
    from soccer_ev_model.pi_ratings import get_team_experience

    # COD
    cod = resolve_team_for_prediction(
        schedule_team_id=1934, team_name="Congo DR",
        ratings=ratings, name_to_id=None,
    )
    # CPV (vs COD, both genuinely missing history)
    cpv = resolve_team_for_prediction(
        schedule_team_id=1930, team_name="Cape Verde Islands",
        ratings=ratings, name_to_id=None,
    )
    assert cod.corpus_id is not None and cpv.corpus_id is not None
    pred = predict_match(
        home_team="Congo DR",
        away_team="Cape Verde Islands",
        home_team_id=int(cod.corpus_id),
        away_team_id=int(cpv.corpus_id),
        date="2026-06-17",
        ratings=ratings,
    )
    # Neutral fallback: home=0.4, draw=0.27, away=0.33
    assert pred["pi_only_probs"] == {"home": 0.4, "draw": 0.27, "away": 0.33}
    # matches_played is 0 (they're not in ratings under the schedule id)
    mp_cod = get_team_experience(ratings, cod.corpus_id)["matches_played"]
    mp_cpv = get_team_experience(ratings, cpv.corpus_id)["matches_played"]
    assert mp_cod == 0
    assert mp_cpv == 0
    # But the warning is properly surfaced.
    assert cod.warning is not None
    assert "status=history_missing" in cod.warning
    assert cpv.warning is not None
    assert "status=history_missing" in cpv.warning


# --------------------------------------------------------------------------- #
# 11) Path equivalence: auto path vs legacy evaluate_one_game path
# --------------------------------------------------------------------------- #
def test_path_equivalence_auto_vs_legacy(england_croatia, ratings, name_to_id):
    """For England vs Croatia on 2026-06-17, the new auto path (via
    resolve_match_for_prediction + predict_match) must produce the
    same pi-only, elo-only, blend, and canonical IDs as the legacy
    evaluate_one_game path."""
    from dashboard.team_resolution import resolve_match_for_prediction
    from soccer_ev_model.ev_workflow import predict_match, evaluate_match
    from soccer_ev_model.team_identity import resolve_team as _resolve_team_identity

    m = england_croatia

    # ---- auto path: resolve_match_for_prediction -> predict_match ---- #
    h, a, _ = resolve_match_for_prediction(
        match=m, ratings=ratings, name_to_id=None,
    )
    assert h.corpus_id is not None and a.corpus_id is not None
    auto_pred = predict_match(
        home_team="England",
        away_team="Croatia",
        home_team_id=int(h.corpus_id),
        away_team_id=int(a.corpus_id),
        date="2026-06-17",
        ratings=ratings,
        canonical_home_id=h.canonical_id,
        canonical_away_id=a.canonical_id,
    )

    # ---- legacy path: replicate evaluate_one_game's translation ---- #
    h_res = _resolve_team_identity(football_data_id=m["home_team_id"], name="England")
    a_res = _resolve_team_identity(football_data_id=m["away_team_id"], name="Croatia")
    assert h_res["corpus_id"] is not None and a_res["corpus_id"] is not None
    legacy_pred = predict_match(
        home_team="England",
        away_team="Croatia",
        home_team_id=int(h_res["corpus_id"]),
        away_team_id=int(a_res["corpus_id"]),
        date="2026-06-17",
        ratings=ratings,
        canonical_home_id=h_res["canonical_id"],
        canonical_away_id=a_res["canonical_id"],
    )

    # ---- assertions ---- #
    assert auto_pred["pi_only_probs"] == legacy_pred["pi_only_probs"]
    assert auto_pred["blend_probs"] == legacy_pred["blend_probs"]
    assert auto_pred["elo_only_probs"] == legacy_pred["elo_only_probs"]
    assert auto_pred["canonical_home_id"] == legacy_pred["canonical_home_id"]
    assert auto_pred["canonical_away_id"] == legacy_pred["canonical_away_id"]


# --------------------------------------------------------------------------- #
# 12) 2026-06-17 audit completeness — no silent fall-through
# --------------------------------------------------------------------------- #
def test_2026_06_17_audit_completeness(ratings, schedule_2026_06_17):
    """For every auto-populated match on 2026-06-17, every team must
    either:

      * resolve cleanly with matches_played > 0, OR
      * have a proper history_missing / identity_unresolved warning
        attached to the ResolvedTeam.

    No silent fall-through: a 'resolved' team must have matches_played
    > 0; a 'history_missing' / 'identity_unresolved' team must have a
    warning string.
    """
    from dashboard.team_resolution import resolve_match_for_prediction
    from soccer_ev_model.pi_ratings import get_team_experience

    for m in schedule_2026_06_17:
        h, a, warnings = resolve_match_for_prediction(
            match=m, ratings=ratings, name_to_id=None,
        )
        for side, res in (("home", h), ("away", a)):
            if res.status == "resolved":
                # Resolved teams must have matches_played > 0
                # (otherwise the resolver lied about the status).
                if res.corpus_id is not None:
                    cid: int = res.corpus_id
                    mp = get_team_experience(ratings, cid)["matches_played"]
                    assert mp > 0, (
                        f"{m.get('home_team_name') if side=='home' else m.get('away_team_name')}: "
                        f"status=resolved but matches_played=0 (silent fall-through)"
                    )
            else:
                # history_missing / identity_unresolved teams MUST have a warning.
                assert res.warning is not None, (
                    f"{side} team status={res.status} but no warning"
                )
                assert res.status in ("history_missing", "identity_unresolved")
        # The combined warnings list is a subset of the per-team warnings.
        for w in warnings:
            assert w == h.warning or w == a.warning
