"""Dashboard "context cards" — Tournament Snapshot + Highest Model Confidence.

Phase 8 (dashboard context polish) introduces two new top-of-page cards
that sit between the top-level nav and the per-view body:

* **Tournament Snapshot** — a one-line summary of the *currently
  loaded* matches (count, stage, matchday, group list).
* **Highest Model Confidence** — the single highest model probability
  across all loaded predictions (regardless of which view is active).

This module is the single source of truth for those two cards' content
rules.  The functions are split into:

* **Pure helpers** — :func:`build_tournament_snapshot`,
  :func:`pick_smart_default_date`, :func:`highest_model_confidence`.
  These take plain Python data and return plain Python data, so they
  can be unit-tested without a running Streamlit session.
* **Thin renderers** — :func:`render_tournament_snapshot` and
  :func:`render_highest_confidence`.  Each emits one ``.wc-card``
  Streamlit block.  The renderers are intentionally tiny — they only
  translate the helper output into ``st.markdown`` / ``st.caption``
  calls.  All content decisions live in the helpers so the tests can
  lock them down.

The smart date default (Feature 2 of the brief) is a *pure* helper
(``pick_smart_default_date``) — the wire-up into
``st.date_input(..., value=...)`` lives in :mod:`dashboard.app`.

The cards read from ``st.session_state`` via the namespaced keys in
:mod:`dashboard.session_state` (``KEYS.LOADED_MATCHES`` and
``KEYS.PREDICTIONS_BY_MATCH``).  On a fresh visit, those keys are
absent, so the cards render the calm placeholder copy.  Once the user
clicks "Show Predictions" the session state populates and the cards
re-render with real data.

Phase 9 (dashboard autoload) adds :func:`autoload_context_for_date`
and its pure helper :func:`_autoload_pure` so the cards populate on
page open *without* requiring the user to click "Show Predictions".
The helper writes to the same keys the per-view renderers already
read, so the existing cards remain the single source of truth.
"""
from __future__ import annotations

from datetime import date as _date
from typing import Any, Callable, Optional

import streamlit as st

from dashboard.session_state import KEYS
from dashboard.text_format import (
    format_group_label as _format_group_label,
    format_matchday_label as _format_matchday_label,
)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
#: Maximum number of groups to enumerate in the snapshot sub-line
#: before falling back to the "A, B, C and N more" truncation.
_MAX_GROUPS_IN_SNAPSHOT = 3

#: Header shown by the snapshot renderer when the matches list is empty
#: (or yields a count of 0).
_SNAPSHOT_UNAVAILABLE_TEXT = "Tournament snapshot unavailable"

#: Caption shown by the highest-confidence renderer when no predictions
#: are available yet (e.g. fresh visit before the user clicks
#: "Show Predictions").
_CONFIDENCE_PLACEHOLDER_TEXT = (
    "Run predictions to see today's highest-confidence result."
)
#: Caption shown beneath the highest-confidence number.  Explicit
#: non-betting language so we don't accidentally nudge the user into
#: treating the model probability as a tip.
_CONFIDENCE_DISCLAIMER_TEXT = (
    "Based on model probability only. Not a betting recommendation."
)


# --------------------------------------------------------------------------- #
# Pure helper: Tournament Snapshot
# --------------------------------------------------------------------------- #
def build_tournament_snapshot(matches: list[dict]) -> dict:
    """Return a snapshot dict describing the loaded matches.

    The returned dict is consumed by :func:`render_tournament_snapshot`
    and has the following keys (all guaranteed to exist; missing
    metadata collapses to the documented fallback strings):

    * ``count`` — int, number of matches in the input list.
    * ``stage`` — raw stage code (e.g. ``"GROUP_STAGE"``) or ``""`` if
      no match has a stage.
    * ``matchday`` — int matchday (1/2/3) for group-stage days or
      ``None`` otherwise.
    * ``matchdays`` — sorted list of *all* matchdays seen (used by the
      renderer for the multi-matchday sub-line).
    * ``header`` — the headline string for the card.  Examples:
      ``"Group Stage · Matchday 2"``, ``"Final"``, ``"Tournament snapshot"``.
    * ``count_label`` — match-count sub-line.  Uses
      "X matches scheduled today" for group-stage and
      "X matches remaining" for knockout / final, per the brief.
    * ``groups_label`` — the one-line list of groups (e.g. ``"Groups A, B, C"``
      or ``"Groups A, B, C and 3 more"``), or ``""`` for non-group
      stages or when no group metadata is present.
    * ``groups_full`` — sorted list of every group code seen (used by
      the renderer; may be longer than the truncated groups_label).
    * ``is_empty`` — bool, True if ``count == 0`` so the renderer
      can pick the placeholder copy.

    Rule for multi-matchday group-stage days (documented in
    ``tests/test_context_cards.py::test_group_stage_multiple_matchdays_same_day``):
    the header uses the *earliest* matchday present, and
    ``matchdays`` carries the full sorted set so the renderer can list
    each one in a sub-line.
    """
    count = len(matches or [])

    # ---- Pull stage / matchday / group metadata from the input ---- #
    stages: list[str] = []
    matchdays_seen: list[int] = []
    groups_seen: list[str] = []
    for m in matches or []:
        stage = (m.get("stage") or "").strip() if isinstance(m, dict) else ""
        if stage:
            stages.append(stage)
        md = m.get("matchday") if isinstance(m, dict) else None
        if isinstance(md, int):
            matchdays_seen.append(md)
        grp = (m.get("group") or "").strip() if isinstance(m, dict) else ""
        if grp:
            groups_seen.append(grp)

    # ---- Pick a representative stage / matchday ---- #
    # We use the first non-empty stage we saw (the matches list is
    # already sorted by kickoff in the loader, so this is the
    # chronologically-earliest match on the day).
    stage = stages[0] if stages else ""
    # The earliest matchday is the one that "starts the day".
    matchday = min(matchdays_seen) if matchdays_seen else None
    all_matchdays = sorted(set(matchdays_seen))

    # ---- Build the header string ---- #
    if not stage and matchday is None:
        # Missing stage/metadata: graceful fallback.
        header = "Tournament snapshot"
    else:
        # Reuse the existing text_format helper for the human label.
        header = _format_matchday_label(stage, matchday)

    # ---- Count label depends on stage type ---- #
    is_group_stage = stage.upper() == "GROUP_STAGE" if stage else False
    is_knockout_or_final = bool(stage) and not is_group_stage
    if is_group_stage:
        count_label = _pluralize(count, "match scheduled today")
    elif is_knockout_or_final:
        count_label = _pluralize(count, "match remaining")
    else:
        # Stage missing or unknown → use the neutral "scheduled today" form.
        count_label = _pluralize(count, "match scheduled today")

    # ---- Groups label ---- #
    unique_groups = sorted(set(groups_seen))
    if is_group_stage and unique_groups:
        groups_label = _format_groups_list(unique_groups)
    else:
        groups_label = ""

    return {
        "count": count,
        "stage": stage,
        "matchday": matchday,
        "matchdays": all_matchdays,
        "header": header,
        "count_label": count_label,
        "groups_label": groups_label,
        "groups_full": unique_groups,
        "is_empty": count == 0,
    }


def _pluralize(n: int, singular_phrase: str) -> str:
    """``_pluralize(1, "match remaining")`` → ``"1 match remaining"``,
    ``_pluralize(5, "match remaining")`` → ``"5 matches remaining"``."""
    word, _, rest = singular_phrase.partition(" ")
    plural_word = word + ("es" if not word.endswith("s") else "")
    return f"{n} {word if n == 1 else plural_word} {rest}".strip()


def _format_groups_list(groups: list[str]) -> str:
    """Format a sorted list of group codes for the snapshot sub-line.

    Examples
    --------
    >>> _format_groups_list(["A"])
    'Group A'
    >>> _format_groups_list(["A", "B", "C"])
    'Groups A, B, C'
    >>> _format_groups_list(["A", "B", "C", "D", "E", "F"])
    'Groups A, B, C and 3 more'
    """
    if not groups:
        return ""
    # Strip the GROUP_ prefix that format_group_label already handles.
    letters: list[str] = []
    for g in groups:
        gu = g.upper()
        letters.append(gu[6:] if gu.startswith("GROUP_") else gu)
    if len(letters) == 1:
        return f"Group {letters[0]}"
    if len(letters) <= _MAX_GROUPS_IN_SNAPSHOT:
        return "Groups " + ", ".join(letters)
    head = ", ".join(letters[:_MAX_GROUPS_IN_SNAPSHOT])
    remaining = len(letters) - _MAX_GROUPS_IN_SNAPSHOT
    return f"Groups {head} and {remaining} more"


# --------------------------------------------------------------------------- #
# Pure helper: Smart Date Default
# --------------------------------------------------------------------------- #
def pick_smart_default_date(
    today: _date, available_dates: list[str]
) -> _date:
    """Pick the best default date for the date_input widget.

    Rules (in priority order):

    1. If ``available_dates`` is empty, return ``today`` (defensive —
       no schedule on disk at all, the user can still pick any date).
    2. If ``today_iso`` is in ``available_dates``, return ``today``.
    3. Otherwise find the nearest future date ``>= today``; if any
       exists, return it.
    4. Otherwise return the most recent past date ``<= today``.

    ``available_dates`` is expected to be a list of ``"YYYY-MM-DD"``
    strings (the format :func:`dashboard.data_loader.list_dates_with_unplayed`
    returns).  Unparseable entries are silently skipped.
    """
    if not available_dates:
        return today

    parsed: list[_date] = []
    for s in available_dates:
        try:
            parsed.append(_date.fromisoformat(str(s).strip()))
        except (ValueError, TypeError, AttributeError):
            continue
    if not parsed:
        return today

    today_iso = today.isoformat()
    if today_iso in {d.isoformat() for d in parsed}:
        return today

    future = sorted(d for d in parsed if d >= today)
    if future:
        return future[0]

    past = sorted((d for d in parsed if d < today), reverse=True)
    if past:
        return past[0]

    # Defensive fallthrough: parsed list was non-empty but every entry
    # compared oddly. Fall back to today so we never crash the page.
    return today


# --------------------------------------------------------------------------- #
# Pure helper: Highest Model Confidence
# --------------------------------------------------------------------------- #
def _probs_for(prediction: dict) -> dict:
    """Return the canonical display probs dict for a prediction.

    Prefers ``blend_probs`` (the canonical display probs from
    ``predict_match``), then ``pi_probs`` as a fallback — same
    priority as :func:`dashboard.prediction_card._extract_most_likely`.
    """
    return (
        prediction.get("blend_probs")
        or prediction.get("pi_probs")
        or {}
    )


def highest_model_confidence(
    matches: list[dict],
    predictions: dict[int, dict],
) -> Optional[dict]:
    """Return the single highest model probability across all loaded matches.

    Iterates over every match in ``matches``; for each, looks up its
    prediction in ``predictions`` keyed by ``int(match_id)`` (matches
    are stored under :data:`dashboard.session_state.KEYS.LOADED_MATCHES`
    and predictions under ``KEYS.PREDICTIONS_BY_MATCH``).  For each
    prediction, reads the model probability from ``blend_probs`` if
    present else ``pi_probs``.

    The returned dict has keys:

    * ``match_id`` — int, the match whose top market won.
    * ``market`` — ``"home"`` / ``"draw"`` / ``"away"``.
    * ``probability`` — float in 0..1, the model's probability for the
      winning market.
    * ``home_team`` / ``away_team`` — display strings (from the
      prediction dict, which has them as ``"home_team"`` / ``"away_team"``).

    Tie-break: prefer the market from the *earlier* match in the list;
    if the same match, prefer ``'home' > 'draw' > 'away'`` (matches
    the order in the snapshot and stays deterministic).

    Returns ``None`` if no matches or no predictions were supplied, OR
    if no prediction has a usable probability dict.
    """
    if not matches or not predictions:
        return None

    # Iterate in the user-supplied order so the tie-break rule
    # ("earlier match wins") is just "first hit wins" without a
    # comparator hack.
    market_order = ("home", "draw", "away")
    best: Optional[dict] = None

    for m in matches:
        if not isinstance(m, dict):
            continue
        mid_raw = m.get("match_id")
        if mid_raw is None:
            continue
        try:
            mid = int(mid_raw)
        except (TypeError, ValueError):
            continue

        pred = predictions.get(mid)
        if not isinstance(pred, dict):
            continue

        probs = _probs_for(pred)
        if not probs:
            continue

        for market in market_order:
            p = probs.get(market)
            try:
                p_val = float(p)
            except (TypeError, ValueError):
                continue
            if best is None or p_val > best["probability"]:
                best = {
                    "match_id": mid,
                    "market": market,
                    "probability": p_val,
                    "home_team": pred.get("home_team") or "Home",
                    "away_team": pred.get("away_team") or "Away",
                }
            # No need to consider later markets in the same match —
            # 'home' beats 'draw' beats 'away' on a tie, and we've
            # already updated if p_val is strictly greater.

    return best


# --------------------------------------------------------------------------- #
# Renderer: Tournament Snapshot
# --------------------------------------------------------------------------- #
def render_tournament_snapshot(snapshot: dict) -> None:
    """Render the Tournament Snapshot card.

    Emits a single ``.wc-card`` block whose contents come from
    ``snapshot`` (a dict built by :func:`build_tournament_snapshot`).
    Renders a calm placeholder if the snapshot is empty / missing
    metadata; never raises.
    """
    if not snapshot or snapshot.get("is_empty"):
        st.markdown(
            f"<div class='wc-card wc-snapshot-card'>"
            f"<strong>Tournament snapshot</strong><br/>"
            f"{_SNAPSHOT_UNAVAILABLE_TEXT}"
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    header = snapshot.get("header") or "Tournament snapshot"
    count_label = snapshot.get("count_label") or ""
    groups_label = snapshot.get("groups_label") or ""

    # The sub-line is the groups label, if any.  Empty string → omit it.
    if groups_label:
        body = (
            f"{count_label}<br/>"
            f"<span style='opacity:0.85;'>{groups_label}</span>"
        )
    else:
        body = count_label

    st.markdown(
        f"<div class='wc-card wc-snapshot-card'>"
        f"<strong>{header}</strong><br/>"
        f"{body}"
        f"</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Renderer: Highest Model Confidence
# --------------------------------------------------------------------------- #
def render_highest_confidence(confidence: Optional[dict]) -> None:
    """Render the Highest Model Confidence card.

    ``confidence`` is the dict returned by
    :func:`highest_model_confidence` (or ``None``).  Renders the
    placeholder copy when the input is ``None`` so the user sees
    actionable guidance instead of an empty box.
    """
    if not confidence:
        st.markdown(
            f"<div class='wc-card wc-confidence-card'>"
            f"<strong>Highest Model Confidence</strong><br/>"
            f"{_CONFIDENCE_PLACEHOLDER_TEXT}"
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    market = confidence.get("market", "")
    home = confidence.get("home_team") or "Home"
    away = confidence.get("away_team") or "Away"
    prob = confidence.get("probability")
    try:
        prob_pct = f"{float(prob) * 100:.1f}%"
    except (TypeError, ValueError):
        prob_pct = "—"

    if market == "draw":
        headline = "Match to End in a Draw"
    elif market == "home":
        headline = f"{home} to Win"
    elif market == "away":
        headline = f"{away} to Win"
    else:
        headline = "TBD"

    st.markdown(
        f"<div class='wc-card wc-confidence-card'>"
        f"<strong>Highest Model Confidence</strong><br/>"
        f"{headline}<br/>"
        f"<span style='opacity:0.85;'>{prob_pct} confidence</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    # The brief calls for a caption beneath the card so the user sees
    # the "model probability only" disclaimer at the same eye level
    # as the number itself.
    st.caption(_CONFIDENCE_DISCLAIMER_TEXT)


# --------------------------------------------------------------------------- #
# Autoload helpers (Phase 9 — dashboard context cards populate on page open)
# --------------------------------------------------------------------------- #
def _fallback_prediction(
    home: str,
    away: str,
    home_id: Any,
    away_id: Any,
    picked_iso: str,
    exc: BaseException,
    *,
    canonical_home_id: str = "",
    canonical_away_id: str = "",
    identity_warnings: Optional[list[str]] = None,
) -> dict:
    """Build the same fallback dict shape the Predictions view uses.

    Mirrors the body of the ``except Exception`` branch in
    :func:`dashboard.app._render_predictions_view` (lines 1710-1736):
    neutral 0.40 / 0.30 / 0.30 probs, a tier-C ``confidence`` block
    with the error message, and the same identity-resolution metadata
    fields so the rest of the dashboard can read the prediction
    without special-casing.

    Kept here (not in app.py) so the autoload pure helper can build
    the same shape without importing the Streamlit-bound renderer.
    """
    return {
        "home_team": home,
        "away_team": away,
        "home_team_id": int(home_id) if home_id is not None else 0,
        "away_team_id": int(away_id) if away_id is not None else 0,
        "date": picked_iso,
        "pi_probs": {"home": 0.4, "draw": 0.3, "away": 0.3},
        "blend_probs": {"home": 0.4, "draw": 0.3, "away": 0.3},
        "pi_only_probs": {"home": 0.4, "draw": 0.3, "away": 0.3},
        "elo_only_probs": None,
        "blend_was_used": False,
        "confidence": {
            "tier": "C",
            "tier_description": "Limited data",
            "warnings": [f"prediction error: {exc!s}"],
        },
        "banner": "Limited data",
        "canonical_home_id": canonical_home_id,
        "canonical_away_id": canonical_away_id,
        "identity_warnings": list(identity_warnings or []),
    }


def _autoload_pure(
    date_iso: str,
    corpus: list[dict],
    elo_snapshots: Any,
    *,
    load_unplayed_fn: Callable[[str], list[dict]],
    predict_match_fn: Callable[..., dict],
    resolve_match_fn: Optional[Callable[..., tuple]] = None,
    get_ratings_fn: Optional[Callable[..., dict]] = None,
    cutoff_iso_override: Optional[str] = None,
) -> dict:
    """Pure helper that builds the autoload payload for ``date_iso``.

    Returns a dict with keys:

    * ``matches`` — ``list[dict]`` from ``load_unplayed_fn(date_iso)``.
      Empty list when the schedule has nothing on this date.
    * ``predictions`` — ``dict[int, dict]`` keyed by match_id; one
      :func:`dashboard.app._predict_match_cached` result per match, or
      a tier-C fallback dict when ``predict_match_fn`` raises for a
      single match (per the existing per-view fallback rule).
    * ``ratings_id`` / ``elo_id`` / ``corpus_id`` — small integer
      handles so the caller can register them in the module-level
      ``_RATINGS_BY_ID`` / ``_ELO_BY_ID`` / ``_CORPUS_BY_ID`` dicts
      used by :func:`dashboard.app._predict_match_cached`.
    * ``date_iso`` — echoed for the caller's convenience.
    * ``ratings_cutoff_iso`` — the cutoff ISO string actually used for
      the pi-ratings snapshot (whichever of ``date_iso + T23:59:59Z``
      and ``date_iso + T00:00:00Z`` succeeded).

    All side-effects (Streamlit session state, module-level registries)
    are the *caller's* responsibility.  This makes the heavy lifting
    unit-testable without a Streamlit ScriptRun.

    Parameters
    ----------
    date_iso
        ``"YYYY-MM-DD"`` ISO date string.  No validation beyond
        string-concatenation; callers should ensure it matches the
        loader's format.
    corpus, elo_snapshots
        The same data structures the per-view renderers hold.  Passed
        through only so the caller can compute ``id(...)`` handles
        without re-importing the registries.
    load_unplayed_fn
        Callable taking ``(date_iso,)`` and returning a list of match
        dicts.  In production this is
        :func:`dashboard.app._load_unplayed_for_date`.
    predict_match_fn
        Callable matching the per-view signature
        ``(home_team, away_team, home_team_id, away_team_id,
        date_iso, _ratings_id, _elo_snapshots_id, _corpus_id) -> dict``.
        In production this is
        :func:`dashboard.app._predict_match_cached`.
    resolve_match_fn
        Optional ``(match, ratings, name_to_id) -> (home_res,
        away_res, warnings)`` resolver.  In production this is
        :func:`dashboard.app._resolve_match_for_prediction`.  When
        ``None`` we treat ``canonical_home_id`` / ``canonical_away_id``
        as empty strings and skip the corpus-id translation.
    get_ratings_fn
        Optional ``(cutoff_iso, corpus) -> dict``.  In production this
        is :func:`dashboard.app.get_ratings`.  When ``None`` we skip
        the cutoff-cutoff two-pass logic and use a 0-id handle (the
        tests typically don't need a real ratings dict).
    cutoff_iso_override
        Test-only knob: when set, use this exact cutoff string instead
        of ``date_iso + "T23:59:59Z"``.  Production callers leave it
        ``None``.
    """
    matches = list(load_unplayed_fn(date_iso) or [])
    if not matches:
        return {
            "matches": [],
            "predictions": {},
            "ratings_id": 0,
            "elo_id": id(elo_snapshots) if elo_snapshots is not None else 0,
            "corpus_id": id(corpus) if corpus is not None else 0,
            "date_iso": date_iso,
            "ratings_cutoff_iso": (cutoff_iso_override or (date_iso + "T23:59:59Z")),
        }

    # ---- Build the pi-ratings snapshot (mirrors the per-view logic) ---- #
    ratings: dict = {}
    ratings_cutoff_iso = cutoff_iso_override or (date_iso + "T23:59:59Z")
    if get_ratings_fn is not None:
        try:
            ratings = get_ratings_fn(ratings_cutoff_iso, corpus)
        except Exception:
            ratings_cutoff_iso = date_iso + "T00:00:00Z"
            try:
                ratings = get_ratings_fn(ratings_cutoff_iso, corpus)
            except Exception:
                # Even the fallback failed; proceed with an empty
                # ratings dict. _predict_match_cached tolerates this
                # (and our per-match try/except handles anything that
                # still goes wrong).
                ratings = {}
    ratings_id = id(ratings)

    elo_id = id(elo_snapshots) if elo_snapshots is not None else 0
    corpus_id = id(corpus) if corpus is not None else 0

    # ---- Per-match prediction (mirrors the per-view logic) ---- #
    match_cutoff = date_iso + "T00:00:00Z"
    predictions: dict[int, dict] = {}
    for m in matches:
        if not isinstance(m, dict):
            continue
        mid = m.get("match_id")
        if mid is None:
            continue
        try:
            mid_int = int(mid)
        except (TypeError, ValueError):
            continue

        home = m.get("home_team_name") or "Home"
        away = m.get("away_team_name") or "Away"
        home_id = m.get("home_team_id")
        away_id = m.get("away_team_id")

        # Identity resolution (translate schedule ids → corpus ids).
        _home_res = _away_res = None
        _id_warnings: list[str] = []
        if resolve_match_fn is not None and home_id is not None and away_id is not None:
            try:
                _home_res, _away_res, _id_warnings = resolve_match_fn(
                    match=m, ratings=ratings, name_to_id=None,
                )
            except Exception:
                _home_res = _away_res = None
                _id_warnings = []

        # Translate to corpus ids when resolved, else fall back to schedule ids.
        home_corpus_id = (
            int(_home_res.corpus_id) if (_home_res is not None and _home_res.corpus_id is not None)
            else (int(home_id) if home_id is not None else 0)
        )
        away_corpus_id = (
            int(_away_res.corpus_id) if (_away_res is not None and _away_res.corpus_id is not None)
            else (int(away_id) if away_id is not None else 0)
        )

        try:
            pred = predict_match_fn(
                home_team=home,
                away_team=away,
                home_team_id=home_corpus_id,
                away_team_id=away_corpus_id,
                date_iso=match_cutoff,
                _ratings_id=ratings_id,
                _elo_snapshots_id=elo_id,
                _corpus_id=corpus_id,
            )
        except Exception as exc:
            pred = _fallback_prediction(
                home=home,
                away=away,
                home_id=home_corpus_id,
                away_id=away_corpus_id,
                picked_iso=date_iso,
                exc=exc,
                canonical_home_id=(
                    _home_res.canonical_id if _home_res is not None else ""
                ),
                canonical_away_id=(
                    _away_res.canonical_id if _away_res is not None else ""
                ),
                identity_warnings=_id_warnings,
            )
        else:
            # Surface identity metadata the same way the per-view renderer does.
            if _id_warnings:
                pred["identity_warnings"] = list(_id_warnings)
            if not pred.get("canonical_home_id") and _home_res is not None and _home_res.canonical_id:
                pred["canonical_home_id"] = _home_res.canonical_id
            if not pred.get("canonical_away_id") and _away_res is not None and _away_res.canonical_id:
                pred["canonical_away_id"] = _away_res.canonical_id

        # Surface the source-match metadata so per-match renderers can read
        # the human-readable kickoff / group / stage labels.
        pred["_match_meta"] = {
            "group": m.get("group", ""),
            "stage": m.get("stage", ""),
            "matchday": m.get("matchday"),
            "kickoff_iso": m.get("kickoff_iso") or date_iso,
        }
        predictions[mid_int] = pred

    return {
        "matches": matches,
        "predictions": predictions,
        "ratings_id": ratings_id,
        "elo_id": elo_id,
        "corpus_id": corpus_id,
        "date_iso": date_iso,
        "ratings_cutoff_iso": ratings_cutoff_iso,
    }


def autoload_context_for_date(
    date_iso: str,
    corpus: list[dict],
    elo_snapshots: Any,
) -> dict:
    """Populate session state with the context cards' data for ``date_iso``.

    This is the Streamlit-aware wrapper around :func:`_autoload_pure`.
    It is intended to be called from :func:`dashboard.app.main` *before*
    the Tournament Snapshot and Highest Model Confidence renderers so
    the cards populate on page open without requiring the user to
    click "Show Predictions".

    The helper is cheap to call on every rerun: it short-circuits via
    the :data:`KEYS.CONTEXT_AUTOLOAD_DATE` sentinel when the same date
    has already been loaded.  When the user changes the date in the
    per-view picker the sentinel mismatch forces a fresh load.

    Parameters
    ----------
    date_iso
        ``"YYYY-MM-DD"`` ISO date string.  ``datetime.date`` objects
        are accepted and normalised.
    corpus, elo_snapshots
        The same data structures held by :func:`dashboard.app.main`.

    Returns
    -------
    dict
        The assembled payload — same shape as
        :func:`_autoload_pure` returns — so the caller can read it
        without going back through session state.  Also written to
        :data:`KEYS.LOADED_MATCHES`, ``KEYS.LOADED_MATCHES + ".date"``,
        and :data:`KEYS.PREDICTIONS_BY_MATCH`.
    """
    # ---- Defensive normalisation ---- #
    if hasattr(date_iso, "isoformat") and not isinstance(date_iso, str):
        try:
            date_iso = date_iso.isoformat()
        except Exception:
            pass
    if not isinstance(date_iso, str) or not date_iso:
        # Without a usable date string we can't cache-key anything;
        # return an empty payload rather than raising — the cards
        # then render their calm placeholders.
        empty = {
            "matches": [],
            "predictions": {},
            "ratings_id": 0,
            "elo_id": 0,
            "corpus_id": 0,
            "date_iso": "",
            "ratings_cutoff_iso": "",
        }
        st.session_state[KEYS.LOADED_MATCHES] = []
        st.session_state[KEYS.LOADED_MATCHES + ".date"] = ""
        st.session_state[KEYS.PREDICTIONS_BY_MATCH] = {}
        st.session_state[KEYS.CONTEXT_AUTOLOAD_DATE] = ""
        return empty

    # ---- Cache short-circuit (the hot path on every rerun) ---- #
    cached_date = st.session_state.get(KEYS.CONTEXT_AUTOLOAD_DATE)
    cached_matches = st.session_state.get(KEYS.LOADED_MATCHES)
    if (
        cached_date == date_iso
        and cached_matches is not None
        and st.session_state.get(KEYS.LOADED_MATCHES + ".date") == date_iso
    ):
        # Same date, already loaded → return the cached payload
        # without touching the loader, get_ratings, or predict_match.
        return {
            "matches": cached_matches or [],
            "predictions": st.session_state.get(KEYS.PREDICTIONS_BY_MATCH) or {},
            "ratings_id": 0,
            "elo_id": id(elo_snapshots) if elo_snapshots is not None else 0,
            "corpus_id": id(corpus) if corpus is not None else 0,
            "date_iso": date_iso,
            "ratings_cutoff_iso": date_iso + "T23:59:59Z",
        }

    # ---- Lazy imports so context_cards stays import-cheap for callers
    # that only want the pure helpers / renderers. ---- #
    from dashboard.app import (
        _CORPUS_BY_ID,
        _ELO_BY_ID,
        _load_unplayed_for_date,
        _predict_match_cached,
        get_ratings,
    )
    from dashboard.team_resolution import resolve_match_for_prediction

    payload = _autoload_pure(
        date_iso,
        corpus,
        elo_snapshots,
        load_unplayed_fn=_load_unplayed_for_date,
        predict_match_fn=_predict_match_cached,
        resolve_match_fn=resolve_match_for_prediction,
        get_ratings_fn=get_ratings,
    )

    # ---- Register handles so _predict_match_cached can find them ---- #
    if payload.get("corpus_id"):
        _CORPUS_BY_ID[payload["corpus_id"]] = corpus
    if payload.get("elo_id"):
        _ELO_BY_ID[payload["elo_id"]] = elo_snapshots

    # ---- Write the same keys the per-view renderers already use ---- #
    st.session_state[KEYS.LOADED_MATCHES] = payload["matches"]
    st.session_state[KEYS.LOADED_MATCHES + ".date"] = date_iso
    st.session_state[KEYS.PREDICTIONS_BY_MATCH] = payload["predictions"]
    st.session_state[KEYS.CONTEXT_AUTOLOAD_DATE] = date_iso
    return payload
