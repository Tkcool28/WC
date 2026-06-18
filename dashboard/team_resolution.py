"""Shared team-id resolution for auto-populated schedule matches.

The new auto-populated flows in :mod:`dashboard.app` (🎯 Predictions,
💰 Bets, 🔬 Analysis — Phases 3-5 of PR #9) read
``m["home_team_id"]`` / ``m["away_team_id"]`` directly from the 2026
schedule cache (``data/raw/matches_2026.json``) and pass them into
:soccer_ev_model.ev_workflow.predict_match:func:.

Those cache IDs are **football-data.org integers** (small, typically
in the 700-10000 range), but the pi-rating layer expects **training-
corpus integers** (large, derived from openfootball). When the two
spaces are mixed, ``get_team_experience(ratings, team_id)`` looks up
the wrong key, returns ``matches_played=0``, and pi-rating produces a
neutral home-draw-away fallback. The most-visible symptom is that
high-history teams like England and Croatia appear with no rating
history at all.

The legacy ``evaluate_one_game(...)`` path (Phase 2 and earlier)
resolved this correctly via a nested ``_resolve_team_id`` helper that
translated ``football_data_id`` → canonical identity → ``corpus_id``
before calling :func:`soccer_ev_model.ev_workflow.predict_match`. The
Phase 3-5 auto loops skipped that translation.

This module lifts that nested helper into a small, public, pure-
function library that all three auto loops can share. It is the
single source of truth for:

  * translating a schedule id → canonical id → corpus id
  * preserving the canonical id separately for diagnostics (squad
    context panel, Raw Diagnostics block)
  * emitting raw per-team warnings that downstream ``translate_warning``
    can clean up before display

The math layers (:mod:`soccer_ev_model.pi_ratings`,
:mod:`soccer_ev_model.elo_ratings`, :mod:`soccer_ev_model.no_vig`,
:mod:`soccer_ev_model.confidence`, :mod:`soccer_ev_model.prediction_summary`,
:mod:`dashboard.context_loader`) are NOT touched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from soccer_ev_model.team_identity import resolve_team as _resolve_team_identity


# --------------------------------------------------------------------------- #
# Public dataclass
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ResolvedTeam:
    """Resolution result for one team.

    Attributes
    ----------
    canonical_id
        The 3-letter canonical id (``"ENG"``, ``"CRO"``, ``"COD"``,
        ...), or ``None`` if the team could not be resolved.
    corpus_id
        The integer that should be passed to ``predict_match`` /
        ``evaluate_match`` as the team id. Falls back to the schedule
        id when no corpus id is available — this is the
        "history_missing" case where pi-rating will produce a neutral
        result rather than hard-fail the row.
    source_team_id
        The schedule's team id (always populated for 2026 matches).
        Kept for diagnostics.
    source_team_name
        The schedule's display name (e.g. ``"England"``).
    status
        One of ``"resolved"`` | ``"history_missing"`` |
        ``"identity_unresolved"``. Mirrors the taxonomy defined in
        :mod:`soccer_ev_model.team_identity`.
    warning
        Raw per-team warning string, or ``None``. The dashboard pipes
        this through :func:`dashboard.ux_presenters.translate_warning`
        before rendering. The format is intentionally stable so the
        existing translator keeps working:

        * ``canonical=<id> status=history_missing fd_id=<n> name=<nm>``
        * ``canonical=None status=identity_unresolved fd_id=<n> name=<nm>``
    """

    canonical_id: str | None
    corpus_id: int | None
    source_team_id: int | None
    source_team_name: str | None
    status: str
    warning: str | None


__all__ = [
    "ResolvedTeam",
    "resolve_team_for_prediction",
    "resolve_match_for_prediction",
]


# --------------------------------------------------------------------------- #
# Core resolver
# --------------------------------------------------------------------------- #
def resolve_team_for_prediction(
    *,
    schedule_team_id: int | None,
    team_name: str | None,
    ratings: dict | None,
    name_to_id: dict | None,
) -> ResolvedTeam:
    """Translate a schedule's team id into the corpus id used by pi-rating.

    Parameters
    ----------
    schedule_team_id
        football-data.org id from the 2026 schedule cache, or any other
        integer. May also be a corpus id if the cache was built from a
        different source — handled below.
    team_name
        Display name (e.g. ``"England"``); used as a fuzzy last-ditch
        fallback against the corpus name map.
    ratings
        The pi-ratings dict for the cutoff (used to detect when
        ``schedule_team_id`` is ALREADY a corpus id — i.e. it appears
        as a key in ``ratings``). May be empty or ``None``.
    name_to_id
        ``name -> corpus_id`` map from the training corpus. Final
        fallback when both the registry and the ratings lookup miss.

    Returns
    -------
    ResolvedTeam
        ``corpus_id`` is the value to pass into
        ``predict_match(home_team_id=..., away_team_id=...)``.
        ``canonical_id`` is preserved for diagnostics (squad context,
        Raw Diagnostics). ``warning`` is a raw warning (or ``None``);
        downstream code should run it through
        :func:`dashboard.ux_presenters.translate_warning` before display.

    Resolution priority
    -------------------

    1. ``schedule_team_id`` is already a corpus key in ``ratings`` →
       use it unchanged (corpus id is correct as-is).
    2. Canonical identity registry resolves a corpus id → use the
       translated corpus id.
    3. Canonical identity registry resolves a canonical id but
       ``corpus_id`` is null (``history_missing``) → return
       ``status='history_missing'``, surface a warning, fall back to
       ``schedule_team_id`` (so pi-rating produces a neutral result
       rather than hard-failing the whole row).
    4. Corpus name map (``name_to_id``) as a final fallback for callers
       that hand us a corpus-shaped id directly.
    5. Truly unresolved → ``status='identity_unresolved'``, surface a
       warning, do NOT fabricate an id.
    """
    h_name = (team_name or "").strip() or None

    # 1) If schedule_team_id is already a key in ratings, it is a
    #    corpus id — use it unchanged. This is the fast path for
    #    caches that store corpus ids directly.
    if (
        schedule_team_id is not None
        and ratings
        and int(schedule_team_id) in ratings
    ):
        res = _resolve_team_identity(
            corpus_id=int(schedule_team_id), name=h_name
        )
        return ResolvedTeam(
            canonical_id=res.get("canonical_id"),
            corpus_id=int(schedule_team_id),
            source_team_id=schedule_team_id,
            source_team_name=h_name,
            status=res.get("status", "resolved"),
            warning=None,
        )

    # 2) Try the canonical identity registry (fd_id → canonical → corpus).
    res: dict[str, Any] = _resolve_team_identity(
        football_data_id=schedule_team_id, name=h_name
    )
    canonical_id = res.get("canonical_id")
    registry_corpus_id = res.get("corpus_id")
    registry_status = res.get("status", "identity_unresolved")

    # 3) Registry gave us a corpus_id — use it (the happy path).
    if registry_corpus_id is not None:
        return ResolvedTeam(
            canonical_id=canonical_id,
            corpus_id=int(registry_corpus_id),
            source_team_id=schedule_team_id,
            source_team_name=h_name,
            status=registry_status or "resolved",
            warning=None,
        )

    # 4) Registry resolved a canonical_id but corpus_id is null →
    #    history_missing (e.g. CPV, COD, CUW in the 2026 cycle).
    #    Use schedule_team_id (so pi-rating produces a neutral
    #    result rather than a hard fail) and surface the warning.
    #    NEVER fabricate a fake id.
    if canonical_id is not None:
        warning = (
            f"canonical={canonical_id} status=history_missing "
            f"fd_id={schedule_team_id} name={h_name}"
        )
        return ResolvedTeam(
            canonical_id=canonical_id,
            corpus_id=schedule_team_id,
            source_team_id=schedule_team_id,
            source_team_name=h_name,
            status="history_missing",
            warning=warning,
        )

    # 5) Last resort: name_to_id corpus name map (the manual flow's
    #    primary path; the auto flow's safety net if both the
    #    registry and the ratings lookup miss).
    if name_to_id and h_name:
        fallback = name_to_id.get(h_name)
        if fallback is not None:
            res2 = _resolve_team_identity(
                corpus_id=int(fallback), name=h_name
            )
            return ResolvedTeam(
                canonical_id=res2.get("canonical_id"),
                corpus_id=int(fallback),
                source_team_id=schedule_team_id,
                source_team_name=h_name,
                status=res2.get("status", "resolved"),
                warning=None,
            )

    # 6) Truly unresolved — preserve warning, do NOT fabricate an id.
    warning = (
        f"canonical=None status=identity_unresolved "
        f"fd_id={schedule_team_id} name={h_name}"
    )
    return ResolvedTeam(
        canonical_id=None,
        corpus_id=schedule_team_id,  # pass through; pi-rating will be neutral
        source_team_id=schedule_team_id,
        source_team_name=h_name,
        status="identity_unresolved",
        warning=warning,
    )


def resolve_match_for_prediction(
    *,
    match: dict,
    ratings: dict | None,
    name_to_id: dict | None,
) -> tuple[ResolvedTeam, ResolvedTeam, list[str]]:
    """Resolve both teams for one auto-populated schedule match.

    Parameters
    ----------
    match
        Schedule match dict (must include ``home_team_id``,
        ``home_team_name``, ``away_team_id``, ``away_team_name``).
    ratings
        Pi-ratings snapshot for the cutoff (see
        :func:`resolve_team_for_prediction`).
    name_to_id
        Corpus name map (see :func:`resolve_team_for_prediction`).

    Returns
    -------
    (home_resolved, away_resolved, raw_warnings_list)
        ``raw_warnings_list`` is the union of any per-team warnings
        (empty if both teams resolved cleanly). Callers should run
        each entry through
        :func:`dashboard.ux_presenters.translate_warning` before
        rendering, then attach to the prediction dict under
        ``identity_warnings`` so the existing UI surfaces them the
        same way it does for the legacy ``evaluate_one_game`` path.
    """
    home = resolve_team_for_prediction(
        schedule_team_id=match.get("home_team_id"),
        team_name=match.get("home_team_name"),
        ratings=ratings,
        name_to_id=name_to_id,
    )
    away = resolve_team_for_prediction(
        schedule_team_id=match.get("away_team_id"),
        team_name=match.get("away_team_name"),
        ratings=ratings,
        name_to_id=name_to_id,
    )
    warnings: list[str] = []
    if home.warning:
        warnings.append(home.warning)
    if away.warning:
        warnings.append(away.warning)
    return home, away, warnings
