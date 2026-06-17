"""Canonical national-team identity layer.

The 2026 World Cup fixture cache (football-data.org) and the training
corpus (openfootball) use different ID spaces. This module provides a
single stable mapping — a 3-letter canonical ID like "ARG", "USA", "ALG"
— that all internal joins should be routed through.

The registry is read lazily from `data/team_identity.json` on first
call and cached in a module-level dict. This module does no I/O at
import time and has no Streamlit dependency, so it is safe to import
from tests, scripts, and the dashboard alike.

Status taxonomy
---------------
A `resolve_team()` call returns one of three statuses:

* ``resolved``       — canonical_id found AND corpus_id is registered
                       (the team has training history)
* ``history_missing``— canonical_id found, but corpus_id is null
                       (e.g. CPV, COD, CUW in the 2026 cycle)
* ``identity_unresolved`` — no canonical_id found from any of the
                            three lookup paths (football_data_id,
                            corpus_id, name)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Registry path
# --------------------------------------------------------------------------- #

# soccer_ev_model/ -> ../data/team_identity.json
_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent
DEFAULT_REGISTRY_PATH = _PROJECT_ROOT / "data" / "team_identity.json"


# --------------------------------------------------------------------------- #
# Module-level cache (filled on first load_registry call)
# --------------------------------------------------------------------------- #

_REGISTRY: dict[str, dict[str, Any]] | None = None
_REGISTRY_PATH: Path | None = None

# Reverse indexes built once alongside _REGISTRY, for O(1) lookups.
_BY_FOOTBALL_DATA: dict[int, str] = {}
_BY_CORPUS: dict[int, str] = {}
_BY_ALIAS: dict[str, str] = {}  # lower-cased alias -> canonical_id


def _norm(s: str) -> str:
    """Normalise a name/alias for case+whitespace-insensitive comparison."""
    return (s or "").strip().lower()


def _build_indexes(registry: dict[str, dict[str, Any]]) -> None:
    """Populate the three reverse indexes from the canonical registry."""
    _BY_FOOTBALL_DATA.clear()
    _BY_CORPUS.clear()
    _BY_ALIAS.clear()
    for cid, entry in registry.items():
        fd_id = entry.get("football_data_id")
        if isinstance(fd_id, int):
            _BY_FOOTBALL_DATA[fd_id] = cid
        corpus_id = entry.get("corpus_id")
        if isinstance(corpus_id, int):
            _BY_CORPUS[corpus_id] = cid
        # Index the canonical id, the display name, and every alias.
        names = [entry.get("name", ""), cid] + list(entry.get("aliases") or [])
        for n in names:
            n_norm = _norm(n)
            if n_norm and n_norm not in _BY_ALIAS:
                _BY_ALIAS[n_norm] = cid


def load_registry(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Return the canonical registry as {canonical_id: entry}.

    Cached at module level so subsequent calls are free. The cache is
    keyed on the path argument, so calling with a different path will
    rebuild the cache. The JSON file is the source of truth.
    """
    global _REGISTRY, _REGISTRY_PATH
    target = Path(path) if path is not None else DEFAULT_REGISTRY_PATH

    if _REGISTRY is not None and _REGISTRY_PATH == target:
        return _REGISTRY

    with open(target, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    # Strip the _meta block — callers want {canonical_id: entry}.
    registry: dict[str, dict[str, Any]] = {
        cid: entry for cid, entry in raw.items() if not cid.startswith("_")
    }

    _REGISTRY = registry
    _REGISTRY_PATH = target
    _build_indexes(registry)
    return registry


def _ensure_loaded() -> None:
    """Lazily load the default registry if no caller has done so yet."""
    if _REGISTRY is None:
        load_registry()


# --------------------------------------------------------------------------- #
# Public lookup helpers
# --------------------------------------------------------------------------- #

def canonical_id_for_football_data_id(fd_id: int) -> str | None:
    """Return canonical_id for a football-data.org integer, or None."""
    if fd_id is None:
        return None
    _ensure_loaded()
    return _BY_FOOTBALL_DATA.get(int(fd_id))


def canonical_id_for_corpus_id(corpus_id: int) -> str | None:
    """Return canonical_id for an openfootball integer, or None."""
    if corpus_id is None:
        return None
    _ensure_loaded()
    return _BY_CORPUS.get(int(corpus_id))


def canonical_id_for_name(name: str) -> str | None:
    """Return canonical_id for a team name (case + whitespace insensitive)."""
    if not name:
        return None
    _ensure_loaded()
    return _BY_ALIAS.get(_norm(name))


def display_name(canonical_id: str) -> str:
    """Return the human-readable name for a canonical_id.

    Falls back to the canonical_id itself if the entry is missing.
    """
    _ensure_loaded()
    entry = (_REGISTRY or {}).get(canonical_id) or {}
    return entry.get("name") or canonical_id


def corpus_id_for_canonical(canonical_id: str) -> int | None:
    """Return the corpus integer for a canonical_id, or None if history_missing."""
    _ensure_loaded()
    entry = (_REGISTRY or {}).get(canonical_id) or {}
    cid = entry.get("corpus_id")
    return int(cid) if isinstance(cid, int) else None


def football_data_id_for_canonical(canonical_id: str) -> int | None:
    """Return the football-data integer for a canonical_id, or None."""
    _ensure_loaded()
    entry = (_REGISTRY or {}).get(canonical_id) or {}
    fid = entry.get("football_data_id")
    return int(fid) if isinstance(fid, int) else None


def all_canonical_ids() -> list[str]:
    """Return a sorted list of all registered canonical_ids."""
    _ensure_loaded()
    return sorted((_REGISTRY or {}).keys())


# --------------------------------------------------------------------------- #
# Status resolver
# --------------------------------------------------------------------------- #

def resolve_team(
    *,
    football_data_id: int | None = None,
    corpus_id: int | None = None,
    name: str | None = None,
) -> dict:
    """Resolve any of {fd_id, corpus_id, name} to a canonical identity status.

    Resolution priority: football_data_id → corpus_id → name.

    Returns a dict with the canonical_id (or None), a status string, the
    inputs as observed, the resolution source, the display name, and the
    corpus id that should be used downstream by the pi-rating lookup.

    Status values:

    * ``resolved``            — canonical_id found AND corpus_id registered
    * ``history_missing``     — canonical_id found but corpus_id is null
    * ``identity_unresolved`` — no canonical_id found from any path
    """
    _ensure_loaded()

    # Normalise inputs to None for missing values
    fd_id = int(football_data_id) if football_data_id is not None else None
    cor_id = int(corpus_id) if corpus_id is not None else None
    nm = (name or "").strip() or None

    canonical_id: str | None = None
    source: str | None = None

    # 1) football_data_id has the highest priority (it's the most
    #    authoritative for the 2026 cache).
    if fd_id is not None:
        canonical_id = _BY_FOOTBALL_DATA.get(fd_id)
        if canonical_id is not None:
            source = "football_data"

    # 2) corpus_id is the fallback for "we know the corpus id but not
    #    the football-data id" (e.g. callers from the manual tab).
    if canonical_id is None and cor_id is not None:
        canonical_id = _BY_CORPUS.get(cor_id)
        if canonical_id is not None:
            source = "corpus"

    # 3) name is the last-ditch fuzzy lookup (case + whitespace).
    if canonical_id is None and nm is not None:
        canonical_id = _BY_ALIAS.get(_norm(nm))
        if canonical_id is not None:
            source = "name"

    if canonical_id is None:
        status = "identity_unresolved"
        registry_corpus_id: int | None = None
        dn: str | None = None
    else:
        entry = (_REGISTRY or {}).get(canonical_id) or {}
        registry_corpus_id_raw = entry.get("corpus_id")
        registry_corpus_id = (
            int(registry_corpus_id_raw)
            if isinstance(registry_corpus_id_raw, int)
            else None
        )
        dn = entry.get("name") or canonical_id
        status = "resolved" if registry_corpus_id is not None else "history_missing"

    return {
        "canonical_id": canonical_id,
        "status": status,
        "source_team_id": fd_id,
        "source_team_name": nm,
        "source": source,
        "display_name": dn,
        "corpus_id": registry_corpus_id,
    }
