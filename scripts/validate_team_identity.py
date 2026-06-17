"""Standalone validation script for the canonical team identity layer.

Loads the registry, the 2026 fixture cache, the historical corpus, and
the Elo snapshot file, and reports how well the registry covers the
teams in the 2026 World Cup.

Run with:
    .venv/bin/python3 scripts/validate_team_identity.py

Exits 0 on success, non-zero on unexpected failure (e.g. registry or
cache file missing). The report is the whole point — the exit code
exists only to flag infrastructure breakage, not registry gaps.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _project_root() -> Path:
    """Locate the repo root from this script's path."""
    return Path(__file__).resolve().parent.parent


def _load_intl_corpus(root: Path) -> list[dict]:
    """Load all openfootball historical matches (intl + WC years)."""
    from soccer_ev_model.pi_backtest import load_matches

    intl_path = root / "data" / "processed" / "international_matches.json"
    intl = json.loads(intl_path.read_text())

    wc: list[dict] = []
    wc_years = (2010, 2014, 2018, 2022)
    for y in wc_years:
        try:
            wc.extend(load_matches(y))
        except FileNotFoundError:
            pass

    corpus = list(intl) + list(wc)
    corpus.sort(key=lambda m: m.get("date", ""))
    return corpus


def _load_elo(root: Path) -> dict:
    """Load the Elo snapshot. Empty dict if file is missing."""
    elo_path = root / "data" / "raw" / "elo_ratings.json"
    if not elo_path.exists():
        return {}
    from soccer_ev_model.elo_ratings import load_elo_ratings
    return load_elo_ratings(elo_path)


def _build_corpus_index(corpus: list[dict]) -> tuple[set[int], set[str]]:
    """Return ({team_ids_in_corpus}, {team_names_in_corpus})."""
    ids: set[int] = set()
    names: set[str] = set()
    for m in corpus:
        for side in ("home", "away"):
            tid = m.get(f"{side}_team_id")
            tn = m.get(f"{side}_team")
            if isinstance(tid, int):
                ids.add(tid)
            if isinstance(tn, str) and tn.strip():
                names.add(tn.strip())
    return ids, names


def _has_elo(elo: dict, team_name: str, cutoff: str = "2026-06-16") -> bool:
    """True if `team_name` has any Elo rating on/before the cutoff.

    The `elo` dict is the parsed output of `load_elo_ratings` where
    dates are `datetime.date` objects (not raw strings).
    """
    if not elo:
        return False
    from datetime import date as _date
    cutoff_d = _date.fromisoformat(cutoff)
    records = elo.get(team_name)
    if not records:
        return False
    for record in records:
        d = record.get("date")
        if isinstance(d, _date) and d <= cutoff_d:
            return True
    return False


def main() -> int:
    root = _project_root()

    # 1) Load registry
    registry_path = root / "data" / "team_identity.json"
    if not registry_path.exists():
        print(f"ERROR: registry not found at {registry_path}", file=sys.stderr)
        return 2
    registry_raw = json.loads(registry_path.read_text())
    registry = {k: v for k, v in registry_raw.items() if not k.startswith("_")}

    # 2) Load 2026 cache
    cache_path = root / "data" / "raw" / "matches_2026.json"
    if not cache_path.exists():
        print(f"ERROR: 2026 cache not found at {cache_path}", file=sys.stderr)
        return 2
    cache = json.loads(cache_path.read_text())

    # 3) Load historical corpus
    corpus = _load_intl_corpus(root)
    corpus_ids, corpus_names = _build_corpus_index(corpus)

    # 4) Load Elo
    elo = _load_elo(root)

    # 5) Lazy import of the identity helpers (after path is set up)
    from soccer_ev_model.team_identity import (
        resolve_team,
        canonical_id_for_football_data_id,
        canonical_id_for_corpus_id,
    )

    # ---- Gather teams from the 2026 cache ----
    seen_teams: dict[str, dict] = {}  # canonical name -> {fd_id, name}
    for m in cache.get("matches", []):
        for side in ("home", "away"):
            name = m.get(f"{side}_team_name")
            fd = m.get(f"{side}_team_id")
            if not name or not isinstance(name, str):
                continue
            name = name.strip()
            if not name:
                continue
            if name not in seen_teams:
                seen_teams[name] = {"fd_id": fd, "name": name}

    total = len(seen_teams)

    # ---- Resolve each team three ways and aggregate ----
    canonical_counts: Counter[str] = Counter()
    in_corpus = 0
    not_in_corpus: list[str] = []
    in_elo = 0
    not_in_elo: list[str] = []
    unresolved: list[dict] = []

    for name, info in sorted(seen_teams.items(), key=lambda kv: kv[0].lower()):
        fd = info["fd_id"]
        # 1) fd_id-based resolve
        cid_via_fd = canonical_id_for_football_data_id(fd) if fd is not None else None
        # 2) name-based resolve
        cid_via_name = None
        if cid_via_fd is None:
            from soccer_ev_model.team_identity import canonical_id_for_name
            cid_via_name = canonical_id_for_name(name)
        # 3) full resolve
        res = resolve_team(football_data_id=fd, name=name)
        cid = res.get("canonical_id")
        if cid:
            canonical_counts[cid] += 1
        # In corpus?
        corpus_id = res.get("corpus_id")
        if corpus_id is not None and corpus_id in corpus_ids:
            in_corpus += 1
        else:
            not_in_corpus.append(name)
        # In Elo?
        if _has_elo(elo, name):
            in_elo += 1
        else:
            not_in_elo.append(name)
        # Unresolved?
        if cid is None:
            unresolved.append({
                "name": name,
                "fd_id": fd,
                "corpus_id": None,
                "canonical_id": None,
                "reason": "no canonical entry found by fd_id or name",
            })

    # ---- Print report ----
    print("=" * 72)
    print("Team identity validation report")
    print("=" * 72)
    print(f"Registry:        {registry_path}")
    print(f"2026 cache:      {cache_path}  ({len(cache.get('matches', []))} matches)")
    print(f"Corpus:          {len(corpus)} matches  ({len(corpus_ids)} unique team ids)")
    print(f"Elo:             {len(elo)} teams in snapshot")
    print()
    print(f"Total unique teams in 2026 cache:  {total}")
    print(f"Mapped to canonical IDs:           {sum(1 for c in canonical_counts if c)} "
          f"({len(canonical_counts)} distinct canonical ids)")
    print()

    # history_missing
    history_missing: list[str] = []
    for name in sorted(seen_teams, key=lambda x: x.lower()):
        res = resolve_team(
            football_data_id=seen_teams[name]["fd_id"],
            name=name,
        )
        if res.get("status") == "history_missing":
            history_missing.append(name)
    if history_missing:
        print(f"History-missing teams (canonical resolved, no corpus_id): "
              f"{len(history_missing)}")
        for n in history_missing:
            print(f"  - {n}")
        print()

    print(f"Found in historical corpus:        {in_corpus} / {total}")
    if not_in_corpus:
        print(f"  Not in corpus ({len(not_in_corpus)}): {', '.join(not_in_corpus)}")
    print()

    print(f"Found in Elo data:                 {in_elo} / {total}")
    if not_in_elo:
        print(f"  Not in Elo ({len(not_in_elo)}): {', '.join(not_in_elo)}")
    print()

    if unresolved:
        print(f"Unresolved teams: {len(unresolved)}")
        for u in unresolved:
            print(f"  - {u['name']}: fd_id={u['fd_id']}, "
                  f"corpus_id={u['corpus_id']}, canonical_id={u['canonical_id']}, "
                  f"reason={u['reason']}")
    else:
        print("Unresolved teams: 0 — all 2026 cache teams have a canonical entry")

    print()
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
