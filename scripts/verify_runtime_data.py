#!/usr/bin/env python3
"""Verify the runtime data files committed to this repo are present and well-formed.

Stdlib-only: pathlib, json, sys. Exits 0 on success, 1 on any failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

REQUIRED_FILES = [
    # raw/
    "raw/matches_2026.json",
    "raw/elo_ratings.json",
    "raw/matches_2010_openfootball.json",
    "raw/matches_2014_openfootball.json",
    "raw/matches_2018_openfootball.json",
    "raw/matches_2022_openfootball.json",
    "raw/matches_1930_openfootball.json",
    "raw/matches_1934_openfootball.json",
    "raw/matches_1938_openfootball.json",
    "raw/matches_1950_openfootball.json",
    "raw/matches_1954_openfootball.json",
    "raw/matches_1958_openfootball.json",
    "raw/matches_1962_openfootball.json",
    "raw/matches_1966_openfootball.json",
    "raw/matches_1970_openfootball.json",
    "raw/matches_1974_openfootball.json",
    "raw/matches_1978_openfootball.json",
    "raw/matches_1982_openfootball.json",
    "raw/matches_1986_openfootball.json",
    "raw/matches_1990_openfootball.json",
    "raw/matches_1994_openfootball.json",
    "raw/matches_1998_openfootball.json",
    "raw/matches_2002_openfootball.json",
    "raw/matches_2006_openfootball.json",
    # processed/
    "processed/international_matches.json",
    # root
    "team_identity.json",
]

MAX_BYTES = 100 * 1024 * 1024  # 100 MB


def _nonempty(value) -> bool:
    if isinstance(value, (list, dict, str)):
        return len(value) > 0
    return value is not None


def check(rel: str) -> tuple[str, str]:
    """Return (status, detail). status is OK or FAIL with reason."""
    path = DATA_DIR / rel
    if not path.exists():
        return "FAIL", "missing"
    try:
        size = path.stat().st_size
    except OSError as e:
        return "FAIL", f"stat error: {e}"
    if size == 0:
        return "FAIL", "empty file (0 bytes)"
    if size > MAX_BYTES:
        return "FAIL", f"too large ({size} bytes > {MAX_BYTES})"
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as e:
        return "FAIL", f"invalid JSON: {e}"
    except OSError as e:
        return "FAIL", f"read error: {e}"

    # Per-file shape checks
    name = path.name
    if name == "elo_ratings.json":
        if isinstance(data, dict) and len(data) > 0:
            return "OK", f"dict len={len(data)}"
        if isinstance(data, dict) and isinstance(data.get("teams"), list) and len(data["teams"]) > 0:
            return "OK", f"teams len={len(data['teams'])}"
        return "FAIL", "elo_ratings must be non-empty dict (or dict with non-empty 'teams' list)"
    if name == "international_matches.json":
        if isinstance(data, list) and len(data) > 0:
            return "OK", f"list len={len(data)}"
        if isinstance(data, dict) and isinstance(data.get("matches"), list) and len(data["matches"]) > 0:
            return "OK", f"matches len={len(data['matches'])}"
        return "FAIL", "international_matches must be non-empty list (or dict with non-empty 'matches')"
    if name == "matches_2026.json":
        if not _nonempty(data):
            return "FAIL", "matches_2026 must be non-empty dict/list"
        return "OK", f"len={len(data)}"
    if name == "team_identity.json":
        if not (isinstance(data, dict) and len(data) > 0):
            return "FAIL", "team_identity must be non-empty dict"
        return "OK", f"dict len={len(data)}"
    if name.endswith("_openfootball.json"):
        if not _nonempty(data):
            return "FAIL", "openfootball file must be non-empty dict/list"
        return "OK", f"len={len(data)}"

    # Generic fallback
    if not _nonempty(data):
        return "FAIL", "empty JSON container"
    return "OK", f"len={len(data)}"


def main() -> int:
    print(f"Verifying {len(REQUIRED_FILES)} runtime data files under {DATA_DIR}")
    print("-" * 78)
    print(f"{'STATUS':<6} {'BYTES':>10}  {'PATH'}")
    print("-" * 78)
    failures: list[tuple[str, str]] = []
    for rel in REQUIRED_FILES:
        path = DATA_DIR / rel
        size = path.stat().st_size if path.exists() else 0
        status, detail = check(rel)
        print(f"{status:<6} {size:>10}  {rel}  ({detail})")
        if status != "OK":
            failures.append((rel, detail))
    print("-" * 78)
    print(f"Total: {len(REQUIRED_FILES)}  OK: {len(REQUIRED_FILES) - len(failures)}  FAIL: {len(failures)}")
    if failures:
        for rel, detail in failures:
            print(f"FAILED: {rel}: {detail}", file=sys.stderr)
        return 1
    print("All runtime data files present and well-formed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
