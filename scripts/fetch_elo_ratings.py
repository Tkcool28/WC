#!/usr/bin/env python3
"""Fetch Elo ratings from eloratings.net and cache them locally.

Output: data/raw/elo_ratings.json (~5MB, 247 teams, 1930-2026).

Usage:
    python scripts/fetch_elo_ratings.py
    python scripts/fetch_elo_ratings.py --force     # re-download
    python scripts/fetch_elo_ratings.py --quiet     # suppress progress
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `soccer_ev_model` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soccer_ev_model.elo_ratings import fetch_and_build  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if cache exists at data/raw/elo_ratings.json")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress logging")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output cache path (default: data/raw/elo_ratings.json)")
    args = parser.parse_args()

    out = args.out or (Path(__file__).resolve().parent.parent / "data" / "raw" / "elo_ratings.json")

    cache = fetch_and_build(cache_path=out, force=args.force, quiet=args.quiet)
    teams = cache.get("teams", {})
    years = cache.get("years_covered", [])
    print()
    print(f"  ✅ {len(teams)} teams, {len(years)} years "
          f"({years[0] if years else '?'}..{years[-1] if years else '?'})")
    print(f"  → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
