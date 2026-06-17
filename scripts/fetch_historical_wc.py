"""Bulk-fetch historical World Cup data from openfootball.

Pulls WC 2010, 2014, 2018, 2022, and 2026 (the modern 32/48-team era) from
the openfootball/worldcup.json GitHub repo. Caches each year to
data/raw/matches_<year>_openfootball.json.

This is a one-shot script. Re-running it does nothing if the files already
exist (use --force to re-fetch).

Usage:
    python scripts/fetch_historical_wc.py
    python scripts/fetch_historical_wc.py --force
    python scripts/fetch_historical_wc.py --years 2022,2018
"""

import argparse
import sys
import time
from pathlib import Path

# Make the package importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soccer_ev_model.fetch_data import fetch_openfootball_year


# Modern-era WC years we care about (32-team format through 2022, 48-team 2026)
DEFAULT_YEARS = [2010, 2014, 2018, 2022, 2026]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years",
        type=str,
        default=",".join(str(y) for y in DEFAULT_YEARS),
        help="Comma-separated list of WC years to fetch",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even if cache file exists",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw",
        help="Output directory for cached files",
    )
    args = parser.parse_args()

    years = [int(y) for y in args.years.split(",")]
    print(f"Fetching WC data for years: {years}")
    print(f"Output dir: {args.out_dir}")
    print(f"Force re-fetch: {args.force}")
    print()

    t0 = time.time()
    succeeded = []
    failed = []
    not_found = []

    for year in years:
        try:
            path = fetch_openfootball_year(year, args.out_dir, force=args.force)
            if path is None:
                print(f"  ⚠ {year}: not found in openfootball repo (404)")
                not_found.append(year)
            else:
                # Read count for summary
                import json
                data = json.loads(path.read_text())
                print(f"  ✅ {year}: {data['count']} matches -> {path.name}")
                succeeded.append(year)
        except Exception as e:
            print(f"  ❌ {year}: {type(e).__name__}: {e}")
            failed.append((year, e))

    elapsed = time.time() - t0
    print()
    print(f"Done in {elapsed:.1f}s. Succeeded: {len(succeeded)}, "
          f"failed: {len(failed)}, not in repo: {len(not_found)}")
    if failed:
        for y, e in failed:
            print(f"  FAILED {y}: {e}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
