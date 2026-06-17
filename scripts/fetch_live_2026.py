"""Fetch the LIVE 2026 World Cup data from football-data.org.

This gets today's matches, live scores, and the full current tournament
(2026). It uses the football-data.org API client (rate-limited, polite UA).

Use this DURING the tournament to keep your local cache fresh. The
openfootball fetch_historical_wc.py script handles past tournaments.

Usage:
    python scripts/fetch_live_2026.py
    python scripts/fetch_live_2026.py --force
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soccer_ev_model.api_client import FootballDataClient
from soccer_ev_model.fetch_data import fetch_year


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if cache exists")
    parser.add_argument("--min-delay", type=float, default=6.0,
                        help="Seconds between API calls (default 6 = safe)")
    args = parser.parse_args()

    print(f"Fetching live 2026 WC from football-data.org")
    print(f"Min delay between calls: {args.min_delay}s (be polite)")
    print()

    client = FootballDataClient(min_delay=args.min_delay)
    t0 = time.time()
    try:
        path = fetch_year(client, year=2026,
                          out_dir=Path(__file__).resolve().parent.parent / "data" / "raw",
                          force=args.force)
        elapsed = time.time() - t0
        import json
        data = json.loads(path.read_text())
        print(f"  ✅ 2026: {data['count']} matches -> {path.name} ({elapsed:.1f}s)")
        return 0
    except Exception as e:
        print(f"  ❌ {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
