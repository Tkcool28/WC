"""Load and query historical Elo ratings from eloratings.net.

We fetch per-year TSV files from https://eloratings.net/YYYY_results.tsv
(each row is a single match with post-match Elo and rating change). We
reconstruct the pre-match Elo per (team, date) and expose a lookup API
that is safe to call from inside the leak-safe feature builder.

Output cache format (JSON) at data/raw/elo_ratings.json:

    {
      "source": "eloratings.net",
      "fetched_at": "2026-06-16T...",
      "years_covered": [1930, 1931, ...],
      "teams": {
        "Argentina": [
          {"date": "1930-07-15", "elo": 1743, "rank": 7},
          ...
        ],
        ...
      }
    }

Why store pre-match Elo (not post)? The feature builder queries "what was
team X's Elo *just before* match M?" and we apply a strict-less-than date
filter, so we only need pre-match snapshots. Storing the pre-match value
directly avoids having to re-derive it on every lookup.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import urllib.error
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)


# Default Elo for teams with no data (or whose data starts after the match)
DEFAULT_ELO = 1500

# Range of years to fetch. 1930 = first World Cup. We grab through
# current year + 1 (the +1 is just to be safe in case eloratings publishes
# early results for next year).
MIN_YEAR = 1930


# ---- Team code -> canonical name ----
#
# The en.teams.tsv from eloratings.net already gives us a canonical name
# (col 2) and aliases (cols 3+). We download that at fetch time and turn
# it into a {code: canonical_name} map. Below is a small extension table
# for names that are common in our training data but missing / different
# in en.teams.tsv. Keys are our-side team names; values are eloratings
# canonical names.

# Map: our_team_name -> eloratings canonical name
# Used by normalize_team_name() to bridge between our data and elo's
OUR_TO_ELO: dict[str, str] = {
    "United States": "USA",
    "South Korea": "Korea Republic",
    "North Korea": "Korea DPR",
    "Iran": "IR Iran",
    "Ivory Coast": "Côte d'Ivoire",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "Cape Verde": "Cabo Verde",
    "DR Congo": "Congo DR",
    "Congo": "Congo",
    "Czech Republic": "Czechia",
    "Czechia": "Czechia",
    "Macedonia": "North Macedonia",
    "FYR Macedonia": "North Macedonia",
    "Republic of Ireland": "Ireland",
    "Ireland": "Ireland",
    "The Gambia": "Gambia",
    "Gambia": "Gambia",
    "Kyrgyzstan": "Kyrgyz Republic",
    "Syria": "Syria",
    "St. Lucia": "Saint Lucia",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "Trinidad and Tobago": "Trinidad & Tobago",
    "Antigua and Barbuda": "Antigua and Barbuda",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Timor-Leste": "Timor-Leste",
    "East Timor": "Timor-Leste",
    "Eswatini": "Eswatini",
    "Swaziland": "Eswatini",
    "Brunei": "Brunei",
    "Chinese Taipei": "Chinese Taipei",
    "Taiwan": "Chinese Taipei",
    "Hong Kong": "Hong Kong",
    "Macau": "Macau",
    "Western Samoa": "Samoa",
    "Samoa": "Samoa",
    "American Samoa": "American Samoa",
    "US Virgin Islands": "US Virgin Islands",
    "British Virgin Islands": "British Virgin Islands",
    "Cayman Islands": "Cayman Islands",
    "Turks and Caicos Islands": "Turks and Caicos Islands",
    "Burma": "Myanmar",
    "Myanmar": "Myanmar",
    "Laos": "Laos",
    "Vietnam": "Vietnam",
    "Viet Nam": "Vietnam",
}


ELO_TO_OUR: dict[str, str] = {
    # Most are 1:1 with our data, but a few need remapping for display
    "USA": "United States",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "Czechia": "Czechia",
    "North Macedonia": "North Macedonia",
    "Ireland": "Ireland",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Trinidad & Tobago": "Trinidad and Tobago",
}


# Sources of data
ELO_BASE = "https://eloratings.net"
ELO_TEAMS_URL = f"{ELO_BASE}/en.teams.tsv"        # code -> canonical name (+aliases)
ELO_SUCCESSOR_URL = f"{ELO_BASE}/teams.tsv"        # old_code -> new_code
ELO_YEAR_URL = lambda y: f"{ELO_BASE}/{y}_results.tsv"  # one file per year


# --- HTTP helpers ---


def _http_get(url: str, timeout: float = 30.0) -> str:
    """GET a URL, return text. Raise on HTTP error."""
    req = urllib.request.Request(url, headers={"User-Agent": "soccer-ev-model/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


# --- Parsing ---


def _parse_signed_int(s: str) -> int:
    """Parse a signed integer. Handles '+12', '-3', '−3' (Unicode minus), '0'."""
    s = s.strip().replace("\u2212", "-")  # Unicode minus
    if not s:
        return 0
    return int(s)


def _parse_year_results_tsv(text: str) -> Iterable[dict]:
    """Yield per-match dicts from one year of eloratings TSV.

    Columns (tab-separated, no header):
        0  year
        1  month
        2  day
        3  home_team_code
        4  away_team_code
        5  home_goals
        6  away_goals
        7  tournament_code
        8  neutral_country (empty if not neutral)
        9  rank_change_home  (signed; + means home's rank improved post-match,
                              i.e. pre_rank > post_rank)
        10 home_elo_post     (Elo after this match)
        11 away_elo_post     (Elo after this match)
        12 rating_change_home (signed)
        13 rating_change_away (signed)
        14 home_rank_post
        15 away_rank_post
    """
    reader = csv.reader(io.StringIO(text), delimiter="\t")
    for row in reader:
        if len(row) < 16 or not row[0].strip().isdigit():
            continue
        try:
            y, m, d = int(row[0]), int(row[1]), int(row[2])
            match_date = date(y, m, d)
            home_elo_post = int(row[10])
            away_elo_post = int(row[11])
            home_d_elo = _parse_signed_int(row[12])
            away_d_elo = _parse_signed_int(row[13])
            home_rank_chg = _parse_signed_int(row[9])
            away_rank_chg = _parse_signed_int(row[15])  # not used
            yield {
                "date": match_date,
                "home_code": row[3].strip(),
                "away_code": row[4].strip(),
                "home_elo_pre": home_elo_post - home_d_elo,
                "away_elo_pre": away_elo_post - away_d_elo,
                "home_rank_pre": int(row[14]) + home_rank_chg,
                "away_rank_pre": int(row[15]) - _parse_signed_int(row[9])
                if False
                else None,  # we don't actually need pre-rank; keep None
                "tournament": row[7].strip(),
                "neutral_country": row[8].strip(),
                "home_goals": int(row[5]) if row[5].strip() else None,
                "away_goals": int(row[6]) if row[6].strip() else None,
            }
        except (ValueError, IndexError) as e:
            log.debug("Skipping bad row %r: %s", row, e)
            continue


def parse_en_teams_tsv(text: str) -> dict[str, list[str]]:
    """Parse en.teams.tsv into {code: [canonical_name, alias1, ...]}.

    The first non-code column is the canonical name. Subsequent columns
    are alternate names/spellings.
    """
    out: dict[str, list[str]] = {}
    reader = csv.reader(io.StringIO(text), delimiter="\t")
    for row in reader:
        if len(row) < 2:
            continue
        code = row[0].strip()
        if not code:
            continue
        names = [c.strip() for c in row[1:] if c.strip()]
        if not names:
            continue
        out[code] = names
    return out


def parse_successor_tsv(text: str) -> dict[str, str]:
    """Parse teams.tsv successor mapping into {old_code: new_code}."""
    out: dict[str, str] = {}
    reader = csv.reader(io.StringIO(text), delimiter="\t")
    for row in reader:
        if len(row) < 2:
            continue
        old = row[0].strip()
        new = row[1].strip()
        if old and new:
            out[old] = new
    return out


# --- Public API: build the per-team snapshots from raw year files ---


def build_elo_snapshots(
    year_texts: dict[int, str],
    en_teams: dict[str, list[str]],
    successor: dict[str, str],
) -> dict[str, list[dict]]:
    """Build per-team Elo snapshots from raw year files.

    Returns:
        {team_name (canonical, our-side): [{"date": date, "elo": int, "rank": int}, ...]}
        The lists are sorted by date.
    """
    # Map code -> canonical name (the FIRST name in en.teams is canonical)
    code_to_name: dict[str, str] = {}
    for code, names in en_teams.items():
        canonical = names[0]
        # Apply successor: if this code has been superseded, route to successor
        if code in successor:
            succ = successor[code]
            succ_names = en_teams.get(succ)
            if succ_names:
                canonical = succ_names[0]
        code_to_name[code] = canonical

    # Now we still need OUR-side names for matching. The en.teams.tsv
    # canonical names are "USA", "Korea Republic", etc. - not always the
    # spelling in our data. We'll keep two parallel views:
    # - snapshots_by_elo_name: {elo_canonical: [(date, elo)]}
    # - snapshots_by_our_name: {our_canonical: [(date, elo)]}
    #
    # At lookup time the caller can ask for either form; we'll also
    # apply OUR_TO_ELO when looking up.

    snapshots: dict[str, list[dict]] = {}

    for year, text in sorted(year_texts.items()):
        for m in _parse_year_results_tsv(text):
            for code, pre_elo, pre_rank in (
                (m["home_code"], m["home_elo_pre"], m["home_rank_pre"]),
                (m["away_code"], m["away_elo_pre"], None),
            ):
                name = code_to_name.get(code)
                if not name:
                    # Unknown code - skip rather than crash. Most are
                    # very-low-rank teams (Tuvalu, etc.) that won't appear
                    # in our training data.
                    continue
                snapshots.setdefault(name, []).append(
                    {"date": m["date"], "elo": pre_elo, "rank": pre_rank}
                )

    # Sort each team's list chronologically; dedupe same-day duplicates
    # (keep the first occurrence for that date).
    for name, snaps in snapshots.items():
        snaps.sort(key=lambda s: s["date"])
        deduped: list[dict] = []
        last_date: date | None = None
        for s in snaps:
            if last_date is not None and s["date"] == last_date:
                continue
            deduped.append(s)
            last_date = s["date"]
        snapshots[name] = deduped

    return snapshots


# --- Public API: caching ---


def fetch_and_build(
    years: Iterable[int] | None = None,
    cache_path: Path | None = None,
    force: bool = False,
    quiet: bool = False,
) -> dict:
    """Fetch the Elo data from eloratings.net and build a per-team cache.

    Args:
        years: which years to fetch (default = 1930 .. current_year)
        cache_path: where to write the JSON cache. If it exists and force
            is False, we load from cache instead of re-fetching.
        force: re-download even if cache exists.
        quiet: suppress progress logging.

    Returns:
        A dict suitable for json.dump with the structure:
            {
                "source": "eloratings.net",
                "fetched_at": ISO-8601 string,
                "years_covered": [...],
                "teams": {team_name: [snapshots]}
            }
    """
    if cache_path is None:
        cache_path = Path("data/raw/elo_ratings.json")
    cache_path = Path(cache_path)

    if cache_path.exists() and not force:
        if not quiet:
            log.info("Loading Elo cache from %s", cache_path)
        return json.loads(cache_path.read_text())

    if years is None:
        end_year = datetime.utcnow().year + 1
        years = range(MIN_YEAR, end_year + 1)

    year_texts: dict[int, str] = {}
    if not quiet:
        log.info("Fetching en.teams.tsv ...")
    en_teams = parse_en_teams_tsv(_http_get(ELO_TEAMS_URL))
    if not quiet:
        log.info("Fetching teams.tsv (successor map) ...")
    successor = parse_successor_tsv(_http_get(ELO_SUCCESSOR_URL))

    years_list = list(years)
    for y in years_list:
        url = ELO_YEAR_URL(y)
        try:
            text = _http_get(url, timeout=20.0)
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            log.warning("Skipping year %d: %s", y, e)
            continue
        if not text or text.lstrip().startswith("<"):
            log.warning("Skipping year %d: non-TSV response", y)
            continue
        year_texts[y] = text
        if not quiet:
            log.info("  %d: %d bytes", y, len(text))

    if not quiet:
        log.info("Building per-team snapshots ...")
    snapshots = build_elo_snapshots(year_texts, en_teams, successor)

    cache = {
        "source": "eloratings.net",
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "years_covered": sorted(year_texts.keys()),
        "teams": {
            name: [
                {"date": s["date"].isoformat(), "elo": s["elo"],
                 "rank": s["rank"]}
                for s in snaps
            ]
            for name, snaps in snapshots.items()
        },
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache))
    if not quiet:
        log.info("Wrote %d teams to %s", len(cache["teams"]), cache_path)
    return cache


# --- Public API: lookup ---


def normalize_team_name(name: str) -> str:
    """Map an our-side team name to its eloratings canonical form.

    If the name is already in OUR_TO_ELO, return that. Otherwise return
    the name unchanged. The lookup is conservative - we don't try to be
    clever about substring matching. If a team isn't in OUR_TO_ELO, the
    lookup will just fail to find Elo data for it (we'll fall back to
    the default 1500 with the *_elo_missing flag).
    """
    return OUR_TO_ELO.get(name, name)


def load_elo_ratings(cache_path: Path | str) -> dict[str, list[dict]]:
    """Load the per-team Elo snapshots from the JSON cache.

    Returns:
        {team_name: [{"date": date, "elo": int, "rank": int|None}, ...]}
        Lists are sorted by date.
    """
    cache_path = Path(cache_path)
    raw = json.loads(cache_path.read_text())
    out: dict[str, list[dict]] = {}
    for name, snaps in raw.get("teams", {}).items():
        out[name] = [
            {
                "date": date.fromisoformat(s["date"]),
                "elo": int(s["elo"]),
                "rank": s.get("rank"),
            }
            for s in snaps
        ]
        out[name].sort(key=lambda s: s["date"])
    return out


def elo_at(
    snapshots: dict[str, list[dict]],
    team_name: str,
    match_date: str | date,
) -> tuple[int, bool]:
    """Return (elo, missing) for a team at a given date.

    The Elo returned is the team's Elo strictly BEFORE `match_date` (i.e.
    the most recent snapshot with date < match_date). If no such snapshot
    exists, returns (DEFAULT_ELO, True).

    The `missing` flag is True when the team has no Elo data at all OR
    no snapshot before the match date.
    """
    if isinstance(match_date, str):
        # Allow "2022-11-20T19:00:00Z" or "2022-11-20"
        d = date.fromisoformat(match_date[:10])
    else:
        d = match_date

    # Try the raw name first, then the eloratings canonical
    candidates = [team_name, normalize_team_name(team_name)]
    snaps: list[dict] | None = None
    for c in candidates:
        if c in snapshots:
            snaps = snapshots[c]
            break
    if snaps is None or not snaps:
        return DEFAULT_ELO, True

    # Binary search for the last snapshot with date < d.
    # Linear is fine for the typical list length (~30-100 entries).
    chosen = DEFAULT_ELO
    found = False
    for s in snaps:
        if s["date"] < d:
            chosen = s["elo"]
            found = True
        else:
            break
    if not found:
        return DEFAULT_ELO, True
    return chosen, False
