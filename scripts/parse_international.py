"""
Parse martj42 international results CSV into our internal match format.

Output: data/processed/international_matches.json with all international
matches from 1990 onwards (more data than we need, easier to slice).

Format matches our pi-rating pipeline:
  - date (YYYY-MM-DD)
  - home_team_id, away_team_id (synthesised int from name hash, same as openfootball)
  - home_team, away_team (string names)
  - home_goals, away_goals (int)
  - result ('home' / 'draw' / 'away')
  - tournament (string: 'Friendly', 'FIFA World Cup', 'Euro', etc.)
  - neutral (bool)
"""
import csv
import hashlib
import json
from pathlib import Path

RAW = Path(__file__).parent.parent / "data" / "raw"
OUT = Path(__file__).parent.parent / "data" / "processed" / "international_matches.json"


def team_id(name: str) -> int:
    """Stable int id from team name (matches openfootball convention)."""
    return int(hashlib.md5(name.lower().strip().encode()).hexdigest()[:8], 16)


# Common aliases — some sources use different names for the same team.
# Keys are the SOURCE name; values are the CANONICAL name used in the
# output JSON (which is then hashed to compute team_id).
#
# Rule: "United States" is canonical. Any source that uses "USA" or
# "U.S.A." gets mapped here. Since the intl CSV and the 2026 WC file
# both end up on "United States", the same hash() will give the same
# team_id, and pi-ratings can track the USMNT across the intl history
# and the live 2026 tournament.
ALIASES = {
    "Czech Republic": "Czechia",
    "Bosnia & Herzegovina": "Bosnia-Herzegovina",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "USA": "United States",
    "U.S.A.": "United States",
    "United States": "United States",  # explicit canonical, idempotent
    "US": "United States",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Cape Verde Islands": "Cape Verde",
    "Cabo Verde": "Cape Verde",
    "Curaçao": "Curacao",
    "St. Lucia": "Saint Lucia",
    "Trinidad and Tobago": "Trinidad & Tobago",
}


def normalise(name: str) -> str:
    return ALIASES.get(name, name)


def parse_csv(path: Path) -> list[dict]:
    matches = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                date = row["date"]
                # Only include matches from 1990 onwards — that's 35 years of
                # data, plenty for pi-rating. Older data has too many disbanded
                # teams (USSR, Yugoslavia, etc.) and we'd need a separate
                # mapping for those.
                if date < "1990-01-01":
                    continue

                hg = int(row["home_score"])
                ag = int(row["away_score"])
            except (KeyError, ValueError, TypeError):
                continue

            home = normalise(row["home_team"].strip())
            away = normalise(row["away_team"].strip())
            if hg > ag:
                result = "home"
            elif hg < ag:
                result = "away"
            else:
                result = "draw"

            matches.append({
                "date": date,
                "home_team": home,
                "away_team": away,
                "home_team_id": team_id(home),
                "away_team_id": team_id(away),
                "home_goals": hg,
                "away_goals": ag,
                "result": result,
                "tournament": row.get("tournament", ""),
                "neutral": row.get("neutral", "FALSE") == "TRUE",
            })
    return matches


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    matches = parse_csv(RAW / "international_results.csv")
    matches.sort(key=lambda m: m["date"])
    with open(OUT, "w") as f:
        json.dump(matches, f)

    # Quick summary
    by_year = {}
    for m in matches:
        by_year[m["date"][:4]] = by_year.get(m["date"][:4], 0) + 1
    print(f"Wrote {len(matches)} matches to {OUT}")
    print(f"Date range: {matches[0]['date']} to {matches[-1]['date']}")
    print(f"Tournaments: {len({m['tournament'] for m in matches})}")
    # Top tournaments
    from collections import Counter
    t_count = Counter(m["tournament"] for m in matches)
    print("Top 5 tournaments by match count:")
    for t, n in t_count.most_common(5):
        print(f"  {t}: {n}")


if __name__ == "__main__":
    main()
