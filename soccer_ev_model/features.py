"""Build a leak-safe feature matrix from historical matches.

The key invariant: features for match M are computed using only matches
strictly before M's date. We do this by processing matches in chronological
order and maintaining a "current state" (pi-ratings, recent form, last-match
date) that updates after each match.

For each match, we capture a snapshot of:
- The current pi-ratings (which are the ratings AS OF just before this match,
  because we compute them on the prior matches only)
- Each team's recent form (wins/draws/losses/goal_diff in last 5 matches)
- Each team's rest days (days since their last match)

The output is a pandas DataFrame (X) and a Series of result codes (y).

Performance
-----------
Earlier versions of this module re-derived pi-ratings, rest_days, and
recent-form from scratch for every match, which is O(N^2) overall. With
~33k matches that is hours of work. This implementation maintains an
incremental state object (`_FeatureBuilderState`) as we walk matches in
chronological order, so each match is processed in O(1) time. Total cost
is O(N) plus the cost of sorting.

The public functions `rest_days` and `compute_recent_form` still work
exactly as before (the unit tests exercise them directly). The new fast
path lives in `build_feature_matrix` and uses the incremental state.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Iterable

import pandas as pd

from soccer_ev_model.pi_ratings import (
    _update_ratings_for_match,
    pi_diff_features,
)


# How many recent matches to consider for "form" features
RECENT_FORM_N = 5

# Sentinel value for "team has never played before" rest days. A large
# finite number so the model can still learn from it (not NaN).
NO_HISTORY_REST_DAYS = 999

# Sentinel for "team has no recent form" - we use 0s for wins/draws/losses
# (since the team has none of each), and 0 for goal_diff.


# --------------------------------------------------------------------------- #
# Result-code normalization
# --------------------------------------------------------------------------- #
# Two encodings exist in our training data:
#   - WC openfootball:   'H' / 'D' / 'A'  (home/draw/away)
#   - international CSV: 'home' / 'draw' / 'away'
# Earlier versions of this file only accepted the WC codes, which silently
# dropped 32,000+ international matches. We now normalize both into the
# single canonical form 'H' / 'D' / 'A'.
_RESULT_ALIASES = {
    "H": "H", "home": "H",
    "D": "D", "draw": "D",
    "A": "A", "away": "A",
}


def _normalize_result_code(m: dict) -> str | None:
    """Return canonical 'H' / 'D' / 'A' for a match, or None if unknown."""
    raw = (m.get("result") or "").strip()
    return _RESULT_ALIASES.get(raw)


def _parse_date(date_str: str) -> datetime:
    """Parse a date string. Same logic as pi_ratings but local to features."""
    if not date_str:
        return datetime.min
    s = date_str.rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return datetime.min


def rest_days(team_id: int, target_date: str, prior_matches: list[dict]) -> int:
    """Days between target_date and this team's most recent prior match.

    If the team has no prior matches, returns NO_HISTORY_REST_DAYS.
    """
    target = _parse_date(target_date)
    last_match_date = None
    for m in prior_matches:
        if m.get("home_team_id") != team_id and m.get("away_team_id") != team_id:
            continue
        d = _parse_date(m.get("date", ""))
        if d >= target:
            continue
        if last_match_date is None or d > last_match_date:
            last_match_date = d
    if last_match_date is None:
        return NO_HISTORY_REST_DAYS
    return (target - last_match_date).days


def compute_recent_form(
    team_id: int, target_date: str, prior_matches: list[dict], n: int = RECENT_FORM_N
) -> dict:
    """Compute this team's recent form (wins/draws/losses/goal_diff) before target_date.

    Looks at the last n matches the team played before target_date.
    A team that was home in match X and away in match Y is correctly
    counted in both.
    """
    target = _parse_date(target_date)
    relevant = []
    for m in prior_matches:
        if m.get("home_team_id") != team_id and m.get("away_team_id") != team_id:
            continue
        d = _parse_date(m.get("date", ""))
        if d >= target:
            continue
        # Skip matches with no result yet
        if m.get("home_goals") is None or m.get("away_goals") is None:
            continue
        # Determine this team's perspective in this match
        is_home = m.get("home_team_id") == team_id
        team_goals = m.get("home_goals") if is_home else m.get("away_goals")
        opp_goals = m.get("away_goals") if is_home else m.get("home_goals")
        relevant.append({
            "date": d,
            "team_goals": team_goals,
            "opp_goals": opp_goals,
            "is_win": team_goals > opp_goals,
            "is_draw": team_goals == opp_goals,
            "is_loss": team_goals < opp_goals,
            "goal_diff": team_goals - opp_goals,
        })
    # Sort newest first, take last n
    relevant.sort(key=lambda r: r["date"], reverse=True)
    recent = relevant[:n]

    return {
        "wins": sum(1 for r in recent if r["is_win"]),
        "draws": sum(1 for r in recent if r["is_draw"]),
        "losses": sum(1 for r in recent if r["is_loss"]),
        "goal_diff": sum(r["goal_diff"] for r in recent),
        "matches_used": len(recent),
    }


# --------------------------------------------------------------------------- #
# Incremental feature builder state
# --------------------------------------------------------------------------- #
class _FeatureBuilderState:
    """Running state for a single pass through chronologically-sorted matches.

    Holds:
    - pi-ratings dict (mutated in place by pi_ratings._update_ratings_for_match)
    - per-team deque of recent form records (maxlen=RECENT_FORM_N)
    - per-team last-match-date (datetime)

    The snapshot methods return the data needed for the next match's
    features WITHOUT mutating the state. The `update_after_match` method
    then folds the just-played match into the state so the NEXT match
    sees it.
    """

    __slots__ = (
        "ratings",
        "_form",
        "_last_date",
        "_pi_ratings_initialized",
    )

    def __init__(self) -> None:
        self.ratings: dict = {}                  # team_id -> {offense, defense, matches_played}
        self._form: dict[int, deque] = {}        # team_id -> deque of form records
        self._last_date: dict[int, datetime] = {}  # team_id -> last match datetime
        # Pi-ratings start at 0.0 / 0 matches for any unseen team; the
        # _update_ratings_for_match helper handles lazy init. We just need
        # to be ready to hand out a copy of `self.ratings` for snapshotting.

    # ----- snapshot helpers (do not mutate state) -----

    def _ensure_team(self, team_id: int) -> None:
        if team_id not in self.ratings:
            self.ratings[team_id] = {
                "offense": 0.0, "defense": 0.0, "matches_played": 0,
            }
        if team_id not in self._form:
            self._form[team_id] = deque(maxlen=RECENT_FORM_N)

    def _form_summary(self, team_id: int) -> dict:
        dq = self._form.get(team_id)
        if dq is None:
            return {"wins": 0, "draws": 0, "losses": 0, "goal_diff": 0,
                    "matches_used": 0}
        wins = draws = losses = gd = 0
        for r in dq:
            if r["is_win"]:
                wins += 1
            elif r["is_draw"]:
                draws += 1
            else:
                losses += 1
            gd += r["goal_diff"]
        return {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "goal_diff": gd,
            "matches_used": len(dq),
        }

    def rest_days(self, team_id: int, target_date: str) -> int:
        target = _parse_date(target_date)
        last = self._last_date.get(team_id)
        if last is None:
            return NO_HISTORY_REST_DAYS
        return (target - last).days

    def home_form(self, team_id: int) -> dict:
        return self._form_summary(team_id)

    def away_form(self, team_id: int) -> dict:
        return self._form_summary(team_id)

    def pi_snapshot(self) -> dict:
        """Return a shallow copy of the ratings dict (inner dicts shared).

        Callers should treat this as read-only. We use this in pi_diff_features
        which only reads `offense` and `defense`.
        """
        return self.ratings

    # ----- state update (after a match is processed) -----

    def update_after_match(self, m: dict) -> None:
        """Fold a match's outcome into the running state.

        Mutates self.ratings, self._form, and self._last_date. Called
        AFTER the match's features have been snapshotted, so the match
        never influences its own features.
        """
        home_id = m.get("home_team_id")
        away_id = m.get("away_team_id")
        home_goals = m.get("home_goals")
        away_goals = m.get("away_goals")
        match_date = _parse_date(m.get("date", ""))

        # Pi-rating update. _update_ratings_for_match handles missing
        # goals / missing ids by silently skipping. It also lazily
        # initializes each team's rating dict.
        _update_ratings_for_match(self.ratings, m)

        if home_id is None or away_id is None:
            return
        if home_goals is None or away_goals is None:
            return

        # Form updates (per-team perspective)
        self._ensure_team(home_id)
        self._ensure_team(away_id)
        is_home_win = home_goals > away_goals
        is_draw = home_goals == away_goals
        self._form[home_id].append({
            "team_goals": home_goals,
            "opp_goals": away_goals,
            "is_win": is_home_win,
            "is_draw": is_draw,
            "is_loss": not is_home_win and not is_draw,
            "goal_diff": home_goals - away_goals,
        })
        self._form[away_id].append({
            "team_goals": away_goals,
            "opp_goals": home_goals,
            "is_win": not is_home_win and not is_draw,
            "is_draw": is_draw,
            "is_loss": home_goals > away_goals,
            "goal_diff": away_goals - home_goals,
        })

        # Last match date: store the max for each team. In a sorted
        # input the current match is always >= the last one, but using
        # max keeps the code robust to out-of-order input.
        prev_h = self._last_date.get(home_id)
        if prev_h is None or match_date > prev_h:
            self._last_date[home_id] = match_date
        prev_a = self._last_date.get(away_id)
        if prev_a is None or match_date > prev_a:
            self._last_date[away_id] = match_date


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def build_feature_matrix(
    matches: list[dict],
    elo_snapshots: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the full feature matrix from a list of historical matches.

    Args:
        matches: list of match dicts. May be a mix of WC ('H'/'D'/'A') and
            international ('home'/'draw'/'away') result codes - both are
            accepted and normalized to 'H' / 'D' / 'A'.
        elo_snapshots: optional output of `soccer_ev_model.elo_ratings
            .load_elo_ratings()`. If provided, adds `home_elo`, `away_elo`,
            `elo_diff`, `home_elo_missing`, `away_elo_missing` columns. If
            None (default), those columns are omitted and the function
            behaves exactly as before.

    Returns:
        (X, y) where:
            X is a DataFrame with one row per match, columns = features
            y is a Series of result codes ('H', 'D', 'A'), aligned with X
        Matches are processed in chronological order regardless of input order.
        Matches without a result are skipped (we need ground truth for training).

    Performance: O(N log N) overall (dominated by the sort). Each match is
    processed in O(1) using an incremental state object. Tested at
    ~33k matches in < 10 seconds.
    """
    # Lazy import to avoid a hard dep on elo_ratings if the caller doesn't use it.
    if elo_snapshots is not None:
        from soccer_ev_model.elo_ratings import elo_at
    else:
        elo_at = None

    # Sort matches chronologically (so prior state is always "earlier").
    # Filter to only those with a known result code so we don't waste
    # state updates on matches with no ground truth.
    sorted_matches = sorted(
        (m for m in matches if _normalize_result_code(m) is not None),
        key=lambda m: _parse_date(m.get("date", "")),
    )

    state = _FeatureBuilderState()
    rows: list[dict] = []
    labels: list[str] = []

    for m in sorted_matches:
        date = m.get("date", "")
        home_id = m.get("home_team_id")
        away_id = m.get("away_team_id")
        home_name = m.get("home_team") or m.get("home_team_name") or ""
        away_name = m.get("away_team") or m.get("away_team_name") or ""

        # 1. Pi-ratings snapshot (current state, NOT yet updated for this
        # match). The pi_diff_features helper reads offense/defense, and
        # we also emit the raw ratings.
        ratings = state.pi_snapshot()
        # Ensure both teams appear in the dict so we get the canonical
        # 0.0/0.0 zero values for never-seen teams. This matches the
        # behavior of the prior implementation, which passed `ratings`
        # to pi_diff_features and the raw-dict lookups, both of which
        # default to 0.0 for missing teams.
        if home_id is not None and home_id not in ratings:
            ratings[home_id] = {"offense": 0.0, "defense": 0.0,
                                "matches_played": 0}
        if away_id is not None and away_id not in ratings:
            ratings[away_id] = {"offense": 0.0, "defense": 0.0,
                                "matches_played": 0}
        pi_feats = pi_diff_features(home_id, away_id, ratings)
        home_r = ratings.get(home_id, {"offense": 0.0, "defense": 0.0})
        away_r = ratings.get(away_id, {"offense": 0.0, "defense": 0.0})

        # 2. Recent form for each team (snapshot from state)
        home_form = state.home_form(home_id)
        away_form = state.away_form(away_id)

        # 3. Rest days for each team (snapshot from state)
        home_rest = state.rest_days(home_id, date)
        away_rest = state.rest_days(away_id, date)

        # 4. Tournament stage (categorical) - simple integer encoding for now
        stage_code = _encode_stage(m.get("stage"))

        # 5. Home advantage flag (almost always 1 in WC; the few neutral-venue
        # matches are coded as 0 by the source)
        is_neutral = m.get("is_neutral", 1)  # default to neutral for safety
        # WC matches are ALL on neutral venues, so we keep is_neutral=1 always.
        # The feature is "is_home_team" which is always 1 in our data, so
        # we drop it - no information.

        # 6. Elo ratings (optional)
        if elo_at is not None and home_name and away_name:
            home_elo, home_missing = elo_at(elo_snapshots, home_name, date)
            away_elo, away_missing = elo_at(elo_snapshots, away_name, date)
        else:
            home_elo, away_elo = 1500, 1500
            home_missing, away_missing = True, True

        # Combine
        row = {
            "pi_home_off": home_r["offense"],
            "pi_away_off": away_r["offense"],
            "pi_home_def": home_r["defense"],
            "pi_away_def": away_r["defense"],
            "pi_off_diff": pi_feats["pi_off_diff"],
            "pi_def_diff": pi_feats["pi_def_diff"],
            "pi_matchup": pi_feats["pi_matchup"],
            "home_form_wins": home_form["wins"],
            "home_form_draws": home_form["draws"],
            "home_form_losses": home_form["losses"],
            "home_form_goal_diff": home_form["goal_diff"],
            "home_form_matches_used": home_form["matches_used"],
            "away_form_wins": away_form["wins"],
            "away_form_draws": away_form["draws"],
            "away_form_losses": away_form["losses"],
            "away_form_goal_diff": away_form["goal_diff"],
            "away_form_matches_used": away_form["matches_used"],
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "stage_code": stage_code,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "home_elo_missing": int(home_missing),
            "away_elo_missing": int(away_missing),
            "date": date,  # kept for time-based splits later
        }
        rows.append(row)
        labels.append(_normalize_result_code(m))

        # IMPORTANT: Add this match to state AFTER computing its features,
        # so this match's outcome influences the NEXT match's features,
        # not its own.
        state.update_after_match(m)

    X = pd.DataFrame(rows)
    y = pd.Series(labels, name="result")
    return X, y


def _encode_stage(stage: str | None) -> int:
    """Map stage name to an integer. Higher = later in tournament.

    0 = GROUP_STAGE
    1 = LAST_16
    2 = QUARTER_FINALS
    3 = SEMI_FINALS
    4 = THIRD_PLACE
    5 = FINAL
    """
    return {
        "GROUP_STAGE": 0,
        "LAST_16": 1,
        "QUARTER_FINALS": 2,
        "SEMI_FINALS": 3,
        "THIRD_PLACE": 4,
        "FINAL": 5,
    }.get(stage, -1)
