"""Independent, interpretable international-football goal models.

The models in this file use historical scorelines directly.  They do not derive
expected goals from the existing Pi/Elo 1X2 blend.

All models are transparent and deterministic:
  - GlobalPoisson: single rate per context (home/away/neutral)
  - EloPoisson: Elo-driven independent Poisson with fitted slope
  - RegularizedTeamPoisson: team attack/defense with shrinkage

No black-box ML.  No random train/test splits.  No future leakage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Iterable, Mapping, Optional

import numpy as np

from .goal_model_data import GoalMatch, classify_tournament

MODEL_VERSION = "goal-model-research-v0.2"

# Epsilon for safe log computations
_EPS = 1e-12

# Default max goals for scoreline matrix
_DEFAULT_MAX_GOALS = 15


def _poisson_pmf(k: int, lam: float) -> float:
    """Poisson PMF.  Returns 0 for negative lambda."""
    if lam <= 0:
        return 0.0
    return math.exp(-lam) * lam ** k / math.factorial(k)


def scoreline_matrix(
    home_xg: float,
    away_xg: float,
    max_goals: int = _DEFAULT_MAX_GOALS,
) -> tuple[np.ndarray, float, float]:
    """Build a normalized scoreline probability matrix.

    Returns:
        (matrix, raw_mass, tail_mass):
            matrix: (max_goals+1, max_goals+1) normalized so sum == 1
            raw_mass: sum before normalization (should be close to 1.0)
            tail_mass: 1.0 - raw_mass (mass lost to truncation)

    Raises:
        ValueError: if expected goals are non-finite or negative.
    """
    if not (math.isfinite(home_xg) and math.isfinite(away_xg)):
        raise ValueError("expected goals must be finite")
    if home_xg < 0 or away_xg < 0:
        raise ValueError("expected goals must be nonnegative")

    hp = np.array([_poisson_pmf(i, home_xg) for i in range(max_goals + 1)], dtype=float)
    ap = np.array([_poisson_pmf(i, away_xg) for i in range(max_goals + 1)], dtype=float)
    matrix = np.outer(hp, ap)
    raw_mass = float(matrix.sum())
    if raw_mass <= 0:
        raise ValueError("scoreline matrix has zero mass")
    tail_mass = 1.0 - raw_mass
    matrix = matrix / raw_mass
    return matrix, raw_mass, tail_mass


def summarize_prediction(
    home_xg: float,
    away_xg: float,
    *,
    max_goals: int = _DEFAULT_MAX_GOALS,
    low_data_flags: list[str] | None = None,
    data_cutoff: str | None = None,
    model_version: str = MODEL_VERSION,
) -> dict:
    """Build a full prediction summary from expected goals.

    Returns a dict with:
        - home_xg, away_xg
        - score_probs: the full normalized matrix
        - hda_probs: home/draw/away probabilities
        - most_likely_score: [hg, ag]
        - expected_total_goals
        - raw_matrix_mass, tail_mass: truncation diagnostics
        - model_version, data_cutoff, low_data_flags
    """
    matrix, raw_mass, tail_mass = scoreline_matrix(home_xg, away_xg, max_goals=max_goals)
    h = float(np.tril(matrix, -1).sum())
    d = float(np.trace(matrix))
    a = float(np.triu(matrix, 1).sum())
    idx = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
    return {
        "home_xg": float(home_xg),
        "away_xg": float(away_xg),
        "score_probs": matrix.tolist(),
        "hda_probs": {"home": h, "draw": d, "away": a},
        "most_likely_score": [int(idx[0]), int(idx[1])],
        "expected_total_goals": float(home_xg + away_xg),
        "raw_matrix_mass": raw_mass,
        "tail_mass": tail_mass,
        "model_version": model_version,
        "data_cutoff": data_cutoff,
        "low_data_flags": list(low_data_flags or []),
    }


# ---------------------------------------------------------------------------
# Global Poisson
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GlobalPoissonModel:
    """Single-rate Poisson for all matches in a context.

    Three rates are estimated:
        home_rate: mean goals for non-neutral home teams
        away_rate: mean goals for non-neutral away teams
        neutral_rate: mean goals per team in neutral matches

    For neutral matches, both teams get the same rate (no home advantage).
    """

    home_rate: float
    away_rate: float
    neutral_rate: float
    data_cutoff: str | None = None

    @classmethod
    def fit(cls, matches: Iterable[GoalMatch]) -> "GlobalPoissonModel":
        rows = list(matches)
        if not rows:
            raise ValueError("cannot fit on an empty match set")
        non_neutral = [m for m in rows if not m.neutral]
        neutral = [m for m in rows if m.neutral]

        if non_neutral:
            home_rate = sum(m.home_goals for m in non_neutral) / len(non_neutral)
            away_rate = sum(m.away_goals for m in non_neutral) / len(non_neutral)
        else:
            # Fallback: use overall mean
            total_goals = sum(m.home_goals + m.away_goals for m in rows)
            home_rate = total_goals / (2 * len(rows))
            away_rate = home_rate

        if neutral:
            neutral_rate = sum(m.home_goals + m.away_goals for m in neutral) / (2 * len(neutral))
        else:
            neutral_rate = (home_rate + away_rate) / 2

        return cls(
            home_rate, away_rate, neutral_rate,
            max(m.match_date for m in rows).isoformat(),
        )

    def predict(self, *, neutral: bool = False, **_: object) -> dict:
        if neutral:
            hxg = self.neutral_rate
            axg = self.neutral_rate
        else:
            hxg = self.home_rate
            axg = self.away_rate
        return summarize_prediction(
            hxg, axg, data_cutoff=self.data_cutoff,
            model_version=f"{MODEL_VERSION}-global",
        )


# ---------------------------------------------------------------------------
# Elo-driven Poisson
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EloPoissonModel:
    """Elo-driven independent Poisson model.

    log(lambda_home) = intercept_home + elo_slope * (home_elo - away_elo)
    log(lambda_away) = intercept_away - elo_slope * (home_elo - away_elo)

    For neutral matches, the intercepts are adjusted so that
    the average of home and away rates equals the neutral_rate,
    but the Elo differential still affects both directions.

    The elo_slope is selected by grid search on training data NLL.
    """

    intercept_home: float
    intercept_away: float
    elo_slope: float
    neutral_rate: float
    data_cutoff: str | None = None
    slope_grid_tested: tuple = ()
    slope_nll_scores: tuple = ()

    @classmethod
    def fit(
        cls,
        matches: Iterable[GoalMatch],
        elo_by_match: Mapping[tuple, tuple[float, float]],
        slope_grid: Iterable[float] = (-0.0015, -0.001, -0.0005, 0.0, 0.0005, 0.001, 0.0015),
    ) -> "EloPoissonModel":
        """Fit the Elo Poisson model.

        Args:
            matches: training matches
            elo_by_match: mapping of (date_iso, home_name, away_name) -> (home_elo, away_elo)
            slope_grid: grid of slope values to test

        Returns:
            Fitted EloPoissonModel with best slope.
        """
        rows = [m for m in matches if (m.match_date.isoformat(), m.home_team, m.away_team) in elo_by_match]
        if not rows:
            raise ValueError("no matches have Elo inputs")

        slope_list = list(slope_grid)
        best = None

        for slope in slope_list:
            # For each slope, compute optimal intercepts via MLE for Poisson:
            # intercept_home = mean(log(home_goals) - slope * diff)
            # intercept_away = mean(log(away_goals) + slope * diff)
            # We use log(goals + 0.25) to handle 0-goal matches gracefully.
            h_logs = []
            a_logs = []
            for m in rows:
                he, ae = elo_by_match[(m.match_date.isoformat(), m.home_team, m.away_team)]
                diff = he - ae
                h_logs.append(math.log(m.home_goals + 0.25) - slope * diff)
                a_logs.append(math.log(m.away_goals + 0.25) + slope * diff)

            ih = float(np.mean(h_logs))
            ia = float(np.mean(a_logs))

            # Compute NLL with these intercepts
            nll = 0.0
            for m in rows:
                he, ae = elo_by_match[(m.match_date.isoformat(), m.home_team, m.away_team)]
                diff = he - ae
                hxg = math.exp(ih + slope * diff)
                axg = math.exp(ia - slope * diff)
                # Poisson NLL: lambda - k * log(lambda)
                nll += hxg - m.home_goals * math.log(max(hxg, _EPS))
                nll += axg - m.away_goals * math.log(max(axg, _EPS))

            if best is None or nll < best[0]:
                best = (nll, ih, ia, slope)

        assert best is not None
        _, ih, ia, best_slope = best

        # Compute neutral rate from neutral matches in the training set
        neutral_rows = [m for m in rows if m.neutral]
        if neutral_rows:
            neutral_rate = sum(m.home_goals + m.away_goals for m in neutral_rows) / (2 * len(neutral_rows))
        else:
            neutral_rate = (math.exp(ih) + math.exp(ia)) / 2

        # Compute NLL for all slopes for reporting
        all_nll = []
        for slope in slope_list:
            nll = 0.0
            for m in rows:
                he, ae = elo_by_match[(m.match_date.isoformat(), m.home_team, m.away_team)]
                diff = he - ae
                hxg = math.exp(ih + slope * diff)
                axg = math.exp(ia - slope * diff)
                nll += hxg - m.home_goals * math.log(max(hxg, _EPS))
                nll += axg - m.away_goals * math.log(max(axg, _EPS))
            all_nll.append(round(nll, 4))

        return cls(
            intercept_home=ih,
            intercept_away=ia,
            elo_slope=best_slope,
            neutral_rate=neutral_rate,
            data_cutoff=max(m.match_date for m in rows).isoformat(),
            slope_grid_tested=tuple(slope_list),
            slope_nll_scores=tuple(all_nll),
        )

    def predict(
        self,
        *,
        home_elo: float = 1500,
        away_elo: float = 1500,
        neutral: bool = False,
        **_: object,
    ) -> dict:
        diff = home_elo - away_elo
        if neutral:
            # For neutral matches: remove home advantage but keep Elo effect
            # Base rate is the neutral_rate, split equally on average
            base = math.log(max(self.neutral_rate / 2, _EPS))
            hxg = math.exp(base + self.elo_slope * diff)
            axg = math.exp(base - self.elo_slope * diff)
        else:
            hxg = math.exp(self.intercept_home + self.elo_slope * diff)
            axg = math.exp(self.intercept_away - self.elo_slope * diff)
        return summarize_prediction(
            hxg, axg, data_cutoff=self.data_cutoff,
            model_version=f"{MODEL_VERSION}-elo",
        )


# ---------------------------------------------------------------------------
# Regularized team attack/defense Poisson
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegularizedTeamPoissonModel:
    """Team-level attack/defense Poisson with shrinkage.

    log(lambda_home) = log(global_rate) + home_advantage + attack_home - defense_away
    log(lambda_away) = log(global_rate) + attack_away - defense_home

    Shrinkage pulls team effects toward zero (population mean).
    Higher shrinkage = more regularization = stronger pull to mean.

    Identifiability: mean(attacks) = 0, mean(defenses) = 0 after each iteration.
    """

    global_rate: float
    home_advantage: float
    attacks: dict[int, float]
    defenses: dict[int, float]
    counts: dict[int, int]
    shrinkage: float
    data_cutoff: str | None = None
    iterations_run: int = 0
    converged: bool = False

    @classmethod
    def fit(
        cls,
        matches: Iterable[GoalMatch],
        *,
        shrinkage: float = 20.0,
        iterations: int = 50,
        tolerance: float = 1e-4,
        max_goals_cap: float = 6.0,
    ) -> "RegularizedTeamPoissonModel":
        """Fit the regularized team Poisson model.

        Args:
            matches: training matches
            shrinkage: regularization strength (higher = more shrinkage)
            iterations: maximum EM iterations
            tolerance: convergence tolerance on max parameter change
            max_goals_cap: defensive upper bound on expected goals

        Returns:
            Fitted RegularizedTeamPoissonModel.
        """
        rows = list(matches)
        if not rows:
            raise ValueError("cannot fit on an empty match set")

        n = len(rows)
        total_goals = sum(m.home_goals + m.away_goals for m in rows)
        global_rate = max(total_goals / (2 * n), 1e-6)

        # Home advantage from non-neutral matches
        nn = [m for m in rows if not m.neutral]
        if nn:
            home_adv = math.log(
                max(sum(m.home_goals for m in nn) / len(nn), 1e-6) / global_rate
            )
        else:
            home_adv = 0.0

        # Initialize team parameters
        teams = sorted({m.home_team_id for m in rows} | {m.away_team_id for m in rows})
        attacks = {t: 0.0 for t in teams}
        defenses = {t: 0.0 for t in teams}
        counts = {t: 0 for t in teams}
        for m in rows:
            counts[m.home_team_id] += 1
            counts[m.away_team_id] += 1

        converged = False
        iterations_run = 0

        for it in range(iterations):
            iterations_run = it + 1

            # Accumulate scored/conceded vs expected
            scored = {t: 0.0 for t in teams}
            exp_scored = {t: 0.0 for t in teams}
            conceded = {t: 0.0 for t in teams}
            exp_conceded = {t: 0.0 for t in teams}

            for m in rows:
                ha = 0.0 if m.neutral else home_adv
                eh = global_rate * math.exp(ha + attacks[m.home_team_id] - defenses[m.away_team_id])
                ea = global_rate * math.exp(attacks[m.away_team_id] - defenses[m.home_team_id])
                scored[m.home_team_id] += m.home_goals
                exp_scored[m.home_team_id] += eh
                scored[m.away_team_id] += m.away_goals
                exp_scored[m.away_team_id] += ea
                conceded[m.home_team_id] += m.away_goals
                exp_conceded[m.home_team_id] += ea
                conceded[m.away_team_id] += m.home_goals
                exp_conceded[m.home_team_id] += eh

            # Update with shrinkage
            max_change = 0.0
            new_attacks = {}
            new_defenses = {}
            for t in teams:
                # Shrinkage: add pseudo-counts toward global mean (0 in log space)
                raw_a = math.log(
                    (scored[t] + shrinkage * global_rate) /
                    (exp_scored[t] + shrinkage * global_rate)
                )
                raw_d = -math.log(
                    (conceded[t] + shrinkage * global_rate) /
                    (exp_conceded[t] + shrinkage * global_rate)
                )
                # Damped update for stability
                new_a = 0.5 * attacks[t] + 0.5 * raw_a
                new_d = 0.5 * defenses[t] + 0.5 * raw_d
                max_change = max(max_change, abs(new_a - attacks[t]), abs(new_d - defenses[t]))
                new_attacks[t] = new_a
                new_defenses[t] = new_d

            attacks = new_attacks
            defenses = new_defenses

            # Enforce identifiability: mean = 0
            mean_a = float(np.mean(list(attacks.values())))
            mean_d = float(np.mean(list(defenses.values())))
            attacks = {t: v - mean_a for t, v in attacks.items()}
            defenses = {t: v - mean_d for t, v in defenses.items()}

            if max_change < tolerance:
                converged = True
                break

        return cls(
            global_rate=global_rate,
            home_advantage=home_adv,
            attacks=attacks,
            defenses=defenses,
            counts=counts,
            shrinkage=shrinkage,
            data_cutoff=max(m.match_date for m in rows).isoformat(),
            iterations_run=iterations_run,
            converged=converged,
        )

    def predict(
        self,
        *,
        home_team_id: int,
        away_team_id: int,
        neutral: bool = False,
        **_: object,
    ) -> dict:
        flags = []
        for label, tid in (("home", home_team_id), ("away", away_team_id)):
            n = self.counts.get(tid, 0)
            if n == 0:
                flags.append(f"{label}_unseen")
            elif n < 10:
                flags.append(f"{label}_low_history")

        ha = 0.0 if neutral else self.home_advantage
        hxg = self.global_rate * math.exp(
            ha + self.attacks.get(home_team_id, 0.0) - self.defenses.get(away_team_id, 0.0)
        )
        axg = self.global_rate * math.exp(
            self.attacks.get(away_team_id, 0.0) - self.defenses.get(home_team_id, 0.0)
        )
        # Defensive safeguard only — wide bounds
        hxg = min(max(hxg, 0.01), 6.0)
        axg = min(max(axg, 0.01), 6.0)
        return summarize_prediction(
            hxg, axg, low_data_flags=flags, data_cutoff=self.data_cutoff,
            model_version=f"{MODEL_VERSION}-regularized-team",
        )


# ---------------------------------------------------------------------------
# Dixon–Coles low-score correction
# ---------------------------------------------------------------------------


def dixon_coles_correction(
    matrix: np.ndarray,
    home_xg: float,
    away_xg: float,
    rho: float,
    max_goals: int = _DEFAULT_MAX_GOALS,
) -> np.ndarray:
    """Apply Dixon–Coles low-score correction to a scoreline matrix.

    Adjusts only the four lowest-score cells:
        (0,0), (0,1), (1,0), (1,1)

    The correction factor for cell (i, j) is:
        tau(i, j) = 1 - rho * lambda_home * lambda_away    if (i,j) == (0,0)
        tau(i, j) = 1 + rho * lambda_home                   if (i,j) == (0,1)
        tau(i, j) = 1 + rho * lambda_away                   if (i,j) == (1,0)
        tau(i, j) = 1 - rho                                  if (i,j) == (1,1)

    After adjustment, negative probabilities are clipped to 0 and the
    matrix is renormalized to sum to 1.

    Reference: Dixon & Coles (1997), "Modelling Association Football Scores
    and Inefficiencies in the Football Betting Market".

    Args:
        matrix: raw (unnormalized) scoreline probability matrix
        home_xg: home expected goals
        away_xg: away expected goals
        rho: correlation parameter (typically small and negative)
        max_goals: size of the matrix dimension

    Returns:
        Corrected and renormalized matrix.
    """
    if not (-0.25 <= rho <= 0.25):
        raise ValueError(f"rho={rho} outside reasonable range [-0.25, 0.25]")

    corrected = matrix.copy()

    # Dixon–Coles tau function for low-score cells
    lam_h = home_xg
    lam_a = away_xg

    tau = {}
    tau[(0, 0)] = 1.0 - rho * lam_h * lam_a
    tau[(0, 1)] = 1.0 + rho * lam_h
    tau[(1, 0)] = 1.0 + rho * lam_a
    tau[(1, 1)] = 1.0 - rho

    for (i, j), factor in tau.items():
        if i <= max_goals and j <= max_goals:
            corrected[i, j] *= factor

    # Clip negatives
    corrected = np.maximum(corrected, 0.0)

    # Renormalize
    total = float(corrected.sum())
    if total <= 0:
        raise ValueError("Dixon–Coles correction produced zero mass")
    return corrected / total


# ---------------------------------------------------------------------------
# Recency weighting
# ---------------------------------------------------------------------------


def recency_weights(
    matches: list[GoalMatch],
    cutoff_date: date,
    half_life_days: float | None = None,
) -> np.ndarray:
    """Compute exponential decay weights based on match age.

    Args:
        matches: list of matches
        cutoff_date: the prediction date (weights are relative to this)
        half_life_days: half-life in days.  None = uniform weights.

    Returns:
        Array of weights, same length as matches.
    """
    if half_life_days is None or half_life_days <= 0:
        return np.ones(len(matches), dtype=float)

    decay = math.log(2) / half_life_days
    weights = np.array([
        math.exp(-decay * (cutoff_date - m.match_date).days)
        for m in matches
    ], dtype=float)
    return weights


# ---------------------------------------------------------------------------
# Match-importance weights
# ---------------------------------------------------------------------------


def importance_weights(
    matches: list[GoalMatch],
    weight_map: Mapping[str, float],
) -> np.ndarray:
    """Compute per-match importance weights from tournament class.

    Args:
        matches: list of matches
        weight_map: mapping from tournament class to weight

    Returns:
        Array of weights, same length as matches.
    """
    return np.array([
        weight_map.get(classify_tournament(m.tournament), 1.0)
        for m in matches
    ], dtype=float)
