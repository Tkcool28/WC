"""Independent, interpretable international-football goal models.

The models in this file use historical scorelines directly.  They do not derive
expected goals from the existing Pi/Elo 1X2 blend.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Iterable, Mapping

import numpy as np

from .goal_model_data import GoalMatch

MODEL_VERSION = "goal-model-research-v0.1"


def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


def scoreline_matrix(home_xg: float, away_xg: float, max_goals: int = 10) -> np.ndarray:
    if not (math.isfinite(home_xg) and math.isfinite(away_xg)):
        raise ValueError("expected goals must be finite")
    if home_xg < 0 or away_xg < 0:
        raise ValueError("expected goals must be nonnegative")
    hp = np.array([_poisson_pmf(i, home_xg) for i in range(max_goals + 1)], dtype=float)
    ap = np.array([_poisson_pmf(i, away_xg) for i in range(max_goals + 1)], dtype=float)
    matrix = np.outer(hp, ap)
    total = float(matrix.sum())
    if total <= 0:
        raise ValueError("scoreline matrix has zero mass")
    return matrix / total


def summarize_prediction(home_xg: float, away_xg: float, *, max_goals: int = 10,
                         low_data_flags: list[str] | None = None,
                         data_cutoff: str | None = None,
                         model_version: str = MODEL_VERSION) -> dict:
    matrix = scoreline_matrix(home_xg, away_xg, max_goals=max_goals)
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
        "model_version": model_version,
        "data_cutoff": data_cutoff,
        "low_data_flags": list(low_data_flags or []),
    }


@dataclass(frozen=True)
class GlobalPoissonModel:
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
        home_rate = sum(m.home_goals for m in non_neutral) / max(1, len(non_neutral))
        away_rate = sum(m.away_goals for m in non_neutral) / max(1, len(non_neutral))
        neutral_rate = (
            sum(m.home_goals + m.away_goals for m in neutral) / max(1, 2 * len(neutral))
            if neutral else (home_rate + away_rate) / 2
        )
        return cls(home_rate, away_rate, neutral_rate, max(m.match_date for m in rows).isoformat())

    def predict(self, *, neutral: bool = False, **_: object) -> dict:
        hxg, axg = ((self.neutral_rate, self.neutral_rate) if neutral else (self.home_rate, self.away_rate))
        return summarize_prediction(hxg, axg, data_cutoff=self.data_cutoff,
                                    model_version=f"{MODEL_VERSION}-global")


@dataclass(frozen=True)
class EloPoissonModel:
    intercept_home: float
    intercept_away: float
    elo_slope: float
    neutral_rate: float
    data_cutoff: str | None = None

    @classmethod
    def fit(cls, matches: Iterable[GoalMatch], elo_by_match: Mapping[tuple, tuple[float, float]],
            slope_grid: Iterable[float] = (-0.0015, -0.001, -0.0005, 0.0, 0.0005, 0.001, 0.0015)) -> "EloPoissonModel":
        rows = [m for m in matches if (m.match_date.isoformat(), m.home_team, m.away_team) in elo_by_match]
        if not rows:
            raise ValueError("no matches have Elo inputs")
        best = None
        for slope in slope_grid:
            h_logs, a_logs = [], []
            for m in rows:
                he, ae = elo_by_match[(m.match_date.isoformat(), m.home_team, m.away_team)]
                diff = he - ae
                h_logs.append(math.log(m.home_goals + 0.25) - slope * diff)
                a_logs.append(math.log(m.away_goals + 0.25) + slope * diff)
            ih, ia = float(np.mean(h_logs)), float(np.mean(a_logs))
            nll = 0.0
            for m in rows:
                he, ae = elo_by_match[(m.match_date.isoformat(), m.home_team, m.away_team)]
                diff = he - ae
                hxg = math.exp(ih + slope * diff)
                axg = math.exp(ia - slope * diff)
                nll += hxg - m.home_goals * math.log(max(hxg, 1e-12))
                nll += axg - m.away_goals * math.log(max(axg, 1e-12))
            if best is None or nll < best[0]:
                best = (nll, ih, ia, slope)
        assert best is not None
        _, ih, ia, slope = best
        neutral_rows = [m for m in rows if m.neutral]
        neutral_rate = (sum(m.home_goals + m.away_goals for m in neutral_rows) / max(1, 2 * len(neutral_rows)))
        return cls(ih, ia, slope, neutral_rate, max(m.match_date for m in rows).isoformat())

    def predict(self, *, home_elo: float = 1500, away_elo: float = 1500,
                neutral: bool = False, **_: object) -> dict:
        diff = home_elo - away_elo
        if neutral:
            base = math.log(max(self.neutral_rate, 1e-6))
            hxg = math.exp(base + self.elo_slope * diff)
            axg = math.exp(base - self.elo_slope * diff)
        else:
            hxg = math.exp(self.intercept_home + self.elo_slope * diff)
            axg = math.exp(self.intercept_away - self.elo_slope * diff)
        return summarize_prediction(hxg, axg, data_cutoff=self.data_cutoff,
                                    model_version=f"{MODEL_VERSION}-elo")


@dataclass(frozen=True)
class RegularizedTeamPoissonModel:
    global_rate: float
    home_advantage: float
    attacks: dict[int, float]
    defenses: dict[int, float]
    counts: dict[int, int]
    shrinkage: float
    data_cutoff: str | None = None

    @classmethod
    def fit(cls, matches: Iterable[GoalMatch], *, shrinkage: float = 20.0,
            iterations: int = 30) -> "RegularizedTeamPoissonModel":
        rows = list(matches)
        if not rows:
            raise ValueError("cannot fit on an empty match set")
        total_goals = sum(m.home_goals + m.away_goals for m in rows)
        global_rate = max(total_goals / (2 * len(rows)), 1e-6)
        nn = [m for m in rows if not m.neutral]
        home_advantage = math.log(max(sum(m.home_goals for m in nn) / max(1, len(nn)), 1e-6) / global_rate)
        teams = sorted({m.home_team_id for m in rows} | {m.away_team_id for m in rows})
        attacks = {t: 0.0 for t in teams}
        defenses = {t: 0.0 for t in teams}
        counts = {t: 0 for t in teams}
        for m in rows:
            counts[m.home_team_id] += 1
            counts[m.away_team_id] += 1
        for _ in range(iterations):
            scored = {t: 0.0 for t in teams}; exp_scored = {t: 0.0 for t in teams}
            conceded = {t: 0.0 for t in teams}; exp_conceded = {t: 0.0 for t in teams}
            for m in rows:
                ha = 0.0 if m.neutral else home_advantage
                eh = global_rate * math.exp(ha + attacks[m.home_team_id] - defenses[m.away_team_id])
                ea = global_rate * math.exp(attacks[m.away_team_id] - defenses[m.home_team_id])
                scored[m.home_team_id] += m.home_goals; exp_scored[m.home_team_id] += eh
                scored[m.away_team_id] += m.away_goals; exp_scored[m.away_team_id] += ea
                conceded[m.home_team_id] += m.away_goals; exp_conceded[m.home_team_id] += ea
                conceded[m.away_team_id] += m.home_goals; exp_conceded[m.away_team_id] += eh
            for t in teams:
                raw_a = math.log((scored[t] + shrinkage * global_rate) /
                                 (exp_scored[t] + shrinkage * global_rate))
                raw_d = -math.log((conceded[t] + shrinkage * global_rate) /
                                  (exp_conceded[t] + shrinkage * global_rate))
                attacks[t] = 0.5 * attacks[t] + 0.5 * raw_a
                defenses[t] = 0.5 * defenses[t] + 0.5 * raw_d
            mean_a = float(np.mean(list(attacks.values())))
            mean_d = float(np.mean(list(defenses.values())))
            attacks = {t: v - mean_a for t, v in attacks.items()}
            defenses = {t: v - mean_d for t, v in defenses.items()}
        return cls(global_rate, home_advantage, attacks, defenses, counts, shrinkage,
                   max(m.match_date for m in rows).isoformat())

    def predict(self, *, home_team_id: int, away_team_id: int, neutral: bool = False,
                **_: object) -> dict:
        flags = []
        for label, tid in (("home", home_team_id), ("away", away_team_id)):
            n = self.counts.get(tid, 0)
            if n == 0:
                flags.append(f"{label}_unseen")
            elif n < 10:
                flags.append(f"{label}_low_history")
        ha = 0.0 if neutral else self.home_advantage
        hxg = self.global_rate * math.exp(ha + self.attacks.get(home_team_id, 0.0) - self.defenses.get(away_team_id, 0.0))
        axg = self.global_rate * math.exp(self.attacks.get(away_team_id, 0.0) - self.defenses.get(home_team_id, 0.0))
        hxg = min(max(hxg, 0.05), 6.0); axg = min(max(axg, 0.05), 6.0)
        return summarize_prediction(hxg, axg, low_data_flags=flags, data_cutoff=self.data_cutoff,
                                    model_version=f"{MODEL_VERSION}-regularized-team")
