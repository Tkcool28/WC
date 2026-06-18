"""Enhanced goal model with prior injection and stage effects.

Extends the base RegularizedTeamPoissonModel with:
  - FIFA ranking prior (small additive shift to attack/defense)
  - Squad-strength prior (small additive shift)
  - Tournament stage effects (group/knockout intercept adjustments)

All extensions are optional and default to zero effect.
Every coefficient is transparent and inspectable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np

from .goal_model import (
    RegularizedTeamPoissonModel,
    summarize_prediction,
    MODEL_VERSION,
    _EPS,
    _DEFAULT_MAX_GOALS,
)
from .goal_model_priors import (
    FifaRankingPrior,
    SquadStrengthPrior,
    fifa_points_to_attack_shift,
    squad_value_to_attack_shift,
)
from .goal_model_stage import StageContext

ENHANCED_MODEL_VERSION = "goal-model-enhanced-v0.1"


@dataclass(frozen=True)
class PriorConfig:
    """Configuration for prior injection."""
    weight: float = 0.0          # 0.0 = no prior effect
    fallback_when_missing: bool = True  # if True, skip prior when data missing


@dataclass(frozen=True)
class StageEffectConfig:
    """Configuration for tournament stage effects.

    Each parameter is an additive adjustment to log(lambda).
    Positive = more goals; negative = fewer goals.
    """
    group_stage_adj: float = 0.0
    knockout_adj: float = 0.0
    final_group_adj: float = 0.0


@dataclass(frozen=True)
class EnhancedGoalModelConfig:
    """Full configuration for the enhanced goal model."""
    shrinkage: float = 5.0
    max_iterations: int = 50
    tolerance: float = 1e-4
    max_goals_cap: float = 6.0
    importance_weight_key: str = "mild"  # "none", "mild", "strong"
    fifa_prior: PriorConfig = field(default_factory=PriorConfig)
    squad_prior: PriorConfig = field(default_factory=PriorConfig)
    stage_effects: StageEffectConfig = field(default_factory=StageEffectConfig)
    # Team identity bridge: 3-letter code -> corpus_id
    team_code_to_id: dict = field(default_factory=dict)
    # Reverse: corpus_id -> 3-letter code
    team_id_to_code: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EnhancedTeamPoissonModel:
    """Regularized team Poisson with optional prior and stage effects.

    log(lambda_home) = log(global_rate) + home_advantage
                       + attack_home - defense_away
                       + fifa_shift + squad_shift
                       + stage_adj

    log(lambda_away) = log(global_rate)
                       + attack_away - defense_home
                       - fifa_shift - squad_shift
                       + stage_adj

    The fifa_shift and squad_shift are symmetric: positive for home,
    negative for away.  Stage adjustments are the same for both teams.
    """

    base_model: RegularizedTeamPoissonModel
    config: EnhancedGoalModelConfig
    data_cutoff: str

    def predict(
        self,
        *,
        home_team_id: int,
        away_team_id: int,
        neutral: bool = False,
        stage_context: Optional[StageContext] = None,
        home_fifa_points: Optional[float] = None,
        away_fifa_points: Optional[float] = None,
        home_squad_value: Optional[float] = None,
        away_squad_value: Optional[float] = None,
        **_: object,
    ) -> dict:
        """Predict with optional prior and stage adjustments."""
        flags = list(self.base_model.predict(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            neutral=neutral,
        ).get("low_data_flags", []))

        ha = 0.0 if neutral else self.base_model.home_advantage
        hxg = self.base_model.global_rate * math.exp(
            ha
            + self.base_model.attacks.get(home_team_id, 0.0)
            - self.base_model.defenses.get(away_team_id, 0.0)
        )
        axg = self.base_model.global_rate * math.exp(
            self.base_model.attacks.get(away_team_id, 0.0)
            - self.base_model.defenses.get(home_team_id, 0.0)
        )

        # FIFA ranking prior shift
        fifa_shift = 0.0
        if self.config.fifa_prior.weight > 0.0:
            if home_fifa_points is not None and away_fifa_points is not None:
                fifa_shift = fifa_points_to_attack_shift(
                    home_fifa_points, away_fifa_points, self.config.fifa_prior.weight
                )
            elif not self.config.fifa_prior.fallback_when_missing:
                flags.append("fifa_prior_missing")
            hxg *= math.exp(fifa_shift)
            axg *= math.exp(-fifa_shift)

        # Squad-strength prior shift
        squad_shift = 0.0
        if self.config.squad_prior.weight > 0.0:
            if home_squad_value is not None and away_squad_value is not None:
                squad_shift = squad_value_to_attack_shift(
                    home_squad_value, away_squad_value, self.config.squad_prior.weight
                )
            elif not self.config.squad_prior.fallback_when_missing:
                flags.append("squad_prior_missing")
            hxg *= math.exp(squad_shift)
            axg *= math.exp(-squad_shift)

        # Stage effects
        stage_adj = 0.0
        if stage_context is not None:
            se = self.config.stage_effects
            if stage_context.is_group_stage:
                stage_adj += se.group_stage_adj
                if stage_context.is_final_group:
                    stage_adj += se.final_group_adj
            if stage_context.is_knockout:
                stage_adj += se.knockout_adj
            if stage_adj != 0.0:
                hxg *= math.exp(stage_adj)
                axg *= math.exp(stage_adj)

        # Defensive bounds
        hxg = min(max(hxg, 0.01), self.config.max_goals_cap)
        axg = min(max(axg, 0.01), self.config.max_goals_cap)

        return summarize_prediction(
            hxg, axg,
            low_data_flags=flags,
            data_cutoff=self.data_cutoff,
            model_version=f"{ENHANCED_MODEL_VERSION}-sh{self.config.shrinkage}",
        )


def fit_enhanced_model(
    matches,
    config: EnhancedGoalModelConfig,
    data_cutoff: str | None = None,
) -> EnhancedTeamPoissonModel:
    """Fit the base regularized team model and wrap with enhanced config.

    Priors are NOT fitted here — they are applied at prediction time
    using externally-provided values (FIFA points, squad values).
    """
    base = RegularizedTeamPoissonModel.fit(
        matches,
        shrinkage=config.shrinkage,
        iterations=config.max_iterations,
        tolerance=config.tolerance,
        max_goals_cap=config.max_goals_cap,
    )
    return EnhancedTeamPoissonModel(
        base_model=base,
        config=config,
        data_cutoff=data_cutoff or base.data_cutoff or "",
    )
