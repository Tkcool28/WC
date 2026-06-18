"""Production-ready goal model module.

Provides a clean, stable API for loading a fitted goal model artifact
and making predictions.  No dashboard imports.  No side effects.

This module is the intended integration point for the goal model
into the production EV workflow — once the final recommendation is made.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np

from .goal_model import (
    RegularizedTeamPoissonModel,
    summarize_prediction,
    scoreline_matrix,
    MODEL_VERSION,
    _EPS,
    _DEFAULT_MAX_GOALS,
)
from .goal_model_priors import (
    FifaRankingPrior,
    SquadStrengthPrior,
    fifa_points_to_attack_shift,
    squad_value_to_attack_shift,
    ContextFlags,
    load_context_flags,
)
from .goal_model_stage import StageContext, classify_stage_context

ARTIFACT_VERSION = "goal-model-artifact-v1"


# ── Artifact schema ─────────────────────────────────────────────────────────

@dataclass
class GoalModelArtifact:
    """Serializable, inspectable goal model artifact.

    Contains all fitted parameters needed to reproduce predictions.
    """
    # Metadata
    artifact_version: str
    model_version: str
    data_cutoff: str
    training_row_count: int
    excluded_row_count: int
    identity_mapping_version: str
    created_at: str

    # Model hyperparameters
    shrinkage: float
    global_rate: float
    home_advantage: float
    max_goals_cap: float

    # Fitted parameters
    attacks: dict[str, float]      # team_id (as str) -> attack effect
    defenses: dict[str, float]     # team_id (as str) -> defense effect
    counts: dict[str, int]         # team_id (as str) -> match count

    # Prior coefficients (if used)
    fifa_prior_weight: float = 0.0
    squad_prior_weight: float = 0.0

    # Stage coefficients (if used)
    stage_group_adj: float = 0.0
    stage_knockout_adj: float = 0.0
    stage_final_group_adj: float = 0.0

    # Diagnostics
    iterations_run: int = 0
    converged: bool = False

    # Source metadata
    source_files: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "GoalModelArtifact":
        required = [
            "artifact_version", "model_version", "data_cutoff",
            "training_row_count", "shrinkage", "global_rate",
            "home_advantage", "attacks", "defenses", "counts",
        ]
        for k in required:
            if k not in d:
                raise ValueError(f"Missing required field: {k}")
        if d["artifact_version"] != ARTIFACT_VERSION:
            raise ValueError(
                f"Artifact version mismatch: expected {ARTIFACT_VERSION}, "
                f"got {d['artifact_version']}"
            )
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# ── Artifact I/O ────────────────────────────────────────────────────────────

def save_artifact(
    artifact: GoalModelArtifact,
    path: str | Path,
) -> Path:
    """Save artifact to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def load_artifact(path: str | Path) -> GoalModelArtifact:
    """Load and validate artifact from JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return GoalModelArtifact.from_dict(data)


# ── Production predictor ────────────────────────────────────────────────────

@dataclass
class GoalPrediction:
    """Full prediction output from the production goal model."""
    home_xg: float
    away_xg: float
    hda_probs: dict[str, float]
    most_likely_score: list[int]
    expected_total_goals: float
    score_probs: Optional[list] = None  # optional full matrix
    model_version: str = ""
    data_cutoff: str = ""
    low_data_flags: list[str] = field(default_factory=list)
    missing_prior_flags: list[str] = field(default_factory=list)
    stage_context_flags: list[str] = field(default_factory=list)
    context_flags: Optional[dict] = None
    confidence: str = "normal"  # "normal", "low", "very_low"
    disagreement: Optional[dict] = None


class GoalModelPredictor:
    """Production predictor wrapping a loaded artifact.

    Stateless after construction.  Thread-safe for prediction.
    """

    def __init__(
        self,
        artifact: GoalModelArtifact,
        fifa_prior: Optional[FifaRankingPrior] = None,
        squad_prior: Optional[SquadStrengthPrior] = None,
        context_flags: Optional[dict[str, ContextFlags]] = None,
    ):
        self.artifact = artifact
        self.fifa_prior = fifa_prior
        self.squad_prior = squad_prior
        self.context_flags = context_flags or {}

    @classmethod
    def from_artifact_file(
        cls,
        path: str | Path,
        fifa_prior_path: Optional[str | Path] = None,
        squad_prior_path: Optional[str | Path] = None,
        context_notes_path: Optional[str | Path] = None,
    ) -> "GoalModelPredictor":
        """Load predictor from artifact file and optional prior sources."""
        artifact = load_artifact(path)
        fifa = FifaRankingPrior.load(fifa_prior_path) if fifa_prior_path else None
        squad = SquadStrengthPrior.load(squad_prior_path) if squad_prior_path else None
        ctx = load_context_flags(context_notes_path) if context_notes_path else None
        return cls(artifact, fifa, squad, ctx)

    def predict(
        self,
        home_team_id: int,
        away_team_id: int,
        match_date: str | date,
        neutral: bool = False,
        stage_context: Optional[StageContext] = None,
        home_fifa_points: Optional[float] = None,
        away_fifa_points: Optional[float] = None,
        home_squad_value: Optional[float] = None,
        away_squad_value: Optional[float] = None,
        home_team_code: Optional[str] = None,
        away_team_code: Optional[str] = None,
    ) -> GoalPrediction:
        """Make a prediction for a single match.

        Args:
            home_team_id: corpus home team ID
            away_team_id: corpus away team ID
            match_date: match date (for prior lookup)
            neutral: whether match is at neutral venue
            stage_context: optional tournament stage context
            home_fifa_points: optional FIFA ranking points for home team
            away_fifa_points: optional FIFA ranking points for away team
            home_squad_value: optional squad market value for home team
            away_squad_value: optional squad market value for away team
            home_team_code: optional 3-letter code (for prior lookup)
            away_team_code: optional 3-letter code (for prior lookup)

        Returns:
            GoalPrediction with all outputs.
        """
        art = self.artifact
        flags: list[str] = []
        prior_flags: list[str] = []
        stage_flags: list[str] = []

        # Base expected goals
        ha = 0.0 if neutral else art.home_advantage
        hid = str(home_team_id)
        aid = str(away_team_id)

        hxg = art.global_rate * math.exp(
            ha
            + art.attacks.get(hid, 0.0)
            - art.defenses.get(aid, 0.0)
        )
        axg = art.global_rate * math.exp(
            art.attacks.get(aid, 0.0)
            - art.defenses.get(hid, 0.0)
        )

        # Low data flags
        home_count = art.counts.get(hid, 0)
        away_count = art.counts.get(aid, 0)
        if home_count == 0:
            flags.append("home_unseen")
        elif home_count < 10:
            flags.append("home_low_history")
        if away_count == 0:
            flags.append("away_unseen")
        elif away_count < 10:
            flags.append("away_low_history")

        # FIFA ranking prior
        if art.fifa_prior_weight > 0.0:
            if home_fifa_points is None and home_team_code and self.fifa_prior:
                md = match_date if isinstance(match_date, date) else date.fromisoformat(str(match_date)[:10])
                home_fifa_points, _, _ = self.fifa_prior.lookup(home_team_code, md)
            if away_fifa_points is None and away_team_code and self.fifa_prior:
                md = match_date if isinstance(match_date, date) else date.fromisoformat(str(match_date)[:10])
                away_fifa_points, _, _ = self.fifa_prior.lookup(away_team_code, md)

            if home_fifa_points is not None and away_fifa_points is not None:
                shift = fifa_points_to_attack_shift(
                    home_fifa_points, away_fifa_points, art.fifa_prior_weight
                )
                hxg *= math.exp(shift)
                axg *= math.exp(-shift)
            else:
                prior_flags.append("fifa_prior_missing")

        # Squad-strength prior
        if art.squad_prior_weight > 0.0:
            if home_squad_value is None and home_team_code and self.squad_prior:
                md = match_date if isinstance(match_date, date) else date.fromisoformat(str(match_date)[:10])
                home_squad_value, _ = self.squad_prior.lookup(home_team_code, md)
            if away_squad_value is None and away_team_code and self.squad_prior:
                md = match_date if isinstance(match_date, date) else date.fromisoformat(str(match_date)[:10])
                away_squad_value, _ = self.squad_prior.lookup(away_team_code, md)

            if home_squad_value is not None and away_squad_value is not None:
                shift = squad_value_to_attack_shift(
                    home_squad_value, away_squad_value, art.squad_prior_weight
                )
                hxg *= math.exp(shift)
                axg *= math.exp(-shift)
            else:
                prior_flags.append("squad_prior_missing")

        # Stage effects
        if stage_context is not None and stage_context.stage:
            adj = 0.0
            if stage_context.is_group_stage:
                adj += art.stage_group_adj
                if stage_context.is_final_group:
                    adj += art.stage_final_group_adj
            if stage_context.is_knockout:
                adj += art.stage_knockout_adj
            if adj != 0.0:
                hxg *= math.exp(adj)
                axg *= math.exp(adj)
                stage_flags.append(f"stage_adj={adj:.4f}")

        # Bounds
        hxg = min(max(hxg, 0.01), art.max_goals_cap)
        axg = min(max(axg, 0.01), art.max_goals_cap)

        # Build summary
        summary = summarize_prediction(hxg, axg, low_data_flags=flags, data_cutoff=art.data_cutoff)

        # Confidence
        confidence = "normal"
        if home_count < 5 or away_count < 5:
            confidence = "very_low"
        elif home_count < 10 or away_count < 10:
            confidence = "low"

        return GoalPrediction(
            home_xg=summary["home_xg"],
            away_xg=summary["away_xg"],
            hda_probs=summary["hda_probs"],
            most_likely_score=summary["most_likely_score"],
            expected_total_goals=summary["expected_total_goals"],
            model_version=art.model_version,
            data_cutoff=art.data_cutoff,
            low_data_flags=flags,
            missing_prior_flags=prior_flags,
            stage_context_flags=stage_flags,
            confidence=confidence,
        )


# ── Artifact builder ────────────────────────────────────────────────────────

def build_artifact(
    model: RegularizedTeamPoissonModel,
    matches,
    excluded_count: int = 0,
    identity_version: str = "v1",
    source_files: Optional[dict] = None,
    fifa_prior_weight: float = 0.0,
    squad_prior_weight: float = 0.0,
    stage_group_adj: float = 0.0,
    stage_knockout_adj: float = 0.0,
    stage_final_group_adj: float = 0.0,
) -> GoalModelArtifact:
    """Build a GoalModelArtifact from a fitted model.

    Args:
        model: fitted RegularizedTeamPoissonModel
        matches: training matches (for row count)
        excluded_count: number of excluded rows
        identity_version: version of team identity mapping
        source_files: optional dict of source file metadata
        fifa_prior_weight: FIFA prior weight used (0.0 = none)
        squad_prior_weight: squad prior weight used (0.0 = none)
        stage_group_adj: group stage adjustment
        stage_knockout_adj: knockout adjustment
        stage_final_group_adj: final group match adjustment

    Returns:
        GoalModelArtifact ready for serialization.
    """
    from datetime import datetime
    rows = list(matches)
    return GoalModelArtifact(
        artifact_version=ARTIFACT_VERSION,
        model_version=MODEL_VERSION,
        data_cutoff=model.data_cutoff or "",
        training_row_count=len(rows),
        excluded_row_count=excluded_count,
        identity_mapping_version=identity_version,
        created_at=datetime.utcnow().isoformat() + "Z",
        shrinkage=model.shrinkage,
        global_rate=model.global_rate,
        home_advantage=model.home_advantage,
        max_goals_cap=6.0,
        attacks={str(k): v for k, v in model.attacks.items()},
        defenses={str(k): v for k, v in model.defenses.items()},
        counts={str(k): v for k, v in model.counts.items()},
        fifa_prior_weight=fifa_prior_weight,
        squad_prior_weight=squad_prior_weight,
        stage_group_adj=stage_group_adj,
        stage_knockout_adj=stage_knockout_adj,
        stage_final_group_adj=stage_final_group_adj,
        iterations_run=model.iterations_run,
        converged=model.converged,
        source_files=source_files or {},
    )
