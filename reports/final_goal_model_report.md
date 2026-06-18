# Final Goal Model Report: Independent International-Football Goal Model

**Branch:** `feat/independent-goal-model`
**Latest commit:** CHECKPOINT_H
**PR:** #10 (draft)
**Data cutoff:** 2026-06-15

---

## Executive Summary

The independent goal model (regularized team attack/defense Poisson, shrinkage=5)
was evaluated against the existing Pi/Elo system across multiple holdout periods,
with controlled priors, tournament stage effects, transparent blending, and
robustness checks.

**Final recommendation:** RECOMMENDATION_PLACEHOLDER

---

## 1. First-Half Results (Phases 1-5, Completed)

| Parameter | Value | Evidence |
|-----------|-------|----------|
| Model | Regularized team Poisson | Beats global Poisson by 0.02-0.12 log loss |
| Shrinkage | **5** | Best in grid (0.9373 vs 0.9547 baseline) |
| Recency | None | No improvement at any half-life |
| Importance | Mild (optional) | Marginal +0.001 improvement |
| Dixon-Coles | None | No improvement at any rho |

**Best first-half Log Loss: 0.9373** (shrinkage=5, 2023+ holdout, N=3,613)
**Baseline Log Loss: 0.9547** (shrinkage=20)
**Improvement: 0.0174 log loss (1.8% relative)**

---

## 2. Phase 6: Prior Source Inventory

### Sources Found

| Source | Path | Historical? | Pre-match Safe? | Backtestable? |
|--------|------|-------------|-----------------|---------------|
| FIFA ranking snapshot | data/manual/fifa_ranking_snapshot.csv | ❌ Single date (2026-05-22) | ❌ | ❌ |
| Squad-strength snapshot | data/manual/squad_strength_snapshot.csv | ❌ Single date (2026-06-17) | ❌ | ❌ |
| Team context notes | data/manual/team_context_notes.csv | ❌ Free-text, 2026 only | ❌ | ❌ |
| Elo ratings | data/raw/elo_ratings.json | ✅ 1930-2026, 247 teams | ✅ | ✅ |

### FIFA Ranking Prior
- **Status:** REJECTED for historical backtests
- **Reason:** Single snapshot (2026-05-22). No historical snapshots exist.
- **Production use:** Optional context feature with `FifaRankingPrior` interface.
- 48 of 59 FIFA codes map to corpus team IDs (11 unmapped: ITA, DEN, UKR, WAL, CHI, JAM, SRB, PER, POL, NGA, CMR).

### Squad-Strength Prior
- **Status:** REJECTED for historical backtests
- **Reason:** Single snapshot (2026-06-17). No historical snapshots exist.
- **Production use:** Optional context feature with `SquadStrengthPrior` interface.

### Elo Ratings
- **Status:** Already integrated via `EloPoissonModel` and `elo_at()` API.
- **Leak-safe:** Strict less-than date filtering ensures no future leakage.
- **Coverage:** 247 teams, 1930-2026, ~30-100 snapshots per team.

### Absence/Rotation Notes
- **Status:** Production-only context flags only.
- **Implementation:** `load_context_flags()` produces `ContextFlags` with boolean warnings.
- **Not used:** No numerical adjustments from free-text notes.

---

## 3. Phase 7: Tournament Stage

### Enrichment Coverage

| World Cup | Matches in Raw | Enriched | Join Coverage |
|-----------|---------------|----------|---------------|
| 2014 | 64 | ~64 | ~100% |
| 2018 | 64 | ~64 | ~100% |
| 2022 | 64 | ~64 | ~100% |

### Stage Effect Results (2022 WC)

| Config | Log Loss | RPS | N |
|--------|----------|-----|---|
| no_stage (baseline) | BASELINE_LL | BASELINE_RPS | N_2022 |
| group_adj=+0.05 | GROUP_05_LL | GROUP_05_RPS | N_2022 |
| knockout_adj=+0.05 | KO_05_LL | KO_05_RPS | N_2022 |
| final_group_adj=+0.05 | FG_05_LL | FG_05_RPS | N_2022 |

### Stage Effect Verdict
STAGE_VERDICT

---

## 4. Phase 8: Direct Model Comparison

All models evaluated on identical common sample (2023+ holdout).

### Overall Metrics

| Model | Log Loss | RPS | Brier | Top-1 | N |
|-------|----------|-----|-------|-------|---|
| Pi-only | PI_LL | PI_RPS | PI_BRIER | PI_TOP1 | N_COMMON |
| Elo-only | ELO_LL | ELO_RPS | ELO_BRIER | ELO_TOP1 | N_COMMON |
| Current blend | BLEND_LL | BLEND_RPS | BLEND_BRIER | BLEND_TOP1 | N_COMMON |
| Goal model | GOAL_LL | GOAL_RPS | GOAL_BRIER | GOAL_TOP1 | N_COMMON |

### Goal Model Additional Metrics

| Metric | Value |
|--------|-------|
| Scoreline NLL | GOAL_SCNLL |
| MAE Home Goals | GOAL_MAEH |
| MAE Away Goals | GOAL_MAEA |
| MAE Total Goals | GOAL_MAET |
| Exact Score Acc | GOAL_EXACT |

### Subgroup Analysis

COMPARISON_SUBGROUP_TABLE

### Interpretation
COMPARISON_INTERPRETATION

---

## 5. Phase 9: Transparent Blending

### Blend Grid Results (2023+ holdout)

| Blend | w_current | w_goal | Log Loss | RPS | Top-1 |
|-------|-----------|--------|----------|-----|-------|
| baseline (current) | 1.0 | 0.0 | BLEND_BASE_LL | BLEND_BASE_RPS | BLEND_BASE_TOP1 |
| blend_90_10 | 0.9 | 0.1 | BLEND_90_10_LL | BLEND_90_10_RPS | BLEND_90_10_TOP1 |
| blend_80_20 | 0.8 | 0.2 | BLEND_80_20_LL | BLEND_80_20_RPS | BLEND_80_20_TOP1 |
| blend_70_30 | 0.7 | 0.3 | BLEND_70_30_LL | BLEND_70_30_RPS | BLEND_70_30_TOP1 |
| blend_60_40 | 0.6 | 0.4 | BLEND_60_40_LL | BLEND_60_40_RPS | BLEND_60_40_TOP1 |
| blend_50_50 | 0.5 | 0.5 | BLEND_50_50_LL | BLEND_50_50_RPS | BLEND_50_50_TOP1 |

### Three-Way Blend Results

| Blend | w_pi | w_elo | w_goal | Log Loss | RPS |
|-------|------|-------|--------|----------|-----|
| 3way_40_40_20 | 0.40 | 0.40 | 0.20 | 3WAY_40_40_20_LL | 3WAY_40_40_20_RPS |
| 3way_35_35_30 | 0.35 | 0.35 | 0.30 | 3WAY_35_35_30_LL | 3WAY_35_35_30_RPS |
| 3way_30_30_40 | 0.30 | 0.30 | 0.40 | 3WAY_30_30_40_LL | 3WAY_30_30_40_RPS |
| 3way_25_25_50 | 0.25 | 0.25 | 0.50 | 3WAY_25_25_50_LL | 3WAY_25_25_50_RPS |

### Best Blend
BEST_BLEND_RESULT

### Confirmation/Disagreement Analysis
CONFIRMATION_RESULT

---

## 6. Phase 10: Robustness and Calibration

### Sensitivity to Shrinkage

| Shrinkage | Log Loss | RPS | Top-1 |
|-----------|----------|-----|-------|
| 3 | SENS_S3_LL | SENS_S3_RPS | SENS_S3_TOP1 |
| 5 (selected) | SENS_S5_LL | SENS_S5_RPS | SENS_S5_TOP1 |
| 8 | SENS_S8_LL | SENS_S8_RPS | SENS_S8_TOP1 |
| 10 | SENS_S10_LL | SENS_S10_RPS | SENS_S10_TOP1 |

### Calibration Table (2023+ holdout)

CALIBRATION_TABLE

### Reliability by Confidence

| Bucket | Count | Avg Top Prob | Top-1 Acc |
|--------|-------|-------------|-----------|
| below_0.40 | CAL_LOW_N | CAL_LOW_PROB | CAL_LOW_ACC |
| 0.40-0.50 | CAL_40_N | CAL_40_PROB | CAL_40_ACC |
| 0.50-0.60 | CAL_50_N | CAL_50_PROB | CAL_50_ACC |
| 0.60-0.70 | CAL_60_N | CAL_60_PROB | CAL_60_ACC |
| above_0.70 | CAL_70_N | CAL_70_PROB | CAL_70_ACC |

### Bootstrap Uncertainty (2023+ holdout, 50 samples)

| Metric | Value |
|--------|-------|
| Mean Log Loss | BOOT_MEAN |
| Std Log Loss | BOOT_STD |
| 95% CI | [BOOT_CI_LOW, BOOT_CI_HIGH] |
| Median | BOOT_MEDIAN |

---

## 7. Phase 11: Production Module

### Artifact Schema

Stored at: `data/artifacts/goal_model_sh5.json`

| Field | Value |
|-------|-------|
| artifact_version | goal-model-artifact-v1 |
| model_version | goal-model-research-v0.2 |
| data_cutoff | 2026-06-15 |
| training_row_count | 32,302 |
| shrinkage | 5.0 |
| global_rate | 1.3802 |
| home_advantage | 0.2036 |
| teams | 326 |
| converged | False (50 iters) |

### Production API Summary

Module: `soccer_ev_model/goal_model_production.py`

```python
# Load from artifact file
predictor = GoalModelPredictor.from_artifact_file(
    "data/artifacts/goal_model_sh5.json",
    fifa_prior_path="data/manual/fifa_ranking_snapshot.csv",  # optional
    squad_prior_path="data/manual/squad_strength_snapshot.csv",  # optional
)

# Predict
pred = predictor.predict(
    home_team_id=710061511,  # Argentina
    away_team_id=4213855446,  # Australia
    match_date="2026-06-20",
    neutral=True,
    stage_context=stage_ctx,  # optional
    home_fifa_points=1867,  # optional
    away_fifa_points=1650,  # optional
)

# Returns: GoalPrediction with home_xg, away_xg, hda_probs,
# most_likely_score, expected_total_goals, low_data_flags, confidence
```

### Build Script

```bash
python3 scripts/build_goal_model_artifact.py \
  --output data/artifacts/goal_model_sh5.json \
  --shrinkage 5
```

Deterministic, no network calls, validates output, prints summary.

---

## 8. Phase 12: Final Decision

### Selected Configuration

| Parameter | Value |
|-----------|-------|
| Model | Regularized team Poisson |
| Shrinkage | 5 |
| Priors | None (no historical snapshots available) |
| Stage effects | STAGE_SELECTED |
| Blending | BLEND_SELECTED |

### Rejected Variants

| Variant | Reason |
|---------|--------|
| FIFA ranking prior | Single snapshot — not leak-safe for history |
| Squad-strength prior | Single snapshot — not leak-safe for history |
| Dixon-Coles | No improvement at any rho (first half) |
| Recency weighting | No improvement at any half-life (first half) |
| High shrinkage (40+) | Degrades performance (first half) |
| REJECTED_STAGE | STAGE_REJECT_REASON |
| REJECTED_BLEND | BLEND_REJECT_REASON |

### Final Recommendation

**RECOMMENDATION**

Justification:
- AGGREGATE_PERFORMANCE
- SUBGROUP_PERFORMANCE
- ROBUSTNESS_ASSESSMENT
- OPERATIONAL_COMPLEXITY
- DATA_AVAILABILITY

### Test Results

| Suite | Tests | Status | Runtime |
|-------|-------|--------|---------|
| test_goal_model.py | 52 | ✅ PASS | <1s |
| test_backtest.py | 11 | ✅ PASS | <1s |
| test_second_half.py | 63 | ✅ PASS | <1s |
| **Total** | **126** | **✅ PASS** | **~1.4s** |

### Files Added/Modified

| File | Description |
|------|-------------|
| soccer_ev_model/goal_model_priors.py | Prior loading, transforms, context flags |
| soccer_ev_model/goal_model_stage.py | Stage enrichment, join, context |
| soccer_ev_model/goal_model_enhanced.py | Prior + stage injection |
| soccer_ev_model/goal_model_comparison.py | Multi-model comparison, blending |
| soccer_ev_model/goal_model_production.py | Production API, artifact I/O |
| scripts/build_goal_model_artifact.py | Deterministic artifact builder |
| scripts/run_second_half_experiments.py | Master experiment runner |
| tests/test_second_half.py | 63 new tests |
| data/artifacts/goal_model_sh5.json | Fitted artifact (shrinkage=5) |
| reports/second_half_results.json | Machine-readable experiment results |
| reports/final_goal_model_report.md | This report |

### Files Retained from First Half

| File | Reason |
|------|--------|
| reports/goal_model_data_audit.json/.md | Data audit evidence |
| reports/backtest_results.json/.md | Backtest evidence |
| reports/first_half_report.md | First-half checkpoint |

### Files Cleaned Up

| File | Reason |
|------|--------|
| reports/experiment_log.txt | Superseded by second_half_results.json |
| reports/focused_experiment_log.txt | Superseded by second_half_results.json |

### Confirmation of Boundaries

- ✅ No dashboard/UI files changed
- ✅ No CSS changed
- ✅ No session state changed
- ✅ No Caddy config changed
- ✅ No systemd config changed
- ✅ No firewall rules changed
- ✅ No cron config changed
- ✅ No Hermes config changed
- ✅ /root/WC production checkout untouched
- ✅ PR #10 remains draft
- ✅ Nothing merged or deployed
- ✅ PR #9 untouched
