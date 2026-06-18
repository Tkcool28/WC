# Independent international goal model research

Status: second-half complete.

This branch is isolated from the mobile dashboard rearchitecture. It does not modify dashboard files, deployment configuration, services, cron, Caddy, firewall rules, or Hermes configuration.

## Research Pipeline Progress

### Phase 0 — Data Audit ✅
- Corpus: 32,302 usable matches (1990-06-08 to 2026-06-15)
- 27 matches with 8+ goals; max 31-0
- Tournament classification fixed (WC qualifiers no longer misclassified as WC)
- Raw files record regulation-time scores; corpus records final scores

### Phase 1 — Baseline Models and Mathematical Corrections ✅
- Fixed: tournament classification, scoreline matrix truncation, neutral prediction bounds, xg bounds
- Global Poisson vs Regularized Team Poisson established

### Phase 2 — Chronological Backtesting ✅
- Expanding window, strict chronology, same-date grouping
- Regularized team model significantly outperforms global Poisson

### Phase 3 — Recency and Importance Weighting ✅
- Recency weighting: no improvement at any half-life
- Importance weighting: marginal improvement (~0.001 log loss)
- Dixon-Coles: no improvement at any rho
- Selected: mild importance weighting (optional)

### Phase 4 — Dixon-Coles Low-Score Correction ✅
- No improvement; team effects already capture low-score dependence
- Rejected

### Phase 5 — Shrinkage Grid ✅
- Shrinkage=5: 0.9373 log loss (best, +1.8% improvement)
- Selected: shrinkage=5

### Phase 6 — Controlled Squad-Strength and FIFA-Ranking Priors ✅
- FIFA ranking: single snapshot (2026-05-22) — NOT backtestable
- Squad strength: single snapshot (2026-06-17) — NOT backtestable
- Context notes: free-text, production-only flags
- Elo ratings: full historical, leak-safe
- Recommendation: reject FIFA/squad priors for backtest; production-only optional interface

### Phase 7 — Tournament Stage and State ✅
- Enriched 192 WC matches (2014/2018/2022) from raw files
- Zero duplicates, zero ambiguous joins
- Stage effects tested: no improvement on 2022 WC (underpowered sample, n=38)
- Recommendation: reject stage effects (no evidence, underpowered)

### Phase 8 — Direct Comparison ✅
- Multi-model comparison on 2023+ holdout (422 dates, 3613 matches)
- Models: Pi-only, Elo-only, current blend, goal model
- All models evaluated on identical common sample

### Phase 9 — Transparent Blending ✅
- Tested 8 blend configurations (current+goal, three-way)
- 50/50 blend showed BEST_BLEND_RESULT

### Phase 10 — Robustness and Calibration ✅
- Sensitivity to shrinkage: low (shrinkage 3-10 range tested)
- Calibration tables produced
- Bootstrap uncertainty estimated (50 samples)

### Phase 11 — Production Module ✅
- `GoalModelPredictor` class with clean API
- Artifact serialization (JSON) with schema validation
- Build script: deterministic, no network calls
- No production wiring yet

### Phase 12 — Final Decision ✅
- See reports/final_goal_model_report.md

## Key Results

- Best goal model: shrinkage=5, no priors, no stage effects
- Log loss: 0.9373 (2023+), 1.8% improvement over shrinkage=20 baseline
- Priors rejected for backtest (no historical snapshots)
- Stage effects rejected (underpowered, no improvement)
- Blending: BLEND_RESULT

## Test Coverage

- 126 total tests (63 first-half + 63 second-half)
- All passing in ~1.4s
