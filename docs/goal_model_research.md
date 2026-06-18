# Independent international goal model research

Status: Phase 3 evidence audit complete.

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
- Decision: reject FIFA/squad priors for backtest; production-only optional interface built

### Phase 7 — Tournament Stage and State ✅
- Enriched 192 WC matches (2014/2018/2022) from raw files
- Zero duplicates, zero ambiguous joins
- Stage effects tested: no improvement; underpowered sample
- Decision: reject stage effects

### Phase 8 — Direct Comparison ✅
- 4 models on 3,805-match aggregate (4 blocks: 2014 WC, 2018 WC, 2022 WC, 2023+)
- Pi-only: LL=0.969857, Elo-only: LL=0.927100, Goal-only: LL=0.934044
- Elo60/Goal40 blend: LL=0.918450 (best aggregate)
- See reports/model_comparison.json and reports/blend_grid.json for per-block breakdowns

### Phase 9 — Transparent Blending ✅
- Best blend: Elo60/Goal40 (LL=0.918450 on 3,805 matches)
- Pi30/Elo40/Goal30 tied at LL=0.918450 but with worse draw calibration
- Blend grid shows flat region; Elo60/Goal40 chosen for draw accuracy and simplicity

### Phase 10 — Robustness and Calibration ✅
- Shrinkage 3-10: identical results (4 decimal places)
- Score grid max 4-7: identical results
- Bootstrap 95% CI for log loss: resolved per-block analysis
- Calibration: model well-calibrated on draw (0.228 predicted vs 0.230 actual)

### Phase 11 — Production Module ✅
- `GoalModelPredictor` class with clean API
- Artifact serialization (JSON) with schema validation
- Build script: deterministic, no network calls
- No production wiring yet

### Phase 12 — Final Decision ✅
- **Recommendation: Elo60/Goal40 as shadow-mode candidate**
- Keep current production unchanged initially
- Use goal model for display-only xG, scoreline, totals, disagreement context
- See reports/final_goal_model_report.md

## Key Results

| Model | N | Log Loss | RPS | Brier | Top-1 |
|-------|---|----------|-----|-------|-------|
| Pi-only | 3,805 | 0.969857 | 0.197281 | 0.573289 | 0.5545 |
| Elo-only | 3,805 | 0.927100 | 0.184872 | 0.543168 | 0.5882 |
| Goal-only | 3,805 | 0.934044 | 0.188964 | 0.551193 | 0.5669 |
| **Elo60/Goal40** | **3,805** | **0.918450** | **0.182779** | **0.538440** | **0.5934** |
| Pi30/Elo40/Goal30 | 3,805 | 0.918450 | 0.182752 | 0.539085 | 0.5869 |

### Sample Definitions (Corrected)
- 2014 WC: 64 matches (pure tournament block)
- 2018 WC: 64 matches (pure tournament block)
- 2022 WC: 64 matches (pure tournament block)
- 2023+: 3,613 matches (all internationals since 2023-01-01)
- Aggregate: 3,805 matches

### Phase 1 Corrections
- Elo60, not Pi30, is closer on aggregate draw average
- Pi30 actual H/D/A is the same sample: approx 0.470 / 0.230 / 0.300
- Earlier 14/18/22 date-window counts (87/23) were invalid; pure WC blocks are 64 each

## Test Coverage
- Focused tests: 129 passed (goal model, backtest, second half)
- Full suite: run after merge verification
- No dashboard or production behavior changed