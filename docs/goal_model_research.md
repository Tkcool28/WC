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
- Decision: reject FIFA/squad priors for backtest; production-only optional interface built

### Phase 7 — Tournament Stage and State ✅
- Enriched 192 WC matches (2014/2018/2022) from raw files
- Zero duplicates, zero ambiguous joins
- Stage effects tested: no improvement on 2022 WC (underpowered sample)
- Decision: reject stage effects (no evidence, underpowered)

### Phase 8 — Direct Comparison ✅
- 4 models on 87-match common sample (2022 WC window)
- Pi-only: LL=1.0482, Elo-only: LL=1.0111, Current blend: LL=1.0482, Goal model: LL=1.0197
- Goal model has highest Top1 (0.517); Elo-only has lowest LL (1.0111)

### Phase 9 — Transparent Blending ✅
- Best blend: 40% Pi / 60% Goal = LL=1.0119 (matches Elo-only)
- Blend grid flat between 30-70% goal weight
- Confirmation signal: 87.4% same-top agreement

### Phase 10 — Robustness and Calibration ✅
- Shrinkage 3-10: identical results (4 decimal places)
- Score grid max 4-7: identical results
- Bootstrap 95% CI for log loss: [1.0063, 1.0249]
- Calibration: well-calibrated in 0.2-0.5 range

### Phase 11 — Production Module ✅
- `GoalModelPredictor` class with clean API
- Artifact serialization (JSON) with schema validation
- Build script: deterministic, no network calls
- No production wiring yet

### Phase 12 — Final Decision ✅
- **Recommendation: Blend goal model with current system (40% Pi / 60% Goal)**
- See reports/final_goal_model_report.md

## Key Results

- Best goal model: shrinkage=5, no priors, no stage effects
- Log loss: 1.0197 (2022 WC window, 87 matches)
- Best blend: 40/60 Pi/Goal = 1.0119
- Priors rejected for backtest (no historical snapshots)
- Stage effects rejected (underpowered, no improvement)

## Test Coverage

- 514 total tests passed (63 new second-half tests)
- 1 pre-existing failure from first-half branch (ev_workflow.py)
- Runtime: ~5s full suite
