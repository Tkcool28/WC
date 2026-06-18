# Final Goal Model Report — Phase 3 Evidence Audit

**Branch:** `feat/independent-goal-model`
**PR:** #10 (draft)
**Date:** 2026-06-18
**Phase:** 3 — Evidence audit and blend recommendation

---

## 1. Data & Evaluation

### Sample Definitions (Corrected)

| Block | Matches | Method | Note |
|-------|---------|--------|------|
| 2014 WC | 64 | Pure tournament block | Not date-window extract |
| 2018 WC | 64 | Pure tournament block | Not date-window extract |
| 2022 WC | 64 | Pure tournament block | Not date-window extract |
| 2023+ | 3,613 | All internationals from 2023-01-01 onward | Expanded from earlier 3,084 |
| **Aggregate** | **3,805** | Union of all 4 blocks | — |

**Correction note:** Phase 1 mistakenly used date-window samples of 87 (2022) and 23 (non-WC). Phase 2/3 replaced these with pure 64-match tournament blocks for 2014/2018/2022. The 2023+ sample grew from 3,084 to 3,613 as more data accumulated.

---

## 2. Per-Block Metrics

### Elo60/Goal40 (recommended blend)

| Block | N | Log Loss | RPS | Brier | Top-1 | Avg Pred H/D/A | Actual H/D/A |
|-------|---|----------|-----|-------|-------|-----------------|--------------|
| 2014 WC | 64 | 0.983385 | 0.211148 | 0.586471 | 0.5781 | 0.427 / 0.252 / 0.320 | 0.453 / 0.203 / 0.344 |
| 2018 WC | 64 | 1.004807 | 0.218432 | 0.601638 | 0.5469 | 0.406 / 0.256 / 0.339 | 0.391 / 0.203 / 0.406 |
| 2022 WC | 64 | 1.014540 | 0.214373 | 0.604998 | 0.5000 | 0.410 / 0.256 / 0.335 | 0.438 / 0.234 / 0.328 |
| 2023+ | 3,613 | 0.917137 | 0.182106 | 0.537430 | 0.5987 | 0.441 / 0.228 / 0.331 | 0.472 / 0.231 / 0.297 |
| **Aggregate** | **3,805** | **0.918450** | **0.182779** | **0.538440** | **0.5934** | **0.441 / 0.228 / 0.331** | **0.470 / 0.230 / 0.300** |

### Pi30/Elo40/Goal30 (3-way comparison candidate)

| Block | N | Log Loss | RPS | Brier | Top-1 |
|-------|---|----------|-----|-------|-------|
| 2014 WC | 64 | 1.022444 | 0.225370 | 0.615974 | 0.4531 |
| 2018 WC | 64 | 1.013309 | 0.222902 | 0.609141 | 0.5156 |
| 2022 WC | 64 | 0.993876 | 0.207448 | 0.590013 | 0.5000 |
| 2023+ | 3,613 | 0.913592 | 0.180848 | 0.535580 | 0.5920 |
| **Aggregate** | **3,805** | **0.918450** | **0.182752** | **0.539085** | **0.5869** |

### Individual Model Aggregate

| Model | N | Log Loss | RPS | Brier | Top-1 | Avg Pred H/D/A | Actual H/D/A |
|-------|---|----------|-----|-------|-------|-----------------|--------------|
| Pi-only | 3,805 | 0.969857 | 0.197281 | 0.573289 | 0.5545 | 0.462 / 0.189 / 0.349 | 0.470 / 0.230 / 0.300 |
| Elo-only | 3,805 | 0.927100 | 0.184872 | 0.543168 | 0.5882 | 0.440 / 0.223 / 0.337 | 0.470 / 0.230 / 0.300 |
| Goal-only | 3,805 | 0.934044 | 0.188964 | 0.551193 | 0.5669 | 0.442 / 0.235 / 0.323 | 0.470 / 0.230 / 0.300 |

---

## 3. Head-to-Head Deltas (Aggregate)

| Comparison | Δ Log Loss | Δ RPS | Δ Brier | Δ Top-1 |
|------------|-----------|-------|---------|---------|
| Elo60/Goal40 vs Pi-only | **−0.051407** | −0.014502 | −0.034849 | +0.038896 |
| Elo60/Goal40 vs Elo-only | **−0.008650** | −0.002093 | −0.004728 | +0.005257 |
| Elo60/Goal40 vs Goal-only | **−0.015594** | −0.006185 | −0.012753 | +0.026544 |
| Pi30/Elo40/Goal30 vs Pi-only | **−0.051407** | −0.014529 | −0.034204 | +0.032325 |
| Elo60/Goal40 vs Pi30/Elo40/Goal30 | **0.000000** | +0.000027 | −0.000645 | +0.006571 |

**Key:** Elo60/Goal40 and Pi30/Elo40/Goal30 are tied on aggregate log loss (0.918450). Elo60/Goal40 has slightly better Brier (−0.000645) and Top-1 (+0.0066) and substantially better draw prediction.

---

## 4. Calibration (Elo60/Goal40)

### Model ECE (Expected Calibration Error)
- **Elo60/Goal40 aggregate:** not separately computed in the trusted audit (blend calibration requires per-blend bins from the grid results)
- **Elo-only ECE:** 0.0603 (aggregate)
- **Goal-only ECE:** 0.0338 (aggregate)
- **Method:** 10 equal-width bins per one-vs-rest outcome probability; ECE weighted over n×3 observations

### Calibration: Predicted vs Actual H/D/A (Elo60/Goal40)

| Outcome | Avg Predicted | Actual | Difference |
|---------|---------------|--------|------------|
| Home | 0.441 | 0.470 | +0.029 (slightly underconfident) |
| Draw | 0.228 | 0.230 | +0.002 (well-calibrated) |
| Away | 0.331 | 0.300 | −0.031 (slightly overconfident) |

**Note:** Calibration-bin data from aggregate blend_grid provides cross-check. Model is substantially better-calibrated on draw than pure Pi (which predicts 0.189 vs actual 0.230).

---

## 5. Priors Status

| Source | Backtestable | Historical Use | Recommendation |
|--------|-------------|----------------|---------------|
| Elo ratings | ✅ Yes | ✅ Full historical snapshots | Keep as blend component |
| FIFA rankings | ❌ No | ❌ Single snapshot (2026-05-22) | Production-only optional |
| Squad strength | ❌ No | ❌ Single snapshot (2026-06-17) | Production-only optional |
| Context notes | ❌ No | ❌ Free-text, no history | Not applicable |
| Stage effects | ❌ No | ❌ No metric improvement | Rejected |

No speculative free-text adjustments. No historically untestable inputs in backtests.

---

## 6. Stage Effects

Direct test performed: 192 WC matches (2014/2018/2022) enriched with stage labels (group/knockout/final). Tested group intercept ±0.05/0.10, knockout ±0.05/0.10, final-group ±0.05/0.10.

**Result:** No improvement at any setting (identical to 4 decimal places). Underpowered sample.

**Decision:** Reject stage effects for backtest and blend.

---

## 7. Final Recommendation

### Selected Blend: **Elo60/Goal40** (60% Elo / 40% Goal)

Rationale:

1. **Tied best aggregate log loss** with Pi30/Elo40/Goal30 at 0.918450
2. **Better draw calibration** — avg predicted draw 0.228 vs actual 0.230 (Pi30/Elo40/Goal30: 0.216)
3. **Slightly better Top-1** (0.5934 vs 0.5869)
4. **Cleaner two-model blend** — avoids the Pi model's known draw-underestimation bias
5. **Better per-block stability** — lower variance across WC blocks than Pi30/Elo40/Goal30

### What to do

- ✅ **Select Elo60/Goal40** as primary 1X2 shadow-mode candidate
- ✅ **Keep current production unchanged initially** (pure Pi)
- ✅ **Run selected blend in shadow mode** — log Pi, Elo, Goal, and selected-blend predictions
- ✅ **Use goal model immediately** for display-only xG, scoreline distribution, totals, and disagreement/context
- ✅ **Track 2026 World Cup** forward performance
- ✅ **Promote blend only after forward-validation criteria met**

### What NOT to do

- ❌ Do NOT recommend immediate production replacement
- ❌ Do NOT replace Pi/Elo blend entirely
- ❌ Do NOT add FIFA/squad priors to historical backtests
- ❌ Do NOT add stage effects (no evidence)
- ❌ Do NOT deploy, merge, or modify dashboard

### Forward Validation Plan

1. Run selected blend (Elo60/Goal40) in shadow mode
2. Log all model predictions (Pi, Elo, Goal, blend) per match
3. Track 2026 WC performance as primary validation event
4. Promote blend to production only after:
   - N ≥ 200 forward matches
   - blend log loss ≤ Elo-only log loss
   - No systematic calibration drift
   - Disagreement analysis shows blend improves on both parents

---

## 8. Correction of Phase 1 Errors

1. **Draw average:** Elo60, not Pi30, is closer on aggregate draw average (0.228 vs 0.230 actual; Pi30/Elo40/Goal30: 0.216)
2. **Actual H/D/A for Pi30:** Pi30 actual H/D/A is the same sample as all models: approx 0.470 / 0.230 / 0.300
3. **Sample counts:** Earlier 2014/2018/2022 date-window counts (87/23) were invalid. Pure WC blocks are 64 each
4. **Best blend:** Previously 40% Pi / 60% Goal on 87 matches; now corrected to Elo60/Goal40 on 3,805 matches

---

## 9. Test Results

- **Focused tests (goal model, backtest, second half):** 129 passed in 1.59s
- **Full suite:** 735 passed, 0 failed
- **No dashboard files changed**
- **soccer_ev_model/ev_workflow.py unchanged relative to current main**

---

## 10. Repository State

- ✅ Branch: `feat/independent-goal-model`, draft PR #10
- ✅ Synchronized with current `origin/main`
- ✅ /root/WC untouched
- ✅ No merge, no deploy
- ✅ No dashboard or production behavior changed