# Soccer EV Model — Phase 4 (Squad-Strength Context Layer) — DESIGN ONLY

> **Status:** Design note. **Do not implement yet.** No code, no CSV files, no
> PR opened in this pass. This document is a planning artifact only.
>
> Phase 4 is **context-only**. It must not change `blend_probs`, must not
> change Elo, must not change Pi, must not change Poisson, and must not
> influence pick recommendations. It exists to give a human (and future
> models) an explainable squad-strength signal alongside the existing model.

**Parent task:** WC dashboard safe deploy + Pi audit + Phase 4 planning.
**Phase 4 goal:** Add a *context-only* squad-strength layer that helps explain
matchups without changing probabilities.

**Non-goals (hard):**
- No Transfermarkt scraping. Data entry is manual from public pages.
- No changes to `evaluate_match`, `blend_probs`, `pi_only_probs`,
  `elo_only_probs`, `calibrated_pi`, `edges`, `plus_ev_flags`.
- No pick-recommendation changes.
- No live deploy of Phase 4 code in this pass.

---

## 1. Why context-only, why now

The dashboard's model probabilities (Pi+Elo blend, calibrated) are the
authoritative signal. Phase 4 is a parallel narrative layer that the
dashboard can render *next to* those probabilities so a human can see
"Argentina has a 2.3B € squad vs Algeria's 95M €" alongside
"P(home) = 0.57".

The current code (post-PR #1) already has 4 context layers that follow
this pattern:
- **Market baseline** — model vs no-vig book.
- **Market divergence label** — qualitative band of disagreement.
- **Poisson secondary view** — independent-Poisson xG sanity check.
- **Group-stage context warnings** — narrative flags about tournament stage.

Phase 4 adds a 5th: **squad-strength context** (squad value + FIFA rank
fallback + manual notes).

---

## 2. Data sources (all manual, no scraping)

### 2.1 Primary: `data/manual/squad_strength_snapshot.csv`

Manually populated from public Transfermarkt-style squad market value pages.
**One row per national team.** Snapshot dated — the file is a point-in-time
view, not a live feed.

| column | type | description |
|---|---|---|
| `team` | str | Display name (matches `team_identity.json` `name` field) |
| `fifa_team_code` | str | 3-letter canonical id (matches `team_identity.json` key, e.g. `ARG`, `USA`) |
| `confederation` | str | `UEFA` / `CONMEBOL` / `CONCACAF` / `CAF` / `AFC` / `OFC` |
| `squad_market_value_eur` | int | Total estimated squad market value, in euros. **Primary metric.** |
| `avg_player_value_eur` | int | Average per-player squad value, in euros |
| `top_5_player_value_eur` | int | Sum of top-5 most valuable players' values, in euros |
| `most_valuable_player` | str | Name of the highest-valued player in the squad |
| `source_name` | str | e.g. `"Transfermarkt (public page)"` |
| `source_url` | str | URL of the public squad/market-value page used |
| `snapshot_date` | str (ISO) | When this row's values were captured (`YYYY-MM-DD`) |
| `notes` | str | Free-form context (e.g. "tournament squad only", "post-window update") |

**Primary metric:** `squad_market_value_eur`. This is the headline number
the dashboard displays.

**Derived display fields (computed at load, not stored):**
- `gap_vs_opponent_pct` — `(team_value / opponent_value - 1) * 100`
- `value_tier` — bucketed label, see §2.4

### 2.2 Fallback: `data/manual/fifa_ranking_snapshot.csv`

Used when a team is missing from the squad-strength file (no public
market-value page, or the user hasn't filled it in yet). FIFA ranking is
a weaker signal but always available for WC participants.

| column | type | description |
|---|---|---|
| `team` | str | Display name |
| `fifa_team_code` | str | 3-letter canonical id |
| `fifa_rank` | int | FIFA/Coca-Cola world ranking position |
| `ranking_points` | int | FIFA ranking points (optional, for finer sorting) |
| `snapshot_date` | str (ISO) | Capture date |
| `source_url` | str | URL of the FIFA ranking page used |
| `notes` | str | Free-form context |

### 2.3 Notes overlay: `data/manual/team_context_notes.csv`

Free-form injuries / absences / rotation / motivation notes. **Not
numeric, not used in the model.** Rendered as a bullet list in the
context panel.

| column | type | description |
|---|---|---|
| `team` | str | Display name |
| `fifa_team_code` | str | 3-letter canonical id |
| `snapshot_date` | str (ISO) | Capture date |
| `injury_notes` | str | Key injuries (`"Mbappé (hamstring, doubtful)"`) |
| `absence_notes` | str | Suspended / not-called-up players |
| `rotation_notes` | str | Manager rotation signals (e.g. "Group-stage rotation expected") |
| `motivation_notes` | str | Tournament-stakes context (e.g. "must-win to advance") |
| `reliability` | str | `high` / `medium` / `low` — editor's confidence in the notes |
| `source_url` | str | Source(s) used for the notes |
| `notes` | str | Free-form misc |

### 2.4 Value tiers

Manual editorial bands for quick visual reading. Tiers are intentionally
crude so the dashboard doesn't promise a calibrated signal.

| tier | squad_market_value_eur (rough) |
|---|---|
| `Elite` | ≥ 1.0 B € |
| `Strong` | 300 M € – 1.0 B € |
| `Mid` | 50 M € – 300 M € |
| `Low` | < 50 M € |
| `Unknown` | no row in `squad_strength_snapshot.csv` |

**These thresholds are the only place Phase 4 hardcodes numbers** and
they are documented, non-secret, and easy to adjust.

### 2.5 File format conventions

- Plain CSV, UTF-8, header row required.
- `fifa_team_code` is the canonical join key in all three files
  (matches `team_identity.json` keys).
- `team` is for human readability; join on `fifa_team_code`.
- One snapshot per team per file. A new snapshot = a new row with a
  newer `snapshot_date`. The loader picks the most recent per team.
- Empty cells (not "null", not "0") for missing data — keeps the file
  git-diff friendly.

### 2.6 What's intentionally out of scope

- **No Transfermarkt scraping** — explicit user constraint. Future phase
  only after manual workflow proves useful.
- **No auto-merge with model** — Phase 4 is read by the dashboard, never
  by `evaluate_match`.
- **No time-decay weighting** — a single snapshot is a single snapshot.
  If we need a time series, that's Phase 5+.
- **No per-player rows** — totals only. Per-player would balloon the CSV
  and is unused by the dashboard.

---

## 3. Loader design (sketch — not implemented here)

```
soccer_ev_model/squad_context.py
    load_squad_strength(path) -> dict[canonical_id, SquadRow]
        - read CSV with pandas
        - validate: required columns present
        - validate: fifa_team_code is in team_identity registry
          (warn on unknown, don't fail the dashboard)
        - dedupe: keep most recent snapshot_date per team
        - return {canonical_id: row}

    load_fifa_ranking(path) -> dict[canonical_id, RankingRow]
        - same pattern

    load_team_context_notes(path) -> dict[canonical_id, NotesRow]
        - same pattern

    build_squad_context_view(home_cid, away_cid, sq, rk, nt) -> dict
        - returns the merged view the dashboard renders
        - includes: home_value, away_value, gap_pct, value_tier_home,
          value_tier_away, fifa_rank_home, fifa_rank_away,
          notes_home, notes_away, source_attribution, label
        - if both files missing, returns an empty view (the dashboard
          must handle that gracefully)
```

**Key safety properties:**
- All loaders are pure functions of the file contents + the registry.
- No I/O at import time.
- Missing file = empty dict, not exception.
- Unknown `fifa_team_code` = warning + skip, not exception.
- Loaders never touch `ratings`, `ev_workflow`, or any model code.

---

## 4. Dashboard behavior (sketch — not implemented here)

### 4.1 New context panel: `_render_squad_context(view)`

Position in the per-game render stack:
1. Header (matchup, date, book odds)
2. Model probabilities (existing — `blend_probs` + market baseline + divergence label)
3. **← new: `_render_squad_context(view)` sits here**
4. Poisson secondary view (existing)
5. Group-stage context warnings (existing)
6. +EV flags (existing)

### 4.2 Panel contents (top to bottom)

- **Headline:** "Squad value: $ARG 2.30 B € (Elite) vs $ALG 95 M € (Mid) — gap +2321%."
- **Tier badges:** two pills showing `Elite / Strong / Mid / Low / Unknown`.
- **Squad-value bar:** a simple horizontal bar comparison.
  - Fall back to FIFA rank comparison if squad value missing for either side.
- **Notes bullets** (if `team_context_notes` has rows for either team):
  - injury_notes, absence_notes, rotation_notes, motivation_notes.
- **Source attribution line:** small text, "Source: Transfermarkt (manual entry, snapshot 2026-06-01)".
- **Label banner (always visible):**
  > "Context only — not included in the probability model yet."

### 4.3 Hard "do not touch" rules in the renderer

- Does not call `evaluate_match`.
- Does not read or write `blend_probs`, `pi_probs`, `pi_only_probs`,
  `elo_only_probs`, `calibrated_pi`, `edges`, `plus_ev_flags`.
- Does not alter pick recommendations.
- Does not run if `view` is empty (shows nothing, not an error).

### 4.4 Mobile-first

Same as existing dashboard: single column, large touch targets, no
horizontal scroll. The bar comparison should not require a tooltip to be
useful.

---

## 5. Tests (planned for the future PR)

| test | asserts |
|---|---|
| `test_squad_loader_missing_file` | loader returns `{}` when file absent, no exception |
| `test_squad_loader_malformed_csv` | loader raises a clear `ValueError` naming the bad column/row |
| `test_squad_loader_unknown_team_code` | loader warns + skips the row, does not crash |
| `test_squad_loader_dedupes_to_latest_snapshot` | two rows for `ARG` → keeps the newer `snapshot_date` |
| `test_squad_view_with_both_sides_present` | view has `home_value`, `away_value`, `gap_pct`, both tiers |
| `test_squad_view_with_fifa_fallback` | squad missing → view shows FIFA rank instead |
| `test_squad_view_empty_when_all_missing` | view is `{}` — dashboard renders nothing |
| `test_squad_panel_does_not_alter_blend` | render a matchup; assert `result["blend_probs"]` is byte-identical before/after the panel renders |
| `test_squad_panel_renders_label_banner` | the literal "Context only — not included in the probability model yet." string is present in the panel output |
| `test_no_transfertmarkt_network_calls` | grep / mock-check that no Phase 4 code path touches the network |

The most important test is the **"does not alter blend"** assertion.
That's the contract.

---

## 6. Doc updates (planned for the future PR)

- `docs/manual_data_update_guide.md` — how Todd edits the three CSVs
  manually before each tournament, with copy-pasteable header rows and
  a worked example.
- `README.md` — link the new guide from the main README.
- `dashboard/README.md` — note the new panel and that it's context-only.

---

## 7. PR plan (when this gets implemented)

Small, reviewable PR. No mixed changes.

**Branch:** `feat/squad-strength-context`
**Files touched (estimated, not yet created):**
- `data/manual/squad_strength_snapshot.csv` (new, ~10 sample rows)
- `data/manual/fifa_ranking_snapshot.csv` (new, ~10 sample rows)
- `data/manual/team_context_notes.csv` (new, ~5 sample rows)
- `soccer_ev_model/squad_context.py` (new, ~150 lines)
- `dashboard/app.py` (add one `_render_squad_context` call, ~30 lines)
- `tests/test_squad_context.py` (new, ~10 tests)
- `docs/manual_data_update_guide.md` (new)
- `docs/plans/2026-06-17-soccer-model-phase4-design.md` (this file, finalize)

**Forbidden in this PR:**
- Any change to `ev_workflow.py`, `pi_ratings.py`, `elo_ratings.py`,
  `no_vig.py`, `confidence.py`, `prediction_summary.py`.
- Any change to `blend_probs` / `pi_probs` / `elo_only_probs` semantics.
- Any Transfermarkt / HTTP scraping code.
- Auto-merge. PR sits for review.

---

## 8. Open questions (answer before opening the PR)

1. **Snapshot cadence:** manual update before every tournament? Or
   also at group-stage midpoint? (Default: pre-tournament only.)
2. **Tier thresholds:** are the §2.4 ranges right for international
   football? They look high because Transfermarkt totals are aggregated
   Euro values, not US-market caps. Easy to adjust.
3. **FIFA ranking source:** FIFA.com only, or also `worldfootball.net` /
   `eloratings.net` for cross-checking?
4. **Notes reliability flag:** does `reliability = low` need a visual
   treatment in the panel? (Default: yes — grey the notes text.)
5. **Confederation field:** is it ever used in the dashboard, or is it
   pure metadata for Todd? (Default: metadata only.)

---

## 9. Confirmation of no-probability-change invariant

This design explicitly **does not** modify:

- `soccer_ev_model/ev_workflow.py` — `evaluate_match`, `_probs_from_ratings`,
  `_probs_from_ratings_blend`, `_logistic_matchup`, `_calibrate_probs`,
  `_BASE_H/_BASE_D/_BASE_A`.
- `soccer_ev_model/pi_ratings.py` — `compute_pi_ratings`,
  `pi_diff_features`, `LEARNING_RATE`, `NEUTRAL_RATING`.
- `soccer_ev_model/elo_ratings.py` — `elo_at`, `load_elo_ratings`.
- `soccer_ev_model/prediction_summary.py` — `calculate_market_deltas`,
  `market_divergence_label`, `poisson_outcome_probs`,
  `expected_goals_from_blend`, `poisson_agreement_label`,
  `group_context_warnings`, `confidence_tier`.
- `soccer_ev_model/no_vig.py` — `implied_probs`, `remove_vig`.
- `soccer_ev_model/confidence.py` — `assess_match_confidence`,
  `render_warning_banner`.

The Phase 4 panel **reads** `result` for display purposes only and never
**writes** back to it. Pick recommendations, calibration, edges, and
+EV flags are untouched.

---

## 10. Out of scope reminders

- **No Phase 5 yet.** Even if Phase 4 goes well, a future "squad value
  feeds the blend" change is a separate design and a separate PR.
- **No Elo adjustments from squad value.** That's a different model
  question with different validation needs.
- **No manual override of probabilities** via these files. If a future
  user wants to nudge probabilities manually, that's a different
  feature with its own audit trail.
