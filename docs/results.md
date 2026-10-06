# UFC Elo — Results & Decisions

This is the project's living record of experiments, findings, recommendations, and important technical decisions.

Do not use this file as a task list. Tasks belong in `tasks.md`.

---

# Project Findings

Record important discoveries about the dataset, architecture, models, and system here.

## Dataset

### Findings

* Tables: `fighter.csv` (4,581 fighters), `event.csv` (1,259 events, 1993-11-12 → 2026-08-08), `fight.csv` (11,441 fights), `round.csv` (25,131 round rows), `fighter_bonus.csv` (2,409 bonus records), `master.csv` (11,441 rows — a pre-joined, denormalized fight+fighter+round-totals table).
* Keys are clean: `fight_id`/`fighter_id`/`event_id` are unique per row in their source tables, zero exact duplicate rows anywhere, and `winner_id` is always either `r_id`, `b_id`, or null — no orphaned foreign keys between tables.
* `result_status`: win 11,238 / no_contest 118 / draw 85. `winner_id` is null for exactly the 203 non-`win` rows — consistent.
* `method` has 11 categories; decisions (unanimous+split+majority) are ~43% of fights, KO/TKO ~32%, submission ~21%.
* Volume grows over time: <100 fights/year before 2000, ramps through the 2000s, stabilizes at ~500-600 fights/year from 2014 onward (2026 partial-year to date).
* Round-by-round data (`round.csv`) is incomplete for older fights: 330 of 11,441 fights (2.9%) have no round rows at all, and `r_ctrl`/`b_ctrl` (control time) is null for 2,658 of 25,131 round rows.
* `fighter.csv` career-stat columns (`slpm`, `str_acc`, `sapm`, `str_def`, `td_avg`, `td_acc`, `td_def`, `sub_avg`) use `0` for both "genuinely zero" and "no UFC.com record available" — 858 fighters (~19%) have `slpm == 0`, which is a missingness marker, not a true rate, and must not be imputed/used as a literal 0 in features.
* `height` is a free-text string (`5' 10"`), not numeric — needs parsing before use.
* `reach_inches` is missing for 1,992 fighters (43%) and `stance` for 897 (20%) — both meaningfully missing-not-at-random (older/lower-profile fighters).
* 8 pairs of fighters share an identical `fighter_name` but have distinct `fighter_id`, different `dob`/`weight_lbs` (e.g. two "Bruno Silva", two "Michael McDonald") — real namesakes, not a dedup bug. **Never join/dedupe on `fighter_name`; always use `fighter_id`.**
* 2,239 of 4,124 fighters (54%) have fought from both the red (`r_id`) and blue (`b_id`) corner across their career — corner is not a fixed attribute of a fighter, it's assigned per fight.

### Data Quality Issues

* **Red-corner leakage in historical data (critical, affects modeling).** Red corner (`r_id`) wins 67.6% of decided fights overall — far from the 50/50 a coin-flip-corner-assignment would produce. Breaking this out by year shows why: red wins **~100% of fights from 1993–2009**, then drops sharply to a stable **55–60% from 2010 onward**. The pre-2010 near-100% rate means `r_id` was effectively backfilled as "the winner" for early UFCStats records rather than reflecting a real pre-fight corner assignment — using `r_id`/`b_id` as a naive feature would leak the outcome for any fight before ~2010. Post-2010, red still wins ~57% (and 68–77% in title fights specifically), which looks like a real convention (red corner = higher-ranked/betting-favorite fighter) rather than leakage, but it's still a strong prior baked into the label that any model comparison must account for.
* **Naive baseline is not 50%.** Because of the above, "always predict red" scores ~67.6% accuracy over the full dataset and ~57% even restricted to 2010+. Any classifier must be benchmarked against this corner-prior baseline, not a coin flip, or its offline metrics will look better than they are.
* `weight_lbs` has extreme outliers (up to 770 lbs) — these are legitimate historical UFC 1-era fighters (e.g. Emmanuel Yarborough), not data errors, but they will skew scaling/normalization if not treated (e.g. winsorized or excluded from early modern-era modeling).
* 330 fights (2.9%) have no round-level detail in `round.csv` — any round-level feature will need to handle these as missing rather than assuming coverage.
* `event.csv.location` missing for 476 of 1,259 events (38%); `bonuses` in `master.csv` is null for 9,090 of 11,441 rows (most fights simply have no bonus, consistent with `fighter_bonus.csv` covering only 2,351 fights).

### Important Assumptions

* `fighter_id` is the correct join key for fighter identity everywhere; fighter name is not unique and must not be used for matching or deduplication.
* Career stat columns in `fighter.csv` (slpm, str_acc, etc.) appear to be **as-of-latest-scrape** aggregates, not point-in-time-of-fight snapshots — using them directly as pre-fight features for older fights would leak future career performance into the past. This needs to be confirmed before use in features/Elo, and until confirmed, treat these columns as unsuitable for time-aware modeling without per-fight historical reconstruction from `round.csv`/`fight.csv`.
* Given the red-corner artifact above, any model trained across the full date range should either (a) restrict training/eval to 2010+ fights where corner reflects a real signal, or (b) explicitly model/neutralize corner as a confound (e.g. symmetrize features by swapping red/blue at training time) rather than treating `r_id`/`b_id` as arbitrary labels.

---

# Experiments

Record meaningful experiments here.

## Experiment Template

### Experiment — [Name]

**Date:**

**Goal:**

**Model:**

**Features:**

**Configuration:**

**Dataset:**

**Evaluation:**

**Results:**

```text
Metric: Value
Metric: Value
Metric: Value
```

**Observations:**

*

**Conclusion:**

*

**Next Step:**

*

---

# Model Results

Track important model results here.

| Model                                    | Features                       | Mean Accuracy | Mean Log Loss | Mean ROC-AUC | Notes                                                        |
| ----------------------------------------- | ------------------------------- | -------------: | -------------: | -------------: | ------------------------------------------------------------- |
| Logistic regression (`logistic_regression.py`) | 14 (`FEATURE_COLUMNS`)     |         0.6195 |         0.6563 |         0.6639 | Current selected baseline, re-run 2026-10-06 via `run_backtest` with ROC-AUC now tracked; numbers match the prior accuracy/log_loss run exactly (reproducible, same 24 walk-forward folds 2003-2026). Early folds (2003-2008) show `RuntimeWarning: overflow in matmul` from unregularized/small-fold instability - known issue, not yet resolved. |
| Random forest (`random_forest.py`)       | same 14 (`FEATURE_COLUMNS`)      |         0.5997 |         0.7101 |         0.6387 | Worse than logistic regression on all three metrics - too few/too linear features for a tree to gain from |
| Random forest (`random_forest.py`)       | ~90 (full `ALL_CANDIDATE_COLUMNS`) |         0.6164 |         0.6539 |              — | Better than 14-feature random forest, still 0.31pp below logistic regression on accuracy - see Feature Findings below. ROC-AUC not yet computed for this run. |
| XGBoost (`xgboost_model.py`)             | same 14 (`FEATURE_COLUMNS`)      |         0.6146 |         0.6533 |         0.6632 | Slightly below logistic regression on accuracy, slightly better log_loss, slightly below on ROC-AUC |
| **XGBoost (`xgboost_model.py`)**         | **~90 (full `ALL_CANDIDATE_COLUMNS`)** |     **0.6225** |     **0.6468** |              — | **Best of every model/feature combination tried so far on accuracy/log_loss** (2026-10-06). ROC-AUC not yet computed for this run. |
| Logistic regression, **nested walk-forward** (`src/evaluation/nested_validation.py`) | 14 (`FEATURE_COLUMNS`) | 0.6194 | 0.6495 | — | Per-outer-fold `C` tuned on an inner walk-forward within that fold's own training window only (2026-10-06); consistently chose `C=0.01` every fold. Accuracy ~unchanged vs. fixed `C=1.0`, log_loss improved (0.6563 -> 0.6495) - the fixed C=1.0 was mildly under-regularized. ROC-AUC not yet computed for this run. |
| Random forest, **nested walk-forward** (`src/evaluation/nested_validation.py`) | ~90 (full `ALL_CANDIDATE_COLUMNS`) | 0.6221 | 0.6536 | — | Per-outer-fold `n_estimators`/`max_depth` tuned on an inner walk-forward within that fold's own training window only (2026-10-06), grid `{100,200} x {3,6,None}`. `max_depth=None` (the fixed run's default) was **never** chosen by inner validation in any of the 24 outer folds - `max_depth=6` won 18/24 folds, `max_depth=3` the other 6, split roughly evenly between `n_estimators` 100/200. Accuracy improved from the fixed-hyperparameter full-pool run (0.6164 -> 0.6221, now best-of-all on accuracy), log_loss about flat (0.6539 -> 0.6536). ROC-AUC not yet computed for this run. |
| XGBoost, **nested walk-forward** (`src/evaluation/nested_validation.py`) | ~90 (full `ALL_CANDIDATE_COLUMNS`) | 0.6214 | 0.6439 | — | Per-outer-fold `n_estimators`/`max_depth`/`learning_rate` tuned on an inner walk-forward within that fold's own training window only (2026-10-06), grid `{100,200} x {3,5} x {0.05,0.1}`. Mostly chose `max_depth=3, learning_rate=0.05` (shallower/slower-learning than the fixed run's defaults). Accuracy slightly below the fixed-hyperparameter full-pool run (0.6225 -> 0.6214) but log_loss improved to the best of every run so far (0.6468 -> 0.6439) - better-calibrated probabilities from the tuned, more conservative trees. ROC-AUC not yet computed for this run. |

All three nested walk-forward runs are complete as of 2026-10-06.

**Baseline re-confirmed 2026-10-06** with ROC-AUC added to `src/models/*.py` and `src/evaluation/nested_validation.py` (`roc_auc_score` on predicted probabilities). This is the "exact baseline" the feature-ablation and nested-validation comparison work below is measured against: 14 features (`age_diff`, `experience_diff`, `prior_win_rate_diff`, `win_streak_diff`, `loss_streak_diff`, `form_last5_win_rate_diff`, `reach_diff_x_<division>` x8), default hyperparameters (`C=1.0` / `n_estimators=200,max_depth=None` / `n_estimators=200,max_depth=3,learning_rate=0.1`), unchanged walk-forward splitting/preprocessing.

### baseline (14) vs. expanded (39) x untuned vs. tuned (2026-10-06)

After promoting `career_volume_stats`/`career_style_stats`/`stance_mismatch` into `FEATURE_COLUMNS` (see Feature Findings), ran the 2x2x3 comparison: `BASELINE_FEATURE_COLUMNS` (14, pre-promotion) vs. `FEATURE_COLUMNS` (39, post-promotion) x fixed default hyperparameters (`run_backtest`) vs. per-outer-fold nested-tuned (`run_nested_backtest`, each model's own `PARAM_GRID`), for all three models. **Stopped early by request after 3 of 6 cells** - random forest's nested tuning took ~16 minutes for one cell (grid x inner-fold x outer-fold refits add up fast), so random forest (expanded, tuned) and both XGBoost (tuned) cells were not run.

| Model | Features | Tuning | Accuracy | Log Loss | ROC-AUC |
| --- | --- | --- | ---: | ---: | ---: |
| logistic_regression | baseline (14) | untuned | 0.6195 | 0.6563 | 0.6639 |
| logistic_regression | baseline (14) | tuned | 0.6194 | 0.6495 | 0.6645 |
| logistic_regression | expanded (39) | untuned | 0.6200 | 0.6604 | 0.6643 |
| logistic_regression | expanded (39) | tuned | 0.6204 | 0.6533 | 0.6656 |
| random_forest | baseline (14) | untuned | 0.5997 | 0.7101 | 0.6387 |
| random_forest | baseline (14) | tuned | 0.6194 | 0.6493 | 0.6664 |
| random_forest | expanded (39) | untuned | 0.6135 | 0.6793 | 0.6538 |
| random_forest | expanded (39) | tuned | — not run — | — | — |
| xgboost_model | baseline (14) | untuned | 0.6146 | 0.6533 | 0.6632 |
| xgboost_model | baseline (14) | tuned | — not run — | — | — |
| xgboost_model | expanded (39) | untuned | 0.6265 | 0.6421 | 0.6740 |
| xgboost_model | expanded (39) | tuned | — not run — | — | — |

**Observations on the completed cells:** tuning helped logistic regression modestly (lower log_loss, slightly higher ROC-AUC, accuracy flat) on both feature sets. Tuning helped random forest *enormously* on the baseline feature set - `max_depth=None` (its hardcoded default) was badly overfitting; nested tuning picked a shallower tree per fold and accuracy jumped from 0.5997 to 0.6194, log_loss from 0.7101 to 0.6493, now competitive with logistic regression. The expanded feature set helped every model that was tested on it untuned (clearest for XGBoost: 0.6146 -> 0.6265 accuracy, 0.6533 -> 0.6421 log_loss).

**Best known config per model (incomplete comparison - see gaps above):**
* **logistic_regression: expanded (39), tuned** - best accuracy (0.6204) and ROC-AUC (0.6656) of its 4 cells; log_loss (0.6533) is close to but not quite baseline-tuned's 0.6495.
* **random_forest: baseline (14), tuned** - best of its 3 completed cells on all three metrics; expanded+tuned was never run, so this is not confirmed as random forest's true best, just its best *known* configuration.
* **xgboost_model: expanded (39), untuned** - best of its 2 completed cells on all three metrics; no tuned run exists for XGBoost at all, so tuning might still improve on this.

**Follow-up (recorded in `tasks.md`):** run the 3 missing cells (random_forest expanded+tuned, xgboost_model baseline+tuned, xgboost_model expanded+tuned) to complete the matrix before treating any "best config" above as final.

---

# Elo Results

Record meaningful findings from the Elo system.

### Configuration

```text
Initial Rating:
K Factor:
Other Parameters:
```

### Findings

*

### Experiments

*

---

# Feature Findings

Record useful discoveries about feature engineering.

### Useful Features

* Not yet determined for the new feature groups below - see Notes. The existing selected set (`src/models/logistic_regression.py FEATURE_COLUMNS`) still reflects the pre-expansion ablation only.
* **`career_volume_stats`/`career_style_stats`/`finish_tendency` carry real signal, just not for a linear model (2026-10-06).** These three groups hurt logistic regression's accuracy (negative `accuracy_drop` in the ablation above) but, fed to `src/models/random_forest.py` instead, raise its mean accuracy from 0.5997 to 0.6164. Fed to `src/models/xgboost_model.py`, they raise its accuracy from 0.6146 to 0.6225 and its log_loss to 0.6468 - both the best of any model/feature combination tried so far, beating logistic regression on both metrics. Confirms the earlier guess that a linear model couldn't use this ~80-column, correlated, nonlinear feature set rather than the features themselves being noise. Next step: an ablation *within XGBoost* (which of these groups specifically help it, and whether its default hyperparameters - `n_estimators=200, max_depth=3, learning_rate=0.1` - are even close to good) rather than assuming all groups contribute equally or that these defaults are tuned.
* **Promoted into `FEATURE_COLUMNS` (2026-10-06): `career_volume_stats`, `career_style_stats`, `stance_mismatch`.** Ablation methodology this time was add-one-group-to-the-exact-14-feature-baseline (not leave-one-out-of-the-full-~90-column-pool, see the entry above - different methodology, kept both since they answer different questions), run for all three models with ROC-AUC now tracked too:

  | Model | Variant | Accuracy | Log Loss | ROC-AUC |
  | --- | --- | ---: | ---: | ---: |
  | logistic_regression | baseline (14) | 0.6195 | 0.6563 | 0.6639 |
  | logistic_regression | + career_volume_stats | 0.6233 | 0.6604 | 0.6654 |
  | logistic_regression | + career_style_stats | 0.6191 | 0.6566 | 0.6638 |
  | logistic_regression | + finish_tendency | 0.6155 | 0.6615 | 0.6604 |
  | logistic_regression | + stance_mismatch | 0.6195 | 0.6563 | 0.6639 |
  | random_forest | baseline (14) | 0.5997 | 0.7101 | 0.6387 |
  | random_forest | + career_volume_stats | 0.6147 | 0.6958 | 0.6510 |
  | random_forest | + career_style_stats | 0.6079 | 0.6930 | 0.6492 |
  | random_forest | + finish_tendency | 0.6014 | 0.6925 | 0.6427 |
  | random_forest | + stance_mismatch | 0.6017 | 0.6964 | 0.6407 |
  | xgboost_model | baseline (14) | 0.6146 | 0.6533 | 0.6632 |
  | xgboost_model | + career_volume_stats | 0.6249 | 0.6450 | 0.6716 |
  | xgboost_model | + career_style_stats | 0.6181 | 0.6474 | 0.6673 |
  | xgboost_model | + finish_tendency | 0.6140 | 0.6523 | 0.6627 |
  | xgboost_model | + stance_mismatch | 0.6146 | 0.6523 | 0.6639 |

  Decision: **promote** `career_volume_stats` (helps all three models on accuracy/ROC-AUC, only a negligible log_loss dip for logistic regression), `career_style_stats` (negligible/flat for logistic regression, clear gain for random forest and XGBoost on all three metrics), and `stance_mismatch` (flat for logistic regression and XGBoost, small genuine gain for random forest on all three metrics, no harm anywhere). **Do not promote** `finish_tendency` - clearly hurts logistic regression on all three metrics and is flat-to-mixed for random forest/XGBoost, so no model benefits enough to justify it. `FEATURE_COLUMNS` is shared across all three models (single list in `logistic_regression.py`), so this is a per-group net-across-models call, not a per-model optimum - `src/models/logistic_regression.py` now also exposes `BASELINE_FEATURE_COLUMNS` (the original 14) so the pre-promotion set stays referenceable for comparisons.

### Features That Did Not Help

* `finish_tendency` (per-method-category prior win/occurrence rate, average finish round) - see promotion decision above. Clearly hurts logistic regression (-0.0040 accuracy, +0.0052 log_loss, -0.0035 ROC-AUC vs. baseline) and is flat-to-mixed for random forest/XGBoost - not promoted into `FEATURE_COLUMNS` (2026-10-06).

### Features Removed

*

### Notes

* **Expanded `src/features/engineering.py` (2026-10-06)** with ~190 new candidate columns, all leakage-safe (prior-fights-only): `add_career_stat_features` (career-to-date per-minute rate + accuracy/output-share ratios for every detailed stat in master.csv - strikes by target/position, takedowns, control time, reversals, sub attempts, knockdowns), `add_finish_tendency_features` (per-method-category prior win/occurrence rate + average finish round), `add_recent_form_features` (win/loss streak, last-5 win rate), `add_stance_matchup_features` (orthodox/southpaw mismatch flag). Added to `src/evaluation/feature_selection.py CANDIDATE_FEATURES` as 4 new groups (`career_volume_stats`, `career_style_stats`, `finish_tendency`, `recent_form`) but **not yet promoted into `baseline.py FEATURE_COLUMNS`** - selection should happen via ablation per the project's existing workflow, not by assumption.
* **Missing round-detail masking.** 330/11,441 fights (2.9%) have `rounds_fought == 0` - their `r_total_*`/`b_total_*` columns are `0` as a missingness marker, not a real zero-output fight (see Data Quality Issues above). `_prior_career_totals` zeroes out BOTH the stat and the fight's duration contribution for these fights before accumulating, so they contribute to neither the numerator nor denominator of any career rate - the alternative (counting duration but not strikes, or vice versa) would silently bias every subsequent career rate for that fighter.
* **`method_label` keeps every raw `method` category distinct** (`src/data/cleaning.py METHOD_LABEL_MAP`) - including all three decision types (unanimous/split/majority, so a fighter who wins clearly vs. one who repeatedly escapes with a split decision are distinguishable) and the rare ones (DQ, doctor stoppage, overturned, could-not-continue, other). None are collapsed into a generic "other" - an initial pass did this and was corrected; a fighter's raw rate of a rare outcome (e.g. `prior_dq_rate`) is itself a meaningful feature for the fighters who actually have history with it, and merging it away would have erased that signal for exactly the fighters where it matters.
* `Overturned` and `Could Not Continue` end up with **zero rows** in the modeling dataset regardless of `method_label` - both always coincide with a null `winner_id` (no_contest/draw), which `clean_fights` already drops since there's no win/loss label to learn from. This is pre-existing `clean_fights` behavior, not something the new method labeling did - if a future target needs to model those outcomes directly, the winner-only filter in `clean_fights` would need revisiting.
* **Ablation collinearity, fixed (2026-10-06).** `src/evaluation/feature_selection.py CANDIDATE_FEATURES` had 4 exact linear dependencies, confirmed directly (row-sums equal to 0 up to float error, and `matrix_rank` deficiency matching column-for-column): the 11 `method_label` prior-rate columns are mutually exclusive and sum to ~1 (both the plain and win-conditioned versions), and `sig_str_landed`/`sig_str_atmp` are each exactly the sum of their head/body/leg breakdown *and* separately the sum of their distance/clinch/ground breakdown. Fixed with standard k-1 reference-category encoding (drop one column per group - no information lost, the intercept + remaining coefficients span the same model). Also dropped `overturned`/`could_not_continue` (structurally always 0 - both methods only occur in no-contest fights, which `clean_fights` already excludes) and `other` (real, but only 2 fights in the entire 30+ year dataset, so effectively constant-zero through nearly every walk-forward fold) from the rate candidates specifically - `method_label` itself still tracks all of these correctly, this only concerns which columns feed the ablation/selection step. Verified full column rank (deficiency 0) across early/mid/late walk-forward slices after the fix.
  * The `RuntimeWarning: divide by zero encountered in matmul` seen during backtests is a **separate, pre-existing, benign issue** - confirmed it also fires with the original ~12-column `FEATURE_COLUMNS` (1260 times across the existing backtest, final accuracy/log_loss always finite), so it predates today's feature work and isn't caused by it. Looks like an environment-level sklearn/BLAS lbfgs quirk, not a correctness bug - not something to chase further right now.
* **Ablation results for the new groups (2026-10-06, logistic regression, full walk-forward backtest).** `recent_form` (win/loss streak, last-5 win rate) is the only new group with a positive `accuracy_drop` (+0.0072 - removing it hurts), putting it in the same tier as `experience_diff`/`reach_diff_x_division`. `career_volume_stats`, `finish_tendency`, and `career_style_stats` all show a **negative** `accuracy_drop` (-0.0011, -0.0040, -0.0049 respectively) - the full model does *better* without them. This doesn't necessarily mean the underlying signal is worthless - logistic regression on ~80 mostly-correlated per-minute/share columns at once is exactly the setting where a linear model drowns in noise; a tree-based model (Phase 6) may extract value these ablation numbers can't see. For now, do not add `career_volume_stats`/`career_style_stats`/`finish_tendency` to `baseline.py FEATURE_COLUMNS`; `recent_form` is a reasonable promotion candidate pending a decision.
* **`stance_mismatch` is corner-symmetric, not an antisymmetric diff** - it's the same value regardless of which fighter is red/blue. `src/models/logistic_regression.py _symmetrize` negates every listed feature column for the blue-perspective row, which is only correct for `r - b` style diffs. Left out of `CANDIDATE_FEATURES` for now rather than silently fed through `_symmetrize`, which would flip its sign incorrectly for the blue row.

---

# Error Analysis

Record meaningful patterns in model errors.

### Observed Errors

*

### Patterns

*

### Potential Explanations

*

### Follow-Up Experiments

*

---

# Recommendations

Record recommendations made during development.

## Recommendation — [Topic]

**Recommendation:**

**Reason:**

**Alternatives considered:**

**Decision:**

**Status:**

* [ ] Pending
* [ ] Accepted
* [ ] Rejected

---

# Technical Decisions

Record decisions that affect the architecture or implementation.

## Decision — Nested walk-forward validation for hyperparameter tuning

**Decision:** Added `src/evaluation/nested_validation.py` (`select_hyperparameters`, `run_nested_backtest`), used by all three models via a small `build_model(**params)` + `PARAM_GRID` + `SCALE` contract added to each (`logistic_regression.py`, `random_forest.py`, `xgboost_model.py`). For each outer walk-forward fold, hyperparameters are chosen via an *inner* walk-forward confined entirely to that fold's own training window (the outer test year is never passed into selection), then a final model is trained on the full outer training window with the chosen hyperparameters and scored once on the outer test year.

**Why:** the existing `run_backtest` functions used fixed, hand-picked hyperparameters (`C=1.0`, `n_estimators=200`, etc.) - correct but untuned. Once tuning starts, picking hyperparameters using the same year being predicted would leak future information into model selection, inflating offline metrics relative to real deployment. Nested validation (inner loop for selection, outer loop for the one-shot final estimate) is the standard fix.

**Alternatives:** a single inner validation split per outer fold (e.g. last N months of the training window) instead of a full inner walk-forward - simpler but throws away most of the inner data and only validates on one period instead of averaging over several. Grid search (not random/Bayesian) was used for the search itself - each model's hyperparameter space here is 1-3 knobs with a handful of values, where exhaustive search is simpler, fully transparent (every candidate's inner score is inspectable, matching the existing ablation style in `feature_selection.py`), and needs no new dependency.

**Consequences:** nested validation is slower - every outer fold now refits the model once per (inner fold x grid candidate) in addition to the final fit, so runtime scales with grid size. Confirmed via `tests/test_nested_validation.py` that the outer test year's fight_ids are never passed into `select_hyperparameters` (spy-based test) and that an outer fold too small to produce any inner fold falls back to the grid's first candidate rather than erroring - that fallback should be treated as "untuned" (`n_inner_folds == 0` in the results), not a real selection.

---

## Decision — `stance_mismatch` given an explicit symmetric-feature path instead of being forced through `_symmetrize`

**Decision:** Added `SYMMETRIC_FEATURE_COLUMNS = ["stance_mismatch"]` to `src/models/logistic_regression.py`. `_symmetrize(df, feature_columns, symmetric_columns=SYMMETRIC_FEATURE_COLUMNS)` now only negates `feature_columns` entries that are *not* in `symmetric_columns` for the blue-perspective row; symmetric columns are left unchanged. `stance_mismatch` added to `feature_selection.py CANDIDATE_FEATURES` now that this is safe.

**Why:** `_symmetrize` mirrors every fight into a red- and blue-perspective row and negates every feature column for the blue row, which is only correct for antisymmetric r-minus-b diffs (e.g. `age_diff`). `stance_mismatch` is corner-symmetric - true/false regardless of which fighter is labeled r or b - so negating it would turn a `0`/`1` flag into `0`/`-1`, which is meaningless to a model and was never a real transformation of the feature (confirmed directly: `_symmetrize` on a one-row mismatch example now returns `stance_mismatch=1` for both the red and blue copies, `age_diff` still flips sign as expected).

**Alternatives considered:** computing two separate corner-specific stance columns (`r_is_orthodox`, `b_is_orthodox`) and letting `_symmetrize`'s existing negation apply - rejected as more columns for no extra information, and it would reintroduce a different asymmetry (which column means "self" vs "opponent") that `_symmetrize` would still get wrong without a separate carve-out. The chosen fix generalizes to any future corner-symmetric feature by just adding it to `SYMMETRIC_FEATURE_COLUMNS`.

**Consequences:** `_symmetrize`'s signature gained a `symmetric_columns` parameter with a module-level default, so every existing call site (`random_forest.py`, `xgboost_model.py`, `nested_validation.py`) picks up the fix automatically with no changes needed there. Verified with `tests/test_nested_validation.py` and the full `tests/` suite (14/14 passing) after the change. `stance_mismatch` was then ablated against the 14-feature baseline and promoted into `FEATURE_COLUMNS` - see Feature Findings above.

---

## Decision — Model artifact persistence, saved from the best-known (incomplete) config per model

**Decision:** Added `src/models/persistence.py` (`fit_final_model`, `save_model_artifact`, `load_model_artifact`). `fit_final_model` fits one model on the *entire* fights history (no held-out test fold - this is the deployable fit, not a backtest fold) using the same `engineer_fold_features`/`_symmetrize`/scaling pipeline as every backtest. `save_model_artifact` writes `models/<name>.joblib` (model + scaler) plus a sidecar `models/<name>.json` recording the feature list, hyperparameters, and the metrics it was selected with. Saved one artifact per model (2026-10-06), using the best-known cell from the baseline-vs-expanded x untuned-vs-tuned comparison above:

| Artifact | Features | Params | Source cell |
| --- | --- | --- | --- |
| `models/logistic_regression.joblib` | expanded (39) | `C=0.01` (chosen by `select_hyperparameters` over the full dataset) | expanded, tuned |
| `models/random_forest.joblib` | baseline (14) | `n_estimators=200, max_depth=6` (chosen by `select_hyperparameters` over the full dataset) | baseline, tuned |
| `models/xgboost_model.joblib` | expanded (39) | `n_estimators=200, max_depth=3, learning_rate=0.1` (fixed default - this was the winning cell) | expanded, untuned |

**Why:** Phase 5 (`tasks.md`) calls for saving model artifacts, and the project now has enough validated configurations to pick from. For the two "tuned" picks, a single final hyperparameter choice doesn't exist directly in a walk-forward backtest (every outer fold can choose differently) - `select_hyperparameters` was re-run once over the *entire* dataset (not per-fold) to get one concrete choice, which is the standard way to pick hyperparameters for a model about to be deployed on all available history.

**Caveat - this is a best-*known*-not-best-possible pick.** The comparison matrix was stopped after 3 of 6 cells (see above) - `random_forest` (expanded, tuned) and both `xgboost_model` tuned cells were never run. If those turn out better, these artifacts should be regenerated. Treat these three as "good enough to unblock Phase 5/persistence," not as the final answer to "which config is best" - that needs the matrix completed first (tracked in `tasks.md`).

**Consequences:** `models/*.joblib` and `models/*.json` are now real files on disk (not committed as sample data - should these be in `.gitignore` or tracked in git? not yet decided, see `tasks.md`). `joblib` added to `requirements.txt` (previously only a transitive dependency via scikit-learn, now imported directly).

---

# Lessons Learned

Record useful lessons discovered during development.

*

---

# Limitations

Document known limitations of the data, Elo system, models, evaluation, API, or deployment.

*

---

# Final Results

Complete this section when the major project work is finished.

## Final Model

**Model:**

**Features:**

**Configuration:**

## Evaluation

| Metric      | Result |
| ----------- | -----: |
| Accuracy    |      — |
| ROC-AUC     |      — |
| Log Loss    |      — |
| Brier Score |      — |

## Final Findings

*

## Known Limitations

*

## Future Work

*
