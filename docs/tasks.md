# UFC Elo — Tasks

This file tracks implementation work for the UFC Elo project.

`scope.md` is the source of truth for the overall project scope.

Tasks should be completed incrementally. Do not implement future tasks prematurely unless required by the current task.

---

# Phase 1 — Understand the Project

* [ ] Read `scope.md`
* [ ] Inspect the existing repository
* [ ] Inspect the UFC dataset
* [ ] Understand the dataset schema
* [ ] Identify missing values and data quality issues
* [ ] Identify how fighters, fights, dates, and outcomes are represented
* [ ] Document important dataset assumptions

---

# Phase 2 — Data Pipeline

* [ ] Implement dataset loading
* [ ] Add data validation
* [ ] Handle required cleaning/transformation
* [ ] Establish consistent fighter identity handling
* [ ] Implement appropriate dataset splitting
* [ ] Add tests for data validation
* [ ] Verify the resulting dataset

---

# Phase 3 — Feature Engineering

* [x] Identify useful features from the available dataset
* [x] Implement fighter-level features
* [x] Implement matchup-level features
* [x] Implement historical/rolling features where required
* [x] Implement shared feature transformations
* [x] Add feature validation tests
* [x] Document important feature decisions
* [x] Run ablation on the new candidate groups (career_volume_stats, career_style_stats, finish_tendency, stance_mismatch; recent_form was already in `FEATURE_COLUMNS`) and promote whichever help into `logistic_regression.py FEATURE_COLUMNS` - done 2026-10-06, see `results.md`: promoted career_volume_stats, career_style_stats, stance_mismatch; left out finish_tendency (hurt logistic regression, flat-to-mixed elsewhere)
* [x] Resolve `stance_mismatch` vs. `_symmetrize`'s antisymmetric-diff assumption before adding it to `CANDIDATE_FEATURES` - done 2026-10-06: added `SYMMETRIC_FEATURE_COLUMNS` to `logistic_regression.py`, `_symmetrize` now only negates antisymmetric columns

---

# Phase 4 — Elo

* [ ] Implement basic Elo calculation
* [ ] Make Elo configuration-driven
* [ ] Add Elo update logic
* [ ] Handle relevant fight outcomes
* [ ] Generate pre-fight Elo features
* [ ] Validate Elo calculations
* [ ] Add Elo unit tests
* [ ] Establish Elo as a baseline

---

# Phase 5 — Baseline Model

* [x] Establish a simple prediction baseline - logistic regression, `src/models/logistic_regression.py`
* [x] Create the training pipeline - `run_backtest` (walk-forward) + `src/evaluation/nested_validation.py` (nested walk-forward tuning)
* [x] Train an initial ML model
* [x] Establish evaluation metrics - accuracy, log loss, ROC-AUC (2026-10-06)
* [x] Save model artifacts appropriately - `src/models/persistence.py`, one `.joblib`+`.json` per model in `models/` (2026-10-06) - built from the best-*known*, not best-possible, config; see follow-up below
* [x] Record baseline results in `results.md`

---

# Phase 6 — Model Development

* [x] Test additional model types - random forest, XGBoost
* [x] Compare feature sets - 14-feature baseline vs. 39-feature expanded set (2026-10-06)
* [ ] Experiment with Elo + ML
* [x] Tune important hyperparameters where justified - nested walk-forward tuning implemented and run for logistic regression (both feature sets) and random forest (baseline only)
* [ ] **Follow-up: complete the baseline-vs-expanded x untuned-vs-tuned matrix** - stopped early 2026-10-06 (random forest's nested tuning took ~16 min/cell). Still need: random_forest (expanded, tuned), xgboost_model (baseline, tuned), xgboost_model (expanded, tuned). The three saved artifacts in `models/` are the best *known* config per model, not confirmed best overall - rerun and resave if these cells change the picture.
* [ ] Compare experiments consistently
* [ ] Analyze model errors
* [ ] Perform relevant slice analysis
* [ ] Record meaningful experiments in `results.md`

---

# Phase 7 — Evaluation

* [ ] Establish final evaluation methodology
* [ ] Run final model comparisons
* [ ] Evaluate probability quality where applicable
* [ ] Perform error analysis
* [ ] Investigate unexpected results
* [ ] Document limitations
* [ ] Select the implementation to carry forward based on documented results

---

# Phase 8 — API

Only complete this phase if required by `scope.md`.

* [ ] Define prediction API
* [ ] Implement model loading
* [ ] Implement request validation
* [ ] Implement feature preparation
* [ ] Implement prediction endpoint
* [ ] Add API tests
* [ ] Verify training/serving feature consistency

---

# Phase 9 — Frontend

Only complete this phase if required by `scope.md`.

* [ ] Define required UI
* [ ] Connect frontend to API
* [ ] Implement prediction flow
* [ ] Display relevant prediction information
* [ ] Handle loading and error states
* [ ] Test frontend/API integration

---

# Phase 10 — Monitoring & Deployment

Only complete this phase if required by `scope.md`.

* [ ] Define relevant logging
* [ ] Implement monitoring
* [ ] Add drift detection where appropriate
* [ ] Containerize application
* [ ] Configure deployment
* [ ] Test deployed system

---

# Phase 11 — Documentation

* [ ] Update README
* [ ] Document architecture
* [ ] Document important decisions
* [ ] Document experiments
* [ ] Document model limitations
* [ ] Create/update model card
* [ ] Ensure `results.md` reflects final findings

---

# Recommendations

Claude should add recommendations here when a meaningful decision needs to be made.

Format:

```text
## Recommendation — [Topic]

Recommendation:
[What should be done]

Reason:
[Why]

Alternatives:
[Other reasonable options]

Status:
[Pending / Accepted / Rejected]
```

Recommendations should not automatically become tasks until they are accepted or clearly required by the project scope.

---

# Current Focus

The current implementation priority should always be the smallest unfinished task that unblocks the next stage of the project.
