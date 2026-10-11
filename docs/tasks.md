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
* [ ] **Follow-up: reconcile `split_fights` with the walk-forward approach.** `src/data/splitting.py split_fights` (one-shot chronological 70/15/15 train/val/test) predates `walk_forward_splits` and is now only used in `notebooks/eda.ipynb` and its own test - no model (`src/models/*.py`) uses it, they all use `walk_forward_splits` instead. Come back and either update `eda.ipynb` to reflect the walk-forward split actually used for modeling, or otherwise reconcile/remove `split_fights` once this phase is revisited.
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
* [x] Run final model comparisons - winner target only: `src/evaluation/train_predict_evaluate.py` runs all three production models' walk-forward backtest side by side, saves per-fold predictions/metrics/summary + comparison charts under `docs/` (2026-10-10, see `results.md`). Multi-target prediction (method of victory, strikes, takedowns, control time, etc.) deliberately deferred.
* [ ] Evaluate probability quality where applicable
* [ ] Perform error analysis
* [ ] Investigate unexpected results
* [ ] Document limitations
* [ ] Select the implementation to carry forward based on documented results

---

# Phase 8 — API

Only complete this phase if required by `scope.md`.

* [x] Define prediction API - `api/main.py` (FastAPI): `/divisions`, `/models`, `/fighters?division=`, `POST /predict`
* [x] Implement model loading - all three saved artifacts loaded once at startup via `src/models/persistence.load_model_artifact`
* [x] Implement request validation - Pydantic `PredictRequest`; unknown model/fighter/pre-debut year all return 400 with a message (2026-10-08)
* [x] Implement feature preparation - `src/features/snapshot.py` (new): point-in-time fighter snapshots via a synthetic "phantom fight" row through the unmodified `engineer_fold_features`, then `build_matchup_features` assembles exactly whichever model's own `feature_columns` calls for (2026-10-08)
* [x] Implement prediction endpoint - `POST /predict`, see above
* [x] Add API tests - `tests/test_api.py`, `tests/test_snapshot.py`
* [x] Verify training/serving feature consistency - snapshot feature formulas cross-checked against `src/features/engineering.py`/`src/models/logistic_regression.py` directly (see commit); `test_build_fighter_snapshot_is_inclusive_of_a_real_fight_on_the_cutoff_date` pins the exact semantics

Known gaps carried forward, not blocking: `models/random_forest.json` was saved with the stale 14-feature baseline (not the current 39-feature `FEATURE_COLUMNS` - see Phase 6 follow-up above); the API reads each model's own saved `feature_columns` generically so this doesn't break anything, it's just a smaller feature set for that one model until it's retrained. Saved artifacts also emit sklearn/xgboost version-mismatch warnings (pickled with older library versions) - functional today, worth re-saving next time models are retrained.

---

# Phase 9 — Frontend

Only complete this phase if required by `scope.md`.

* [x] Define required UI - Gradio: model + division dropdowns, a fighter + optional "year" dropdown per corner, Predict button, per-corner stats panel, past-meetings history (2026-10-08)
* [x] Connect frontend to API - `frontend/gradio.py` calls `api/main.py` over HTTP only, no model/feature logic in the frontend (per `docs/proposal.md`)
* [x] Implement prediction flow - division/fighter dropdowns cascade (division limits fighter choices, fighter selection populates that fighter's real fight years); Predict calls `POST /predict`
* [x] Display relevant prediction information - win probabilities, full striking/grappling/physical/fight-pace stats panel per corner, and (when the two fighters have already met) a predicted-vs-actual card per past meeting including rematches
* [x] Handle loading and error states - API 400s (unknown model, fighter with no data by the chosen year) surface as an inline error message instead of a crash
* [ ] Test frontend/API integration - verified manually end-to-end (uvicorn + gradio, several real matchups incl. a rematch and a same-model corner-swap symmetry check) and via `tests/test_api.py`; no browser/UI-level automated test yet

Note: `frontend/gradio.py` is literally named the same as the `gradio` package, which self-shadows `import gradio` when run as `python frontend/gradio.py` directly (the script's own directory lands first on `sys.path`). Run it as `python -m frontend.gradio` from the project root instead (needs `frontend/__init__.py`, added 2026-10-08).

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
