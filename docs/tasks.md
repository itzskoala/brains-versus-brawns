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

* [ ] Identify useful features from the available dataset
* [ ] Implement fighter-level features
* [ ] Implement matchup-level features
* [ ] Implement historical/rolling features where required
* [ ] Implement shared feature transformations
* [ ] Add feature validation tests
* [ ] Document important feature decisions

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

* [ ] Establish a simple prediction baseline
* [ ] Create the training pipeline
* [ ] Train an initial ML model
* [ ] Establish evaluation metrics
* [ ] Save model artifacts appropriately
* [ ] Record baseline results in `results.md`

---

# Phase 6 — Model Development

* [ ] Test additional model types
* [ ] Compare feature sets
* [ ] Experiment with Elo + ML
* [ ] Tune important hyperparameters where justified
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
