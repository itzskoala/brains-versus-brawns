# UFC Elo — Project Proposal

## Overview

UFC Elo is a machine learning project that uses an existing UFC dataset to build a fighter Elo rating system and explore machine learning approaches for analyzing and predicting UFC fight outcomes.

The project is intended to be a practical end-to-end ML system rather than a single modeling experiment.

The project scope is defined in `scope.md`. This proposal describes the intended approach for completing that scope.

---

## Objectives

The project will:

1. Understand and validate the existing UFC dataset.
2. Build a reliable data processing pipeline.
3. Create meaningful fighter and matchup features.
4. Implement a UFC-specific Elo rating system.
5. Train and evaluate ML models.
6. Compare model performance against appropriate baselines.
7. Analyze model errors and performance across different situations.
8. Provide a way to serve predictions through an API.
9. Build the supporting frontend if required by `scope.md`.
10. Document important technical and modeling decisions.

---

## Proposed Architecture

The project will follow a modular architecture:

```text
Raw / Existing Dataset
        ↓
Data Validation
        ↓
Data Processing
        ↓
Feature Engineering
        ↓
Elo Ratings
        ↓
Model Training
        ↓
Evaluation
        ↓
Model Registry / Artifacts
        ↓
API
        ↓
Frontend
```

Each stage should have a clear responsibility.

Training and serving should share the same feature transformations wherever possible to reduce inconsistencies between development and production.

---

## Elo System

The first version should implement a straightforward Elo rating system.

The Elo implementation should be:

* Configurable
* Testable
* Reproducible
* Independent from the ML models

Once a reliable baseline exists, modifications to the Elo methodology can be tested as separate experiments.

The project should avoid prematurely creating a highly complicated rating system.

---

## Machine Learning

The ML component should begin with simple baselines before progressing to more sophisticated models.

A reasonable progression is:

```text
Baseline
    ↓
Logistic Regression
    ↓
Tree-Based Models
    ↓
Advanced Models
```

The exact models should be determined by the available data and project scope.

Model complexity should be justified by measurable improvement rather than added for its own sake.

---

## Feature Engineering

Features should describe fighters and the matchup in a way that is useful for prediction.

Potential feature categories include:

* Fighter attributes
* Historical performance
* Elo ratings
* Recent performance
* Experience
* Striking
* Grappling
* Physical differences
* Matchup characteristics

The exact feature set should be determined from the available dataset and the requirements in `scope.md`.

Feature engineering should live in `src/features/`.

---

## Evaluation

Models should be evaluated systematically rather than relying on a single metric.

Evaluation should include appropriate classification and probability metrics, along with error analysis and relevant performance slices.

The evaluation process should make it possible to answer:

* Does the model actually outperform simple baselines?
* Where does the model perform well?
* Where does it fail?
* Which features appear useful?
* Does additional complexity provide meaningful improvement?

Results should be recorded in `results.md`.

---

## Experimentation

Experiments should be reproducible.

Important configuration should be stored in `configs/`.

Each meaningful experiment should record:

* Model
* Features
* Configuration
* Dataset/version
* Evaluation period
* Metrics
* Observations
* Decision

Avoid making undocumented changes that make experiments difficult to reproduce.

---

## Deployment

If required by `scope.md`, the final system should expose the trained model through the API layer.

The API should use the same feature-processing logic as training wherever possible.

The frontend should consume the API rather than duplicating model logic.

Deployment configuration belongs in `deployment/`.

---

## Recommendations

During development, Claude should make recommendations when there is a meaningful engineering or modeling decision to make.

Recommendations should consider:

* Project scope
* Simplicity
* Maintainability
* Data availability
* Model performance
* Reproducibility
* Deployment requirements

Recommendations should not automatically become implementation tasks.

Record significant recommendations and whether they were accepted or rejected in `results.md`.

---

## Expected Outcome

The completed project should provide:

* A validated UFC data pipeline
* A functioning Elo system
* Reusable feature engineering
* Trained ML models
* Meaningful evaluation
* Documented experiments
* A reproducible workflow
* API/model serving where required
* A frontend where required
* Clear documentation of results and decisions

The final system should be understandable, reproducible, and maintainable.
