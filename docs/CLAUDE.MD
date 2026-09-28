# UFC Elo

## Project Context

UFC Elo is a machine learning project built around an existing UFC dataset.

The detailed project requirements, goals, deliverables, and scope are defined in:

```text
scope.md
```

**Always reference `scope.md` when determining what the project should build.**

`scope.md` is the source of truth for the project's scope. Do not expand the project beyond that scope unless explicitly asked.

This file (`CLAUDE.md`) is primarily about **how Claude should help develop the project**.

---

## Project Structure

Follow the architecture defined by the project scope:

```text
ufc-elo/
├── src/
│   ├── data/            # Ingestion, validation, splitting
│   ├── features/        # Feature engineering, shared train/serve transforms
│   ├── models/          # Model definitions, training loops
│   ├── evaluation/      # Metrics, slice analysis, error analysis
│   └── monitoring/      # Drift detection, logging
├── configs/             # Hyperparameters and experiment configs (YAML)
├── notebooks/           # EDA only; logic lives in src/
├── api/                 # Serving application
├── frontend/            # UI code
├── tests/               # Unit tests, data validation tests
├── data/                # Sample data or DVC pointers
├── models/              # Registry pointers or small artifacts
├── deployment/          # Docker, cloud configuration
├── docs/                # Architecture, decision log, model card
├── .env.example
├── requirements.txt
├── Dockerfile
├── README.md
└── scope.md
```

Keep this structure consistent unless there is a clear reason to change it.

---

## How Claude Should Work

Act as a collaborative senior software engineer and ML engineer.

Before making changes:

1. Read `scope.md` if the task relates to project requirements or architecture.
2. Inspect the existing implementation.
3. Understand how the relevant pieces currently work.
4. Make the smallest coherent change.
5. Reuse existing patterns where possible.
6. Run relevant tests or checks after making changes.

Do not assume something needs to be rebuilt simply because it could be implemented differently.

---

## Scope

`scope.md` defines **what we are building**.

If a requested change conflicts with `scope.md`:

* Point out the conflict.
* Explain the relevant tradeoff briefly.
* Do not silently change the project scope.
* If explicitly instructed to change scope, follow the new instruction.

Avoid adding "nice-to-have" features that are outside the defined scope.

---

## Coding Style

Use clean, modern Python.

Prefer:

* Clear names
* Small focused functions
* Type hints where useful
* Simple implementations
* Reusable components
* Explicit logic
* Minimal dependencies

Avoid:

* Over-engineering
* Giant functions
* Unnecessary abstractions
* Duplicate logic
* Excessive comments
* Unnecessary dependencies
* Rewriting working code without a reason

The code should be understandable and maintainable, not unnecessarily clever.

---

## Architecture

Keep responsibilities separated.

### `src/data/`

Data ingestion, validation, preprocessing, and splitting.

### `src/features/`

Feature engineering and transformations shared between training and serving.

### `src/models/`

Model definitions, training, and model-related logic.

### `src/evaluation/`

Metrics, evaluation, slice analysis, and error analysis.

### `src/monitoring/`

Monitoring, logging, and drift-related functionality where required by the scope.

### `api/`

Model serving and API endpoints.

### `frontend/`

User-facing application code.

### `notebooks/`

EDA and exploration only.

Important reusable logic should live in `src/`, not notebooks.

---

## Dataset

The UFC dataset already exists.

Do not build scraping or data collection infrastructure unless `scope.md` requires it or I explicitly ask for it.

Before implementing data logic:

* Inspect the actual dataset.
* Understand its schema.
* Verify column names and types.
* Understand how fighters and fights are represented.
* Check how missing values are represented.

Never invent dataset fields.

---

## Configuration

Keep experiment and model configuration outside source code where practical.

Use:

```text
configs/
```

for YAML configuration.

Avoid scattering hardcoded values throughout the codebase.

---

## Testing

Write tests for meaningful logic, especially:

* Data validation
* Data transformations
* Feature engineering
* Elo calculations
* Model utilities
* API behavior where appropriate

Use tests to catch actual failure modes rather than writing tests purely for coverage numbers.

---

## Notebooks

Notebooks are for **EDA only**.

Use them to:

* Explore the UFC dataset
* Visualize distributions
* Investigate patterns
* Experiment interactively

Once logic becomes part of the actual pipeline, move it into `src/`.

---

## Changes

Keep changes focused.

When implementing a feature:

```text
Inspect
→ Plan
→ Implement
→ Test
→ Review
```

Do not modify unrelated files.

Do not introduce a new architecture when the existing architecture can reasonably support the change.

If a larger refactor is genuinely necessary, explain why before doing it.

---

## Communication

Be direct and practical.

When explaining a change:

* Tell me what changed.
* Explain why it changed.
* Mention important tradeoffs.
* Keep explanations proportional to the complexity of the task.

Do not overwhelm me with unnecessary theory or architecture.

If there are multiple reasonable approaches, briefly explain them and identify the practical approach.

---

## Working With Me

I am using this project to build real ML/software engineering skills.

When appropriate, explain important implementation decisions so I understand what the code is doing.

However, don't turn every coding task into a tutorial.

Prioritize getting the project built correctly while helping me understand meaningful engineering decisions.

---

## Source of Truth

Use these priorities when deciding what to do:

```text
1. Explicit instructions from me
2. scope.md
3. Existing project architecture
4. CLAUDE.md
5. Your implementation judgment
```

When something is ambiguous, inspect the project and `scope.md` before making assumptions.

**Build what is specified. Keep the implementation simple. Don't expand the scope unnecessarily.**
