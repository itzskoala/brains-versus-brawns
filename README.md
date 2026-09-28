# UFC Elo

Machine learning project for modeling UFC fight outcomes with Elo-based ratings.

Project scope, requirements, and deliverables are defined in [`docs/scoping.md`](docs/scoping.md).

## Project Structure

```text
ufc-elo/
├── src/
│   ├── data/            # Ingestion, validation, splitting
│   ├── features/        # Feature engineering, shared train/serve transforms
│   ├── models/           # Model definitions, training loops
│   ├── evaluation/       # Metrics, slice analysis, error analysis
│   └── monitoring/       # Drift detection, logging
├── configs/             # Hyperparameters and experiment configs (YAML)
├── notebooks/           # EDA only; logic lives in src/
├── api/                 # Serving application
├── frontend/            # UI code
├── tests/               # Unit tests, data validation tests
├── data/                # Sample data or DVC pointers (no raw data in git)
├── models/              # Registry pointers or small artifacts
├── deployment/          # Docker, cloud config
├── docs/                # Architecture diagram, decision log, model card
├── .env.example
├── requirements.txt
├── Dockerfile
└── README.md
```

## Data Pipeline

`src/data/cleaning.py` cleans the raw fight- and round-level CSVs (missing values, dtypes), `src/data/splitting.py` splits fights chronologically for training, and `src/features/engineering.py` builds leakage-safe matchup and fighter-history features - everything is computed from data known before the fight being predicted. EDA lives in `notebooks/` (`eda.ipynb`, `univariate_questions.ipynb`, `bivariate_questions.ipynb`, `multivariate_questions.ipynb`), including a documented red-corner recording bias in the source data that any model built on this dataset needs to account for.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Testing

```bash
pytest tests/
```
