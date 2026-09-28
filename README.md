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
