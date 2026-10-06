import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.features.engineering import engineer_fold_features
from src.features.scaling import apply_scaler, fit_scaler
from src.models.logistic_regression import _symmetrize


def fit_final_model(fights, build_model, params, feature_columns, scale):
    """Fit one model on the full fights history with the given hyperparameters
    and feature set - the "deployable" fit, as opposed to a walk-forward
    backtest fold. Uses the same feature engineering and corner-symmetrization
    as every backtest, just with no held-out test period, since a final
    artifact should use all available history. Returns (model, scaler) -
    scaler is None when scale is False."""
    empty_test = fights.iloc[0:0]
    train, _ = engineer_fold_features(fights, empty_test)
    train = _symmetrize(train, feature_columns)

    X_train = train[feature_columns].fillna(0)
    y_train = train["label"]

    scaler = None
    if scale:
        scaler = fit_scaler(X_train, feature_columns)
        X_train = apply_scaler(X_train, feature_columns, scaler)

    model = build_model(**params)
    model.fit(X_train, y_train)
    return model, scaler


def save_model_artifact(model, scaler, feature_columns, params, metrics, model_name, out_dir="models"):
    """Persist a fitted model (plus its scaler, if any) to
    out_dir/model_name.joblib, with a sidecar out_dir/model_name.json
    recording the feature set, hyperparameters, and evaluation metrics it
    was selected with - so a saved artifact is never just a bare pickle
    with no record of what it is or how it was validated."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump({"model": model, "scaler": scaler}, out_dir / f"{model_name}.joblib")

    metadata = {
        "model_name": model_name,
        "feature_columns": feature_columns,
        "params": params,
        "metrics": metrics,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_dir / f"{model_name}.json", "w") as f:
        json.dump(metadata, f, indent=2)


def load_model_artifact(model_name, out_dir="models"):
    """Load a model artifact saved by save_model_artifact. Returns
    (model, scaler, metadata)."""
    out_dir = Path(out_dir)
    bundle = joblib.load(out_dir / f"{model_name}.joblib")
    with open(out_dir / f"{model_name}.json") as f:
        metadata = json.load(f)
    return bundle["model"], bundle["scaler"], metadata
