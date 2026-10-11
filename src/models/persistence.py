import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.features.engineering import engineer_fold_features
from src.features.imputation import apply_imputer, fit_imputer, missing_indicator_columns
from src.features.scaling import apply_scaler, fit_scaler
from src.models.logistic_regression import _symmetrize


def fit_final_model(fights, build_model, params, feature_columns, scale):
    """Fit one model on the full fights history with the given hyperparameters
    and feature set - the "deployable" fit, as opposed to a walk-forward
    backtest fold. Uses the same feature engineering, corner-symmetrization,
    and missing-value imputation (src.features.imputation - fit on this
    full history, since there is no held-out split here to leak into) as
    every backtest. Returns (model, scaler, imputer, fitted_feature_columns)
    - scaler is None when scale is False; fitted_feature_columns is
    feature_columns plus the missingness indicators the imputer added, in
    the exact order the model was fit on (serving must rebuild X in this
    same order)."""
    empty_test = fights.iloc[0:0]
    train, _ = engineer_fold_features(fights, empty_test)
    train = _symmetrize(train, feature_columns)

    imputer = fit_imputer(train, feature_columns)
    train_imputed = apply_imputer(train, feature_columns, imputer)
    fitted_feature_columns = feature_columns + missing_indicator_columns(imputer)
    X_train = train_imputed[fitted_feature_columns]
    y_train = train["label"]

    scaler = None
    if scale:
        scaler = fit_scaler(X_train, fitted_feature_columns)
        X_train = apply_scaler(X_train, fitted_feature_columns, scaler)

    model = build_model(**params)
    model.fit(X_train, y_train)
    return model, scaler, imputer, fitted_feature_columns


def save_model_artifact(model, scaler, imputer, feature_columns, fitted_feature_columns,
                         params, metrics, model_name, out_dir="models"):
    """Persist a fitted model (plus its scaler and imputer, if any) to
    out_dir/model_name.joblib, with a sidecar out_dir/model_name.json
    recording the feature set, hyperparameters, and evaluation metrics it
    was selected with - so a saved artifact is never just a bare pickle
    with no record of what it is or how it was validated.

    feature_columns is the base set snapshot.build_matchup_features needs
    to construct from two fighter snapshots; fitted_feature_columns is that
    plus the imputer's missingness indicators, in the exact column order
    the model expects - serving needs both."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump({"model": model, "scaler": scaler, "imputer": imputer}, out_dir / f"{model_name}.joblib")

    metadata = {
        "model_name": model_name,
        "feature_columns": feature_columns,
        "fitted_feature_columns": fitted_feature_columns,
        "params": params,
        "metrics": metrics,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_dir / f"{model_name}.json", "w") as f:
        json.dump(metadata, f, indent=2)


def load_model_artifact(model_name, out_dir="models"):
    """Load a model artifact saved by save_model_artifact. Returns
    (model, scaler, imputer, metadata). imputer is None for an artifact
    saved before src.features.imputation existed - callers must fall back
    to zero-fill in that case (see src.features.snapshot.build_matchup_features)."""
    out_dir = Path(out_dir)
    bundle = joblib.load(out_dir / f"{model_name}.joblib")
    with open(out_dir / f"{model_name}.json") as f:
        metadata = json.load(f)
    return bundle["model"], bundle["scaler"], bundle.get("imputer"), metadata


if __name__ == "__main__":
    # Re-fits and re-saves all three deployed artifacts on the new combined
    # (individual + diff) feature representation and the median+indicator
    # imputer (2026-10), keeping each artifact's own previously-saved
    # hyperparameters exactly as-is - this refreshes the feature
    # set/preprocessing, not the hyperparameter choice (that's still the
    # separate, not-yet-complete nested-tuning follow-up in docs/tasks.md).
    from src.data.cleaning import clean_fights
    from src.data.ingestion import build_master_equivalent, load_fighters, load_fights
    from src.models import logistic_regression, random_forest, xgboost_model
    from src.models.logistic_regression import FEATURE_COLUMNS

    master = build_master_equivalent(load_fights(), load_fighters())
    fights = clean_fights(master)

    MODELS = {
        "logistic_regression": {
            "module": logistic_regression,
            "build_model": logistic_regression.build_model,
            "scale": logistic_regression.SCALE,
        },
        "random_forest": {
            "module": random_forest,
            "build_model": random_forest.build_model,
            "scale": random_forest.SCALE,
        },
        "xgboost_model": {
            "module": xgboost_model,
            "build_model": xgboost_model.build_model,
            "scale": xgboost_model.SCALE,
        },
    }

    for model_name, spec in MODELS.items():
        with open(f"models/{model_name}.json") as f:
            previous_metadata = json.load(f)
        params = previous_metadata["params"]

        print(f"[persistence] refitting {model_name} with preserved params {params} "
              f"on {len(FEATURE_COLUMNS)} combined features")
        backtest_metrics = spec["module"].run_backtest(fights, feature_columns=FEATURE_COLUMNS, **params)

        model, scaler, imputer, fitted_feature_columns = fit_final_model(
            fights, spec["build_model"], params, FEATURE_COLUMNS, spec["scale"])

        save_model_artifact(
            model, scaler, imputer, FEATURE_COLUMNS, fitted_feature_columns, params,
            metrics={
                "mean_accuracy": round(float(backtest_metrics["accuracy"].mean()), 4),
                "mean_log_loss": round(float(backtest_metrics["log_loss"].mean()), 4),
                "mean_roc_auc": round(float(backtest_metrics["roc_auc"].mean()), 4),
                "source": (
                    "walk-forward run_backtest, combined(%d) feature set, "
                    "median+indicator imputer, 2026-10-10, hyperparameters "
                    "preserved from prior artifact" % len(FEATURE_COLUMNS)
                ),
            },
            model_name=model_name,
        )
        print(f"[persistence] saved models/{model_name}.joblib + .json "
              f"({len(fitted_feature_columns)} fitted features)")
