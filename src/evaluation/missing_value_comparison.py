"""Compare the original zero-fill baseline against the leakage-safe
median+missingness-indicator alternative in src.features.imputation,
across all three models, on identical walk-forward folds and the
diffs-only DIFF_FEATURE_COLUMNS (src.models.logistic_regression) - the
feature set production used at the time this comparison was first run
(2026-10).

Historical note: this comparison is what led to adopting median+indicator
imputation as the new default in logistic_regression.py/random_forest.py/
xgboost_model.py/nested_validation.py (see docs/results.md) - so
"zero_fill" here no longer matches current production, which now uses the
imputer by default. Kept using DIFF_FEATURE_COLUMNS specifically (not
FEATURE_COLUMNS, which is now the combined representation) so this
module's already-reported numbers stay reproducible on the exact feature
set they were computed on.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

from src.data.splitting import walk_forward_splits
from src.features.engineering import engineer_fold_features
from src.features.imputation import apply_imputer, fit_imputer, missing_indicator_columns
from src.features.scaling import apply_scaler, fit_scaler
from src.models.logistic_regression import DIFF_FEATURE_COLUMNS as FEATURE_COLUMNS
from src.models.logistic_regression import _symmetrize

MODEL_CONFIGS = {
    "logistic_regression": {
        "build": lambda: LogisticRegression(C=1.0, max_iter=1000),
        "scale": True,
    },
    "random_forest": {
        "build": lambda: RandomForestClassifier(n_estimators=200, max_depth=None, random_state=0),
        "scale": False,
    },
    "xgboost_model": {
        "build": lambda: XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=0),
        "scale": False,
    },
}

STRATEGIES = ["zero_fill", "median_indicator"]


def run_backtest(fights, model_name, strategy, rounds=None, freq="Y", min_train_size=500):
    """Walk-forward backtest of model_name on FEATURE_COLUMNS (unchanged -
    no feature definitions touched), identical folds/labels/symmetrization
    as production, differing only in how missing values are handled:
    - "zero_fill": .fillna(0), byte-for-byte what every production
      run_backtest does today.
    - "median_indicator": src.features.imputation, fit on each fold's
      train rows only, applied unchanged to that fold's test rows.
    Returns a DataFrame of per-fold metrics."""
    config = MODEL_CONFIGS[model_name]
    results = []

    for period, train, test in walk_forward_splits(fights, freq=freq, min_train_size=min_train_size):
        train, test = engineer_fold_features(train, test, rounds=rounds)
        train = _symmetrize(train, FEATURE_COLUMNS)
        test = _symmetrize(test, FEATURE_COLUMNS)

        y_train, y_test = train["label"], test["label"]

        if strategy == "zero_fill":
            X_train = train[FEATURE_COLUMNS].fillna(0)
            X_test = test[FEATURE_COLUMNS].fillna(0)
            model_feature_columns = FEATURE_COLUMNS
        elif strategy == "median_indicator":
            imputer = fit_imputer(train, FEATURE_COLUMNS)
            train_imputed = apply_imputer(train, FEATURE_COLUMNS, imputer)
            test_imputed = apply_imputer(test, FEATURE_COLUMNS, imputer)
            model_feature_columns = FEATURE_COLUMNS + missing_indicator_columns(imputer)
            X_train = train_imputed[model_feature_columns]
            X_test = test_imputed[model_feature_columns]
        else:
            raise ValueError(f"unknown strategy {strategy!r}")

        if config["scale"]:
            scaler = fit_scaler(X_train, model_feature_columns)
            X_train = apply_scaler(X_train, model_feature_columns, scaler)
            X_test = apply_scaler(X_test, model_feature_columns, scaler)

        model = config["build"]()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        results.append({
            "period": str(period),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": len(model_feature_columns),
            "accuracy": accuracy_score(y_test, preds),
            "log_loss": log_loss(y_test, probs, labels=[0, 1]),
            "roc_auc": roc_auc_score(y_test, probs),
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    from src.data.cleaning import clean_fights
    from src.data.ingestion import build_master_equivalent, load_fighters, load_fights

    master = build_master_equivalent(load_fights(), load_fighters())
    fights = clean_fights(master)

    rows = []
    for model_name in MODEL_CONFIGS:
        for strategy in STRATEGIES:
            print(f"[missing_value_comparison] running {model_name} / {strategy}")
            metrics = run_backtest(fights, model_name, strategy)
            rows.append({
                "model": model_name,
                "strategy": strategy,
                "n_features": metrics["n_features"].iloc[0],
                "mean_accuracy": metrics["accuracy"].mean(),
                "mean_log_loss": metrics["log_loss"].mean(),
                "mean_roc_auc": metrics["roc_auc"].mean(),
            })

    summary = pd.DataFrame(rows)
    pivot = summary.pivot(index="model", columns="strategy",
                           values=["mean_accuracy", "mean_log_loss", "mean_roc_auc"])
    print()
    print(summary.to_string(index=False))
    print()
    print(pivot.to_string())
