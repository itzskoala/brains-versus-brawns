"""Compare three feature representations - corner differences, individual
per-corner values, and a combination of both - across all three production
models, on identical walk-forward folds, eligible fights, and labels.

Reuses src.data.splitting.walk_forward_splits, src.features.engineering's
engineer_fold_features, and src.models.logistic_regression's
_symmetrize/DIFF_FEATURE_COLUMNS/INDIVIDUAL_FEATURE_COLUMNS/
COMBINED_FEATURE_COLUMNS unchanged - those are the canonical definitions
production now uses by default (FEATURE_COLUMNS == COMBINED_FEATURE_COLUMNS,
promoted after this comparison first showed plain diffs were the worst of
the three on every metric, for every model - see docs/results.md). Only
each model's own default hyperparameters are duplicated here (as plain
sklearn/xgboost constructors), so this file has no production side effects.
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
from src.models.logistic_regression import (
    COMBINED_FEATURE_COLUMNS,
    DIFF_FEATURE_COLUMNS,
    INDIVIDUAL_FEATURE_COLUMNS,
    _symmetrize,
)

REPRESENTATIONS = {
    "diffs": DIFF_FEATURE_COLUMNS,
    "individual": INDIVIDUAL_FEATURE_COLUMNS,
    "combined": COMBINED_FEATURE_COLUMNS,
}

# Each model's own default configuration, copied from its
# src.models.<name>.run_backtest default arguments - same classifier, same
# hyperparameters, same scale-or-not choice as the production backtests.
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


def run_backtest(fights, model_name, representation, rounds=None, freq="Y", min_train_size=500):
    """Walk-forward backtest of model_name using the named representation's
    feature set - identical folds (walk_forward_splits), eligible fights,
    engineered features (engineer_fold_features), symmetrization
    (_symmetrize - same function production uses), and label definition as
    every production run_backtest; only the feature set differs. Missing
    values are handled by src.features.imputation (median + missingness
    indicator, fit on each fold's train rows only) for all three
    representations, so the comparison isn't skewed by zero-fill's bias
    (see src.evaluation.missing_value_comparison). The scaler (when the
    model needs one) is fit on each fold's train rows only, same as
    production. Returns a DataFrame of per-fold metrics."""
    feature_columns = REPRESENTATIONS[representation]
    config = MODEL_CONFIGS[model_name]
    results = []

    for period, train, test in walk_forward_splits(fights, freq=freq, min_train_size=min_train_size):
        train, test = engineer_fold_features(train, test, rounds=rounds)
        train = _symmetrize(train, feature_columns)
        test = _symmetrize(test, feature_columns)

        y_train, y_test = train["label"], test["label"]

        imputer = fit_imputer(train, feature_columns)
        train_imputed = apply_imputer(train, feature_columns, imputer)
        test_imputed = apply_imputer(test, feature_columns, imputer)
        model_feature_columns = feature_columns + missing_indicator_columns(imputer)
        X_train = train_imputed[model_feature_columns]
        X_test = test_imputed[model_feature_columns]

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


def run_comparison(fights, rounds=None):
    """Run all 3 models x 3 representations on identical folds. Returns a
    dict {(model_name, representation): per-fold metrics DataFrame}."""
    comparison = {}
    for model_name in MODEL_CONFIGS:
        for representation in REPRESENTATIONS:
            print(f"[representation_comparison] running {model_name} / {representation} "
                  f"(n_features={len(REPRESENTATIONS[representation])})")
            comparison[(model_name, representation)] = run_backtest(
                fights, model_name, representation, rounds=rounds)
    return comparison


if __name__ == "__main__":
    from src.data.cleaning import clean_fights
    from src.data.ingestion import build_master_equivalent, load_fighters, load_fights

    master = build_master_equivalent(load_fights(), load_fighters())
    fights = clean_fights(master)

    comparison = run_comparison(fights)

    rows = []
    for (model_name, representation), metrics in comparison.items():
        rows.append({
            "model": model_name,
            "representation": representation,
            "n_base_features": len(REPRESENTATIONS[representation]),
            "n_features_with_indicators": metrics["n_features"].mean(),
            "mean_accuracy": metrics["accuracy"].mean(),
            "mean_log_loss": metrics["log_loss"].mean(),
            "mean_roc_auc": metrics["roc_auc"].mean(),
            "std_roc_auc": metrics["roc_auc"].std(),
        })
    summary = pd.DataFrame(rows).sort_values(["model", "representation"])
    print()
    print(summary.to_string(index=False))

    summary.to_csv("docs/representation_comparison_summary.csv", index=False)
    for (model_name, representation), metrics in comparison.items():
        metrics.to_csv(f"docs/representation_comparison_{model_name}_{representation}_folds.csv", index=False)
    print("\nwrote docs/representation_comparison_summary.csv and per-(model,representation) fold csvs")
