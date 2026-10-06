import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

from src.data.splitting import walk_forward_splits
from src.features.engineering import engineer_fold_features
from src.models.logistic_regression import FEATURE_COLUMNS, _symmetrize

# n_estimators x max_depth x learning_rate grid - kept small (8 candidates)
# since each one gets refit across every inner fold during nested validation.
PARAM_GRID = [
    {"n_estimators": n, "max_depth": d, "learning_rate": lr}
    for n in (100, 200) for d in (3, 5) for lr in (0.05, 0.1)
]
SCALE = False  # tree splits use raw thresholds, not a scaled distance


def build_model(**params):
    """Construct a fresh XGBClassifier for one hyperparameter candidate -
    used by src.evaluation.nested_validation so the same nested
    walk-forward tuning code works for every model in src/models/."""
    return XGBClassifier(random_state=0, **params)


def run_backtest(fights, rounds=None, freq="Y", min_train_size=500, feature_columns=None,
                  n_estimators=200, max_depth=3, learning_rate=0.1, random_state=0):
    """Walk-forward backtest of an XGBoost classifier predicting fight
    outcomes - same folds, features, and corner-symmetrization as
    src.models.logistic_regression.run_backtest, swapping logistic
    regression for gradient-boosted trees so the two are comparable on
    identical folds and features. Trees split on raw thresholds rather than
    a scaled distance, so there's no scaler step here. Returns a DataFrame
    of per-fold metrics."""
    feature_columns = feature_columns or FEATURE_COLUMNS
    results = []

    print(f"[xgboost_model] starting walk-forward backtest: freq={freq}, min_train_size={min_train_size}, "
          f"n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}, "
          f"n_features={len(feature_columns)}")

    for period, train, test in walk_forward_splits(fights, freq=freq, min_train_size=min_train_size):
        train, test = engineer_fold_features(train, test, rounds=rounds)
        train = _symmetrize(train, feature_columns)
        test = _symmetrize(test, feature_columns)

        y_train = train["label"]
        y_test = test["label"]

        X_train = train[feature_columns].fillna(0)
        X_test = test[feature_columns].fillna(0)

        model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                               learning_rate=learning_rate, random_state=random_state)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        fold_accuracy = accuracy_score(y_test, preds)
        fold_log_loss = log_loss(y_test, probs, labels=[0, 1])
        fold_roc_auc = roc_auc_score(y_test, probs)
        print(f"[xgboost_model] fold {period}: n_train={len(X_train)}, n_test={len(X_test)}, "
              f"accuracy={fold_accuracy:.4f}, log_loss={fold_log_loss:.4f}, roc_auc={fold_roc_auc:.4f}")

        results.append({
            "period": str(period),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "accuracy": fold_accuracy,
            "log_loss": fold_log_loss,
            "roc_auc": fold_roc_auc,
        })

    print(f"[xgboost_model] backtest complete: {len(results)} folds")
    return pd.DataFrame(results)


if __name__ == "__main__":
    from src.data.cleaning import clean_fights

    print("[xgboost_model] loading data/master.csv")
    master = pd.read_csv("data/master.csv")
    print(f"[xgboost_model] loaded {len(master)} raw rows; cleaning fights")
    fights = clean_fights(master)
    print(f"[xgboost_model] {len(fights)} fights after cleaning; running backtest")

    metrics = run_backtest(fights)
    print(metrics.to_string(index=False))
    print("\nmean accuracy:", metrics["accuracy"].mean())
    print("mean log_loss:", metrics["log_loss"].mean())
    print("mean roc_auc:", metrics["roc_auc"].mean())
