import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_auc_score

from src.data.splitting import walk_forward_splits
from src.features.engineering import engineer_fold_features
from src.features.imputation import apply_imputer, fit_imputer, missing_indicator_columns
from src.models.logistic_regression import FEATURE_COLUMNS, _symmetrize

# n_estimators x max_depth grid - the two knobs most likely to matter for a
# forest at this dataset size; kept small (6 candidates) since each one
# gets refit across every inner fold during nested validation.
PARAM_GRID = [{"n_estimators": n, "max_depth": d} for n in (100, 200) for d in (3, 6, None)]
SCALE = False  # tree splits use raw thresholds, not a scaled distance


def build_model(**params):
    """Construct a fresh RandomForestClassifier for one hyperparameter
    candidate - used by src.evaluation.nested_validation so the same
    nested walk-forward tuning code works for every model in src/models/."""
    return RandomForestClassifier(random_state=0, **params)


def run_backtest(fights, rounds=None, freq="Y", min_train_size=500, feature_columns=None,
                  n_estimators=200, max_depth=None, random_state=0, return_predictions=False):
    """Walk-forward backtest of a random forest classifier predicting fight
    outcomes - same folds, features, corner-symmetrization, and missing-value
    imputation (src.features.imputation, fit on each fold's train rows only)
    as src.models.logistic_regression.run_backtest, swapping logistic
    regression for a random forest so the two are comparable on identical
    folds and features. Trees split on raw thresholds rather than a scaled
    distance, so there's no scaler step here. Returns a DataFrame of
    per-fold metrics; if return_predictions is True, also returns a long
    DataFrame of every test-fold row's own period/fight_id/y_true/y_pred/
    y_prob as a second return value - see
    src.models.logistic_regression.run_backtest's docstring."""
    feature_columns = feature_columns or FEATURE_COLUMNS
    results = []
    predictions = []

    print(f"[random_forest] starting walk-forward backtest: freq={freq}, min_train_size={min_train_size}, "
          f"n_estimators={n_estimators}, max_depth={max_depth}, n_features={len(feature_columns)}")

    for period, train, test in walk_forward_splits(fights, freq=freq, min_train_size=min_train_size):
        train, test = engineer_fold_features(train, test, rounds=rounds)
        train = _symmetrize(train, feature_columns)
        test = _symmetrize(test, feature_columns)

        y_train = train["label"]
        y_test = test["label"]

        imputer = fit_imputer(train, feature_columns)
        train_imputed = apply_imputer(train, feature_columns, imputer)
        test_imputed = apply_imputer(test, feature_columns, imputer)
        fitted_feature_columns = feature_columns + missing_indicator_columns(imputer)
        X_train = train_imputed[fitted_feature_columns]
        X_test = test_imputed[fitted_feature_columns]

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        fold_accuracy = accuracy_score(y_test, preds)
        fold_log_loss = log_loss(y_test, probs, labels=[0, 1])
        fold_roc_auc = roc_auc_score(y_test, probs)
        fold_mse = mean_squared_error(y_test, probs)
        # var(y_test) is the MSE a model gets by always predicting the mean
        # of y_test - the textbook "worse than predicting the average"
        # baseline. ddof=0 so it's exactly that mean-prediction MSE, not the
        # sample-variance estimate (ddof=1).
        fold_target_variance = y_test.var(ddof=0)
        fold_beats_mean_baseline = bool(fold_mse < fold_target_variance)
        print(f"[random_forest] fold {period}: n_train={len(X_train)}, n_test={len(X_test)}, "
              f"accuracy={fold_accuracy:.4f}, log_loss={fold_log_loss:.4f}, roc_auc={fold_roc_auc:.4f}, "
              f"mse={fold_mse:.4f}, target_variance={fold_target_variance:.4f}, "
              f"beats_mean_baseline={fold_beats_mean_baseline}")

        results.append({
            "period": str(period),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "accuracy": fold_accuracy,
            "log_loss": fold_log_loss,
            "roc_auc": fold_roc_auc,
            "mse": fold_mse,
            "target_variance": fold_target_variance,
            "beats_mean_baseline": fold_beats_mean_baseline,
        })

        if return_predictions:
            predictions.append(pd.DataFrame({
                "period": str(period),
                "fight_id": test["fight_id"].values,
                "y_true": y_test.values,
                "y_pred": preds,
                "y_prob": probs,
            }))

    print(f"[random_forest] backtest complete: {len(results)} folds")
    metrics = pd.DataFrame(results)
    if return_predictions:
        predictions_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(
            columns=["period", "fight_id", "y_true", "y_pred", "y_prob"])
        return metrics, predictions_df
    return metrics


if __name__ == "__main__":
    from src.data.cleaning import clean_fights
    from src.data.ingestion import build_master_equivalent, load_fighters, load_fights

    print("[random_forest] loading data/fights.csv + data/fighter.csv")
    master = build_master_equivalent(load_fights(), load_fighters())
    print(f"[random_forest] loaded {len(master)} raw rows; cleaning fights")
    fights = clean_fights(master)
    print(f"[random_forest] {len(fights)} fights after cleaning; running backtest")

    metrics = run_backtest(fights)
    print(metrics.to_string(index=False))
    print("\nmean accuracy:", metrics["accuracy"].mean())
    print("mean log_loss:", metrics["log_loss"].mean())
    print("mean roc_auc:", metrics["roc_auc"].mean())
    print("mean mse:", metrics["mse"].mean())
    print("mean target_variance:", metrics["target_variance"].mean())
    print("folds beating mean-prediction baseline:",
          f"{metrics['beats_mean_baseline'].sum()}/{len(metrics)}")
