import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_auc_score

from src.data.splitting import walk_forward_splits
from src.features.engineering import engineer_fold_features
from src.features.imputation import apply_imputer, fit_imputer, missing_indicator_columns
from src.features.scaling import apply_scaler, fit_scaler
from src.models.logistic_regression import _symmetrize


def _fit_and_score(train, test, feature_columns, build_model, params, scale):
    """Fit one model on train, score it on test. Shared by the inner
    hyperparameter search and the outer final fit so both go through the
    same impute-then-scale-then-fit-then-predict steps. Missing values are
    imputed (src.features.imputation: training-fold median plus a
    missingness indicator) fit on train only, same as every
    src.models.*.run_backtest."""
    imputer = fit_imputer(train, feature_columns)
    train_imputed = apply_imputer(train, feature_columns, imputer)
    test_imputed = apply_imputer(test, feature_columns, imputer)
    fitted_feature_columns = feature_columns + missing_indicator_columns(imputer)
    X_train = train_imputed[fitted_feature_columns]
    X_test = test_imputed[fitted_feature_columns]

    if scale:
        scaler = fit_scaler(X_train, fitted_feature_columns)
        X_train = apply_scaler(X_train, fitted_feature_columns, scaler)
        X_test = apply_scaler(X_test, fitted_feature_columns, scaler)

    model = build_model(**params)
    model.fit(X_train, train["label"])

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    mse = mean_squared_error(test["label"], probs)
    # var(y_test) is the MSE a model gets by always predicting the mean of
    # y_test (ddof=0, so it's exactly that mean-prediction MSE).
    target_variance = test["label"].var(ddof=0)
    return {
        "accuracy": accuracy_score(test["label"], preds),
        "log_loss": log_loss(test["label"], probs, labels=[0, 1]),
        "roc_auc": roc_auc_score(test["label"], probs),
        "mse": mse,
        "target_variance": target_variance,
        "beats_mean_baseline": bool(mse < target_variance),
    }


def select_hyperparameters(train_fights, build_model, param_grid, feature_columns, rounds=None,
                            scale=False, freq="Y", min_train_size=250):
    """Pick the best-scoring hyperparameters using an INNER walk-forward
    split generated entirely from train_fights. The outer fold's test year
    is never passed into this function - it is not in train_fights at all
    - so it cannot influence which candidate wins.

    Every param_grid candidate is scored by its mean accuracy across every
    inner (inner_train, inner_test) fold, using grid search: param_grid is
    small (1-3 hyperparameters, a handful of values each) for all three
    models used in this project, so an exhaustive, transparent search over
    named candidates is simpler and more inspectable than random or
    Bayesian search, and needs no extra dependency.

    Inner folds are engineered once and reused across every candidate,
    since feature engineering doesn't depend on the hyperparameters being
    tested - recomputing it per-candidate would be pure waste.

    min_train_size defaults lower than the outer loop's, since an outer
    fold's own training window is often much smaller than the full
    dataset (e.g. selecting hyperparameters for a 2020 prediction only has
    2003-2019 to search within). If train_fights is too small to produce
    even one inner fold, every candidate scores as unusable and the first
    candidate in param_grid is returned - callers should treat n_inner_folds
    == 0 in the returned scores as "this choice wasn't actually validated".

    Returns (best_params, scores_df) where scores_df has one row per
    candidate with its mean_accuracy, mean_log_loss, and n_inner_folds."""
    inner_folds = []
    for _, inner_train, inner_test in walk_forward_splits(train_fights, freq=freq, min_train_size=min_train_size):
        inner_train, inner_test = engineer_fold_features(inner_train, inner_test, rounds=rounds)
        inner_train = _symmetrize(inner_train, feature_columns)
        inner_test = _symmetrize(inner_test, feature_columns)
        inner_folds.append((inner_train, inner_test))

    rows = []
    for params in param_grid:
        if not inner_folds:
            rows.append({"params": params, "mean_accuracy": float("-inf"),
                         "mean_log_loss": float("inf"), "mean_roc_auc": float("-inf"),
                         "mean_mse": float("inf"), "mean_target_variance": float("nan"),
                         "n_inner_folds": 0})
            continue

        fold_scores = pd.DataFrame(
            _fit_and_score(tr, te, feature_columns, build_model, params, scale) for tr, te in inner_folds)
        rows.append({
            "params": params,
            "mean_accuracy": fold_scores["accuracy"].mean(),
            "mean_log_loss": fold_scores["log_loss"].mean(),
            "mean_roc_auc": fold_scores["roc_auc"].mean(),
            "mean_mse": fold_scores["mse"].mean(),
            "mean_target_variance": fold_scores["target_variance"].mean(),
            "n_inner_folds": len(fold_scores),
        })

    scores_df = pd.DataFrame(rows)
    best_params = scores_df.loc[scores_df["mean_accuracy"].idxmax(), "params"]
    return best_params, scores_df


def run_nested_backtest(fights, build_model, param_grid, feature_columns, rounds=None, scale=False,
                         freq="Y", min_train_size=500, inner_min_train_size=250):
    """Nested walk-forward backtest: for each outer fold, hyperparameters
    are chosen by select_hyperparameters using only that fold's own
    training window - never the outer test year - then a final model is
    trained on the FULL outer training window with those hyperparameters
    and scored once on the outer test year. Inner temporal validation for
    model selection, outer temporal validation for the final performance
    estimate - the outer test year is only ever touched by that single
    final prediction. Returns a DataFrame of per-outer-fold metrics plus
    the chosen hyperparameters and how many inner folds backed that
    choice."""
    results = []

    for period, train, test in walk_forward_splits(fights, freq=freq, min_train_size=min_train_size):
        best_params, inner_scores = select_hyperparameters(
            train, build_model, param_grid, feature_columns, rounds=rounds, scale=scale,
            freq=freq, min_train_size=inner_min_train_size)

        train_f, test_f = engineer_fold_features(train, test, rounds=rounds)
        train_f = _symmetrize(train_f, feature_columns)
        test_f = _symmetrize(test_f, feature_columns)

        outer_score = _fit_and_score(train_f, test_f, feature_columns, build_model, best_params, scale)

        results.append({
            "period": str(period),
            "n_train": len(train_f),
            "n_test": len(test_f),
            "n_inner_folds": int(inner_scores["n_inner_folds"].iloc[0]) if len(inner_scores) else 0,
            "chosen_params": best_params,
            **outer_score,
        })

    return pd.DataFrame(results)
