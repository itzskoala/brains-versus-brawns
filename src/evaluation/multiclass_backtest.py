"""Walk-forward backtest + final-fit, generalized from the binary winner
target (src.models.*.run_backtest) to a multiclass fight-outcome target -
currently "method_bucket" (win method: ko_tko/submission/decision/other)
and "finish_round" (round number) - the two new prediction targets from
the "New prediction targets" Recommendation in docs/tasks.md.

Same leakage-safety contract as every binary run_backtest: each fold's
features come from engineer_fold_features on that fold's own fights only,
imputation/scaling are fit on the fold's train rows only, and
is_predictable_outcome excludes draws/no-contests/DQs/overturned results
from ever being an (X, y) example - round/method are exactly as
undefined for those outcomes as a winner is. Reuses
src.models.logistic_regression._symmetrize for feature-building (the
method_bucket/finish_round columns it carries through are corner-invariant
- the same value for both the red- and blue-perspective row of one fight -
so no new symmetrization logic is needed there).

Hyperparameters are NOT re-tuned here: each target reuses the SAME
params already chosen (and validated) for that model family's winner
target (models/<name>.json), rather than running a new nested search -
these are a different target but the same model-complexity tradeoff, and
a fresh search for two new targets x three model families would be a lot
of extra training for marginal expected benefit on a first pass. Revisit
via nested_validation.py (parameterized by target_column) if a future
pass shows it's worth it.

Usage: python -m src.evaluation.multiclass_backtest
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from src.data.splitting import walk_forward_splits
from src.features.engineering import engineer_fold_features, is_predictable_outcome
from src.features.imputation import apply_imputer, fit_imputer, missing_indicator_columns
from src.features.scaling import apply_scaler, fit_scaler
from src.models.logistic_regression import _symmetrize


def _expand_proba(raw_proba, present_classes, n_classes):
    """model.predict_proba's columns are ordered by model.classes_, which
    can be a STRICT SUBSET of the full class list if a fold's training
    data happens to be missing a rare class (e.g. no round-5 fights yet in
    an early fold) - log_loss against the full label set needs one
    probability column per class, so missing ones are filled with 0 (that
    class genuinely got no probability mass from a model that never saw
    it), not dropped or guessed."""
    full = np.zeros((raw_proba.shape[0], n_classes))
    for col, cls in enumerate(present_classes):
        full[:, cls] = raw_proba[:, col]
    return full


def run_multiclass_backtest(fights, build_model, params, feature_columns, target_column, classes,
                             scale=False, rounds=None, freq="Y", min_train_size=500,
                             return_predictions=False, model_label="model"):
    """Walk-forward backtest of one multiclass fight-outcome target.
    classes is the full, fixed label set (e.g. engineering.METHOD_BUCKETS)
    - labels are encoded to that list's integer positions before fitting
    (XGBoost's sklearn API requires integer class labels; encoding
    uniformly here keeps one code path for all three model families
    instead of special-casing one). Returns a DataFrame of per-fold
    metrics; if return_predictions is True, also returns a long DataFrame
    of period/fight_id/fighter_id/y_true/y_pred/confidence (the predicted
    class's own probability) for every test-fold row, mirroring the binary
    run_backtest functions' return_predictions shape."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    results = []
    predictions = []

    print(f"[multiclass_backtest] {model_label}/{target_column}: starting walk-forward backtest: "
          f"freq={freq}, min_train_size={min_train_size}, classes={classes}")

    for period, train, test in walk_forward_splits(fights, freq=freq, min_train_size=min_train_size):
        train, test = engineer_fold_features(train, test, rounds=rounds)
        # Hard constraint: never a training/evaluation example for a draw,
        # no-contest, DQ, or overturned result - round/method are exactly
        # as undefined for those as a winner is (is_predictable_outcome).
        train = train[is_predictable_outcome(train)]
        test = test[is_predictable_outcome(test)]
        train = _symmetrize(train, feature_columns)
        test = _symmetrize(test, feature_columns)
        if train.empty or test.empty:
            continue

        y_train = train[target_column].map(class_to_idx)
        y_test = test[target_column].map(class_to_idx)

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
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        raw_proba = model.predict_proba(X_test)
        proba = _expand_proba(raw_proba, model.classes_, n_classes)

        fold_accuracy = accuracy_score(y_test, preds)
        fold_log_loss = log_loss(y_test, proba, labels=list(range(n_classes)))
        majority_class = y_train.mode().iloc[0]
        majority_baseline_accuracy = float((y_test == majority_class).mean())
        beats_majority_baseline = bool(fold_accuracy > majority_baseline_accuracy)

        print(f"[multiclass_backtest] {model_label}/{target_column} fold {period}: "
              f"n_train={len(X_train)}, n_test={len(X_test)}, accuracy={fold_accuracy:.4f}, "
              f"log_loss={fold_log_loss:.4f}, majority_baseline_accuracy={majority_baseline_accuracy:.4f}, "
              f"beats_majority_baseline={beats_majority_baseline}")

        results.append({
            "period": str(period), "n_train": len(X_train), "n_test": len(X_test),
            "accuracy": fold_accuracy, "log_loss": fold_log_loss,
            "majority_baseline_accuracy": majority_baseline_accuracy,
            "beats_majority_baseline": beats_majority_baseline,
        })

        if return_predictions:
            confidence = proba[np.arange(len(preds)), preds]
            predictions.append(pd.DataFrame({
                "period": str(period),
                "fight_id": test["fight_id"].values,
                "fighter_id": test["fighter_id"].values,
                "y_true": [classes[i] for i in y_test],
                "y_pred": [classes[i] for i in preds],
                "confidence": confidence,
            }))

    print(f"[multiclass_backtest] {model_label}/{target_column} backtest complete: {len(results)} folds")
    metrics = pd.DataFrame(results)
    if return_predictions:
        predictions_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(
            columns=["period", "fight_id", "fighter_id", "y_true", "y_pred", "confidence"])
        return metrics, predictions_df
    return metrics


def fit_final_multiclass_model(fights, build_model, params, feature_columns, target_column, classes, scale):
    """Fit one multiclass target on the full fights history - the
    deployable fit, mirroring src.models.persistence.fit_final_model.
    Returns (model, scaler, imputer, fitted_feature_columns)."""
    class_to_idx = {c: i for i, c in enumerate(classes)}

    empty_test = fights.iloc[0:0]
    train, _ = engineer_fold_features(fights, empty_test)
    train = train[is_predictable_outcome(train)]
    train = _symmetrize(train, feature_columns)

    imputer = fit_imputer(train, feature_columns)
    train_imputed = apply_imputer(train, feature_columns, imputer)
    fitted_feature_columns = feature_columns + missing_indicator_columns(imputer)
    X_train = train_imputed[fitted_feature_columns]
    y_train = train[target_column].map(class_to_idx)

    scaler = None
    if scale:
        scaler = fit_scaler(X_train, fitted_feature_columns)
        X_train = apply_scaler(X_train, fitted_feature_columns, scaler)

    model = build_model(**params)
    model.fit(X_train, y_train)
    return model, scaler, imputer, fitted_feature_columns


if __name__ == "__main__":
    from src.data.cleaning import clean_fights
    from src.data.ingestion import build_master_equivalent, load_fighters, load_fights
    from src.features.engineering import METHOD_BUCKETS
    from src.models import logistic_regression, random_forest, xgboost_model
    from src.models.persistence import save_model_artifact

    print("[multiclass_backtest] loading data/fights.csv + data/fighter.csv")
    master = build_master_equivalent(load_fights(), load_fighters())
    fights = clean_fights(master)
    print(f"[multiclass_backtest] {len(fights)} fights after cleaning")

    # Round classes are data-driven (the real observed finish_round values
    # among predictable fights), not hardcoded 1-5 - robust to whatever the
    # real distribution actually contains (confirmed: 1-6 appear).
    from src.features.engineering import engineer_fold_features as _eff, is_predictable_outcome as _ipo
    _all_features, _ = _eff(fights, fights.iloc[0:0])
    ROUND_CLASSES = sorted(int(r) for r in _all_features.loc[_ipo(_all_features), "finish_round"].unique())

    # method_bucket needs no feature beyond the usual winner-target set;
    # the round target additionally gets rounds_scheduled (see
    # add_fight_outcome_targets) so it isn't blind to why some fights can
    # only end at round 5 - a 3-round fight and a 5-round title fight that
    # both "went the distance" would otherwise look identical to it.
    METHOD_FEATURE_COLUMNS = logistic_regression.FEATURE_COLUMNS
    ROUND_FEATURE_COLUMNS = logistic_regression.FEATURE_COLUMNS + ["rounds_scheduled"]

    MODELS = {
        "logistic_regression": {"build_model": logistic_regression.build_model, "scale": logistic_regression.SCALE},
        "random_forest": {"build_model": random_forest.build_model, "scale": random_forest.SCALE},
        "xgboost_model": {"build_model": xgboost_model.build_model, "scale": xgboost_model.SCALE},
    }
    TARGETS = {
        "method": ("method_bucket", METHOD_BUCKETS, METHOD_FEATURE_COLUMNS),
        "round": ("finish_round", ROUND_CLASSES, ROUND_FEATURE_COLUMNS),
    }

    for model_name, spec in MODELS.items():
        with open(f"models/{model_name}.json") as f:
            params = __import__("json").load(f)["params"]

        for target_label, (target_column, classes, feature_columns) in TARGETS.items():
            artifact_name = f"{model_name}_{target_label}"
            backtest_metrics = run_multiclass_backtest(
                fights, spec["build_model"], params, feature_columns, target_column, classes,
                scale=spec["scale"], model_label=model_name)

            model, scaler, imputer, fitted_feature_columns = fit_final_multiclass_model(
                fights, spec["build_model"], params, feature_columns, target_column, classes, spec["scale"])

            save_model_artifact(
                model, scaler, imputer, feature_columns, fitted_feature_columns, params,
                metrics={
                    "mean_accuracy": round(float(backtest_metrics["accuracy"].mean()), 4),
                    "mean_log_loss": round(float(backtest_metrics["log_loss"].mean()), 4),
                    "mean_majority_baseline_accuracy": round(float(backtest_metrics["majority_baseline_accuracy"].mean()), 4),
                    "folds_beating_majority_baseline": f"{int(backtest_metrics['beats_majority_baseline'].sum())}/{len(backtest_metrics)}",
                    "source": (f"walk-forward run_multiclass_backtest, target={target_column}, "
                               f"hyperparameters reused from {model_name}'s winner-target artifact"),
                },
                model_name=artifact_name,
                extra_metadata={"classes": classes, "target_column": target_column},
            )
            print(f"[multiclass_backtest] saved models/{artifact_name}.joblib + .json")
