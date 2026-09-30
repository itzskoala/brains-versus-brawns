import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from src.data.splitting import walk_forward_splits
from src.features.engineering import MAIN_DIVISIONS, engineer_fold_features
from src.features.scaling import apply_scaler, fit_scaler

FEATURE_COLUMNS = [
    "age_diff", "experience_diff", "prior_win_rate_diff",
] + [f"reach_diff_x_{division.lower().replace(' ', '_')}" for division in MAIN_DIVISIONS]
# height_diff, weight_diff, reach_diff dropped - see src/evaluation/feature_selection.py


def _symmetrize(df, feature_columns):
    """Mirror each fight into a red-perspective and a blue-perspective row so
    the model can't exploit which corner is "r" vs "b" as a shortcut. r_id
    was effectively backfilled as "the winner" for pre-2010 records rather
    than reflecting a genuine pre-fight assignment - red wins ~100% of
    fights before 2010 as a labeling artifact, not skill (docs/results.md) -
    so predicting r_win directly would let the model learn that artifact
    instead of the actual matchup features. Doubles the row count; label is
    corner-agnostic (1 if that perspective's fighter won)."""
    red = df.copy()
    red["label"] = (red["winner_id"] == red["r_fighter_id"]).astype(int)

    blue = df.copy()
    blue["label"] = (blue["winner_id"] == blue["b_fighter_id"]).astype(int)
    blue[feature_columns] = -blue[feature_columns]

    return pd.concat([red, blue], ignore_index=True)


def run_backtest(fights, rounds=None, freq="Y", min_train_size=500, C=1.0, feature_columns=None):
    """Walk-forward backtest of a logistic regression baseline predicting
    fight outcomes. For each fold, features are built fresh from only that
    fold's own fights via engineer_fold_features, the scaler is fit on the
    fold's train rows only, and the model is trained and scored on that fold
    alone - so nothing about a later fold ever influences an earlier one.
    feature_columns defaults to FEATURE_COLUMNS; pass a subset for feature
    selection. Returns a DataFrame of per-fold metrics."""
    feature_columns = feature_columns or FEATURE_COLUMNS
    results = []

    for period, train, test in walk_forward_splits(fights, freq=freq, min_train_size=min_train_size):
        train, test = engineer_fold_features(train, test, rounds=rounds)
        train = _symmetrize(train, feature_columns)
        test = _symmetrize(test, feature_columns)

        y_train = train["label"]
        y_test = test["label"]

        X_train = train[feature_columns].fillna(0)
        X_test = test[feature_columns].fillna(0)

        scaler = fit_scaler(X_train, feature_columns)
        X_train = apply_scaler(X_train, feature_columns, scaler)
        X_test = apply_scaler(X_test, feature_columns, scaler)

        model = LogisticRegression(C=C, max_iter=1000)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        results.append({
            "period": str(period),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "accuracy": accuracy_score(y_test, preds),
            "log_loss": log_loss(y_test, probs, labels=[0, 1]),
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    from src.data.cleaning import clean_fights

    master = pd.read_csv("data/master.csv")
    fights = clean_fights(master)

    metrics = run_backtest(fights)
    print(metrics.to_string(index=False))
    print("\nmean accuracy:", metrics["accuracy"].mean())
    print("mean log_loss:", metrics["log_loss"].mean())
