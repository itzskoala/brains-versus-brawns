import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from src.data.splitting import walk_forward_splits
from src.features.engineering import MAIN_DIVISIONS, engineer_fold_features
from src.features.scaling import apply_scaler, fit_scaler

# career_volume_stats and career_style_stats column lists mirror
# src.evaluation.feature_selection.CANDIDATE_FEATURES exactly (duplicated
# here, not imported, since feature_selection.py imports run_backtest from
# this module - importing back would be circular). If engineering.py's
# stat columns change, update both places.
_CAREER_VOLUME_STATS_DIFF = [
    "kd_per_min_diff", "total_str_landed_per_min_diff", "total_str_atmp_per_min_diff",
    "td_success_per_min_diff", "td_atmp_per_min_diff", "sub_att_per_min_diff", "rev_per_min_diff",
    "sig_str_landed_head_per_min_diff", "sig_str_atmp_head_per_min_diff",
    "sig_str_landed_body_per_min_diff", "sig_str_atmp_body_per_min_diff",
    "sig_str_landed_leg_per_min_diff", "sig_str_atmp_leg_per_min_diff",
    "sig_str_landed_distance_per_min_diff", "sig_str_atmp_distance_per_min_diff",
    "sig_str_landed_clinch_per_min_diff", "sig_str_atmp_clinch_per_min_diff",
    "ctrl_seconds_per_min_diff",
]
_CAREER_STYLE_STATS_DIFF = [
    "career_sig_str_acc_diff", "career_td_acc_diff",
    "body_strike_share_diff", "leg_strike_share_diff",
    "clinch_strike_share_diff", "ground_strike_share_diff",
]

# The pre-expansion 14-feature set, kept as its own named constant (rather
# than inlined into FEATURE_COLUMNS) so the "baseline" feature set stays
# referenceable on its own - e.g. for the baseline-vs-expanded nested
# validation comparison in docs/results.md.
BASELINE_FEATURE_COLUMNS = [
    "age_diff", "experience_diff", "prior_win_rate_diff",
    "win_streak_diff", "loss_streak_diff", "form_last5_win_rate_diff",
] + [f"reach_diff_x_{division.lower().replace(' ', '_')}" for division in MAIN_DIVISIONS]

FEATURE_COLUMNS = BASELINE_FEATURE_COLUMNS + _CAREER_VOLUME_STATS_DIFF + _CAREER_STYLE_STATS_DIFF + ["stance_mismatch"]
# height_diff, weight_diff, reach_diff, finish_tendency dropped - see
# src/evaluation/feature_selection.py and docs/results.md Feature Findings
# (2026-10-06 ablation vs. the 14-feature baseline: career_volume_stats,
# career_style_stats, and stance_mismatch all improved or were neutral
# across logistic regression/random forest/XGBoost; finish_tendency
# clearly hurt logistic regression and was flat-to-mixed for the others)

# Features that are corner-invariant (true/false regardless of which
# fighter is labeled r or b) rather than antisymmetric r-minus-b diffs -
# _symmetrize must not negate these for the blue-perspective row.
SYMMETRIC_FEATURE_COLUMNS = ["stance_mismatch"]

# Regularization strength is the one hyperparameter worth tuning for a
# plain logistic regression on this few features - grid search over C
# covers the practically meaningful range without needing more than this.
PARAM_GRID = [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)]
SCALE = True  # distance-based: needs StandardScaler, unlike the tree models


def build_model(**params):
    """Construct a fresh LogisticRegression for one hyperparameter
    candidate - used by src.evaluation.nested_validation so the same
    nested walk-forward tuning code works for every model in src/models/."""
    return LogisticRegression(max_iter=1000, **params)


def _symmetrize(df, feature_columns, symmetric_columns=SYMMETRIC_FEATURE_COLUMNS):
    """Mirror each fight into a red-perspective and a blue-perspective row so
    the model can't exploit which corner is "r" vs "b" as a shortcut. r_id
    was effectively backfilled as "the winner" for pre-2010 records rather
    than reflecting a genuine pre-fight assignment - red wins ~100% of
    fights before 2010 as a labeling artifact, not skill (docs/results.md) -
    so predicting r_win directly would let the model learn that artifact
    instead of the actual matchup features. Doubles the row count; label is
    corner-agnostic (1 if that perspective's fighter won).

    Most feature_columns are antisymmetric r-minus-b diffs (e.g. age_diff =
    r_age - b_age), so flipping perspective negates them. symmetric_columns
    are corner-invariant matchup features (e.g. stance_mismatch: true
    regardless of which fighter is labeled r or b) - those are left
    unchanged for the blue-perspective row instead of being negated, since
    negating a 0/1 flag into -1 would be meaningless."""
    antisymmetric_columns = [c for c in feature_columns if c not in symmetric_columns]

    red = df.copy()
    red["label"] = (red["winner_id"] == red["r_fighter_id"]).astype(int)

    blue = df.copy()
    blue["label"] = (blue["winner_id"] == blue["b_fighter_id"]).astype(int)
    blue[antisymmetric_columns] = -blue[antisymmetric_columns]

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

    print(f"[logistic_regression] starting walk-forward backtest: freq={freq}, "
          f"min_train_size={min_train_size}, C={C}, n_features={len(feature_columns)}")

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

        fold_accuracy = accuracy_score(y_test, preds)
        fold_log_loss = log_loss(y_test, probs, labels=[0, 1])
        fold_roc_auc = roc_auc_score(y_test, probs)
        print(f"[logistic_regression] fold {period}: n_train={len(X_train)}, n_test={len(X_test)}, "
              f"accuracy={fold_accuracy:.4f}, log_loss={fold_log_loss:.4f}, roc_auc={fold_roc_auc:.4f}")

        results.append({
            "period": str(period),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "accuracy": fold_accuracy,
            "log_loss": fold_log_loss,
            "roc_auc": fold_roc_auc,
        })

    print(f"[logistic_regression] backtest complete: {len(results)} folds")
    return pd.DataFrame(results)


if __name__ == "__main__":
    from src.data.cleaning import clean_fights

    print("[logistic_regression] loading data/master.csv")
    master = pd.read_csv("data/master.csv")
    print(f"[logistic_regression] loaded {len(master)} raw rows; cleaning fights")
    fights = clean_fights(master)
    print(f"[logistic_regression] {len(fights)} fights after cleaning; running backtest")

    metrics = run_backtest(fights)
    print(metrics.to_string(index=False))
    print("\nmean accuracy:", metrics["accuracy"].mean())
    print("mean log_loss:", metrics["log_loss"].mean())
    print("mean roc_auc:", metrics["roc_auc"].mean())
