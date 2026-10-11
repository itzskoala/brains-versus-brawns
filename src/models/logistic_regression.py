import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_auc_score

from src.data.splitting import walk_forward_splits
from src.features.engineering import MAIN_DIVISIONS, engineer_fold_features
from src.features.imputation import apply_imputer, fit_imputer, missing_indicator_columns
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
# than inlined into DIFF_FEATURE_COLUMNS) so the "baseline" feature set
# stays referenceable on its own - e.g. for the baseline-vs-expanded nested
# validation comparison in docs/results.md.
BASELINE_FEATURE_COLUMNS = [
    "age_diff", "experience_diff", "prior_win_rate_diff",
    "win_streak_diff", "loss_streak_diff", "form_last5_win_rate_diff",
] + [f"reach_diff_x_{division.lower().replace(' ', '_')}" for division in MAIN_DIVISIONS]

# Corner differences (r_X - b_X) - this was the sole production feature set
# through the 2026-10 ablation/promotion work (docs/results.md) and is kept
# as its own name since feature_selection.py's ablation and
# src.evaluation.missing_value_comparison's already-reported numbers are
# specifically about this set, not whatever FEATURE_COLUMNS currently is.
DIFF_FEATURE_COLUMNS = (
    BASELINE_FEATURE_COLUMNS + _CAREER_VOLUME_STATS_DIFF + _CAREER_STYLE_STATS_DIFF + ["stance_mismatch"]
)
# height_diff, weight_diff, reach_diff, finish_tendency dropped - see
# src/evaluation/feature_selection.py and docs/results.md Feature Findings
# (2026-10-06 ablation vs. the 14-feature baseline: career_volume_stats,
# career_style_stats, and stance_mismatch all improved or were neutral
# across logistic regression/random forest/XGBoost; finish_tendency
# clearly hurt logistic regression and was flat-to-mixed for the others)

# experience_diff is built from r_prior_fights/b_prior_fights, not
# r_experience/b_experience (same override src.features.snapshot's
# _DIFF_KEY_OVERRIDES uses) - spelled out rather than guessed.
_BASE_KEY_OVERRIDES = {"experience": "prior_fights"}

_DIVISION_SUFFIXES = [d.lower().replace(" ", "_") for d in MAIN_DIVISIONS]

# The underlying r_/b_ paired signal behind each non-division diff in
# BASELINE_FEATURE_COLUMNS.
_SIMPLE_BASES = ["age", "experience", "prior_win_rate", "win_streak", "loss_streak", "form_last5_win_rate"]
# reach_diff_x_<division> has no per-corner equivalent in engineering.py
# (only the diff is computed there) - _add_reach_division_interactions
# below builds r_/b_reach_inches_x_<division> here, from the same
# r_reach_inches/b_reach_inches/weight_class columns the diff version uses.
_REACH_DIVISION_BASES = [f"reach_inches_x_{s}" for s in _DIVISION_SUFFIXES]
# strip "_diff" from the exact diff lists above, so "individual" represents
# precisely the same underlying stats as the diffs - not a superset or
# subset of them.
_VOLUME_BASES = [name[: -len("_diff")] for name in _CAREER_VOLUME_STATS_DIFF]
_STYLE_BASES = [name[: -len("_diff")] for name in _CAREER_STYLE_STATS_DIFF]

# Every underlying r_/b_ paired signal with both a diff (above) and an
# individual (self_/opp_, see _symmetrize) representation. Does NOT include
# finish_tendency or the dropped height_diff/reach_diff/weight_diff - those
# have no individual counterpart defined (never promoted into
# DIFF_FEATURE_COLUMNS, so there's nothing to pair against); see
# src.evaluation.feature_selection for the ablation groups this mirrors.
PAIRED_BASES = _SIMPLE_BASES + _REACH_DIVISION_BASES + _VOLUME_BASES + _STYLE_BASES

INDIVIDUAL_FEATURE_COLUMNS = (
    [f"{side}_{base}" for base in PAIRED_BASES for side in ("self", "opp")] + ["stance_mismatch"]
)

# The new default (2026-10): individual per-corner values plus the existing
# meaningful diffs/interactions, promoted after
# src.evaluation.representation_comparison showed plain diffs were the
# worst of the three representations tested, on every metric, for all
# three models - see docs/results.md.
COMBINED_FEATURE_COLUMNS = sorted(set(DIFF_FEATURE_COLUMNS) | set(INDIVIDUAL_FEATURE_COLUMNS))
FEATURE_COLUMNS = COMBINED_FEATURE_COLUMNS

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


def _add_reach_division_interactions(df):
    """Per-corner r_/b_reach_inches_x_<division>, mirroring the existing
    reach_diff_x_<division> but kept undifferenced per corner - lets the
    individual/combined representations learn the same division-specific
    reach effect the diff representation already could."""
    df = df.copy()
    for division in MAIN_DIVISIONS:
        suffix = division.lower().replace(" ", "_")
        in_division = df["weight_class"] == division
        df[f"r_reach_inches_x_{suffix}"] = df["r_reach_inches"].where(in_division, 0)
        df[f"b_reach_inches_x_{suffix}"] = df["b_reach_inches"].where(in_division, 0)
    return df


def _symmetrize(df, feature_columns, symmetric_columns=SYMMETRIC_FEATURE_COLUMNS):
    """Mirror each fight into a red-perspective and a blue-perspective row so
    the model can't exploit which corner is "r" vs "b" as a shortcut. r_id
    was effectively backfilled as "the winner" for pre-2010 records rather
    than reflecting a genuine pre-fight assignment - red wins ~100% of
    fights before 2010 as a labeling artifact, not skill (docs/results.md) -
    so predicting r_win directly would let the model learn that artifact
    instead of the actual matchup features. Doubles the row count; label is
    corner-agnostic (1 if that perspective's fighter won).

    Produces every representation's features in one pass, regardless of
    which ones feature_columns actually selects:
    - DIFF_FEATURE_COLUMNS' antisymmetric r-minus-b diffs (e.g. age_diff =
      r_age - b_age) are negated for the blue-perspective row.
    - symmetric_columns (e.g. stance_mismatch: true regardless of which
      fighter is red/blue) are left unchanged - negating a 0/1 flag into -1
      would be meaningless.
    - INDIVIDUAL_FEATURE_COLUMNS' self_<base>/opp_<base> pairs (see
      PAIRED_BASES) are SWAPPED, not negated, for the blue-perspective row:
      self_ is always this row's own fighter, opp_ the other, so whichever
      corner they were originally labeled, "self" means "this row"."""
    df = _add_reach_division_interactions(df)
    antisymmetric_columns = [c for c in DIFF_FEATURE_COLUMNS if c not in symmetric_columns]

    def base_cols(base):
        key = _BASE_KEY_OVERRIDES.get(base, base)
        return f"r_{key}", f"b_{key}"

    red = df.copy()
    red["label"] = (red["winner_id"] == red["r_fighter_id"]).astype(int)
    red_pairs = pd.DataFrame({
        f"{side}_{base}": red[col]
        for base in PAIRED_BASES
        for side, col in zip(("self", "opp"), base_cols(base))
    }, index=red.index)
    red = pd.concat([red, red_pairs], axis=1)

    blue = df.copy()
    blue["label"] = (blue["winner_id"] == blue["b_fighter_id"]).astype(int)
    blue[antisymmetric_columns] = -blue[antisymmetric_columns]
    blue_pairs = pd.DataFrame({
        f"{side}_{base}": blue[col]
        for base in PAIRED_BASES
        for side, col in zip(("self", "opp"), reversed(base_cols(base)))
    }, index=blue.index)
    blue = pd.concat([blue, blue_pairs], axis=1)

    return pd.concat([red, blue], ignore_index=True)


def run_backtest(fights, rounds=None, freq="Y", min_train_size=500, C=1.0, feature_columns=None,
                  return_predictions=False):
    """Walk-forward backtest of a logistic regression baseline predicting
    fight outcomes. For each fold, features are built fresh from only that
    fold's own fights via engineer_fold_features, missing values are
    imputed (src.features.imputation: training-fold median plus a
    missingness indicator - see docs/results.md on why this replaced
    zero-fill) and the scaler is fit, both using the fold's train rows
    only, and the model is trained and scored on that fold alone - so
    nothing about a later fold ever influences an earlier one.
    feature_columns defaults to FEATURE_COLUMNS (the combined
    individual+diff representation); pass a subset for feature selection.
    Returns a DataFrame of per-fold metrics; if return_predictions is True,
    also returns a long DataFrame of every test-fold row's own
    period/fight_id/y_true/y_pred/y_prob (one row per fight per corner
    perspective, see _symmetrize) as a second return value - used by
    src.evaluation.train_predict_evaluate to save per-fold predictions
    without duplicating this fold loop."""
    feature_columns = feature_columns or FEATURE_COLUMNS
    results = []
    predictions = []

    print(f"[logistic_regression] starting walk-forward backtest: freq={freq}, "
          f"min_train_size={min_train_size}, C={C}, n_features={len(feature_columns)}")

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

        scaler = fit_scaler(X_train, fitted_feature_columns)
        X_train = apply_scaler(X_train, fitted_feature_columns, scaler)
        X_test = apply_scaler(X_test, fitted_feature_columns, scaler)

        model = LogisticRegression(C=C, max_iter=1000)
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
        print(f"[logistic_regression] fold {period}: n_train={len(X_train)}, n_test={len(X_test)}, "
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

    print(f"[logistic_regression] backtest complete: {len(results)} folds")
    metrics = pd.DataFrame(results)
    if return_predictions:
        predictions_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(
            columns=["period", "fight_id", "y_true", "y_pred", "y_prob"])
        return metrics, predictions_df
    return metrics


if __name__ == "__main__":
    from src.data.cleaning import clean_fights
    from src.data.ingestion import build_master_equivalent, load_fighters, load_fights

    print("[logistic_regression] loading data/fights.csv + data/fighter.csv")
    master = build_master_equivalent(load_fights(), load_fighters())
    print(f"[logistic_regression] loaded {len(master)} raw rows; cleaning fights")
    fights = clean_fights(master)
    print(f"[logistic_regression] {len(fights)} fights after cleaning; running backtest")

    metrics = run_backtest(fights)
    print(metrics.to_string(index=False))
    print("\nmean accuracy:", metrics["accuracy"].mean())
    print("mean log_loss:", metrics["log_loss"].mean())
    print("mean roc_auc:", metrics["roc_auc"].mean())
    print("mean mse:", metrics["mse"].mean())
    print("mean target_variance:", metrics["target_variance"].mean())
    print("folds beating mean-prediction baseline:",
          f"{metrics['beats_mean_baseline'].sum()}/{len(metrics)}")
