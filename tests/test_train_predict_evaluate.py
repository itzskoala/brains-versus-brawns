from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from src.data.splitting import walk_forward_splits
from src.evaluation import train_predict_evaluate as tpe
from src.features.engineering import FIGHT_STAT_COLUMNS
from src.models import logistic_regression, random_forest, xgboost_model

ALL_MODULES = [logistic_regression, random_forest, xgboost_model]


def make_fight(fight_id, date, r_id, b_id, winner_id, method_label="decision_unanimous",
               finish_round=3, rounds_fought=3, r_stance="Orthodox", b_stance="Orthodox"):
    # full master.csv-shaped row, matching tests/test_engineering.py's and
    # tests/test_nested_validation.py's helper - engineer_fold_features
    # needs every one of these columns to exist.
    fight = {
        "fight_id": fight_id, "event_date": pd.Timestamp(date),
        "r_fighter_id": r_id, "b_fighter_id": b_id, "winner_id": winner_id,
        "r_dob": pd.Timestamp("1990-01-01"), "b_dob": pd.Timestamp("1990-01-01"),
        "r_height": 70, "b_height": 70, "r_reach_inches": 70, "b_reach_inches": 70,
        "r_weight_lbs": 170, "b_weight_lbs": 170, "weight_class": "Lightweight",
        "r_stance": r_stance, "b_stance": b_stance,
        "method_label": method_label, "finish_round": finish_round,
        "finish_time": 300, "rounds_fought": rounds_fought,
    }
    # Unlike test_nested_validation.py's helper (which zeroes every stat and
    # only ever runs with a small explicit feature_columns override), this
    # file exercises run_backtest with its FULL default FEATURE_COLUMNS -
    # including the career_sig_str_acc/career_td_acc/*_share ratio features,
    # each some_landed/some_attempted. An all-zero fixture makes every one
    # of those a 0/0 = NaN for every row in every fold, which plain
    # LogisticRegression (unlike the tree models) correctly refuses to fit
    # on. Nonzero, distinct landed/attempted values keep every ratio
    # well-defined, matching what a real fight's totals always look like.
    for r_col, b_col in FIGHT_STAT_COLUMNS.values():
        fight[r_col] = 6
        fight[b_col] = 4
    return fight


def _make_multi_year_fights():
    """6 years x 4 fights/year, four recurring fighters so career-history
    features actually accumulate - enough for several walk-forward folds at
    a small min_train_size, matching this test file's fold size throughout."""
    pairs = [("A", "B"), ("C", "D"), ("E", "F"), ("B", "C")]
    rows = []
    i = 0
    for year in range(2015, 2021):
        for month, (r, b) in enumerate(pairs, start=1):
            i += 1
            rows.append(make_fight(f"f{i}", f"{year}-{month:02d}-01", r, b, winner_id=r))
    return pd.DataFrame(rows)


MIN_TRAIN_SIZE = 4


# ---------------------------------------------------------------------------
# Temporal leakage: the full run_backtest pipeline (not just its sub-pieces,
# which tests/test_splitting.py and tests/test_engineering.py already
# cover) must never fit train-only preprocessing on a test-fold row.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", ALL_MODULES)
def test_run_backtest_fits_imputer_on_train_fold_fights_only(module):
    fights = _make_multi_year_fights()
    expected_folds = list(walk_forward_splits(fights, freq="Y", min_train_size=MIN_TRAIN_SIZE))

    seen_train_fight_ids = []
    original_fit_imputer = module.fit_imputer

    def spy(df, columns):
        seen_train_fight_ids.append(set(df["fight_id"]))
        return original_fit_imputer(df, columns)

    with patch.object(module, "fit_imputer", side_effect=spy):
        module.run_backtest(fights, freq="Y", min_train_size=MIN_TRAIN_SIZE)

    assert len(seen_train_fight_ids) == len(expected_folds) > 0
    for (period, train, test), seen_ids in zip(expected_folds, seen_train_fight_ids):
        assert seen_ids == set(train["fight_id"])
        assert seen_ids.isdisjoint(set(test["fight_id"]))


def test_logistic_regression_run_backtest_fits_scaler_on_train_fold_rows_only():
    # Only logistic_regression scales (SCALE=True) - random_forest/
    # xgboost_model never call fit_scaler at all (tree splits use raw
    # thresholds, see their run_backtest docstrings).
    fights = _make_multi_year_fights()
    expected_folds = list(walk_forward_splits(fights, freq="Y", min_train_size=MIN_TRAIN_SIZE))

    seen_train_row_counts = []
    original_fit_scaler = logistic_regression.fit_scaler

    def spy(df, columns):
        seen_train_row_counts.append(len(df))
        return original_fit_scaler(df, columns)

    with patch.object(logistic_regression, "fit_scaler", side_effect=spy):
        logistic_regression.run_backtest(fights, freq="Y", min_train_size=MIN_TRAIN_SIZE)

    # fit_scaler only ever sees X_train (feature columns, no fight_id) -
    # check its row count against each fold's own training size instead.
    # _symmetrize doubles every fight into a red + blue perspective row.
    assert len(seen_train_row_counts) == len(expected_folds) > 0
    for (period, train, test), n_seen in zip(expected_folds, seen_train_row_counts):
        assert n_seen == 2 * len(train)


# ---------------------------------------------------------------------------
# Evaluation correctness: the metrics run_backtest reports must be an exact
# function of the predictions it actually made - recomputed independently
# from the returned predictions frame using plain sklearn.metrics calls.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", ALL_MODULES)
def test_run_backtest_metrics_match_direct_recomputation_from_its_own_predictions(module):
    fights = _make_multi_year_fights()
    metrics, predictions = module.run_backtest(
        fights, freq="Y", min_train_size=MIN_TRAIN_SIZE, return_predictions=True)

    assert len(metrics) > 0
    for _, fold in metrics.iterrows():
        fold_preds = predictions[predictions["period"] == fold["period"]]
        assert len(fold_preds) == fold["n_test"]

        expected_accuracy = accuracy_score(fold_preds["y_true"], fold_preds["y_pred"])
        expected_log_loss = log_loss(fold_preds["y_true"], fold_preds["y_prob"], labels=[0, 1])
        expected_roc_auc = roc_auc_score(fold_preds["y_true"], fold_preds["y_prob"])

        assert fold["accuracy"] == pytest.approx(expected_accuracy)
        assert fold["log_loss"] == pytest.approx(expected_log_loss)
        assert fold["roc_auc"] == pytest.approx(expected_roc_auc)


def test_return_predictions_doubles_rows_per_fight_for_both_corner_perspectives():
    fights = _make_multi_year_fights()
    _, predictions = logistic_regression.run_backtest(
        fights, freq="Y", min_train_size=MIN_TRAIN_SIZE, return_predictions=True)

    counts = predictions.groupby(["period", "fight_id"]).size()
    assert (counts == 2).all()
    # exactly one perspective per fight wins (label 1) and the other loses.
    labels = predictions.groupby(["period", "fight_id"])["y_true"].apply(sorted)
    assert all(pair == [0, 1] for pair in labels)


@pytest.mark.parametrize("module", ALL_MODULES)
def test_run_backtest_default_return_shape_is_unchanged(module):
    # return_predictions defaults to False - every existing caller
    # (persistence.py, feature_selection.py, representation_comparison.py,
    # missing_value_comparison.py, each module's own __main__) expects a
    # single metrics DataFrame back, not a tuple.
    fights = _make_multi_year_fights()
    metrics = module.run_backtest(fights, freq="Y", min_train_size=MIN_TRAIN_SIZE)

    assert isinstance(metrics, pd.DataFrame)
    assert list(metrics.columns) == [
        "period", "n_train", "n_test", "accuracy", "log_loss",
        "roc_auc", "mse", "target_variance", "beats_mean_baseline",
    ]


# ---------------------------------------------------------------------------
# src.evaluation.train_predict_evaluate: orchestration across the three
# models, and that the saved summary can never disagree with the saved
# per-fold metrics it's derived from.
# ---------------------------------------------------------------------------

def test_run_all_tags_and_concatenates_every_model():
    fights = _make_multi_year_fights()
    predictions_df, metrics_df = tpe.run_all(fights, freq="Y", min_train_size=MIN_TRAIN_SIZE)

    assert set(metrics_df["model"]) == set(tpe.MODELS.keys())
    assert set(predictions_df["model"]) == set(tpe.MODELS.keys())
    for model_name in tpe.MODELS:
        n_folds = (metrics_df["model"] == model_name).sum()
        assert n_folds == len(list(walk_forward_splits(fights, freq="Y", min_train_size=MIN_TRAIN_SIZE)))


def test_summarize_metrics_equals_groupby_mean_of_the_per_fold_metrics():
    fights = _make_multi_year_fights()
    _, metrics_df = tpe.run_all(fights, freq="Y", min_train_size=MIN_TRAIN_SIZE)

    summary = tpe.summarize_metrics(metrics_df)

    for _, row in summary.iterrows():
        fold_rows = metrics_df[metrics_df["model"] == row["model"]]
        assert row["accuracy"] == pytest.approx(fold_rows["accuracy"].mean())
        assert row["log_loss"] == pytest.approx(fold_rows["log_loss"].mean())
        assert row["roc_auc"] == pytest.approx(fold_rows["roc_auc"].mean())
        assert row["n_folds"] == len(fold_rows)
        assert row["folds_beating_baseline"] == fold_rows["beats_mean_baseline"].sum()


def test_save_outputs_and_save_charts_write_the_expected_files(tmp_path):
    fights = _make_multi_year_fights()
    predictions_df, metrics_df = tpe.run_all(fights, freq="Y", min_train_size=MIN_TRAIN_SIZE)

    tpe.save_outputs(predictions_df, metrics_df, out_dir=tmp_path)
    tpe.save_charts(metrics_df, out_dir=tmp_path)

    for name in ["winner_predictions.csv", "winner_metrics_by_fold.csv", "winner_metrics_summary.csv",
                 "winner_accuracy_by_fold.png", "winner_model_comparison.png"]:
        assert (tmp_path / name).exists()

    saved_predictions = pd.read_csv(tmp_path / "winner_predictions.csv")
    assert list(saved_predictions.columns) == ["model", "period", "fight_id", "y_true", "y_pred", "y_prob"]
    assert len(saved_predictions) == len(predictions_df)

    saved_summary = pd.read_csv(tmp_path / "winner_metrics_summary.csv")
    assert set(saved_summary["model"]) == set(tpe.MODELS.keys())
