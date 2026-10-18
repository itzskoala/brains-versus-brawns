import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.evaluation.multiclass_backtest import (
    _expand_proba,
    fit_final_multiclass_model,
    run_multiclass_backtest,
)
from src.features.engineering import FIGHT_STAT_COLUMNS


def make_fight(fight_id, date, r_id, b_id, winner_id, method_label="decision_unanimous",
               finish_round=3, time_format="3 Rnd (5-5-5)", rounds_fought=3,
               result_status="win"):
    fight = {
        "fight_id": fight_id, "event_date": pd.Timestamp(date),
        "r_fighter_id": r_id, "b_fighter_id": b_id, "winner_id": winner_id,
        "result_status": result_status,
        "r_dob": pd.Timestamp("1990-01-01"), "b_dob": pd.Timestamp("1990-01-01"),
        "r_height": 70, "b_height": 70, "r_reach_inches": 70, "b_reach_inches": 70,
        "r_weight_lbs": 170, "b_weight_lbs": 170, "weight_class": "Lightweight",
        "r_stance": "Orthodox", "b_stance": "Orthodox",
        "method_label": method_label, "finish_round": finish_round, "time_format": time_format,
        "finish_time": 300, "rounds_fought": rounds_fought,
    }
    for r_col, b_col in FIGHT_STAT_COLUMNS.values():
        fight[r_col] = 6
        fight[b_col] = 4
    return fight


def _make_multi_year_fights():
    pairs = [("A", "B"), ("C", "D"), ("E", "F"), ("B", "C")]
    methods = ["ko_tko", "submission", "decision_unanimous", "decision_split"]
    rows = []
    i = 0
    for year in range(2015, 2022):
        for month, (r, b) in enumerate(pairs, start=1):
            i += 1
            rows.append(make_fight(f"f{i}", f"{year}-{month:02d}-01", r, b, winner_id=r,
                                    method_label=methods[month % len(methods)],
                                    finish_round=(month % 3) + 1))
    return pd.DataFrame(rows)


MIN_TRAIN_SIZE = 4
METHOD_FEATURE_COLUMNS = ["age_diff", "experience_diff"]
ROUND_FEATURE_COLUMNS = ["age_diff", "experience_diff", "rounds_scheduled"]


def build_model(**params):
    return LogisticRegression(max_iter=1000, **params)


def test_expand_proba_fills_zero_for_a_class_missing_from_this_folds_training():
    raw = np.array([[0.3, 0.7], [0.6, 0.4]])
    # model only ever saw classes 0 and 2 in its training fold (class 1 -
    # e.g. a rare "other" method bucket - never appeared) - its own
    # predict_proba columns are ordered [0, 2], not [0, 1, 2].
    expanded = _expand_proba(raw, present_classes=[0, 2], n_classes=3)
    assert expanded.shape == (2, 3)
    assert list(expanded[0]) == [0.3, 0.0, 0.7]
    assert list(expanded[1]) == [0.6, 0.0, 0.4]


def test_run_multiclass_backtest_produces_sane_metrics_and_predictions():
    fights = _make_multi_year_fights()
    classes = ["decision", "ko_tko", "submission"]
    metrics, predictions = run_multiclass_backtest(
        fights, build_model, {"C": 1.0}, METHOD_FEATURE_COLUMNS, "method_bucket", classes,
        scale=True, min_train_size=MIN_TRAIN_SIZE, return_predictions=True)

    assert len(metrics) > 0
    assert set(metrics.columns) >= {"accuracy", "log_loss", "majority_baseline_accuracy", "beats_majority_baseline"}
    assert (metrics["accuracy"] >= 0).all() and (metrics["accuracy"] <= 1).all()

    assert len(predictions) > 0
    assert set(predictions["y_true"]).issubset(set(classes))
    assert set(predictions["y_pred"]).issubset(set(classes))
    assert (predictions["confidence"] >= 0).all() and (predictions["confidence"] <= 1).all()
    assert "fighter_id" in predictions.columns


def test_run_multiclass_backtest_never_trains_or_tests_on_a_non_predictable_outcome():
    fights = _make_multi_year_fights()
    rows = list(fights.to_dict("records"))
    rows += [
        make_fight("draw1", "2018-06-15", "A", "B", None, result_status="draw"),
        make_fight("nc1", "2019-06-15", "A", "B", None, result_status="no_contest"),
        make_fight("dq1", "2020-06-15", "A", "B", "A", method_label="dq"),
    ]
    fights_with_bad_outcomes = pd.DataFrame(rows)
    non_predictable_ids = {"draw1", "nc1", "dq1"}

    _, predictions = run_multiclass_backtest(
        fights_with_bad_outcomes, build_model, {"C": 1.0}, METHOD_FEATURE_COLUMNS, "method_bucket",
        ["decision", "ko_tko", "submission"], scale=True, min_train_size=MIN_TRAIN_SIZE, return_predictions=True)

    assert set(predictions["fight_id"]).isdisjoint(non_predictable_ids)


def test_fit_final_multiclass_model_produces_a_usable_model():
    fights = _make_multi_year_fights()
    classes = ["decision", "ko_tko", "submission"]
    model, scaler, imputer, fitted_feature_columns = fit_final_multiclass_model(
        fights, build_model, {"C": 1.0}, METHOD_FEATURE_COLUMNS, "method_bucket", classes, scale=True)

    assert scaler is not None
    assert set(METHOD_FEATURE_COLUMNS).issubset(set(fitted_feature_columns))
    # the fitted model must only ever predict an index into `classes`.
    dummy_X = pd.DataFrame([[0.0] * len(fitted_feature_columns)], columns=fitted_feature_columns)
    pred = model.predict(dummy_X)[0]
    assert 0 <= pred < len(classes)


def test_round_target_uses_rounds_scheduled_as_a_feature_not_a_leak():
    # rounds_scheduled (parsed from time_format) must be usable as an
    # input feature for the ROUND target without engineer_fold_features
    # ever treating finish_round itself as anything but the label.
    fights = _make_multi_year_fights()
    classes = sorted(fights["finish_round"].unique().tolist())
    metrics, predictions = run_multiclass_backtest(
        fights, build_model, {"C": 1.0}, ROUND_FEATURE_COLUMNS, "finish_round", classes,
        scale=True, min_train_size=MIN_TRAIN_SIZE, return_predictions=True)
    assert len(metrics) > 0
    assert set(predictions["y_true"]).issubset(set(classes))
