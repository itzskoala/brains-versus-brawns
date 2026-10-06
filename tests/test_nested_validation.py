from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data.splitting import walk_forward_splits
from src.evaluation import nested_validation as nv
from src.features.engineering import FIGHT_STAT_COLUMNS


def make_fight(fight_id, date, r_id, b_id, winner_id, method_label="decision_unanimous",
                finish_round=3, rounds_fought=3, r_stance="Orthodox", b_stance="Orthodox"):
    # full master.csv-shaped row, matching tests/test_engineering.py's helper -
    # engineer_fold_features needs every one of these columns to exist.
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
    for r_col, b_col in FIGHT_STAT_COLUMNS.values():
        fight[r_col] = 0
        fight[b_col] = 0
    return fight


class _DummyModel:
    """Predicts the majority class seen in training - enough to exercise
    the pipeline without needing a real sklearn estimator."""

    def fit(self, X, y):
        self._majority = int(round(y.mean())) if len(y) else 1
        return self

    def predict(self, X):
        return [self._majority] * len(X)

    def predict_proba(self, X):
        p = 0.99 if self._majority == 1 else 0.01
        return np.array([[1 - p, p]] * len(X))


def _build_dummy(**params):
    return _DummyModel()


def test_select_hyperparameters_only_uses_inner_folds_from_train_fights():
    # f5 (2021) is not included in train_fights at all - select_hyperparameters
    # only ever sees f1-f4, so it structurally cannot use f5.
    train_fights = pd.DataFrame([
        make_fight("f1", "2018-01-01", "X", "A", "X"),
        make_fight("f2", "2018-06-01", "Y", "B", "Y"),
        make_fight("f3", "2019-01-01", "X", "B", "X"),
        make_fight("f4", "2019-06-01", "Y", "A", "Y"),
    ])

    best_params, scores_df = nv.select_hyperparameters(
        train_fights, _build_dummy, param_grid=[{"a": 1}, {"a": 2}],
        feature_columns=["age_diff"], min_train_size=1,
    )

    assert best_params in ({"a": 1}, {"a": 2})
    assert set(scores_df["n_inner_folds"]) != {0}  # sanity: it actually ran some inner folds


def test_run_nested_backtest_never_leaks_outer_test_fights_into_inner_selection():
    fights = pd.DataFrame([
        make_fight("f1", "2018-01-01", "X", "A", "X"),
        make_fight("f2", "2018-06-01", "Y", "B", "Y"),
        make_fight("f3", "2019-01-01", "X", "B", "X"),
        make_fight("f4", "2019-06-01", "Y", "A", "Y"),
        make_fight("f5", "2020-01-01", "X", "Y", "X"),
        make_fight("f6", "2020-06-01", "A", "B", "A"),
    ])

    seen_train_fight_ids = []
    original = nv.select_hyperparameters

    def spy(train_fights, *args, **kwargs):
        seen_train_fight_ids.append(set(train_fights["fight_id"]))
        return original(train_fights, *args, **kwargs)

    with patch.object(nv, "select_hyperparameters", side_effect=spy):
        result = nv.run_nested_backtest(
            fights, _build_dummy, param_grid=[{"a": 1}],
            feature_columns=["age_diff"], min_train_size=1, inner_min_train_size=1,
        )

    outer_folds = list(walk_forward_splits(fights, min_train_size=1))
    assert len(seen_train_fight_ids) == len(outer_folds) == len(result)

    for (period, train, test), seen_ids in zip(outer_folds, seen_train_fight_ids):
        assert seen_ids == set(train["fight_id"])
        assert seen_ids.isdisjoint(set(test["fight_id"]))


def test_run_nested_backtest_falls_back_when_outer_fold_too_small_for_inner_folds():
    # the first outer fold's own training window is too small to produce
    # any inner fold at the requested inner_min_train_size - should fall
    # back to the first grid candidate rather than erroring out.
    fights = pd.DataFrame([
        make_fight("f1", "2018-01-01", "X", "A", "X"),
        make_fight("f2", "2019-01-01", "X", "B", "X"),
    ])

    result = nv.run_nested_backtest(
        fights, _build_dummy, param_grid=[{"a": 1}, {"a": 2}],
        feature_columns=["age_diff"], min_train_size=1, inner_min_train_size=5,
    )

    assert len(result) == 1
    assert result.iloc[0]["chosen_params"] == {"a": 1}
    assert result.iloc[0]["n_inner_folds"] == 0
