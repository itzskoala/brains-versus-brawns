import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "data"))

from splitting import split_fights, walk_forward_splits


def make_fights(dates):
    return pd.DataFrame({
        "fight_id": [f"f{i}" for i in range(len(dates))],
        "event_date": pd.to_datetime(dates),
    })


def test_split_fights_is_chronological_and_covers_all_rows():
    dates = pd.date_range("2000-01-01", periods=100, freq="10D")
    df = make_fights(dates).sample(frac=1, random_state=0)  # shuffled input

    train, val, test = split_fights(df)

    assert len(train) + len(val) + len(test) == len(df)
    assert train["event_date"].max() < val["event_date"].min()
    assert val["event_date"].max() < test["event_date"].min()


def test_walk_forward_splits_never_trains_on_the_future():
    dates = pd.date_range("2000-01-01", "2010-12-31", freq="15D")
    df = make_fights(dates)

    folds = list(walk_forward_splits(df, freq="Y", min_train_size=20))

    assert len(folds) > 0
    for period, train, test in folds:
        assert train["event_date"].max() < test["event_date"].min()
        assert len(train) >= 20
        assert "_period" not in train.columns and "_period" not in test.columns


def test_walk_forward_splits_training_window_expands():
    dates = pd.date_range("2000-01-01", "2010-12-31", freq="15D")
    df = make_fights(dates)

    folds = list(walk_forward_splits(df, freq="Y", min_train_size=20))
    train_sizes = [len(train) for _, train, _ in folds]

    assert train_sizes == sorted(train_sizes)


def test_walk_forward_splits_skips_periods_below_min_train_size():
    dates = pd.date_range("2000-01-01", "2002-12-31", freq="60D")
    df = make_fights(dates)

    folds = list(walk_forward_splits(df, freq="Y", min_train_size=1000))

    assert folds == []
