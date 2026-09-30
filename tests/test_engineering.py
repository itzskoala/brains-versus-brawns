import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "features"))

from engineering import engineer_fold_features


def make_fight(fight_id, date, r_id, b_id, winner_id):
    return {
        "fight_id": fight_id, "event_date": pd.Timestamp(date),
        "r_fighter_id": r_id, "b_fighter_id": b_id, "winner_id": winner_id,
        "r_dob": pd.Timestamp("1990-01-01"), "b_dob": pd.Timestamp("1990-01-01"),
        "r_height": 70, "b_height": 70, "r_reach_inches": 70, "b_reach_inches": 70,
        "r_weight_lbs": 170, "b_weight_lbs": 170, "weight_class": "Lightweight",
    }


def test_engineer_fold_features_only_sees_this_folds_fights():
    # fighter "X" has two fights in train and one in test; a later, unrelated
    # fight for "X" is never passed in at all - simulating a future fold.
    train = pd.DataFrame([
        make_fight("f1", "2020-01-01", "X", "A", "X"),
        make_fight("f2", "2020-06-01", "X", "B", "X"),
    ])
    test = pd.DataFrame([
        make_fight("f3", "2021-01-01", "X", "C", "C"),
    ])

    train_out, test_out = engineer_fold_features(train, test)

    assert set(train_out["fight_id"]) == {"f1", "f2"}
    assert set(test_out["fight_id"]) == {"f3"}

    f1 = train_out[train_out["fight_id"] == "f1"].iloc[0]
    f2 = train_out[train_out["fight_id"] == "f2"].iloc[0]
    f3 = test_out[test_out["fight_id"] == "f3"].iloc[0]

    assert f1["r_prior_fights"] == 0
    assert f2["r_prior_fights"] == 1
    assert f3["r_prior_fights"] == 2
    assert f3["r_prior_wins"] == 2


def test_engineer_fold_features_preserves_row_counts():
    train = pd.DataFrame([
        make_fight("f1", "2020-01-01", "X", "A", "X"),
        make_fight("f2", "2020-06-01", "Y", "B", "Y"),
    ])
    test = pd.DataFrame([
        make_fight("f3", "2021-01-01", "X", "Y", "X"),
    ])

    train_out, test_out = engineer_fold_features(train, test)

    assert len(train_out) == len(train)
    assert len(test_out) == len(test)
