import pandas as pd

from src.features.engineering import FIGHT_STAT_COLUMNS, engineer_fold_features
from src.models.logistic_regression import (
    COMBINED_FEATURE_COLUMNS,
    DIFF_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    INDIVIDUAL_FEATURE_COLUMNS,
    _symmetrize,
)


def make_fight(fight_id, date, r_id, b_id, winner_id, r_reach_inches=70, b_reach_inches=74,
               weight_class="Lightweight"):
    fight = {
        "fight_id": fight_id, "event_date": pd.Timestamp(date),
        "r_fighter_id": r_id, "b_fighter_id": b_id, "winner_id": winner_id,
        "r_dob": pd.Timestamp("1990-01-01"), "b_dob": pd.Timestamp("1985-01-01"),
        "r_height": 70, "b_height": 72, "r_reach_inches": r_reach_inches, "b_reach_inches": b_reach_inches,
        "r_weight_lbs": 170, "b_weight_lbs": 185, "weight_class": weight_class,
        "r_stance": "Orthodox", "b_stance": "Orthodox",
        "method_label": "decision_unanimous", "finish_round": 3,
        "finish_time": 300, "rounds_fought": 3,
    }
    for r_col, b_col in FIGHT_STAT_COLUMNS.values():
        fight[r_col] = 0
        fight[b_col] = 0
    return fight


def test_feature_columns_defaults_to_combined():
    assert FEATURE_COLUMNS == COMBINED_FEATURE_COLUMNS
    assert set(COMBINED_FEATURE_COLUMNS) == set(DIFF_FEATURE_COLUMNS) | set(INDIVIDUAL_FEATURE_COLUMNS)
    # no duplicates - sorted(set(...)) in logistic_regression.py should
    # have already collapsed stance_mismatch's double membership.
    assert len(COMBINED_FEATURE_COLUMNS) == len(set(COMBINED_FEATURE_COLUMNS))


def test_symmetrize_swaps_individual_pairs_for_the_blue_perspective_row():
    fights = pd.DataFrame([
        make_fight("f1", "2019-01-01", "X", "A", "X"),
        make_fight("f2", "2019-06-01", "X", "B", "X"),
        make_fight("f3", "2020-01-01", "X", "C", "X"),
    ])

    engineered, _ = engineer_fold_features(fights, fights.iloc[0:0])
    out = _symmetrize(engineered, ["self_age", "opp_age", "age_diff"])

    # two rows for f3: one red-perspective, one blue-perspective.
    f3 = out[out["fight_id"] == "f3"]
    assert len(f3) == 2

    r_age = engineered[engineered["fight_id"] == "f3"].iloc[0]["r_age"]
    b_age = engineered[engineered["fight_id"] == "f3"].iloc[0]["b_age"]

    by_self_age = f3.set_index("self_age")
    # the row with self_age == r_age must have opp_age == b_age and the
    # ORIGINAL (unflipped) age_diff sign; the row with self_age == b_age
    # (blue perspective) must have opp_age == r_age and the NEGATED diff.
    red_perspective = by_self_age.loc[r_age]
    blue_perspective = by_self_age.loc[b_age]

    assert red_perspective["opp_age"] == b_age
    assert blue_perspective["opp_age"] == r_age
    assert red_perspective["age_diff"] == r_age - b_age
    assert blue_perspective["age_diff"] == -(r_age - b_age)


def test_symmetrize_reach_division_interaction_matches_weight_class_only():
    fights = pd.DataFrame([
        make_fight("f1", "2019-01-01", "X", "A", "X"),
        make_fight("f2", "2020-01-01", "X", "B", "X", r_reach_inches=72, b_reach_inches=68,
                   weight_class="Lightweight"),
    ])

    engineered, _ = engineer_fold_features(fights, fights.iloc[0:0])
    out = _symmetrize(engineered, ["self_reach_inches_x_lightweight", "opp_reach_inches_x_lightweight",
                                    "self_reach_inches_x_heavyweight", "opp_reach_inches_x_heavyweight"])

    f2 = out[out["fight_id"] == "f2"]
    red_row = f2[f2["self_reach_inches_x_lightweight"] == 72.0].iloc[0]
    blue_row = f2[f2["self_reach_inches_x_lightweight"] == 68.0].iloc[0]

    assert red_row["opp_reach_inches_x_lightweight"] == 68.0
    assert blue_row["opp_reach_inches_x_lightweight"] == 72.0
    # wrong division -> zeroed for both corners, both perspectives.
    assert red_row["self_reach_inches_x_heavyweight"] == 0.0
    assert red_row["opp_reach_inches_x_heavyweight"] == 0.0
    assert blue_row["self_reach_inches_x_heavyweight"] == 0.0


def test_symmetrize_label_is_corner_agnostic_own_fighter_won():
    fights = pd.DataFrame([make_fight("f1", "2020-01-01", "X", "A", "X")])

    engineered, _ = engineer_fold_features(fights, fights.iloc[0:0])
    out = _symmetrize(engineered, ["age_diff"])

    # red-perspective row: r_fighter_id (X) is the winner -> label 1.
    # blue-perspective row: b_fighter_id (A) is not the winner -> label 0.
    labels = sorted(out[out["fight_id"] == "f1"]["label"].tolist())
    assert labels == [0, 1]
