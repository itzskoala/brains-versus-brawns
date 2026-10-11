import pandas as pd
import pytest

from src.features.engineering import FIGHT_STAT_COLUMNS
from src.features.snapshot import (
    build_fighter_snapshot,
    build_matchup_features,
    fighter_fight_years,
    latest_fight_date,
)


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


def make_fighters_df(rows):
    # rows: list of (fighter_id, height_text, weight_lbs, reach_inches, stance, dob)
    return pd.DataFrame(rows, columns=[
        "fighter_id", "height", "weight_lbs", "reach_inches", "stance", "dob"])


@pytest.fixture
def fights():
    return pd.DataFrame([
        make_fight("f1", "2010-01-01", "X", "A", "X"),
        make_fight("f2", "2012-06-01", "X", "B", "B"),
        make_fight("f3", "2018-01-01", "X", "C", "X"),
    ])


@pytest.fixture
def fighters_df():
    return make_fighters_df([
        ("X", "5' 11\"", 170.0, 72.0, "Orthodox", "1990-01-01"),
        ("Y", "6' 0\"", 185.0, 74.0, "Southpaw", "1988-05-05"),
    ])


def test_fighter_fight_years(fights):
    assert fighter_fight_years(fights, "X") == [2010, 2012, 2018]


def test_latest_fight_date_within_year(fights):
    assert latest_fight_date(fights, "X", year=2012) == pd.Timestamp("2012-06-01")


def test_latest_fight_date_overall(fights):
    assert latest_fight_date(fights, "X") == pd.Timestamp("2018-01-01")


def test_latest_fight_date_raises_when_fighter_has_no_fight_in_year(fights):
    with pytest.raises(ValueError):
        latest_fight_date(fights, "X", year=2099)


def test_build_fighter_snapshot_is_inclusive_of_a_real_fight_on_the_cutoff_date(fights, fighters_df):
    # as_of_date = f3's own date means "data valid through f3" (the UI's
    # year-cutoff semantics: "Jon Jones @ 2015" includes his last 2015
    # fight's result) - so the snapshot entering a hypothetical NEXT fight
    # must already include f1, f2, AND f3.
    snapshot = build_fighter_snapshot(fights, fighters_df, "X", pd.Timestamp("2018-01-01"),
                                       weight_class="Lightweight")
    assert snapshot["prior_fights"] == 3
    assert snapshot["prior_wins"] == 2  # f1 (win) + f3 (win); f2 was a loss
    assert snapshot["win_streak"] == 1  # f3 (the most recent) was a win
    assert snapshot["loss_streak"] == 0


def test_build_fighter_snapshot_excludes_fights_strictly_after_the_cutoff(fights, fighters_df):
    # As of f2's own date (2012-06-01), f2 itself counts but f3 (2018) must not.
    snapshot = build_fighter_snapshot(fights, fighters_df, "X", pd.Timestamp("2012-06-01"),
                                       weight_class="Lightweight")
    assert snapshot["prior_fights"] == 2
    assert snapshot["prior_wins"] == 1


def test_build_fighter_snapshot_raises_before_debut(fights, fighters_df):
    with pytest.raises(ValueError):
        build_fighter_snapshot(fights, fighters_df, "X", pd.Timestamp("2000-01-01"),
                                weight_class="Lightweight")


def test_build_fighter_snapshot_raises_for_unknown_fighter(fights, fighters_df):
    with pytest.raises(ValueError):
        build_fighter_snapshot(fights, fighters_df, "nonexistent", pd.Timestamp("2018-01-01"),
                                weight_class="Lightweight")


def test_build_matchup_features_diff_and_overrides():
    red = {"age": 30.0, "prior_fights": 5, "stance": "Orthodox", "reach_inches": 74.0}
    blue = {"age": 28.0, "prior_fights": 2, "stance": "Southpaw", "reach_inches": 70.0}

    feats = build_matchup_features(red, blue, "Heavyweight",
                                    ["age_diff", "experience_diff", "stance_mismatch",
                                     "reach_diff_x_heavyweight", "reach_diff_x_lightweight"])

    assert feats["age_diff"] == 2.0
    assert feats["experience_diff"] == 3  # experience_diff is a prior_fights diff, not an "experience" key
    assert feats["stance_mismatch"] == 1
    assert feats["reach_diff_x_heavyweight"] == 4.0  # matches the selected division
    assert feats["reach_diff_x_lightweight"] == 0.0  # zeroed out for every other division


def test_build_matchup_features_fills_missing_with_zero():
    red = {"prior_win_rate": float("nan")}
    blue = {"prior_win_rate": 0.5}
    feats = build_matchup_features(red, blue, "Heavyweight", ["prior_win_rate_diff"])
    assert feats["prior_win_rate_diff"] == 0.0


def test_build_matchup_features_self_opp_reads_own_vs_other_corner():
    red = {"age": 30.0, "prior_fights": 5}
    blue = {"age": 28.0, "prior_fights": 2}

    feats = build_matchup_features(red, blue, "Heavyweight",
                                    ["self_age", "opp_age", "self_experience", "opp_experience"])

    # self_ is always red here (a live request has no second, mirrored
    # "blue is self" row the way a training-time symmetrized row would).
    assert feats["self_age"] == 30.0
    assert feats["opp_age"] == 28.0
    # experience maps to the prior_fights key, not a literal "experience".
    assert feats["self_experience"] == 5
    assert feats["opp_experience"] == 2


def test_build_matchup_features_self_opp_reach_division_interaction():
    red = {"reach_inches": 74.0}
    blue = {"reach_inches": 70.0}

    feats = build_matchup_features(
        red, blue, "Heavyweight",
        ["self_reach_inches_x_heavyweight", "opp_reach_inches_x_heavyweight",
         "self_reach_inches_x_lightweight", "opp_reach_inches_x_lightweight"])

    assert feats["self_reach_inches_x_heavyweight"] == 74.0
    assert feats["opp_reach_inches_x_heavyweight"] == 70.0
    # wrong division -> zeroed for both corners.
    assert feats["self_reach_inches_x_lightweight"] == 0.0
    assert feats["opp_reach_inches_x_lightweight"] == 0.0


def test_build_matchup_features_missing_indicator_tracks_the_paired_value():
    red = {"prior_win_rate": float("nan"), "age": 30.0}
    blue = {"prior_win_rate": 0.5, "age": 28.0}

    feats = build_matchup_features(
        red, blue, "Heavyweight",
        ["prior_win_rate_diff", "prior_win_rate_diff_was_missing", "age_diff", "age_diff_was_missing"])

    # red's prior_win_rate was NaN -> the diff itself was NaN before fill.
    assert feats["prior_win_rate_diff_was_missing"] == 1
    assert feats["age_diff_was_missing"] == 0


def test_build_matchup_features_uses_imputer_fill_value_not_zero():
    red = {"prior_win_rate": float("nan")}
    blue = {"prior_win_rate": 0.5}
    imputer = {"fill_values": {"prior_win_rate_diff": 0.1}, "missing_columns": ["prior_win_rate_diff"]}

    feats = build_matchup_features(
        red, blue, "Heavyweight", ["prior_win_rate_diff", "prior_win_rate_diff_was_missing"], imputer=imputer)

    assert feats["prior_win_rate_diff"] == 0.1  # the imputer's fill value, not 0.0
    assert feats["prior_win_rate_diff_was_missing"] == 1
