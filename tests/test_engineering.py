import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "features"))

from engineering import FIGHT_STAT_COLUMNS, engineer_fold_features


def make_fight(fight_id, date, r_id, b_id, winner_id, method_label="decision_unanimous",
                finish_round=3, rounds_fought=3, r_stance="Orthodox", b_stance="Orthodox"):
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


def test_win_loss_streaks_reset_on_opposite_result():
    # X: win, win, loss, win -> entering the 4th fight X has a loss_streak
    # of 1 (not 0, not folded into the earlier win streak).
    train = pd.DataFrame([
        make_fight("f1", "2020-01-01", "X", "A", "X"),
        make_fight("f2", "2020-02-01", "X", "B", "X"),
        make_fight("f3", "2020-03-01", "X", "C", "C"),
    ])
    test = pd.DataFrame([
        make_fight("f4", "2020-04-01", "X", "D", "X"),
    ])

    train_out, test_out = engineer_fold_features(train, test)

    f4 = test_out[test_out["fight_id"] == "f4"].iloc[0]
    assert f4["r_win_streak"] == 0
    assert f4["r_loss_streak"] == 1
    assert f4["r_form_last5_win_rate"] == 2 / 3


def test_career_stat_features_ignore_fights_with_no_round_detail():
    # X's only prior fight has rounds_fought == 0, i.e. the round-level
    # scrape is missing and its r_total_sig_landed == 0 is a missingness
    # marker, not a genuine zero-strike fight (docs/results.md). That prior
    # fight must not drag X's career sig_str_landed_per_min to 0.
    missing_detail = make_fight("f1", "2020-01-01", "X", "A", "X", rounds_fought=0)
    missing_detail["r_total_sig_landed"] = 0

    normal = make_fight("f2", "2020-02-01", "X", "B", "X", rounds_fought=3)
    normal["r_total_sig_landed"] = 30
    normal["finish_time"] = 0  # 2 full 5-min rounds + 0:00 of round 3 = 10 min

    train = pd.DataFrame([missing_detail])
    test = pd.DataFrame([normal])

    _, test_out = engineer_fold_features(train, test)

    f2 = test_out[test_out["fight_id"] == "f2"].iloc[0]
    assert f2["r_prior_fights"] == 1
    # if the missing-detail fight counted as 0 strikes over 5 real minutes,
    # this would be far below 0 instead of NaN (no valid prior fight yet).
    assert pd.isna(f2["r_sig_str_landed_per_min"])


def test_finish_tendency_tracks_prior_method_and_finish_round():
    # X: win by KO in round 1, then win by submission in round 2 -> entering
    # a 3rd fight, X has finished both prior fights, one by each method.
    train = pd.DataFrame([
        make_fight("f1", "2020-01-01", "X", "A", "X", method_label="ko_tko", finish_round=1),
        make_fight("f2", "2020-02-01", "X", "B", "X", method_label="submission", finish_round=2),
    ])
    test = pd.DataFrame([
        make_fight("f3", "2020-03-01", "X", "C", "X"),
    ])

    _, test_out = engineer_fold_features(train, test)

    f3 = test_out[test_out["fight_id"] == "f3"].iloc[0]
    assert f3["r_prior_ko_tko_rate"] == 0.5
    assert f3["r_prior_win_ko_tko_rate"] == 0.5
    assert f3["r_prior_submission_rate"] == 0.5
    assert f3["r_avg_finish_round_prior"] == 1.5


def test_finish_tendency_distinguishes_decision_subtypes():
    # X wins a dominant unanimous decision, then barely survives a split
    # decision - the two should not blur into one "decision" rate.
    train = pd.DataFrame([
        make_fight("f1", "2020-01-01", "X", "A", "X", method_label="decision_unanimous"),
    ])
    test = pd.DataFrame([
        make_fight("f2", "2020-02-01", "X", "B", "X", method_label="decision_split"),
    ])

    _, test_out = engineer_fold_features(train, test)

    f2 = test_out[test_out["fight_id"] == "f2"].iloc[0]
    assert f2["r_prior_decision_unanimous_rate"] == 1.0
    assert f2["r_prior_decision_split_rate"] == 0.0


def test_stance_mismatch_flags_orthodox_vs_southpaw_only():
    same = make_fight("f1", "2020-01-01", "X", "A", "X", r_stance="Orthodox", b_stance="Orthodox")
    mismatch = make_fight("f2", "2020-01-01", "Y", "B", "Y", r_stance="Orthodox", b_stance="Southpaw")
    unknown = make_fight("f3", "2020-01-01", "Z", "C", "Z", r_stance="Orthodox", b_stance="Unknown")

    out, _ = engineer_fold_features(pd.DataFrame([same, mismatch, unknown]), pd.DataFrame([]))

    by_id = out.set_index("fight_id")["stance_mismatch"]
    assert by_id["f1"] == 0
    assert by_id["f2"] == 1
    assert by_id["f3"] == 0
