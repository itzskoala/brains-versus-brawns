import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "features"))

from engineering import FIGHT_STAT_COLUMNS, add_early_pace_features, engineer_fold_features, is_predictable_outcome


def make_fight(fight_id, date, r_id, b_id, winner_id, method_label="decision_unanimous",
                finish_round=3, finish_time=300, rounds_fought=3, r_stance="Orthodox", b_stance="Orthodox",
                result_status="win", time_format="3 Rnd (5-5-5)"):
    fight = {
        "fight_id": fight_id, "event_date": pd.Timestamp(date),
        "r_fighter_id": r_id, "b_fighter_id": b_id, "winner_id": winner_id,
        "result_status": result_status,
        "r_dob": pd.Timestamp("1990-01-01"), "b_dob": pd.Timestamp("1990-01-01"),
        "r_height": 70, "b_height": 70, "r_reach_inches": 70, "b_reach_inches": 70,
        "r_weight_lbs": 170, "b_weight_lbs": 170, "weight_class": "Lightweight",
        "r_stance": r_stance, "b_stance": b_stance,
        "method_label": method_label, "finish_round": finish_round, "time_format": time_format,
        "finish_time": finish_time, "rounds_fought": rounds_fought,
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


def make_round(fight_id, round_no, r_id, b_id, r_sig_landed, b_sig_landed):
    return {
        "fight_id": fight_id, "round_no": round_no, "r_id": r_id, "b_id": b_id,
        "r_sig_landed": r_sig_landed, "b_sig_landed": b_sig_landed,
    }


def test_future_fights_never_change_an_earlier_fights_features():
    # A future-leakage smoke test independent of any specific feature: the
    # whole feature matrix for f1/f2 must come out identical whether or not
    # a later fight (f3, after both) is even present in the input.
    train_without_future = pd.DataFrame([
        make_fight("f1", "2020-01-01", "X", "A", "X"),
        make_fight("f2", "2020-06-01", "X", "B", "B"),
    ])
    train_with_future = pd.DataFrame([
        make_fight("f1", "2020-01-01", "X", "A", "X"),
        make_fight("f2", "2020-06-01", "X", "B", "B"),
        make_fight("f3", "2021-01-01", "X", "C", "X"),
    ])

    out_without, _ = engineer_fold_features(train_without_future, train_without_future.iloc[0:0])
    out_with, _ = engineer_fold_features(train_with_future, train_with_future.iloc[0:0])

    shared = out_without.set_index("fight_id").sort_index()
    comparable = out_with.set_index("fight_id").loc[shared.index].sort_index()
    pd.testing.assert_frame_equal(shared[comparable.columns], comparable)


def test_same_date_fights_do_not_leak_into_each_other():
    # X fights twice on the exact same date (an early-tournament-format
    # doubleheader, same as real rows in fights.csv) - winning the first,
    # losing the second. Neither same-date fight may see the other's
    # result: both must report identical prior_fights/prior_wins/win_streak,
    # as of strictly before that shared date, regardless of row order.
    train = pd.DataFrame([
        make_fight("f0", "2019-01-01", "X", "W", "X"),
    ])
    same_date_a = make_fight("f1", "2020-01-01", "X", "A", "X")
    same_date_b = make_fight("f2", "2020-01-01", "X", "B", "B")
    later = make_fight("f3", "2020-02-01", "X", "C", "X")

    for ordering in [[same_date_a, same_date_b], [same_date_b, same_date_a]]:
        test = pd.DataFrame(ordering + [later])
        _, test_out = engineer_fold_features(train, test)
        by_id = test_out.set_index("fight_id")

        assert by_id.loc["f1", "r_prior_fights"] == 1
        assert by_id.loc["f2", "r_prior_fights"] == 1
        assert by_id.loc["f1", "r_prior_wins"] == 1
        assert by_id.loc["f2", "r_prior_wins"] == 1
        assert by_id.loc["f1", "r_win_streak"] == by_id.loc["f2", "r_win_streak"]
        assert by_id.loc["f1", "r_days_since_last_fight"] == by_id.loc["f2", "r_days_since_last_fight"] == 365

        # the LATER fight must see both same-date results (one win, one loss).
        assert by_id.loc["f3", "r_prior_fights"] == 3
        assert by_id.loc["f3", "r_prior_wins"] == 2


def test_unrecognized_result_status_raises_instead_of_silently_mislabeling():
    # result_status (not winner_id alone - see _add_outcome_flags) is what
    # distinguishes a loss from a draw/no-contest. A value outside
    # "win"/"draw"/"no_contest" means something upstream broke (clean_fights
    # is supposed to always produce one of these three) - fail loudly rather
    # than silently mislabel every outcome.
    bad = make_fight("f1", "2020-01-01", "X", "A", "X", result_status="bogus")
    train = pd.DataFrame([bad])

    with pytest.raises(AssertionError):
        engineer_fold_features(train, train.iloc[0:0])


def test_draw_counts_as_neither_a_win_nor_a_loss_for_win_rate():
    # X: win, then a draw -> prior_win_rate entering the 3rd fight excludes
    # the draw from the denominator (1 win / 1 decided fight = 1.0, not
    # 1/2), while prior_fights still counts all 2 and prior_draw_rate
    # reflects the draw.
    train = pd.DataFrame([
        make_fight("f1", "2020-01-01", "X", "A", "X"),
        make_fight("f2", "2020-02-01", "X", "B", None, result_status="draw"),
    ])
    test = pd.DataFrame([make_fight("f3", "2020-03-01", "X", "C", "X")])

    _, test_out = engineer_fold_features(train, test)

    f3 = test_out[test_out["fight_id"] == "f3"].iloc[0]
    assert f3["r_prior_fights"] == 2
    assert f3["r_prior_wins"] == 1
    assert f3["r_prior_win_rate"] == 1.0
    assert f3["r_prior_draw_rate"] == 0.5
    assert f3["r_prior_no_contest_rate"] == 0.0


def test_no_contest_counts_as_neither_a_win_nor_a_loss_for_win_rate():
    train = pd.DataFrame([
        make_fight("f1", "2020-01-01", "X", "A", "X"),
        make_fight("f2", "2020-02-01", "X", "B", None, result_status="no_contest"),
    ])
    test = pd.DataFrame([make_fight("f3", "2020-03-01", "X", "C", "X")])

    _, test_out = engineer_fold_features(train, test)

    f3 = test_out[test_out["fight_id"] == "f3"].iloc[0]
    assert f3["r_prior_fights"] == 2
    assert f3["r_prior_wins"] == 1
    assert f3["r_prior_win_rate"] == 1.0
    assert f3["r_prior_no_contest_rate"] == 0.5
    assert f3["r_prior_draw_rate"] == 0.0


def test_draw_and_no_contest_do_not_count_as_a_loss_for_opponent_either():
    # B was on the other side of X's draw and no-contest - neither counts
    # as a win OR a loss for B, same as for X.
    train = pd.DataFrame([
        make_fight("f1", "2020-01-01", "X", "B", None, result_status="draw"),
        make_fight("f2", "2020-02-01", "C", "B", None, result_status="no_contest"),
    ])
    test = pd.DataFrame([make_fight("f3", "2020-03-01", "B", "D", "D")])

    _, test_out = engineer_fold_features(train, test)

    f3 = test_out[test_out["fight_id"] == "f3"].iloc[0]
    assert f3["r_prior_fights"] == 2
    assert f3["r_prior_wins"] == 0
    assert pd.isna(f3["r_prior_win_rate"])  # 0 decided fights -> no rate, not 0.0
    assert f3["r_prior_draw_rate"] == 0.5
    assert f3["r_prior_no_contest_rate"] == 0.5


def test_win_loss_streaks_reset_on_a_draw_not_extended_as_a_loss():
    # X: win, win, draw, win -> entering the 4th fight, the draw must have
    # broken the win streak (not left it at 2) AND must not itself start a
    # loss streak (a draw is not a loss) - both streaks are 0 entering f4.
    train = pd.DataFrame([
        make_fight("f1", "2020-01-01", "X", "A", "X"),
        make_fight("f2", "2020-02-01", "X", "B", "X"),
        make_fight("f3", "2020-03-01", "X", "C", None, result_status="draw"),
    ])
    test = pd.DataFrame([make_fight("f4", "2020-04-01", "X", "D", "X")])

    _, test_out = engineer_fold_features(train, test)

    f4 = test_out[test_out["fight_id"] == "f4"].iloc[0]
    assert f4["r_win_streak"] == 0
    assert f4["r_loss_streak"] == 0


def test_draw_and_no_contest_stats_still_count_toward_career_totals():
    # Per the user's explicit ask: strikes/other action recorded before a
    # draw/no-contest must still accumulate into career stat totals - only
    # the win/loss outcome bookkeeping changes, not the stat volume.
    draw = make_fight("f1", "2020-01-01", "X", "A", None, result_status="draw",
                       rounds_fought=2, finish_round=2, finish_time=0)
    draw["r_total_sig_landed"] = 20  # over 1 full round = 5 min
    nc = make_fight("f2", "2020-02-01", "X", "B", None, result_status="no_contest",
                     rounds_fought=2, finish_round=2, finish_time=0)
    nc["r_total_sig_landed"] = 5  # over 1 full round = 5 min

    train = pd.DataFrame([draw, nc])
    test = pd.DataFrame([make_fight("f3", "2020-03-01", "X", "C", "X")])

    _, test_out = engineer_fold_features(train, test)

    f3 = test_out[test_out["fight_id"] == "f3"].iloc[0]
    # 20 + 5 = 25 sig str landed over 5 + 5 = 10 minutes.
    assert f3["r_sig_str_landed_per_min"] == pytest.approx(25 / 10)


def test_prior_record_counts_wins_losses_draws_and_no_contests_separately():
    # X: win, loss, draw, no-contest, DQ win (counts as an ordinary win,
    # same as any other decided fight) -> entering the 6th fight: 2 wins,
    # 1 loss, 1 draw, 1 no-contest, out of 5 prior fights.
    train = pd.DataFrame([
        make_fight("f1", "2020-01-01", "X", "A", "X"),
        make_fight("f2", "2020-02-01", "X", "B", "B"),
        make_fight("f3", "2020-03-01", "X", "C", None, result_status="draw"),
        make_fight("f4", "2020-04-01", "X", "D", None, result_status="no_contest"),
        make_fight("f5", "2020-05-01", "X", "E", "X", method_label="dq"),
    ])
    test = pd.DataFrame([make_fight("f6", "2020-06-01", "X", "F", "X")])

    _, test_out = engineer_fold_features(train, test)

    f6 = test_out[test_out["fight_id"] == "f6"].iloc[0]
    assert f6["r_prior_fights"] == 5
    assert f6["r_prior_wins"] == 2
    assert f6["r_prior_losses"] == 1
    assert f6["r_prior_draws"] == 1
    assert f6["r_prior_no_contests"] == 1


def test_finish_tendency_counts_a_draws_method_but_not_as_a_win():
    # X's prior fight was a majority-decision draw - it must count toward
    # prior_decision_majority_rate (the fight really did end that way) but
    # NOT toward prior_win_decision_majority_rate (nobody won it).
    train = pd.DataFrame([
        make_fight("f1", "2020-01-01", "X", "A", None, result_status="draw", method_label="decision_majority"),
    ])
    test = pd.DataFrame([make_fight("f2", "2020-02-01", "X", "B", "X")])

    _, test_out = engineer_fold_features(train, test)

    f2 = test_out[test_out["fight_id"] == "f2"].iloc[0]
    assert f2["r_prior_decision_majority_rate"] == 1.0
    # 0 prior wins -> 0/0, a real "no rate to compute" NaN, not a 0.0 that
    # would claim X has a known 0% win-by-this-method rate.
    assert pd.isna(f2["r_prior_win_decision_majority_rate"])


def test_early_pace_feature_still_populates_when_the_current_fight_lacks_round_detail():
    # X's prior fight (f1) has round-1 detail; X's current fight (f2) does
    # not (e.g. an old fight with no round.csv/fight_rounds.csv scrape).
    # f2 must still receive X's prior round-1 average, not NaN just because
    # f2 itself has no round-level row.
    train = pd.DataFrame([make_fight("f1", "2020-01-01", "X", "A", "X", rounds_fought=3)])
    test = pd.DataFrame([make_fight("f2", "2020-02-01", "X", "B", "X", rounds_fought=0)])
    rounds = pd.DataFrame([make_round("f1", 1, "X", "A", 10, 4)])

    fold = pd.concat([train, test])
    out = add_early_pace_features(fold, rounds)

    f2 = out[out["fight_id"] == "f2"].iloc[0]
    assert f2["r_r1_avg_sig_landed_prior"] == 10


def test_early_pace_feature_is_nan_not_zero_for_a_true_debut():
    train = pd.DataFrame([make_fight("f1", "2020-01-01", "X", "A", "X", rounds_fought=3)])
    rounds = pd.DataFrame([make_round("f1", 1, "X", "A", 10, 4)])

    out = add_early_pace_features(train, rounds)
    f1 = out[out["fight_id"] == "f1"].iloc[0]
    assert pd.isna(f1["r_r1_avg_sig_landed_prior"])


def test_feature_engineering_is_reproducible_regardless_of_input_row_order():
    rows = [
        make_fight("f1", "2020-01-01", "X", "A", "X"),
        make_fight("f2", "2020-02-01", "X", "B", "X"),
        make_fight("f3", "2020-03-01", "X", "C", "C"),
        make_fight("f4", "2020-04-01", "X", "D", "X"),
    ]
    forward, _ = engineer_fold_features(pd.DataFrame(rows), pd.DataFrame(rows).iloc[0:0])
    shuffled, _ = engineer_fold_features(pd.DataFrame(rows[::-1]), pd.DataFrame(rows[::-1]).iloc[0:0])

    a = forward.set_index("fight_id").sort_index()
    b = shuffled.set_index("fight_id").sort_index()[a.columns]
    pd.testing.assert_frame_equal(a, b)


def test_is_predictable_outcome_true_for_an_ordinary_win():
    rows = pd.DataFrame([make_fight("f1", "2020-01-01", "X", "A", "X")])
    assert list(is_predictable_outcome(rows)) == [True]


def test_is_predictable_outcome_false_for_draw_no_contest_dq_and_overturned():
    rows = pd.DataFrame([
        make_fight("f1", "2020-01-01", "X", "A", None, result_status="draw"),
        make_fight("f2", "2020-01-01", "X", "A", None, result_status="no_contest"),
        make_fight("f3", "2020-01-01", "X", "A", "X", method_label="dq"),
        make_fight("f4", "2020-01-01", "X", "A", None, result_status="no_contest", method_label="overturned"),
    ])
    assert list(is_predictable_outcome(rows)) == [False, False, False, False]


def test_stance_mismatch_flags_orthodox_vs_southpaw_only():
    same = make_fight("f1", "2020-01-01", "X", "A", "X", r_stance="Orthodox", b_stance="Orthodox")
    mismatch = make_fight("f2", "2020-01-01", "Y", "B", "Y", r_stance="Orthodox", b_stance="Southpaw")
    unknown = make_fight("f3", "2020-01-01", "Z", "C", "Z", r_stance="Orthodox", b_stance="Unknown")

    out, _ = engineer_fold_features(pd.DataFrame([same, mismatch, unknown]), pd.DataFrame([]))

    by_id = out.set_index("fight_id")["stance_mismatch"]
    assert by_id["f1"] == 0
    assert by_id["f2"] == 1
    assert by_id["f3"] == 0
