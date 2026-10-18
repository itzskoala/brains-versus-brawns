import pandas as pd

from src.evaluation.debutant_bias_analysis import _bucket_label, build_calibration_table
from src.features.engineering import FIGHT_STAT_COLUMNS


def test_bucket_label_covers_every_boundary():
    assert _bucket_label(0) == "0"
    assert _bucket_label(1) == "1-2"
    assert _bucket_label(2) == "1-2"
    assert _bucket_label(3) == "3-5"
    assert _bucket_label(5) == "3-5"
    assert _bucket_label(6) == "6-10"
    assert _bucket_label(10) == "6-10"
    assert _bucket_label(11) == "11+"
    assert _bucket_label(100) == "11+"


def make_fight(fight_id, date, r_id, b_id, winner_id, method_label="decision_unanimous",
               finish_round=3, rounds_fought=3, result_status="win"):
    fight = {
        "fight_id": fight_id, "event_date": pd.Timestamp(date),
        "r_fighter_id": r_id, "b_fighter_id": b_id, "winner_id": winner_id,
        "result_status": result_status,
        "r_dob": pd.Timestamp("1990-01-01"), "b_dob": pd.Timestamp("1990-01-01"),
        "r_height": 70, "b_height": 70, "r_reach_inches": 70, "b_reach_inches": 70,
        "r_weight_lbs": 170, "b_weight_lbs": 170, "weight_class": "Lightweight",
        "r_stance": "Orthodox", "b_stance": "Orthodox",
        "method_label": method_label, "finish_round": finish_round, "time_format": "3 Rnd (5-5-5)",
        "finish_time": 300, "rounds_fought": rounds_fought,
    }
    for r_col, b_col in FIGHT_STAT_COLUMNS.values():
        fight[r_col] = 6
        fight[b_col] = 4
    return fight


def _make_multi_year_fights():
    pairs = [("A", "B"), ("C", "D"), ("E", "F"), ("B", "C")]
    rows = []
    i = 0
    for year in range(2015, 2022):
        for month, (r, b) in enumerate(pairs, start=1):
            i += 1
            rows.append(make_fight(f"f{i}", f"{year}-{month:02d}-01", r, b, winner_id=r))
    return pd.DataFrame(rows)


def test_build_calibration_table_attaches_a_real_prior_fights_to_every_prediction():
    fights = _make_multi_year_fights()
    predictions, calibration = build_calibration_table(fights, freq="Y", min_train_size=4)

    # every prediction row must have a real (non-null) prior_fights looked
    # up for its own fighter - a failed join would silently produce NaN
    # buckets instead of a loud failure.
    assert predictions["prior_fights"].notna().all()
    assert (predictions["bucket"] != "unknown").all()

    # the calibration table's row counts must exactly partition every
    # prediction - no row silently dropped or double-counted by the join.
    assert calibration["n"].sum() == len(predictions)

    for col in ("mean_predicted_prob", "actual_win_rate", "overconfidence"):
        assert col in calibration.columns
