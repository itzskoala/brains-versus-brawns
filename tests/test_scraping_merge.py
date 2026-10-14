import pandas as pd
import pytest

from src.data.scraping.merge import (
    ValidationError,
    append_new,
    build_fight_bundle,
    upsert_fighters,
    validate_new_data,
)
from src.features.engineering import FIGHT_STAT_COLUMNS


def _minimal_valid_dataset():
    # Small but schema-complete synthetic dataset (not the real data/*.csv -
    # matching the project's existing test convention of building minimal
    # fixtures, e.g. tests/test_snapshot.py's make_fight) so
    # validate_new_data's full clean_fights/clean_rounds run stays fast and
    # isolated rather than depending on 11k+ real rows.
    fighter = pd.DataFrame([
        {"fighter_id": "r1", "fighter_name": "Red Corner", "fighter_nick_name": "",
         "height": "5' 10\"", "weight_lbs": 170, "reach_inches": 70, "stance": "Orthodox",
         "dob": "1990-01-01", "slpm": 3.0, "str_acc": 45, "sapm": 2.5, "str_def": 55,
         "td_avg": 1.0, "td_acc": 40, "td_def": 60, "sub_avg": 0.5},
        {"fighter_id": "b1", "fighter_name": "Blue Corner", "fighter_nick_name": "",
         "height": "6' 0\"", "weight_lbs": 170, "reach_inches": 72, "stance": "Switch",
         "dob": "1991-01-01", "slpm": 2.8, "str_acc": 42, "sapm": 2.7, "str_def": 50,
         "td_avg": 0.8, "td_acc": 35, "td_def": 55, "sub_avg": 0.3},
    ])

    totals = {col: 0 for pair in FIGHT_STAT_COLUMNS.values() for col in pair}
    fights = pd.DataFrame([{
        "fight_id": "f1", "event_id": "e1", "event_name": "UFC Fight Night: Test",
        "event_date": "2026-09-01", "event_location": "Las Vegas, Nevada, USA",
        "weight_class": "Lightweight", "title_fight": False, "winner_id": "r1",
        "result_status": "win", "method": "KO/TKO", "finish_round": 2, "finish_time": "1:10",
        "time_format": "3 Rnd (5-5-5)", "referee": "Keith Peterson", "details": "Punch",
        "bonuses": None, "r_fighter_id": "r1", "b_fighter_id": "b1", "rounds_fought": 2,
        **totals,
    }])

    fight_rounds = pd.DataFrame([{
        "fight_id": "f1", "round_no": 1, "r_id": "r1", "b_id": "b1",
        "r_kd": 0, "b_kd": 0, "r_sig_landed": 5, "r_sig_atmp": 10, "b_sig_landed": 3, "b_sig_atmp": 8,
        "r_total_str_landed": 5, "r_total_str_atmp": 10, "b_total_str_landed": 3, "b_total_str_atmp": 8,
        "r_td_success": 0, "r_td_atmp": 0, "b_td_success": 0, "b_td_atmp": 0,
        "r_sub_att": 0, "b_sub_att": 0, "r_rev": 0, "b_rev": 0, "r_ctrl": "0:30", "b_ctrl": "0:10",
        **{f"{side}_sig_str_{kind}_{target}": 0 for side in "rb" for kind in ("landed", "atmp")
           for target in ("head", "body", "leg", "distance", "clinch", "ground")},
    }])

    event = pd.DataFrame([{"event_id": "e1", "event_name": "UFC Fight Night: Test",
                            "date": "2026-09-01", "location": "Las Vegas, Nevada, USA"}])
    fight = pd.DataFrame([{"event_id": "e1", "fight_id": "f1", "weight_class": "Lightweight",
                            "title_fight": False, "r_id": "r1", "b_id": "b1", "winner_id": "r1",
                            "result_status": "win", "method": "KO/TKO", "round": 2, "time": "1:10",
                            "time_format": "3 Rnd (5-5-5)", "referee": "Keith Peterson", "details": "Punch"}])
    fighter_bonus = pd.DataFrame(columns=["fight_id", "bonus_type"])

    return {"event.csv": event, "fight.csv": fight, "fights.csv": fights,
            "fight_rounds.csv": fight_rounds, "fighter.csv": fighter, "fighter_bonus.csv": fighter_bonus}

EVENT_META = {"event_id": "e1", "event_name": "UFC Fight Night: Test", "event_date": "2026-09-01",
              "location": "Las Vegas, Nevada, USA"}
EVENT_FIGHT_META = {"fight_id": "f1", "weight_class": "Lightweight", "title_fight": False, "bonus_types": []}


def _fight_detail(round_rows=None):
    return {
        "fight_id": "f1", "r_fighter_id": "r1", "b_fighter_id": "b1", "winner_id": "r1",
        "result_status": "win", "method": "KO/TKO", "round": 2, "time": "1:10",
        "time_format": "3 Rnd (5-5-5)", "referee": "Keith Peterson", "details": "Punch",
        "totals": {
            "r_total_kd": 1, "b_total_kd": 0,
            "r_total_sig_landed": 10, "r_total_sig_atmp": 20, "b_total_sig_landed": 5, "b_total_sig_atmp": 15,
            "r_total_total_str_landed": 10, "r_total_total_str_atmp": 20,
            "b_total_total_str_landed": 5, "b_total_total_str_atmp": 15,
            "r_total_td_success": 0, "r_total_td_atmp": 0, "b_total_td_success": 0, "b_total_td_atmp": 0,
            "r_total_sub_att": 0, "b_total_sub_att": 0, "r_total_rev": 0, "b_total_rev": 0,
            "r_total_ctrl_seconds": 0, "b_total_ctrl_seconds": 0,
            **{f"{side}_total_sig_str_landed_{t}": 0 for side in "rb" for t in
               ("head", "body", "leg", "distance", "clinch", "ground")},
            **{f"{side}_total_sig_str_atmp_{t}": 0 for side in "rb" for t in
               ("head", "body", "leg", "distance", "clinch", "ground")},
        },
        "round_rows": round_rows or [],
    }


def test_build_fight_bundle_fight_row():
    bundle = build_fight_bundle(EVENT_META, EVENT_FIGHT_META, _fight_detail())
    assert bundle["fight_row"]["r_id"] == "r1"
    assert bundle["fight_row"]["b_id"] == "b1"
    assert bundle["fight_row"]["winner_id"] == "r1"
    assert bundle["fight_row"]["title_fight"] == 0


def test_build_fight_bundle_fights_row_has_totals_and_rounds_fought():
    bundle = build_fight_bundle(EVENT_META, EVENT_FIGHT_META, _fight_detail())
    assert bundle["fights_row"]["r_fighter_id"] == "r1"
    assert bundle["fights_row"]["r_total_sig_landed"] == 10
    assert bundle["fights_row"]["rounds_fought"] == 2  # no round_rows -> falls back to finish round


def test_build_fight_bundle_bonus_text_joined_in_canonical_order():
    meta = dict(EVENT_FIGHT_META, bonus_types=["Performance of the Night", "Fight of the Night"])
    bundle = build_fight_bundle(EVENT_META, meta, _fight_detail())
    assert bundle["fights_row"]["bonuses"] == "Fight of the Night, Performance of the Night"
    assert bundle["bonus_rows"] == [
        {"fight_id": "f1", "bonus_type": "Performance of the Night"},
        {"fight_id": "f1", "bonus_type": "Fight of the Night"},
    ]


def test_build_fight_bundle_no_bonuses_is_none():
    bundle = build_fight_bundle(EVENT_META, EVENT_FIGHT_META, _fight_detail())
    assert bundle["fights_row"]["bonuses"] is None


def test_build_fight_bundle_round_rows_get_fight_and_corner_ids():
    round_rows = [{"round_no": 1, "r_sig_landed": 5}]
    bundle = build_fight_bundle(EVENT_META, EVENT_FIGHT_META, _fight_detail(round_rows))
    assert bundle["fight_rounds_rows"][0]["fight_id"] == "f1"
    assert bundle["fight_rounds_rows"][0]["r_id"] == "r1"
    assert bundle["fight_rounds_rows"][0]["b_id"] == "b1"


def test_append_new_is_idempotent_on_rerun():
    df = pd.DataFrame([{"fight_id": "existing", "x": 1}])
    new_rows = [{"fight_id": "existing", "x": 1}, {"fight_id": "new", "x": 2}]
    updated, added, skipped = append_new(df, new_rows, "fight_id")
    assert added == 1
    assert skipped == 1
    assert set(updated["fight_id"]) == {"existing", "new"}

    # running the exact same new_rows again against the already-updated df adds nothing
    updated2, added2, skipped2 = append_new(updated, new_rows, "fight_id")
    assert added2 == 0
    assert skipped2 == 2
    assert len(updated2) == len(updated)


def test_append_new_composite_key():
    df = pd.DataFrame([{"fight_id": "f1", "round_no": 1, "v": 1}])
    new_rows = [{"fight_id": "f1", "round_no": 1, "v": 1}, {"fight_id": "f1", "round_no": 2, "v": 2}]
    updated, added, skipped = append_new(df, new_rows, ["fight_id", "round_no"])
    assert added == 1
    assert skipped == 1


def test_upsert_fighters_adds_new_and_overwrites_existing():
    existing = pd.DataFrame([{"fighter_id": "a", "fighter_name": "Old Name", "slpm": 1.0}])
    new_rows = [
        {"fighter_id": "a", "fighter_name": "New Name", "slpm": 2.0},
        {"fighter_id": "b", "fighter_name": "Debut Fighter", "slpm": 3.0},
    ]
    updated, n_new, n_updated = upsert_fighters(existing, new_rows)
    assert n_new == 1
    assert n_updated == 1
    assert len(updated) == 2
    assert updated.set_index("fighter_id").loc["a", "fighter_name"] == "New Name"


def test_validate_new_data_passes_on_a_well_formed_dataset():
    validate_new_data(_minimal_valid_dataset())  # should not raise


def test_validate_new_data_rejects_duplicate_fight_id():
    dfs = _minimal_valid_dataset()
    dfs["fights.csv"] = pd.concat([dfs["fights.csv"], dfs["fights.csv"].iloc[[0]]], ignore_index=True)
    with pytest.raises(ValidationError):
        validate_new_data(dfs)


def test_validate_new_data_rejects_duplicate_round_no():
    dfs = _minimal_valid_dataset()
    dfs["fight_rounds.csv"] = pd.concat(
        [dfs["fight_rounds.csv"], dfs["fight_rounds.csv"].iloc[[0]]], ignore_index=True)
    with pytest.raises(ValidationError):
        validate_new_data(dfs)


def test_validate_new_data_rejects_orphan_fighter_id():
    dfs = _minimal_valid_dataset()
    dfs["fights.csv"].loc[0, "r_fighter_id"] = "does-not-exist-in-fighter-csv"
    with pytest.raises(ValidationError):
        validate_new_data(dfs)


def test_validate_new_data_rejects_malformed_control_time():
    # clean_rounds splits r_ctrl/b_ctrl on ":" - a scrape that produced
    # garbage instead of "mm:ss" must fail validation, not silently corrupt
    # downstream feature engineering.
    dfs = _minimal_valid_dataset()
    dfs["fight_rounds.csv"].loc[0, "r_ctrl"] = "not-a-time"
    with pytest.raises(ValidationError):
        validate_new_data(dfs)
