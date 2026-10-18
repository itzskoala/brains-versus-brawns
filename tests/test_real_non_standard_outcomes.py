"""Regression tests anchored to real, specific historical fights - not
synthetic fixtures - for non-standard outcome handling (draw/no-contest/
DQ/overturned). See docs/results.md's "Complete fight data" decision for
the full rationale; these pin the exact real-world cases that motivated it
so a future change can't silently regress them.

Each fight_id/corner below was looked up directly from data/fights.csv
(UFCStats' own id, not something this pipeline generates) and is expected
to be stable.
"""
import pandas as pd
import pytest

from src.data.cleaning import clean_fights
from src.data.ingestion import build_master_equivalent, load_fighters, load_fights
from src.features.engineering import is_predictable_outcome

UFC_214_CORMIER_VS_JONES_2 = "5ab286735ccee803"  # no-contest, Overturned (failed drug test)
BARRY_VS_JACKSON_DQ = "7cf7960c6ccb5fa1"  # DQ win for Mike Jackson (b corner)
JACKSON_VS_MAKASHVILI_DRAW = "df8391ca39160d8c"  # Decision - Majority draw


@pytest.fixture(scope="module")
def real_fights():
    master = build_master_equivalent(load_fights(), load_fighters())
    return clean_fights(master)


def _row(real_fights, fight_id):
    match = real_fights[real_fights["fight_id"] == fight_id]
    assert len(match) == 1, f"expected exactly one row for {fight_id}, found {len(match)}"
    return match.iloc[0]


def test_ufc_214_no_contest_survives_cleaning_with_stats_preserved(real_fights):
    row = _row(real_fights, UFC_214_CORMIER_VS_JONES_2)
    assert row["result_status"] == "no_contest"
    assert pd.isna(row["winner_id"])
    assert row["method"] == "Overturned"
    # action recorded before the fight was overturned must still be here -
    # this fight went to a round-3 stoppage with real strikes landed.
    assert row["r_total_sig_landed"] > 0 or row["b_total_sig_landed"] > 0


def test_ufc_214_no_contest_excluded_from_predictable_outcomes(real_fights):
    row = _row(real_fights, UFC_214_CORMIER_VS_JONES_2)
    assert is_predictable_outcome(pd.DataFrame([row])).iloc[0] == False


def test_dq_fight_survives_cleaning_with_a_real_winner(real_fights):
    row = _row(real_fights, BARRY_VS_JACKSON_DQ)
    assert row["result_status"] == "win"
    assert row["method"] == "DQ"
    assert row["winner_id"] == row["b_fighter_id"]  # Mike Jackson, the DQ beneficiary


def test_dq_fight_excluded_from_predictable_outcomes(real_fights):
    # Hard constraint: even though a DQ has a real winner_id (unlike a
    # draw/no-contest), it is still excluded - decided by a rule
    # violation, not fighting skill.
    row = _row(real_fights, BARRY_VS_JACKSON_DQ)
    assert is_predictable_outcome(pd.DataFrame([row])).iloc[0] == False


def test_draw_survives_cleaning_with_stats_preserved(real_fights):
    row = _row(real_fights, JACKSON_VS_MAKASHVILI_DRAW)
    assert row["result_status"] == "draw"
    assert pd.isna(row["winner_id"])
    assert row["r_total_sig_landed"] > 0 or row["b_total_sig_landed"] > 0


def test_draw_excluded_from_predictable_outcomes(real_fights):
    row = _row(real_fights, JACKSON_VS_MAKASHVILI_DRAW)
    assert is_predictable_outcome(pd.DataFrame([row])).iloc[0] == False
