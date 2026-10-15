import numpy as np
import pandas as pd

from src.data.cleaning import parse_height_to_inches


def test_parse_height_to_inches_parses_feet_and_inches_text():
    heights = parse_height_to_inches(pd.Series(["5' 11\"", "6' 0\""]))
    assert heights.tolist() == [71.0, 72.0]


def test_parse_height_to_inches_full_column_with_nan_mixed_in_stays_object_dtype():
    # fighter.csv's real shape: a column of mostly text with some blank
    # (NaN) rows - already object dtype because of the strings, so this
    # case always worked even before the fix.
    series = pd.Series(["5' 10\"", np.nan, "6' 2\""])
    heights = parse_height_to_inches(series)
    assert heights.tolist()[0] == 70.0
    assert pd.isna(heights.tolist()[1])
    assert heights.tolist()[2] == 74.0


def test_parse_height_to_inches_single_nan_scalar_does_not_crash():
    # Regression test: src.features.snapshot wraps one fighter's height in
    # pd.Series([value]) to reuse this function for a single lookup. When
    # that one value is NaN (a fighter with no recorded height - ~8% of
    # fighter.csv), pandas infers float64 dtype for the one-element Series
    # (no string in the list to force object dtype), which has no `.str`
    # accessor at all - raising AttributeError instead of returning NaN.
    # This crashed /predict for any such fighter before the fix.
    heights = parse_height_to_inches(pd.Series([np.nan]))
    assert pd.isna(heights.iloc[0])


def test_parse_height_to_inches_single_none_scalar_does_not_crash():
    heights = parse_height_to_inches(pd.Series([None]))
    assert pd.isna(heights.iloc[0])


def test_parse_height_to_inches_malformed_text_is_nan_not_a_crash():
    heights = parse_height_to_inches(pd.Series(["unknown", "N/A", ""]))
    assert heights.isna().all()
