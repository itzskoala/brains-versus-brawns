import numpy as np
import pandas as pd

from src.features.imputation import apply_imputer, fit_imputer, missing_indicator_columns


def test_genuine_zero_and_missing_remain_distinguishable():
    # column median is 10, not 0, so a genuine 0 and an imputed-from-NaN
    # value land on clearly different numbers, and only the NaN row gets
    # flagged by the indicator.
    df = pd.DataFrame({"x": [10.0, 10.0, 0.0, np.nan]})
    imputer = fit_imputer(df, ["x"])

    out = apply_imputer(df, ["x"], imputer)

    assert out["x"].tolist() == [10.0, 10.0, 0.0, 10.0]
    assert out["x_was_missing"].tolist() == [0, 0, 0, 1]
    # the genuine zero is NOT flagged as missing - it's a real observed 0.
    assert out.loc[2, "x_was_missing"] == 0


def test_column_with_no_missingness_gets_no_indicator():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    imputer = fit_imputer(df, ["x"])

    out = apply_imputer(df, ["x"], imputer)

    assert "x_was_missing" not in out.columns
    assert missing_indicator_columns(imputer) == []


def test_fitting_only_on_train_and_applying_unchanged_to_test():
    # the imputer's fill value must come entirely from train - changing
    # test's own values (even its missingness) must not change what test
    # gets filled with.
    train = pd.DataFrame({"x": [10.0, 20.0, 30.0, np.nan]})
    imputer = fit_imputer(train, ["x"])

    test_a = pd.DataFrame({"x": [np.nan, 5.0]})
    test_b = pd.DataFrame({"x": [np.nan, 999.0]})  # different test data

    out_a = apply_imputer(test_a, ["x"], imputer)
    out_b = apply_imputer(test_b, ["x"], imputer)

    # both fills come from TRAIN's median (20.0), unaffected by whatever
    # test itself contains.
    assert out_a.loc[0, "x"] == 20.0
    assert out_b.loc[0, "x"] == 20.0
    assert imputer["fill_values"]["x"] == 20.0


def test_apply_imputer_does_not_mutate_the_fitted_values_from_df_passed_in():
    # a regression guard against apply_imputer silently re-deriving fill
    # values from whatever df it's called on (which would reintroduce
    # train/test leakage) instead of strictly using the passed-in imputer.
    train = pd.DataFrame({"x": [1.0, 1.0, 1.0, np.nan]})
    imputer = fit_imputer(train, ["x"])
    before = imputer["fill_values"]["x"]

    weird_test = pd.DataFrame({"x": [500.0, 500.0, np.nan]})
    apply_imputer(weird_test, ["x"], imputer)

    assert imputer["fill_values"]["x"] == before


def test_debut_fighter_ratio_is_imputed_to_a_typical_rate_not_zero():
    # the exact scenario from the audit: a ratio feature that's NaN for a
    # debut fighter (0/0) must not collapse to 0 (read as "worst possible
    # fighter") - it should land near the training fold's typical rate.
    train = pd.DataFrame({"prior_win_rate": [0.6, 0.5, 0.7, 0.4, np.nan]})
    imputer = fit_imputer(train, ["prior_win_rate"])

    out = apply_imputer(train, ["prior_win_rate"], imputer)

    debut_row = out.iloc[4]
    assert debut_row["prior_win_rate"] == 0.55  # training median, not 0
    assert debut_row["prior_win_rate_was_missing"] == 1
