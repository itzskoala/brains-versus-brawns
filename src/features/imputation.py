"""Leakage-safe alternative to filling every missing feature with 0.

A debut (or early-career) fighter's prior_win_rate, career_td_acc, and
every per-minute/share ratio built in src.features.engineering are
genuinely undefined (0/0), not 0 - see _prior_career_totals and
add_career_stat_features. Every model's run_backtest and
src.evaluation.nested_validation currently call plain .fillna(0) on the
full feature matrix before fitting, which encodes "unknown" as "the worst
possible fighter" for every ratio feature (a career win rate of 0%, a
striking accuracy of 0%, etc.) - a real, systematic bias, not a neutral
default.

fit_imputer/apply_imputer follow the exact same fit-on-train-only contract
as src.features.scaling.fit_scaler/apply_scaler: fit once per fold on the
training rows, then apply the same fitted values to train, test, and
serving. The fill value is each column's training-fold median (a typical
fighter's rate, not a zero), and a <col>_was_missing indicator is added for
every column that had any missingness in the training fold, so the model
can still tell a genuine 0 apart from an imputed unknown - zero-filling
alone collapses that distinction permanently.
"""

import pandas as pd


def fit_imputer(df, columns):
    """Fit on the training split only - fitting on validation/test data
    leaks information into it, the same rule fit_scaler follows. Returns
    {"fill_values": per-column median (a Series), "missing_columns": the
    subset of columns that had at least one missing value in df} - only
    those columns get a missingness indicator in apply_imputer, so a
    column that's never missing doesn't pick up a useless always-zero
    indicator."""
    fill_values = df[columns].median()
    has_missing = df[columns].isna().any()
    missing_columns = [c for c in columns if has_missing[c]]
    return {"fill_values": fill_values, "missing_columns": missing_columns}


def apply_imputer(df, columns, imputer):
    """Apply an already-fit imputer. Returns a new DataFrame with: the
    given columns' missing values filled with the training fold's
    per-column median (not 0), plus one new <col>_was_missing column (1 if
    that row was missing before filling, else 0) for every column
    fit_imputer found to have missingness in the training fold. Use the
    same imputer (fit on train) for validation, test, and serving, the
    same rule apply_scaler follows - this function never looks at df's own
    values to decide the fill value, only imputer's."""
    df = df.copy()
    indicators = pd.DataFrame({
        f"{c}_was_missing": df[c].isna().astype(int) for c in imputer["missing_columns"]
    }, index=df.index)
    df[columns] = df[columns].fillna(imputer["fill_values"])
    return pd.concat([df, indicators], axis=1)


def missing_indicator_columns(imputer):
    """The <col>_was_missing column names apply_imputer will add for this
    fitted imputer - use to extend a feature_columns list before fitting a
    model on the imputed frame."""
    return [f"{c}_was_missing" for c in imputer["missing_columns"]]
