"""One-off analysis (not part of the training/serving pipeline - same
category as feature_selection.py/missing_value_comparison.py/
representation_comparison.py): is the model's predicted win probability
systematically too confident for a fighter with few prior UFC fights?

Uses ONLY already-correct, already-leakage-safe data: logistic_regression's
walk-forward run_backtest(return_predictions=True) (every prediction is
genuinely out-of-sample - the model that produced it never saw that fold's
test year during training) joined back to each fighter's own point-in-time
prior_fights count via engineer_fold_features run once over the full
history (the same technique api/main.py's load_state() uses for
fights_with_features - a real fight's own prior-history features don't
change depending on which later fold happens to include it, so this join
introduces no leakage of its own).

Usage: python -m src.evaluation.debutant_bias_analysis
"""
import pandas as pd

from src.data.cleaning import clean_fights
from src.data.ingestion import build_master_equivalent, load_fighters, load_fights
from src.features.engineering import engineer_fold_features
from src.models import logistic_regression

PRIOR_FIGHTS_BUCKETS = [(0, 0), (1, 2), (3, 5), (6, 10), (11, None)]


def _bucket_label(n):
    for lo, hi in PRIOR_FIGHTS_BUCKETS:
        if hi is None and n >= lo:
            return f"{lo}+"
        if hi is not None and lo <= n <= hi:
            return f"{lo}-{hi}" if lo != hi else str(lo)
    return "unknown"


def build_calibration_table(fights, **run_backtest_kwargs):
    """Returns (predictions_with_prior_fights, calibration_by_bucket).
    predictions_with_prior_fights: every out-of-sample prediction row
    (one per fight per corner perspective - see _symmetrize) with its own
    fighter's prior_fights attached. calibration_by_bucket: one row per
    prior_fights bucket with n, mean_predicted_prob, actual_win_rate, and
    their difference (positive = overconfident: the model predicts a
    higher win probability than that bucket's fighters actually achieve).
    run_backtest_kwargs passes through to logistic_regression.run_backtest
    (e.g. freq/min_train_size for a small test fixture)."""
    _, predictions = logistic_regression.run_backtest(
        fights, return_predictions=True, **run_backtest_kwargs)

    fights_with_features, _ = engineer_fold_features(fights, fights.iloc[0:0])
    long_prior_fights = pd.concat([
        fights_with_features[["fight_id", "r_fighter_id", "r_prior_fights"]].rename(
            columns={"r_fighter_id": "fighter_id", "r_prior_fights": "prior_fights"}),
        fights_with_features[["fight_id", "b_fighter_id", "b_prior_fights"]].rename(
            columns={"b_fighter_id": "fighter_id", "b_prior_fights": "prior_fights"}),
    ], ignore_index=True)

    merged = predictions.merge(long_prior_fights, on=["fight_id", "fighter_id"], how="left")
    merged["bucket"] = merged["prior_fights"].apply(_bucket_label)

    rows = []
    bucket_order = [_bucket_label(lo) for lo, _ in PRIOR_FIGHTS_BUCKETS]
    for bucket in bucket_order:
        group = merged[merged["bucket"] == bucket]
        if group.empty:
            continue
        mean_predicted = group["y_prob"].mean()
        actual_win_rate = group["y_true"].mean()
        rows.append({
            "prior_fights_bucket": bucket,
            "n": len(group),
            "mean_predicted_prob": mean_predicted,
            "actual_win_rate": actual_win_rate,
            "overconfidence": mean_predicted - actual_win_rate,
        })
    calibration = pd.DataFrame(rows)
    return merged, calibration


if __name__ == "__main__":
    master = build_master_equivalent(load_fights(), load_fighters())
    fights = clean_fights(master)
    print(f"[debutant_bias_analysis] {len(fights)} fights after cleaning; running backtest")

    _, calibration = build_calibration_table(fights)
    print()
    print(calibration.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
