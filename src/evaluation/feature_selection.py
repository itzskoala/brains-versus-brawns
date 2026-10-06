import pandas as pd

from src.data.cleaning import METHOD_LABELS
from src.features.engineering import MAIN_DIVISIONS, STAT_NAMES
from src.models.logistic_regression import run_backtest

# The full candidate pool to select from - independent of whatever
# src.models.logistic_regression.FEATURE_COLUMNS currently is, so this
# always re-evaluates every candidate rather than just whatever was last
# selected.
#
# Grouped by the engineering.py function that produces them (one group per
# add_*_features call), not one entry per column - ablating ~70 individual
# diff columns one at a time would be both slow and hard to read. Most
# entries are _diff columns (antisymmetric r-minus-b diffs, correctly
# negated by _symmetrize for the blue-perspective row). stance_mismatch is
# the one exception - it's corner-symmetric (true regardless of which
# fighter is red/blue) - so it's listed in
# src.models.logistic_regression.SYMMETRIC_FEATURE_COLUMNS, which tells
# _symmetrize to leave it unchanged for the blue-perspective row instead of
# negating it.
#
# One reference category is dropped from each mutually-exclusive share/rate
# group below (method_label's 11 categories always sum to ~1 per fighter,
# and likewise the head/body/leg and distance/clinch/ground strike-target
# shares) - confirmed by direct check to be an exact linear dependency
# (row-sum ~0 up to float error), which made the logistic regression in
# run_backtest diverge (RuntimeWarning: overflow in matmul) on the smallest
# early walk-forward folds. Dropping the reference category is the standard
# fix (k-1 encoding) and loses no information: the intercept plus the
# remaining k-1 coefficients fully capture the same decision boundary as
# using all k. "decision_unanimous" (the most common method) and
# "head"/"distance" (the most common strike target/position) are the
# categories dropped, so the remaining coefficients read as relative to
# that baseline.
#
# "overturned" and "could_not_continue" are dropped outright (not just as a
# reference) rather than renamed as a baseline: both methods only ever occur
# in no_contest/draw fights, which clean_fights already drops for having no
# winner to learn from, so their prior-rate columns are a literal constant
# zero for every fighter in this dataset - a degenerate column, not a
# genuine-but-rare signal.
#
# "other" is dropped too, for a related but distinct reason: it's a real
# category (occurs twice, 2000 and 2007, each with a winner - confirmed by
# direct lookup) rather than structurally impossible, but with only 2
# fights in the entire 30+ year dataset, at most 4 fighters will ever carry
# a nonzero prior_other_rate, in only their own fights after that. That
# makes it an effective constant-zero column through nearly every
# walk-forward fold - the exact same singularity as overturned/
# could_not_continue, just data-dependent rather than guaranteed by
# clean_fights. If "other" becomes more common in future data this exclusion
# should be revisited - method_label itself still tracks it correctly either
# way, this only concerns the ablation/selection candidate list.
_METHOD_LABELS_FOR_RATES = [m for m in METHOD_LABELS
                            if m not in ("decision_unanimous", "overturned", "could_not_continue", "other")]

# sig_str_landed is exactly the sum of its head/body/leg breakdown, and
# separately the exact sum of its distance/clinch/ground breakdown (same
# total, two partitions) - same for sig_str_atmp. Keeping all of
# {total, head, body, leg, distance, clinch, ground} is 7 columns for 2
# independent equations (rank 5), another exact dependency confirmed by
# direct check. Dropping the total and the ground component from each
# (landed and atmp) resolves it while keeping leg strikes - explicitly
# called out as a feature of interest - as its own named column; ground is
# still represented separately via ground_strike_share in career_style_stats.
_VOLUME_STATS = [s for s in STAT_NAMES
                  if s not in ("sig_str_landed", "sig_str_atmp", "sig_str_landed_ground", "sig_str_atmp_ground")]

CANDIDATE_FEATURES = {
    "height_diff": ["height_diff"],
    "reach_diff": ["reach_diff"],
    "weight_diff": ["weight_diff"],
    "age_diff": ["age_diff"],
    "experience_diff": ["experience_diff"],
    "prior_win_rate_diff": ["prior_win_rate_diff"],
    "reach_diff_x_division": [f"reach_diff_x_{d.lower().replace(' ', '_')}" for d in MAIN_DIVISIONS],
    "career_volume_stats": [f"{stat}_per_min_diff" for stat in _VOLUME_STATS],
    "career_style_stats": [
        "career_sig_str_acc_diff", "career_td_acc_diff",
        "body_strike_share_diff", "leg_strike_share_diff",
        "clinch_strike_share_diff", "ground_strike_share_diff",
    ],
    "finish_tendency": (
        [f"prior_{o}_rate_diff" for o in _METHOD_LABELS_FOR_RATES] +
        [f"prior_win_{o}_rate_diff" for o in _METHOD_LABELS_FOR_RATES] +
        ["avg_finish_round_prior_diff"]
    ),
    "recent_form": ["win_streak_diff", "loss_streak_diff", "form_last5_win_rate_diff"],
    "stance_mismatch": ["stance_mismatch"],
}
ALL_CANDIDATE_COLUMNS = [c for cols in CANDIDATE_FEATURES.values() for c in cols]


def run_ablation(fights, rounds=None):
    """Leave-one-feature-out check: for each feature, remove it and compare
    mean walk-forward accuracy to the full candidate set. accuracy_drop is
    the full model's accuracy minus the accuracy with that feature removed -
    a positive value means the feature helps; negative or ~zero means it
    doesn't. Returns a DataFrame, most helpful first."""
    baseline_acc = run_backtest(fights, rounds=rounds, feature_columns=ALL_CANDIDATE_COLUMNS)["accuracy"].mean()

    rows = [{"feature": "(full feature set)", "accuracy": baseline_acc, "accuracy_drop": 0.0}]

    for name, cols in CANDIDATE_FEATURES.items():
        kept = [c for c in ALL_CANDIDATE_COLUMNS if c not in cols]
        acc = run_backtest(fights, rounds=rounds, feature_columns=kept)["accuracy"].mean()
        rows.append({"feature": name, "accuracy": acc, "accuracy_drop": baseline_acc - acc})

    result = pd.DataFrame(rows)
    return result.sort_values("accuracy_drop", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from src.data.cleaning import clean_fights

    master = pd.read_csv("data/master.csv")
    fights = clean_fights(master)

    print(run_ablation(fights).to_string(index=False))
