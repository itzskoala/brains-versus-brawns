import pandas as pd

from src.features.engineering import MAIN_DIVISIONS
from src.models.baseline import run_backtest

# The full candidate pool to select from - independent of whatever
# src.models.baseline.FEATURE_COLUMNS currently is, so this always
# re-evaluates every candidate rather than just whatever was last selected.
CANDIDATE_FEATURES = {
    "height_diff": ["height_diff"],
    "reach_diff": ["reach_diff"],
    "weight_diff": ["weight_diff"],
    "age_diff": ["age_diff"],
    "experience_diff": ["experience_diff"],
    "prior_win_rate_diff": ["prior_win_rate_diff"],
    "reach_diff_x_division": [f"reach_diff_x_{d.lower().replace(' ', '_')}" for d in MAIN_DIVISIONS],
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
