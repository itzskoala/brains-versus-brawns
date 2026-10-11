"""One-time, repeatable migration from master.csv/round.csv to the
fights.csv/fight_rounds.csv schema (fighter.csv is unchanged).

Regenerates fights.csv and fight_rounds.csv into staging files, runs every
validation in the migration plan, and only on a full pass: moves the staging
files into place and deletes master.csv/round.csv. Any failure leaves
master.csv and round.csv untouched and exits non-zero - nothing destructive
happens unless every check passes.

Run with: python -m src.data.migrate_to_three_files
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.data.cleaning import clean_fights
from src.evaluation.feature_selection import run_ablation
from src.features.engineering import engineer_fold_features
from src.models import logistic_regression, random_forest, xgboost_model

DATA_DIR = Path("data")

# Columns that are really fighter.csv's bio/career-rate data, duplicated per
# corner per fight in master.csv - verified identical to fighter.csv's row
# for that fighter_id (see the ingestion.py consolidation work). These move
# to fighter.csv only; fights.csv keeps just the r_fighter_id/b_fighter_id
# foreign keys.
BIO_COLUMNS_PER_SIDE = [
    "fighter_name", "fighter_nick_name", "height", "weight_lbs",
    "reach_inches", "stance", "dob", "slpm", "str_acc", "sapm", "str_def",
    "td_avg", "td_acc", "td_def", "sub_avg",
]


def _bio_columns():
    return [f"{side}_{col}" for side in ("r", "b") for col in BIO_COLUMNS_PER_SIDE]


def build_fights(master):
    """fights.csv: master.csv minus the bio columns that belong in fighter.csv."""
    return master.drop(columns=_bio_columns())


def build_master_equivalent(fights, fighters):
    """Reconstruct master.csv's exact shape from fights.csv + fighter.csv,
    for feeding into clean_fights unchanged."""
    r_fighters = fighters.add_prefix("r_")
    b_fighters = fighters.add_prefix("b_")
    merged = fights.merge(r_fighters, on="r_fighter_id", how="left")
    merged = merged.merge(b_fighters, on="b_fighter_id", how="left")
    return merged


class CheckRunner:
    def __init__(self):
        self.failures = []

    def check(self, name, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
        if not condition:
            self.failures.append(name)

    def ok(self):
        return not self.failures


def validate_schema(master, rounds, fighters, fights, fight_rounds, c):
    print("\n--- 1-6: schema, keys, FKs, nulls, corners ---")

    c.check("fights.csv row count == master.csv", len(fights) == len(master),
            f"{len(fights)} vs {len(master)}")
    c.check("fight_rounds.csv row count == round.csv", len(fight_rounds) == len(rounds),
            f"{len(fight_rounds)} vs {len(rounds)}")

    expected_fights_cols = set(master.columns) - set(_bio_columns())
    c.check("fights.csv columns == master.csv minus bio columns",
            set(fights.columns) == expected_fights_cols,
            f"missing={expected_fights_cols - set(fights.columns)} "
            f"extra={set(fights.columns) - expected_fights_cols}")
    c.check("fight_rounds.csv columns == round.csv columns",
            set(fight_rounds.columns) == set(rounds.columns))

    c.check("fights.csv fight_id is unique", bool(fights["fight_id"].is_unique))
    c.check("fight_rounds.csv (fight_id, round_no) is unique",
            not fight_rounds.duplicated(subset=["fight_id", "round_no"]).any())

    fighter_ids = set(fighters["fighter_id"])
    for label, df, cols in [
        ("fights.csv", fights, ["r_fighter_id", "b_fighter_id"]),
        ("fight_rounds.csv", fight_rounds, ["r_id", "b_id"]),
    ]:
        for col in cols:
            orphans = set(df[col]) - fighter_ids
            c.check(f"{label}.{col} all present in fighter.csv.fighter_id",
                    not orphans, f"{len(orphans)} orphan ids")

    null_mismatches = []
    for col in expected_fights_cols:
        a, b = master[col].isna().sum(), fights[col].isna().sum()
        if a != b:
            null_mismatches.append((col, a, b))
    c.check("fights.csv null counts match master.csv for every shared column",
            not null_mismatches, str(null_mismatches))

    null_mismatches_r = []
    for col in rounds.columns:
        a, b = rounds[col].isna().sum(), fight_rounds[col].isna().sum()
        if a != b:
            null_mismatches_r.append((col, a, b))
    c.check("fight_rounds.csv null counts match round.csv for every column",
            not null_mismatches_r, str(null_mismatches_r))

    corner_check = master[["fight_id", "r_fighter_id", "b_fighter_id"]].merge(
        fights[["fight_id", "r_fighter_id", "b_fighter_id"]], on="fight_id",
        suffixes=("_orig", "_new"))
    corner_ok = (corner_check["r_fighter_id_orig"] == corner_check["r_fighter_id_new"]).all() and \
                (corner_check["b_fighter_id_orig"] == corner_check["b_fighter_id_new"]).all()
    c.check("fights.csv corner assignment (r_/b_fighter_id) matches master.csv row-for-row",
            bool(corner_ok))

    round_corner_check = rounds[["fight_id", "round_no", "r_id", "b_id"]].merge(
        fight_rounds[["fight_id", "round_no", "r_id", "b_id"]],
        on=["fight_id", "round_no"], suffixes=("_orig", "_new"))
    round_corner_ok = (round_corner_check["r_id_orig"] == round_corner_check["r_id_new"]).all() and \
                      (round_corner_check["b_id_orig"] == round_corner_check["b_id_new"]).all()
    c.check("fight_rounds.csv corner assignment matches round.csv row-for-row",
            bool(round_corner_ok))

    print("\n--- 7: exact reconstruction equality vs master.csv ---")
    recon = build_master_equivalent(fights, fighters)
    recon = recon[list(master.columns)].sort_values("fight_id").reset_index(drop=True)
    master_sorted = master.sort_values("fight_id").reset_index(drop=True)
    c.check("build_master_equivalent(fights.csv, fighter.csv) == master.csv exactly",
            recon.equals(master_sorted))


def validate_pipeline_parity(master, rounds, fights, fighters, fight_rounds, c):
    print("\n--- 8: cleaning + feature engineering parity (old vs new loading path) ---")

    old_cleaned = clean_fights(master).sort_values("fight_id").reset_index(drop=True)
    new_master_equiv = build_master_equivalent(fights, fighters)
    new_cleaned = clean_fights(new_master_equiv).sort_values("fight_id").reset_index(drop=True)
    new_cleaned = new_cleaned[old_cleaned.columns]
    c.check("clean_fights(old master.csv) == clean_fights(new fights.csv+fighter.csv)",
            old_cleaned.equals(new_cleaned))

    old_features, _ = engineer_fold_features(old_cleaned, old_cleaned.iloc[0:0], rounds=rounds)
    new_features, _ = engineer_fold_features(new_cleaned, new_cleaned.iloc[0:0], rounds=fight_rounds)
    old_features = old_features.sort_values("fight_id").reset_index(drop=True)
    new_features = new_features.sort_values("fight_id").reset_index(drop=True)[old_features.columns]
    c.check("engineer_fold_features identical on old vs new loading path (full feature matrix)",
            old_features.equals(new_features))


def validate_models_and_tests(master, rounds, fights, fighters, c):
    print("\n--- 9: model backtests + feature selection, old vs new loading path ---")

    old_cleaned = clean_fights(master)
    new_cleaned = clean_fights(build_master_equivalent(fights, fighters))

    for label, module in [
        ("logistic_regression", logistic_regression),
        ("random_forest", random_forest),
        ("xgboost_model", xgboost_model),
    ]:
        old_metrics = module.run_backtest(old_cleaned)
        new_metrics = module.run_backtest(new_cleaned)
        c.check(f"{label}.run_backtest identical old vs new loading path",
                old_metrics.reset_index(drop=True).equals(new_metrics.reset_index(drop=True)))

    old_ablation = run_ablation(old_cleaned)
    new_ablation = run_ablation(new_cleaned)
    c.check("feature_selection.run_ablation identical old vs new loading path",
            old_ablation.reset_index(drop=True).equals(new_ablation.reset_index(drop=True)))

    print("\n--- 10: full pytest suite ---")
    result = subprocess.run(["python3", "-m", "pytest", "-q"], capture_output=True, text=True)
    print(result.stdout[-4000:])
    if result.returncode != 0:
        print(result.stderr[-2000:])
    c.check("pytest suite passes", result.returncode == 0,
            f"exit code {result.returncode}")


def main():
    master = pd.read_csv(DATA_DIR / "master.csv")
    rounds = pd.read_csv(DATA_DIR / "round.csv")
    fighters = pd.read_csv(DATA_DIR / "fighter.csv")

    fights_staged_path = DATA_DIR / "fights.csv.staging"
    fight_rounds_staged_path = DATA_DIR / "fight_rounds.csv.staging"

    fights = build_fights(master)
    fight_rounds = rounds.copy()

    fights.to_csv(fights_staged_path, index=False)
    fight_rounds.to_csv(fight_rounds_staged_path, index=False)
    print(f"staged {len(fights)} rows -> {fights_staged_path}")
    print(f"staged {len(fight_rounds)} rows -> {fight_rounds_staged_path}")

    # re-read from disk so validation covers the actual CSV round-trip, not
    # just the in-memory DataFrame.
    fights_reloaded = pd.read_csv(fights_staged_path)
    fight_rounds_reloaded = pd.read_csv(fight_rounds_staged_path)

    c = CheckRunner()
    validate_schema(master, rounds, fighters, fights_reloaded, fight_rounds_reloaded, c)
    validate_pipeline_parity(master, rounds, fights_reloaded, fighters, fight_rounds_reloaded, c)
    validate_models_and_tests(master, rounds, fights_reloaded, fighters, c)

    print(f"\n{len(c.failures)} failing check(s)" if c.failures else "\nAll checks passed.")

    if not c.ok():
        print("Validation failed - leaving master.csv, round.csv, and staging files untouched.")
        for name in c.failures:
            print(f"  - {name}")
        return 1

    (DATA_DIR / "fights.csv").write_bytes(fights_staged_path.read_bytes())
    (DATA_DIR / "fight_rounds.csv").write_bytes(fight_rounds_staged_path.read_bytes())
    fights_staged_path.unlink()
    fight_rounds_staged_path.unlink()
    (DATA_DIR / "master.csv").unlink()
    (DATA_DIR / "round.csv").unlink()

    print("\nMigration complete. data/ now contains:")
    for p in sorted(DATA_DIR.glob("*.csv")):
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
