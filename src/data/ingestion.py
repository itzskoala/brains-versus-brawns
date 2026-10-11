"""Single place that reads the raw CSVs in data/. Everything else (models,
evaluation scripts, the API) should call load_fights()/load_fighters()/
load_fight_rounds() instead of its own pd.read_csv.

Canonical schema (master.csv and round.csv were retired by
src/data/migrate_to_three_files.py once every validation in that script
passed):
- fights.csv: one row per fight (all 11,441). Fight metadata/outcome and
  totals; r_fighter_id/b_fighter_id are foreign keys into fighter.csv, not
  bio data.
- fight_rounds.csv: one row per round (25,131, covering the 11,111 of
  11,441 fights that have round-by-round detail). r_id/b_id are foreign
  keys into fighter.csv.
- fighter.csv: one row per fighter (4,581). Bio/career-rate stats, as of
  the latest scrape.

build_master_equivalent() reconstructs the old master.csv shape in memory
(fights.csv + fighter.csv joined per corner) so src.data.cleaning.clean_fights
and everything downstream of it is unchanged.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")


def load_fights(data_dir=DATA_DIR):
    """Load fights.csv - one row per fight. Columns: fight metadata
    (fight_id, event_id, event_name, event_date, event_location,
    weight_class, title_fight, winner_id, result_status, method,
    finish_round, finish_time, time_format, referee, details, bonuses,
    rounds_fought), r_fighter_id/b_fighter_id (foreign keys into
    fighter.csv - no bio data here), and both corners' per-fight totals
    (knockdowns, sig/total strikes landed+attempted by target/position,
    takedowns, sub attempts, reversals, control seconds).
    """
    return pd.read_csv(Path(data_dir) / "fights.csv")


def load_fighters(data_dir=DATA_DIR):
    """Load fighter.csv - one row per fighter, their stats as of the latest
    scrape. Columns: fighter_id, fighter_name, fighter_nick_name, height,
    weight_lbs, reach_inches, stance, dob, slpm, str_acc, sapm, str_def,
    td_avg, td_acc, td_def, sub_avg.

    Used for live snapshots (api/main.py) and to recover bio attributes for
    fights.csv/fight_rounds.csv rows via their fighter_id foreign keys.
    """
    return pd.read_csv(Path(data_dir) / "fighter.csv")


def load_fight_rounds(data_dir=DATA_DIR):
    """Load fight_rounds.csv - one row per round per fight (fight_id,
    round_no are the key). 330 of fights.csv's 11,441 fights have no rows
    here (older fights where round-by-round detail wasn't tracked -
    fights.csv only has their fight totals). Columns: fight_id, round_no,
    r_id, b_id (fighter_id of each corner - matches fights.csv's
    r_fighter_id/b_fighter_id for every fight that has round data), then
    per-corner (r_/b_ prefixed) knockdowns, sig/total strikes
    landed+attempted by target/position, takedowns, submission attempts,
    reversals, and control time ("mm:ss" text - see clean_rounds in
    src/data/cleaning.py).
    """
    return pd.read_csv(Path(data_dir) / "fight_rounds.csv")


def build_master_equivalent(fights, fighters):
    """Reconstruct the old master.csv's exact shape (fights.csv's columns
    plus each corner's bio/career-rate stats from fighter.csv, r_/b_
    prefixed) so clean_fights - written against that shape - needs no
    changes. Verified byte-for-byte equal to the retired master.csv by
    src/data/migrate_to_three_files.py before master.csv was deleted."""
    r_fighters = fighters.add_prefix("r_")
    b_fighters = fighters.add_prefix("b_")
    merged = fights.merge(r_fighters, on="r_fighter_id", how="left")
    merged = merged.merge(b_fighters, on="b_fighter_id", how="left")
    return merged
