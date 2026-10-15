"""Turns parsed UFCStats data (src.data.scraping.ufcstats_parse's plain
dicts) into rows for all six files under data/, and writes them - but only
after validate_new_data() has run every check and passed. Nothing here
calls DataFrame.to_csv unless the caller has already done that, and
write_all() is the only function that touches disk.

Two tiers, matching what's actually wired into the model vs. kept for
completeness:
  - fights.csv + fighter.csv + fight_rounds.csv: the ones
    src.data.ingestion/src.models.persistence actually load and train on.
  - event.csv + fight.csv + fighter_bonus.csv: the normalized layer the
    very first commit shipped (nothing in src/ or api/ reads these today -
    confirmed via grep - but they're part of "the data", so new events go
    into them too, held to the same validation gate).
"""
from pathlib import Path

import pandas as pd

from src.data.cleaning import clean_fights, clean_rounds
from src.data.ingestion import build_master_equivalent

# event.csv deliberately writes LAST: write_all() iterates this list in
# order, and a crash/kill between writes is the one way a run could still
# leave inconsistent state on disk despite per-event validation. With
# event.csv last, a crash can only ever leave fight/fighter rows on disk
# for an event that event.csv doesn't list yet - find_missing_events (keyed
# on event.csv's event_id) still calls that event missing, so the next run
# retries it; append_new/upsert_fighters are idempotent, so re-scraping it
# just fills in whatever didn't make it to disk last time. A crash could
# never produce the opposite (event.csv listing an event whose fights
# didn't make it to disk), which is the actual danger this guards against.
FILES = ["fight.csv", "fights.csv", "fight_rounds.csv", "fighter.csv", "fighter_bonus.csv", "event.csv"]

# UFCStats' own "bonuses" text join order (verified against data/fights.csv).
_BONUS_JOIN_ORDER = ["Fight of the Night", "Knockout of the Night",
                     "Performance of the Night", "Submission of the Night"]


class ValidationError(Exception):
    """Newly scraped data failed a safety check - nothing gets written."""


def load_existing(data_dir="data"):
    data_dir = Path(data_dir)
    return {name: pd.read_csv(data_dir / name) for name in FILES}


def build_fight_bundle(event_meta, event_fight_meta, fight_detail):
    """One fight's rows for every file, from:
    event_meta: {"event_id", "event_name", "event_date", "location"}
    event_fight_meta: parse_event_page()'s per-fight dict (weight_class,
        title_fight, bonus_types) - NOT corner info, see ufcstats_parse.
    fight_detail: parse_fight_page()'s return value - the only source of
        r_fighter_id/b_fighter_id/winner_id.
    Returns {"fight_row", "fights_row", "fight_rounds_rows", "bonus_rows"}.
    """
    fight_id = fight_detail["fight_id"]
    bonuses_text = ", ".join(b for b in _BONUS_JOIN_ORDER if b in event_fight_meta["bonus_types"]) or None

    fight_row = {
        "event_id": event_meta["event_id"],
        "fight_id": fight_id,
        "weight_class": event_fight_meta["weight_class"],
        "title_fight": int(event_fight_meta["title_fight"]),
        "r_id": fight_detail["r_fighter_id"],
        "b_id": fight_detail["b_fighter_id"],
        "winner_id": fight_detail["winner_id"],
        "result_status": fight_detail["result_status"],
        "method": fight_detail["method"],
        "round": fight_detail["round"],
        "time": fight_detail["time"],
        "time_format": fight_detail["time_format"],
        "referee": fight_detail["referee"],
        "details": fight_detail["details"],
    }

    fights_row = {
        "fight_id": fight_id,
        "event_id": event_meta["event_id"],
        "event_name": event_meta["event_name"],
        "event_date": event_meta["event_date"],
        "event_location": event_meta["location"],
        "weight_class": event_fight_meta["weight_class"],
        "title_fight": int(event_fight_meta["title_fight"]),
        "winner_id": fight_detail["winner_id"],
        "result_status": fight_detail["result_status"],
        "method": fight_detail["method"],
        "finish_round": fight_detail["round"],
        "finish_time": fight_detail["time"],
        "time_format": fight_detail["time_format"],
        "referee": fight_detail["referee"],
        "details": fight_detail["details"],
        "bonuses": bonuses_text,
        "r_fighter_id": fight_detail["r_fighter_id"],
        "b_fighter_id": fight_detail["b_fighter_id"],
        "rounds_fought": len(fight_detail["round_rows"]) or fight_detail["round"],
        **fight_detail["totals"],
    }

    fight_rounds_rows = []
    for rr in fight_detail["round_rows"]:
        row = dict(rr)
        row["fight_id"] = fight_id
        row["r_id"] = fight_detail["r_fighter_id"]
        row["b_id"] = fight_detail["b_fighter_id"]
        fight_rounds_rows.append(row)

    bonus_rows = [{"fight_id": fight_id, "bonus_type": b} for b in event_fight_meta["bonus_types"]]

    return {"fight_row": fight_row, "fights_row": fights_row,
            "fight_rounds_rows": fight_rounds_rows, "bonus_rows": bonus_rows}


def append_new(df, new_rows, key_cols):
    """Appends new_rows (list[dict]) to df, skipping any whose key already
    exists - the idempotency guarantee: re-running against an event already
    on disk is a no-op, not a duplicate. Returns (updated_df, num_added,
    num_skipped)."""
    if not new_rows:
        return df, 0, 0
    new_df = pd.DataFrame(new_rows)
    if isinstance(key_cols, str):
        key_cols = [key_cols]
    existing_keys = set(map(tuple, df[key_cols].values)) if len(df) else set()
    mask = ~new_df[key_cols].apply(tuple, axis=1).isin(existing_keys)
    to_add = new_df[mask]
    skipped = len(new_df) - len(to_add)
    if to_add.empty:
        return df, 0, skipped
    combined = pd.concat([df, to_add[df.columns.tolist()]], ignore_index=True)
    return combined, len(to_add), skipped


def upsert_fighters(fighter_df, new_fighter_rows):
    """Debuting fighters get appended; a fighter already in fighter.csv gets
    their row fully replaced (bio + career-rate stats are "as of latest
    scrape" - see src/data/ingestion.py's load_fighters docstring - not
    history to preserve). Returns (updated_df, num_new, num_updated)."""
    if not new_fighter_rows:
        return fighter_df, 0, 0
    new_df = pd.DataFrame(new_fighter_rows)[fighter_df.columns.tolist()]
    existing_ids = set(fighter_df["fighter_id"])
    new_ids = set(new_df["fighter_id"])
    updated_ids = existing_ids & new_ids
    added_ids = new_ids - existing_ids

    kept = fighter_df[~fighter_df["fighter_id"].isin(updated_ids)]
    combined = pd.concat([kept, new_df], ignore_index=True)
    return combined, len(added_ids), len(updated_ids)


def validate_new_data(updated):
    """Runs the exact cleaning path every historical row already goes
    through (build_master_equivalent + clean_fights for fights.csv,
    clean_rounds for fight_rounds.csv) against the full updated dataset.
    Raises ValidationError - with nothing written - if any of it breaks,
    or if a basic sanity/integrity check fails. This is the gate: a scrape
    that doesn't pass this never reaches data/*.csv."""
    fights, fighters, fight_rounds = updated["fights.csv"], updated["fighter.csv"], updated["fight_rounds.csv"]

    if not fights["fight_id"].is_unique:
        raise ValidationError("fights.csv has duplicate fight_id after merge")
    if fight_rounds.duplicated(subset=["fight_id", "round_no"]).any():
        raise ValidationError("fight_rounds.csv has duplicate (fight_id, round_no) after merge")
    if not fighters["fighter_id"].is_unique:
        raise ValidationError("fighter.csv has duplicate fighter_id after merge")

    fighter_ids = set(fighters["fighter_id"])
    for label, df, cols in [("fights.csv", fights, ["r_fighter_id", "b_fighter_id"]),
                             ("fight_rounds.csv", fight_rounds, ["r_id", "b_id"])]:
        for col in cols:
            orphans = set(df[col]) - fighter_ids
            if orphans:
                raise ValidationError(f"{label}.{col} has {len(orphans)} fighter_id(s) not in fighter.csv")

    try:
        master = build_master_equivalent(fights, fighters)
        clean_fights(master)
    except Exception as exc:  # noqa: BLE001 - any cleaning failure means "don't write"
        raise ValidationError(f"clean_fights raised on the updated dataset: {exc}") from exc

    try:
        clean_rounds(fight_rounds)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"clean_rounds raised on the updated dataset: {exc}") from exc


def write_all(data_dir, dfs):
    data_dir = Path(data_dir)
    for name, df in dfs.items():
        df.to_csv(data_dir / name, index=False)
