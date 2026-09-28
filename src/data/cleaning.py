import pandas as pd


def clean_fights(df):
    """Clean the fight-level dataset (master.csv). Returns a new DataFrame."""
    df = df.copy()

    # winner_id is null only for draws/no-contests - there is no target to
    # learn from these rows, so they are dropped rather than imputed.
    df = df[df["winner_id"].notna()]

    # height is stored as text like 5' 10" - convert to inches.
    for col in ["r_height", "b_height"]:
        feet_inches = df[col].str.extract(r"(\d+)'\s*(\d+)")
        feet = pd.to_numeric(feet_inches[0])
        inches = pd.to_numeric(feet_inches[1])
        df[col] = feet * 12 + inches

    # physical attributes: missing at random, small share of rows - median impute.
    for col in ["r_height", "b_height", "r_weight_lbs", "b_weight_lbs",
                "r_reach_inches", "b_reach_inches"]:
        df[col] = df[col].fillna(df[col].median())

    # stance is missing for ~20% of fighters, too much to impute a value -
    # keep it as its own category instead.
    df["r_stance"] = df["r_stance"].fillna("Unknown")
    df["b_stance"] = df["b_stance"].fillna("Unknown")
    df["referee"] = df["referee"].fillna("Unknown")

    # not useful for modeling - dropped rather than cleaned.
    df = df.drop(columns=[
        "r_fighter_nick_name", "b_fighter_nick_name",
        "details", "bonuses", "event_location",
    ])

    # dates were plain strings - needed as datetime for age/time-based features.
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["r_dob"] = pd.to_datetime(df["r_dob"])
    df["b_dob"] = pd.to_datetime(df["b_dob"])

    # finish_time is "mm:ss" text - convert to seconds like r_ctrl/b_ctrl.
    minutes_seconds = df["finish_time"].str.split(":", expand=True)
    minutes = pd.to_numeric(minutes_seconds[0])
    seconds = pd.to_numeric(minutes_seconds[1])
    df["finish_time"] = minutes * 60 + seconds

    # title_fight is a 0/1 flag, not a count - store as bool.
    df["title_fight"] = df["title_fight"].astype(bool)

    # low-cardinality text fields - store as category, not free-form object.
    for col in ["weight_class", "result_status", "method", "time_format",
                "referee", "r_stance", "b_stance"]:
        df[col] = df[col].astype("category")

    return df


def clean_rounds(df):
    """Clean the round-level dataset (round.csv). Returns a new DataFrame."""
    df = df.copy()

    # r_ctrl/b_ctrl ("mm:ss") are null for older fights where control time
    # simply wasn't tracked - that's unknown, not zero, so it stays missing
    # after conversion rather than being filled in.
    for col in ["r_ctrl", "b_ctrl"]:
        minutes_seconds = df[col].str.split(":", expand=True)
        minutes = pd.to_numeric(minutes_seconds[0])
        seconds = pd.to_numeric(minutes_seconds[1])
        df[col] = minutes * 60 + seconds

    return df
