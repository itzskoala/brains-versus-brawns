import pandas as pd

# Slugified 1:1 mapping of the 11 raw `method` values - used both as a
# feature (a fighter's historical finish tendency, including how often they
# win a close split/majority decision vs. a clear unanimous one) and as a
# future prediction target. Every genuine outcome keeps its own category,
# including the rare ones (DQ, doctor stoppage, overturned, could-not-
# continue) - none of them are impossible or data errors, so none are
# collapsed into a generic "other".
METHOD_LABEL_MAP = {
    "KO/TKO": "ko_tko",
    "TKO - Doctor's Stoppage": "tko_doctor_stoppage",
    "Submission": "submission",
    "Decision - Unanimous": "decision_unanimous",
    "Decision - Split": "decision_split",
    "Decision - Majority": "decision_majority",
    "Decision": "decision_unspecified",
    "Overturned": "overturned",
    "Could Not Continue": "could_not_continue",
    "DQ": "dq",
    "Other": "other",
}
METHOD_LABELS = sorted(set(METHOD_LABEL_MAP.values()))
DECISION_LABELS = {"decision_unanimous", "decision_split", "decision_majority", "decision_unspecified"}


def parse_height_to_inches(series):
    """Convert a height column stored as text like 5' 10" into inches.
    Shared by clean_fights and src.features.snapshot, which both need to
    turn fighter.csv's raw height string into the same numeric form."""
    feet_inches = series.str.extract(r"(\d+)'\s*(\d+)")
    feet = pd.to_numeric(feet_inches[0])
    inches = pd.to_numeric(feet_inches[1])
    return feet * 12 + inches


def clean_fights(df):
    """Clean the fight-level dataset (master.csv). Returns a new DataFrame."""
    df = df.copy()

    # winner_id is null only for draws/no-contests - there is no target to
    # learn from these rows, so they are dropped rather than imputed.
    df = df[df["winner_id"].notna()]

    for col in ["r_height", "b_height"]:
        df[col] = parse_height_to_inches(df[col])

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

    # coarse finish-type grouping - see METHOD_LABEL_MAP.
    df["method_label"] = df["method"].map(METHOD_LABEL_MAP).astype("category")

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
