import pandas as pd

MAIN_DIVISIONS = ["Flyweight", "Bantamweight", "Featherweight", "Lightweight",
                   "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight"]


def add_matchup_features(df):
    """Add physical and age differences between the two fighters. These are
    static attributes known before the fight, so there is no leakage risk.
    Returns a new DataFrame."""
    df = df.copy()

    df["r_age"] = (df["event_date"] - df["r_dob"]).dt.days / 365.25
    df["b_age"] = (df["event_date"] - df["b_dob"]).dt.days / 365.25

    # ages outside a plausible fighting range point to a bad dob, not a real
    # age - treat as missing rather than guessing a corrected value.
    df.loc[(df["r_age"] < 16) | (df["r_age"] > 60), "r_age"] = float("nan")
    df.loc[(df["b_age"] < 16) | (df["b_age"] > 60), "b_age"] = float("nan")

    df["height_diff"] = df["r_height"] - df["b_height"]
    df["reach_diff"] = df["r_reach_inches"] - df["b_reach_inches"]
    df["weight_diff"] = df["r_weight_lbs"] - df["b_weight_lbs"]
    df["age_diff"] = df["r_age"] - df["b_age"]

    # reach advantage matters far more in some divisions than others - a ~40
    # point win-rate swing in Heavyweight, almost nothing in Lightweight (see
    # notebooks/multivariate_questions.ipynb). One column per division, zero
    # everywhere else, lets a model learn a separate reach effect per weight
    # class instead of one effect averaged across all of them.
    for division in MAIN_DIVISIONS:
        column = "reach_diff_x_" + division.lower().replace(" ", "_")
        df[column] = df["reach_diff"].where(df["weight_class"] == division, 0)

    return df


def add_fighter_history_features(df):
    """Add each fighter's record coming into the fight: prior fight count,
    prior wins, prior win rate, and days since their last fight. Each value
    is computed from fights strictly before the current one, so nothing
    about the fight being predicted leaks in. Returns a new DataFrame."""
    df = df.copy()

    r_side = df[["fight_id", "event_date", "r_fighter_id", "winner_id"]].rename(
        columns={"r_fighter_id": "fighter_id"})
    b_side = df[["fight_id", "event_date", "b_fighter_id", "winner_id"]].rename(
        columns={"b_fighter_id": "fighter_id"})
    long = pd.concat([r_side, b_side])
    long["is_winner"] = (long["winner_id"] == long["fighter_id"]).astype(int)
    long = long.sort_values(["fighter_id", "event_date"])

    long["prior_fights"] = long.groupby("fighter_id").cumcount()
    long["prior_wins"] = long.groupby("fighter_id")["is_winner"].cumsum() - long["is_winner"]
    long["prior_win_rate"] = long["prior_wins"] / long["prior_fights"]
    long["days_since_last_fight"] = long.groupby("fighter_id")["event_date"].diff().dt.days

    history = long[["fight_id", "fighter_id", "prior_fights", "prior_wins",
                     "prior_win_rate", "days_since_last_fight"]]

    r_history = history.rename(columns={
        "fighter_id": "r_fighter_id", "prior_fights": "r_prior_fights",
        "prior_wins": "r_prior_wins", "prior_win_rate": "r_prior_win_rate",
        "days_since_last_fight": "r_days_since_last_fight",
    })
    b_history = history.rename(columns={
        "fighter_id": "b_fighter_id", "prior_fights": "b_prior_fights",
        "prior_wins": "b_prior_wins", "prior_win_rate": "b_prior_win_rate",
        "days_since_last_fight": "b_days_since_last_fight",
    })

    df = df.merge(r_history, on=["fight_id", "r_fighter_id"], how="left")
    df = df.merge(b_history, on=["fight_id", "b_fighter_id"], how="left")

    df["experience_diff"] = df["r_prior_fights"] - df["b_prior_fights"]
    df["prior_win_rate_diff"] = df["r_prior_win_rate"] - df["b_prior_win_rate"]

    return df


def add_early_pace_features(fights_df, rounds_df):
    """Add each fighter's historical round-1 output: significant strikes
    landed in the first round, averaged over their past fights. This needs
    round.csv - master.csv only has fight totals, not per-round detail.
    Only prior fights are used, so nothing about the fight being predicted
    leaks in. Returns a new DataFrame."""
    fights_df = fights_df.copy()

    round1 = rounds_df[rounds_df["round_no"] == 1]

    r_side = round1[["fight_id", "r_id", "r_sig_landed"]].rename(
        columns={"r_id": "fighter_id", "r_sig_landed": "round1_sig_landed"})
    b_side = round1[["fight_id", "b_id", "b_sig_landed"]].rename(
        columns={"b_id": "fighter_id", "b_sig_landed": "round1_sig_landed"})
    long = pd.concat([r_side, b_side])

    long = long.merge(fights_df[["fight_id", "event_date"]], on="fight_id")
    long = long.sort_values(["fighter_id", "event_date"])

    prior_count = long.groupby("fighter_id").cumcount()
    prior_sum = long.groupby("fighter_id")["round1_sig_landed"].cumsum() - long["round1_sig_landed"]
    long["r1_avg_sig_landed_prior"] = prior_sum / prior_count

    history = long[["fight_id", "fighter_id", "r1_avg_sig_landed_prior"]]

    r_history = history.rename(columns={
        "fighter_id": "r_fighter_id", "r1_avg_sig_landed_prior": "r_r1_avg_sig_landed_prior"})
    b_history = history.rename(columns={
        "fighter_id": "b_fighter_id", "r1_avg_sig_landed_prior": "b_r1_avg_sig_landed_prior"})

    fights_df = fights_df.merge(r_history, on=["fight_id", "r_fighter_id"], how="left")
    fights_df = fights_df.merge(b_history, on=["fight_id", "b_fighter_id"], how="left")

    return fights_df


def engineer_fold_features(train, test, rounds=None):
    """Build matchup, fighter-history, and (if rounds is given) early-pace
    features for one walk-forward fold. train and test are concatenated
    before engineering so that a fighter's history/pace features - for
    their fights in test as well as train - are computed from fights up
    through this fold's test period only, never from a later fold. Returns
    (train, test) with the same rows as the inputs, features attached."""
    fold = pd.concat([train, test])

    fold = add_matchup_features(fold)
    fold = add_fighter_history_features(fold)
    if rounds is not None:
        fold = add_early_pace_features(fold, rounds)

    train_ids = set(train["fight_id"])
    train_out = fold[fold["fight_id"].isin(train_ids)]
    test_out = fold[~fold["fight_id"].isin(train_ids)]

    return train_out, test_out
