import pandas as pd

from src.data.cleaning import DECISION_LABELS, METHOD_LABELS

MAIN_DIVISIONS = ["Flyweight", "Bantamweight", "Featherweight", "Lightweight",
                   "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight"]

# Every detailed per-fight stat in master.csv (landed/attempted strikes by
# target and position, takedowns, control time, reversals, knockdowns, sub
# attempts), mapped to its r_/b_ source columns. Used to build career-to-date
# volume and style features below - one entry here is one entry in the
# resulting per-minute-rate and share-of-output feature set.
FIGHT_STAT_COLUMNS = {
    "kd": ("r_total_kd", "b_total_kd"),
    "sig_str_landed": ("r_total_sig_landed", "b_total_sig_landed"),
    "sig_str_atmp": ("r_total_sig_atmp", "b_total_sig_atmp"),
    "total_str_landed": ("r_total_total_str_landed", "b_total_total_str_landed"),
    "total_str_atmp": ("r_total_total_str_atmp", "b_total_total_str_atmp"),
    "td_success": ("r_total_td_success", "b_total_td_success"),
    "td_atmp": ("r_total_td_atmp", "b_total_td_atmp"),
    "sub_att": ("r_total_sub_att", "b_total_sub_att"),
    "rev": ("r_total_rev", "b_total_rev"),
    "sig_str_landed_head": ("r_total_sig_str_landed_head", "b_total_sig_str_landed_head"),
    "sig_str_atmp_head": ("r_total_sig_str_atmp_head", "b_total_sig_str_atmp_head"),
    "sig_str_landed_body": ("r_total_sig_str_landed_body", "b_total_sig_str_landed_body"),
    "sig_str_atmp_body": ("r_total_sig_str_atmp_body", "b_total_sig_str_atmp_body"),
    "sig_str_landed_leg": ("r_total_sig_str_landed_leg", "b_total_sig_str_landed_leg"),
    "sig_str_atmp_leg": ("r_total_sig_str_atmp_leg", "b_total_sig_str_atmp_leg"),
    "sig_str_landed_distance": ("r_total_sig_str_landed_distance", "b_total_sig_str_landed_distance"),
    "sig_str_atmp_distance": ("r_total_sig_str_atmp_distance", "b_total_sig_str_atmp_distance"),
    "sig_str_landed_clinch": ("r_total_sig_str_landed_clinch", "b_total_sig_str_landed_clinch"),
    "sig_str_atmp_clinch": ("r_total_sig_str_atmp_clinch", "b_total_sig_str_atmp_clinch"),
    "sig_str_landed_ground": ("r_total_sig_str_landed_ground", "b_total_sig_str_landed_ground"),
    "sig_str_atmp_ground": ("r_total_sig_str_atmp_ground", "b_total_sig_str_atmp_ground"),
    "ctrl_seconds": ("r_total_ctrl_seconds", "b_total_ctrl_seconds"),
}
STAT_NAMES = list(FIGHT_STAT_COLUMNS.keys())


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


def _prior_career_totals(df):
    """Build each fighter's career-to-date cumulative totals (as of just
    before the current fight) for every stat in FIGHT_STAT_COLUMNS, plus
    cumulative fight duration. Shared by add_career_stat_features so the
    long-format reshape only happens once.

    330 of 11,441 fights (rounds_fought == 0) have no round.csv detail, so
    their r_total_*/b_total_* columns are all 0 as a missingness marker, not
    a genuine zero-output fight (docs/results.md). Both the stat values and
    the duration for those fights are zeroed out here before accumulating,
    so they contribute nothing to either side of a later rate and don't
    drag a fighter's career rate down with fabricated zero-output time.

    Returns a long DataFrame: one row per fighter per fight, with
    "prior_<stat>" cumulative-sum columns and "prior_duration_min"."""
    has_detail = df["rounds_fought"] > 0
    duration_min = ((df["finish_round"] - 1) * 300 + df["finish_time"]) / 60
    duration_min = duration_min.where(has_detail, 0)

    sides = []
    for corner, id_col in [("r", "r_fighter_id"), ("b", "b_fighter_id")]:
        cols = ["fight_id", "event_date", id_col] + [FIGHT_STAT_COLUMNS[s][0 if corner == "r" else 1] for s in STAT_NAMES]
        side = df[cols].copy()
        side.columns = ["fight_id", "event_date", "fighter_id"] + STAT_NAMES
        side[STAT_NAMES] = side[STAT_NAMES].mul(has_detail.values, axis=0)
        side["duration_min"] = duration_min.values
        sides.append(side)

    long = pd.concat(sides, ignore_index=True).sort_values(["fighter_id", "event_date"])
    grouped = long.groupby("fighter_id")

    long["prior_duration_min"] = grouped["duration_min"].cumsum() - long["duration_min"]
    for stat in STAT_NAMES:
        long[f"prior_{stat}"] = grouped[stat].cumsum() - long[stat]

    return long[["fight_id", "fighter_id", "prior_duration_min"] + [f"prior_{s}" for s in STAT_NAMES]]


def add_career_stat_features(df):
    """Add career-to-date volume and style features built from every
    detailed per-fight stat (strikes by target/position, takedowns, control
    time, reversals, sub attempts, knockdowns): a per-minute rate for each
    raw stat, plus derived accuracy and output-share ratios (e.g. what share
    of a fighter's landed strikes come from the ground, a grappler/striker
    signal). All values use only fights strictly before the current one, so
    nothing about the fight being predicted leaks in. Returns a new
    DataFrame."""
    df = df.copy()
    totals = _prior_career_totals(df)

    rate_cols = {f"{stat}_per_min": totals[f"prior_{stat}"] / totals["prior_duration_min"] for stat in STAT_NAMES}

    def share(numerator, denominator):
        return totals[f"prior_{numerator}"] / totals[f"prior_{denominator}"]

    style_cols = {
        # "career_" prefix: master.csv already has a raw r_td_acc/b_td_acc
        # (fighter.csv's as-of-latest-scrape stat, not point-in-time-safe -
        # see docs/results.md) - this is a different, leakage-safe column
        # reconstructed from prior fights only, so it needs a distinct name.
        "career_sig_str_acc": share("sig_str_landed", "sig_str_atmp"),
        "career_td_acc": share("td_success", "td_atmp"),
        "head_strike_share": share("sig_str_landed_head", "sig_str_landed"),
        "body_strike_share": share("sig_str_landed_body", "sig_str_landed"),
        "leg_strike_share": share("sig_str_landed_leg", "sig_str_landed"),
        "distance_strike_share": share("sig_str_landed_distance", "sig_str_landed"),
        "clinch_strike_share": share("sig_str_landed_clinch", "sig_str_landed"),
        "ground_strike_share": share("sig_str_landed_ground", "sig_str_landed"),
    }

    history = totals[["fight_id", "fighter_id"]].assign(**rate_cols, **style_cols)
    feature_names = list(rate_cols.keys()) + list(style_cols.keys())

    r_history = history.rename(columns={"fighter_id": "r_fighter_id", **{c: f"r_{c}" for c in feature_names}})
    b_history = history.rename(columns={"fighter_id": "b_fighter_id", **{c: f"b_{c}" for c in feature_names}})

    df = df.merge(r_history, on=["fight_id", "r_fighter_id"], how="left")
    df = df.merge(b_history, on=["fight_id", "b_fighter_id"], how="left")

    for c in feature_names:
        df[f"{c}_diff"] = df[f"r_{c}"] - df[f"b_{c}"]

    return df


def add_finish_tendency_features(df):
    """Add each fighter's historical finish profile: how often their prior
    fights ended by each method (as either fighter, from method_label -
    see src/data/cleaning.py), how often they win specifically by each
    method, and the average round their non-decision fights have ended in
    (an early- vs late-finisher signal). Every method_label category gets
    its own rate - including the three decision types (unanimous/split/
    majority, e.g. a fighter who often wins a close split decision vs. one
    who wins clearly) and the rare ones (DQ, doctor stoppage, overturned,
    could-not-continue) - none are merged into a generic "other", since
    each is a real, distinct thing that happened. Only prior fights are
    used, so nothing about the fight being predicted leaks in. Returns a
    new DataFrame."""
    df = df.copy()

    r_side = df[["fight_id", "event_date", "r_fighter_id", "winner_id", "method_label", "finish_round"]].rename(
        columns={"r_fighter_id": "fighter_id"})
    b_side = df[["fight_id", "event_date", "b_fighter_id", "winner_id", "method_label", "finish_round"]].rename(
        columns={"b_fighter_id": "fighter_id"})
    long = pd.concat([r_side, b_side], ignore_index=True)
    long["is_winner"] = (long["winner_id"] == long["fighter_id"]).astype(int)
    long = long.sort_values(["fighter_id", "event_date"])

    grouped = long.groupby("fighter_id")
    prior_fights = grouped.cumcount()

    # cumulative counts of each method_label among ALL prior fights (as
    # either fighter), and among prior WINS only (how they finish, not just
    # how their fights happen to end).
    for outcome in METHOD_LABELS:
        long[f"_is_{outcome}"] = (long["method_label"] == outcome).astype(int)
        long[f"_is_win_{outcome}"] = long[f"_is_{outcome}"] * long["is_winner"]

    prior_wins = grouped["is_winner"].cumsum() - long["is_winner"]
    for outcome in METHOD_LABELS:
        prior_outcome = grouped[f"_is_{outcome}"].cumsum() - long[f"_is_{outcome}"]
        long[f"prior_{outcome}_rate"] = prior_outcome / prior_fights

        prior_win_outcome = grouped[f"_is_win_{outcome}"].cumsum() - long[f"_is_win_{outcome}"]
        long[f"prior_win_{outcome}_rate"] = prior_win_outcome / prior_wins

    # finish_round for any non-decision method, 0 elsewhere - filled rather
    # than left as NaN so a decision (or any later fight) doesn't turn every
    # subsequent cumsum for that fighter into NaN (pandas cumsum propagates
    # NaN forward from the position it occurs).
    is_finish = ~long["method_label"].isin(DECISION_LABELS)
    long["_finish_round_if_finish"] = long["finish_round"].where(is_finish, 0)
    long["_is_finish"] = is_finish.astype(int)

    prior_finish_round_sum = grouped["_finish_round_if_finish"].cumsum() - long["_finish_round_if_finish"]
    prior_finish_count = grouped["_is_finish"].cumsum() - long["_is_finish"]
    long["avg_finish_round_prior"] = prior_finish_round_sum / prior_finish_count

    feature_cols = (
        [f"prior_{o}_rate" for o in METHOD_LABELS] +
        [f"prior_win_{o}_rate" for o in METHOD_LABELS] +
        ["avg_finish_round_prior"]
    )
    history = long[["fight_id", "fighter_id"] + feature_cols]

    r_history = history.rename(columns={"fighter_id": "r_fighter_id", **{c: f"r_{c}" for c in feature_cols}})
    b_history = history.rename(columns={"fighter_id": "b_fighter_id", **{c: f"b_{c}" for c in feature_cols}})

    df = df.merge(r_history, on=["fight_id", "r_fighter_id"], how="left")
    df = df.merge(b_history, on=["fight_id", "b_fighter_id"], how="left")

    for c in feature_cols:
        df[f"{c}_diff"] = df[f"r_{c}"] - df[f"b_{c}"]

    return df


def add_recent_form_features(df):
    """Add each fighter's current win/loss streak and win rate over their
    last 5 fights, as of just before the current fight - more sensitive to
    recent form than the career-long prior_win_rate in
    add_fighter_history_features. Only prior fights are used, so nothing
    about the fight being predicted leaks in. Returns a new DataFrame."""
    df = df.copy()

    r_side = df[["fight_id", "event_date", "r_fighter_id", "winner_id"]].rename(
        columns={"r_fighter_id": "fighter_id"})
    b_side = df[["fight_id", "event_date", "b_fighter_id", "winner_id"]].rename(
        columns={"b_fighter_id": "fighter_id"})
    long = pd.concat([r_side, b_side], ignore_index=True)
    long["is_winner"] = (long["winner_id"] == long["fighter_id"]).astype(int)
    long = long.sort_values(["fighter_id", "event_date"])

    grouped = long.groupby("fighter_id")

    # length of the run of identical results (all wins or all losses) each
    # fight belongs to, then shift by one fight so a fight only reflects the
    # streak entering it, never its own result.
    is_new_block = long["is_winner"] != grouped["is_winner"].shift()
    long["_block"] = is_new_block.groupby(long["fighter_id"]).cumsum()
    long["_streak_len"] = long.groupby(["fighter_id", "_block"]).cumcount() + 1

    entering_streak = grouped["_streak_len"].shift(1).fillna(0)
    entering_was_win = grouped["is_winner"].shift(1)
    long["win_streak"] = entering_streak.where(entering_was_win == 1, 0)
    long["loss_streak"] = entering_streak.where(entering_was_win == 0, 0)

    long["form_last5_win_rate"] = grouped["is_winner"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())

    feature_cols = ["win_streak", "loss_streak", "form_last5_win_rate"]
    history = long[["fight_id", "fighter_id"] + feature_cols]

    r_history = history.rename(columns={"fighter_id": "r_fighter_id", **{c: f"r_{c}" for c in feature_cols}})
    b_history = history.rename(columns={"fighter_id": "b_fighter_id", **{c: f"b_{c}" for c in feature_cols}})

    df = df.merge(r_history, on=["fight_id", "r_fighter_id"], how="left")
    df = df.merge(b_history, on=["fight_id", "b_fighter_id"], how="left")

    for c in feature_cols:
        df[f"{c}_diff"] = df[f"r_{c}"] - df[f"b_{c}"]

    return df


def add_stance_matchup_features(df):
    """Add a flag for the classic orthodox-vs-southpaw stance mismatch,
    which changes striking angles for both fighters regardless of either
    fighter's individual stance. Static matchup attribute, no leakage risk.
    Returns a new DataFrame."""
    df = df.copy()

    orthodox_southpaw = {"Orthodox", "Southpaw"}
    is_mismatch = (
        df["r_stance"].isin(orthodox_southpaw) &
        df["b_stance"].isin(orthodox_southpaw) &
        (df["r_stance"] != df["b_stance"])
    )
    df["stance_mismatch"] = is_mismatch.astype(int)

    return df


def engineer_fold_features(train, test, rounds=None):
    """Build matchup, fighter-history, career-stat, finish-tendency,
    recent-form, stance, and (if rounds is given) early-pace features for
    one walk-forward fold. train and test are concatenated before
    engineering so that a fighter's history/pace features - for their
    fights in test as well as train - are computed from fights up through
    this fold's test period only, never from a later fold. Returns (train,
    test) with the same rows as the inputs, features attached."""
    fold = pd.concat([train, test])

    fold = add_matchup_features(fold)
    fold = add_fighter_history_features(fold)
    fold = add_career_stat_features(fold)
    fold = add_finish_tendency_features(fold)
    fold = add_recent_form_features(fold)
    fold = add_stance_matchup_features(fold)
    if rounds is not None:
        fold = add_early_pace_features(fold, rounds)

    train_ids = set(train["fight_id"])
    train_out = fold[fold["fight_id"].isin(train_ids)]
    test_out = fold[~fold["fight_id"].isin(train_ids)]

    return train_out, test_out
