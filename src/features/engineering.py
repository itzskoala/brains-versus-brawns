import pandas as pd

from src.data.cleaning import DECISION_LABELS, METHOD_LABELS

MAIN_DIVISIONS = ["Flyweight", "Bantamweight", "Featherweight", "Lightweight",
                   "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight"]

# Every detailed per-fight stat available on the reconstructed master-shaped
# dataframe (landed/attempted strikes by target and position, takedowns,
# control time, reversals, knockdowns, sub attempts), mapped to its r_/b_
# source columns. Used to build career-to-date volume and style features
# below - one entry here is one entry in the resulting per-minute-rate and
# share-of-output feature set.
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


def _prior_sum_by_date(long, value_cols, fighter_col="fighter_id", date_col="event_date"):
    """For each row, the sum of value_cols over this fighter's STRICTLY
    earlier dates only. Rows that share a fighter's exact event_date - e.g.
    the same-card doubleheaders early tournament-format events ran, roughly
    130 fighters in this dataset - are treated as simultaneous: none of
    them can see any other same-date row's value, no matter which way
    pandas' stable sort happens to break the tie internally. Without this,
    a plain "cumsum then subtract my own value" (what every caller below
    used to do) lets whichever same-date fight sorts first leak into the
    other's "prior" features - confirmed on real data (see
    tests/test_engineering.py's same-date tests).

    Returns a DataFrame of prior_<col> columns, aligned to long's index."""
    value_cols = list(value_cols)
    per_date = long.groupby([fighter_col, date_col], as_index=False)[value_cols].sum()
    per_date = per_date.sort_values([fighter_col, date_col])

    cumsum = per_date.groupby(fighter_col)[value_cols].cumsum()
    prior_cols = [f"prior_{c}" for c in value_cols]
    prior = (cumsum - per_date[value_cols]).set_axis(prior_cols, axis=1)
    per_date = pd.concat([per_date[[fighter_col, date_col]], prior], axis=1)

    merged = long[[fighter_col, date_col]].merge(per_date, on=[fighter_col, date_col], how="left")
    merged.index = long.index
    return merged[prior_cols]


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
    is computed from fights strictly before the current one - including
    same-date fights, see _prior_sum_by_date - so nothing about the fight
    being predicted leaks in. Returns a new DataFrame."""
    df = df.copy()

    r_side = df[["fight_id", "event_date", "r_fighter_id", "winner_id"]].rename(
        columns={"r_fighter_id": "fighter_id"})
    b_side = df[["fight_id", "event_date", "b_fighter_id", "winner_id"]].rename(
        columns={"b_fighter_id": "fighter_id"})
    long = pd.concat([r_side, b_side], ignore_index=True)
    long["is_winner"] = (long["winner_id"] == long["fighter_id"]).astype(int)
    long["_one"] = 1

    prior = _prior_sum_by_date(long, ["_one", "is_winner"])
    long["prior_fights"] = prior["prior__one"]
    long["prior_wins"] = prior["prior_is_winner"]
    long["prior_win_rate"] = long["prior_wins"] / long["prior_fights"]

    # days since the fighter's most recent STRICTLY earlier date - a
    # same-date fight (see _prior_sum_by_date) shares this value with its
    # same-date sibling(s) rather than reporting 0 days for whichever one
    # happens to sort second.
    unique_dates = long[["fighter_id", "event_date"]].drop_duplicates().sort_values(
        ["fighter_id", "event_date"])
    unique_dates["_prev_date"] = unique_dates.groupby("fighter_id")["event_date"].shift(1)
    prev_date = long[["fighter_id", "event_date"]].merge(
        unique_dates, on=["fighter_id", "event_date"], how="left")["_prev_date"]
    long["days_since_last_fight"] = (long["event_date"] - prev_date.values).dt.days

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
    fight_rounds.csv - fights.csv only has fight totals, not per-round
    detail. Built from every fighter-fight appearance (not just the ones
    with round-1 detail), so a fight that itself lacks round-level data
    (see src.data.ingestion.load_fight_rounds) still gets a prior average
    from its fighter's earlier fights that do have it - a fight only drops
    out of the rolling INPUT when it lacks detail, never out of the
    feature's OUTPUT. Only prior fights are used (same-date-safe, see
    _prior_sum_by_date), so nothing about the fight being predicted leaks
    in. Returns a new DataFrame."""
    fights_df = fights_df.copy()

    round1 = rounds_df[rounds_df["round_no"] == 1]
    r_detail = round1[["fight_id", "r_id", "r_sig_landed"]].rename(
        columns={"r_id": "fighter_id", "r_sig_landed": "round1_sig_landed"})
    b_detail = round1[["fight_id", "b_id", "b_sig_landed"]].rename(
        columns={"b_id": "fighter_id", "b_sig_landed": "round1_sig_landed"})
    detail = pd.concat([r_detail, b_detail], ignore_index=True)

    r_all = fights_df[["fight_id", "event_date", "r_fighter_id"]].rename(
        columns={"r_fighter_id": "fighter_id"})
    b_all = fights_df[["fight_id", "event_date", "b_fighter_id"]].rename(
        columns={"b_fighter_id": "fighter_id"})
    long = pd.concat([r_all, b_all], ignore_index=True).merge(
        detail, on=["fight_id", "fighter_id"], how="left")

    # same zero-fill-paired-with-a-count-flag pattern as _prior_career_totals:
    # a fight with no round-1 detail contributes 0 to both the sum and the
    # count of detailed fights, so it never drags the average toward 0 or
    # produces a 0/0 that should instead be a real number.
    long["_has_detail"] = long["round1_sig_landed"].notna().astype(int)
    long["_sig_landed_filled"] = long["round1_sig_landed"].fillna(0)

    prior = _prior_sum_by_date(long, ["_has_detail", "_sig_landed_filled"])
    long["r1_avg_sig_landed_prior"] = prior["prior__sig_landed_filled"] / prior["prior__has_detail"]

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
    before the current fight, same-date-safe - see _prior_sum_by_date) for
    every stat in FIGHT_STAT_COLUMNS, plus cumulative fight duration. Shared
    by add_career_stat_features so the long-format reshape only happens
    once.

    330 of 11,441 fights (rounds_fought == 0) have no fight_rounds.csv
    detail, so their r_total_*/b_total_* columns are all 0 as a missingness
    marker, not a genuine zero-output fight (docs/results.md). Both the
    stat values and the duration for those fights are zeroed out here
    before accumulating, so they contribute nothing to either side of a
    later rate and don't drag a fighter's career rate down with fabricated
    zero-output time.

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

    long = pd.concat(sides, ignore_index=True)
    prior = _prior_sum_by_date(long, STAT_NAMES + ["duration_min"])

    return pd.concat([long[["fight_id", "fighter_id"]], prior], axis=1)


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
        # "career_" prefix: the working dataframe already has a raw
        # r_td_acc/b_td_acc (fighter.csv's as-of-latest-scrape stat, not
        # point-in-time-safe - see docs/results.md) - this is a different,
        # leakage-safe column reconstructed from prior fights only, so it
        # needs a distinct name.
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
    used (same-date-safe, see _prior_sum_by_date), so nothing about the
    fight being predicted leaks in. Returns a new DataFrame."""
    df = df.copy()

    r_side = df[["fight_id", "event_date", "r_fighter_id", "winner_id", "method_label", "finish_round"]].rename(
        columns={"r_fighter_id": "fighter_id"})
    b_side = df[["fight_id", "event_date", "b_fighter_id", "winner_id", "method_label", "finish_round"]].rename(
        columns={"b_fighter_id": "fighter_id"})
    long = pd.concat([r_side, b_side], ignore_index=True)
    long["is_winner"] = (long["winner_id"] == long["fighter_id"]).astype(int)
    long["_one"] = 1

    # cumulative counts of each method_label among ALL prior fights (as
    # either fighter), and among prior WINS only (how they finish, not just
    # how their fights happen to end).
    for outcome in METHOD_LABELS:
        long[f"_is_{outcome}"] = (long["method_label"] == outcome).astype(int)
        long[f"_is_win_{outcome}"] = long[f"_is_{outcome}"] * long["is_winner"]

    # finish_round for any non-decision method, 0 elsewhere - filled rather
    # than left as NaN so a decision (or any later fight) doesn't turn every
    # subsequent cumsum for that fighter into NaN (pandas cumsum propagates
    # NaN forward from the position it occurs).
    is_finish = ~long["method_label"].isin(DECISION_LABELS)
    long["_finish_round_if_finish"] = long["finish_round"].where(is_finish, 0)
    long["_is_finish"] = is_finish.astype(int)

    sum_cols = (["_one", "is_winner"] + [f"_is_{o}" for o in METHOD_LABELS] +
                [f"_is_win_{o}" for o in METHOD_LABELS] +
                ["_finish_round_if_finish", "_is_finish"])
    prior = _prior_sum_by_date(long, sum_cols)

    prior_fights = prior["prior__one"]
    prior_wins = prior["prior_is_winner"]
    for outcome in METHOD_LABELS:
        long[f"prior_{outcome}_rate"] = prior[f"prior__is_{outcome}"] / prior_fights
        long[f"prior_win_{outcome}_rate"] = prior[f"prior__is_win_{outcome}"] / prior_wins
    long["avg_finish_round_prior"] = prior["prior__finish_round_if_finish"] / prior["prior__is_finish"]

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

    # same-date fights (see _prior_sum_by_date) must not see each other's
    # result: override every row's "entering" snapshot with the one held by
    # the first row of its (fighter_id, event_date) group. Ties sit
    # contiguously under the stable sort above, so that first row's
    # snapshot already reflects only strictly earlier dates, regardless of
    # how the tie itself happened to order internally. Later dates still
    # progress correctly off the last row of a tied group, which already
    # folds in every same-date result.
    snapshot_cols = ["win_streak", "loss_streak", "form_last5_win_rate"]
    long["_date_group"] = long.groupby(["fighter_id", "event_date"]).ngroup()
    first_of_group = ~long.duplicated(subset=["fighter_id", "event_date"], keep="first")
    group_snapshot = long.loc[first_of_group, ["_date_group"] + snapshot_cols]
    long = long.drop(columns=snapshot_cols).merge(group_snapshot, on="_date_group", how="left")

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

    # Every is_winner computation below treats a null winner_id as a loss
    # for both fighters (NaN == anything is False) - silently wrong for a
    # draw/no-contest. clean_fights already drops rows with no winner, so
    # this should never trip; it's here so a future change that skips
    # clean_fights fails loudly instead of quietly mislabeling outcomes.
    assert fold["winner_id"].notna().all(), (
        "engineer_fold_features assumes draws/no-contests/missing winners "
        "were already dropped (see clean_fights) - a null winner_id would "
        "silently compute is_winner=False (a loss) for both fighters."
    )

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
