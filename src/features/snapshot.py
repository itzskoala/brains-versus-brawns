import pandas as pd

from src.data.cleaning import DECISION_LABELS, METHOD_LABELS, parse_height_to_inches
from src.features.engineering import FIGHT_STAT_COLUMNS, MAIN_DIVISIONS, engineer_fold_features
from src.models.logistic_regression import PAIRED_BASES
from src.models.logistic_regression import _BASE_KEY_OVERRIDES as _PAIRED_KEY_OVERRIDES

# A handful of FEATURE_COLUMNS names don't match their snapshot source key
# by simply stripping "_diff" (see add_fighter_history_features in
# engineering.py: experience_diff is r_prior_fights - b_prior_fights, not a
# "r_experience" column) - spelled out explicitly rather than guessed.
_DIFF_KEY_OVERRIDES = {"experience_diff": "prior_fights"}

# reach_inches_x_<division> is in PAIRED_BASES (so self_/opp_ versions
# exist for the individual/combined representation) but, unlike every
# other PAIRED_BASES entry, has no matching key in a fighter snapshot -
# src.models.logistic_regression._add_reach_division_interactions builds
# it from reach_inches + weight_class at symmetrize time, not inside
# engineer_fold_features. build_matchup_features computes it the same way,
# so it's handled as its own case below rather than the generic
# self_/opp_ lookup every other PAIRED_BASES entry uses.

_DIVISION_SUFFIXES = {d.lower().replace(" ", "_"): d for d in MAIN_DIVISIONS}

# Sentinel opponent id for the synthetic "phantom fight" appended per
# snapshot - never collides with a real fighter_id, and only ever appears
# once (per call), so it accumulates no history of its own to contaminate
# anything.
_PLACEHOLDER_FIGHTER_ID = "__snapshot_placeholder_opponent__"


def fighter_fight_years(fights, fighter_id):
    """Sorted distinct calendar years in which fighter_id actually fought -
    used to populate a "pick a year" control with only real, selectable
    years rather than an arbitrary range."""
    in_fight = (fights["r_fighter_id"] == fighter_id) | (fights["b_fighter_id"] == fighter_id)
    years = fights.loc[in_fight, "event_date"].dt.year.unique()
    return sorted(int(y) for y in years)


def latest_fight_date(fights, fighter_id, year=None):
    """The cutoff date for a fighter's snapshot: their latest fight overall
    (year=None), or their latest fight within the given calendar year -
    "Jon Jones @ 2015" means Jones's own last fight that happened in 2015,
    not January 1st of that year. Raises ValueError if the fighter has no
    qualifying fight."""
    in_fight = (fights["r_fighter_id"] == fighter_id) | (fights["b_fighter_id"] == fighter_id)
    if year is not None:
        in_fight &= fights["event_date"].dt.year == year

    dates = fights.loc[in_fight, "event_date"]
    if dates.empty:
        if year is not None:
            raise ValueError(f"fighter {fighter_id} has no fight recorded in {year}")
        raise ValueError(f"fighter {fighter_id} has no fight history")
    return dates.max()


def _static_attributes(fighters_df, fighter_id, fallbacks):
    """A fighter's physical attributes as of the only fighter.csv record
    there is for them (height/weight/reach/stance/dob are treated as
    static, same as clean_fights does for every real fight row). Missing
    values fall back to the same training-set medians clean_fights used,
    so a snapshot with a missing attribute behaves the same way a real
    fight row would."""
    match = fighters_df.loc[fighters_df["fighter_id"] == fighter_id]
    if match.empty:
        raise ValueError(f"unknown fighter_id {fighter_id}")
    row = match.iloc[0]

    height = parse_height_to_inches(pd.Series([row["height"]])).iloc[0]
    if pd.isna(height):
        height = fallbacks["height"]

    weight = row["weight_lbs"] if pd.notna(row["weight_lbs"]) else fallbacks["weight_lbs"]
    reach = row["reach_inches"] if pd.notna(row["reach_inches"]) else fallbacks["reach_inches"]
    stance = row["stance"] if pd.notna(row["stance"]) else "Unknown"
    dob = pd.to_datetime(row["dob"]) if pd.notna(row["dob"]) else pd.NaT

    return {"height": height, "weight_lbs": weight, "reach_inches": reach, "stance": stance, "dob": dob}


def build_fighter_snapshot(fights, fighters_df, fighter_id, as_of_date, weight_class):
    """Build one fighter's point-in-time feature snapshot as of as_of_date,
    using only their real fights strictly before it. Returns a dict of the
    r_-prefixed feature values engineer_fold_features would have computed
    for a real fight on that date (age, prior record, career stats, finish
    tendency, recent form, stance, static attributes).

    Implementation: append one synthetic "phantom fight" row - fighter_id
    vs. a placeholder opponent, dated an instant after as_of_date - to the
    real fights table and run it through the unmodified
    engineer_fold_features. engineering.py's prior-feature helpers treat
    same-date rows as simultaneous (neither can see the other, see
    _prior_sum_by_date) so the offset makes this row unambiguously the
    first fight *after* as_of_date rather than tied with a real fight that
    happened to land exactly on it - which is what "data valid through
    as_of_date" (this function's documented, tested cutoff semantics)
    requires: a real fight dated as_of_date must count as prior, while one
    dated any later must not. The fighter's real fights after as_of_date
    are still automatically excluded by event_date ordering, with no need
    to filter the input; the synthetic row's own stat contribution is
    explicitly zeroed (not left NaN) so it cancels itself out exactly in
    that row's own cumsum() - self computation rather than corrupting it.
    Only the synthetic row's r_* columns are ever read back - the
    placeholder's b_* side is discarded."""
    in_fight = (fights["r_fighter_id"] == fighter_id) | (fights["b_fighter_id"] == fighter_id)
    if not (in_fight & (fights["event_date"] <= as_of_date)).any():
        raise ValueError(f"no data for fighter {fighter_id} by {as_of_date.date()}")

    fallbacks = {
        "height": fights["r_height"].median(),
        "weight_lbs": fights["r_weight_lbs"].median(),
        "reach_inches": fights["r_reach_inches"].median(),
    }
    attrs = _static_attributes(fighters_df, fighter_id, fallbacks)

    # FIGHT_STAT_COLUMNS values are already the full r_/b_ column names
    # (e.g. "r_total_kd", "b_total_kd") - zero every one of them so the
    # synthetic row's own (fake) stat contribution exactly cancels out in
    # its own cumsum() - self computation instead of leaving NaN (NaN * 0
    # is NaN, not 0 - see _prior_career_totals' has_detail masking).
    zero_stats = {col: 0 for pair in FIGHT_STAT_COLUMNS.values() for col in pair}

    synthetic_row = {
        "fight_id": "__snapshot__",
        "event_date": as_of_date + pd.Timedelta(seconds=1),
        "weight_class": weight_class,
        "winner_id": _PLACEHOLDER_FIGHTER_ID + "_winner_never_used",
        "title_fight": False,
        "finish_round": 1,
        "finish_time": 0,
        "rounds_fought": 0,
        "method_label": METHOD_LABELS[0],
        "r_fighter_id": fighter_id,
        "r_height": attrs["height"],
        "r_weight_lbs": attrs["weight_lbs"],
        "r_reach_inches": attrs["reach_inches"],
        "r_stance": attrs["stance"],
        "r_dob": attrs["dob"],
        "b_fighter_id": _PLACEHOLDER_FIGHTER_ID,
        "b_height": None,
        "b_weight_lbs": None,
        "b_reach_inches": None,
        "b_stance": "Unknown",
        "b_dob": pd.NaT,
        **zero_stats,
    }

    fold = pd.concat([fights, pd.DataFrame([synthetic_row])], ignore_index=True)
    fold, _ = engineer_fold_features(fold, fold.iloc[0:0])
    snapshot_row = fold[fold["fight_id"] == "__snapshot__"].iloc[0]

    return {col[2:]: snapshot_row[col] for col in snapshot_row.index if col.startswith("r_")}


def build_matchup_features(red_snapshot, blue_snapshot, weight_class, feature_columns, imputer=None):
    """Assemble exactly the requested feature_columns (a saved model's own
    metadata.feature_columns - different saved artifacts want different
    sets, see models/*.json) from two independent fighter snapshots, the
    same way engineer_fold_features/_symmetrize would for a real fight at
    a single shared event_date:
    - "<base>_diff", "stance_mismatch", "reach_diff_x_<division>": as
      before (corner differences/the one symmetric flag).
    - "self_<base>"/"opp_<base>" for base in
      src.models.logistic_regression.PAIRED_BASES: self_ reads
      red_snapshot's own value, opp_ reads blue_snapshot's - red is always
      "self" here, since (unlike a real symmetrized training row) a live
      matchup request has no second, mirrored "blue is self" row to
      compute too.
    - "self_reach_inches_x_<division>"/"opp_reach_inches_x_<division>":
      the per-corner reach-by-division interaction (see the module-level
      note on reach_inches_x_<division> above).
    - "<col>_was_missing": 1 if <col> was NaN before filling, 0 otherwise -
      request it alongside <col> itself, same pairing
      src.features.imputation produces at training time.
    Missing values are filled from imputer's training-fold fill values
    when one is given (matching how the model itself was trained);
    without one (an artifact saved before src.features.imputation
    existed), falls back to 0.0, matching the old flat fillna(0). Returns
    a plain dict, one entry per requested column."""
    orthodox_southpaw = {"Orthodox", "Southpaw"}

    def reach_division_value(snapshot, suffix):
        return snapshot["reach_inches"] if weight_class == _DIVISION_SUFFIXES[suffix] else 0.0

    base_features = {}
    indicator_requests = []

    for col in feature_columns:
        if col.endswith("_was_missing"):
            indicator_requests.append(col)
        elif col == "stance_mismatch":
            base_features[col] = int(
                red_snapshot["stance"] in orthodox_southpaw
                and blue_snapshot["stance"] in orthodox_southpaw
                and red_snapshot["stance"] != blue_snapshot["stance"]
            )
        elif col.startswith("reach_diff_x_") and col[len("reach_diff_x_"):] in _DIVISION_SUFFIXES:
            division = _DIVISION_SUFFIXES[col[len("reach_diff_x_"):]]
            reach_diff = red_snapshot["reach_inches"] - blue_snapshot["reach_inches"]
            base_features[col] = reach_diff if weight_class == division else 0.0
        elif col.startswith("self_reach_inches_x_") and col[len("self_reach_inches_x_"):] in _DIVISION_SUFFIXES:
            base_features[col] = reach_division_value(red_snapshot, col[len("self_reach_inches_x_"):])
        elif col.startswith("opp_reach_inches_x_") and col[len("opp_reach_inches_x_"):] in _DIVISION_SUFFIXES:
            base_features[col] = reach_division_value(blue_snapshot, col[len("opp_reach_inches_x_"):])
        elif col.startswith("self_") and col[len("self_"):] in PAIRED_BASES:
            base = col[len("self_"):]
            base_features[col] = red_snapshot[_PAIRED_KEY_OVERRIDES.get(base, base)]
        elif col.startswith("opp_") and col[len("opp_"):] in PAIRED_BASES:
            base = col[len("opp_"):]
            base_features[col] = blue_snapshot[_PAIRED_KEY_OVERRIDES.get(base, base)]
        elif col.endswith("_diff"):
            base = _DIFF_KEY_OVERRIDES.get(col, col[: -len("_diff")])
            base_features[col] = red_snapshot[base] - blue_snapshot[base]
        else:
            raise ValueError(f"unrecognized feature column: {col}")

    is_missing = {col: bool(pd.isna(val)) for col, val in base_features.items()}

    def fill_value(col):
        if imputer is not None and col in imputer["fill_values"]:
            return imputer["fill_values"][col]
        return 0.0

    filled = {col: (fill_value(col) if is_missing[col] else val) for col, val in base_features.items()}
    for col in indicator_requests:
        filled[col] = int(is_missing.get(col[: -len("_was_missing")], False))

    return filled


def _safe_div(numerator, denominator):
    return numerator / denominator if denominator else float("nan")


def build_fighter_opponent_stats(fights, fighter_id, as_of_date):
    """Stats that need the OPPONENT's side of history, not the fighter's
    own cumulative totals: strikes/takedowns absorbed (for SApM, striking
    defense, takedown defense) and average fight time. Display-only - none
    of this feeds any trained model's FEATURE_COLUMNS - so it's computed as
    one direct filter+aggregate over real history rather than through
    engineer_fold_features' per-row incremental machinery: a single
    point-in-time total needs no row ordering, just a sum."""
    is_red = fights["r_fighter_id"] == fighter_id
    is_blue = fights["b_fighter_id"] == fighter_id
    mask = (is_red | is_blue) & (fights["event_date"] <= as_of_date)
    if not mask.any():
        raise ValueError(f"no data for fighter {fighter_id} by {as_of_date.date()}")

    history = fights.loc[mask]
    is_red = is_red.loc[mask]

    # Same "untrustworthy without round.csv detail" rule _prior_career_totals
    # uses (docs/results.md): 330/11,441 fights have rounds_fought == 0 and
    # their r_total_*/b_total_* columns are a missingness marker, not a
    # genuine zero-output fight - excluded from the stat, the opponent's
    # stat, and the duration side of every ratio below.
    has_detail = history["rounds_fought"] > 0

    def opponent_total(stat_name):
        r_col, b_col = FIGHT_STAT_COLUMNS[stat_name]
        vals = history[b_col].where(is_red, history[r_col])
        return vals.where(has_detail, 0).sum()

    duration_min = (((history["finish_round"] - 1) * 300 + history["finish_time"]) / 60).where(has_detail, 0)
    total_duration = duration_min.sum()
    n_detailed_fights = int(has_detail.sum())

    sig_landed_against = opponent_total("sig_str_landed")
    sig_atmp_against = opponent_total("sig_str_atmp")
    td_success_against = opponent_total("td_success")
    td_atmp_against = opponent_total("td_atmp")

    return {
        "sapm": _safe_div(sig_landed_against, total_duration),
        "striking_defense": 1 - _safe_div(sig_landed_against, sig_atmp_against),
        "takedown_defense": 1 - _safe_div(td_success_against, td_atmp_against),
        "avg_fight_time_min": _safe_div(total_duration, n_detailed_fights),
    }


def strength_of_schedule(fights_with_features, fighter_id, as_of_date):
    """Average quality of opponents faced: for each prior fight, the
    opponent's own prior_win_rate AS OF that fight (not their eventual
    career record, which would leak the opponent's future into this
    fighter's schedule-strength). fights_with_features must already have
    r_prior_win_rate/b_prior_win_rate attached - i.e.
    engineer_fold_features(fights, fights.iloc[0:0])[0], computed once over
    the full dataset (this is exactly what every real fight row's
    prior_win_rate already is from the standard training pipeline, so this
    is a lookup+average, not a new estimate)."""
    is_red = fights_with_features["r_fighter_id"] == fighter_id
    is_blue = fights_with_features["b_fighter_id"] == fighter_id
    mask = (is_red | is_blue) & (fights_with_features["event_date"] <= as_of_date)
    if not mask.any():
        return float("nan")

    history = fights_with_features.loc[mask]
    is_red = is_red.loc[mask]
    opponent_prior_win_rate = history["b_prior_win_rate"].where(is_red, history["r_prior_win_rate"])
    return opponent_prior_win_rate.mean()


def _method_rate_buckets(snapshot, prefix):
    """Collapse the 11 raw method_label categories into the 3 buckets a
    prop-betting audience actually asks about (KO/TKO, Submission,
    Decision) plus a catch-all "other" (DQ, doctor stoppage, overturned,
    could-not-continue, the 2-fight "other" category) - prefix is
    "prior_" (rate among ALL prior fights) or "prior_win_" (rate among
    prior WINS only, i.e. "when I win, how do I usually win")."""
    decision = sum(snapshot.get(f"{prefix}{m}_rate", 0.0) for m in DECISION_LABELS)
    other = sum(snapshot.get(f"{prefix}{m}_rate", 0.0) for m in METHOD_LABELS
                if m not in DECISION_LABELS and m not in ("ko_tko", "submission"))
    return {
        "ko_tko": snapshot.get(f"{prefix}ko_tko_rate", 0.0),
        "submission": snapshot.get(f"{prefix}submission_rate", 0.0),
        "decision": decision,
        "other": other,
    }


def build_display_stats(fights, fights_with_features, fighters_df, fighter_id, as_of_date, weight_class):
    """The full fighter stats panel: Striking, Grappling/Wrestling,
    Physical/Structural, and Fight Pace & Outcome metrics, all as of
    as_of_date. Combines build_fighter_snapshot (self-side career stats,
    already correctly point-in-time) with build_fighter_opponent_stats and
    strength_of_schedule (the opponent-side stats build_fighter_snapshot
    doesn't compute)."""
    snapshot = build_fighter_snapshot(fights, fighters_df, fighter_id, as_of_date, weight_class)
    opponent_stats = build_fighter_opponent_stats(fights, fighter_id, as_of_date)
    schedule = strength_of_schedule(fights_with_features, fighter_id, as_of_date)

    return {
        "striking": {
            "slpm": snapshot["sig_str_landed_per_min"],
            "striking_accuracy": snapshot["career_sig_str_acc"],
            "sapm": opponent_stats["sapm"],
            "striking_defense": opponent_stats["striking_defense"],
        },
        "grappling": {
            "takedown_avg_per_15min": snapshot["td_success_per_min"] * 15,
            "takedown_accuracy": snapshot["career_td_acc"],
            "takedown_defense": opponent_stats["takedown_defense"],
            "sub_att_per_15min": snapshot["sub_att_per_min"] * 15,
        },
        "physical": {
            "age": snapshot["age"],
            "stance": snapshot["stance"],
            "height_inches": snapshot["height"],
            "reach_inches": snapshot["reach_inches"],
            "strength_of_schedule": schedule,
        },
        "fight_pace": {
            "avg_fight_time_min": opponent_stats["avg_fight_time_min"],
            "method_of_victory": {
                "occurrence": _method_rate_buckets(snapshot, "prior_"),
                "wins_by_method": _method_rate_buckets(snapshot, "prior_win_"),
            },
        },
        "record": {
            "prior_fights": snapshot["prior_fights"],
            "prior_wins": snapshot["prior_wins"],
            "prior_win_rate": snapshot["prior_win_rate"],
            "win_streak": snapshot["win_streak"],
            "loss_streak": snapshot["loss_streak"],
            "form_last5_win_rate": snapshot["form_last5_win_rate"],
        },
    }


def find_past_meetings(fights, fighter_a, fighter_b):
    """Every real fight already on record between these two fighters,
    oldest first - handles rematches/trilogies naturally by returning every
    matching row, not just the first or most recent."""
    a_vs_b = (fights["r_fighter_id"] == fighter_a) & (fights["b_fighter_id"] == fighter_b)
    b_vs_a = (fights["r_fighter_id"] == fighter_b) & (fights["b_fighter_id"] == fighter_a)
    return fights.loc[a_vs_b | b_vs_a].sort_values("event_date")


def actual_fight_stats(fight_row, fighter_id):
    """This fighter's own raw, single-fight numbers (not a career rate)
    for one real fight row - the "what actually happened" half of a
    predicted-vs-actual comparison."""
    is_red = fight_row["r_fighter_id"] == fighter_id
    prefix = "r_" if is_red else "b_"
    opp_prefix = "b_" if is_red else "r_"

    return {
        "sig_str_landed": fight_row[f"{prefix}total_sig_landed"],
        "sig_str_atmp": fight_row[f"{prefix}total_sig_atmp"],
        "td_success": fight_row[f"{prefix}total_td_success"],
        "td_atmp": fight_row[f"{prefix}total_td_atmp"],
        "sub_att": fight_row[f"{prefix}total_sub_att"],
        "sig_str_landed_absorbed": fight_row[f"{opp_prefix}total_sig_landed"],
        "sig_str_atmp_absorbed": fight_row[f"{opp_prefix}total_sig_atmp"],
        "won": bool(fight_row["winner_id"] == fighter_id),
        "method": fight_row["method"],
        "finish_round": fight_row["finish_round"],
        "finish_time_seconds": fight_row["finish_time"],
    }
