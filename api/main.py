from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data.cleaning import clean_fights
from src.data.ingestion import build_master_equivalent, load_fighters, load_fights
from src.features.engineering import MAIN_DIVISIONS, engineer_fold_features
from src.features.scaling import apply_scaler
from src.features.snapshot import (
    actual_fight_stats,
    build_display_stats,
    build_fighter_snapshot,
    build_matchup_features,
    find_past_meetings,
    latest_fight_date,
)
from src.models.persistence import load_model_artifact

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost_model"]
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost_model": "XGBoost",
}

_state = {}


@asynccontextmanager
async def lifespan(app):
    load_state()
    yield


app = FastAPI(title="UFC Elo Matchup Predictor", lifespan=lifespan)


def _native(value):
    """Recursively convert numpy/pandas scalar types to plain Python types
    so FastAPI's default JSON encoder can serialize them (numpy int64/bool_
    aren't JSON-serializable out of the box)."""
    if isinstance(value, dict):
        return {k: _native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if pd.isna(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return value


def _meeting_winner(meeting, red_fighter_id):
    """"red"/"blue"/"draw"/"no_contest" for one past-meeting row. winner_id
    is null for BOTH a draw and a no-contest, so this reads result_status
    first rather than inferring the outcome from winner_id alone (which
    would silently mislabel every draw/no-contest as a blue win - meeting["
    winner_id"] == red_fighter_id is False for both a blue win and a
    draw/NC)."""
    if meeting["result_status"] == "draw":
        return "draw"
    if meeting["result_status"] == "no_contest":
        return "no_contest"
    return "red" if meeting["winner_id"] == red_fighter_id else "blue"


def _build_fighter_index(fights, fighters_df):
    long = pd.concat([
        fights[["r_fighter_id", "weight_class", "event_date"]].rename(columns={"r_fighter_id": "fighter_id"}),
        fights[["b_fighter_id", "weight_class", "event_date"]].rename(columns={"b_fighter_id": "fighter_id"}),
    ], ignore_index=True)

    name_lookup = fighters_df.set_index("fighter_id")
    index = {}
    for fighter_id, group in long.groupby("fighter_id"):
        if fighter_id in name_lookup.index:
            row = name_lookup.loc[fighter_id]
            name = row["fighter_name"]
            dob = row["dob"]
            display_name = f"{name} (b. {dob})" if pd.notna(dob) else name
            nick_name = row["fighter_nick_name"] if pd.notna(row["fighter_nick_name"]) else None
        else:
            name = fighter_id
            display_name = fighter_id
            nick_name = None

        index[fighter_id] = {
            "fighter_id": fighter_id,
            "fighter_name": name,
            "nick_name": nick_name,
            "display_name": display_name,
            "divisions": set(group["weight_class"].dropna().unique()),
            "years": sorted(int(y) for y in group["event_date"].dt.year.unique()),
        }

    # Two different real fighters can share both a name AND have no DOB on
    # record to tell them apart (verified against real fighter.csv, e.g.
    # two distinct "Anthony Figueroa" fighter_ids) - the "(b. ...)"
    # disambiguator above does nothing for them, so without this they'd
    # show as two visually-identical options in the Fighter dropdown with
    # no way for a user to tell which is which before picking (a fighter
    # IDENTITY integrity gap, not a missing-data one - both records are
    # correct and distinct, just indistinguishable in the UI). Append a
    # short, stable fighter_id suffix to every display_name that still
    # collides after the dob check, for every such group, not just the
    # no-dob case - any future collision (e.g. two same-named fighters who
    # happen to share a dob too) gets the same guarantee of uniqueness.
    by_display_name = {}
    for info in index.values():
        by_display_name.setdefault(info["display_name"], []).append(info)
    for group in by_display_name.values():
        if len(group) > 1:
            for info in group:
                info["display_name"] = f"{info['display_name']} [id:{info['fighter_id'][:8]}]"

    return index


def load_state():
    fighters_df = load_fighters(DATA_DIR)
    master = build_master_equivalent(load_fights(DATA_DIR), fighters_df)
    fights = clean_fights(master)
    # computed once over the full dataset so strength_of_schedule can look
    # up each real fight's own point-in-time r_prior_win_rate/b_prior_win_rate
    # without recomputing engineer_fold_features per request.
    fights_with_features, _ = engineer_fold_features(fights, fights.iloc[0:0])

    _state["fights"] = fights
    _state["fighters_df"] = fighters_df
    _state["fights_with_features"] = fights_with_features
    # MAIN_DIVISIONS (the 8 standard men's divisions), not every raw
    # weight_class string in the data - that includes 1990s tournament
    # bracket labels ("13 Heavyweight Tournament") and women's divisions,
    # neither of which the model's reach_diff_x_division feature was
    # trained to recognize (it zeroes out for any division outside
    # MAIN_DIVISIONS - a known scope limit, not a bug, see engineering.py).
    _state["divisions"] = MAIN_DIVISIONS
    _state["fighter_index"] = _build_fighter_index(fights, fighters_df)
    _state["models"] = {name: load_model_artifact(name) for name in MODEL_NAMES}

    # Round/win-method targets (docs/tasks.md "New prediction targets"
    # Recommendation) - one pair of artifacts per MODEL_NAMES entry, saved
    # by src.evaluation.multiclass_backtest as "<model_name>_round"/
    # "<model_name>_method". Loaded only if present: an older models/
    # directory without these yet must not crash startup - _predict_multiclass
    # falls back to None (the frontend already renders that as "Unavailable").
    _state["multiclass_models"] = {}
    for name in MODEL_NAMES:
        for target in ("round", "method"):
            artifact_name = f"{name}_{target}"
            try:
                _state["multiclass_models"][artifact_name] = load_model_artifact(artifact_name)
            except FileNotFoundError:
                pass


class PredictRequest(BaseModel):
    model_name: str
    division: str
    red_fighter_id: str
    red_year: Optional[int] = None
    blue_fighter_id: str
    blue_year: Optional[int] = None


def _predict(model_name, division, red_fighter_id, red_cutoff, blue_fighter_id, blue_cutoff,
             fights, fighters_df):
    model, scaler, imputer, meta = _state["models"][model_name]
    feature_columns = meta["feature_columns"]
    # pre-imputation artifacts have no "fitted_feature_columns" - the model
    # was fit on feature_columns directly, with no missingness indicators.
    fitted_feature_columns = meta.get("fitted_feature_columns", feature_columns)

    red_snapshot = build_fighter_snapshot(fights, fighters_df, red_fighter_id, red_cutoff, division)
    blue_snapshot = build_fighter_snapshot(fights, fighters_df, blue_fighter_id, blue_cutoff, division)
    feats = build_matchup_features(red_snapshot, blue_snapshot, division, fitted_feature_columns, imputer=imputer)

    X = pd.DataFrame([feats])[fitted_feature_columns]
    if scaler is not None:
        X = apply_scaler(X, fitted_feature_columns, scaler)

    red_win_probability = float(model.predict_proba(X)[0, 1])
    return red_win_probability


# method_bucket's internal keys (src.features.engineering.METHOD_BUCKET_MAP)
# to a display label - never invented, just the same coarse buckets
# build_display_stats' method_of_victory already shows.
_METHOD_DISPLAY = {"ko_tko": "KO/TKO", "submission": "Submission", "decision": "Decision", "other": "Other"}


def _predict_multiclass(artifact_key, division, red_fighter_id, red_cutoff, blue_fighter_id, blue_cutoff,
                         fights, fighters_df, static_features=None):
    """The single most-likely class for a round or win-method artifact
    (meta["classes"], decoded from the model's integer prediction - see
    src.evaluation.multiclass_backtest), or None if artifact_key has no
    saved artifact at all (an older models/ directory, or - per the hard
    red/blue-only constraint - this was never trained for anything but a
    predictable outcome, which is the only kind of fight this is ever
    called for anyway). Mirrors _predict's snapshot-based feature
    construction exactly; never substitutes a guess for a missing
    artifact - the caller gets None back and must pass that through
    as-is, not silently invent a round/method."""
    artifact = _state["multiclass_models"].get(artifact_key)
    if artifact is None:
        return None
    model, scaler, imputer, meta = artifact
    feature_columns = meta["feature_columns"]
    fitted_feature_columns = meta.get("fitted_feature_columns", feature_columns)
    classes = meta["classes"]

    red_snapshot = build_fighter_snapshot(fights, fighters_df, red_fighter_id, red_cutoff, division)
    blue_snapshot = build_fighter_snapshot(fights, fighters_df, blue_fighter_id, blue_cutoff, division)
    feats = build_matchup_features(red_snapshot, blue_snapshot, division, fitted_feature_columns,
                                    imputer=imputer, static_features=static_features)

    X = pd.DataFrame([feats])[fitted_feature_columns]
    if scaler is not None:
        X = apply_scaler(X, fitted_feature_columns, scaler)

    predicted_idx = int(model.predict(X)[0])
    return classes[predicted_idx]


@app.get("/divisions")
def get_divisions():
    return _state["divisions"]


@app.get("/models")
def get_models():
    return [{"id": name, "label": MODEL_LABELS[name]} for name in MODEL_NAMES]


@app.get("/fighters")
def get_fighters(division: str):
    fighter_index = _state["fighter_index"]
    results = [
        {"fighter_id": info["fighter_id"], "fighter_name": info["fighter_name"],
         "nick_name": info["nick_name"], "display_name": info["display_name"], "years": info["years"]}
        for info in fighter_index.values()
        if division in info["divisions"]
    ]
    results.sort(key=lambda r: r["display_name"])
    return results


@app.post("/predict")
def predict(req: PredictRequest):
    if req.model_name not in MODEL_NAMES:
        raise HTTPException(400, f"unknown model_name {req.model_name}")

    fights = _state["fights"]
    fighters_df = _state["fighters_df"]
    fights_with_features = _state["fights_with_features"]

    try:
        red_cutoff = latest_fight_date(fights, req.red_fighter_id, req.red_year)
        blue_cutoff = latest_fight_date(fights, req.blue_fighter_id, req.blue_year)

        red_win_probability = _predict(req.model_name, req.division, req.red_fighter_id, red_cutoff,
                                        req.blue_fighter_id, blue_cutoff, fights, fighters_df)

        red_display = build_display_stats(fights, fights_with_features, fighters_df,
                                           req.red_fighter_id, red_cutoff, req.division)
        blue_display = build_display_stats(fights, fights_with_features, fighters_df,
                                            req.blue_fighter_id, blue_cutoff, req.division)
    except ValueError as e:
        raise HTTPException(400, str(e))

    # Rematches/trilogies: for every real fight already on record between
    # these two, show what the chosen model would have predicted right
    # before that specific fight (its own row excluded from history, cutoff
    # = its own event_date) against what actually happened - including a
    # draw or no-contest, which clean_fights no longer drops (see
    # src/data/cleaning.py).
    history = []
    past_meetings = find_past_meetings(fights, req.red_fighter_id, req.blue_fighter_id)
    for _, meeting in past_meetings.iterrows():
        trimmed_fights = fights[fights["fight_id"] != meeting["fight_id"]]
        trimmed_features = fights_with_features[fights_with_features["fight_id"] != meeting["fight_id"]]
        meeting_division = meeting["weight_class"]
        meeting_date = meeting["event_date"]

        try:
            historical_red_prob = _predict(req.model_name, meeting_division, req.red_fighter_id, meeting_date,
                                            req.blue_fighter_id, meeting_date, trimmed_fights, fighters_df)

            # rounds_scheduled is a fight-level fact (3 vs. 5 scheduled
            # rounds), not derivable from either fighter's own snapshot -
            # read directly off this REAL historical fight's own row
            # (fights_with_features, not trimmed_features, since trimming
            # excludes the meeting itself). See
            # src.features.engineering.add_fight_outcome_targets.
            meeting_rounds_scheduled = fights_with_features.loc[
                fights_with_features["fight_id"] == meeting["fight_id"], "rounds_scheduled"]
            meeting_rounds_scheduled = (
                meeting_rounds_scheduled.iloc[0] if not meeting_rounds_scheduled.empty else None)

            predicted_round = _predict_multiclass(
                f"{req.model_name}_round", meeting_division, req.red_fighter_id, meeting_date,
                req.blue_fighter_id, meeting_date, trimmed_fights, fighters_df,
                static_features={"rounds_scheduled": meeting_rounds_scheduled})
            predicted_method = _predict_multiclass(
                f"{req.model_name}_method", meeting_division, req.red_fighter_id, meeting_date,
                req.blue_fighter_id, meeting_date, trimmed_fights, fighters_df)

            red_pre_fight = build_display_stats(trimmed_fights, trimmed_features, fighters_df,
                                                 req.red_fighter_id, meeting_date, meeting_division)
            blue_pre_fight = build_display_stats(trimmed_fights, trimmed_features, fighters_df,
                                                  req.blue_fighter_id, meeting_date, meeting_division)
            predicted = {
                "red_win_probability": historical_red_prob,
                "blue_win_probability": 1 - historical_red_prob,
                "predicted_winner": "red" if historical_red_prob >= 0.5 else "blue",
                # None only if this model_name has no round/method artifact
                # at all (see _predict_multiclass) - never a guessed value.
                "predicted_round": predicted_round,
                "predicted_method": _METHOD_DISPLAY.get(predicted_method, predicted_method),
                "red_stats": red_pre_fight,
                "blue_stats": blue_pre_fight,
            }
        except ValueError:
            # one side had no fight history before this meeting (e.g. it
            # was their UFC debut) - no pre-fight snapshot to predict from,
            # but the meeting itself is still real: show the actual result
            # without a predicted-vs-actual comparison rather than hiding
            # the meeting entirely.
            predicted = None

        history.append({
            "fight_id": meeting["fight_id"],
            "event_name": meeting["event_name"],
            "event_date": meeting_date,
            "predicted": predicted,
            "actual": {
                # result_status ("win"/"draw"/"no_contest") - not winner_id
                # alone - decides this: a draw and a no-contest both have a
                # null winner_id, so "winner_id == red_fighter_id" being
                # False does NOT mean blue won (see src/data/cleaning.py).
                "winner": _meeting_winner(meeting, req.red_fighter_id),
                "result_status": meeting["result_status"],
                "method": meeting["method"],
                "details": meeting["details"] or None,
                "finish_round": meeting["finish_round"],
                "finish_time_seconds": meeting["finish_time"],
                "red": actual_fight_stats(meeting, req.red_fighter_id),
                "blue": actual_fight_stats(meeting, req.blue_fighter_id),
            },
        })

    response = {
        "model_used": req.model_name,
        "red_fighter_id": req.red_fighter_id,
        "red_snapshot_date": red_cutoff,
        "blue_fighter_id": req.blue_fighter_id,
        "blue_snapshot_date": blue_cutoff,
        "red_win_probability": red_win_probability,
        "blue_win_probability": 1 - red_win_probability,
        "predicted_winner": "red" if red_win_probability >= 0.5 else "blue",
        "red_stats": red_display,
        "blue_stats": blue_display,
        "past_meetings": history,
    }
    return _native(response)
