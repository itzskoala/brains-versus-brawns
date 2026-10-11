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
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


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
        else:
            display_name = fighter_id

        index[fighter_id] = {
            "fighter_id": fighter_id,
            "display_name": display_name,
            "divisions": set(group["weight_class"].dropna().unique()),
            "years": sorted(int(y) for y in group["event_date"].dt.year.unique()),
        }
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
        {"fighter_id": info["fighter_id"], "display_name": info["display_name"], "years": info["years"]}
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
    # = its own event_date) against what actually happened. Note: a meeting
    # that was later overturned to a no-contest won't appear here -
    # clean_fights already drops no-decision rows (no label to learn from).
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
            red_pre_fight = build_display_stats(trimmed_fights, trimmed_features, fighters_df,
                                                 req.red_fighter_id, meeting_date, meeting_division)
            blue_pre_fight = build_display_stats(trimmed_fights, trimmed_features, fighters_df,
                                                  req.blue_fighter_id, meeting_date, meeting_division)
        except ValueError:
            # a fighter's very first-ever fight (no history to snapshot
            # before it) - skip rather than error the whole request.
            continue

        history.append({
            "fight_id": meeting["fight_id"],
            "event_name": meeting["event_name"],
            "event_date": meeting_date,
            "predicted": {
                "red_win_probability": historical_red_prob,
                "blue_win_probability": 1 - historical_red_prob,
                "predicted_winner": "red" if historical_red_prob >= 0.5 else "blue",
                "red_stats": red_pre_fight,
                "blue_stats": blue_pre_fight,
            },
            "actual": {
                "winner": "red" if meeting["winner_id"] == req.red_fighter_id else "blue",
                "method": meeting["method"],
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
