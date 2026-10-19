import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import _build_fighter_index, app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _make_fights(rows):
    # rows: list of (fight_id, r_id, b_id, weight_class, year)
    return pd.DataFrame([
        {"fight_id": fid, "r_fighter_id": r, "b_fighter_id": b, "weight_class": wc,
         "event_date": pd.Timestamp(f"{year}-01-01")}
        for fid, r, b, wc, year in rows
    ])


def _make_fighters(rows):
    # rows: list of (fighter_id, fighter_name, fighter_nick_name, dob)
    return pd.DataFrame(rows, columns=["fighter_id", "fighter_name", "fighter_nick_name", "dob"])


# ---------------------------------------------------------------------------
# Fighter identity integrity: _build_fighter_index must never let two
# distinct real fighter_ids collapse into one indistinguishable entry, and
# must never mix one fighter's divisions/years into another's.
# ---------------------------------------------------------------------------

def test_build_fighter_index_keeps_distinct_ids_with_same_name_separate():
    # Real shape, e.g. UFCStats' two distinct "Anthony Figueroa" fighter_ids
    # (verified against data/fighter.csv) - same name, no DOB on record for
    # either, so the usual "(b. ...)" disambiguator does nothing.
    fights = _make_fights([
        ("f1", "id-a", "opp1", "Bantamweight", 2015),
        ("f2", "id-b", "opp2", "Featherweight", 2020),
    ])
    fighters = _make_fighters([
        ("id-a", "Same Name", None, None),
        ("id-b", "Same Name", None, None),
        ("opp1", "Opponent One", None, None),
        ("opp2", "Opponent Two", None, None),
    ])

    index = _build_fighter_index(fights, fighters)

    assert index["id-a"]["display_name"] != index["id-b"]["display_name"]
    # each still legible as "Same Name", just not identical to one another.
    assert index["id-a"]["display_name"].startswith("Same Name")
    assert index["id-b"]["display_name"].startswith("Same Name")
    # the disambiguator must not bleed into fighter_name/nick_name, which
    # other code (predict()'s fight-title formatting) reads separately.
    assert index["id-a"]["fighter_name"] == "Same Name"
    assert index["id-b"]["fighter_name"] == "Same Name"


def test_build_fighter_index_does_not_mix_divisions_or_years_across_same_named_fighters():
    fights = _make_fights([
        ("f1", "id-a", "opp1", "Bantamweight", 2015),
        ("f2", "id-b", "opp2", "Featherweight", 2020),
    ])
    fighters = _make_fighters([
        ("id-a", "Same Name", None, None),
        ("id-b", "Same Name", None, None),
        ("opp1", "Opponent One", None, None),
        ("opp2", "Opponent Two", None, None),
    ])

    index = _build_fighter_index(fights, fighters)

    assert index["id-a"]["divisions"] == {"Bantamweight"}
    assert index["id-b"]["divisions"] == {"Featherweight"}
    assert index["id-a"]["years"] == [2015]
    assert index["id-b"]["years"] == [2020]


def test_build_fighter_index_dob_alone_already_disambiguates_no_extra_suffix_needed():
    # One has a recorded DOB, the other doesn't - their display_names
    # already differ ("Name (b. ...)" vs "Name"), so no [id:...] suffix
    # should be added to either (it would only add noise).
    fights = _make_fights([
        ("f1", "id-a", "opp1", "Bantamweight", 2015),
        ("f2", "id-b", "opp2", "Featherweight", 2020),
    ])
    fighters = _make_fighters([
        ("id-a", "Same Name", None, "1990-01-01"),
        ("id-b", "Same Name", None, None),
        ("opp1", "Opponent One", None, None),
        ("opp2", "Opponent Two", None, None),
    ])

    index = _build_fighter_index(fights, fighters)

    assert index["id-a"]["display_name"] == "Same Name (b. 1990-01-01)"
    assert index["id-b"]["display_name"] == "Same Name"
    assert "[id:" not in index["id-a"]["display_name"]
    assert "[id:" not in index["id-b"]["display_name"]


def test_build_fighter_index_unique_names_are_never_suffixed():
    fights = _make_fights([("f1", "id-a", "id-b", "Bantamweight", 2015)])
    fighters = _make_fighters([
        ("id-a", "Unique Fighter One", None, None),
        ("id-b", "Unique Fighter Two", None, None),
    ])

    index = _build_fighter_index(fights, fighters)

    assert index["id-a"]["display_name"] == "Unique Fighter One"
    assert index["id-b"]["display_name"] == "Unique Fighter Two"


def test_divisions_lists_the_eight_main_divisions(client):
    resp = client.get("/divisions")
    assert resp.status_code == 200
    assert "Heavyweight" in resp.json()
    assert "Light Heavyweight" in resp.json()


def test_models_lists_all_three(client):
    resp = client.get("/models")
    ids = {m["id"] for m in resp.json()}
    assert ids == {"logistic_regression", "random_forest", "xgboost_model"}


def test_fighters_scoped_to_division_contains_known_fighter(client):
    resp = client.get("/fighters", params={"division": "Light Heavyweight"})
    assert resp.status_code == 200
    names = [f["display_name"] for f in resp.json()]
    assert any("Jon Jones" in n for n in names)


def test_fighters_include_raw_name_and_nickname_for_frontend_search_and_titles(client):
    fighters = client.get("/fighters", params={"division": "Bantamweight"}).json()
    omalley = next(f for f in fighters if f["fighter_name"] == "Sean O'Malley")
    assert omalley["nick_name"] == "Suga"
    # display_name is unchanged (still DOB-disambiguated, no nickname) -
    # the new fields are additive, not a replacement.
    assert "Suga" not in omalley["display_name"]


def test_fighters_nickname_is_none_not_nan_when_absent(client):
    fighters = client.get("/fighters", params={"division": "Light Heavyweight"}).json()
    jones = next(f for f in fighters if f["fighter_name"] == "Jon Jones")
    assert jones["nick_name"] is None or isinstance(jones["nick_name"], str)
    no_nickname = [f for f in fighters if f["nick_name"] is None]
    assert len(no_nickname) > 0


def _fighter_id(client, division, name_fragment):
    fighters = client.get("/fighters", params={"division": division}).json()
    return next(f["fighter_id"] for f in fighters if name_fragment in f["display_name"])


def test_predict_happy_path_with_per_fighter_years(client):
    jones = _fighter_id(client, "Light Heavyweight", "Jon Jones")
    cormier = _fighter_id(client, "Light Heavyweight", "Daniel Cormier")

    resp = client.post("/predict", json={
        "model_name": "xgboost_model",
        "division": "Light Heavyweight",
        "red_fighter_id": jones,
        "red_year": 2015,
        "blue_fighter_id": cormier,
        "blue_year": 2020,
    })
    assert resp.status_code == 200
    data = resp.json()

    assert data["red_snapshot_date"] == "2015-01-03"
    assert data["blue_snapshot_date"] == "2020-08-15"
    assert 0.0 <= data["red_win_probability"] <= 1.0
    assert data["red_win_probability"] + data["blue_win_probability"] == pytest.approx(1.0)
    assert data["predicted_winner"] in ("red", "blue")
    assert "striking" in data["red_stats"]
    assert "grappling" in data["blue_stats"]

    # Round/win-method targets (docs/tasks.md Recommendation) must be
    # predicted for this upcoming/hypothetical matchup too, not just for
    # past meetings - see test_past_meeting_predicted_round_and_method_
    # appear_alongside_winner_probability below.
    assert isinstance(data["predicted_round"], int)
    assert 1 <= data["predicted_round"] <= 5
    assert data["predicted_method"] in ("KO/TKO", "Submission", "Decision", "Other")

    # a real past meeting (UFC 182) should show up with its own
    # predicted-vs-actual comparison.
    assert len(data["past_meetings"]) >= 1
    meeting = data["past_meetings"][0]
    assert meeting["actual"]["winner"] in ("red", "blue")
    assert 0.0 <= meeting["predicted"]["red_win_probability"] <= 1.0


# ---------------------------------------------------------------------------
# Regression: UFC 214 (Cormier vs. Jones 2) was overturned to a no-contest
# after the fact - a real historical case of exactly the outcome the hard
# red/blue-only prediction constraint must never be trained/scored against,
# while the UI must still retain and show what the model WOULD have
# predicted beforehand, separately from the actual (non-)result.
# ---------------------------------------------------------------------------

def test_past_meeting_retains_predicted_winner_for_a_real_no_contest(client):
    jones = _fighter_id(client, "Light Heavyweight", "Jon Jones")
    cormier = _fighter_id(client, "Light Heavyweight", "Daniel Cormier")

    resp = client.post("/predict", json={
        "model_name": "xgboost_model",
        "division": "Light Heavyweight",
        "red_fighter_id": jones,
        "red_year": None,
        "blue_fighter_id": cormier,
        "blue_year": None,
    })
    assert resp.status_code == 200
    meetings = {m["event_name"]: m for m in resp.json()["past_meetings"]}
    assert "UFC 214: Cormier vs. Jones 2" in meetings
    ufc_214 = meetings["UFC 214: Cormier vs. Jones 2"]

    # The actual result is correctly a no-contest, not a fabricated winner.
    assert ufc_214["actual"]["winner"] == "no_contest"
    assert ufc_214["actual"]["result_status"] == "no_contest"
    assert ufc_214["actual"]["method"] == "Overturned"
    assert ufc_214["actual"]["details"]  # the overturn reason, not dropped

    # Fight stats recorded before the stoppage are still shown.
    assert ufc_214["actual"]["red"]["sig_str_landed"] > 0 or ufc_214["actual"]["blue"]["sig_str_landed"] > 0

    # The model's pre-fight Red/Blue prediction is retained and displayed
    # separately - never suppressed just because the real outcome wasn't
    # something the model can predict.
    predicted = ufc_214["predicted"]
    assert predicted is not None
    assert predicted["predicted_winner"] in ("red", "blue")
    assert 0.0 <= predicted["red_win_probability"] <= 1.0
    assert predicted["red_win_probability"] + predicted["blue_win_probability"] == pytest.approx(1.0)

    # Round/win-method targets are now trained (docs/tasks.md
    # Recommendation, accepted) - the model's pre-fight round/method call
    # is retained exactly like the winner call, even though the real
    # outcome here is a no-contest with no official method/round to
    # compare it against.
    assert "predicted_round" in predicted
    assert "predicted_method" in predicted
    assert isinstance(predicted["predicted_round"], int)
    assert predicted["predicted_method"] in ("KO/TKO", "Submission", "Decision", "Other")


@pytest.mark.parametrize("model_name", ["logistic_regression", "random_forest", "xgboost_model"])
def test_past_meeting_predicted_round_and_method_appear_alongside_winner_probability(client, model_name):
    # Trace through the full pipeline for an ordinary decisive past meeting
    # (UFC 182, a normal win, not a non-standard outcome): every one of the
    # three model families must return a real round/method prediction
    # alongside the existing winner probability, for every corner.
    jones = _fighter_id(client, "Light Heavyweight", "Jon Jones")
    cormier = _fighter_id(client, "Light Heavyweight", "Daniel Cormier")

    resp = client.post("/predict", json={
        "model_name": model_name,
        "division": "Light Heavyweight",
        "red_fighter_id": jones,
        "red_year": None,
        "blue_fighter_id": cormier,
        "blue_year": None,
    })
    assert resp.status_code == 200
    meetings = {m["event_name"]: m for m in resp.json()["past_meetings"]}
    ufc_182 = meetings["UFC 182: Jones vs Cormier"]
    predicted = ufc_182["predicted"]

    assert predicted is not None
    assert predicted["predicted_winner"] in ("red", "blue")
    assert 0.0 <= predicted["red_win_probability"] <= 1.0
    assert isinstance(predicted["predicted_round"], int)
    assert 1 <= predicted["predicted_round"] <= 5
    assert predicted["predicted_method"] in ("KO/TKO", "Submission", "Decision", "Other")


def test_predict_record_breaks_out_wins_losses_draws_and_no_contests_separately(client):
    jones = _fighter_id(client, "Light Heavyweight", "Jon Jones")
    cormier = _fighter_id(client, "Light Heavyweight", "Daniel Cormier")

    resp = client.post("/predict", json={
        "model_name": "xgboost_model",
        "division": "Light Heavyweight",
        "red_fighter_id": jones,
        "red_year": None,
        "blue_fighter_id": cormier,
        "blue_year": None,
    })
    assert resp.status_code == 200
    record = resp.json()["red_stats"]["record"]

    for key in ("prior_fights", "prior_wins", "prior_losses", "prior_draws", "prior_no_contests"):
        assert key in record

    # wins + losses + draws + no-contests must reconstruct prior_fights
    # exactly - a draw/no-contest is neither a win nor a loss, so it's not
    # double-counted or dropped.
    assert (record["prior_wins"] + record["prior_losses"]
            + record["prior_draws"] + record["prior_no_contests"]) == record["prior_fights"]


def test_predict_defaults_to_full_history_when_year_omitted(client):
    jones = _fighter_id(client, "Light Heavyweight", "Jon Jones")
    cormier = _fighter_id(client, "Light Heavyweight", "Daniel Cormier")

    resp = client.post("/predict", json={
        "model_name": "logistic_regression",
        "division": "Light Heavyweight",
        "red_fighter_id": jones,
        "blue_fighter_id": cormier,
    })
    assert resp.status_code == 200
    data = resp.json()
    # full history = each fighter's own actual latest fight overall.
    assert data["red_snapshot_date"] != data["blue_snapshot_date"]


def test_predict_rejects_unknown_model(client):
    jones = _fighter_id(client, "Light Heavyweight", "Jon Jones")
    cormier = _fighter_id(client, "Light Heavyweight", "Daniel Cormier")

    resp = client.post("/predict", json={
        "model_name": "not_a_real_model",
        "division": "Light Heavyweight",
        "red_fighter_id": jones,
        "blue_fighter_id": cormier,
    })
    assert resp.status_code == 400


def test_predict_rejects_year_before_fighter_debut(client):
    jones = _fighter_id(client, "Light Heavyweight", "Jon Jones")
    cormier = _fighter_id(client, "Light Heavyweight", "Daniel Cormier")

    resp = client.post("/predict", json={
        "model_name": "xgboost_model",
        "division": "Light Heavyweight",
        "red_fighter_id": jones,
        "red_year": 1990,
        "blue_fighter_id": cormier,
    })
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Fight format (3 vs. 5 scheduled rounds) - the frontend's toggle. This is a
# real model input (rounds_scheduled, see src.features.engineering.
# add_fight_outcome_targets), not just a display label, so these tests
# confirm the selection actually changes what's fed into the round model,
# not merely what the response happens to echo back.
# ---------------------------------------------------------------------------

def test_predict_defaults_rounds_scheduled_to_3_when_omitted(client):
    jones = _fighter_id(client, "Light Heavyweight", "Jon Jones")
    cormier = _fighter_id(client, "Light Heavyweight", "Daniel Cormier")

    resp = client.post("/predict", json={
        "model_name": "xgboost_model", "division": "Light Heavyweight",
        "red_fighter_id": jones, "blue_fighter_id": cormier,
    })
    assert resp.status_code == 200
    assert resp.json()["rounds_scheduled"] == 3


@pytest.mark.parametrize("rounds_scheduled", [3, 5])
def test_predict_rounds_scheduled_round_trips_in_response(client, rounds_scheduled):
    jones = _fighter_id(client, "Light Heavyweight", "Jon Jones")
    cormier = _fighter_id(client, "Light Heavyweight", "Daniel Cormier")

    resp = client.post("/predict", json={
        "model_name": "xgboost_model", "division": "Light Heavyweight",
        "red_fighter_id": jones, "blue_fighter_id": cormier,
        "rounds_scheduled": rounds_scheduled,
    })
    assert resp.status_code == 200
    assert resp.json()["rounds_scheduled"] == rounds_scheduled


@pytest.mark.parametrize("rounds_scheduled", [0, 1, 2, 4, 6, -3])
def test_predict_rejects_rounds_scheduled_outside_3_or_5(client, rounds_scheduled):
    jones = _fighter_id(client, "Light Heavyweight", "Jon Jones")
    cormier = _fighter_id(client, "Light Heavyweight", "Daniel Cormier")

    resp = client.post("/predict", json={
        "model_name": "xgboost_model", "division": "Light Heavyweight",
        "red_fighter_id": jones, "blue_fighter_id": cormier,
        "rounds_scheduled": rounds_scheduled,
    })
    assert resp.status_code == 422


def test_predict_rounds_scheduled_actually_feeds_the_round_model_not_just_the_echoed_label(client, monkeypatch):
    # Spy on _predict_multiclass to capture the static_features it's
    # actually called with for the top-level (non-past-meeting) round
    # prediction - the first "<model>_round" call made per request, since
    # predict() makes it before it ever loops over past_meetings (whose own
    # calls use a REAL historical rounds_scheduled, unrelated to the
    # request's). A fixed/ignored default would show the same captured
    # value regardless of what the request asked for.
    jones = _fighter_id(client, "Light Heavyweight", "Jon Jones")
    cormier = _fighter_id(client, "Light Heavyweight", "Daniel Cormier")
    original = api_main._predict_multiclass

    def _capture_first_round_call(rounds_scheduled):
        captured = []

        def _spy(artifact_key, *args, static_features=None, **kwargs):
            if artifact_key.endswith("_round") and not captured:
                captured.append(static_features["rounds_scheduled"])
            return original(artifact_key, *args, static_features=static_features, **kwargs)

        monkeypatch.setattr(api_main, "_predict_multiclass", _spy)
        resp = client.post("/predict", json={
            "model_name": "xgboost_model", "division": "Light Heavyweight",
            "red_fighter_id": jones, "blue_fighter_id": cormier,
            "rounds_scheduled": rounds_scheduled,
        })
        assert resp.status_code == 200
        return captured[0]

    assert _capture_first_round_call(3) == 3.0
    assert _capture_first_round_call(5) == 5.0


# ---------------------------------------------------------------------------
# Data freshness and no-leakage: api.main.load_state() only runs once, at
# process startup (see the lifespan hook) - _state["fights"] is NOT reloaded
# per request. The snapshot-reuse refactor in predict() (building one
# snapshot per corner/cutoff and sharing it across the win/round/method/
# display-stats targets) must not add any ADDITIONAL caching on top of
# that: every request still has to (a) read whatever is currently in
# _state at call time, and (b) build a genuinely distinct snapshot for
# every distinct (fighter, cutoff) pair it needs - the live matchup's own
# cutoff, and each past meeting's own, real historical cutoff - never
# reusing one for another.
# ---------------------------------------------------------------------------

def test_predict_builds_exactly_one_snapshot_per_fighter_cutoff_with_no_cross_contamination(client, monkeypatch):
    jones = _fighter_id(client, "Light Heavyweight", "Jon Jones")
    cormier = _fighter_id(client, "Light Heavyweight", "Daniel Cormier")
    original = api_main.build_fighter_snapshot
    calls = []

    def _spy(fights, fighters_df, fighter_id, as_of_date, weight_class):
        calls.append((fighter_id, as_of_date))
        return original(fights, fighters_df, fighter_id, as_of_date, weight_class)

    monkeypatch.setattr(api_main, "build_fighter_snapshot", _spy)

    resp = client.post("/predict", json={
        "model_name": "xgboost_model", "division": "Light Heavyweight",
        "red_fighter_id": jones, "blue_fighter_id": cormier,
    })
    assert resp.status_code == 200
    data = resp.json()
    meeting_dates = {pd.Timestamp(m["event_date"]) for m in data["past_meetings"]}
    assert meeting_dates, "this test needs a real rivalry with at least one past meeting to prove anything"

    red_cutoff = pd.Timestamp(data["red_snapshot_date"])
    blue_cutoff = pd.Timestamp(data["blue_snapshot_date"])
    # Jones has fought many times since his UFC 214/182 meetings with
    # Cormier - if this ever stops being true (dataset cut off earlier),
    # the coincident-date edge case below needs separate handling rather
    # than this assertion silently passing on a false premise.
    assert red_cutoff not in meeting_dates

    jones_calls = [c for c in calls if c[0] == jones]
    cormier_calls = [c for c in calls if c[0] == cormier]

    # The live matchup's own cutoff is built exactly once per corner (the
    # win/round/method/display-stats targets all reuse it - see
    # api.main.predict) - not the 4 separate rebuilds from before the
    # snapshot-reuse refactor.
    assert sum(1 for c in jones_calls if c[1] == red_cutoff) == 1
    assert sum(1 for c in cormier_calls if c[1] == blue_cutoff) == 1

    # Every past meeting gets its own real historical cutoff, also built
    # exactly once per corner - never reusing the live matchup's cutoff or
    # another meeting's.
    for meeting_date in meeting_dates:
        assert sum(1 for c in jones_calls if c[1] == meeting_date) == 1
        assert sum(1 for c in cormier_calls if c[1] == meeting_date) == 1

    # No extra/duplicate builds anywhere: exactly one call per corner per
    # distinct cutoff actually needed (live matchup + one per meeting).
    assert len(jones_calls) == 1 + len(meeting_dates)
    assert len(cormier_calls) == 1 + len(meeting_dates)


def test_predict_uses_the_current_state_fights_table_not_a_copy_frozen_at_startup(client, monkeypatch):
    # Regression for the weekly scrape-and-retrain cron
    # (.github/workflows/update-fight-data.yml): once new data replaces
    # _state["fights"] (whether via a future live-reload or, today, a
    # fresh process after that cron's push redeploys the service),
    # /predict must reflect it on the very next call - not a fights table
    # captured once by an earlier call or cached anywhere else.
    jones = _fighter_id(client, "Light Heavyweight", "Jon Jones")
    cormier = _fighter_id(client, "Light Heavyweight", "Daniel Cormier")

    baseline = client.post("/predict", json={
        "model_name": "xgboost_model", "division": "Light Heavyweight",
        "red_fighter_id": jones, "blue_fighter_id": cormier,
    }).json()
    baseline_cutoff = baseline["red_snapshot_date"]
    baseline_prior_fights = baseline["red_stats"]["record"]["prior_fights"]

    original_fights = api_main._state["fights"]
    new_fight_date = original_fights["event_date"].max() + pd.Timedelta(days=30)
    # Template off Jones's own most recent real fight row so every column
    # engineer_fold_features needs is already present and valid - only the
    # fight-identifying fields are overwritten.
    template = original_fights[original_fights["r_fighter_id"] == jones].iloc[-1].to_dict()
    template.update({
        "fight_id": "__regression_new_fight__",
        "event_date": new_fight_date,
        "winner_id": jones,
        "result_status": "win",
        "weight_class": "Light Heavyweight",
        "b_fighter_id": "__regression_new_opponent__",
    })
    refreshed_fights = pd.concat([original_fights, pd.DataFrame([template])], ignore_index=True)
    monkeypatch.setitem(api_main._state, "fights", refreshed_fights)

    resp = client.post("/predict", json={
        "model_name": "xgboost_model", "division": "Light Heavyweight",
        "red_fighter_id": jones, "blue_fighter_id": cormier,
    })
    assert resp.status_code == 200
    data = resp.json()

    # Picked up immediately, no restart/reload step beyond the dict
    # assignment above - proving nothing holds a stale copy of the fights
    # table across requests.
    assert data["red_snapshot_date"] == new_fight_date.date().isoformat()
    assert data["red_snapshot_date"] != baseline_cutoff
    assert data["red_stats"]["record"]["prior_fights"] == baseline_prior_fights + 1

    # The real past meetings against Cormier (unaffected by this synthetic
    # fight, which is against a different, fake opponent) must still be
    # exactly the same ones as before - the new row must not shadow or
    # duplicate any real historical data.
    assert {m["fight_id"] for m in data["past_meetings"]} == {m["fight_id"] for m in baseline["past_meetings"]}


def test_predict_excludes_a_fight_the_requested_year_cutoff_precedes(client, monkeypatch):
    # The flip side of the freshness test above: a newly-added fight must
    # NOT leak into a snapshot whose requested cutoff is strictly before
    # it - no look-ahead, even once the fight exists somewhere in the
    # table.
    jones = _fighter_id(client, "Light Heavyweight", "Jon Jones")
    cormier = _fighter_id(client, "Light Heavyweight", "Daniel Cormier")

    original_fights = api_main._state["fights"]
    jones_fights = original_fights[original_fights["r_fighter_id"] == jones]
    pre_cutoff_year = int(jones_fights["event_date"].max().year)
    # Forced into the NEXT calendar year outright (not "+30 days", which
    # could land in the same year as pre_cutoff_year and get picked up by
    # red_year=pre_cutoff_year itself, defeating the point of this test).
    new_fight_date = pd.Timestamp(f"{pre_cutoff_year + 1}-01-15")

    template = jones_fights.iloc[-1].to_dict()
    template.update({
        "fight_id": "__regression_future_fight__",
        "event_date": new_fight_date,
        "winner_id": jones,
        "result_status": "win",
        "weight_class": "Light Heavyweight",
        "b_fighter_id": "__regression_future_opponent__",
    })
    refreshed_fights = pd.concat([original_fights, pd.DataFrame([template])], ignore_index=True)
    monkeypatch.setitem(api_main._state, "fights", refreshed_fights)

    resp = client.post("/predict", json={
        "model_name": "xgboost_model", "division": "Light Heavyweight",
        "red_fighter_id": jones, "red_year": pre_cutoff_year,
        "blue_fighter_id": cormier,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["red_snapshot_date"] != new_fight_date.date().isoformat()
    assert pd.Timestamp(data["red_snapshot_date"]).year == pre_cutoff_year
