import pandas as pd
import pytest
from fastapi.testclient import TestClient

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

    # a real past meeting (UFC 182) should show up with its own
    # predicted-vs-actual comparison.
    assert len(data["past_meetings"]) >= 1
    meeting = data["past_meetings"][0]
    assert meeting["actual"]["winner"] in ("red", "blue")
    assert 0.0 <= meeting["predicted"]["red_win_probability"] <= 1.0


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
