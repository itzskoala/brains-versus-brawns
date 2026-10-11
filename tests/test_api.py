import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


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
