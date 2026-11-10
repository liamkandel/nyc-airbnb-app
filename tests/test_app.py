"""Smoke tests: model loads, predictions are sane, every page renders, API answers."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from pathlib import Path

from src.model import load_model

ROOT = Path(__file__).resolve().parent.parent

ROW = dict(name="Sunny 2BR loft", neighbourhood_group="Manhattan", neighbourhood="Chelsea",
           room_type="Entire home/apt", latitude=40.7465, longitude=-74.0014, minimum_nights=2,
           number_of_reviews=20, reviews_per_month=1.0, calculated_host_listings_count=1,
           availability_365=120, days_since_last_review=30)


def test_prediction_is_sane():
    p = load_model().predict(pd.DataFrame([ROW])).iloc[0]
    assert 50 < p["predicted_price"] < 1000
    assert p["price_low"] <= p["predicted_price"] <= p["price_high"]


def test_room_type_ordering():
    m = load_model()
    rows = pd.DataFrame([{**ROW, "room_type": r, "name": ""} for r in ["Entire home/apt", "Private room", "Shared room"]])
    p = m.predict(rows)["predicted_price"].tolist()
    assert p[0] > p[1] > p[2]


@pytest.mark.parametrize("page", ["home", "explore", "neighborhoods", "estimator", "performance", "api_docs"])
def test_pages_render(page):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    if page != "home":
        at.switch_page(f"views/{page}.py").run()
    assert not at.exception, at.exception
    assert at.title[0].value  # every page has a title


def test_api():
    from api import app
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/predict", json={"neighbourhood": "Williamsburg", "room_type": "Private room", "name": "cozy room"})
    assert r.status_code == 200 and 30 < r.json()["predicted_price"] < 300
    assert c.post("/predict", json={"neighbourhood": "Atlantis", "room_type": "Private room"}).status_code == 422
    b = c.post("/predict/batch", json=[{"neighbourhood": "Harlem", "room_type": "Entire home/apt"}] * 3)
    assert len(b.json()) == 3
    lst = c.get("/listings", params={"borough": "Queens", "max_price": 80, "limit": 5}).json()
    assert lst["total"] > 0 and len(lst["results"]) == 5 and all(x["price"] <= 80 for x in lst["results"])
    assert len(c.get("/neighbourhoods", params={"borough": "Bronx"}).json()) > 10
