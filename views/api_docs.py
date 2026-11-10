import json

import streamlit as st

from src.features import BOROUGHS, ROOM_TYPES
from src.ui import listings, model

st.title("REST API")
st.markdown(
    "The same model and data are available as a JSON API built with **FastAPI**, for use from scripts, "
    "notebooks or other apps. The API runs on your own machine (see the user guide for hosting it online)."
)

st.subheader("1. Start the API")
st.code("pip install -r requirements.txt\nuvicorn api:app --port 8000", language="bash")
st.markdown("Interactive docs, where you can try every endpoint in the browser, open at "
            "**http://127.0.0.1:8000/docs**.")

st.subheader("2. Endpoints")
st.markdown(
    """
| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Status, model version and number of listings |
| GET | `/neighbourhoods?borough=Brooklyn` | Valid neighborhood names with listing counts and median prices |
| POST | `/predict` | Price estimate and 80% range for one listing |
| POST | `/predict/batch` | Estimates for up to 10,000 listings in one request |
| GET | `/listings?borough=Manhattan&room_type=Private room&max_price=120&limit=20` | Filtered listings with the model's estimate for each |
| GET | `/metrics` | Test-set accuracy of the model and the baseline |
"""
)

st.subheader("3. Build a request")
c1, c2, c3 = st.columns(3)
df = listings()
borough = c1.selectbox("Borough", BOROUGHS)
hood = c2.selectbox("Neighborhood", sorted(df.loc[df["neighbourhood_group"] == borough, "neighbourhood"].unique()))
room = c3.selectbox("Room type", ROOM_TYPES)
title = st.text_input("Title", "Cozy studio near the subway")
body = {"neighbourhood": hood, "room_type": room, "name": title, "minimum_nights": 2, "availability_365": 180}

import pandas as pd  # noqa: E402

c = df[df["neighbourhood"] == hood][["latitude", "longitude"]].median()
row = pd.DataFrame([{**body, "neighbourhood_group": borough, "latitude": c["latitude"], "longitude": c["longitude"],
                     "number_of_reviews": 20, "reviews_per_month": 1.0, "days_since_last_review": 30,
                     "calculated_host_listings_count": 1}])
p = model().predict(row).iloc[0]
response = {"predicted_price": p["predicted_price"], "price_low": p["price_low"], "price_high": p["price_high"],
            "neighbourhood_group": borough}

t1, t2, t3 = st.tabs(["curl", "Python", "Response"])
with t1:
    st.code(f"curl -X POST http://127.0.0.1:8000/predict \\\n  -H 'Content-Type: application/json' \\\n"
            f"  -d '{json.dumps(body)}'", language="bash")
with t2:
    st.code(
        "import requests\n\n"
        f"listing = {json.dumps(body, indent=4)}\n"
        "r = requests.post('http://127.0.0.1:8000/predict', json=listing)\n"
        "print(r.json())", language="python")
with t3:
    st.code(json.dumps(response, indent=2), language="json")
    st.caption("This is the response the API returns for the request above (computed live here with the same model).")
