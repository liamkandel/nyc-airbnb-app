"""REST API for the NYC Airbnb price model.

Run:   uvicorn api:app --reload        (then open http://127.0.0.1:8000/docs)
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.features import BOROUGHS, ROOM_TYPES, SCORED_PARQUET
from src.model import load_metrics, load_model

app = FastAPI(
    title="NYC Airbnb Price API",
    version="1.0",
    description="Nightly price predictions and listing data for 48,895 NYC Airbnb listings (July 2019).",
)

MODEL = load_model()
LISTINGS = pd.read_parquet(SCORED_PARQUET)
LISTINGS["room_type"] = LISTINGS["room_type"].astype(str)
LISTINGS["neighbourhood_group"] = LISTINGS["neighbourhood_group"].astype(str)
NB_BOROUGH = LISTINGS.groupby("neighbourhood")["neighbourhood_group"].first().to_dict()
CENTROIDS = LISTINGS.groupby("neighbourhood")[["latitude", "longitude"]].median()

Borough = Enum("Borough", {b.replace(" ", "_"): b for b in BOROUGHS}, type=str)
RoomType = Enum("RoomType", {r.split()[0].split("/")[0]: r for r in ROOM_TYPES}, type=str)


class ListingIn(BaseModel):
    neighbourhood: str = Field(..., examples=["Williamsburg"], description="One of GET /neighbourhoods")
    room_type: RoomType = Field(..., examples=["Entire home/apt"])
    name: str = Field("", description="Listing title; signals like '2BR', 'studio', 'luxury' are used",
                      examples=["Sunny 2BR with balcony near the park"])
    latitude: Optional[float] = Field(None, ge=40.49, le=40.92, description="Defaults to the neighborhood center")
    longitude: Optional[float] = Field(None, ge=-74.26, le=-73.69)
    minimum_nights: int = Field(2, ge=1, le=365)
    availability_365: int = Field(180, ge=0, le=365)
    number_of_reviews: int = Field(20, ge=0)
    reviews_per_month: float = Field(1.0, ge=0)
    days_since_last_review: Optional[int] = Field(30, ge=0, description="null for a listing with no reviews")
    calculated_host_listings_count: int = Field(1, ge=1)


class PredictionOut(BaseModel):
    predicted_price: float
    price_low: float = Field(..., description="Lower end of the 80% range")
    price_high: float = Field(..., description="Upper end of the 80% range")
    neighbourhood_group: str


def _to_frame(items: list[ListingIn]) -> pd.DataFrame:
    rows = []
    for it in items:
        if it.neighbourhood not in NB_BOROUGH:
            raise HTTPException(422, f"Unknown neighbourhood '{it.neighbourhood}'. See GET /neighbourhoods.")
        r = it.model_dump()
        r["room_type"] = it.room_type.value
        r["neighbourhood_group"] = NB_BOROUGH[it.neighbourhood]
        c = CENTROIDS.loc[it.neighbourhood]
        r["latitude"] = r["latitude"] if r["latitude"] is not None else float(c["latitude"])
        r["longitude"] = r["longitude"] if r["longitude"] is not None else float(c["longitude"])
        if r["number_of_reviews"] == 0:
            r["days_since_last_review"], r["reviews_per_month"] = None, 0.0
        rows.append(r)
    return pd.DataFrame(rows)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL.version, "listings": len(LISTINGS)}


@app.get("/neighbourhoods")
def neighbourhoods(borough: Optional[Borough] = None):
    """Neighborhoods the model knows, with listing counts and median price."""
    df = LISTINGS if borough is None else LISTINGS[LISTINGS["neighbourhood_group"] == borough.value]
    g = df.groupby(["neighbourhood_group", "neighbourhood"]).agg(
        listings=("id", "size"), median_price=("price", "median")).reset_index()
    return g.sort_values(["neighbourhood_group", "neighbourhood"]).to_dict(orient="records")


@app.post("/predict", response_model=PredictionOut)
def predict(listing: ListingIn):
    """Predict the nightly price of one listing."""
    df = _to_frame([listing])
    p = MODEL.predict(df).iloc[0]
    return PredictionOut(**p.to_dict(), neighbourhood_group=df["neighbourhood_group"].iloc[0])


@app.post("/predict/batch", response_model=list[PredictionOut])
def predict_batch(listings: list[ListingIn]):
    """Predict up to 10,000 listings in one call."""
    if not listings:
        return []
    if len(listings) > 10_000:
        raise HTTPException(413, "Send at most 10,000 listings per request.")
    df = _to_frame(listings)
    p = MODEL.predict(df)
    p["neighbourhood_group"] = df["neighbourhood_group"]
    return p.to_dict(orient="records")


@app.get("/listings")
def get_listings(
    borough: Optional[Borough] = None,
    neighbourhood: Optional[str] = None,
    room_type: Optional[RoomType] = None,
    min_price: int = Query(0, ge=0),
    max_price: int = Query(100_000, ge=0),
    value_label: Optional[str] = Query(None, description="'Below typical range', 'Typical' or 'Above typical range'"),
    sort: str = Query("price", pattern="^(price|-price|price_vs_predicted_pct|-number_of_reviews)$"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Filter listings. Each result includes the model's out-of-fold estimate."""
    df = LISTINGS
    if borough:
        df = df[df["neighbourhood_group"] == borough.value]
    if neighbourhood:
        df = df[df["neighbourhood"] == neighbourhood]
    if room_type:
        df = df[df["room_type"] == room_type.value]
    df = df[df["price"].between(min_price, max_price)]
    if value_label:
        df = df[df["value_label"] == value_label]
    desc = sort.startswith("-")
    df = df.sort_values(sort.lstrip("-"), ascending=not desc)
    cols = ["id", "name", "neighbourhood_group", "neighbourhood", "room_type", "latitude", "longitude",
            "price", "predicted_price", "price_low", "price_high", "price_vs_predicted_pct", "value_label",
            "minimum_nights", "number_of_reviews", "availability_365"]
    page = df[cols].iloc[offset:offset + limit]
    return {"total": int(len(df)), "offset": offset, "limit": limit,
            "results": page.astype(object).where(page.notna(), None).to_dict(orient="records")}


@app.get("/metrics")
def metrics():
    """Hold-out accuracy of the model and of the baseline."""
    m = load_metrics()
    return {k: m[k] for k in ["model", "baseline", "by_room_type", "n_listings_modelled", "trained_at"]}
