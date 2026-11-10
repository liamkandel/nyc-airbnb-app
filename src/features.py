"""Data cleaning and feature engineering shared by training, the app and the API.

Everything that turns a raw listing into model features lives here, so the
Streamlit app, the REST API and the training script can never drift apart.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "AB_NYC_2019.csv"
CLEAN_PARQUET = ROOT / "data" / "listings_clean.parquet"
SCORED_PARQUET = ROOT / "data" / "listings_scored.parquet"
MODEL_PATH = ROOT / "models" / "price_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

# Date the data was scraped (last review in the file is 2019-07-08).
SNAPSHOT_DATE = pd.Timestamp("2019-07-08")

# Price bounds used for training. Listings outside are almost always
# placeholders ($0) or "don't book me" prices ($10,000).
MIN_PRICE, MAX_PRICE = 10, 1000

BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room"]

LANDMARKS = {
    "dist_times_sq_km": (40.7580, -73.9855),
    "dist_wall_st_km": (40.7060, -74.0090),
    "dist_williamsburg_km": (40.7141, -73.9614),
}

# Keywords hosts put in listing titles that carry price information.
KEYWORDS = {
    "kw_luxury": r"luxur|upscale|deluxe|elegant",
    "kw_studio": r"studio",
    "kw_loft": r"loft",
    "kw_penthouse": r"penthouse",
    "kw_townhouse": r"townhouse|brownstone|house\b",
    "kw_duplex": r"duplex|triplex",
    "kw_view": r"view|skyline",
    "kw_outdoor": r"garden|terrace|balcony|roof|patio|backyard|yard",
    "kw_spacious": r"spacious|huge|large|big|massive",
    "kw_cozy": r"cozy|cosy|small|tiny|cute",
    "kw_renovated": r"renovat|new|modern|designer",
    "kw_budget": r"cheap|budget|affordable|bunk|hostel",
    "kw_group": r"group|family|families|sleeps",
}

BEDROOM_RE = re.compile(r"(\d+)\s*-?\s*(?:br\b|bd\b|bdr|bdrm|bed ?room|bedroom|bed\b|beds\b)")
BATH_RE = re.compile(r"(\d+(?:\.5)?)\s*-?\s*(?:ba\b|bath)")
WORD_NUM = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6"}

CATEGORICAL = ["neighbourhood_group", "neighbourhood", "room_type"]
NUMERIC = [
    "latitude",
    "longitude",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
    "days_since_last_review",
    "title_bedrooms",
    "title_baths",
    "title_length",
    *LANDMARKS.keys(),
    *KEYWORDS.keys(),
]
FEATURES = CATEGORICAL + NUMERIC


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _normalise_title(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.lower()
    for word, digit in WORD_NUM.items():
        s = s.str.replace(rf"\b{word}\b", digit, regex=True)
    return s


def title_features(titles: pd.Series) -> pd.DataFrame:
    t = _normalise_title(titles)
    out = pd.DataFrame(index=titles.index)
    beds = t.str.extract(BEDROOM_RE, expand=False).astype(float)
    beds = beds.where(beds <= 10)  # "100 bed hostel" etc. are not bedrooms
    beds = beds.mask(t.str.contains("studio") & beds.isna(), 0.0)
    out["title_bedrooms"] = beds
    out["title_baths"] = t.str.extract(BATH_RE, expand=False).astype(float).where(lambda x: x <= 6)
    out["title_length"] = t.str.len().astype(float)
    for col, pattern in KEYWORDS.items():
        out[col] = t.str.contains(pattern, regex=True).astype(float)
    return out


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Inside Airbnb export. Drops host names (personal data)."""
    df = raw.drop(columns=[c for c in ["host_name"] if c in raw.columns]).copy()
    df["name"] = df["name"].fillna("").astype(str)
    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0.0)
    df["minimum_nights"] = df["minimum_nights"].clip(upper=365)
    df["neighbourhood_group"] = pd.Categorical(df["neighbourhood_group"], categories=BOROUGHS)
    df["room_type"] = pd.Categorical(df["room_type"], categories=ROOM_TYPES)
    df["neighbourhood"] = df["neighbourhood"].astype(str)
    df["in_training_range"] = df["price"].between(MIN_PRICE, MAX_PRICE)
    return df


def build_features(df: pd.DataFrame, neighbourhoods: list[str] | None = None) -> pd.DataFrame:
    """Turn cleaned listings (or user input with the same columns) into model features."""
    X = pd.DataFrame(index=df.index)
    X["neighbourhood_group"] = pd.Categorical(df["neighbourhood_group"], categories=BOROUGHS)
    X["neighbourhood"] = pd.Categorical(df["neighbourhood"], categories=neighbourhoods)
    X["room_type"] = pd.Categorical(df["room_type"], categories=ROOM_TYPES)
    for col in ["latitude", "longitude", "minimum_nights", "number_of_reviews",
                "reviews_per_month", "calculated_host_listings_count", "availability_365"]:
        X[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    if "days_since_last_review" in df:
        X["days_since_last_review"] = pd.to_numeric(df["days_since_last_review"], errors="coerce")
    else:
        last = pd.to_datetime(df.get("last_review"), errors="coerce")
        X["days_since_last_review"] = (SNAPSHOT_DATE - last).dt.days.astype(float)
    X = X.join(title_features(df.get("name", pd.Series("", index=df.index))))
    for col, (lat, lon) in LANDMARKS.items():
        X[col] = haversine_km(X["latitude"], X["longitude"], lat, lon)
    return X[FEATURES]


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_CSV)
