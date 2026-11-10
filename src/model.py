"""Loading the trained price model and turning inputs into predictions."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from .features import FEATURES, METRICS_PATH, MODEL_PATH, build_features

# Human-readable names for the explanation chart.
FEATURE_LABELS = {
    "neighbourhood_group": "Borough",
    "neighbourhood": "Neighborhood",
    "room_type": "Room type",
    "latitude": "Exact location (lat)",
    "longitude": "Exact location (lon)",
    "minimum_nights": "Minimum nights",
    "number_of_reviews": "Number of reviews",
    "reviews_per_month": "Reviews per month",
    "calculated_host_listings_count": "Host's listing count",
    "availability_365": "Days available per year",
    "days_since_last_review": "Days since last review",
    "title_bedrooms": "Bedrooms (from title)",
    "title_baths": "Bathrooms (from title)",
    "title_length": "Title length",
    "dist_times_sq_km": "Distance to Times Square",
    "dist_wall_st_km": "Distance to Wall Street",
    "dist_williamsburg_km": "Distance to Williamsburg",
    "kw_luxury": "Title says luxury",
    "kw_studio": "Title says studio",
    "kw_loft": "Title says loft",
    "kw_penthouse": "Title says penthouse",
    "kw_townhouse": "Title says townhouse/brownstone",
    "kw_duplex": "Title says duplex",
    "kw_view": "Title mentions a view",
    "kw_outdoor": "Title mentions outdoor space",
    "kw_spacious": "Title says spacious/large",
    "kw_cozy": "Title says cozy/small",
    "kw_renovated": "Title says new/renovated/modern",
    "kw_budget": "Title says budget/cheap",
    "kw_group": "Title mentions groups/families",
}


@dataclass
class PriceModel:
    point: object  # LightGBM regressor on log1p(price)
    low: object  # 10th percentile quantile model
    high: object  # 90th percentile quantile model
    neighbourhoods: list[str]
    version: str
    interval_adjust: float = 0.0  # conformal widening of the interval, in log space

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        return build_features(df, self.neighbourhoods)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return predicted nightly price in USD with an 80% prediction interval."""
        X = self.features(df)
        mid = np.expm1(self.point.predict(X))
        lo = np.expm1(self.low.predict(X) - self.interval_adjust)
        hi = np.expm1(self.high.predict(X) + self.interval_adjust)
        # Quantile models are trained separately, so enforce lo <= mid <= hi.
        lo = np.minimum(lo, mid)
        hi = np.maximum(hi, mid)
        return pd.DataFrame(
            {"predicted_price": mid, "price_low": lo, "price_high": hi}, index=df.index
        ).round(0)

    def explain(self, df_one_row: pd.DataFrame) -> tuple[float, pd.DataFrame]:
        """Per-feature contributions for a single listing.

        Returns the base price (average listing) and a table of each feature's
        multiplicative effect on the price, from LightGBM's TreeSHAP values.
        """
        X = self.features(df_one_row)
        contrib = self.point.predict(X, pred_contrib=True)[0]
        base = float(np.expm1(contrib[-1]))
        effects = pd.DataFrame({"feature": FEATURES, "log_effect": contrib[:-1]})
        # Merge lat/lon into a single "exact location" effect: they only make
        # sense together.
        loc = effects["feature"].isin(["latitude", "longitude"])
        if loc.any():
            merged = pd.DataFrame({"feature": ["location"], "log_effect": [effects.loc[loc, "log_effect"].sum()]})
            effects = pd.concat([effects[~loc], merged], ignore_index=True)
        labels = {**FEATURE_LABELS, "location": "Exact location (lat/lon)"}
        effects["label"] = effects["feature"].map(labels)
        effects["pct_effect"] = (np.exp(effects["log_effect"]) - 1) * 100
        effects = effects.reindex(effects["log_effect"].abs().sort_values(ascending=False).index)
        return base, effects.reset_index(drop=True)


@lru_cache(maxsize=1)
def load_model() -> PriceModel:
    bundle = joblib.load(MODEL_PATH)
    return PriceModel(**bundle)


@lru_cache(maxsize=1)
def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text())
