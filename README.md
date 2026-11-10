# NYC Airbnb Price Explorer & Estimator

An interactive data product built on **48,895 real New York City Airbnb listings** (Inside Airbnb, July 2019).
It has two parts that share one trained model:

- **A Streamlit web app** for exploring listings on a map, comparing neighborhoods and getting an
  instant, explainable nightly-price estimate for any listing you describe.
- **A FastAPI REST API** that serves the same predictions and filtered data as JSON.

**Live app:** _add your Streamlit Community Cloud link here after deploying (see section 4)._

---

## 1. Run the app on your own computer (about 3 minutes)

You need **Python 3.11, 3.12 or 3.13** ([python.org/downloads](https://www.python.org/downloads/)).

```bash
# 1. Open a terminal in this folder (the one that contains app.py)
cd nyc-airbnb-app

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install the dependencies
pip install -r requirements.txt

# 4. Start the app
streamlit run app.py
```

The app opens in your browser at **http://localhost:8501**. Stop it with `Ctrl+C` in the terminal.

The trained model and the processed data are already included, so there is no training step. To rebuild them
from the raw CSV, run `python train.py` (about one minute).

## 2. Using the app

Use the menu on the left to move between pages.

### Overview
Headline numbers (listings, neighborhoods, median price, model error) and median price by borough and room type.

### Explore listings
1. Set filters in the sidebar: **borough**, **neighborhood**, **room type**, **price range**, **maximum minimum-stay**,
   **minimum number of reviews**, **price compared with the model**, and **words in the title**.
2. The numbers, the map and the table all update immediately.
3. **Color points by** switches the map between price, room type, and *price vs model*
   (blue = priced below the model's typical range, red = above it).
4. Hover a point on the map to see the listing's title, price and the model's estimate.
5. Sort the table, for example *Best value first* to find listings priced furthest below what the model expects.
6. **Download matching listings (CSV)** exports every listing that matches your filters.

### Compare neighborhoods
1. Pick a **borough** and a **room type**. Neighborhoods with fewer listings than the threshold are hidden.
2. The ranking chart shows the most expensive neighborhoods. Hover a bar for the middle-50% price range.
3. **Head-to-head**: choose up to three neighborhoods to compare their price distributions and key statistics.
4. **Is cheaper more popular?** plots neighborhood price against review counts.

### Price estimator (the model)
1. Describe the listing: **title** (optional, but words like "2BR", "studio", "luxury", "loft", "view" matter),
   **borough**, **neighborhood**, **room type**, **minimum nights** and **days available per year**.
2. Optional: open **Location and review history** to enter exact coordinates, mark the listing as new
   (no reviews), or change the review and host details.
3. Optional: enter **your asking price** to see whether it is below, inside or above the typical range.
4. The page shows, live:
   - the **estimated nightly price** and an **80% range** (about 8 in 10 similar real listings fall inside it),
   - **Why this price?**: how much each factor raised or lowered the estimate,
   - **What if…**: the estimate for each room type and for different minimum stays,
   - **Similar listings nearby**: the 15 closest real listings of the same room type, on a map and in a table.

**Try this example:** Title `Luxury 2BR loft with skyline view`, Manhattan, Chelsea, Entire home/apt, 2 nights.
Then change the room type to Private room, or remove "Luxury" from the title, and watch the estimate move.

### Model performance
Test-set accuracy compared with a simple baseline, predicted vs real prices, feature importance,
accuracy by room type and a model card with limitations.

### REST API
How to start and call the API, with a request builder that shows a ready-to-copy `curl` and Python call.

## 3. Using the REST API

Start it (in the same activated environment):

```bash
uvicorn api:app --port 8000
```

Open **http://127.0.0.1:8000/docs** for interactive documentation where you can try every endpoint.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Status and model version |
| GET | `/neighbourhoods?borough=Brooklyn` | Valid neighborhood names, listing counts, median prices |
| POST | `/predict` | Price estimate and 80% range for one listing |
| POST | `/predict/batch` | Up to 10,000 listings per request |
| GET | `/listings?borough=Manhattan&room_type=Private room&max_price=120&limit=20` | Filtered listings with model estimates (paged with `limit` and `offset`) |
| GET | `/metrics` | Test-set accuracy of the model and the baseline |

Example:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"neighbourhood": "Williamsburg", "room_type": "Private room", "name": "Cozy room near the L train"}'
```

```json
{"predicted_price": 89.0, "price_low": 59.0, "price_high": 124.0, "neighbourhood_group": "Brooklyn"}
```

Only `neighbourhood` and `room_type` are required. Other fields (`latitude`, `longitude`, `minimum_nights`,
`availability_365`, `number_of_reviews`, `reviews_per_month`, `days_since_last_review`,
`calculated_host_listings_count`) have sensible defaults and are validated; an unknown neighborhood returns
HTTP 422 with a helpful message.

## 4. Deploy the app online (Streamlit Community Cloud, free)

1. Put this folder in a **public GitHub repository** (it is about 25 MB including data and model).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub and click **Create app**.
3. Choose the repository and branch, set **Main file path** to `app.py`, and under **Advanced settings**
   choose Python **3.12** or **3.13**.
4. Click **Deploy**. The first start takes a few minutes while packages install. Copy the URL into the top of this file.

## 5. How it works

**Data.** [New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data),
originally from [Inside Airbnb](http://insideairbnb.com/get-the-data/) (CC BY 4.0). Host names are dropped on load.
Listings priced below $10 or above $1,000 (250 rows, mostly $0 placeholders and $10,000 "do not book" prices) are
excluded from training but still shown in the explorer.

**Features.** Borough, neighborhood, room type, latitude/longitude, distance to Times Square, Wall Street and
Williamsburg, minimum nights, days available, review count, reviews per month, days since last review, host
listing count, and signals parsed from the title (bedroom and bathroom counts, and keyword groups such as luxury,
studio, loft, view, outdoor space, cozy, budget).

**Model.** LightGBM gradient-boosted trees on log(price), plus two LightGBM quantile models (10th and 90th percentile)
for the range. The range is widened with conformal calibration so that about 80% of real prices fall inside it.
Explanations are per-listing TreeSHAP contributions.

**Results on 9,729 held-out listings**

| | Model | Baseline (median of same neighborhood + room type) |
|---|---|---|
| Typical (median) error | **20.5%** | 25.0% |
| Average error (MAE) | **$40** | $51 |
| R² on log price | **0.71** | 0.55 |
| 80% range coverage | 81% | – |

**Scalability and design choices**
- Data is stored as Parquet and loaded once per server with `st.cache_data`; the model is loaded once with
  `st.cache_resource`, so every interaction is a fast in-memory prediction.
- All feature engineering lives in one module (`src/features.py`) shared by training, the app and the API, so they
  can never disagree.
- The API supports batch prediction (10,000 listings per call) and paged listing queries, and can be run with
  several workers (`uvicorn api:app --workers 4`) or behind any container host.
- The "price vs model" label for each listing uses 5-fold out-of-fold predictions, so no listing is judged by a model
  that was trained on it.
- `python train.py` rebuilds everything from the raw CSV, so the pipeline can be rerun on a newer Inside Airbnb export
  with the same columns.

**Limitations.** Prices are 2019 nightly list prices and predate the 2023 NYC short-term rental rules. The dataset has
no bedroom, bathroom, amenity or photo fields, so size can only come from the listing title.

## 6. Project structure

```
app.py                 Streamlit entry point and navigation
views/                 One file per app page
src/features.py        Cleaning and feature engineering (shared)
src/model.py           Model loading, prediction, explanations
src/ui.py              Cached loaders, colors, chart styling
api.py                 FastAPI REST API
train.py               Training pipeline (writes data/*.parquet and models/*)
data/AB_NYC_2019.csv   Raw data
models/                Trained model and metrics
tests/test_app.py      Smoke tests: model, every page, API  (run: python -m pytest tests)
```
