"""Train the NYC Airbnb nightly-price model.

Usage:  python train.py

Outputs
  data/listings_clean.parquet   cleaned listings used by the dashboard
  data/listings_scored.parquet  listings + out-of-fold price predictions
  models/price_model.joblib     point model + 80% interval models
  models/metrics.json           test-set metrics, baseline, feature importance
"""
from __future__ import annotations

import json
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from src.features import (
    CLEAN_PARQUET, FEATURES, METRICS_PATH, MODEL_PATH, SCORED_PARQUET,
    build_features, clean, load_raw,
)

SEED = 42
POINT_PARAMS = dict(
    objective="regression", n_estimators=1200, learning_rate=0.03, num_leaves=63,
    min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=1.0, cat_smooth=20, min_data_per_group=30, random_state=SEED, verbose=-1,
)
QUANTILE_PARAMS = dict(
    n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1,
)


def fit_models(X: pd.DataFrame, y_log: pd.Series):
    point = lgb.LGBMRegressor(**POINT_PARAMS).fit(X, y_log)
    low = lgb.LGBMRegressor(objective="quantile", alpha=0.10, **QUANTILE_PARAMS).fit(X, y_log)
    high = lgb.LGBMRegressor(objective="quantile", alpha=0.90, **QUANTILE_PARAMS).fit(X, y_log)
    return point, low, high


def conformal_adjust(y_log, lo_log, hi_log, coverage=0.80) -> float:
    """How much to widen the quantile interval (in log space) to reach the target coverage."""
    scores = np.maximum(lo_log - y_log, y_log - hi_log)
    n = len(scores)
    return float(np.quantile(scores, min(1.0, np.ceil((n + 1) * coverage) / n)))


def baseline_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Median price for the same neighbourhood and room type (falls back to room type)."""
    by_nb = train.groupby(["neighbourhood", "room_type"], observed=True)["price"].median()
    by_rt = train.groupby("room_type", observed=True)["price"].median()
    keys = list(zip(test["neighbourhood"], test["room_type"]))
    return np.array([by_nb.get(k, by_rt[k[1]]) for k in keys], dtype=float)


def main() -> None:
    t0 = time.time()
    df = clean(load_raw())
    df.to_parquet(CLEAN_PARQUET, index=False)
    neighbourhoods = sorted(df["neighbourhood"].unique())

    train_df = df[df["in_training_range"]].reset_index(drop=True)
    X_all = build_features(train_df, neighbourhoods)
    y_all = np.log1p(train_df["price"])

    # ---- 1. Hold-out evaluation -------------------------------------------
    idx_tr, idx_te = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=SEED)
    # Hold back part of the training split to calibrate the 80% interval
    # (conformalized quantile regression), so test coverage is honest.
    idx_fit, idx_cal = train_test_split(idx_tr, test_size=0.15, random_state=SEED)
    _, low_c, high_c = fit_models(X_all.iloc[idx_fit], y_all.iloc[idx_fit])
    q_hold = conformal_adjust(y_all.iloc[idx_cal].to_numpy(), low_c.predict(X_all.iloc[idx_cal]),
                              high_c.predict(X_all.iloc[idx_cal]))
    point, low, high = fit_models(X_all.iloc[idx_tr], y_all.iloc[idx_tr])
    X_te, price_te = X_all.iloc[idx_te], train_df["price"].iloc[idx_te].to_numpy()
    pred = np.expm1(point.predict(X_te))
    lo, hi = np.expm1(low.predict(X_te) - q_hold), np.expm1(high.predict(X_te) + q_hold)
    base = baseline_predict(train_df.iloc[idx_tr], train_df.iloc[idx_te])

    ape = np.abs(pred - price_te) / price_te
    metrics = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "n_listings_total": int(len(df)),
        "n_listings_modelled": int(len(train_df)),
        "price_range": [10, 1000],
        "test_size": int(len(idx_te)),
        "model": {
            "mae_usd": round(float(mean_absolute_error(price_te, pred)), 2),
            "median_abs_pct_error": round(float(np.median(ape) * 100), 1),
            "within_25pct": round(float(np.mean(ape <= 0.25) * 100), 1),
            "r2_log_price": round(float(r2_score(np.log1p(price_te), np.log1p(pred))), 3),
            "interval_80_coverage": round(float(np.mean((price_te >= lo) & (price_te <= hi)) * 100), 1),
            "interval_80_median_width_usd": round(float(np.median(hi - lo)), 0),
        },
        "baseline": {
            "description": "Median price of listings with the same neighborhood and room type",
            "mae_usd": round(float(mean_absolute_error(price_te, base)), 2),
            "median_abs_pct_error": round(float(np.median(np.abs(base - price_te) / price_te) * 100), 1),
            "r2_log_price": round(float(r2_score(np.log1p(price_te), np.log1p(base))), 3),
        },
    }
    by_room = []
    for rt in train_df["room_type"].cat.categories:
        m = (train_df["room_type"].iloc[idx_te] == rt).to_numpy()
        by_room.append({"room_type": rt, "n": int(m.sum()),
                        "mae_usd": round(float(mean_absolute_error(price_te[m], pred[m])), 2),
                        "median_abs_pct_error": round(float(np.median(ape[m]) * 100), 1)})
    metrics["by_room_type"] = by_room
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(idx_te), size=min(3000, len(idx_te)), replace=False)
    metrics["test_sample"] = {
        "actual": price_te[sample].round(0).tolist(),
        "predicted": pred[sample].round(0).tolist(),
        "room_type": train_df["room_type"].iloc[idx_te].astype(str).to_numpy()[sample].tolist(),
    }
    print("Hold-out:", json.dumps(metrics["model"]), "\nBaseline:", json.dumps(metrics["baseline"]))

    # ---- 2. Out-of-fold predictions for every listing (honest "value" score)
    oof = np.zeros((len(train_df), 3))
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(X_all)):
        p, l, h = fit_models(X_all.iloc[tr], y_all.iloc[tr])
        oof[te] = np.column_stack([p.predict(X_all.iloc[te]), l.predict(X_all.iloc[te]), h.predict(X_all.iloc[te])])
        print(f"  fold {k + 1}/5 done")
    q_final = conformal_adjust(y_all.to_numpy(), oof[:, 1], oof[:, 2])
    oof[:, 1] -= q_final
    oof[:, 2] += q_final
    metrics["interval_log_adjustment"] = round(q_final, 4)
    print(f"  conformal widening (log space): {q_final:.3f}")
    oof = np.expm1(oof)

    # ---- 3. Final model on all modelled listings ------------------------------
    point, low, high = fit_models(X_all, y_all)
    imp = pd.Series(point.booster_.feature_importance("gain"), index=FEATURES)
    metrics["feature_importance"] = (imp / imp.sum() * 100).round(2).sort_values(ascending=False).to_dict()

    bundle = {"point": point, "low": low, "high": high, "neighbourhoods": neighbourhoods,
              "interval_adjust": q_final, "version": metrics["trained_at"]}
    joblib.dump(bundle, MODEL_PATH, compress=3)

    scored = df.copy()
    scored[["predicted_price", "price_low", "price_high"]] = np.nan
    in_range = scored["in_training_range"].to_numpy()
    scored.loc[in_range, ["predicted_price", "price_low", "price_high"]] = oof
    # Listings outside the training range ($0 or > $1000) get the final model's estimate.
    X_out = build_features(scored.loc[~in_range], neighbourhoods)
    scored.loc[~in_range, "predicted_price"] = np.expm1(point.predict(X_out))
    scored.loc[~in_range, "price_low"] = np.expm1(low.predict(X_out) - q_final)
    scored.loc[~in_range, "price_high"] = np.expm1(high.predict(X_out) + q_final)
    scored["price_low"] = np.minimum(scored["price_low"], scored["predicted_price"])
    scored["price_high"] = np.maximum(scored["price_high"], scored["predicted_price"])
    scored[["predicted_price", "price_low", "price_high"]] = scored[["predicted_price", "price_low", "price_high"]].round(0)
    scored["price_vs_predicted_pct"] = ((scored["price"] / scored["predicted_price"] - 1) * 100).round(1)
    scored["value_label"] = np.select(
        [scored["price"] < scored["price_low"], scored["price"] > scored["price_high"]],
        ["Below typical range", "Above typical range"], "Typical",
    )
    scored.loc[~in_range, "value_label"] = "Not modelled (price outlier)"
    scored.to_parquet(SCORED_PARQUET, index=False)

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved model, metrics and scored listings in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
