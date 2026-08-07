"""
Trains one LightGBM model per product (Agile, FreePhase), using horizon_days
as an explicit feature so a single model covers the full 2-14 day range
rather than needing 13 separate models per horizon.

Why gradient boosting rather than a baseline or deep learning here:
  - The feature set is small/tabular (calendar, lags, forecast demand) --
    exactly where GBMs outperform neural nets, which need far more data to
    beat a well-tuned tree ensemble on this kind of problem.
  - Half-hourly rates 14 days out are dominated by seasonal/calendar
    structure and band (Red/Amber/Green) rather than fine continuous
    dynamics, which trees handle natively without scaling or embeddings.
  - Quantile objective gives you an uncertainty band for free, which matters
    more than the point estimate this far out -- useful directly in your
    dashboard as "expect somewhere between X-Y p/kWh" rather than a single
    number pretending to be precise 14 days ahead.

Time-based (not random) train/validation split, because random shuffling on
a time series leaks future information into training via adjacent slots.
"""

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

FEATURE_COLUMNS = [
    "horizon_days", "hour", "day_of_week", "is_weekend", "month", "hh_index",
    "forecast_demand_mean", "forecast_demand_min", "forecast_demand_max",
    "demand_7d_mean", "renewable_7d_mean",
]
CATEGORICAL_COLUMNS = ["freephase_band"]

QUANTILES = [0.1, 0.5, 0.9]


def prepare_features(df, product_key):
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    # lag7d/lag14d are intermediate columns only (conditionally null depending
    # on horizon -- see add_target_anchored_lag) and were combined into
    # price_anchor, which is always populated; lag28d is always safe on its
    # own and kept as a separate, more stable long-baseline reference.
    for suffix in ("price_anchor", "rate_lag28d"):
        col = f"{product_key}_{suffix}"
        if col in df.columns:
            feature_cols.append(col)
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]

    X = df[feature_cols + cat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    y = df[f"{product_key}_rate_p_kwh"]

    # Complete cases only: LightGBM will happily train through NaNs (it has
    # built-in missing-value splitting), which is exactly why this needs to
    # be explicit rather than left to the library -- a row with a missing
    # lag feature or missing demand forecast gets silently routed down a
    # default branch instead of excluded, which is a different thing from
    # what "only use rows where everything's present" means. Require the
    # target and every feature column non-null.
    complete = y.notna()
    for c in feature_cols + cat_cols:
        complete &= X[c].notna()

    n_dropped = (~complete).sum()
    if n_dropped:
        print(f"  [{product_key}] dropping {n_dropped}/{len(df)} rows with missing "
              f"target or feature values ({n_dropped/len(df):.1%})")

    return X[complete], y[complete], feature_cols, cat_cols


def time_based_split(df, calib_fraction=0.15, test_fraction=0.15):
    """Three-way time-ordered split: train / calibration / test.

    Calibration and test must be disjoint -- fitting the conformal margin on
    the same slice you then report coverage on is circular (it will always
    look correctly calibrated on data it was tuned against). Train comes
    first chronologically, then calibration, then test, so nothing before it
    in time leaks into a later split.
    """
    df_sorted = df.sort_values("target_timestamp")
    n = len(df_sorted)
    test_start = int(n * (1 - test_fraction))
    calib_start = int(n * (1 - test_fraction - calib_fraction))
    calib_cutoff = df_sorted.iloc[calib_start]["target_timestamp"]
    test_cutoff = df_sorted.iloc[test_start]["target_timestamp"]
    train_mask = df["target_timestamp"] < calib_cutoff
    calib_mask = (df["target_timestamp"] >= calib_cutoff) & (df["target_timestamp"] < test_cutoff)
    test_mask = df["target_timestamp"] >= test_cutoff
    return train_mask, calib_mask, test_mask


def train_product(df, product_key):
    if f"{product_key}_rate_p_kwh" not in df.columns:
        print(f"! no target column for {product_key}, skipping")
        return None

    X, y, feature_cols, cat_cols = prepare_features(df, product_key)
    aligned_df = df.loc[X.index]
    train_mask, calib_mask, test_mask = time_based_split(aligned_df)

    models = {}
    for q in QUANTILES:
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=q,
            n_estimators=400, learning_rate=0.03, num_leaves=31,
            min_child_samples=30, verbose=-1,
        )
        model.fit(
            X[train_mask], y[train_mask],
            categorical_feature=cat_cols,
            eval_set=[(X[calib_mask], y[calib_mask])],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        models[q] = model

    # Fit the conformal margin on the CALIBRATION slice...
    lo_calib = models[QUANTILES[0]].predict(X[calib_mask])
    hi_calib = models[QUANTILES[-1]].predict(X[calib_mask])
    y_calib = y[calib_mask].to_numpy()
    nonconformity = np.maximum(lo_calib - y_calib, y_calib - hi_calib)
    conformal_margin = float(np.quantile(nonconformity, QUANTILES[-1] - QUANTILES[0]))

    # ...then report both raw and calibrated coverage on the untouched TEST
    # slice, so the number printed below is an honest estimate of what
    # coverage looks like on data neither the models nor the margin have
    # seen -- not a number that's guaranteed to look good by construction.
    median_pred = models[0.5].predict(X[test_mask])
    mae = mean_absolute_error(y[test_mask], median_pred)
    lo_test = models[QUANTILES[0]].predict(X[test_mask])
    hi_test = models[QUANTILES[-1]].predict(X[test_mask])
    y_test = y[test_mask].to_numpy()
    raw_coverage = np.mean((y_test >= lo_test) & (y_test <= hi_test))
    lo_calibrated = lo_test - conformal_margin
    hi_calibrated = hi_test + conformal_margin
    calibrated_coverage = np.mean((y_test >= lo_calibrated) & (y_test <= hi_calibrated))

    print(f"[{product_key}] test MAE (median): {mae:.3f} p/kWh, "
          f"raw {int(QUANTILES[-1]*100 - QUANTILES[0]*100)}% band coverage: {raw_coverage:.1%} -> "
          f"after conformal calibration: {calibrated_coverage:.1%} "
          f"(margin: +/-{conformal_margin:.2f} p/kWh, n_calib={calib_mask.sum()}, n_test={test_mask.sum()})")

    return {"models": models, "feature_cols": feature_cols, "cat_cols": cat_cols,
            "mae": mae, "conformal_margin": conformal_margin}


def main():
    df = pd.read_csv("training_matrix.csv")
    # CSV round-trips don't reliably preserve UTC offsets through parse_dates;
    # go via fetch_neso.to_utc() so this matches the tz-awareness used everywhere
    # else in the pipeline rather than silently going tz-naive here.
    from fetch_neso import to_utc
    df["target_timestamp"] = to_utc(df["target_timestamp"])
    df["issue_timestamp"] = to_utc(df["issue_timestamp"])
    for product_key in ("agile", "freephase"):
        result = train_product(df, product_key)
        if result:
            joblib.dump(result, f"model_{product_key}.joblib")
            print(f"Saved model_{product_key}.joblib")


if __name__ == "__main__":
    main()
