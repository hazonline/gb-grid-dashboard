"""
Live inference: pulls NESO's *current* 2-14 day ahead half-hourly demand
forecast (the live file, refreshed twice daily -- distinct from the historic
archive used for training, which is Cardinal-Point-only; see fetch_neso.py).

Because training features are daily aggregates (forecast_demand_mean/min/max
per target date + horizon, built from a handful of Cardinal Points), this
aggregates the live half-hourly forecast down to the same daily shape before
scoring -- otherwise train/serve feature distributions won't match even
though the column names line up. It's an approximation either way: training
saw a few Cardinal Points per day, inference sees up to 48 real half-hourly
values per day. Worth watching in validation once you have live predictions
to compare against actual outcomes.
"""

import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from fetch_neso import fetch_dataset_records, _first_matching_column, to_utc, get_recent_actual_demand_and_renewables
from fetch_tariffs import fetch_full_tariff_history
from build_training_matrix import add_calendar_features, build as build_matrix
from train_model import QUANTILES, prepare_features

ROLLING_CALIBRATION_WINDOW_DAYS = 30
ROLLING_CALIBRATION_MIN_POINTS = 500  # below this, the recent window is too thin to trust; fall back to training-time margin

LIVE_14DA_RESOURCE = "7c0411cd-2714-4bb5-a408-adb065edf34d"  # ng-demand-14da-hh.csv


def fetch_live_forecast():
    df = fetch_dataset_records(LIVE_14DA_RESOURCE)
    if df.empty:
        return df
    target_col = _first_matching_column(df, ["GDATETIME", "TARGETDATE", "SETTLEMENT_DATE", "FORECASTDATE"])
    value_col = _first_matching_column(df, ["NATIONALDEMAND", "FORECASTDEMAND", "DEMAND", "ND_FORECAST"])
    if target_col is None or value_col is None:
        print("! could not auto-map live 14da forecast columns -- inspect df.columns:", list(df.columns))
        return df
    out = df.rename(columns={target_col: "target_timestamp", value_col: "forecast_demand_mw"})
    out["target_timestamp"] = to_utc(out["target_timestamp"])
    return out


def build_inference_frame():
    now = datetime.now(timezone.utc)
    live = fetch_live_forecast()
    if live.empty or "target_timestamp" not in live.columns:
        raise RuntimeError("Live 14-day forecast unavailable or unparseable -- check schema.")

    live["target_date"] = live["target_timestamp"].dt.normalize()
    live["horizon_days"] = ((live["target_date"] - now.replace(hour=0, minute=0, second=0, microsecond=0))
                             .dt.total_seconds() / 86400).round().astype(int)
    live = live[(live["horizon_days"] >= 2) & (live["horizon_days"] <= 14)]

    daily = (
        live.groupby(["target_date", "horizon_days"])["forecast_demand_mw"]
        .agg(forecast_demand_mean="mean", forecast_demand_min="min", forecast_demand_max="max")
        .reset_index()
    )

    # Build the half-hourly output grid for the next 14 days and attach the
    # daily forecast aggregate to every slot on that date -- same structure
    # add_calendar_features() expects, mirroring build_training_matrix.py.
    slots = pd.date_range(
        start=now.replace(minute=0 if now.minute < 30 else 30, second=0, microsecond=0),
        periods=48 * 14, freq="30min", tz="UTC",
    )
    grid = pd.DataFrame({"target_timestamp": slots})
    grid["target_date"] = grid["target_timestamp"].dt.normalize()
    grid = grid.merge(daily, on="target_date", how="inner")
    grid = add_calendar_features(grid, "target_timestamp")

    # demand_7d_mean / renewable_7d_mean: trailing 7-day actual trend as of
    # right now, computed the same way build_training_matrix.py computes it
    # historically -- just against the rolling "Demand Data Update" resource
    # instead of a full year archive, since all we need here is the last
    # ~7 days, not deep history.
    recent = get_recent_actual_demand_and_renewables()
    if not recent.empty:
        recent_sorted = recent.sort_values("timestamp")
        recent_sorted["demand_7d_mean"] = recent_sorted["actual_demand_mw"].rolling(336, min_periods=48).mean()
        if "renewable_utilisation_pct" in recent_sorted.columns:
            recent_sorted["renewable_7d_mean"] = recent_sorted["renewable_utilisation_pct"].rolling(336, min_periods=48).mean()
        latest = recent_sorted.iloc[-1]
        grid["demand_7d_mean"] = latest.get("demand_7d_mean")
        grid["renewable_7d_mean"] = latest.get("renewable_7d_mean")
    else:
        print("! could not fetch recent actual demand/renewables -- "
              "demand_7d_mean and renewable_7d_mean will be missing")

    # Seasonal lag features, target-anchored -- must match build_training_matrix.py's
    # add_target_anchored_lag() exactly, or train/serve feature definitions
    # silently diverge. This replaced an issue_timestamp-anchored version
    # that collapsed to a single constant across the whole live batch (every
    # row shares issue_timestamp="now"), which was the actual root cause of
    # live predictions looking like a flat, repeating daily template despite
    # the model leaning on these features for up to 69% of its gain.
    #
    # Target-anchoring uses each row's own target_timestamp (which DOES vary
    # across the 14-day grid), matched against the actual rate at
    # (target_timestamp - lag_days) -- using merge_asof/nearest since exact
    # half-hourly alignment isn't guaranteed against real published data.
    for product_key in ("agile", "freephase"):
        tariff = fetch_full_tariff_history(product_key)
        if tariff.empty:
            print(f"! could not fetch {product_key} tariff history for lag features")
            continue
        tariff_sorted = tariff.sort_values("timestamp")
        col = f"{product_key}_rate_p_kwh"

        for lag in (7, 14, 28):
            lookup = pd.DataFrame({
                "target_timestamp": grid["target_timestamp"] - pd.Timedelta(days=lag),
                "_orig_idx": grid.index,
            }).sort_values("target_timestamp")
            matched = pd.merge_asof(
                lookup, tariff_sorted.rename(columns={"timestamp": "target_timestamp"}),
                on="target_timestamp", direction="nearest",
            ).set_index("_orig_idx").sort_index()
            feature_name = f"{product_key}_rate_lag{lag}d"
            grid[feature_name] = matched[col] if col in matched.columns else None
            # leakage guard: null out anywhere lag_days < horizon_days, exactly
            # matching the training-time safety rule
            grid.loc[grid["horizon_days"] > lag, feature_name] = pd.NA

        lag7_col, lag14_col = f"{product_key}_rate_lag7d", f"{product_key}_rate_lag14d"
        grid[f"{product_key}_price_anchor"] = grid[lag7_col].where(grid[lag7_col].notna(), grid[lag14_col])

    return grid


def compute_rolling_conformal_margin(product_key, window_days=ROLLING_CALIBRATION_WINDOW_DAYS,
                                      min_points=ROLLING_CALIBRATION_MIN_POINTS):
    """
    Recompute the conformal margin from the most recent `window_days` of
    ACTUAL outcomes, rather than trusting the static margin baked in at
    training time. In a volatile market, "how wrong was the model on
    average over its whole training history" and "how wrong has it been in
    the last month" can differ a lot -- this recalibrates against the latter
    every time predict.py runs, so the band tracks current conditions
    instead of a snapshot from whenever the model was last trained.

    Reuses build_training_matrix.build(since=...) rather than duplicating
    its feature logic, so a recent calibration row is guaranteed to be built
    the exact same way as a training row -- no risk of the two silently
    drifting into different feature definitions over time.

    Trade-off: this re-fetches all source datasets (NESO forecast archive,
    outturn demand, tariff history) every time predict.py runs, which is
    more network cost than reusing a cached matrix. Acceptable for a
    periodic/scheduled run; revisit if this ends up running much more often.
    """
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=window_days)
    recent_matrix = build_matrix(since=cutoff)
    if recent_matrix.empty:
        print(f"! [{product_key}] rolling calibration fetch returned nothing, "
              "falling back to training-time margin")
        return None, 0

    X, y, feature_cols, cat_cols = prepare_features(recent_matrix, product_key)
    if len(X) < min_points:
        print(f"! [{product_key}] only {len(X)} recent calibration points (need {min_points}+) "
              "-- falling back to training-time margin")
        return None, len(X)

    bundle = joblib.load(f"model_{product_key}.joblib")
    lo = bundle["models"][QUANTILES[0]].predict(X)
    hi = bundle["models"][QUANTILES[-1]].predict(X)
    y_arr = y.to_numpy()
    nonconformity = np.maximum(lo - y_arr, y_arr - hi)
    margin = float(np.quantile(nonconformity, QUANTILES[-1] - QUANTILES[0]))
    print(f"[{product_key}] rolling conformal margin (last {window_days}d, n={len(X)}): +/-{margin:.2f} p/kWh")
    return margin, len(X)


def predict_product(df, product_key):
    bundle = joblib.load(f"model_{product_key}.joblib")
    feature_cols, cat_cols = bundle["feature_cols"], bundle["cat_cols"]

    missing = [c for c in feature_cols + cat_cols if c not in df.columns]
    if missing:
        print(f"! [{product_key}] missing features at inference time: {missing} "
              "-- fill these in build_inference_frame() before trusting the output.")
        for c in missing:
            df[c] = 0

    X = df[feature_cols + cat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")

    for q, model in bundle["models"].items():
        df[f"{product_key}_p{int(q*100)}"] = model.predict(X)

    # Prefer a freshly-computed rolling margin over the training-time static
    # one -- see compute_rolling_conformal_margin() docstring. Falls back
    # cleanly if there isn't enough recent data yet.
    rolling_margin, n_points = compute_rolling_conformal_margin(product_key)
    if rolling_margin is not None:
        margin = rolling_margin
    else:
        margin = bundle.get("conformal_margin", 0.0)
        print(f"[{product_key}] using training-time static margin: +/-{margin:.2f} p/kWh")

    lo_col, hi_col = f"{product_key}_p{int(QUANTILES[0]*100)}", f"{product_key}_p{int(QUANTILES[-1]*100)}"
    if lo_col in df.columns and hi_col in df.columns:
        df[lo_col] = df[lo_col] - margin
        df[hi_col] = df[hi_col] + margin
    return df


def main():
    df = build_inference_frame()
    for product_key in ("agile", "freephase"):
        df = predict_product(df, product_key)

    out_cols = ["target_timestamp", "horizon_days", "freephase_band"]
    out_cols += [c for c in df.columns if c.startswith(("agile_p", "freephase_p"))]
    result = df[out_cols].sort_values("target_timestamp")
    result.to_csv("predicted_rates_14d.csv", index=False)
    print(f"Saved predicted_rates_14d.csv ({len(result)} rows)")
    print(result.head(10))


if __name__ == "__main__":
    main()
