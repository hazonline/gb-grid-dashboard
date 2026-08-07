"""
Builds the historical training matrix for 14-day-ahead rate prediction.

Design principle (this is the part the original script got wrong): every
feature must be something you would ACTUALLY have in hand at the moment
you're making a 2-14 day ahead prediction. Concretely:

  - forecast_demand_mw   -- NESO's own forecast, made at issue_timestamp,
                             for target_timestamp. Safe: it's forecast data,
                             exactly what you'd have at inference time too.
  - recent_actual_demand_* -- actual demand in the days *before* issue_timestamp.
                             Safe: backward-looking only.
  - embedded wind/solar forecast for target date. Safe: forward-looking
                             forecast, available at issue time.
  - calendar / band features of the target slot. Safe: known in advance.
  - lagged tariff rates (same slot N days before issue_timestamp). Safe:
                             backward-looking only.

  - actual_demand_mw AT THE TARGET TIME is deliberately NOT used as a
    feature anywhere -- only as a downstream sanity-check column, because
    you will never have it at prediction time for a date 14 days out.

Target variables: agile_rate_p_kwh, freephase_rate_p_kwh at target_timestamp.
"""

import pandas as pd
from fetch_neso import get_historic_outturn_demand, get_2_14_day_forecast_archive, get_embedded_wind_solar_forecast, to_utc
from fetch_tariffs import fetch_full_tariff_history


def assign_band(hour):
    if 16 <= hour < 19:
        return "Red"
    if 23 <= hour or hour < 6:
        return "Green"
    return "Amber"


def add_calendar_features(df, ts_col):
    # UK tariff bands (and real demand patterns) are defined in UK clock
    # time, which is BST (UTC+1) for roughly seven months of the year --
    # not UTC. Deriving hour/band from raw UTC, as this did before, silently
    # shifts every Red/Amber/Green boundary by an hour whenever BST is in
    # effect (spot-checked: 16:00 UTC in August is 17:00 UK local). Convert
    # to Europe/London first so DST transitions are handled automatically,
    # then derive every calendar feature from local time.
    local_ts = df[ts_col].dt.tz_convert("Europe/London")
    df["hour"] = local_ts.dt.hour
    df["day_of_week"] = local_ts.dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = local_ts.dt.month
    df["hh_index"] = local_ts.dt.hour * 2 + (local_ts.dt.minute >= 30).astype(int)
    df["freephase_band"] = df["hour"].apply(assign_band)
    return df


def add_target_anchored_lag(target_df, source_df, value_col, lag_days, feature_name):
    """
    Seasonal lag join anchored to target_timestamp, not issue_timestamp:
    "the actual price at this same time, `lag_days` before the thing being
    predicted" -- NOT "before today".

    This is the fix for a real problem found by inspecting live predictions:
    issue-anchored lag features (the previous design) are constant across an
    entire live prediction batch, because every row you predict today shares
    the same issue_timestamp ("now"). During training those features varied
    row to row (issue_timestamp = target - horizon, which differs per row),
    so the model leaned on them heavily (up to 69% of gain for Agile) --
    then at serve time that same feature contributes zero differentiation
    between one target day and the next, which is exactly why live output
    collapsed into a near-identical repeating daily template.

    Target-anchoring fixes this: target_timestamp varies per row both in
    training AND at serve time, so the feature stays informative in both
    places.

    Leakage safety: only valid when lag_days >= horizon_days, i.e. the
    reference point (target - lag_days) falls at or before issue_timestamp
    (target - horizon_days). Rows where that's not true get null here --
    handled by pick_safe_seasonal_lag() below, which always finds a safe
    option from {7, 14, 28} for any horizon in [2, 14], so no row is ever
    left without a valid seasonal feature.
    """
    lagged = source_df[["timestamp", value_col]].rename(
        columns={"timestamp": "target_timestamp", value_col: feature_name})
    merged = target_df.merge(lagged, on="target_timestamp", how="left")
    unsafe = target_df["horizon_days"] > lag_days
    merged.loc[unsafe.values, feature_name] = pd.NA
    return merged


def build(since=None):
    """Build the feature matrix. If `since` is given (a tz-aware timestamp),
    restrict to target dates on/after it -- used by predict.py to build a
    small recent-only matrix for rolling conformal recalibration, sharing
    every join/feature-engineering step with the full training build so the
    two can never silently drift apart into different feature definitions.
    """
    print("Fetching source datasets...")
    forecast_archive = get_2_14_day_forecast_archive()
    embedded = get_embedded_wind_solar_forecast()
    agile = fetch_full_tariff_history("agile")
    freephase = fetch_full_tariff_history("freephase")

    # Fetch enough outturn/renewables history to cover the OLDEST tariff data
    # we have, not an arbitrary fixed window. Getting this wrong silently
    # drops however many months of tariff history fall outside the window --
    # demand_7d_mean/renewable_7d_mean come back null for anything older,
    # and the complete-case filter in train_model.py then removes those rows
    # entirely rather than erroring, which is exactly why this is worth
    # computing from the actual data rather than guessing a round number.
    earliest_tariff_dates = [df["timestamp"].min() for df in (agile, freephase) if not df.empty]
    if earliest_tariff_dates:
        earliest = min(earliest_tariff_dates)
        months_needed = int(((pd.Timestamp.now(tz="UTC") - earliest).days / 30) + 2)  # +2 months buffer
        print(f"Oldest tariff data: {earliest.date()} -> requesting {months_needed} months of outturn/renewables history")
    else:
        months_needed = 13
    outturn = get_historic_outturn_demand(months_back=months_needed)

    if forecast_archive.empty or "target_date" not in forecast_archive.columns:
        print("! 2-14 day forecast archive did not resolve to usable columns -- "
              "check the printed schema from fetch_neso.py and fix the column "
              "candidates in get_2_14_day_forecast_archive() before proceeding.")
        return pd.DataFrame()

    # Cardinal Point forecasts are a handful of reference points per day, not
    # half-hourly. Aggregate to (target_date, horizon_days) before joining --
    # mean/min/max across the day's cardinal points becomes a daily "expected
    # demand level" feature; hour/day_of_week/band (added below) carry the
    # intraday shape the model needs on top of that.
    daily_forecast = (
        forecast_archive.groupby(["target_date", "horizon_days"])["forecast_demand_mw"]
        .agg(forecast_demand_mean="mean", forecast_demand_min="min", forecast_demand_max="max")
        .reset_index()
    )
    # Round horizon to whole days so every half-hourly slot on a given
    # calendar date joins against the same handful of horizon buckets NESO
    # actually publishes (2-14), rather than fragmenting on fractional hours.
    daily_forecast["horizon_days"] = daily_forecast["horizon_days"].round().astype(int)
    daily_forecast = daily_forecast[(daily_forecast["horizon_days"] >= 2) & (daily_forecast["horizon_days"] <= 14)]

    # Base grid: every actual half-hourly rate slot we have, crossed with
    # each horizon (2-14) it could plausibly have been predicted at -- this
    # is what lets one model learn "how does accuracy change with horizon".
    targets = []
    if not agile.empty:
        targets.append(agile)
    if not freephase.empty:
        targets.append(freephase)
    if not targets:
        print("! no tariff history available")
        return pd.DataFrame()

    base = targets[0][["timestamp"]].copy()
    for t in targets[1:]:
        base = pd.merge(base, t[["timestamp"]], on="timestamp", how="outer")
    base = base.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    if since is not None:
        base = base[base["timestamp"] >= since]
    base["target_date"] = base["timestamp"].dt.normalize()

    df = base.merge(pd.DataFrame({"horizon_days": range(2, 15)}), how="cross")
    df = df.rename(columns={"timestamp": "target_timestamp"})
    df = df.merge(daily_forecast, on=["target_date", "horizon_days"], how="inner")
    df["issue_timestamp"] = df["target_timestamp"] - pd.to_timedelta(df["horizon_days"], unit="D")

    df = add_calendar_features(df, "target_timestamp")

    # Targets: actual rate at target_timestamp
    if not agile.empty:
        df = df.merge(agile.rename(columns={"timestamp": "target_timestamp"}),
                       on="target_timestamp", how="left")
    if not freephase.empty:
        df = df.merge(freephase.rename(columns={"timestamp": "target_timestamp"}),
                       on="target_timestamp", how="left")

    # Seasonal lag features, target-anchored (see add_target_anchored_lag
    # docstring for why this replaced the old issue-anchored version).
    # price_anchor_Nd: N in {7, 14} is the FRESHEST safe seasonal reference
    # for this row's horizon (7 for horizon<=7, 14 otherwise) -- always
    # populated, never null, for every horizon in [2, 14].
    # rate_lag28d: always safe regardless of horizon, kept as a second,
    # more stable long-baseline reference alongside the adaptive one.
    for lag in (7, 14, 28):
        if not agile.empty:
            df = add_target_anchored_lag(df, agile, "agile_rate_p_kwh", lag, f"agile_rate_lag{lag}d")
        if not freephase.empty:
            df = add_target_anchored_lag(df, freephase, "freephase_rate_p_kwh", lag, f"freephase_rate_lag{lag}d")

    for product_key in ("agile", "freephase"):
        lag7_col, lag14_col = f"{product_key}_rate_lag7d", f"{product_key}_rate_lag14d"
        if lag7_col in df.columns and lag14_col in df.columns:
            # freshest safe: lag7 where horizon<=7 (non-null there by construction),
            # else fall back to lag14 (always safe for horizon<=14)
            df[f"{product_key}_price_anchor"] = df[lag7_col].where(df[lag7_col].notna(), df[lag14_col])

    # Recent actual demand + renewables trend as of issue time (backward-
    # looking, safe). renewable_7d_mean is the feature you were missing:
    # trailing wind+solar utilisation, which both Agile and FreePhase should
    # respond to inversely (more renewables -> lower wholesale price ->
    # lower rate). This is deliberately a *recent trend*, not a forecast for
    # the target date -- the forward-looking renewable forecast archive only
    # spans ~13-14 days historically (see get_embedded_wind_solar_forecast's
    # docstring), nowhere near enough to cover 13 months of training targets,
    # so using it here would mean training on a feature that's mostly null
    # and get dropped by the complete-case filter, or -- worse -- available
    # for training but not symmetric with what's usable in predict.py.
    if not outturn.empty:
        outturn_sorted = outturn.sort_values("timestamp")
        outturn_sorted["demand_7d_mean"] = outturn_sorted["actual_demand_mw"].rolling(336, min_periods=48).mean()
        recent_cols = ["timestamp", "demand_7d_mean"]
        if "renewable_utilisation_pct" in outturn_sorted.columns:
            outturn_sorted["renewable_7d_mean"] = outturn_sorted["renewable_utilisation_pct"].rolling(336, min_periods=48).mean()
            recent_cols.append("renewable_7d_mean")
        else:
            print("! renewable_utilisation_pct not present in outturn data -- "
                  "renewable_7d_mean will be missing from the feature set entirely")
        recent = outturn_sorted[recent_cols].rename(columns={"timestamp": "issue_timestamp"})
        df = pd.merge_asof(
            df.sort_values("issue_timestamp"), recent.sort_values("issue_timestamp"),
            on="issue_timestamp", direction="backward"
        )

    df = df.sort_values("target_timestamp").reset_index(drop=True)
    print(f"Training matrix: {len(df)} rows, columns: {list(df.columns)}")
    return df


if __name__ == "__main__":
    matrix = build()
    if not matrix.empty:
        matrix.to_csv("training_matrix.csv", index=False)
        print("Saved training_matrix.csv")
