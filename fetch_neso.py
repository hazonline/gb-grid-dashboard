"""
NESO (National Energy System Operator) data fetch layer.

IMPORTANT — corrected from the original script:
The resource_id '4dd712a2-ee2c-455d-a9c0-9d3564c80fa0' is NOT historic outturn
demand. Verified against neso.energy/data-portal directly (Aug 2026): it is
'archive_14dayahead.csv', the historic archive of NESO's own 2-14 day AHEAD
FORECASTS. Using it as ground-truth demand would bake NESO's own forecast
error into the training set silently.

This module fetches three distinct, correctly-labelled things:
  1. get_historic_outturn_demand()  -> what demand ACTUALLY was (ND, TSD)
  2. get_2_14_day_forecast_archive() -> what NESO FORECAST demand would be,
     made at a known horizon before the target settlement period. This is
     legitimate to use as a *feature* for horizon-14 prediction, because at
     inference time it's genuinely all you'd have -- but it must never be
     merged in and relabelled as if it were actual demand.
  3. get_embedded_wind_solar_forecast() -> NESO's 14-day-ahead embedded wind/
     solar forecast (daily resolution), another legitimate forward-looking
     feature.

NESO's CKAN "Historic Demand Data" dataset is split into one resource per
calendar year (there is no single resource_id covering a rolling 12 months),
so get_historic_outturn_demand() discovers the right resources dynamically
via `package_show` rather than hardcoding an ID that will silently go stale
every January.

Schema note: I have not been able to verify the exact column headers of
archive_14dayahead.csv from documentation alone (NESO's docs don't publish a
data dictionary for this file). fetch_dataset_records() prints the discovered
columns on first use -- check that output against what the code assumes
before trusting the merge in build_training_matrix.py.
"""

import time
import requests
import pandas as pd
from datetime import datetime, timezone

NESO_API_BASE = "https://api.neso.energy/api/3/action"

# Dataset slugs (stable) -- resource IDs under them are NOT stable, so we
# resolve resources dynamically via package_show instead of hardcoding IDs.
DATASET_SLUGS = {
    "historic_demand": "historic-demand-data",
    "demand_forecast_14da": "2-14-days-ahead-national-demand-forecast",
    "embedded_wind_solar_forecast": "embedded-wind-and-solar-forecasts",
}

# The one resource_id that IS safe to hardcode: this dataset is described by
# NESO as "from the first day of the previous month up to the current day" --
# i.e. a rolling window, refreshed daily. Good for a lightweight top-up fetch,
# not a substitute for full 12-month history.
DEMAND_DATA_UPDATE_RESOURCE = "177f6fa4-ae49-4182-81ea-0c6b35f26ca6"


def list_dataset_resources(dataset_slug):
    """Return [{id, name, url, format, datastore_active}, ...] for a dataset."""
    url = f"{NESO_API_BASE}/package_show"
    resp = requests.get(url, params={"id": dataset_slug}, timeout=30)
    resp.raise_for_status()
    result = resp.json()["result"]
    return [
        {
            "id": r["id"],
            "name": r.get("name", ""),
            "url": r.get("url", ""),
            "format": r.get("format", ""),
            "datastore_active": r.get("datastore_active", False),
        }
        for r in result.get("resources", [])
    ]


def fetch_dataset_records(resource_id, page_size=5000, max_pages=200, verbose=True):
    """
    Fetch every record for a resource_id via datastore_search, paginating on
    offset. Falls back to downloading the raw CSV if the datastore isn't
    active for this resource (NESO doesn't enable it uniformly).
    """
    records = []
    offset = 0
    for _ in range(max_pages):
        resp = requests.get(
            f"{NESO_API_BASE}/datastore_search",
            params={"resource_id": resource_id, "limit": page_size, "offset": offset},
            timeout=30,
        )
        if resp.status_code != 200:
            break
        payload = resp.json()
        if not payload.get("success"):
            break
        page = payload["result"]["records"]
        if not page:
            break
        records.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
        time.sleep(0.1)  # be polite to the API

    if records:
        df = pd.DataFrame(records)
        if verbose:
            print(f"  -> {resource_id}: {len(df)} rows via datastore. Columns: {list(df.columns)}")
        return df

    # Datastore empty/inactive -> fall back to raw CSV download
    if verbose:
        print(f"  -> {resource_id}: datastore returned nothing, trying CSV fallback...")
    resources = None
    for slug in DATASET_SLUGS.values():
        try:
            resources = list_dataset_resources(slug)
        except Exception:
            continue
        match = next((r for r in resources if r["id"] == resource_id), None)
        if match and match["url"]:
            df = pd.read_csv(match["url"])
            if verbose:
                print(f"  -> {resource_id}: {len(df)} rows via CSV fallback. Columns: {list(df.columns)}")
            return df
    print(f"  -> {resource_id}: could not fetch via datastore or CSV fallback.")
    return pd.DataFrame()


def to_utc(series):
    """Parse a column to tz-aware UTC regardless of whether the source gave
    tz-naive or tz-aware timestamps. NESO's SETTLEMENT_DATE columns come back
    tz-naive; Octopus/EDF 'valid_from' fields come back tz-aware (ISO8601 'Z').
    Mixing the two anywhere -- comparisons, merges, or subtraction -- either
    raises or (worse, for a merge) silently matches nothing. Route every
    timestamp column through this before it touches another one.
    """
    dt = pd.to_datetime(series)
    if dt.dt.tz is None:
        return dt.dt.tz_localize("UTC")
    return dt.dt.tz_convert("UTC")


def _first_matching_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def get_recent_actual_demand_and_renewables(verbose=True):
    """
    'Demand Data Update' -- NESO's rolling window from the first of the
    previous month to today, refreshed daily. Same column layout as the
    per-year historic demand resources (confirmed by inspection), so it can
    reuse the same parsing logic rather than needing its own.

    This exists specifically so predict.py can compute "actual demand/
    renewables trend as of right now" without waiting for a new calendar-
    year resource to appear -- the per-year resources won't have today's
    data in them yet if today is early in a new year, and re-fetching a
    whole year of history just to get the last 7 days is wasteful.
    """
    df = fetch_dataset_records(DEMAND_DATA_UPDATE_RESOURCE, verbose=verbose)
    if df.empty:
        return df
    return _parse_outturn_frame(df, verbose=verbose)


def _parse_outturn_frame(df, months_back=None, verbose=True):
    ts_col = _first_matching_column(df, ["SETTLEMENT_DATE", "GDATETIME", "settlement_date"])
    period_col = _first_matching_column(df, ["SETTLEMENT_PERIOD", "settlement_period"])
    demand_col = _first_matching_column(df, ["ND", "TSD", "nd", "tsd"])
    wind_gen_col = _first_matching_column(df, ["EMBEDDED_WIND_GENERATION"])
    wind_cap_col = _first_matching_column(df, ["EMBEDDED_WIND_CAPACITY"])
    solar_gen_col = _first_matching_column(df, ["EMBEDDED_SOLAR_GENERATION"])
    solar_cap_col = _first_matching_column(df, ["EMBEDDED_SOLAR_CAPACITY"])

    if ts_col is None or demand_col is None:
        print("  ! Could not auto-detect timestamp/demand columns. "
              f"Available columns: {list(df.columns)}. Fix column names manually.")
        return pd.DataFrame()

    out = pd.DataFrame()
    if period_col is not None:
        base_date = to_utc(df[ts_col])
        period = pd.to_numeric(df[period_col], errors="coerce")
        out["timestamp"] = base_date + pd.to_timedelta((period - 1) * 30, unit="m")
    else:
        out["timestamp"] = to_utc(df[ts_col])

    out["actual_demand_mw"] = pd.to_numeric(df[demand_col], errors="coerce")

    if wind_gen_col and solar_gen_col:
        wind_gen = pd.to_numeric(df[wind_gen_col], errors="coerce")
        solar_gen = pd.to_numeric(df[solar_gen_col], errors="coerce")
        out["renewable_generation_mw"] = wind_gen.fillna(0) + solar_gen.fillna(0)
        if wind_cap_col and solar_cap_col:
            wind_cap = pd.to_numeric(df[wind_cap_col], errors="coerce")
            solar_cap = pd.to_numeric(df[solar_cap_col], errors="coerce")
            total_cap = wind_cap.fillna(0) + solar_cap.fillna(0)
            out["renewable_utilisation_pct"] = (out["renewable_generation_mw"] / total_cap.replace(0, pd.NA)) * 100

    if months_back is not None:
        now = datetime.now(timezone.utc)
        cutoff = now - pd.DateOffset(months=months_back)
        out = out[out["timestamp"] >= cutoff]

    out = out.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    if verbose:
        print(f"-> Outturn demand/renewables: {len(out)} rows, "
              f"{out['timestamp'].min()} to {out['timestamp'].max()}")
    return out


def get_historic_outturn_demand(months_back=13, verbose=True):
    """
    Actual historic demand (ND/TSD) plus actual embedded wind/solar
    generation, concatenated across however many calendar-year resources are
    needed to cover `months_back` months, discovered dynamically so this
    doesn't rot every New Year.
    """
    now = datetime.now(timezone.utc)
    cutoff_year = (now.replace(day=1) - pd.DateOffset(months=months_back)).year
    years_needed = list(range(cutoff_year, now.year + 1))

    if verbose:
        print(f"Discovering historic demand resources for years: {years_needed}")
    resources = list_dataset_resources(DATASET_SLUGS["historic_demand"])

    frames = []
    for year in years_needed:
        match = next((r for r in resources if str(year) in r["name"] or str(year) in r["url"]), None)
        if not match:
            print(f"  ! no historic demand resource found for {year}, skipping")
            continue
        df = fetch_dataset_records(match["id"], verbose=verbose)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    return _parse_outturn_frame(df, months_back=months_back, verbose=verbose)


def get_2_14_day_forecast_archive(verbose=True):
    """
    NESO's own historic 2-14 day ahead demand FORECASTS.

    Confirmed schema (from a real run against the live API):
    ['_id', 'DAYSAHEAD', 'TARGETDATE', 'FORECASTDEMAND', 'CARDINALPOINT',
     'CP_TYPE', 'CP_ST_TIME', 'CP_END_TIME', 'F_Point', 'FORECAST_TIMESTAMP']

    This is NOT half-hourly. CARDINALPOINT/CP_TYPE/CP_ST_TIME/CP_END_TIME
    mean each row is a forecast for one of a handful of reference points in
    a day (morning peak, evening peak, overnight minimum, etc.) -- NESO
    doesn't appear to retain a historical half-hourly forecast archive, only
    this coarser Cardinal Point one (the half-hourly file is a live-only
    snapshot). DAYSAHEAD gives the forecast horizon directly, which is more
    reliable than deriving it from timestamp subtraction.

    Returned as one row per (target date, horizon, cardinal point) --
    aggregate to daily level in build_training_matrix.py before using it as
    a feature for half-hourly targets; don't try to join it 1:1 onto every
    half-hour slot, most slots simply have no matching row.
    """
    resource_id = "4dd712a2-ee2c-455d-a9c0-9d3564c80fa0"  # archive_14dayahead.csv
    df = fetch_dataset_records(resource_id, verbose=verbose)
    if df.empty:
        return df

    target_col = _first_matching_column(df, ["TARGETDATE", "SETTLEMENT_DATE", "FORECASTDATE"])
    horizon_col = _first_matching_column(df, ["DAYSAHEAD"])
    issue_col = _first_matching_column(df, ["FORECAST_TIMESTAMP", "PUBLISHTIME", "FORECAST_ISSUE_TIME", "PUBLISH_DATE"])
    value_col = _first_matching_column(df, ["FORECASTDEMAND", "DEMAND", "ND_FORECAST"])

    missing = [n for n, c in [("target", target_col), ("horizon", horizon_col),
                               ("issue", issue_col), ("value", value_col)] if c is None]
    if missing:
        print(f"  ! Could not auto-map columns for {missing} in 2-14da forecast archive. "
              f"Available: {list(df.columns)}")
        return df

    out = df.rename(columns={
        target_col: "target_date",
        horizon_col: "horizon_days",
        issue_col: "issue_timestamp",
        value_col: "forecast_demand_mw",
    })
    out["target_date"] = to_utc(out["target_date"]).dt.normalize()
    out["issue_timestamp"] = to_utc(out["issue_timestamp"])
    out["horizon_days"] = pd.to_numeric(out["horizon_days"], errors="coerce")
    out["forecast_demand_mw"] = pd.to_numeric(out["forecast_demand_mw"], errors="coerce")

    if verbose:
        print(f"-> 2-14da forecast archive: {len(out)} cardinal-point rows, "
              f"horizons {out['horizon_days'].min():.0f}-{out['horizon_days'].max():.0f} days")
    return out


def get_embedded_wind_solar_forecast(verbose=True):
    """
    NESO's embedded wind & solar generation FORECAST (as distinct from the
    actual generation now folded into get_historic_outturn_demand()).

    Confirmed schema (from a real run): ['_id', 'DATE_GMT', 'TIME_GMT',
    'SETTLEMENT_DATE', 'SETTLEMENT_PERIOD', 'EMBEDDED_WIND_FORECAST',
    'EMBEDDED_WIND_CAPACITY', 'EMBEDDED_SOLAR_FORECAST', 'EMBEDDED_SOLAR_CAPACITY'].

    Caution before wiring this into training: a real run returned only ~652
    rows -- about 13-14 days at half-hourly resolution. That's consistent
    with this being a live rolling snapshot (like DEMAND_DATA_UPDATE_RESOURCE)
    rather than a deep historical archive. If that's confirmed, it can only
    be used in predict.py at inference time, not in build_training_matrix.py,
    because it simply doesn't go back far enough to cover historical target
    dates. Using it in training but not at inference (or vice versa) would
    give the model a feature column it never saw filled at one end -- worse
    than not having the feature at all. This function prints the actual date
    range on every call so that assumption can be checked rather than trusted.
    """
    resource_id = "db6c038f-98af-4570-ab60-24d71ebd0ae5"
    df = fetch_dataset_records(resource_id, verbose=verbose)
    if df.empty:
        return df

    ts_col = _first_matching_column(df, ["SETTLEMENT_DATE"])
    period_col = _first_matching_column(df, ["SETTLEMENT_PERIOD"])
    wind_col = _first_matching_column(df, ["EMBEDDED_WIND_FORECAST"])
    wind_cap_col = _first_matching_column(df, ["EMBEDDED_WIND_CAPACITY"])
    solar_col = _first_matching_column(df, ["EMBEDDED_SOLAR_FORECAST"])
    solar_cap_col = _first_matching_column(df, ["EMBEDDED_SOLAR_CAPACITY"])

    if ts_col is None or wind_col is None or solar_col is None:
        print(f"  ! Could not auto-map embedded forecast columns. Available: {list(df.columns)}")
        return df

    out = pd.DataFrame()
    if period_col is not None:
        base_date = to_utc(df[ts_col])
        period = pd.to_numeric(df[period_col], errors="coerce")
        out["target_timestamp"] = base_date + pd.to_timedelta((period - 1) * 30, unit="m")
    else:
        out["target_timestamp"] = to_utc(df[ts_col])

    wind = pd.to_numeric(df[wind_col], errors="coerce")
    solar = pd.to_numeric(df[solar_col], errors="coerce")
    out["renewable_forecast_mw"] = wind.fillna(0) + solar.fillna(0)
    if wind_cap_col and solar_cap_col:
        cap = pd.to_numeric(df[wind_cap_col], errors="coerce").fillna(0) + \
              pd.to_numeric(df[solar_cap_col], errors="coerce").fillna(0)
        out["renewable_forecast_utilisation_pct"] = (out["renewable_forecast_mw"] / cap.replace(0, pd.NA)) * 100

    out = out.drop_duplicates("target_timestamp").sort_values("target_timestamp").reset_index(drop=True)
    span_days = (out["target_timestamp"].max() - out["target_timestamp"].min()).total_seconds() / 86400 if len(out) else 0
    if verbose:
        print(f"-> Embedded wind/solar forecast: {len(out)} rows, "
              f"{out['target_timestamp'].min()} to {out['target_timestamp'].max()} "
              f"({span_days:.1f} day span)")
        if span_days < 20:
            print("  ! span looks like a live rolling snapshot, not deep history -- "
                  "see docstring before using this in build_training_matrix.py")
    return out


if __name__ == "__main__":
    demand = get_historic_outturn_demand()
    print(demand.head())
    forecast = get_2_14_day_forecast_archive()
    print(forecast.head())
