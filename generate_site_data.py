"""
Builds every JSON file the static site reads. Run by the twice-daily
refresh workflow. Split into multiple files deliberately, not one blob:

  data/predictions.json   -- small. Region J official recent rates +
                              14-day predictions. This is what the main
                              chart loads on page load, so it needs to be
                              light.
  data/regions.json       -- manifest: which regions exist, which one has
                              deep history vs recent-only, so the site
                              knows what to offer before fetching anything.
  data/history_{R}.json   -- one per region. J gets full ~30-month history
                              (it's what the model trains on anyway, and
                              Harry will want to browse it deeply). Every
                              other region gets a recent window only --
                              fetched fresh each refresh via the lightweight
                              date-filtered endpoint, not a full pagination
                              sweep across 13 extra regions on every run.
                              Loaded on-demand by the site, only when a
                              region is actually selected in the history
                              browser -- not bundled into the initial payload.

All timestamps are epoch seconds (UTC), not ISO strings -- meaningfully
smaller over ~60,000+ rows, and trivial to reconstruct client-side with
`new Date(epoch * 1000)`.

Freshness polling: GitHub Actions scheduled workflows aren't guaranteed to
fire at the exact cron minute, and NESO/Octopus don't publish at the exact
second either. Rather than trust a fixed clock time, wait_for_fresh_forecast()
polls the live NESO forecast until it genuinely looks like today's update
(max target date reaches ~14 days out from now) or a time budget runs out --
then proceeds with whatever's available either way, so a slightly-late
publish delays the site by a few minutes instead of leaving it broken.
"""
import argparse
import json
import os
import time
import pandas as pd

from fetch_tariffs import fetch_full_tariff_history, fetch_recent_tariff_history, REGIONS, DEFAULT_REGION
import predict

OUTPUT_DIR = "site/data"
OTHER_REGION_HISTORY_DAYS = 90  # non-J regions: how far back "go back in time" can browse


def to_epoch_pairs(df, value_col):
    if df.empty or value_col not in df.columns:
        return []
    ts = (df["timestamp"].astype("int64") // 10**9).tolist()
    vals = df[value_col].round(3).tolist()
    return list(zip(ts, vals))


def wait_for_fresh_forecast(max_wait_minutes=40, poll_interval_minutes=5):
    """
    Poll NESO's live 14-day forecast until it looks genuinely fresh, rather
    than trusting the workflow's trigger time to line up with NESO's actual
    publish time. Heuristic: a freshly-published forecast should reach out
    to ~14 days from today; a stale (yesterday's) one will fall a day short.
    Not perfect, but self-contained -- doesn't need to persist state between
    runs to know "has this actually updated since last time".
    """
    deadline = time.time() + max_wait_minutes * 60
    attempt = 0
    while True:
        attempt += 1
        live = predict.fetch_live_forecast()
        if not live.empty and "target_timestamp" in live.columns:
            max_horizon_days = (live["target_timestamp"].max() - pd.Timestamp.now(tz="UTC")).total_seconds() / 86400
            if max_horizon_days >= 13:
                print(f"Forecast looks fresh on attempt {attempt} (reaches {max_horizon_days:.1f} days out)")
                return True
            print(f"Attempt {attempt}: forecast only reaches {max_horizon_days:.1f} days out, "
                  f"expected ~14 -- likely not yet updated today")
        else:
            print(f"Attempt {attempt}: forecast fetch returned nothing")

        if time.time() >= deadline:
            print(f"Gave up waiting for a fresh forecast after {max_wait_minutes} minutes -- "
                  "proceeding with whatever's currently available")
            return False
        time.sleep(poll_interval_minutes * 60)


def build_predictions_payload():
    print("Building predictions.json (region J)...")
    frame = predict.build_inference_frame()
    for product_key in ("agile", "freephase"):
        frame = predict.predict_product(frame, product_key)

    predictions = {}
    for product_key in ("agile", "freephase"):
        cols = ["target_timestamp", "horizon_days", "freephase_band",
                f"{product_key}_p10", f"{product_key}_p50", f"{product_key}_p90"]
        cols = [c for c in cols if c in frame.columns]
        sub = frame[cols].sort_values("target_timestamp")
        predictions[product_key] = [
            {
                "t": int(row["target_timestamp"].timestamp()),
                "h": int(row["horizon_days"]),
                "band": row.get("freephase_band"),
                "p10": round(row[f"{product_key}_p10"], 3) if f"{product_key}_p10" in row else None,
                "p50": round(row[f"{product_key}_p50"], 3) if f"{product_key}_p50" in row else None,
                "p90": round(row[f"{product_key}_p90"], 3) if f"{product_key}_p90" in row else None,
            }
            for _, row in sub.iterrows()
        ]

    print("Fetching region J recent official rates for continuity with the chart...")
    recent_official = {}
    for product_key in ("agile", "freephase"):
        df = fetch_recent_tariff_history(product_key, region=DEFAULT_REGION, days_back=14, days_forward=2)
        recent_official[product_key] = to_epoch_pairs(df, f"{product_key}_rate_p_kwh")

    return {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "region": DEFAULT_REGION,
        "official_recent": recent_official,
        "predictions": predictions,
    }


def build_region_history(region, is_primary):
    print(f"Fetching history for region {region} ({REGIONS.get(region, '?')})"
          f"{' [full]' if is_primary else ' [recent]'}...")
    payload = {"region": region, "name": REGIONS.get(region, region)}
    for product_key in ("agile", "freephase"):
        if is_primary:
            df = fetch_full_tariff_history(product_key, region=region)
        else:
            df = fetch_recent_tariff_history(product_key, region=region,
                                              days_back=OTHER_REGION_HISTORY_DAYS, days_forward=2)
        payload[product_key] = to_epoch_pairs(df, f"{product_key}_rate_p_kwh")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-fresh-forecast", action="store_true",
                         help="Poll NESO's live forecast until it looks freshly published, "
                              "before generating the site (use for the morning refresh run).")
    parser.add_argument("--max-wait-minutes", type=int, default=40)
    parser.add_argument("--poll-interval-minutes", type=int, default=5)
    args = parser.parse_args()

    if args.wait_for_fresh_forecast:
        wait_for_fresh_forecast(args.max_wait_minutes, args.poll_interval_minutes)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    predictions_payload = build_predictions_payload()
    with open(f"{OUTPUT_DIR}/predictions.json", "w") as f:
        json.dump(predictions_payload, f, separators=(",", ":"))
    print(f"Wrote {OUTPUT_DIR}/predictions.json")

    regions_manifest = {
        "regions": REGIONS,
        "primary_region": DEFAULT_REGION,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    with open(f"{OUTPUT_DIR}/regions.json", "w") as f:
        json.dump(regions_manifest, f, separators=(",", ":"))
    print(f"Wrote {OUTPUT_DIR}/regions.json")

    for region in REGIONS:
        history = build_region_history(region, is_primary=(region == DEFAULT_REGION))
        with open(f"{OUTPUT_DIR}/history_{region}.json", "w") as f:
            json.dump(history, f, separators=(",", ":"))
        print(f"Wrote {OUTPUT_DIR}/history_{region}.json "
              f"({len(history.get('agile', []))} agile, {len(history.get('freephase', []))} freephase rows)")

    print("\nSite data generation complete.")


if __name__ == "__main__":
    main()
