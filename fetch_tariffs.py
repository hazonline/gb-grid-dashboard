"""
Historic tariff rate fetcher for Octopus Agile and EDF FreePhase Dynamic.

This mirrors the URL/slug pattern already working in your live
fetch_and_predict.py on gb-grid-dashboard (api.octopus.energy for Agile,
api.edfgb-kraken.energy for FreePhase, slug = E-1R-{product}-{dno_letter}) --
that part of your original script was correct, so it's kept as-is here,
just generalised to page back through full history rather than one page.
"""

import time
import requests
import pandas as pd

OCTOPUS_AGILE_PRODUCT = "AGILE-24-04-03"
EDF_FREEPHASE_PRODUCT = "EDF_FREEPHASE_DYNAMIC_12M_HH"
DEFAULT_REGION = "J"  # South East England -- the region the ML model is trained on

# Full 14 DNO region codes (source: Ofgem / MPAN distributor IDs, cross-checked
# Aug 2026). ML predictions are only valid for DEFAULT_REGION; every region
# below works for fetching real published rates (no ML involved there).
REGIONS = {
    "A": "Eastern England", "B": "East Midlands", "C": "London",
    "D": "Merseyside & North Wales", "E": "West Midlands", "F": "North East England",
    "G": "North West England", "H": "Southern England", "J": "South East England",
    "K": "South Wales", "L": "South West England", "M": "Yorkshire",
    "N": "Southern Scotland", "P": "Northern Scotland",
}

TARIFF_SOURCES = {
    "agile": {
        "base_url": "https://api.octopus.energy/v1",
        "product": OCTOPUS_AGILE_PRODUCT,
    },
    "freephase": {
        "base_url": "https://api.edfgb-kraken.energy/v1",
        "product": EDF_FREEPHASE_PRODUCT,
    },
}


def _parse_rate_results(results, product_key):
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame([
        {
            "timestamp": item["valid_from"],
            f"{product_key}_rate_p_kwh": round(item["value_inc_vat"], 4),
        }
        for item in results
    ])
    from fetch_neso import to_utc
    df["timestamp"] = to_utc(df["timestamp"])
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def fetch_full_tariff_history(product_key, region=DEFAULT_REGION, page_size=1500, max_pages=500, verbose=True):
    """Paginate through a product's standard-unit-rates history to completion."""
    cfg = TARIFF_SOURCES[product_key]
    product = cfg["product"]
    slug = f"E-1R-{product}-{region}"
    url = (
        f"{cfg['base_url']}/products/{product}/electricity-tariffs/"
        f"{slug}/standard-unit-rates/?page_size={page_size}"
    )

    all_results = []
    pages = 0
    while url and pages < max_pages:
        try:
            resp = requests.get(url, timeout=20)
        except requests.RequestException as e:
            print(f"  ! network error fetching {product_key}/{region}: {e}")
            break
        if resp.status_code != 200:
            if pages == 0:
                print(f"  ! {product_key}/{region} returned HTTP {resp.status_code} -- "
                      "product may not be offered in this region")
            break
        data = resp.json()
        all_results.extend(data.get("results", []))
        url = data.get("next")
        pages += 1
        time.sleep(0.05)

    df = _parse_rate_results(all_results, product_key)
    if verbose and not df.empty:
        print(f"-> {product_key}/{region}: {len(df)} half-hourly rate slots, "
              f"{df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


def fetch_recent_tariff_history(product_key, region=DEFAULT_REGION, days_back=7, days_forward=2, verbose=True):
    """
    Lightweight date-filtered fetch for the site's 'current rates' panel --
    NOT the full pagination sweep. Used across up to 14 regions x 2 products
    on every site refresh, so this matters: a full history pull per
    region/product would be slow and mostly wasted (site only needs recent +
    any already-published near-future slots).
    """
    cfg = TARIFF_SOURCES[product_key]
    product = cfg["product"]
    slug = f"E-1R-{product}-{region}"
    now = pd.Timestamp.now(tz="UTC")
    period_from = (now - pd.Timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
    period_to = (now + pd.Timedelta(days=days_forward)).strftime("%Y-%m-%dT%H:%M:%SZ")
    url = (
        f"{cfg['base_url']}/products/{product}/electricity-tariffs/{slug}/standard-unit-rates/"
        f"?period_from={period_from}&period_to={period_to}&page_size=1500"
    )

    all_results = []
    pages = 0
    while url and pages < 10:  # small window, shouldn't ever need many pages
        try:
            resp = requests.get(url, timeout=20)
        except requests.RequestException as e:
            print(f"  ! network error fetching {product_key}/{region} (recent): {e}")
            break
        if resp.status_code != 200:
            break
        data = resp.json()
        all_results.extend(data.get("results", []))
        url = data.get("next")
        pages += 1

    df = _parse_rate_results(all_results, product_key)
    if verbose and not df.empty:
        print(f"  {product_key}/{region}: {len(df)} recent slots "
              f"({df['timestamp'].min()} to {df['timestamp'].max()})")
    return df


if __name__ == "__main__":
    agile = fetch_full_tariff_history("agile")
    freephase = fetch_full_tariff_history("freephase")
    print(agile.tail())
    print(freephase.tail())

