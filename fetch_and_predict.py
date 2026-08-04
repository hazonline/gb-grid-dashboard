import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta

# All 14 UK Distribution Network Operator (DNO) Regions
DNO_REGIONS = {
    "_A": {"name": "A - Eastern England", "kraken_code": "-A"},
    "_B": {"name": "B - East Midlands", "kraken_code": "-B"},
    "_C": {"name": "C - London", "kraken_code": "-C"},
    "_D": {"name": "D - Merseyside & N. Wales", "kraken_code": "-D"},
    "_E": {"name": "E - West Midlands", "kraken_code": "-E"},
    "_F": {"name": "F - North East England", "kraken_code": "-F"},
    "_G": {"name": "G - North West England", "kraken_code": "-G"},
    "_H": {"name": "H - Southern England", "kraken_code": "-H"},
    "_J": {"name": "J - South East England", "kraken_code": "-J"},
    "_K": {"name": "K - South Wales", "kraken_code": "-K"},
    "_L": {"name": "L - South West England", "kraken_code": "-L"},
    "_M": {"name": "M - Yorkshire", "kraken_code": "-M"},
    "_N": {"name": "N - Southern Scotland", "kraken_code": "-N"},
    "_P": {"name": "P - Northern Scotland", "kraken_code": "-P"}
}

OCTOPUS_AGILE_PRODUCT = "AGILE-24-04-03"
EDF_FREEPHASE_PRODUCT = "EDF_FREEPHASE_DYNAMIC_12M_HH"

def fetch_octopus_region(dict_key, kraken_code):
    """Fetch Octopus Agile rates for a single DNO region."""
    url = (
        f"https://api.octopus.energy/v1/products/{OCTOPUS_AGILE_PRODUCT}/"
        f"electricity-tariffs/E-1R-{OCTOPUS_AGILE_PRODUCT}{kraken_code}/standard-unit-rates/?page_size=96"
    )
    rates = {}
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            for item in results:
                ts = item["valid_from"].replace(".000Z", "Z")
                rates[ts] = round(item["value_inc_vat"], 2)
    except Exception as e:
        print(f"Octopus API error ({kraken_code}): {e}")
    return dict_key, rates

def fetch_edf_region(dict_key, kraken_code):
    """Fetch EDF FreePhase rates for a single DNO region."""
    url = (
        f"https://api.edfgb-kraken.energy/v1/products/{EDF_FREEPHASE_PRODUCT}/"
        f"electricity-tariffs/E-1R-{EDF_FREEPHASE_PRODUCT}{kraken_code}/standard-unit-rates/?page_size=96"
    )
    rates = {}
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            for item in results:
                ts = item["valid_from"].replace(".000Z", "Z")
                rates[ts] = round(item["value_inc_vat"], 2)
    except Exception as e:
        print(f"EDF Kraken API error ({kraken_code}): {e}")
    return dict_key, rates

def fetch_carbon_intensity():
    """Fetch national average carbon intensity (gCO2/kWh)."""
    try:
        res = requests.get("https://api.carbonintensity.org.uk/intensity", timeout=10).json()
        data = res["data"][0]["intensity"]
        actual = data.get("actual")
        return actual if actual is not None else data.get("forecast", 0)
    except Exception as e:
        print(f"Error fetching carbon intensity: {e}")
        return 0

def build_timeline():
    """Concurrently fetch rates for all 14 regions and construct 48 half-hour slots."""
    now = datetime.now(timezone.utc)
    start_time = now.replace(minute=0 if now.minute < 30 else 30, second=0, microsecond=0)
    
    agile_by_region = {}
    freephase_by_region = {}

    with ThreadPoolExecutor(max_workers=14) as executor:
        octopus_futures = [
            executor.submit(fetch_octopus_region, key, meta["kraken_code"])
            for key, meta in DNO_REGIONS.items()
        ]
        edf_futures = [
            executor.submit(fetch_edf_region, key, meta["kraken_code"])
            for key, meta in DNO_REGIONS.items()
        ]

        for future in as_completed(octopus_futures):
            key, rates = future.result()
            agile_by_region[key] = rates

        for future in as_completed(edf_futures):
            key, rates = future.result()
            freephase_by_region[key] = rates

    half_hours = []

    for i in range(48):
        slot_dt = start_time + timedelta(minutes=30 * i)
        iso_key = slot_dt.strftime("%Y-%m-%dT%H:%M:00Z")
        display_time = slot_dt.strftime("%H:%M")
        hour = slot_dt.hour

        if 23 <= hour or hour < 6:
            band_name, band_code = "Green (Off-Peak)", "GREEN"
        elif 16 <= hour < 19:
            band_name, band_code = "Red (Peak)", "RED"
        else:
            band_name, band_code = "Amber (Standard)", "AMBER"

        region_pricing = {}
        for key in DNO_REGIONS.keys():
            agile_rate = agile_by_region.get(key, {}).get(iso_key)
            edf_rate = freephase_by_region.get(key, {}).get(iso_key)

            region_pricing[key] = {
                "agile_price": agile_rate,
                "edf_price": edf_rate,
                "is_free_moment": edf_rate == 0.0 if edf_rate is not None else False
            }

        half_hours.append({
            "iso_timestamp": slot_dt.isoformat(),
            "display_time": display_time,
            "band_name": band_name,
            "band_code": band_code,
            "regional_pricing": region_pricing
        })

    return half_hours

def main():
    os.makedirs("data", exist_ok=True)
    
    print("Fetching national carbon intensity...")
    carbon = fetch_carbon_intensity()

    print("Fetching half-hourly tariff rates across 14 DNO regions...")
    timeline = build_timeline()

    payload = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "dno_regions": {k: {"name": v["name"]} for k, v in DNO_REGIONS.items()},
        "carbon_intensity": carbon,
        "half_hourly_timeline": timeline
    }

    output_path = os.path.join("data", "grid_status.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    main()
