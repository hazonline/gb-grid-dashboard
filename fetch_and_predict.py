import os
import json
import requests
from datetime import datetime, timezone, timedelta

DNO_CODES = ["_A", "_B", "_C", "_D", "_E", "_F", "_G", "_H", "_J", "_K", "_L", "_M", "_N", "_P"]

OCTOPUS_AGILE_PRODUCT = "AGILE-24-04-03"
EDF_FREEPHASE_PRODUCT = "EDF_FREEPHASE_DYNAMIC_12M_HH"

def fetch_octopus_agile_rates(dno_code):
    """Fetch Octopus Agile rates directly from Octopus Kraken API."""
    url = (
        f"https://api.octopus.energy/v1/products/{OCTOPUS_AGILE_PRODUCT}/"
        f"electricity-tariffs/E-1R-{OCTOPUS_AGILE_PRODUCT}{dno_code}/standard-unit-rates/?page_size=96"
    )
    try:
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            return {
                item["valid_from"].replace(".000Z", "Z"): round(item["value_inc_vat"], 2)
                for item in results
            }
    except Exception as e:
        print(f"Error fetching Octopus Agile for region {dno_code}: {e}")
    return {}

def fetch_edf_freephase_rates(dno_code):
    """Fetch EDF FreePhase rates directly from EDF Kraken API (edfgb-kraken.energy)."""
    url = (
        f"https://api.edfgb-kraken.energy/v1/products/{EDF_FREEPHASE_PRODUCT}/"
        f"electricity-tariffs/E-1R-{EDF_FREEPHASE_PRODUCT}{dno_code}/standard-unit-rates/?page_size=96"
    )
    try:
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            return {
                item["valid_from"].replace(".000Z", "Z"): round(item["value_inc_vat"], 2)
                for item in results
            }
    except Exception as e:
        print(f"Error fetching EDF FreePhase for region {dno_code}: {e}")
    return {}

def fetch_live_grid():
    """Fetch live MW generation from Elexon BMRS API."""
    fuel_mw = {}
    total_mw = 0.0
    renewable_mw = 0.0

    fuel_map = {
        "CCGT": "Gas (CCGT)", "OIL": "Oil", "COAL": "Coal",
        "NUCLEAR": "Nuclear", "WIND": "Wind", "SOLAR": "Solar",
        "BIOMASS": "Biomass", "HYDRO": "Hydro", "NPSHYD": "Pumped Storage",
        "INTFR": "French Link", "INTIFA2": "IFA2 Link", "INTIRL": "Irish Link",
        "INTNED": "Dutch Link", "INTEW": "EirGrid Link", "INTNEM": "Nemo Link",
        "INTNSL": "North Sea Link", "INTVIK": "Viking Link", "OTHER": "Other"
    }

    try:
        url = "https://data.elexon.co.uk/bmrs/api/v1/datasets/FUELINST"
        res = requests.get(url, timeout=10).json()
        data_items = res if isinstance(res, list) else res.get("data", [])
        
        latest_by_fuel = {}
        for item in data_items:
            fuel = item.get("fuelType", "").upper()
            mw = float(item.get("generation", 0) or 0)
            if fuel in fuel_map:
                latest_by_fuel[fuel] = mw

        for fuel, mw in latest_by_fuel.items():
            name = fuel_map[fuel]
            fuel_mw[name] = round(mw, 1)
            total_mw += mw
            if fuel in ["WIND", "SOLAR", "BIOMASS", "HYDRO"]:
                renewable_mw += mw

    except Exception as e:
        print(f"Elexon API error: {e}")

    # Fallback to Carbon Intensity API if stream/fetch fails
    if total_mw < 5000:
        try:
            res = requests.get("https://api.carbonintensity.org.uk/generation", timeout=10).json()
            mix = res["data"]["generationmix"]
            est_total_demand = 22000.0
            fuel_mw = {}
            total_mw = 0.0
            renewable_mw = 0.0
            for item in mix:
                mw = round((item["perc"] / 100.0) * est_total_demand, 1)
                fuel_mw[item["fuel"].capitalize()] = mw
                total_mw += mw
                if item["fuel"] in ["wind", "solar", "biomass", "hydro"]:
                    renewable_mw += mw
        except Exception as err:
            print(f"Fallback generation fetch failed: {err}")

    renewable_pct = round((renewable_mw / total_mw * 100), 1) if total_mw > 0 else 0.0

    return {
        "total_mw": round(total_mw, 0),
        "renewable_pct": renewable_pct,
        "fuel_breakdown": fuel_mw
    }

def fetch_carbon_intensity():
    try:
        res = requests.get("https://api.carbonintensity.org.uk/intensity", timeout=10).json()
        data = res["data"][0]["intensity"]
        return data["actual"] if data["actual"] is not None else data["forecast"]
    except Exception:
        return 0

def build_timeline():
    """Build 48 half-hour slots using true API endpoints for both Agile and FreePhase."""
    now = datetime.now(timezone.utc)
    start_time = now.replace(minute=0 if now.minute < 30 else 30, second=0, microsecond=0)
    
    agile_by_region = {code: fetch_octopus_agile_rates(code) for code in DNO_CODES}
    freephase_by_region = {code: fetch_edf_freephase_rates(code) for code in DNO_CODES}

    half_hours = []

    for i in range(48):
        slot_dt = start_time + timedelta(minutes=30 * i)
        iso_key = slot_dt.strftime("%Y-%m-%dT%H:%M:00Z")
        display_time = slot_dt.strftime("%H:%M")
        hour = slot_dt.hour

        # Structural Banding for EDF FreePhase
        if 23 <= hour or hour < 6:
            band_name, band_code = "Green (Off-Peak)", "GREEN"
        elif 16 <= hour < 19:
            band_name, band_code = "Red (Peak)", "RED"
        else:
            band_name, band_code = "Amber (Standard)", "AMBER"

        region_pricing = {}
        for code in DNO_CODES:
            agile_rate = agile_by_region.get(code, {}).get(iso_key)
            edf_rate = freephase_by_region.get(code, {}).get(iso_key)

            region_pricing[code] = {
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
    grid = fetch_live_grid()
    carbon = fetch_carbon_intensity()
    timeline = build_timeline()

    payload = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "live_grid": grid,
        "carbon_intensity": carbon,
        "half_hourly_timeline": timeline
    }

    output_path = os.path.join("data", "grid_status.json")
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Refreshed real Kraken data. Saved to {output_path}")

if __name__ == "__main__":
    main()
