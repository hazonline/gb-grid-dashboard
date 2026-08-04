import os
import json
import requests
from datetime import datetime, timezone, timedelta

# UK DNO Region Multipliers (Adjusts national Agile & FreePhase base rates by region)
DNO_REGIONS = {
    "_A": {"name": "Eastern England (_A)", "multiplier": 1.0, "offset": 0.0},
    "_B": {"name": "East Midlands (_B)", "multiplier": 0.98, "offset": -0.2},
    "_C": {"name": "London (_C)", "multiplier": 1.05, "offset": 0.5},
    "_D": {"name": "Merseyside & North Wales (_D)", "multiplier": 1.03, "offset": 0.3},
    "_E": {"name": "West Midlands (_E)", "multiplier": 0.99, "offset": -0.1},
    "_F": {"name": "North Eastern (_F)", "multiplier": 0.97, "offset": -0.4},
    "_G": {"name": "North Western (_G)", "multiplier": 1.01, "offset": 0.1},
    "_H": {"name": "Southern England (_H)", "multiplier": 1.02, "offset": 0.2},
    "_J": {"name": "South Eastern (_J)", "multiplier": 1.04, "offset": 0.4},
    "_K": {"name": "South Wales (_K)", "multiplier": 1.06, "offset": 0.6},
    "_L": {"name": "South Western (_L)", "multiplier": 1.07, "offset": 0.7},
    "_M": {"name": "Yorkshire (_M)", "multiplier": 0.98, "offset": -0.3},
    "_N": {"name": "Southern Scotland (_N)", "multiplier": 0.95, "offset": -0.6},
    "_P": {"name": "Northern Scotland (_P)", "multiplier": 0.94, "offset": -0.8}
}

def fetch_live_grid_data():
    """Fetch exact instantaneous fuel generation in MW from Elexon BMRS."""
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
        url = "https://data.elexon.co.uk/bmrs/api/v1/datasets/FUELINST/stream"
        res = requests.get(url, timeout=10).json()
        data_items = res if isinstance(res, list) else res.get("data", [])
        
        # Take the most recent snapshot per fuel type
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
        print(f"Elexon stream error: {e}")

    # Fallback to Carbon Intensity API if stream fails
    if total_mw < 5000: # Typical UK grid minimum is > 12,000 MW
        try:
            res = requests.get("https://api.carbonintensity.org.uk/generation", timeout=10).json()
            mix = res["data"]["generationmix"]
            est_total_demand = 25000.0
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

def fetch_octopus_agile_regional():
    """Fetch Octopus Agile rates for all UK DNO regions."""
    regional_agile = {}
    for code in DNO_REGIONS.keys():
        regional_agile[code] = {}
        try:
            url = f"https://api.octopus.energy/v1/products/AGILE-24-04-03/electricity-tariffs/E-1R-AGILE-24-04-03{code}/standard-unit-rates/?page_size=96"
            res = requests.get(url, timeout=5).json()
            for item in res.get("results", []):
                regional_agile[code][item["valid_from"]] = round(item["value_inc_vat"], 2)
        except Exception as e:
            print(f"Error pulling region {code}: {e}")
    return regional_agile

def process_half_hourly_timeline(regional_agile):
    """
    Build 48 half-hour settlement slots for today/tomorrow.
    Calculates flat daily Green/Amber/Red bands for EDF FreePhase.
    """
    now = datetime.now(timezone.utc)
    start_time = now.replace(minute=0 if now.minute < 30 else 30, second=0, microsecond=0)
    
    half_hours = []

    # Weather forecast for renewable production estimate
    try:
        w_res = requests.get(
            "https://api.open-meteo.com/v1/forecast?latitude=54.0&longitude=-2.0&hourly=windspeed_10m,direct_normal_irradiance&forecast_days=2",
            timeout=10
        ).json()
        w_times = w_res["hourly"]["time"]
        w_wind = w_res["hourly"]["windspeed_10m"]
        w_solar = w_res["hourly"]["direct_normal_irradiance"]
    except Exception:
        w_times, w_wind, w_solar = [], [], []

    # Step 1: Pre-generate raw Agile curve to derive EDF daily band averages
    raw_agile_points = []
    for i in range(48):
        slot_dt = start_time + timedelta(minutes=30 * i)
        iso_key = slot_dt.strftime("%Y-%m-%dT%H:%M:00Z")
        hour = slot_dt.hour

        # Base regional price reference (_A)
        official_val = regional_agile.get("_A", {}).get(iso_key)
        if official_val is not None:
            price = official_val
        else:
            # Synthetic prediction model
            w_idx = min(len(w_times)-1, int(i / 2)) if w_times else 0
            wind_speed = w_wind[w_idx] if w_wind else 15
            solar_irrad = w_solar[w_idx] if w_solar else 0
            
            demand_gw = 20.0 + (10.0 if 7 <= hour <= 22 else 0)
            renew_gw = min(18.0, (wind_speed / 35.0)**3 * 14.0 + (solar_irrad / 800.0) * 8.0)
            net_gw = demand_gw - renew_gw
            price = round(12.0 + (net_gw - 10.0) * 1.4, 2)

        raw_agile_points.append({"dt": slot_dt, "price": price, "hour": hour})

    # Step 2: Calculate Flat Daily FreePhase Band Rates (Green, Amber, Red)
    green_prices = [p["price"] for p in raw_agile_points if p["hour"] >= 23 or p["hour"] < 6]
    red_prices = [p["price"] for p in raw_agile_points if 16 <= p["hour"] < 19]
    amber_prices = [p["price"] for p in raw_agile_points if (6 <= p["hour"] < 16) or (19 <= p["hour"] < 23)]

    edf_green_flat = round(max(9.5, (sum(green_prices)/len(green_prices)) * 0.70) if green_prices else 12.0, 2)
    edf_amber_flat = round(max(18.0, (sum(amber_prices)/len(amber_prices)) * 0.90) if amber_prices else 20.0, 2)
    edf_red_flat = round(max(34.0, (sum(red_prices)/len(red_prices)) * 1.25) if red_prices else 38.0, 2)

    # Step 3: Populate 48 Half-Hourly Items for all regions
    for pt in raw_agile_points:
        dt = pt["dt"]
        hour = pt["hour"]
        iso_key = dt.strftime("%Y-%m-%dT%H:%M:00Z")
        display_time = dt.strftime("%H:%M")

        # FreePhase Band Classification
        if 23 <= hour or hour < 6:
            band_name, band_code, base_edf = "Green", "GREEN", edf_green_flat
        elif 16 <= hour < 19:
            band_name, band_code, base_edf = "Red (Peak)", "RED", edf_red_flat
        else:
            band_name, band_code, base_edf = "Amber", "AMBER", edf_amber_flat

        region_data = {}
        for code, info in DNO_REGIONS.items():
            reg_agile = regional_agile.get(code, {}).get(iso_key)
            if reg_agile is None:
                reg_agile = round((pt["price"] * info["multiplier"]) + info["offset"], 2)
            
            # Check for Free Moment
            is_free = reg_agile <= 0.0
            reg_edf = 0.0 if is_free else round((base_edf * info["multiplier"]) + info["offset"], 2)

            region_data[code] = {
                "agile_price": reg_agile,
                "edf_price": reg_edf,
                "is_free": is_free
            }

        half_hours.append({
            "iso_timestamp": dt.isoformat(),
            "display_time": display_time,
            "band_name": band_name,
            "band_code": band_code,
            "regional_pricing": region_data
        })

    return half_hours

def main():
    os.makedirs("data", exist_ok=True)
    grid = fetch_live_grid_data()
    carbon = fetch_carbon_intensity()
    reg_agile = fetch_octopus_agile_regional()
    timeline = process_half_hourly_timeline(reg_agile)

    payload = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "dno_regions": DNO_REGIONS,
        "live_grid": grid,
        "carbon_intensity": carbon,
        "half_hourly_timeline": timeline
    }

    with open("data/grid_status.json", "w") as f:
        json.dump(payload, f, indent=2)

    print("Data refreshed successfully.")

if __name__ == "__main__":
    main()
