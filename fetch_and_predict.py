import os
import json
import requests
from datetime import datetime, timezone, timedelta

def fetch_live_grid_data():
    """
    Fetch exact live GB electricity generation outturn by fuel type (in MW)
    directly from Elexon BMRS FUELINST (Instantaneous Generation).
    """
    fuel_mw = {}
    total_mw = 0.0
    renewable_mw = 0.0

    try:
        # Elexon BMRS Instantaneous Generation Outturn API
        url = "https://data.elexon.co.uk/bmrs/api/v1/generation/outturn/summary"
        res = requests.get(url, timeout=12).json()
        
        # Map Elexon fuel types
        fuel_map = {
            "CCGT": "Gas (CCGT)",
            "OIL": "Oil",
            "COAL": "Coal",
            "NUCLEAR": "Nuclear",
            "WIND": "Wind",
            "SOLAR": "Solar",
            "BIOMASS": "Biomass",
            "HYDRO": "Hydro",
            "NPSHYD": "Pumped Storage",
            "INTFR": "French Link",
            "INTIFA2": "IFA2 Link",
            "INTIRL": "Irish Link",
            "INTNED": "Dutch Link",
            "INTEW": "EirGrid Link",
            "INTNEM": "Nemo Link",
            "INTNSL": "North Sea Link",
            "INTVIK": "Viking Link",
            "OTHER": "Other"
        }

        for item in res:
            raw_fuel = item.get("fuelType", "").upper()
            mw = float(item.get("halfHourlyGbOutturn", 0) or item.get("currentGbOutturn", 0) or 0)
            
            if mw > 0 and raw_fuel in fuel_map:
                name = fuel_map[raw_fuel]
                fuel_mw[name] = round(mw, 1)
                total_mw += mw
                
                if raw_fuel in ["WIND", "SOLAR", "BIOMASS", "HYDRO"]:
                    renewable_mw += mw

        renewable_pct = round((renewable_mw / total_mw * 100), 1) if total_mw > 0 else 0.0

    except Exception as e:
        print(f"Error pulling Elexon fuel outturn: {e}")
        # Secondary fallback: Carbon Intensity API Generation Mix %
        try:
            fallback_res = requests.get("https://api.carbonintensity.org.uk/generation", timeout=10).json()
            mix = fallback_res["data"]["generationmix"]
            for item in mix:
                fuel_mw[item["fuel"].capitalize()] = item["perc"]
            renewable_pct = sum([x["perc"] for x in mix if x["fuel"] in ["wind", "solar", "biomass", "hydro"]])
            total_mw = 0.0 # Standard marker for percentage fallback mode
        except Exception as err:
            print(f"Fallback grid fetch failed: {err}")

    return {
        "total_mw": round(total_mw, 0),
        "renewable_pct": renewable_pct,
        "fuel_breakdown": fuel_mw
    }

def fetch_carbon_intensity():
    """Fetch live UK national carbon intensity (gCO2/kWh)."""
    try:
        res = requests.get("https://api.carbonintensity.org.uk/intensity", timeout=10).json()
        data = res["data"][0]["intensity"]
        return data["actual"] if data["actual"] is not None else data["forecast"]
    except Exception as e:
        print(f"Carbon intensity API error: {e}")
        return 0

def fetch_official_octopus_agile():
    """Fetch official published half-hourly unit rates for Octopus Agile (p/kWh)."""
    agile_rates = {}
    try:
        url = "https://api.octopus.energy/v1/products/AGILE-24-04-03/electricity-tariffs/E-1R-AGILE-24-04-03-A/standard-unit-rates/?page_size=96"
        res = requests.get(url, timeout=10).json()
        for item in res.get("results", []):
            start_iso = item["valid_from"]
            agile_rates[start_iso] = round(item["value_inc_vat"], 2)
    except Exception as e:
        print(f"Octopus API fetch error: {e}")
    return agile_rates

def calculate_edf_freephase_band(dt, agile_price):
    """
    Calculate EDF FreePhase rate bands & free moment alerts:
    - Green Band: 23:00 - 06:00 (Cheapest Overnight)
    - Red Band: 16:00 - 19:00 (Peak Evening Surcharge)
    - Amber Band: 06:00 - 16:00 & 19:00 - 23:00 (Standard)
    - FREE MOMENT: Triggered when wholesale/Agile price drops <= 0p/kWh
    """
    hour = dt.hour
    
    is_free_moment = agile_price <= 0.0
    
    if 23 <= hour or hour < 6:
        band_name = "Green"
        band_code = "GREEN"
        base_rate = round(max(8.0, agile_price * 0.65), 2)
    elif 16 <= hour < 19:
        band_name = "Red (Peak)"
        band_code = "RED"
        base_rate = round(max(32.0, agile_price * 1.35), 2)
    else:
        band_name = "Amber"
        band_code = "AMBER"
        base_rate = round(max(18.0, agile_price * 0.95), 2)
        
    if is_free_moment:
        base_rate = 0.0

    return {
        "band_name": band_name,
        "band_code": band_code,
        "estimated_rate": base_rate,
        "is_free_moment": is_free_moment
    }

def forecast_grid_and_prices(official_agile):
    """Forecast prices & generation using Open-Meteo wind/solar models."""
    now = datetime.now(timezone.utc)
    weather_url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=54.0&longitude=-2.0"
        "&hourly=windspeed_10m,direct_normal_irradiance"
        "&forecast_days=2"
    )

    forecasts = []
    try:
        res = requests.get(weather_url, timeout=10).json()
        timestamps = res["hourly"]["time"]
        wind_speeds = res["hourly"]["windspeed_10m"]
        solar_rad = res["hourly"]["direct_normal_irradiance"]

        for i in range(len(timestamps)):
            dt = datetime.fromisoformat(timestamps[i]).replace(tzinfo=timezone.utc)

            # Keep only upcoming half-hours
            if dt < now - timedelta(minutes=30):
                continue

            hour = dt.hour
            iso_start = dt.strftime("%Y-%m-%dT%H:%M:00Z")
            
            # Use official Octopus Agile rate if published, otherwise estimate
            official_price = None
            for k, v in official_agile.items():
                if k.startswith(timestamps[i][:13]):
                    official_price = v
                    break

            # Estimate net demand vs renewable yield
            typical_demand_gw = 18.0 + 11.0 * (1 if 7 <= hour <= 22 else 0)
            est_wind_gw = min(17.5, (wind_speeds[i] / 38.0) ** 3 * 15.0)
            est_solar_gw = min(9.5, (solar_rad[i] / 750.0) * 8.5)
            est_renewables_gw = est_wind_gw + est_solar_gw
            net_demand_gw = typical_demand_gw - est_renewables_gw

            if net_demand_gw < 5.0:
                predicted_price = -3.0 + (net_demand_gw * 0.5)
            elif net_demand_gw < 10.0:
                predicted_price = 1.0 + (net_demand_gw - 5.0) * 1.9
            else:
                predicted_price = 11.0 + (net_demand_gw - 10.0) * 1.3

            agile_price = official_price if official_price is not None else round(predicted_price, 2)
            edf_info = calculate_edf_freephase_band(dt, agile_price)

            forecasts.append({
                "iso_timestamp": dt.isoformat(),
                "agile_price": agile_price,
                "is_official_agile": official_price is not None,
                "edf_band": edf_info["band_name"],
                "edf_band_code": edf_info["band_code"],
                "edf_estimated_rate": edf_info["estimated_rate"],
                "is_free_moment": edf_info["is_free_moment"],
                "est_renewable_pct": min(100, round((est_renewables_gw / typical_demand_gw) * 100, 1))
            })

    except Exception as e:
        print(f"Forecasting pipeline error: {e}")

    return forecasts

def main():
    os.makedirs("data", exist_ok=True)

    grid = fetch_live_grid_data()
    carbon = fetch_carbon_intensity()
    official_agile = fetch_official_octopus_agile()
    forecasts = forecast_grid_and_prices(official_agile)

    payload = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "live_grid": grid,
        "carbon_intensity": carbon,
        "forecasts": forecasts
    }

    with open("data/grid_status.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Grid status successfully updated with {len(forecasts)} forecasts.")

if __name__ == "__main__":
    main()
