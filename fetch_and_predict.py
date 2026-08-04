import os
import json
import requests
from datetime import datetime, timezone

def fetch_live_grid_data():
    """Fetch current GB power generation outturn from Elexon API."""
    try:
        # Elexon Insights API endpoint for current generation summary
        url = "https://data.elexon.co.uk/bmrs/api/v1/generation/outturn/summary"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Parse generation breakdown by fuel type (MW)
        gen = {}
        for item in data:
            fuel = item.get("fuelType", "").upper()
            mw = item.get("halfHourlyGbOutturn", 0)
            gen[fuel] = round(mw, 2)
            
        total_gen = sum(gen.values())
        renewables = gen.get("WIND", 0) + gen.get("SOLAR", 0) + gen.get("BIOMASS", 0) + gen.get("HYDRO", 0)
        renewable_pct = round((renewables / total_gen * 100), 1) if total_gen > 0 else 0

        return {
            "total_mw": round(total_gen, 2),
            "renewable_pct": renewable_pct,
            "breakdown": gen
        }
    except Exception as e:
        print(f"Error fetching live grid data: {e}")
        return {"total_mw": 0, "renewable_pct": 0, "breakdown": {}}

def fetch_carbon_intensity():
    """Fetch current UK carbon intensity (gCO2/kWh)."""
    try:
        url = "https://api.carbonintensity.org.uk/intensity"
        response = requests.get(url, timeout=10)
        data = response.json()
        return data["data"][0]["intensity"]["actual"] or data["data"][0]["intensity"]["forecast"]
    except Exception as e:
        print(f"Error fetching carbon intensity: {e}")
        return 0

def predict_future_prices():
    """
    Fetch weather forecasts for UK wind/solar and run price prediction logic
    for Octopus Agile and EDF FreePhase.
    """
    # Coordinates for UK center (lat: 52.5, lon: -1.5)
    weather_url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=52.5&longitude=-1.5"
        "&hourly=windspeed_10m,direct_normal_irradiance"
        "&forecast_days=2"
    )
    
    try:
        res = requests.get(weather_url, timeout=10).json()
        timestamps = res["hourly"]["time"]
        wind_speeds = res["hourly"]["windspeed_10m"] # km/h
        solar_rad = res["hourly"]["direct_normal_irradiance"] # W/m2
        
        predictions = []
        
        for i in range(len(timestamps)):
            dt_str = timestamps[i]
            hour = int(dt_str.split("T")[1].split(":")[0])
            
            # Simple heuristic model for net generation vs typical demand curve
            # Baseline UK demand fluctuates roughly 18GW (night) to 32GW (day)
            typical_demand_gw = 20 + 8 * (1 if 7 <= hour <= 22 else 0)
            
            # Estimated renewable contribution based on wind speed and irradiance
            est_wind_gw = min(18.0, (wind_speeds[i] / 40.0) ** 3 * 16.0)
            est_solar_gw = min(10.0, (solar_rad[i] / 800.0) * 9.0)
            est_renewables_gw = est_wind_gw + est_solar_gw
            
            # Net demand = Demand minus variable renewables
            net_demand_gw = typical_demand_gw - est_renewables_gw
            
            # Wholesale / Agile price estimation (p/kWh)
            # When Net Demand drops below ~8 GW, prices start approaching/going below zero
            if net_demand_gw < 6.0:
                est_agile_price = -2.5 + (net_demand_gw * 0.5)  # Negative pricing
            elif net_demand_gw < 10.0:
                est_agile_price = 0.0 + (net_demand_gw - 6.0) * 1.5 # 0p to 6p
            else:
                est_agile_price = 8.0 + (net_demand_gw - 10.0) * 1.2
            
            # Determine free/negative probability flags
            predictions.append({
                "time": dt_str,
                "est_agile_price": round(est_agile_price, 2),
                "is_octopus_agile_free_or_negative": est_agile_price <= 0,
                "is_edf_freephase_trigger": est_agile_price <= 1.0,
                "renewable_ratio_pct": min(100, round((est_renewables_gw / typical_demand_gw) * 100, 1))
            })
            
        return predictions
    except Exception as e:
        print(f"Error forecasting prices: {e}")
        return []

def main():
    os.makedirs("data", exist_ok=True)
    
    output = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "live_grid": fetch_live_grid_data(),
        "carbon_intensity_gco2_kwh": fetch_carbon_intensity(),
        "predictions": predict_future_prices()
    }
    
    with open("data/grid_status.json", "w") as f:
        json.dump(output, f, indent=2)
        
    print("Successfully generated data/grid_status.json")

if __name__ == "__main__":
    main()
