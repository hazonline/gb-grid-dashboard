import os
import json
import requests
from datetime import datetime, timezone, timedelta

def fetch_live_grid_data():
    """Fetch current GB generation breakdown in MW and fuel percentages."""
    gen_data = {}
    total_mw = 0
    renewable_pct = 0.0
    
    # 1. Primary: Elexon Current Outturn Endpoint
    try:
        url = "https://data.elexon.co.uk/bmrs/api/v1/generation/outturn/current"
        res = requests.get(url, timeout=10).json()
        
        for item in res:
            fuel = item.get("fuelType", "").upper()
            mw = item.get("currentGbOutturn", 0) or item.get("halfHourlyGbOutturn", 0)
            if fuel and mw > 0:
                gen_data[fuel] = round(mw, 2)
                
        total_mw = sum(gen_data.values())
        renewables = gen_data.get("WIND", 0) + gen_data.get("SOLAR", 0) + gen_data.get("BIOMASS", 0) + gen.get("HYDRO", 0)
        if total_mw > 0:
            renewable_pct = round((renewables / total_mw) * 100, 1)
    except Exception as e:
        print(f"Elexon fetch error: {e}")

    # 2. Fallback / Carbon Intensity API for percentage validation if total_mw is 0
    if total_mw == 0:
        try:
            url = "https://api.carbonintensity.org.uk/generation"
            res = requests.get(url, timeout=10).json()
            generation_mix = res["data"]["generationmix"]
            
            renewables_perc = 0
            for fuel in generation_mix:
                fuel_name = fuel["fuel"].upper()
                perc = fuel["perc"]
                gen_data[fuel_name] = perc  # Percentage mode fallback
                if fuel_name in ["WIND", "SOLAR", "BIOMASS", "HYDRO"]:
                    renewables_perc += perc
                    
            renewable_pct = round(renewables_perc, 1)
            total_mw = 25000 # Estimated nominal grid load
        except Exception as e:
            print(f"Carbon intensity generation fetch error: {e}")

    return {
        "total_mw": round(total_mw, 0),
        "renewable_pct": renewable_pct,
        "breakdown": gen_data
    }

def fetch_carbon_intensity():
    """Fetch UK Carbon Intensity (gCO2/kWh)."""
    try:
        url = "https://api.carbonintensity.org.uk/intensity"
        res = requests.get(url, timeout=10).json()
        data = res["data"][0]["intensity"]
        return data["actual"] if data["actual"] is not None else data["forecast"]
    except Exception as e:
        print(f"Carbon intensity error: {e}")
        return 0

def fetch_official_octopus_agile():
    """Fetch official Octopus Agile rates for upcoming hours."""
    official_rates = {}
    try:
        # Octopus Agile GSP Group A (Eastern/London/National standard API endpoint)
        url = "https://api.octopus.energy/v1/products/AGILE-24-04-03/electricity-tariffs/E-1R-AGILE-24-04-03-A/standard-unit-rates/?page_size=96"
        res = requests.get(url, timeout=10).json()
        
        for item in res.get("results", []):
            valid_from = item["value_exc_vat"] # p/kWh inc VAT is value_inc_vat
            price_inc_vat = item["value_inc_vat"]
            start_time = item["valid_from"]
            official_rates[start_time] = round(price_inc_vat, 2)
    except Exception as e:
        print(f"Octopus API fetch error: {e}")
    return official_rates

def predict_future_prices(official_rates):
    """Forecast prices using weather model + official Octopus rates for next 48 hours."""
    now = datetime.now(timezone.utc)
    weather_url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=54.0&longitude=-2.0" # Center of UK landmass
        "&hourly=windspeed_10m,direct_normal_irradiance"
        "&forecast_days=3"
    )
    
    predictions = []
    try:
        res = requests.get(weather_url, timeout=10).json()
        timestamps = res["hourly"]["time"]
        wind_speeds = res["hourly"]["windspeed_10m"]
        solar_rad = res["hourly"]["direct_normal_irradiance"]
        
        for i in range(len(timestamps)):
            # Parse Open-Meteo ISO time
            dt = datetime.fromisoformat(timestamps[i]).replace(tzinfo=timezone.utc)
            
            # Skip past hours
            if dt < now - timedelta(hours=1):
                continue
                
            dt_iso = dt.strftime("%Y-%m-%d%H:%M:00Z")
            hour = dt.hour
            
            # Check if official Octopus Agile price exists for this period
            official_price = None
            for key in official_rates:
                if key.startswith(timestamps[i][:13]):
                    official_price = official_rates[key]
                    break

            # Model prediction (if official price not yet released)
            typical_demand_gw = 18 + 10 * (1 if 7 <= hour <= 22 else 0)
            est_wind_gw = min(18.0, (wind_speeds[i] / 38.0) ** 3 * 15.0)
            est_solar_gw = min(10.0, (solar_rad[i] / 750.0) * 8.5)
            est_renewables_gw = est_wind_gw + est_solar_gw
            net_demand_gw = typical_demand_gw - est_renewables_gw
            
            if net_demand_gw < 5.0:
                predicted_price = -2.0 + (net_demand_gw * 0.4)
            elif net_demand_gw < 10.0:
                predicted_price = 1.0 + (net_demand_gw - 5.0) * 1.8
            else:
                predicted_price = 10.0 + (net_demand_gw - 10.0) * 1.3

            final_price = official_price if official_price is not None else round(predicted_price, 2)
            is_official = official_price is not None

            predictions.append({
                "time": dt.isoformat(),
                "display_time": dt.strftime("%a %d %b, %H:%M"),
                "est_agile_price": final_price,
                "is_official": is_official,
                "is_octopus_agile_free_or_negative": final_price <= 0,
                "is_edf_freephase_trigger": final_price <= 1.0,
                "renewable_ratio_pct": min(100, round((est_renewables_gw / typical_demand_gw) * 100, 1))
            })
            
    except Exception as e:
        print(f"Price forecast error: {e}")
        
    return predictions

def main():
    os.makedirs("data", exist_ok=True)
    
    grid = fetch_live_grid_data()
    carbon = fetch_carbon_intensity()
    official_agile = fetch_official_octopus_agile()
    predictions = predict_future_prices(official_agile)
    
    output = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "live_grid": grid,
        "carbon_intensity_gco2_kwh": carbon,
        "predictions": predictions
    }
    
    with open("data/grid_status.json", "w") as f:
        json.dump(output, f, indent=2)
        
    print(f"Successfully updated grid_status.json with {len(predictions)} upcoming slots.")

if __name__ == "__main__":
    main()
