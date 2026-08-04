import os
import json
import requests
from datetime import datetime, timezone, timedelta

def fetch_live_grid_data():
    """Fetch current GB generation breakdown using National Grid's Carbon Intensity API."""
    gen_data = {}
    total_mw = 0
    renewable_pct = 0.0

    try:
        # National Grid Carbon Intensity Generation Mix Endpoint
        url = "https://api.carbonintensity.org.uk/generation"
        res = requests.get(url, timeout=10).json()
        generation_mix = res["data"]["generationmix"]

        # Nominal national demand baseline (~28,000 MW average) to estimate MW per fuel
        ESTIMATED_GRID_MW = 28000 
        
        renewables_perc = 0.0
        for item in generation_mix:
            fuel = item["fuel"].upper()
            perc = item["perc"]
            gen_data[fuel] = round((perc / 100.0) * ESTIMATED_GRID_MW, 0)
            
            if fuel in ["WIND", "SOLAR", "BIOMASS", "HYDRO"]:
                renewables_perc += perc

        renewable_pct = round(renewables_perc, 1)
        total_mw = ESTIMATED_GRID_MW

    except Exception as e:
        print(f"Grid fetch error: {e}")

    return {
        "total_mw": total_mw,
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
    """Fetch official Octopus Agile rates."""
    rates = []
    try:
        url = "https://api.octopus.energy/v1/products/AGILE-24-04-03/electricity-tariffs/E-1R-AGILE-24-04-03-A/standard-unit-rates/?page_size=48"
        res = requests.get(url, timeout=10).json()
        for item in res.get("results", []):
            rates.append({
                "valid_from": item["valid_from"],
                "price": round(item["value_inc_vat"], 2)
            })
    except Exception as e:
        print(f"Octopus API fetch error: {e}")
    return rates

def predict_future_prices(official_rates):
    """Forecast prices using weather model + official Octopus rates."""
    now = datetime.now(timezone.utc)
    weather_url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=54.0&longitude=-2.0"
        "&hourly=windspeed_10m,direct_normal_irradiance"
        "&forecast_days=2"
    )

    predictions = []
    try:
        res = requests.get(weather_url, timeout=10).json()
        timestamps = res["hourly"]["time"]
        wind_speeds = res["hourly"]["windspeed_10m"]
        solar_rad = res["hourly"]["direct_normal_irradiance"]

        for i in range(len(timestamps)):
            # Parse Open-Meteo time (YYYY-MM-DDTHH:MM)
            dt = datetime.fromisoformat(timestamps[i]).replace(tzinfo=timezone.utc)

            # Keep only current and upcoming slots
            if dt < now - timedelta(hours=1):
                continue

            hour = dt.hour
            
            # Match official Octopus Agile rates if available
            official_price = None
            dt_str = dt.strftime("%Y-%m-%dT%H:%M:00Z")
            for rate in official_rates:
                if rate["valid_from"].startswith(timestamps[i][:13]):
                    official_price = rate["price"]
                    break

            # Demand vs Renewable prediction heuristic
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

            predictions.append({
                "timestamp": dt.isoformat(),
                "est_agile_price": final_price,
                "is_official": official_price is not None,
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

    print(f"Updated data/grid_status.json successfully with {len(predictions)} forecasts.")

if __name__ == "__main__":
    main()
