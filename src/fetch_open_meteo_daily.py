import requests
import pandas as pd
from pathlib import Path

# =========================
# Configuration
# =========================

LATITUDE = 25.7617
LONGITUDE = -80.1918
START_DATE = "1990-01-01"
END_DATE = "2024-12-31"
TIMEZONE = "UTC"

DAILY_VARIABLES = [
   "temperature_2m_max",
   "temperature_2m_min",
   "temperature_2m_mean",
   "rain_sum",
   "precipitation_sum",
   "precipitation_hours",
   "wind_speed_10m_max",
   "wind_direction_10m_dominant",
   "sunrise",
   "sunset",
   "shortwave_radiation_sum",
   "weather_code",
]

HOURLY_VARIABLES = [
    "cloud_cover",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
]

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

OUTPUT_PATH = Path("data/raw/open_meteo_miami_daily.csv")


# =========================
# Data fetch
# =========================

def fetch_daily_weather():
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": DAILY_VARIABLES,   
        "hourly": HOURLY_VARIABLES,
        "timezone": TIMEZONE,
    }

    if not isinstance(params["daily"], list) or len(params["daily"]) == 0:
        raise ValueError("'daily' must be a non-empty list of variable names")

    response = requests.get(BASE_URL, params=params)

    try:
        response.raise_for_status()
    except requests.HTTPError:
        print("❌ Open-Meteo API request failed")
        print(f"URL: {response.url}")
        print(f"Status code: {response.status_code}")
        print(f"Response text: {response.text}")
        raise

    data = response.json()
    df = pd.DataFrame(data["daily"])
    df["date"] = pd.to_datetime(df["time"]).dt.date

    df_hourly = pd.DataFrame(data["hourly"])
    df_hourly["time"] = pd.to_datetime(df_hourly["time"])
    df_hourly["date"] = df_hourly["time"].dt.date

    hourly_means = (
        df_hourly
        .groupby("date")[HOURLY_VARIABLES]
        .mean()
        .reset_index()
        .rename(columns={var: f"{var}_mean" for var in HOURLY_VARIABLES})
    )

    df = df.merge(hourly_means, on="date", how="left")

    return df


# =========================
# Script entry point
# =========================

def main():
    print("[START] Fetching daily weather data from Open-Meteo")

    df = fetch_daily_weather()

    if df.empty:
        raise RuntimeError("Downloaded dataset is empty")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"[DONE] Dataset saved to {OUTPUT_PATH}")
    print(f"Rows: {len(df)} | Columns: {df.shape[1]}")
    print(df.head())


if __name__ == "__main__":
    main()