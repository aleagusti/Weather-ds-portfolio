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
   "maximum_wind_speed_10m",
   "dominant_wind_direction_10m",
   "sunrise",
   "sunset",
   "shortwave_radiation_sum",
   "weathercode"
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
        "timezone": TIMEZONE
    }

    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()

    data = response.json()
    df = pd.DataFrame(data["daily"])
    df["date"] = pd.to_datetime(df["time"])

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