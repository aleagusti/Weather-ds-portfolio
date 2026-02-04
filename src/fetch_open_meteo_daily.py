import requests
import pandas as pd
from src.config import RAW_DATASET, RAW_DIR

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

# =========================
# Data fetch
# =========================

def fetch_daily_weather() -> pd.DataFrame:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": DAILY_VARIABLES,
        "hourly": HOURLY_VARIABLES,
        "timezone": TIMEZONE,
    }

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

    df_daily = pd.DataFrame(data["daily"])
    df_daily["date"] = pd.to_datetime(df_daily["time"]).dt.date

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

    df = df_daily.merge(hourly_means, on="date", how="left")
    return df

# =========================
# Script entry point
# =========================

def main():
    print("[STEP 1] Fetching raw data")

    if RAW_DATASET.exists():
        print("[SKIP] Raw dataset already exists. Using cached file.")
        print(f"Path: {RAW_DATASET}")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df = fetch_daily_weather()

    if df.empty:
        raise RuntimeError("Downloaded dataset is empty")

    df.to_csv(RAW_DATASET, index=False)

    print(f"[DONE] Dataset saved to {RAW_DATASET}")
    print(f"Rows: {len(df)} | Columns: {df.shape[1]}")
    print(df.head())

if __name__ == "__main__":
    main()