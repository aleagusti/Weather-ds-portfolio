import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import time

load_dotenv()

API_KEY = os.getenv("WEATHER_API_KEY")
CITY = "Buenos Aires"
DAYS_BACK = 180


def fetch_historical_weather(days=DAYS_BACK):
    records = []
    end_date = datetime.now(timezone.utc)

    print(f"Iniciando descarga de clima para {CITY} ({days} días)...")

    for i in range(days):
        date = (end_date - timedelta(days=i)).strftime("%Y-%m-%d")

        if i % 10 == 0:
            print(f"Procesados {i}/{days} días...")

        url = (
            "http://api.weatherapi.com/v1/history.json"
            f"?key={API_KEY}&q={CITY}&dt={date}"
        )

        response = requests.get(url)
        data = response.json()

        if "error" in data:
            print(f"Error en fecha {date}: {data['error']['message']}")
            time.sleep(0.5)
            continue

        forecast_day = data["forecast"]["forecastday"][0]
        day_data = forecast_day["day"]
        hours_data = forecast_day["hour"]

        avg_pressure = sum(h["pressure_mb"] for h in hours_data) / len(hours_data)

        records.append({
            "date": date,
            "temp_max_c": day_data["maxtemp_c"],
            "temp_min_c": day_data["mintemp_c"],
            "humidity_avg": day_data["avghumidity"],
            "precip_mm": day_data["totalprecip_mm"],
            "pressure_mb": round(avg_pressure, 2)
        })

        time.sleep(0.5)

    if not records:
        raise RuntimeError(
            "No se pudo recuperar ningún dato válido. "
            "Verificá la API key y los límites del plan."
        )

    print(f"Descarga finalizada. Días válidos obtenidos: {len(records)}")

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    return df

if __name__ == "__main__":
    df = fetch_historical_weather()
    df.to_csv("data/raw/weather_history_180d.csv", index=False)
    print(df.head())
    print(df.tail())
