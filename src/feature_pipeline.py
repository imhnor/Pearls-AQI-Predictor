import os
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

LAT = 31.5204
LON = 74.3587

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data"
DATA_PATH.mkdir(exist_ok=True)

FILE = DATA_PATH / "raw_live_aqi.csv"


def fetch_live():
    air_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"

    air = requests.get(air_url).json()
    weather = requests.get(weather_url).json()

    p = air["list"][0]["components"]

    ts = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    return {
        "timestamp": ts,
        "aqi": air["list"][0]["main"]["aqi"],
        "pm2_5": p.get("pm2_5"),
        "pm10": p.get("pm10"),
        "co": p.get("co"),
        "no2": p.get("no2"),
        "o3": p.get("o3"),
        "so2": p.get("so2"),
        "nh3": p.get("nh3"),
        "temp": weather["main"]["temp"],
        "humidity": weather["main"]["humidity"],
        "pressure": weather["main"]["pressure"],
    }


def run():
    new_row = pd.DataFrame([fetch_live()])

    if FILE.exists():
        df = pd.read_csv(FILE)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        new_row["timestamp"] = pd.to_datetime(new_row["timestamp"], utc=True)

        df = pd.concat([df, new_row], ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp"])
    else:
        df = new_row

    df.to_csv(FILE, index=False)

    print("✅ Live data updated")
    print(df.tail(1))


if __name__ == "__main__":
    run()