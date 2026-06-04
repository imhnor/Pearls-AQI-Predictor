import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

LAT = 31.5204
LON = 74.3587

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data"
DATA_PATH.mkdir(exist_ok=True)

OUTPUT_FILE = DATA_PATH / "raw_historical_aqi.csv"


def fetch_day(start, end):
    url = (
        f"https://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={LAT}&lon={LON}&start={int(start.timestamp())}"
        f"&end={int(end.timestamp())}&appid={API_KEY}"
    )

    res = requests.get(url, timeout=20)

    if res.status_code != 200:
        print("API error:", res.text)
        return []

    data = res.json()
    records = []

    for item in data.get("list", []):
        c = item["components"]

        records.append({
            "timestamp": datetime.fromtimestamp(item["dt"], tz=timezone.utc),
            "aqi": item["main"]["aqi"],
            "pm2_5": c.get("pm2_5"),
            "pm10": c.get("pm10"),
            "co": c.get("co"),
            "no2": c.get("no2"),
            "o3": c.get("o3"),
            "so2": c.get("so2"),
            "nh3": c.get("nh3"),
        })

    return records


def build(days=360):
    all_data = []
    end = datetime.now(timezone.utc)

    print(f"Building dataset for {days} days...")

    for i in range(days):
        start = end - timedelta(days=i + 1)
        finish = end - timedelta(days=i)

        print("Fetching:", start.date())

        all_data.extend(fetch_day(start, finish))
        time.sleep(1)

    df = pd.DataFrame(all_data)

    df = df.drop_duplicates()
    df = df.sort_values("timestamp")

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ Dataset created:", len(df))
    print("Saved at:", OUTPUT_FILE)


if __name__ == "__main__":
    build(360)