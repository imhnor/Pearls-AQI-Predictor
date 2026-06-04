import os
import requests
import pandas as pd
import datetime
import time
from pathlib import Path
from dotenv import load_dotenv

# =========================
# ENV
# =========================
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not API_KEY:
    raise ValueError("Missing OPENWEATHER_API_KEY")

# =========================
# PROJECT PATH
# =========================
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data"
DATA_PATH.mkdir(exist_ok=True)

OUTPUT_FILE = DATA_PATH / "raw_historical_aqi.csv"

# =========================
# CONFIG
# =========================
LAT = 31.5204
LON = 74.3587


# =========================
# FETCH FUNCTION
# =========================
def fetch_historical_aqi(api_key, lat, lon, start_date, end_date):

    start_ts = int(start_date.timestamp())
    end_ts = int(end_date.timestamp())

    url = (
        f"https://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={lat}&lon={lon}&start={start_ts}&end={end_ts}&appid={api_key}"
    )

    print("Fetching historical AQI data...")

    try:
        res = requests.get(url, timeout=20)
        res.raise_for_status()
        data = res.json()

        records = []

        for item in data.get("list", []):
            comp = item.get("components", {})

            records.append({
                "timestamp": pd.to_datetime(item["dt"], unit="s", utc=True),

                "aqi": item.get("main", {}).get("aqi"),
                "pm2_5": comp.get("pm2_5"),
                "pm10": comp.get("pm10"),
                "co": comp.get("co"),
                "no2": comp.get("no2"),
                "o3": comp.get("o3"),
                "so2": comp.get("so2"),
                "nh3": comp.get("nh3"),
            })

        return pd.DataFrame(records)

    except Exception as e:
        print("Error:", e)
        return pd.DataFrame()


# =========================
# MAIN PIPELINE
# =========================
if __name__ == "__main__":

    end_date = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
    start_date = end_date - datetime.timedelta(days=30)  # SAFE LIMIT (API friendly)

    df = fetch_historical_aqi(API_KEY, LAT, LON, start_date, end_date)

    if df.empty:
        print("No data fetched from API")
        exit()

    # CLEAN
    df = df.drop_duplicates()
    df = df.sort_values("timestamp")

    # SAVE INSIDE PROJECT STRUCTURE
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ Historical dataset created successfully")
    print(f"Rows: {len(df)}")
    print(f"Saved at: {OUTPUT_FILE}")