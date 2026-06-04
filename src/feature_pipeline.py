# What this file does (project-aligned)
# 1.Fetches AQI and pollutant data from OpenWeather Air Pollution API.
# 2.Fetches weather data from OpenWeather Weather API.
# 3.Generates time features: hour, day, month, weekday.
# 4.Appends rows to data/raw_aqi.csv instead of overwriting.
# 5.Prevents duplicate timestamps.
# 6.Handles optional rain/snow fields safely.
# 7.Uses UTC timestamps rounded to the hour so hourly scheduling is deterministic.

import os
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

# =========================
# ENV SETUP
# =========================
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT = os.getenv("LATITUDE")
LON = os.getenv("LONGITUDE")

if not API_KEY:
    raise ValueError("Missing OPENWEATHER_API_KEY in .env")

# =========================
# PATH SETUP
# =========================
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data"
DATA_PATH.mkdir(exist_ok=True)

FILE_PATH = DATA_PATH / "raw_aqi.csv"

# =========================
# API ENDPOINTS
# =========================
AIR_URL = (
    f"https://api.openweathermap.org/data/2.5/air_pollution"
    f"?lat={LAT}&lon={LON}&appid={API_KEY}"
)

WEATHER_URL = (
    f"https://api.openweathermap.org/data/2.5/weather"
    f"?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
)

# =========================
# FETCH AQI DATA
# =========================
try:
    air_res = requests.get(AIR_URL, timeout=30)
    air_res.raise_for_status()
    air_json = air_res.json()
except Exception as e:
    raise RuntimeError(f"AQI API failed: {e}")

pollution = air_json["list"][0]

# =========================
# FETCH WEATHER DATA
# =========================
try:
    weather_res = requests.get(WEATHER_URL, timeout=30)
    weather_res.raise_for_status()
    weather_json = weather_res.json()
except Exception as e:
    raise RuntimeError(f"Weather API failed: {e}")

# =========================
# TIMESTAMP (CONSISTENT HOURLY)
# =========================
timestamp = datetime.now(timezone.utc).replace(
    minute=0,
    second=0,
    microsecond=0
)

# =========================
# FEATURE RECORD (RAW LAYER)
# =========================
record = {
    # time
    "timestamp": timestamp,

    # AQI components
    "aqi": pollution["main"]["aqi"],
    "pm2_5": pollution["components"].get("pm2_5"),
    "pm10": pollution["components"].get("pm10"),
    "co": pollution["components"].get("co"),
    "no": pollution["components"].get("no"),
    "no2": pollution["components"].get("no2"),
    "o3": pollution["components"].get("o3"),
    "so2": pollution["components"].get("so2"),
    "nh3": pollution["components"].get("nh3"),

    # weather
    "temperature": weather_json["main"].get("temp"),
    "humidity": weather_json["main"].get("humidity"),
    "pressure": weather_json["main"].get("pressure"),
    "wind_speed": weather_json["wind"].get("speed"),
    "visibility": weather_json.get("visibility"),

    "clouds": weather_json.get("clouds", {}).get("all", 0),
    "rain_1h": weather_json.get("rain", {}).get("1h", 0),
    "snow_1h": weather_json.get("snow", {}).get("1h", 0),

    # time features
    "hour": timestamp.hour,
    "day": timestamp.day,
    "month": timestamp.month,
    "weekday": timestamp.weekday(),
}

df_new = pd.DataFrame([record])

# =========================
# LOAD EXISTING DATA
# =========================
if FILE_PATH.exists():
    df_old = pd.read_csv(FILE_PATH)

    # FIX TYPE CONSISTENCY
    df_old["timestamp"] = pd.to_datetime(df_old["timestamp"], utc=True, errors="coerce")
    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], utc=True, errors="coerce")

    df = pd.concat([df_old, df_new], ignore_index=True)

    # CLEAN
    df = df.dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp")

else:
    df = df_new

# =========================
# SAVE
# =========================
df.to_csv(FILE_PATH, index=False)

# =========================
# OUTPUT
# =========================
print("\n✅ Feature pipeline executed successfully")
print(df.tail(1))
print(f"\n📁 Saved to: {FILE_PATH}")