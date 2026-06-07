import requests
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
CITY = "Lahore"

def get_coordinates(city):
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
    res = requests.get(url).json()
    return res[0]["lat"], res[0]["lon"]

def fetch_aqi_openweather():
    lat, lon = get_coordinates(CITY)

    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    res = requests.get(url).json()

    if "list" not in res:
        raise Exception(f"API Error: {res}")

    data = res["list"][0]

    record = {
        "datetime": datetime.now(),
        "aqi": data["main"]["aqi"],
        "co": data["components"]["co"],
        "no": data["components"]["no"],
        "no2": data["components"]["no2"],
        "o3": data["components"]["o3"],
        "pm2_5": data["components"]["pm2_5"],
        "pm10": data["components"]["pm10"],
        "so2": data["components"]["so2"],
        "nh3": data["components"]["nh3"],
    }

    os.makedirs("data", exist_ok=True)

    df = pd.DataFrame([record])

    file_path = "data/raw.csv"

    if os.path.exists(file_path):
        old = pd.read_csv(file_path)
        df = pd.concat([old, df], ignore_index=True)

    df.to_csv(file_path, index=False)

    print("OpenWeather AQ data saved")

if __name__ == "__main__":
    fetch_aqi_openweather()