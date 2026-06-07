import pandas as pd
from fetch_data import fetch_aqi_openweather
from datetime import datetime

def build_features():
    data = []

    for _ in range(200):  # small dataset for submission
        row = fetch_aqi_openweather()

        row["hour"] = datetime.now().hour
        row["day"] = datetime.now().day
        row["month"] = datetime.now().month

        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv("data/features.csv", index=False)

    print("Feature dataset created")


if __name__ == "__main__":
    build_features()