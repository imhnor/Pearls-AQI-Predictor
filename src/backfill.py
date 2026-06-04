import os
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data"

INPUT = DATA_PATH / "raw_historical_aqi.csv"
OUTPUT = DATA_PATH / "training_data.csv"


def build_features(df):
    df = df.sort_values("timestamp")

    # time features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["weekday"] = df["timestamp"].dt.weekday

    # lag features
    df["lag_1"] = df["aqi"].shift(1)
    df["lag_3"] = df["aqi"].shift(3)
    df["lag_24"] = df["aqi"].shift(24)

    # rolling
    df["mean_24"] = df["aqi"].rolling(24).mean()
    df["std_24"] = df["aqi"].rolling(24).std()

    # target (forecast next 72 hours)
    df["target"] = df["aqi"].shift(-72)

    df = df.dropna()

    return df


def run():
    if not INPUT.exists():
        print("No dataset found. Run dataset_builder first.")
        return

    df = pd.read_csv(INPUT)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df_ml = build_features(df)

    df_ml.to_csv(OUTPUT, index=False)

    print("✅ Backfill completed")
    print("Rows:", len(df_ml))
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    run()