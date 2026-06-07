import pandas as pd

def build_features():
    df = pd.read_csv("data/raw.csv")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    # time features
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month

    # lag features
    df["aqi_lag1"] = df["aqi"].shift(1)
    df["aqi_change"] = df["aqi"].diff()
    df["aqi_roll3"] = df["aqi"].rolling(3).mean()

    # IMPORTANT: fill instead of drop (prevents empty dataset)
    df = df.fillna(method="bfill").fillna(method="ffill")

    df.to_csv("data/features.csv", index=False)

    print("Features created:", len(df))

if __name__ == "__main__":
    build_features()