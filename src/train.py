import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def train():
    df = pd.read_csv("data/features.csv")
    df = df.dropna()

    X = df[
        ["co", "no", "no2", "o3", "pm10", "so2", "nh3",
         "hour", "day", "month",
         "aqi_lag1", "aqi_change", "aqi_roll3"]
    ]

    y = df["pm2_5"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("MAE:", mean_absolute_error(y_test, preds))
    print("R2:", r2_score(y_test, preds))

    joblib.dump(model, "models/aqi_model.pkl")

    print("Model saved successfully")

if __name__ == "__main__":
    train()