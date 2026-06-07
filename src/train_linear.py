import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def train():
    df = pd.read_csv("data/features.csv").dropna()

    features = [
        "co", "no", "no2", "o3", "pm10", "so2", "nh3",
        "hour", "day", "month",
        "aqi_lag1", "aqi_change", "aqi_roll3"
    ]

    X = df[features]
    y = df["pm2_5"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\n=== LINEAR REGRESSION RESULTS ===")
    print("MAE:", mean_absolute_error(y_test, preds))
    print("R2 :", r2_score(y_test, preds))

    joblib.dump(model, "models/aqi_model.pkl")

    print("\nModel saved successfully → models/aqi_model_linear.pkl")

if __name__ == "__main__":
    train()