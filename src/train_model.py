import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib

def train_dl():
    df = pd.read_csv("data/features.csv").dropna()

    X = df[
        ["co", "no", "no2", "o3", "pm10", "so2", "nh3",
         "hour", "day", "month",
         "aqi_lag1", "aqi_change", "aqi_roll3"]
    ].values

    y = df["pm2_5"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    model.fit(X_train, y_train, epochs=30, batch_size=8, verbose=1)

    loss, mae = model.evaluate(X_test, y_test)

    print("DL MAE:", mae)

    model.save("models/dl_model.h5")
    joblib.dump(scaler, "models/scaler.pkl")

    print("Deep Learning model saved")

if __name__ == "__main__":
    train_dl()