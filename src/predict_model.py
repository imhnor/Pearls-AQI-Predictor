import numpy as np
import tensorflow as tf
import joblib
import pandas as pd

model = tf.keras.models.load_model("models/aqi_lstm_model.keras")
scaler = joblib.load("models/scaler.pkl")

df = pd.read_csv("data/features.csv").dropna()

features = ["co","no","no2","o3","pm10","so2","nh3","hour","day","month"]

X = df[features].values[-24:]  # last 24 steps

X_scaled = scaler.transform(X)
X_input = np.expand_dims(X_scaled, axis=0)

pred = model.predict(X_input)

print("Predicted AQI (pm2.5):", pred[0][0])