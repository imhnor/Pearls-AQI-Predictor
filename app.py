import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("AQI Predictor")

model = joblib.load("models/aqi_model.pkl")

df = pd.read_csv("data/features.csv")

st.subheader("Latest Data")
st.write(df.tail())

latest = df.drop(columns=["aqi", "datetime"]).iloc[-1]

preds = []
current = latest.copy()

for _ in range(3):
    pred = model.predict([current])[0]
    preds.append(pred)
    current["aqi_lag1"] = pred

st.subheader("3-Day Forecast")
st.write(preds)

st.line_chart(preds)

if max(preds) > 150:
    st.error("Hazardous AQI Alert!")