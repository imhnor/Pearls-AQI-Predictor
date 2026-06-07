import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime, timedelta

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌍",
    layout="wide"
)

# --------------------------------------------------
# Custom Styling
# --------------------------------------------------

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.big-title {
    font-size: 42px;
    font-weight: 700;
    color: #1f2937;
}

.subtitle {
    font-size: 18px;
    color: #6b7280;
    margin-bottom: 20px;
}

.metric-card {
    background-color: #f8fafc;
    border-radius: 12px;
    padding: 15px;
    border: 1px solid #e5e7eb;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Loading Screen
# --------------------------------------------------

loading = st.empty()

with loading.container():

    st.markdown(
        "<div class='big-title'>🌍 Pearls AQI Predictor</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>Loading model and environmental data...</div>",
        unsafe_allow_html=True
    )

    progress = st.progress(0)

    progress.progress(20)

    model = tf.keras.models.load_model(
        "models/aqi_lstm_model.keras",
        compile=False
    )

    progress.progress(60)

    scaler = joblib.load("models/scaler.pkl")

    progress.progress(80)

    df = pd.read_csv("data/features.csv").dropna()

    progress.progress(100)

loading.empty()

# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    "<div class='big-title'>🌍 Pearls AQI Predictor</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>AI-powered PM2.5 Forecasting Dashboard</div>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# AQI Category
# --------------------------------------------------

def aqi_category(pm25):

    if pm25 <= 12:
        return "🟢 Good"

    elif pm25 <= 35:
        return "🟡 Moderate"

    elif pm25 <= 55:
        return "🟠 Sensitive Groups"

    elif pm25 <= 150:
        return "🔴 Unhealthy"

    return "⚫ Hazardous"

# --------------------------------------------------
# Forecast Function
# --------------------------------------------------

FEATURES = [
    "co","no","no2","o3",
    "pm10","so2","nh3",
    "hour","day","month"
]

TIME_STEPS = 24

sequence = df[FEATURES].values[-TIME_STEPS:]
sequence_scaled = scaler.transform(sequence)

def forecast_3_days():

    predictions = []

    current = sequence_scaled.copy()

    for i in range(3):

        pred = model.predict(
            np.expand_dims(current, axis=0),
            verbose=0
        )[0][0]

        date = datetime.now() + timedelta(days=i+1)

        predictions.append({
            "Date": date.strftime("%d %b %Y"),
            "PM2.5": round(float(pred), 2),
            "Status": aqi_category(pred)
        })

        new_row = current[-1].copy()

        new_row[-3] = date.hour
        new_row[-2] = date.day
        new_row[-1] = date.month

        current = np.vstack([current[1:], new_row])

    return pd.DataFrame(predictions)

# --------------------------------------------------
# Main Action
# --------------------------------------------------

if st.button("Generate 3-Day Forecast", use_container_width=True):

    with st.spinner("Running AQI forecast model..."):

        forecast_df = forecast_3_days()

    st.success("Forecast generated successfully")

    # KPI Cards

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Day 1 PM2.5",
            forecast_df.iloc[0]["PM2.5"]
        )

    with c2:
        st.metric(
            "Day 2 PM2.5",
            forecast_df.iloc[1]["PM2.5"]
        )

    with c3:
        st.metric(
            "Day 3 PM2.5",
            forecast_df.iloc[2]["PM2.5"]
        )

    st.divider()

    # Forecast Table

    st.subheader("3-Day Forecast")

    st.dataframe(
        forecast_df,
        use_container_width=True,
        hide_index=True
    )

    # Trend Chart

    st.subheader("Forecast Trend")

    chart_df = forecast_df.set_index("Date")[["PM2.5"]]

    st.line_chart(chart_df)

    # AQI Status

    st.subheader("Air Quality Assessment")

    for _, row in forecast_df.iterrows():

        st.info(
            f"{row['Date']} → PM2.5: {row['PM2.5']} | {row['Status']}"
        )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Pearls AQI Predictor • LSTM Deep Learning Forecast Model"
)