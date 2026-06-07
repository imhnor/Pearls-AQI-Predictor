import pandas as pd
import joblib

def predict():
    model = joblib.load("models/aqi_model.pkl")

    df = pd.read_csv("data/features.csv")

    latest = df.iloc[-1]

    row = latest.copy()

    predictions = []

    for _ in range(3):
        input_data = [[
            row["co"], row["no"], row["no2"], row["o3"],
            row["pm10"], row["so2"], row["nh3"],
            row["hour"], row["day"], row["month"],
            row["aqi_lag1"], row["aqi_change"], row["aqi_roll3"]
        ]]

        pred = model.predict(input_data)[0]
        predictions.append(pred)

        # update lag for next step
        row["aqi_lag1"] = pred

    print("3-step PM2.5 forecast:", predictions)

if __name__ == "__main__":
    predict()