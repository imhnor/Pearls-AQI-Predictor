import pandas as pd
import joblib

def predict_next():
    model = joblib.load("models/aqi_model.pkl")

    df = pd.read_csv("data/features.csv")

    latest = df.drop(columns=["aqi", "datetime"]).iloc[-1]

    preds = []

    current = latest.copy()

    for _ in range(3):
        pred = model.predict([current])[0]
        preds.append(pred)

        current["aqi_lag1"] = pred

    print("Next 3-day AQI forecast:", preds)
    return preds

if __name__ == "__main__":
    predict_next()