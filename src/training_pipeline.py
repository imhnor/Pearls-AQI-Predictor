import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import joblib

# -----------------------------
# Convert data into sequences
# -----------------------------
def create_sequences(X, y, time_steps=24):
    Xs, ys = [], []

    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])

    return np.array(Xs), np.array(ys)


def train_lstm():
    df = pd.read_csv("data/features.csv").dropna()

    features = [
        "co", "no", "no2", "o3", "pm10", "so2", "nh3",
        "hour", "day", "month"
    ]

    X = df[features].values
    y = df["pm2_5"].values

    # -----------------------------
    # Scaling (IMPORTANT)
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # Sequence creation
    # -----------------------------
    TIME_STEPS = 24
    X_seq, y_seq = create_sequences(X_scaled, y, TIME_STEPS)

    # -----------------------------
    # Train-test split (time series safe)
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq,
        test_size=0.2,
        shuffle=False
    )

    # -----------------------------
    # LSTM Model
    # -----------------------------
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True,
                             input_shape=(TIME_STEPS, X.shape[1])),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.LSTM(32),

        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    model.summary()

    # -----------------------------
    # Training
    # -----------------------------
    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )

    # -----------------------------
    # Evaluation
    # -----------------------------
    loss, mae = model.evaluate(X_test, y_test)
    print("\nLSTM MAE:", mae)

    # -----------------------------
    # Save model
    # -----------------------------
    model.save("models/aqi_lstm_model.keras")
    joblib.dump(scaler, "models/scaler.pkl")

    print("LSTM model saved successfully")


if __name__ == "__main__":
    train_lstm()