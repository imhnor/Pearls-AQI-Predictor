import hopsworks
import pandas as pd
import joblib
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Input
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from processor import standardize_data
from dotenv import load_dotenv
from hopsworks_common.client.exceptions import ModelRegistryException

load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
def create_gru_model(input_shape):
    model = Sequential([
        Input(shape=(1, input_shape)),
        GRU(64, return_sequences=False),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_register_model():
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    
    print("Fetching data from Hopsworks version 3...")
    fg = fs.get_feature_group("aqi_data_lahore", version=3) 
    df = fg.read()
    
    df = standardize_data(df)
    df = df.dropna()
    
    X = df.drop(['aqi', 'timestamp'], axis=1)
    y = df['aqi']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Classical Models
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    best_r2 = -1
    best_model_name = ""
    best_model_obj = None
    is_deep_learning = False

    # Train Classical
    for name, model in models.items():
        model.fit(X_train, y_train)
        score = r2_score(y_test, model.predict(X_test))
        print(f"{name} R2 Score: {score:.4f}")
        if score > best_r2:
            best_r2, best_model_name, best_model_obj = score, name, model

    # 2. Train GRU
    print("\nTraining GRU (Deep Learning)...")
    X_train_gru = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_gru = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    gru_model = create_gru_model(X_train.shape[1])
    gru_model.fit(X_train_gru, y_train, epochs=20, batch_size=32, verbose=0)
    
    gru_preds = gru_model.predict(X_test_gru)
    gru_score = r2_score(y_test, gru_preds)
    print(f"GRU R2 Score: {gru_score:.4f}")
    
    if gru_score > best_r2:
        best_r2, best_model_name, best_model_obj = gru_score, "GRU", gru_model
        is_deep_learning = True

    # 3. Save & Register
    save_dir = f"aqi_model_{best_model_name.lower()}"
    os.makedirs(save_dir, exist_ok=True)
    
    if is_deep_learning:
        best_model_obj.save(f"{save_dir}/model.keras")
    else:
        joblib.dump(best_model_obj, f"{save_dir}/model.pkl")

    # Registy Logic (Cleaned Up)
    mr = project.get_model_registry()
    model_name = f"aqi_predictor_{best_model_name.lower()}"
    
    # Get all existing versions to prevent collision
    all_models = mr.get_models(name=model_name)
    
    if all_models:
        next_version = max([m.version for m in all_models]) + 1
    else:
        next_version = 1
    
    print(f"Registering model: {model_name}, Version: {next_version}")
    
    # Save to Registry
    aqi_model = mr.python.create_model(name=model_name, version=next_version)
    aqi_model.save(save_dir)
    print(f"Version {next_version}: Model {best_model_name} registered successfully!")

if __name__ == "__main__":
    train_and_register_model()