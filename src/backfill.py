import pandas as pd
import hopsworks
from processor import standardize_data  # Ensure processor.py is in the same directory
from dotenv import load_dotenv
import os

# API Key setup
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

def backfill(file_path):
    print("--- Starting Backfill Pipeline (3-Day Lag Logic) ---")
    print("Loading Excel file...")
    df = pd.read_excel(file_path)
    
    # 1. Rename Mapping
    mapping = {
        'datetime': 'timestamp',
        'main.aqi': 'aqi',
        'components.co': 'co',
        'components.no': 'no',
        'components.no2': 'no2',
        'components.o3': 'o3',
        'components.so2': 'so2',
        'components.pm2_5': 'pm2_5',
        'components.pm10': 'pm10',
        'components.nh3': 'nh3'
    }
    df = df.rename(columns=mapping)
    
    # 2. Sort by time (CRITICAL for lag calculation)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 3. Create 3-Day Lag Features (t-1, t-2, t-3)
    # Using the shift() function to look back in time
    df['aqi_t1'] = df['aqi'].shift(1).fillna(0).astype('int64')
    df['aqi_t2'] = df['aqi'].shift(2).fillna(0).astype('int64')
    df['aqi_t3'] = df['aqi'].shift(3).fillna(0).astype('int64')
    
    # 4. Standardize (Setting is_training=True)
    print("Standardizing data and generating features...")
    df_clean = standardize_data(df, is_training=True)
    
    # Drop rows that are NaN due to shifting (the first 3 rows)
    # df_clean = df_clean.dropna() 
    
    # 5. Upload to Hopsworks (Version 3)
    print("Logging into Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    
    aqi_fg = fs.get_or_create_feature_group(
        name="aqi_data_lahore",
        version=3, 
        primary_key=["timestamp"]
    )
    
    print(f"Uploading {len(df_clean)} rows to Hopsworks version 3...")
    aqi_fg.insert(df_clean)
    
    print("--- Historical Backfill Complete! ---")
    print("Your Feature Store is now ready for 3-Day Forecasting models.")

if __name__ == "__main__":
    # Ensure this path is correct for your environment
    DATA_PATH = "lahore_complete_data.xlsx"
    backfill(DATA_PATH)