import os
import requests
import pandas as pd
import hopsworks
from dotenv import load_dotenv
from processor import standardize_data 

# 1. Load Environment Variables
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
# HOPSWORKS_API_KEY = hopsworks_key
# OPENWEATHER_API_KEY =openweather_key
def fetch_aqi_data(city_name):
    # Step 1: Get coordinates
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={OPENWEATHER_API_KEY}"
    geo_response = requests.get(geo_url).json()
    
    # Validate response is a list with at least one element
    if not isinstance(geo_response, list) or len(geo_response) == 0:
        print(f"City '{city_name}' not found or invalid API response: {geo_response}")
        return None
    
    lat = geo_response[0]['lat']
    lon = geo_response[0]['lon']
    # Step 2: Get Air Pollution data
    aqi_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
    aqi_response = requests.get(aqi_url).json()
    return aqi_response

def fetch_and_process():
        raw_data = fetch_aqi_data("Lahore")
        if not raw_data or 'list' not in raw_data: return None
    
        # Current data
        pollution = raw_data['list'][0]
        data_dict = {'timestamp': pd.Timestamp.now(), 'aqi': pollution['main']['aqi'], **pollution['components']}
        df = pd.DataFrame([data_dict])
        
        # --- 3-DAY LAG LOGIC (Hopsworks se fetch karein) ---
        try:
            project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
            fs = project.get_feature_store()
            fg = fs.get_feature_group("aqi_data_lahore", version=3)
            
            # Pichle 3 records uthayein
            history = fg.read().sort_values('timestamp', ascending=False).head(3)
            
            # Data fill karein
            df['aqi_t1'] = history.iloc[0]['aqi'] if len(history) > 0 else 0
            df['aqi_t2'] = history.iloc[1]['aqi'] if len(history) > 1 else 0
            df['aqi_t3'] = history.iloc[2]['aqi'] if len(history) > 2 else 0
        except:
            # Agar pehli baar chal raha hai toh 0 rakhein
            df['aqi_t1'], df['aqi_t2'], df['aqi_t3'] = 0, 0, 0
        
        # Standardize
        df_clean = standardize_data(df)
        return df_clean

def save_to_hopsworks(df):
    try:
        # Login
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        fs = project.get_feature_store()
        
        # Get or Create Feature Group
        aqi_fg = fs.get_or_create_feature_group(
            name="aqi_data_lahore",
            version=3,
            primary_key=["timestamp"]
        )
        
        # Insert Data
        aqi_fg.insert(df)
        print("Data successfully pushed to Hopsworks!")
    except Exception as e:
        print(f"Error pushing to Hopsworks: {e}")

if __name__ == "__main__":
    print("Running Pipeline...")
    df_clean = fetch_and_process()
    
    if df_clean is not None:
        save_to_hopsworks(df_clean)
    else:
        print("Pipeline aborted.")