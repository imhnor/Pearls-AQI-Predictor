import pandas as pd

# Updated Schema with Lag Features
SCHEMA_TYPES = {
    'aqi': 'int64',
    'co': 'float64', 'no': 'float64', 'no2': 'float64',
    'o3': 'float64', 'so2': 'float64', 'pm2_5': 'float64',
    'pm10': 'float64', 'nh3': 'float64',
    'hour': 'int64', 'day': 'int64', 'month': 'int64',
    'day_of_week': 'int64', 'aqi_change_rate': 'int64',
    # New Lag Features
    'aqi_t1': 'int64', 'aqi_t2': 'int64', 'aqi_t3': 'int64'
}

def standardize_data(df, is_training=False):
    # 1. Ensure Timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 2. Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # 3. AQI Change Rate
    df['aqi_change_rate'] = df['aqi'].diff().fillna(0).astype('int64')
    
    # 4. Lag Features (Only for training/backfilling)
    if is_training:
        df['aqi_t1'] = df['aqi'].shift(1).fillna(0).astype('int64')
        df['aqi_t2'] = df['aqi'].shift(2).fillna(0).astype('int64')
        df['aqi_t3'] = df['aqi'].shift(3).fillna(0).astype('int64')
    else:
        # In live prediction, these will be filled from Feature Store in your app.py
        # For now, we initialize them to 0 if not present
        if 'aqi_t1' not in df.columns: df['aqi_t1'] = 0
        if 'aqi_t2' not in df.columns: df['aqi_t2'] = 0
        if 'aqi_t3' not in df.columns: df['aqi_t3'] = 0

    # 5. Type Casting
    for col, dtype in SCHEMA_TYPES.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
            
    # 6. Final Selection
    cols = ['timestamp'] + list(SCHEMA_TYPES.keys())
    return df[cols]