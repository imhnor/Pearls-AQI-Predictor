import time
from fetch_data import fetch_aqi_openweather

# simulate historical data collection
for i in range(500):   # 500 historical points
    fetch_aqi_openweather()
    print(f"Collected {i+1}/500")
    time.sleep(1)  # small delay (or remove)
    