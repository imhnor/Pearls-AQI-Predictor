# import time
from fetch_data import fetch_aqi_openweather

# # simulate historical data collection
# for i in range(500):   # 500 historical points
#     fetch_aqi_openweather()
#     print(f"Collected {i+1}/500")
#     time.sleep(1)  # small delay (or remove)


from datetime import datetime, timedelta
import time

for i in range(360):
    date = datetime.now() - timedelta(days=i)

    timestamp = int(date.timestamp())

    fetch_aqi_openweather(timestamp=timestamp)

    print(f"Fetched day {i+1}/360")

    time.sleep(1)