## Dataset – Iteration 1 (Baseline)

### Data source
- Provider: WeatherAPI.com
- Endpoint: Historical daily weather
- Location: Buenos Aires, Argentina
- Time range: ~180 days
- Frequency: Daily aggregates

### Retrieved variables
The following variables were collected in the initial dataset:

- `date`: Observation date
- `temp_max_c`: Daily maximum temperature (°C)
- `temp_min_c`: Daily minimum temperature (°C)
- `humidity_avg`: Average daily humidity (%)
- `precip_mm`: Total daily precipitation (mm)
- `pressure_mb`: Average daily atmospheric pressure (mb)

### Data generation
The dataset is generated programmatically using the script:
src/fetch_weather.py

and stored as:
data/raw/weather_history_180d.csv

The raw data is reproducible and can be regenerated at any time using the API.
