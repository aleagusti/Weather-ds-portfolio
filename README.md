Objetive

Predict daily precipitation using physical meteorological variables from an open, production-grade weather API.

Fuente de datos:
Open-Meteo Historical Weather API

Unidad temporal:
Daily

Ubicación:
Miami, FL (primera iteración)

Target:
precipitation_sum (mm)

Features:
temperature (mean / max / min)
relative humidity
surface pressure
wind speed
cloud cover
lagged precipitation
rolling aggregates

Weather-ds-portfolio/
│
├── src/
│   ├── fetch_open_meteo_daily.py
│   ├── features.py
│   └── utils.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
│
├── requirements.txt
└── README.md

## Environment setup

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt