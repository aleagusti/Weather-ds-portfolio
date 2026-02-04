Objetive

Predict daily precipitation using physical meteorological variables from an open, production-grade weather API.

Data source:
Open-Meteo Historical Weather API

Frequency unit:
Daily

Location:
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
# Weather DS Portfolio

## Objective
Predict **daily precipitation (mm)** using physically motivated meteorological variables obtained from a production-grade weather API.  
The project is designed to be **fully reproducible**, **testable**, and **pipeline-driven**, combining exploratory notebooks with production-ready Python modules.

---

## Data Source
- **API:** Open-Meteo Historical Weather API
- **Temporal resolution:** Daily
- **Location:** Miami, FL (initial iteration)
- **Time span:** 1990 – present

---

## Target Variable
- `precipitation_sum` (mm)

---

## Feature Groups

### 1. Base Physical Variables
- Temperature (min / max / mean)
- Daily temperature range
- Wind speed
- Solar radiation
- Weather code

### 2. Seasonal Encoding
- Day-of-year encoded using sine / cosine (cyclical encoding)
- Physically motivated proxy for annual solar forcing

### 3. Atmospheric Memory
- Lagged precipitation (1, 3, 7 days)
- Rolling precipitation accumulations (3, 7 days)

---

## Project Structure

```
Weather-ds-portfolio/
│
├── src/                     # Production code
│   ├── __init__.py
│   ├── fetch_open_meteo_daily.py
│   ├── build_dataset.py
│   ├── features.py
│   ├── modeling_regression.py
│   └── utils.py
│
├── data/
│   ├── raw/                 # Raw API downloads
│   ├── processed/           # Feature-engineered datasets (versioned)
│   └── results/             # Metrics, predictions, feature sets
│
├── notebooks/               # Exploratory & explanatory analysis
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
│
├── tests/                   # Unit + integration tests
│   ├── __init__.py
│   ├── test_features.py
│   ├── test_build_dataset.py
│   ├── test_results.py
│   └── test_pipeline.py
│
├── pytest.ini
├── run_pipeline.sh
├── requirements.txt
└── README.md
```

---

## Pipeline Overview

The project is executed end-to-end via a single reproducible pipeline:

```
./run_pipeline.sh
```

This pipeline performs:
1. Data ingestion from Open-Meteo API
2. Dataset construction and feature engineering
3. Regression modeling experiments
4. Saving versioned outputs:
   - Metrics (`metrics_regression_vXXX.csv`)
   - Predictions (`predictions_best_vXXX.csv`)
   - Feature sets (`feature_sets_vXXX.json`)

---

## Modeling
- Models evaluated: Linear Regression, Ridge, Lasso, Random Forest
- Time-based train / test split (no leakage)
- Metrics:
  - MAE
  - RMSE
  - R²

---

## Testing Strategy

The project includes **automated tests** to guarantee reproducibility:

- Feature correctness (lags, rolling windows, cyclical encoding)
- Dataset construction integrity
- Output artifact creation
- Full pipeline execution (end-to-end)

Run all tests with:
```
pytest -q
```

---

## Environment Setup

Using Conda (recommended):

```
conda create -n weather-ds python=3.12
conda activate weather-ds
pip install -r requirements.txt
```

---

## Reproducibility Guarantee
A fresh clone of the repository can:
1. Install dependencies
2. Run the pipeline
3. Reproduce datasets, metrics, and predictions
4. Pass all automated tests

---

## Status
- Pipeline: ✅ Stable
- Tests: ✅ Passing
- Dataset versioning: ✅ Implemented
- Next steps:
  - Extend to additional locations
  - Add non-linear / deep learning models
  - Probabilistic precipitation modeling