---
# Weather DS Portfolio

## Objective
Predict **daily precipitation (mm)** using physically motivated meteorological variables obtained from a production-grade weather API.

The project is designed to be **fully reproducible**, **config-driven**, **test-covered**, and **pipeline-first**, combining exploratory notebooks with production-ready Python modules.

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

## Pipeline architecture

![Pipeline architecture](assets/pipeline_architecture.png)

---

## Feature Groups

### 1. Base Physical Variables
- Temperature (min / max / mean)
- Daily temperature range
- Relative humidity
- Surface pressure
- Wind speed

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
│   ├── config.py            # Single source of truth (paths, versions, params)
│   ├── fetch_open_meteo_daily.py
│   ├── build_dataset.py
│   ├── features.py
│   └── modeling_regression.py
│
├── data/
│   ├── raw/                 # Cached raw API downloads
│   ├── processed/           # Feature-engineered datasets (versioned)
│   └── results/             # Metrics, predictions, feature sets
│
├── notebooks/               # Exploratory, feature, models analysis and comparison
│   ├── 01_eda.ipynb         
│   ├── 02_eda_precipitation.ipynb 
│   ├── 03_feature_engineering.ipynb 
│   ├── 04_base_model.ipynb 
│   ├── 05_xgboost.ipynb 
│   ├── 06_neural_networks.ipynb 
│   └── 07_model_comparison.ipynb 
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

## Notebook execution order

The notebooks are intended to be read and executed in the following order:

1. 01_eda.ipynb – General exploratory analysis
2. 02_eda_precipitation.ipynb – Detailed precipitation analysis
3. 03_feature_engineering.ipynb – Feature construction and validation
4. 04_base_model.ipynb – Baseline regression model
5. 05_xgboost.ipynb – Gradient boosting experiments 
6. 06_neural_networks.ipynb – Neural network experiments
7. 07_model_comparison.ipynb – Final model comparison and selection

---

## Pipeline Overview

The project is executed end-to-end via a single reproducible pipeline:

```
./run_pipeline.sh
```

Pipeline steps:
1. **Fetch raw data**
   - Downloads data from Open-Meteo
   - Uses local cache if raw dataset already exists (API-safe)
2. **Build processed dataset**
   - Feature engineering
   - Versioned dataset creation
3. **Run modeling**
   - Multiple regression experiments
   - Metrics, predictions, and feature sets saved

Generated artifacts:
- Metrics: `metrics_regression_vXXX.csv`
- Predictions: `predictions_best_vXXX.csv`
- Feature sets: `feature_sets_vXXX.json`

---

## Modeling
- Models evaluated:
  - Linear Regression
  - Ridge
  - Lasso
  - Random Forest
- Time-based train / test split (no leakage)
- Metrics:
  - MAE
  - RMSE
  - R²

---

## Model Selection Rationale

Although EXP6_full and EXP5_base_memory models showed similar RMSE performance, EXP5_base_memory was selected as the final model due to its greater parsimony, robustness, and interpretability. By using a more streamlined feature set focused on base physical variables and atmospheric memory, this model offers easier interpretation and potentially better generalization, aligning with project goals for a stable and maintainable baseline.

---

## Configuration & Versioning

All paths, dataset names, versions, and modeling parameters are centralized in:

```
src/config.py
```

This guarantees:
- No hardcoded paths
- Consistent versioning across scripts and tests
- Easy reproducibility and auditing

---

## Testing Strategy

The project includes **automated unit and integration tests** covering:

- Feature engineering correctness
- Dataset construction integrity
- Output artifact creation
- Full end-to-end pipeline execution

Run all tests with:
```
pytest -q
```

All tests are aligned with `config.py` (no hardcoded paths).

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

A fresh clone of this repository can:
1. Install dependencies
2. Run the pipeline
3. Reproduce datasets, metrics, and predictions
4. Pass all automated tests


---