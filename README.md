# Climate ML Project – Physics‑Driven Weather Modeling

## 1. 🧠 Project Overview

This project explores machine learning approaches for climate and weather analysis using **rich atmospheric datasets with physical variables** (e.g., Oikolab).
The objective is not to replicate a simple weather forecast, but to **model, understand, and predict complex atmospheric behavior** using data‑driven techniques grounded in physics.

Este proyecto explora enfoques de machine learning para el análisis climático y meteorológico utilizando datasets ricos en variables físicas atmosféricas.
El objetivo no es reproducir un pronóstico básico del clima, sino **modelar, comprender y predecir comportamientos atmosféricos complejos** con técnicas de ML apoyadas en fundamentos físicos.

---

## 2. 🎯 Motivation & Problem Framing

Many introductory ML projects focus on predicting temperature or rainfall using limited features.
Here, the goal is to **go beyond surface variables** and experiment with:

* deep atmospheric parameters
* energy and convection‑related variables
* multivariate temporal dynamics

En lugar de predecir únicamente temperatura o precipitación, este proyecto busca trabajar con **variables atmosféricas avanzadas**, explorando señales físicas más profundas y su relación con fenómenos climáticos.

---

## 3. 📦 Dataset Source

### Primary Data Source: **Oikolab Climate Parameters Dataset**

Oikolab provides historical and forecasted climate data with **high‑resolution physical parameters** derived from reanalysis and numerical weather prediction models.

📄 Documentation:
[https://docs.oikolab.com/parameters/](https://docs.oikolab.com/parameters/)

This dataset offers:

* global coverage
* long historical depth
* hundreds of atmospheric variables

Este dataset provee cobertura global, profundidad histórica extensa y una gran variedad de variables físicas atmosféricas.

---

## 4. 🌦 Available Variables (High‑Level)

The project can leverage multiple groups of parameters, including:

### Core meteorological variables

* temperature
* surface pressure
* wind speed and direction
* relative humidity
* precipitation

### Atmospheric structure & dynamics

* boundary layer height
* cloud base height
* cloud cover (low / mid / high)
* zero degree level

### Energy & convection indicators

* convective available potential energy (CAPE)
* convective inhibition (CIN)
* surface latent heat flux
* evaporation

### Moisture & radiation

* total column water vapour
* downward solar / UV radiation
* albedo

Estas variables permiten capturar procesos físicos complejos asociados a tormentas, convección, lluvias intensas y otros fenómenos relevantes.

---

## 5. 🧰 Project Structure

```
Climate‑ML‑Portfolio/
├── data/
│   ├── raw/          # Raw downloaded datasets
│   ├── processed/    # Cleaned and feature‑engineered data
│   └── external/     # External reference datasets
├── src/
│   ├── data_ingest.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── utils.py
├── notebooks/        # EDA and experiments
├── requirements.txt
├── .env
└── README.md
```

* **data/raw/**: original datasets, never modified
* **data/processed/**: ML‑ready datasets
* **src/**: reusable Python scripts
* **notebooks/**: exploratory analysis and reporting

---

## 6. 🚀 Environment Setup

### Python

Python **3.12+** is recommended.

### Environment creation (Conda recommended)

```bash
conda create -n climate-ml python=3.12
conda activate climate-ml
```

### Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Typical dependencies include:

* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* requests
* python-dotenv

---

## 7. 📁 Data Ingestion Strategy

The ingestion pipeline is designed to:

1. Download or load raw climate data
2. Normalize timestamps and spatial references
3. Control temporal resolution (e.g., daily)
4. Store raw and processed data separately

El objetivo es mantener un pipeline reproducible y trazable, evitando fugas de información y facilitando la experimentación.

---

## 8. 📊 Exploratory Data Analysis (EDA)

EDA notebooks focus on:

* temporal patterns and seasonality
* distribution of physical variables
* correlations and multicollinearity
* detection of extreme values or anomalies

El análisis exploratorio es clave para entender la señal física antes de modelar.

---

## 9. 🧪 Modeling Approaches

Depending on the experiment, the project may include:

### Regression tasks

* forecasting continuous variables (e.g., precipitation, CAPE)

### Classification tasks

* detection of anomalous or extreme atmospheric conditions

### Unsupervised learning

* clustering of atmospheric regimes
* dimensionality reduction

Los modelos se evalúan respetando la estructura temporal de los datos.

---

## 10. 📈 Evaluation Metrics

### Regression

* RMSE
* MAE
* R²

### Classification

* Precision / Recall
* F1‑score
* ROC‑AUC / PR‑AUC

Las métricas se eligen según el problema y el desbalance de clases.

