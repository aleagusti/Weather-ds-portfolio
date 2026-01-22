# Extreme Weather Events Prediction – Florida

## 1. Problem Statement

The objective of this project is to predict the occurrence of extreme weather events
using only physical atmospheric variables. Official weather alerts are used exclusively
as ground-truth labels and are never included as predictive features.

El objetivo del proyecto es predecir la ocurrencia de eventos meteorológicos extremos
utilizando únicamente variables físicas de la atmósfera. Las alertas oficiales se usan
solo como etiquetas de validación y nunca como variables predictoras.

---

## 2. Scope

- Region: Miami, Florida (USA)
- Temporal resolution: Daily
- Historical window: ~5 years
- Initial task: Binary classification (event vs no event)

Región: Miami, Florida (EE.UU.)  
Resolución temporal: diaria  
Ventana histórica: ~5 años  
Tarea inicial: clasificación binaria (evento / no evento)

---

## 3. Data Sources

- WeatherAPI – Historical Weather Data
- WeatherAPI – Alerts API (labels only)

WeatherAPI – Datos históricos del clima  
WeatherAPI – API de alertas (solo para etiquetas)

---

## 4. Modeling Strategy

1. Binary classification: extreme event occurrence  
2. Probabilistic risk estimation  
3. (Future work) Time-to-event prediction  

1. Clasificación binaria: ocurrencia de evento extremo  
2. Estimación probabilística de riesgo  
3. (Trabajo futuro) Predicción del tiempo hasta el evento  

---

## 5. Methodological Principles

- No data leakage
- Physics-driven feature engineering
- Explicit assumptions and limitations
- Reproducible data pipeline

- Sin fuga de información
- Ingeniería de features basada en variables físicas
- Supuestos y limitaciones explícitas
- Pipeline reproducible
