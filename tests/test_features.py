import pandas as pd
import numpy as np
from src.features import (
    ensure_sorted_by_date,
    add_temperature_range,
    add_cyclical_dayofyear,
    add_precipitation_lags,
    add_precipitation_rolling,
)

def test_ensure_sorted_by_date():
    df = pd.DataFrame({
        "date": ["2021-01-02", "2021-01-01"],
        "x": [1, 2]
    })
    df_sorted = ensure_sorted_by_date(df, date_col="date")
    assert df_sorted["date"].iloc[0] < df_sorted["date"].iloc[1]

def test_temperature_range():
    df = pd.DataFrame({
        "temperature_2m_max": [20],
        "temperature_2m_min": [10]
    })
    df = add_temperature_range(df)
    assert "temp_range" in df.columns
    assert df["temp_range"].iloc[0] == 10

def test_cyclical_encoding():
    df = pd.DataFrame({"date": ["2021-06-01"]})
    df = add_cyclical_dayofyear(df, date_col="date")
    assert "doy_sin" in df.columns and "doy_cos" in df.columns

def test_precip_lags():
    df = pd.DataFrame({"precipitation_sum": [0, 1, 2, 3]})
    df = add_precipitation_lags(df, lags=[1])
    assert "precip_lag_1" in df.columns
    assert df["precip_lag_1"].isna().iloc[0]

def test_precip_rolling():
    df = pd.DataFrame({"precipitation_sum": [1, 2, 3, 4]})
    df = add_precipitation_rolling(df, windows=[2])
    assert "precip_roll_2" in df.columns
    assert df["precip_roll_2"].iloc[1] == 3