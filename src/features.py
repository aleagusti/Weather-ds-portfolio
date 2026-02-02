"""
Feature Engineering Utilities

Reusable feature construction functions for precipitation modeling.
All functions assume DAILY data indexed or sorted by date.

Design principles:
- No data leakage (only past information)
- Physically interpretable transformations
- Modular and composable feature blocks
"""

import numpy as np
import pandas as pd


# =========================
# Base utilities
# =========================

def ensure_sorted_by_date(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Ensure DataFrame is sorted by date.
    """
    return df.sort_values(date_col).reset_index(drop=True)


# =========================
# Base physical features 
# =========================

def add_temperature_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add daily temperature range (max - min).
    Physical meaning:
    - Proxy for atmospheric stability / cloudiness
    """
    df = df.copy()
    df["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]
    return df


# =========================
# Seasonality (cyclical encoding)
# =========================

def add_cyclical_dayofyear(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Encode annual seasonality using sine/cosine of day of year.
    Physical meaning:
    - Proxy for solar radiation / annual cycle
    """
    df = df.copy()
    doy = pd.to_datetime(df[date_col]).dt.dayofyear

    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
    return df


# =========================
# Precipitation memory 
# =========================

def add_precipitation_lags(
    df: pd.DataFrame,
    target_col: str = "precipitation_sum",
    lags: list[int] = [1, 3, 7]
) -> pd.DataFrame:
    """
    Add lagged precipitation features.
    Physical meaning:
    - Short-term atmospheric memory
    """
    df = df.copy()
    for lag in lags:
        df[f"precip_lag_{lag}"] = df[target_col].shift(lag)
    return df



def add_precipitation_rolling(
    df: pd.DataFrame,
    target_col: str = "precipitation_sum",
    windows: list[int] = [3, 7]
) -> pd.DataFrame:
    """
    Add rolling precipitation accumulation features.
    Physical meaning:
    - Persistent wet-state indicator
    """
    df = df.copy()
    for w in windows:
        df[f"precip_roll_{w}"] = df[target_col].rolling(window=w).sum()
    return df


# =========================
# High-level feature sets
# =========================

def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct base physical feature set.
    """
    df = ensure_sorted_by_date(df)
    df = add_temperature_range(df)
    return df



def build_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct seasonal (cyclical) features.
    """
    df = ensure_sorted_by_date(df)
    df = add_cyclical_dayofyear(df)
    return df



def build_memory_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct precipitation memory features (lags + rolling).
    """
    df = ensure_sorted_by_date(df)
    df = add_precipitation_lags(df)
    df = add_precipitation_rolling(df)
    return df


# =========================
# Feature selection helpers
# =========================

def select_feature_subset(df: pd.DataFrame, feature_list: list[str], target_col: str) -> pd.DataFrame:
    """
    Select features + target and drop rows with missing values.
    """
    return df[feature_list + [target_col]].dropna()
