import pandas as pd
import numpy as np
import pytest

from src.features import (
    ensure_sorted_by_date,
    add_temperature_range,
    add_cyclical_dayofyear,
    add_precipitation_lags,
    add_precipitation_rolling,
)

# =========================
# Fixtures
# =========================

@pytest.fixture
def sample_df():
    """
    Minimal, deterministic daily dataset for feature tests.
    """
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=10, freq="D"),
        "temperature_2m_max": np.arange(10) + 25,
        "temperature_2m_min": np.arange(10) + 15,
        "precipitation_sum": np.arange(10) * 0.5,
    })


# =========================
# Tests
# =========================

def test_ensure_sorted_by_date(sample_df):
    shuffled = sample_df.sample(frac=1, random_state=42)
    sorted_df = ensure_sorted_by_date(shuffled)

    assert sorted_df["date"].is_monotonic_increasing
    assert len(sorted_df) == len(sample_df)


def test_add_temperature_range(sample_df):
    df = add_temperature_range(sample_df)

    assert "temp_range" in df.columns
    expected = df["temperature_2m_max"] - df["temperature_2m_min"]
    assert (df["temp_range"] == expected).all()


def test_add_cyclical_dayofyear(sample_df):
    df = add_cyclical_dayofyear(sample_df)

    assert "doy_sin" in df.columns
    assert "doy_cos" in df.columns
    assert df["doy_sin"].between(-1, 1).all()
    assert df["doy_cos"].between(-1, 1).all()


def test_add_precipitation_lags(sample_df):
    df = add_precipitation_lags(sample_df, lags=[1, 3])

    assert "precip_lag_1" in df.columns
    assert "precip_lag_3" in df.columns
    assert df["precip_lag_1"].isna().iloc[0]
    assert df["precip_lag_3"].isna().iloc[:3].all()


def test_add_precipitation_rolling(sample_df):
    df = add_precipitation_rolling(sample_df, windows=[3])

    assert "precip_roll_3" in df.columns
    assert df["precip_roll_3"].isna().iloc[:2].all()