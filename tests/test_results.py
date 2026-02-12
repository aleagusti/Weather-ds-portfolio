import pandas as pd
from pathlib import Path
import numpy as np
import pytest

from src.config import (
    METRICS_FILE,
    PREDICTIONS_FILE,
)


def test_metrics_file_created():
    path = METRICS_FILE
    assert path.exists()

    df = pd.read_csv(path)

    expected_cols = {"MAE", "RMSE", "R2"}
    assert expected_cols.issubset(df.columns)

    for col in expected_cols:
        assert df[col].notna().all()
        assert pd.api.types.is_numeric_dtype(df[col])
        assert pd.Series(df[col]).replace([float("inf"), -float("inf")], pd.NA).notna().all()

    assert (df["MAE"] >= 0).all()
    assert (df["RMSE"] >= 0).all()
    assert ((df["R2"] >= -1) & (df["R2"] <= 1)).all()


def test_predictions_file_created():
    path = PREDICTIONS_FILE
    assert path.exists()

    df = pd.read_csv(path)

    required_cols = {"date", "precipitation_sum", "y_pred", "error"}
    assert required_cols.issubset(df.columns)
    assert len(df) > 0
    assert df["y_pred"].notna().all()
    assert np.isfinite(df["y_pred"]).all()
    assert np.isfinite(df["error"]).all()


