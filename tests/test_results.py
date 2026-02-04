import pandas as pd
from pathlib import Path


def test_metrics_file_created():
    path = Path("data/results/metrics_regression_v001.csv")
    assert path.exists()

    df = pd.read_csv(path)
    assert "RMSE" in df.columns
    assert df["RMSE"].min() > 0


def test_predictions_file_created():
    path = Path("data/results/predictions_best_v001.csv")
    assert path.exists()

    df = pd.read_csv(path)

    # columnas clave
    assert "date" in df.columns
    assert "precipitation_sum" in df.columns  
    assert "y_pred" in df.columns
    assert "error" in df.columns

    # checks básicos de sanidad
    assert len(df) > 0
    assert df["y_pred"].isna().sum() == 0