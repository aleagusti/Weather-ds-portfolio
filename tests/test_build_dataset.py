import subprocess
import pandas as pd

from src.config import PROCESSED_DATASET


def test_build_dataset_creates_file():
    # borrar si existe
    if PROCESSED_DATASET.exists():
        PROCESSED_DATASET.unlink()

    # ejecutar script como módulo
    subprocess.run(
        ["python", "-m", "src.build_dataset"],
        check=True
    )

    assert PROCESSED_DATASET.exists(), "Processed dataset was not created"


def test_build_dataset_has_expected_columns():
    df = pd.read_csv(PROCESSED_DATASET)

    expected_cols = [
        "date",
        "temperature_2m_mean",
        "precipitation_sum",
        "temp_range",
        "doy_sin",
        "precip_lag_1",
        "precip_roll_3",
    ]

    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"

    assert len(df) > 1000, "Dataset looks too small"