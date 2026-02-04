from pathlib import Path
import pandas as pd
import subprocess


def test_build_dataset_creates_file():
    output_path = Path("data/processed/open_meteo_miami_daily_v001.csv")

    # borrar si existe
    if output_path.exists():
        output_path.unlink()

    # ejecutar script
    subprocess.run(
        ["python", "src/build_dataset.py"],
        check=True
    )

    assert output_path.exists(), "Processed dataset was not created"


def test_build_dataset_has_expected_columns():
    df = pd.read_csv("data/processed/open_meteo_miami_daily_v001.csv")

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