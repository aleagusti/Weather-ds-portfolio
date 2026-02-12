import subprocess
from pathlib import Path
from src.config import RAW_DATASET, PROCESSED_DIR, METRICS_FILE, PREDICTIONS_FILE
import pytest

def test_pipeline_runs_without_error():
    result = subprocess.run(
        ["bash", "run_pipeline.sh"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, f"Pipeline failed:\n{result.stderr}"

from src.config import RESULTS_DIR, SELECTED_MODEL_FAMILY, VERSION

def test_pipeline_outputs_exist():
    # Assuming the test checks for existence of pipeline output files
    output_files = [
        "models/final_model.pkl",
        RESULTS_DIR / f"predictions_best_{SELECTED_MODEL_FAMILY}_{VERSION}.csv",
        RESULTS_DIR / f"metrics_regression_{SELECTED_MODEL_FAMILY}_{VERSION}.csv",
    ]
    for file_path in output_files:
        path = Path(file_path)
        assert path.exists()

