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

@pytest.mark.xfail(reason="Results artifacts generated only after final model selection")
def test_pipeline_outputs_exist():
    # Assuming the test checks for existence of pipeline output files
    output_files = [
        "models/final_model.pkl",
        "results/metrics.csv",
        "results/predictions.csv",
    ]
    for file_path in output_files:
        path = Path(file_path)
        assert path.exists()