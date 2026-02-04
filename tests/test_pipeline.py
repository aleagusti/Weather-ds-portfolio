import subprocess
from pathlib import Path
from src.config import RAW_DATASET, PROCESSED_DIR, METRICS_FILE, PREDICTIONS_FILE

def test_pipeline_runs_without_error():
    result = subprocess.run(
        ["bash", "run_pipeline.sh"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, f"Pipeline failed:\n{result.stderr}"


def test_pipeline_outputs_exist():
    assert Path(RAW_DATASET).exists()
    assert any(Path(PROCESSED_DIR).glob("open_meteo_miami_daily_v*.csv"))
    assert any(Path(METRICS_FILE).parent.glob("metrics_*.csv"))
    assert any(Path(PREDICTIONS_FILE).parent.glob("predictions_*.csv"))