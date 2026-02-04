import subprocess
from pathlib import Path

def test_pipeline_runs_without_error():
    result = subprocess.run(
        ["bash", "run_pipeline.sh"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, f"Pipeline failed:\n{result.stderr}"


def test_pipeline_outputs_exist():
    assert Path("data/raw/open_meteo_miami_daily.csv").exists()
    assert any(Path("data/processed").glob("open_meteo_miami_daily_v*.csv"))
    assert any(Path("data/results").glob("metrics_*.csv"))
    assert any(Path("data/results").glob("predictions_*.csv"))