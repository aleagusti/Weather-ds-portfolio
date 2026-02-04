from pathlib import Path

# =========================
# Project paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

# =========================
# Dataset & experiment versioning
# =========================
DATASET_NAME = "open_meteo_miami_daily"
VERSION = "v001"

RAW_DATASET = RAW_DIR / f"{DATASET_NAME}.csv"
PROCESSED_DATASET = PROCESSED_DIR / f"{DATASET_NAME}_{VERSION}.csv"

# =========================
# Modeling configuration
# =========================
TARGET_COL = "precipitation_sum"
DATE_COL = "date"

TRAIN_SPLIT_DATE = "2017-12-31"

# =========================
# Outputs
# =========================
METRICS_FILE = RESULTS_DIR / f"metrics_regression_{VERSION}.csv"
PREDICTIONS_FILE = RESULTS_DIR / f"predictions_best_{VERSION}.csv"
FEATURE_SETS_FILE = RESULTS_DIR / f"feature_sets_{VERSION}.json"