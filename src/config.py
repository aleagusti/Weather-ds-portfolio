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

# =========================
# Temporal split configuration
# =========================
TEST_FRACTION = 0.20
RANDOM_STATE = 42

# =========================
# Model selection (single source of truth)
# =========================

SELECTED_MODEL_FAMILY = "mlp"
SELECTED_MODEL_NAME = "MLP_small"
SELECTED_EXPERIMENT = "EXP5_base_memory"

# =========================
# Outputs
# =========================
METRICS_FILE = RESULTS_DIR / f"metrics_regression_{SELECTED_MODEL_FAMILY}_{VERSION}.csv"
PREDICTIONS_FILE = RESULTS_DIR / f"predictions_best_{SELECTED_MODEL_FAMILY}_{VERSION}.csv"
FEATURE_SETS_FILE = RESULTS_DIR / f"feature_sets_{SELECTED_EXPERIMENT}_{VERSION}.json"

