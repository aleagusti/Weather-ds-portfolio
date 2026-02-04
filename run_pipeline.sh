#!/usr/bin/env bash
set -e

echo "======================================"
echo " Weather DS Pipeline starting..."
echo "======================================"

# --------------------------------------
# STEP 0: Environment check
# --------------------------------------
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
  echo "[ERROR] Conda environment not activated"
  echo "Run: conda activate weather-ds"
  exit 1
fi

# --------------------------------------
# STEP 1: Fetch raw data
# --------------------------------------
echo "[STEP 1] Fetching raw data"
python -m src.fetch_open_meteo_daily

# --------------------------------------
# STEP 2: Build processed dataset
# --------------------------------------
echo "[STEP 2] Building processed dataset"
python -m src.build_dataset

# --------------------------------------
# STEP 3: Run modeling
# --------------------------------------
echo "[STEP 3] Running modeling"
python -m src.modeling_regression

echo "======================================"
echo " Pipeline finished successfully"
echo "======================================"