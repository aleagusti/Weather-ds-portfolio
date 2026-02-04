"""
Dataset Builder

Constructs the processed daily dataset used for modeling.

Responsibilities:
- Read raw Open-Meteo daily data
- Apply feature engineering blocks (from features.py)
- Handle NaNs induced by lag/rolling features
- Save versioned processed dataset

Design principles:
- No API calls
- No modeling
- Single responsibility: raw -> processed
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

# =========================
# Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "open_meteo_miami_daily.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "open_meteo_miami_daily_v001.csv"

# Allow importing from src/
sys.path.append(str(PROJECT_ROOT))

from src.features import (
    ensure_sorted_by_date,
    build_base_features,
    build_seasonal_features,
    build_memory_features,
)


# =========================
# Dataset construction
# =========================

def build_dataset() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_PATH}")

    # Load raw data
    df = pd.read_csv(RAW_PATH, parse_dates=["date"])

    # Ensure temporal ordering
    df = ensure_sorted_by_date(df, date_col="date")

    # Apply feature blocks
    df = build_base_features(df)
    df = build_seasonal_features(df)
    df = build_memory_features(df)

    # Drop rows with NaNs caused by lags/rolling windows
    df = df.dropna().reset_index(drop=True)

    return df


# =========================
# Script entry point
# =========================

def main() -> None:
    print("[START] Building processed dataset")

    df = build_dataset()

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print(f"[DONE] Processed dataset saved to {PROCESSED_PATH}")
    print(f"Rows: {len(df)} | Columns: {df.shape[1]}")
    print(df.head())


if __name__ == "__main__":
    main()
