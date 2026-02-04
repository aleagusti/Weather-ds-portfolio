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

import pandas as pd

from src.config import RAW_DATASET, PROCESSED_DATASET, PROCESSED_DIR
from src.features import (
    ensure_sorted_by_date,
    build_base_features,
    build_seasonal_features,
    build_memory_features,
)


# =========================
# Dataset construction
# =========================

def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build processed dataset with engineered features.
    """
    df = ensure_sorted_by_date(df, date_col="date")
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

    if not RAW_DATASET.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_DATASET}")

    df_raw = pd.read_csv(RAW_DATASET, parse_dates=["date"])
    df_processed = build_dataset(df_raw)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(PROCESSED_DATASET, index=False)

    print(f"[DONE] Processed dataset saved to {PROCESSED_DATASET}")
    print(f"Rows: {len(df_processed)} | Columns: {df_processed.shape[1]}")
    print(df_processed.head())


if __name__ == "__main__":
    main()
