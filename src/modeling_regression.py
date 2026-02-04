from __future__ import annotations

"""
This script runs the final selected experiment using a fixed 80/20 temporal split.
Model and feature comparisons are performed in notebooks; this script executes
the chosen production configuration only.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from .config import (
    PROCESSED_DATASET,
    RESULTS_DIR,
    METRICS_FILE,
    PREDICTIONS_FILE,
    FEATURE_SETS_FILE,
    VERSION,
    TEST_FRACTION,
)


# =========================
# Paths / Config
# =========================

BEST_EXPERIMENT = {
    "name": "EXP5_base_memory",
    "features": [
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "cloud_cover_mean",
        "precip_lag_1",
        "precip_lag_3",
        "precip_lag_7",
        "precip_roll_3",
        "precip_roll_7",
        "doy_sin",
        "doy_cos",
    ],
    "model": "RandomForest",
    "model_params": {
        "n_estimators": 400,
        "random_state": 0,
        "n_jobs": -1,
        "max_depth": None,
    },
}


# =========================
# Import local features
# =========================

# IMPORTANT:
# This script assumes it is executed as a module from the project root, e.g.:
#   python -m src.modeling_regression
# or via run_pipeline.sh, which sets the correct PYTHONPATH.
from .features import (
    ensure_sorted_by_date,
    build_base_features,
    build_seasonal_features,
    build_memory_features,
)


# =========================
# Helpers
# =========================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


@dataclass
class ExperimentResult:
    experiment: str
    model_name: str
    metrics: Dict[str, float]


def fit_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    target: str,
    model,
) -> Tuple[np.ndarray, np.ndarray]:
    X_train = train_df[features].to_numpy()
    y_train = train_df[target].to_numpy()

    X_test = test_df[features].to_numpy()
    y_test = test_df[target].to_numpy()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_test, y_pred


# =========================
# Main modeling logic
# =========================

def main(test_fraction: float = TEST_FRACTION) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load processed dataset
    if not PROCESSED_DATASET.exists():
        raise FileNotFoundError(f"Processed dataset not found: {PROCESSED_DATASET}")

    df = pd.read_csv(PROCESSED_DATASET, parse_dates=["date"])
    df = ensure_sorted_by_date(df, date_col="date")

    # 2) Build feature blocks (same logic as notebook, but clean)
    df = build_base_features(df)
    df = build_seasonal_features(df)
    df = build_memory_features(df)

    # 3) Drop rows that got NaNs from lag/rolling features
    # Adjust list if you add more lags/rolls later.
    df = df.dropna().reset_index(drop=True)

    split_idx = int(len(df) * (1 - test_fraction))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise RuntimeError("Train/test split produced empty set. Check TEST_FRACTION.")

    TARGET = "precipitation_sum"

    features = [c for c in BEST_EXPERIMENT["features"] if c in df.columns]

    if len(features) == 0:
        raise RuntimeError("No valid features found for BEST_EXPERIMENT.")

    model = RandomForestRegressor(**BEST_EXPERIMENT["model_params"])

    y_true, y_pred = fit_predict(
        train_df,
        test_df,
        features,
        TARGET,
        model,
    )

    metrics = evaluate(y_true, y_pred)

    metrics_row = {
        "experiment": BEST_EXPERIMENT["name"],
        "model": BEST_EXPERIMENT["model"],
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
        "n_train": len(train_df),
        "n_test": len(test_df),
        "test_fraction": test_fraction,
    }

    # Build predictions dataframe (test set only)
    pred_df = test_df[["date", TARGET]].copy()
    pred_df["y_pred"] = y_pred
    pred_df["error"] = pred_df["y_pred"] - pred_df[TARGET]

    # Save outputs
    pd.DataFrame([metrics_row]).to_csv(METRICS_FILE, index=False)
    pred_df.to_csv(PREDICTIONS_FILE, index=False)

    with open(FEATURE_SETS_FILE, "w") as f:
        json.dump(BEST_EXPERIMENT, f, indent=2)

    print("[DONE] Modeling finished")
    print(f"Saved metrics -> {METRICS_FILE}")
    print(f"Saved predictions -> {PREDICTIONS_FILE}")
    print(f"Saved feature sets -> {FEATURE_SETS_FILE}")
    print("Executed experiment:", BEST_EXPERIMENT["name"])


if __name__ == "__main__":
    main()