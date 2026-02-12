from __future__ import annotations

"""
Runs the selected production model using a fixed 80/20 temporal split.

Model selection lives in:
src/config.py

This script:
1. Loads processed dataset
2. Applies temporal split
3. Builds experiment features (same logic as notebooks)
4. Trains selected model
5. Enforces non-negativity
6. Evaluates
7. Saves outputs
"""

# =========================
# Imports
# =========================

from pathlib import Path
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.config import (
    PROCESSED_DATASET,
    TARGET_COL,
    DATE_COL,
    TEST_FRACTION,
    RESULTS_DIR,
    SELECTED_EXPERIMENT,
    RANDOM_STATE,
    SELECTED_MODEL_FAMILY,
    SELECTED_MODEL_NAME,
)

from src.features import ensure_sorted_by_date
from src.evaluation import evaluate_regression

# =========================
# Model factory
# =========================

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


def get_selected_model():
    """Return the selected model instance based on config."""

    if SELECTED_MODEL_FAMILY == "mlp":

        if SELECTED_MODEL_NAME == "MLP_small":
            return MLPRegressor(
                hidden_layer_sizes=(32,),
                activation="relu",
                solver="adam",
                alpha=0.0001,
                learning_rate_init=0.001,
                max_iter=800,
                random_state=RANDOM_STATE,
            )

        elif SELECTED_MODEL_NAME == "MLP_medium":
            return MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.0001,
                learning_rate_init=0.001,
                max_iter=800,
                random_state=RANDOM_STATE,
            )

        else:
            raise ValueError(f"Unknown MLP model: {SELECTED_MODEL_NAME}")

    elif SELECTED_MODEL_FAMILY == "xgb":

        return XGBRegressor(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    elif SELECTED_MODEL_FAMILY == "base":

        if SELECTED_MODEL_NAME == "RandomForest":
            return RandomForestRegressor(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )

        elif SELECTED_MODEL_NAME == "Linear":
            return LinearRegression()

        elif SELECTED_MODEL_NAME == "Ridge":
            return Ridge()

        elif SELECTED_MODEL_NAME == "Lasso":
            return Lasso()

        else:
            raise ValueError(f"Unknown base model: {SELECTED_MODEL_NAME}")

    else:
        raise ValueError(f"Unknown model family: {SELECTED_MODEL_FAMILY}")


# =========================
# Feature groups (copied from notebooks)
# =========================

BASE_FEATURES = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "cloud_cover_mean",
    "surface_pressure_mean",
    "wind_speed_10m_mean",
    "temp_range",
]

SEASONAL_FEATURES = [
    "doy_sin",
    "doy_cos",
]

MEMORY_FEATURES = [
    "precip_lag_1",
    "precip_lag_3",
    "precip_lag_7",
    "precip_roll_3",
    "precip_roll_7",
]


# =========================
# Experiment definition (copied structure)
# =========================

@dataclass(frozen=True)
class Experiment:
    name: str
    feature_list: List[str]


def make_experiments() -> List[Experiment]:

    return [
        Experiment("EXP1_base", BASE_FEATURES),
        Experiment("EXP2_base_season", BASE_FEATURES + SEASONAL_FEATURES),
        Experiment("EXP3_memory", MEMORY_FEATURES),
        Experiment("EXP4_season_memory", SEASONAL_FEATURES + MEMORY_FEATURES),
        Experiment("EXP5_base_memory", BASE_FEATURES + MEMORY_FEATURES),
        Experiment("EXP6_full", BASE_FEATURES + SEASONAL_FEATURES + MEMORY_FEATURES),
        Experiment("EXP7_full_lags_only",
                   BASE_FEATURES + SEASONAL_FEATURES + ["precip_lag_1", "precip_lag_7"]),
        Experiment("EXP8_full_roll_only",
                   BASE_FEATURES + SEASONAL_FEATURES + ["precip_roll_7"]),
    ]


EXPERIMENTS = make_experiments()


# =========================
# Temporal split
# =========================

def temporal_split(df: pd.DataFrame, test_fraction: float):
    split_index = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_index], df.iloc[split_index:]


# =========================
# Main
# =========================

def main():

    print("Running selected production model...\n")

    # 1 Load dataset
    df = pd.read_csv(PROCESSED_DATASET, parse_dates=[DATE_COL])
    df = ensure_sorted_by_date(df, date_col=DATE_COL)

    # 2 Validate experiment
    selected_exp = next(
        (e for e in EXPERIMENTS if e.name == SELECTED_EXPERIMENT),
        None
    )

    if selected_exp is None:
        raise ValueError(f"Unknown experiment: {SELECTED_EXPERIMENT}")

    selected_features = selected_exp.feature_list
    print(f"Selected experiment: {SELECTED_EXPERIMENT}")
    print(f"Using {len(selected_features)} features")

    # 3 Split
    train_df, test_df = temporal_split(df, TEST_FRACTION)

    train_df = train_df[[DATE_COL, TARGET_COL] + selected_features].dropna()
    test_df = test_df[[DATE_COL, TARGET_COL] + selected_features].dropna()

    X_train = train_df[selected_features]
    y_train = train_df[TARGET_COL]

    X_test = test_df[selected_features]
    y_test = test_df[TARGET_COL]

    # 4 Get model
    model = get_selected_model()
    print(f"Selected model: {model.__class__.__name__}")

    # 5 Scaling only for MLP
    if model.__class__.__name__ == "MLPRegressor":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # 6 Train
    model.fit(X_train, y_train)

    # 7 Predict
    y_pred = model.predict(X_test)

    # 8 Enforce non-negativity
    neg_rate = float((y_pred < 0).mean())
    neg_min = float(y_pred.min())
    y_pred = np.clip(y_pred, 0.0, None)

    print(f"Negative predictions (before clip): {neg_rate:.4%} | min={neg_min:.4f}")

    # 9 Evaluate
    metrics = evaluate_regression(y_test.values, y_pred)

    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 10 Save outputs
    from src.config import METRICS_FILE, PREDICTIONS_FILE

    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # --- Save metrics ---
    pd.DataFrame([metrics]).to_csv(METRICS_FILE, index=False)

    # --- Save predictions ---
    pred_df = pd.DataFrame({
        DATE_COL: test_df[DATE_COL].values,
        TARGET_COL: y_test.values,
        "y_pred": y_pred,
    })

    pred_df["error"] = pred_df[TARGET_COL] - pred_df["y_pred"]

    pred_df.to_csv(PREDICTIONS_FILE, index=False)

    # --- Save trained model ---
    import joblib

    model_path = Path("models/final_model.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    print(f"\nSaved metrics → {METRICS_FILE}")
    print(f"Saved predictions → {PREDICTIONS_FILE}")
    print(f"Saved model → {model_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()