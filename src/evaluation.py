from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def inverse_target(y: np.ndarray | "np.typing.ArrayLike", use_log_target: bool) -> np.ndarray:
    """
    Invert log1p transform back to original scale (e.g., mm).

    If use_log_target=True, assumes y is in log1p space and returns expm1(y).
    If False, returns y as a numpy array.
    """
    y_arr = np.asarray(y)
    if use_log_target:
        return np.expm1(y_arr)
    return y_arr


def evaluate_regression(y_true: np.ndarray | "np.typing.ArrayLike",
                        y_pred: np.ndarray | "np.typing.ArrayLike") -> Dict[str, float]:
    """
    Compute MAE, RMSE, R2 on the provided scale.

    IMPORTANT: If your model is trained on log1p(target), then pass inverted
    arrays (original scale) here to keep metrics comparable.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
    r2 = r2_score(y_true_arr, y_pred_arr)

    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def add_run_metadata(row: Dict[str, object],
                     *,
                     split: str,
                     use_log_target: bool,
                     test_fraction: Optional[float] = None,
                     dataset_version: Optional[str] = None,
                     exp_version: Optional[str] = None) -> Dict[str, object]:
    """
    Small helper to standardize metadata fields across experiments.
    Returns a NEW dict (doesn't mutate input).
    """
    out = dict(row)
    out["split"] = split
    out["use_log_target"] = bool(use_log_target)
    if test_fraction is not None:
        out["test_fraction"] = float(test_fraction)
    if dataset_version is not None:
        out["dataset_version"] = str(dataset_version)
    if exp_version is not None:
        out["exp_version"] = str(exp_version)
    return out
