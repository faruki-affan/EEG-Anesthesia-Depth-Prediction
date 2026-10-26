"""Regression metrics for BIS predictions.

Undefined Pearson correlation and R² are returned as None instead of a
substitute number. A constant prediction has no correlation. A constant
target has no R².
"""

from __future__ import annotations

import numpy as np


def regression_metrics(y_true, y_pred) -> dict[str, float | int | None]:
    actual = np.asarray(y_true, dtype=np.float64).reshape(-1)
    predicted = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted BIS must have the same shape.")
    if actual.size == 0:
        raise ValueError("Metrics require at least one window.")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(predicted)):
        raise ValueError("Metrics require finite BIS values.")

    error = predicted - actual
    absolute = np.abs(error)
    ss_res = float(np.sum(error**2))
    centered = actual - float(np.mean(actual))
    ss_tot = float(np.sum(centered**2))
    if ss_tot <= 1e-8:
        r2 = None
    else:
        r2 = float(1.0 - ss_res / ss_tot)
    # A constant prediction has no linear association. Tiny numerical
    # spread left by an otherwise constant predictor is treated as constant.
    if float(np.std(actual)) <= 1e-8 or float(np.std(predicted)) <= 1e-8:
        pearson = None
    else:
        pearson = float(np.corrcoef(actual, predicted)[0, 1])
    return {
        "n_windows": int(actual.size),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": r2,
        "pearson_r": pearson,
        "bias": float(np.mean(error)),
        "median_absolute_error": float(np.median(absolute)),
        "p90_absolute_error": float(np.percentile(absolute, 90)),
        "within_5_percent": float(np.mean(absolute <= 5.0) * 100.0),
        "within_10_percent": float(np.mean(absolute <= 10.0) * 100.0),
    }


def metrics_by_bis_range(y_true, y_pred, case_ids=None) -> list[dict]:
    """MAE and related metrics inside actual-BIS bins."""
    actual = np.asarray(y_true, dtype=np.float64).reshape(-1)
    predicted = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    edges = np.asarray([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
    labels = ["0–20", "20–40", "40–60", "60–80", "80–100"]
    # Right-closed bins, with 0 included in the first bin.
    bin_index = np.digitize(actual, edges[1:-1], right=True)
    rows = []
    for index, label in enumerate(labels):
        mask = bin_index == index
        if case_ids is not None:
            ids = np.asarray(case_ids).reshape(-1)[mask]
            n_cases = int(len(set(ids.tolist())))
        else:
            n_cases = None
        if not np.any(mask):
            rows.append(
                {
                    "bis_range": label,
                    "n_windows": 0,
                    "n_cases": n_cases,
                    "mae": None,
                    "rmse": None,
                    "bias": None,
                }
            )
            continue
        metrics = regression_metrics(actual[mask], predicted[mask])
        rows.append(
            {
                "bis_range": label,
                "n_windows": metrics["n_windows"],
                "n_cases": n_cases,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "bias": metrics["bias"],
                "within_10_percent": metrics["within_10_percent"],
            }
        )
    return rows
