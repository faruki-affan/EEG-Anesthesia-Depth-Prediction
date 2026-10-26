"""Load a Phase 3 model and predict BIS from EEG features.

Artifacts store the estimator, the training feature order, and the metadata
needed to apply the training-only transform. Predictions can be clipped to
the BIS range of 0–100.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


class FeatureValidationError(ValueError):
    """The input features do not match the trained model."""


class MedianBaseline(BaseEstimator, RegressorMixin):
    """Predict the training-set median BIS for every window."""

    def fit(self, X, y):
        matrix = np.asarray(X)
        target = np.asarray(y, dtype=np.float64).reshape(-1)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("MedianBaseline requires a non-empty two-dimensional feature matrix.")
        if target.shape[0] != matrix.shape[0]:
            raise ValueError("Features and BIS targets have different lengths.")
        if not np.all(np.isfinite(target)):
            raise ValueError("BIS targets must be finite.")
        self.median_ = float(np.median(target))
        self.n_features_in_ = int(matrix.shape[1])
        return self

    def predict(self, X):
        check_is_fitted(self, "median_")
        matrix = np.asarray(X)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"MedianBaseline expected {self.n_features_in_} features, got shape {matrix.shape}."
            )
        return np.full(matrix.shape[0], self.median_, dtype=np.float64)


class NonnegativeLog1p(BaseEstimator, TransformerMixin):
    """Apply log1p to right-skewed non-negative columns selected on the training fold.

    A column is transformed only when its training minimum is at least 0, its
    training standard deviation is positive, and its training skewness is at
    least ``skew_threshold``. The mask is not refit at prediction time.
    """

    def __init__(self, skew_threshold: float = 1.0):
        self.skew_threshold = skew_threshold

    def fit(self, X, y=None):
        matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("Log1p input must be two-dimensional.")
        if matrix.shape[0] < 3:
            raise ValueError("Log1p skewness needs at least 3 training rows.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Log1p training features must be finite.")
        skew = np.asarray(stats.skew(matrix, axis=0, bias=False), dtype=np.float64)
        minimum = matrix.min(axis=0)
        deviation = matrix.std(axis=0)
        self.skew_ = skew
        self.log_mask_ = (skew >= float(self.skew_threshold)) & (minimum >= 0.0) & (deviation > 0.0)
        self.n_features_in_ = int(matrix.shape[1])
        return self

    def transform(self, X):
        check_is_fitted(self, "log_mask_")
        matrix = np.array(X, dtype=np.float64, copy=True)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Log1p expected {self.n_features_in_} features, got shape {getattr(matrix, 'shape', None)}."
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Log1p received non-finite features.")
        if np.any(matrix[:, self.log_mask_] < 0.0):
            raise ValueError("A column selected for log1p has a negative value.")
        matrix[:, self.log_mask_] = np.log1p(matrix[:, self.log_mask_])
        return matrix


class BISModel:
    """Trained regressor plus the feature contract needed to apply it."""

    def __init__(self, estimator: Any, feature_names: list[str], metadata: dict[str, Any]):
        self.estimator = estimator
        self.feature_names = list(feature_names)
        self.metadata = dict(metadata)

    def predict(self, features: pd.DataFrame | np.ndarray, clip: bool | None = None) -> np.ndarray:
        matrix = feature_matrix(features, self.feature_names)
        predicted = np.asarray(self.estimator.predict(matrix), dtype=np.float64).reshape(-1)
        if predicted.shape != (matrix.shape[0],):
            raise FeatureValidationError(
                f"Expected {matrix.shape[0]} BIS predictions, got {predicted.shape}."
            )
        if not np.all(np.isfinite(predicted)):
            raise FeatureValidationError("The model returned a non-finite BIS prediction.")
        use_clip = bool(self.metadata.get("clip_predictions", True)) if clip is None else bool(clip)
        if use_clip:
            low = float(self.metadata.get("bis_min", 0.0))
            high = float(self.metadata.get("bis_max", 100.0))
            predicted = np.clip(predicted, low, high)
        return predicted


def feature_matrix(features: pd.DataFrame | np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Arrange inputs in the training column order.

    A DataFrame is matched by column name and then reordered. A positional
    array must already have one column per training feature, in that order.
    """
    if isinstance(features, pd.DataFrame):
        missing = [name for name in feature_names if name not in features.columns]
        if missing:
            raise FeatureValidationError(
                "Input is missing required EEG features: " + ", ".join(missing) + "."
            )
        matrix = features.loc[:, feature_names].to_numpy(dtype=np.float64)
    else:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2:
            raise FeatureValidationError(
                f"A positional feature array must have shape (n_windows, {len(feature_names)})."
            )
        if matrix.shape[1] != len(feature_names):
            raise FeatureValidationError(
                "Positional features must follow the training order "
                f"{feature_names}. Received {matrix.shape[1]} columns, expected {len(feature_names)}."
            )
    if matrix.shape[0] and not np.all(np.isfinite(matrix)):
        problems = []
        for index, name in enumerate(feature_names):
            n_bad = int((~np.isfinite(matrix[:, index])).sum())
            if n_bad:
                problems.append(f"{name} ({n_bad} non-finite)")
        raise FeatureValidationError("Invalid input features: " + "; ".join(problems) + ".")
    return matrix


def predict_bis(
    model: BISModel | str | Path,
    features: pd.DataFrame | np.ndarray,
    clip: bool | None = None,
) -> np.ndarray:
    """Predict BIS from EEG features with a model object or a joblib path."""
    if isinstance(model, (str, Path)):
        model = load_bis_model(model)
    if not isinstance(model, BISModel):
        raise FeatureValidationError("predict_bis expected a BISModel or a path to one.")
    return model.predict(features, clip=clip)


def save_bis_model(model: BISModel, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)


def load_bis_model(path: str | Path) -> BISModel:
    loaded = joblib.load(path)
    if not isinstance(loaded, BISModel):
        raise FeatureValidationError(f"{path} does not contain a BISModel artifact.")
    if not loaded.feature_names:
        raise FeatureValidationError(f"{path} has no feature names.")
    return loaded


def make_estimator(model_key: str, phase3: dict[str, Any]) -> MedianBaseline | Pipeline:
    """Build one unfitted model. Transforms inside a pipeline see training rows only."""
    seed = int(phase3["random_state"])
    if model_key == "median":
        return MedianBaseline()
    transformer = NonnegativeLog1p(skew_threshold=float(phase3["log1p_skew_threshold"]))
    if model_key == "random_forest":
        params = dict(phase3["random_forest"])
        params["random_state"] = seed
        params.setdefault("max_features", 1.0)
        model = RandomForestRegressor(**params)
    elif model_key == "hist_gradient_boosting":
        params = dict(phase3["hist_gradient_boosting"])
        params["random_state"] = seed
        params["early_stopping"] = bool(params.get("early_stopping", False))
        params.setdefault("loss", "squared_error")
        model = HistGradientBoostingRegressor(**params)
    else:
        raise ValueError(f"Unknown model '{model_key}'.")
    return Pipeline([("log1p", transformer), ("model", model)])


def logged_feature_names(estimator: Any, feature_names: list[str]) -> list[str]:
    if not hasattr(estimator, "named_steps") or "log1p" not in estimator.named_steps:
        return []
    mask = np.asarray(estimator.named_steps["log1p"].log_mask_, dtype=bool)
    return [name for name, flag in zip(feature_names, mask) if flag]
