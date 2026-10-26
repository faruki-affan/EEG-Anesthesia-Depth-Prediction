"""Phase 3 figures. Every series comes from the saved out-of-fold predictions."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ml_data import MODEL_KEYS, MODEL_LABELS, feature_group

COLORS = {
    "actual": "#222222",
    "median": "#7f7f7f",
    "random_forest": "#1f77b4",
    "hist_gradient_boosting": "#d62728",
    "frequency": "#1f77b4",
    "time": "#2ca02c",
}


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def plot_bis_over_time(predictions: pd.DataFrame, case_id: int, path: Path) -> None:
    case = predictions.loc[predictions["case_id"] == int(case_id)].sort_values("window_start_sec")
    figure, axis = plt.subplots(figsize=(11, 4.2))
    axis.plot(case["window_start_sec"], case["BIS_target"], color=COLORS["actual"], lw=1.2, label="Recorded BIS")
    for key in MODEL_KEYS:
        axis.plot(
            case["window_start_sec"],
            case[f"pred_{key}"],
            color=COLORS[key],
            lw=1.0,
            alpha=0.9,
            label=MODEL_LABELS[key],
        )
    axis.set_xlabel("EEG window start (s)")
    axis.set_ylabel("BIS (0–100)")
    axis.set_ylim(-5, 105)
    axis.set_title(f"Case {int(case_id)}: recorded BIS and leave-one-case-out predictions")
    axis.legend(loc="best", fontsize=8)
    axis.grid(True, alpha=0.3)
    _save(figure, path)


def plot_scatter(predictions: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for axis, key in zip(axes, MODEL_KEYS):
        axis.scatter(
            predictions["BIS_target"],
            predictions[f"pred_{key}"],
            s=8,
            alpha=0.25,
            c=COLORS[key],
            linewidths=0,
        )
        axis.plot([0, 100], [0, 100], color="black", lw=1)
        axis.set_xlim(0, 100)
        axis.set_ylim(0, 100)
        axis.set_xlabel("Recorded BIS")
        axis.set_title(MODEL_LABELS[key])
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("Predicted BIS")
    figure.suptitle("Pooled out-of-fold predictions (3 held-out cases)")
    _save(figure, path)


def plot_error_histogram(predictions: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(-60, 60, 49)
    for key in MODEL_KEYS:
        error = predictions[f"pred_{key}"] - predictions["BIS_target"]
        axis.hist(error, bins=bins, histtype="step", lw=1.6, color=COLORS[key], label=MODEL_LABELS[key])
    axis.axvline(0.0, color="black", lw=1)
    axis.set_xlabel("Prediction error (predicted − recorded BIS)")
    axis.set_ylabel("Windows")
    axis.set_title("Pooled out-of-fold prediction error")
    axis.legend(fontsize=8)
    axis.grid(True, alpha=0.3)
    _save(figure, path)


def plot_absolute_error(predictions: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.2))
    series = [np.abs(predictions[f"pred_{key}"] - predictions["BIS_target"]).to_numpy() for key in MODEL_KEYS]
    tick_labels = [MODEL_LABELS[key] for key in MODEL_KEYS]
    box = axis.boxplot(series, showfliers=False, patch_artist=True)
    axis.set_xticks(np.arange(1, len(tick_labels) + 1))
    axis.set_xticklabels(tick_labels)
    for patch, key in zip(box["boxes"], MODEL_KEYS):
        patch.set_facecolor(COLORS[key])
        patch.set_alpha(0.7)
    axis.set_ylabel("Absolute error (BIS points)")
    axis.set_title("Pooled out-of-fold absolute error")
    axis.grid(True, axis="y", alpha=0.3)
    _save(figure, path)


def plot_feature_importance(importance: pd.DataFrame, path: Path) -> None:
    ordered = importance.sort_values("permutation_mean")
    figure, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    colors = [COLORS[feature_group(name)] for name in ordered["feature"]]
    axes[0].barh(ordered["feature"], ordered["impurity_mean"], color=colors)
    axes[1].barh(ordered["feature"], ordered["permutation_mean"], color=colors, xerr=ordered["permutation_std"], capsize=2)
    axes[0].set_xlabel("Mean impurity importance")
    axes[1].set_xlabel("Mean permutation importance (held-out MAE increase)")
    axes[0].set_title("Training-set impurity")
    axes[1].set_title("Held-out permutation")
    figure.suptitle("Random Forest feature importance (unweighted mean of 3 cases)")
    figure.text(0.5, 0.01, "Blue: frequency domain. Green: time domain. Importance is not causation.", ha="center", fontsize=8)
    _save(figure, path)


def plot_model_comparison(pooled: dict[str, dict], path: Path) -> None:
    labels = [MODEL_LABELS[key] for key in MODEL_KEYS]
    mae = [pooled[key]["mae"] for key in MODEL_KEYS]
    rmse = [pooled[key]["rmse"] for key in MODEL_KEYS]
    position = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8, 4.2))
    axis.bar(position - width / 2, mae, width, label="MAE", color="#4c78a8")
    axis.bar(position + width / 2, rmse, width, label="RMSE", color="#f58518")
    axis.set_xticks(position, labels)
    axis.set_ylabel("BIS points")
    axis.set_title("Pooled out-of-fold error (3 cases, window-level)")
    axis.legend()
    axis.grid(True, axis="y", alpha=0.3)
    _save(figure, path)


def plot_bis_distribution(features: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    bins = np.linspace(0, 100, 21)
    for axis, (case_id, case_frame) in zip(axes, features.groupby("case_id", sort=True)):
        axis.hist(case_frame["BIS_target"], bins=bins, color="#4c78a8", edgecolor="white")
        axis.set_title(f"Case {int(case_id)} (n={len(case_frame)})")
        axis.set_xlabel("Recorded BIS")
        axis.set_xlim(0, 100)
        axis.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Windows")
    figure.suptitle("Recorded BIS distribution by surgical case")
    _save(figure, path)


def plot_performance_by_range(range_table: pd.DataFrame, path: Path) -> None:
    ranges = list(dict.fromkeys(range_table["bis_range"].tolist()))
    position = np.arange(len(ranges))
    width = 0.25
    figure, axis = plt.subplots(figsize=(9, 4.2))
    for offset, key in enumerate(MODEL_KEYS):
        subset = range_table.loc[range_table["model"] == key].set_index("bis_range").reindex(ranges)
        axis.bar(
            position + (offset - 1) * width,
            subset["mae"].to_numpy(dtype=float),
            width,
            label=MODEL_LABELS[key],
            color=COLORS[key],
        )
    axis.set_xticks(position, ranges)
    axis.set_xlabel("Recorded BIS range")
    axis.set_ylabel("MAE (BIS points)")
    axis.set_title("Pooled out-of-fold MAE by recorded BIS range")
    axis.legend(fontsize=8)
    axis.grid(True, axis="y", alpha=0.3)
    _save(figure, path)


def plot_bland_altman(predictions: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for axis, key in zip(axes, MODEL_KEYS):
        actual = predictions["BIS_target"].to_numpy(dtype=float)
        predicted = predictions[f"pred_{key}"].to_numpy(dtype=float)
        average = (actual + predicted) / 2.0
        delta = predicted - actual
        bias = float(np.mean(delta))
        sd = float(np.std(delta, ddof=1)) if delta.size > 1 else 0.0
        axis.scatter(average, delta, s=8, alpha=0.25, c=COLORS[key], linewidths=0)
        axis.axhline(bias, color="black", lw=1, label=f"Bias {bias:.2f}")
        axis.axhline(bias + 1.96 * sd, color="black", lw=1, ls="--", label=f"±1.96 SD ({sd:.2f})")
        axis.axhline(bias - 1.96 * sd, color="black", lw=1, ls="--")
        axis.set_xlabel("Mean of recorded and predicted BIS")
        axis.set_title(MODEL_LABELS[key])
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7, loc="best")
    axes[0].set_ylabel("Predicted − recorded BIS")
    figure.suptitle("Bland–Altman plot, pooled out-of-fold windows")
    _save(figure, path)
