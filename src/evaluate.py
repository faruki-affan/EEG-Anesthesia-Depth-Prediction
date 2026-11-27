"""Figures for the matched held-out case comparison."""

from matplotlib import pyplot as plt
import numpy as np

from src.ml_data import MODEL_KEYS, MODEL_LABELS
from src.ml_plots import _save


def save_evaluation_figures(predictions, history, directory):
    models = {**{key: MODEL_LABELS[key] for key in MODEL_KEYS}, "cnn": "1D CNN"}
    for case, frame in predictions.groupby("case_id"):
        frame = frame.sort_values("window_end_sec")
        fig, (trace, error_axis) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        time = frame.window_end_sec
        trace.plot(time, frame.BIS_target, ".", ms=2, label="Recorded window BIS")
        for key, label in models.items():
            output = frame[f"pred_{key}"]
            trace.plot(time, output, ".", ms=2, label=label)
            error_axis.plot(time, output - frame.BIS_target, ".", ms=2, label=label)
        trace.set(ylabel="BIS", title=f"Held-out case {case}")
        trace.legend()
        error_axis.axhline(0, color="black", linewidth=0.6)
        error_axis.set(xlabel="Window end (s)", ylabel="Predicted − recorded BIS")
        _save(fig, directory / f"timeline_case_{case:05d}.png")

    fig, axes = plt.subplots(3, len(models), figsize=(15, 10), layout="constrained")
    actual = predictions.BIS_target
    for (key, label), (scatter, histogram, bland) in zip(models.items(), axes.T):
        output = predictions[f"pred_{key}"]
        error = output - actual
        scatter.scatter(actual, output, s=4, alpha=0.25)
        scatter.plot([0, 100], [0, 100], color="black", linewidth=0.7)
        scatter.set(title=label, xlabel="Recorded BIS", ylabel="Predicted BIS")
        histogram.hist(error, bins=35)
        histogram.set(xlabel="Predicted − recorded BIS", ylabel="Windows")
        bland.scatter((actual + output) / 2, error, s=4, alpha=0.25)
        bias, spread = error.mean(), error.std(ddof=1)
        for level in (bias, bias - 1.96 * spread, bias + 1.96 * spread):
            bland.axhline(level, color="black", linestyle="--", linewidth=0.7)
        bland.set(xlabel="Mean of recorded and predicted BIS", ylabel="Bland–Altman difference")
    _save(fig, directory / "held_out_agreement.png")

    fig, axis = plt.subplots(figsize=(9, 4))
    for case, fold in history.groupby("test_case"):
        train, = axis.plot(fold.epoch, fold.train_loss, label=f"Test case {case} training")
        axis.plot(fold.epoch, fold.validation_loss, "--", color=train.get_color(),
                  label=f"Test case {case} validation")
    axis.set(xlabel="Epoch", ylabel="Smooth L1 loss (scaled BIS)")
    axis.legend(ncols=2)
    _save(fig, directory / "loss.png")

