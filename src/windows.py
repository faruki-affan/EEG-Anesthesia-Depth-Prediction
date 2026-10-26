"""EEG window layout, quality rules, and BIS alignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EXCESSIVE_MISSING = "excessive_missing_data"
FLATLINE = "flatline"
SEVERE_ARTIFACT = "severe_artifact"
INVALID_BIS = "invalid_bis"
INVALID_FEATURES = "invalid_features"


@dataclass(frozen=True)
class WindowSpec:
    index: int
    start_index: int
    stop_index: int
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class EegWindowQuality:
    reasons: tuple[str, ...]
    signal: np.ndarray | None
    n_missing: int
    missing_fraction: float
    peak_abs_uv: float | None
    std_uv: float | None


def window_specs(
    n_samples: int,
    t_start_sec: float,
    sample_interval_sec: float,
    length_sec: float,
    overlap_fraction: float,
) -> tuple[int, int, list[WindowSpec]]:
    """Build fixed-length windows on the original sample clock.

    Returns the window length in samples, the hop in samples, and one spec
    per complete window. A window covers ``[start_sec, end_sec)``.
    """
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative.")
    if sample_interval_sec <= 0:
        raise ValueError("sample_interval_sec must be positive.")
    if length_sec <= 0:
        raise ValueError("length_sec must be positive.")
    if not 0 <= overlap_fraction < 1:
        raise ValueError("overlap_fraction must be in [0, 1).")
    n_window = int(round(length_sec / sample_interval_sec))
    if n_window < 4:
        raise ValueError("The window is shorter than 4 samples.")
    hop = int(round(n_window * (1.0 - overlap_fraction)))
    if hop < 1:
        raise ValueError("The hop is shorter than 1 sample.")
    windows: list[WindowSpec] = []
    if n_samples < n_window:
        return n_window, hop, windows
    index = 0
    for start in range(0, n_samples - n_window + 1, hop):
        stop = start + n_window
        windows.append(
            WindowSpec(
                index=index,
                start_index=start,
                stop_index=stop,
                start_sec=float(t_start_sec + start * sample_interval_sec),
                end_sec=float(t_start_sec + stop * sample_interval_sec),
            )
        )
        index += 1
    return n_window, hop, windows


def fill_interior_nans(values: np.ndarray) -> np.ndarray:
    """Linearly fill NaNs that sit between the first and last finite samples.

    Missing samples at either edge are left missing. The input is not modified.
    """
    output = np.array(values, dtype=np.float64, copy=True)
    finite = np.isfinite(output)
    if finite.all() or not finite.any():
        return output
    indices = np.flatnonzero(finite)
    first = int(indices[0])
    last = int(indices[-1])
    if last == first:
        return output
    grid = np.arange(first, last + 1)
    output[first : last + 1] = np.interp(grid, indices, output[indices])
    return output


def evaluate_eeg(
    segment: np.ndarray,
    max_missing_fraction: float,
    flatline_std_uv: float,
    artifact_peak_uv: float,
    artifact_std_uv: float,
) -> EegWindowQuality:
    """Reject a window for missing data, a flatline, or a severe artifact.

    The returned signal is the finite window used for features. It is present
    only when none of those EEG reasons apply. Interior gaps under the missing
    limit are linearly interpolated in that copy.
    """
    segment = np.asarray(segment, dtype=np.float64)
    if segment.ndim != 1:
        raise ValueError("A window must be one-dimensional.")
    n_samples = int(segment.size)
    n_missing = int(np.count_nonzero(~np.isfinite(segment)))
    missing_fraction = float(n_missing / n_samples) if n_samples else 1.0
    reasons: list[str] = []
    filled: np.ndarray | None = None
    if n_samples == 0 or missing_fraction > max_missing_fraction:
        reasons.append(EXCESSIVE_MISSING)
    else:
        filled = fill_interior_nans(segment)
        if not np.all(np.isfinite(filled)):
            reasons.append(EXCESSIVE_MISSING)
            filled = None

    if filled is not None:
        peak = float(np.max(np.abs(filled)))
        std = float(np.std(filled))
        if std < flatline_std_uv:
            reasons.append(FLATLINE)
        if peak > artifact_peak_uv or std > artifact_std_uv:
            reasons.append(SEVERE_ARTIFACT)
    else:
        finite = segment[np.isfinite(segment)]
        if finite.size:
            peak = float(np.max(np.abs(finite)))
            std = float(np.std(finite))
        else:
            peak = None
            std = None
    return EegWindowQuality(
        reasons=tuple(reasons),
        signal=None if reasons else filled,
        n_missing=n_missing,
        missing_fraction=missing_fraction,
        peak_abs_uv=peak,
        std_uv=std,
    )


def bis_in_window(
    start_sec: float,
    end_sec: float,
    bis_time: np.ndarray,
    bis_values: np.ndarray,
    min_samples: int,
    bis_min: float,
    bis_max: float,
    delay_sec: float = 0.0,
) -> dict[str, float | int] | None:
    """Median valid BIS on ``[start + delay, end + delay)``.

    A positive ``delay_sec`` looks up BIS later than the EEG window. That is
    the hook for a processing-delay comparison. The baseline call uses 0.
    Timestamps of the samples that formed the median are returned so the
    choice can be audited without re-reading the decision rule.
    """
    if min_samples < 1:
        raise ValueError("min_samples must be at least 1.")
    if end_sec <= start_sec:
        raise ValueError("The BIS window must have a positive duration.")
    time = np.asarray(bis_time, dtype=np.float64)
    values = np.asarray(bis_values, dtype=np.float64)
    if time.shape != values.shape:
        raise ValueError("BIS time and value arrays must have the same shape.")
    lookup_start = float(start_sec + delay_sec)
    lookup_end = float(end_sec + delay_sec)
    mask = (time >= lookup_start) & (time < lookup_end) & np.isfinite(time) & np.isfinite(values)
    selected_time = time[mask]
    selected_values = values[mask]
    valid = (selected_values >= bis_min) & (selected_values <= bis_max)
    selected_time = selected_time[valid]
    selected_values = selected_values[valid]
    if selected_values.size < min_samples:
        return None
    return {
        "BIS_target": float(np.median(selected_values)),
        "bis_n": int(selected_values.size),
        "bis_first_sec": float(np.min(selected_time)),
        "bis_last_sec": float(np.max(selected_time)),
    }


def median_in_window(
    start_sec: float,
    end_sec: float,
    time: np.ndarray,
    values: np.ndarray,
    delay_sec: float = 0.0,
) -> float:
    """Median of finite samples on ``[start + delay, end + delay)``, or NaN."""
    stamps = np.asarray(time, dtype=np.float64)
    series = np.asarray(values, dtype=np.float64)
    lookup_start = float(start_sec + delay_sec)
    lookup_end = float(end_sec + delay_sec)
    mask = (stamps >= lookup_start) & (stamps < lookup_end) & np.isfinite(series)
    if not np.any(mask):
        return float("nan")
    return float(np.median(series[mask]))
