"""Gap-aware Butterworth bandpass for intraoperative EEG.

The filter is a zero-phase cascade of second-order sections. Long missing
stretches are not interpolated, and each finite side is filtered on its own
so filter state cannot cross the gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class FilterResult:
    values: np.ndarray
    n_long_gaps: int
    n_short_gaps_bridged: int
    n_edge_gaps: int
    n_segments_filtered: int
    n_segments_skipped: int
    n_missing_input: int
    n_missing_output: int
    pad_edge_samples: int


def design_bandpass(order: int, low_hz: float, high_hz: float, fs: float) -> np.ndarray:
    """Return a stable SOS bandpass. ``order`` is the Butterworth order."""
    if int(order) < 1:
        raise ValueError("Butterworth order must be at least 1.")
    if fs <= 0:
        raise ValueError("Sampling rate must be positive.")
    if not 0 < low_hz < high_hz < fs / 2.0:
        raise ValueError(
            f"Bandpass edges {low_hz}–{high_hz} Hz are outside (0, {fs / 2.0}) Hz."
        )
    sos = signal.butter(
        int(order),
        [float(low_hz), float(high_hz)],
        btype="bandpass",
        fs=float(fs),
        output="sos",
    )
    sos = np.asarray(sos, dtype=np.float64)
    if sos.ndim != 2 or sos.shape[1] != 6:
        raise ValueError(f"Expected second-order sections of shape (n, 6), got {sos.shape}.")
    return sos


def sos_pad_edge(sos: np.ndarray) -> int:
    """Default ``sosfiltfilt`` pad length. The input must be longer than this."""
    n_sections = int(sos.shape[0])
    ntaps = 2 * n_sections + 1
    ntaps -= min(int(np.sum(sos[:, 2] == 0)), int(np.sum(sos[:, 5] == 0)))
    return int(ntaps * 3)


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return ``[start, end)`` index pairs for each run of True values."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    changes = np.diff(mask.astype(np.int8))
    starts = (np.flatnonzero(changes == 1) + 1).tolist()
    ends = (np.flatnonzero(changes == -1) + 1).tolist()
    if bool(mask[0]):
        starts.insert(0, 0)
    if bool(mask[-1]):
        ends.append(int(mask.size))
    return list(zip(starts, ends))


def bandpass_filter(
    values: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int,
    max_gap_sec: float,
) -> FilterResult:
    """Bandpass ``values`` without filtering across long missing-data gaps.

    Interior gaps of at most ``max_gap_sec`` are linearly interpolated for the
    duration of the filter and then set back to NaN. Longer gaps, and gaps
    that touch either end of the recording, stay open and split the filter.
    """
    original = np.asarray(values, dtype=np.float64)
    if original.ndim != 1:
        raise ValueError("EEG must be a one-dimensional array.")
    if max_gap_sec < 0:
        raise ValueError("max_gap_sec must be non-negative.")
    sos = design_bandpass(order, low_hz, high_hz, fs)
    edge = sos_pad_edge(sos)
    max_gap_samples = int(round(max_gap_sec * fs))

    missing = ~np.isfinite(original)
    working = original.copy()
    working[missing] = np.nan
    n_long = 0
    n_short = 0
    n_edge = 0
    for start, end in true_runs(missing):
        length = end - start
        at_edge = start == 0 or end == working.size
        if length > max_gap_samples:
            n_long += 1
            continue
        if at_edge:
            n_edge += 1
            continue
        working[start:end] = np.interp(
            np.arange(start, end),
            [start - 1, end],
            [working[start - 1], working[end]],
        )
        n_short += 1

    output = np.full(original.shape, np.nan, dtype=np.float64)
    n_filtered = 0
    n_skipped = 0
    finite = np.isfinite(working)
    for start, end in true_runs(finite):
        segment = working[start:end]
        if segment.size <= edge:
            n_skipped += 1
            continue
        try:
            filtered = signal.sosfiltfilt(sos, segment)
        except ValueError:
            n_skipped += 1
            continue
        output[start:end] = np.asarray(filtered, dtype=np.float64)
        n_filtered += 1

    output[missing] = np.nan
    output[~np.isfinite(output)] = np.nan
    return FilterResult(
        values=output,
        n_long_gaps=n_long,
        n_short_gaps_bridged=n_short,
        n_edge_gaps=n_edge,
        n_segments_filtered=n_filtered,
        n_segments_skipped=n_skipped,
        n_missing_input=int(missing.sum()),
        n_missing_output=int(np.isnan(output).sum()),
        pad_edge_samples=edge,
    )
