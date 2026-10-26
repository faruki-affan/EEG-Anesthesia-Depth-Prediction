"""Spectral and time-domain features for one finite EEG window."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import signal

BAND_NAMES = ("delta", "theta", "alpha", "beta", "gamma")
TIME_FEATURES = (
    "mean",
    "std",
    "rms",
    "variance",
    "hjorth_mobility",
    "hjorth_complexity",
)


def welch_psd(
    segment: np.ndarray,
    fs: float,
    nperseg: int,
    noverlap: int,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided Welch PSD. ``scaling='density'`` so band integrals are power."""
    segment = np.asarray(segment, dtype=np.float64)
    if segment.ndim != 1 or segment.size < 4:
        raise ValueError("Welch input must be a one-dimensional window of at least 4 samples.")
    if not np.all(np.isfinite(segment)):
        raise ValueError("Welch input must be finite. Fill or reject missing samples first.")
    if nperseg > segment.size or nperseg < 4:
        raise ValueError(f"nperseg must be in [4, {segment.size}], got {nperseg}.")
    if not 0 <= noverlap < nperseg:
        raise ValueError("noverlap must satisfy 0 <= noverlap < nperseg.")
    freqs, psd = signal.welch(
        segment,
        fs=float(fs),
        window=window,
        nperseg=int(nperseg),
        noverlap=int(noverlap),
        detrend="constant",
        scaling="density",
        return_onesided=True,
    )
    return np.asarray(freqs, dtype=np.float64), np.maximum(np.asarray(psd, dtype=np.float64), 0.0)


def dominant_frequency(
    segment: np.ndarray,
    fs: float,
    nperseg: int,
    noverlap: int,
    window: str = "hann",
) -> float:
    """Frequency of the largest Welch bin, ignoring the DC bin."""
    freqs, psd = welch_psd(segment, fs, nperseg, noverlap, window)
    if psd.size > 1:
        psd = psd.copy()
        psd[0] = 0.0
    return float(freqs[int(np.argmax(psd))])


def band_power(freqs: np.ndarray, psd: np.ndarray, low_hz: float, high_hz: float) -> float:
    """Integrate PSD from ``low_hz`` to ``high_hz``, interpolating the edges.

    Adjacent bands that share an endpoint tile the frequency axis. The shared
    endpoint does not add the same sliver of power to both bands.
    """
    if freqs.size < 2:
        return 0.0
    low = max(float(low_hz), float(freqs[0]))
    high = min(float(high_hz), float(freqs[-1]))
    if high <= low:
        return 0.0
    mask = (freqs >= low) & (freqs <= high)
    picked_f = freqs[mask]
    picked_p = psd[mask]
    if picked_f.size == 0 or picked_f[0] > low:
        picked_f = np.insert(picked_f, 0, low)
        picked_p = np.insert(picked_p, 0, float(np.interp(low, freqs, psd)))
    if picked_f[-1] < high:
        picked_f = np.append(picked_f, high)
        picked_p = np.append(picked_p, float(np.interp(high, freqs, psd)))
    if picked_f.size < 2:
        return 0.0
    return float(np.trapezoid(picked_p, picked_f))


def spectral_entropy(psd: np.ndarray) -> float:
    """Shannon entropy of the PSD, normalized by ``log(n_bins)`` to ``[0, 1]``."""
    total = float(np.sum(psd))
    if total <= 0 or psd.size < 2:
        return float("nan")
    probability = psd / total
    positive = probability > 0
    entropy = float(-np.sum(probability[positive] * np.log(probability[positive])))
    return entropy / float(np.log(probability.size))


def spectral_edge(freqs: np.ndarray, psd: np.ndarray, fraction: float) -> float:
    """Lowest frequency at which the cumulative PSD reaches ``fraction``."""
    if not 0 < fraction <= 1:
        raise ValueError("Spectral-edge fraction must be in (0, 1].")
    total = float(np.sum(psd))
    if total <= 0 or freqs.size == 0:
        return float("nan")
    cumulative = np.cumsum(psd) / total
    index = int(np.searchsorted(cumulative, fraction, side="left"))
    return float(freqs[min(index, freqs.size - 1)])


def time_features(segment: np.ndarray) -> dict[str, float]:
    """Mean, standard deviation, RMS, variance, and Hjorth parameters.

    Standard deviation and variance use the population definition (``ddof=0``).
    Hjorth mobility and complexity use successive first differences.
    """
    segment = np.asarray(segment, dtype=np.float64)
    if segment.ndim != 1 or segment.size < 3 or not np.all(np.isfinite(segment)):
        raise ValueError("Time-domain features require at least 3 finite samples.")
    mean = float(np.mean(segment))
    variance = float(np.var(segment))
    std = float(np.sqrt(variance))
    rms = float(np.sqrt(np.mean(segment * segment)))
    mobility, complexity = _hjorth(segment)
    return {
        "mean": mean,
        "std": std,
        "rms": rms,
        "variance": variance,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
    }


def _hjorth(segment: np.ndarray) -> tuple[float, float]:
    first = np.diff(segment)
    variance = float(np.var(segment))
    first_variance = float(np.var(first))
    if variance <= 0 or first_variance <= 0:
        return float("nan"), float("nan")
    mobility = float(np.sqrt(first_variance / variance))
    second = np.diff(first)
    second_variance = float(np.var(second))
    if second_variance <= 0:
        return mobility, float("nan")
    mobility_of_derivative = float(np.sqrt(second_variance / first_variance))
    return mobility, float(mobility_of_derivative / mobility)


def extract_features(
    segment: np.ndarray,
    fs: float,
    bands: Sequence[tuple[str, float, float]],
    nperseg: int,
    noverlap: int,
    window: str,
    spectral_edge_fraction: float,
) -> dict[str, float]:
    """Band power, relative power, spectral entropy, spectral edge, and time features."""
    segment = np.asarray(segment, dtype=np.float64)
    if not np.all(np.isfinite(segment)):
        raise ValueError("Feature extraction requires a finite window.")
    freqs, psd = welch_psd(segment, fs, nperseg, noverlap, window)
    powers: dict[str, float] = {}
    for name, low, high in bands:
        powers[name] = band_power(freqs, psd, low, high)
    total = float(sum(powers.values()))
    features: dict[str, float] = {}
    for name, power in powers.items():
        features[f"{name}_power"] = power
        features[f"{name}_relative"] = float(power / total) if total > 0 else float("nan")
    in_band = (freqs >= min(low for _, low, _ in bands)) & (freqs <= max(high for _, _, high in bands))
    features["spectral_entropy"] = spectral_entropy(psd[in_band])
    features["spectral_edge_hz"] = spectral_edge(freqs[in_band], psd[in_band], spectral_edge_fraction)
    features.update(time_features(segment))
    return features
