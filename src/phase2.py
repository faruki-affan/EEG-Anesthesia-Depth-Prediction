"""Phase 2: filter cached EEG, extract features, and align BIS labels.

This module reads the Phase 1 NumPy archives. It does not download VitalDB
tracks and it does not train a model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features import BAND_NAMES, extract_features
from src.preprocess import FilterResult, bandpass_filter
from src.vitaldb_loader import PROJECT_ROOT, load_config, load_numeric, load_waveform
from src.windows import (
    INVALID_BIS,
    INVALID_FEATURES,
    WindowSpec,
    bis_in_window,
    evaluate_eeg,
    median_in_window,
    window_specs,
)

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "case_id",
    "window_index",
    "window_start_sec",
    "window_end_sec",
    "n_samples",
    *[f"{name}_power" for name in BAND_NAMES],
    *[f"{name}_relative" for name in BAND_NAMES],
    "spectral_entropy",
    "spectral_edge_hz",
    "mean",
    "std",
    "rms",
    "variance",
    "hjorth_mobility",
    "hjorth_complexity",
    "BIS_target",
    "bis_delay_sec",
    "missing_fraction",
    "n_missing",
    "peak_abs_uv",
    "bis_n",
    "bis_first_sec",
    "bis_last_sec",
    "sqi_median",
    "sampling_rate_hz",
]

REJECTION_COLUMNS = [
    "case_id",
    "window_index",
    "window_start_sec",
    "window_end_sec",
    "reason",
    "missing_fraction",
    "n_missing",
    "peak_abs_uv",
    "eeg_std_uv",
    "bis_n",
    "bis_value",
]


class Phase2Error(Exception):
    """Phase 2 could not build a feature table from the cached tracks."""


def _under_root(relative: str) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, Path):
        return str(value)
    return value


def _bands(config: dict[str, Any]) -> tuple[tuple[str, float, float], ...]:
    raw = config["features"]["bands"]
    missing = [name for name in BAND_NAMES if name not in raw]
    if missing:
        raise Phase2Error(f"Feature config is missing bands: {missing}")
    bands = []
    for name in BAND_NAMES:
        low, high = raw[name]
        bands.append((name, float(low), float(high)))
    return tuple(bands)


def _accepted_case_ids(config: dict[str, Any]) -> set[int] | None:
    path = _under_root(config["paths"]["metadata_dir"]) / "selected_cases.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(case_id) for case_id in payload.get("accepted_caseids", [])}


def save_processed_windows(path: Path, payload: dict[str, np.ndarray]) -> None:
    """Write one case of fixed-length EEG windows for a later 1D CNN."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_processed_windows(path: Path) -> dict[str, Any]:
    """Load a Phase 2 window archive. Arrays are copied out of the zip."""
    required = (
        "eeg",
        "window_index",
        "window_start_sec",
        "window_end_sec",
        "bis_target",
        "case_id",
        "sampling_rate_hz",
        "sample_interval_sec",
    )
    with np.load(path) as archive:
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise Phase2Error(f"{path.name} is missing {missing}.")
        return {
            "eeg": np.array(archive["eeg"], dtype=np.float32, copy=True),
            "window_index": np.array(archive["window_index"], dtype=np.int32, copy=True),
            "window_start_sec": np.array(archive["window_start_sec"], dtype=np.float64, copy=True),
            "window_end_sec": np.array(archive["window_end_sec"], dtype=np.float64, copy=True),
            "bis_target": np.array(archive["bis_target"], dtype=np.float64, copy=True),
            "case_id": int(archive["case_id"]),
            "sampling_rate_hz": float(archive["sampling_rate_hz"]),
            "sample_interval_sec": float(archive["sample_interval_sec"]),
            "bandpass_low_hz": float(archive["bandpass_low_hz"]) if "bandpass_low_hz" in archive.files else None,
            "bandpass_high_hz": float(archive["bandpass_high_hz"]) if "bandpass_high_hz" in archive.files else None,
        }


def _optional_numeric(case_dir: Path, filename: str) -> dict[str, np.ndarray] | None:
    path = case_dir / filename
    if not path.exists():
        return None
    return load_numeric(path)


def _assert_feature_row(row: dict[str, Any]) -> None:
    relative = sum(float(row[f"{name}_relative"]) for name in BAND_NAMES)
    if abs(relative - 1.0) > 1e-6:
        raise Phase2Error(f"Relative band powers sum to {relative}, not 1.")
    for column in FEATURE_COLUMNS:
        if column == "sqi_median":
            continue
        if not np.isfinite(float(row[column])):
            raise Phase2Error(f"Accepted window has a non-finite {column}.")


def _process_windows(
    case_id: int,
    filtered: np.ndarray,
    specs: list[WindowSpec],
    bis: dict[str, np.ndarray],
    sqi: dict[str, np.ndarray] | None,
    config: dict[str, Any],
    fs: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[np.ndarray]]:
    quality = config["quality"]
    features_cfg = config["features"]
    bands = _bands(config)
    delay = float(quality["bis_delay_sec"])
    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    accepted_signals: list[np.ndarray] = []
    for spec in specs:
        segment = filtered[spec.start_index : spec.stop_index]
        eeg_quality = evaluate_eeg(
            segment,
            max_missing_fraction=float(quality["max_window_missing_fraction"]),
            flatline_std_uv=float(quality["flatline_std_uv"]),
            artifact_peak_uv=float(quality["artifact_peak_uv"]),
            artifact_std_uv=float(quality["artifact_std_uv"]),
        )
        label = bis_in_window(
            spec.start_sec,
            spec.end_sec,
            bis["time"],
            bis["values"],
            min_samples=int(quality["min_bis_samples"]),
            bis_min=float(quality["bis_min"]),
            bis_max=float(quality["bis_max"]),
            delay_sec=delay,
        )
        reasons = list(eeg_quality.reasons)
        if label is None:
            reasons.append(INVALID_BIS)
        feature_row = None
        if not reasons and eeg_quality.signal is not None:
            feature_row = extract_features(
                eeg_quality.signal,
                fs,
                bands,
                nperseg=int(features_cfg["welch_nperseg"]),
                noverlap=int(features_cfg["welch_noverlap"]),
                window=str(features_cfg["welch_window"]),
                spectral_edge_fraction=float(features_cfg["spectral_edge_fraction"]),
            )
            if any(not np.isfinite(value) for value in feature_row.values()):
                reasons.append(INVALID_FEATURES)
                feature_row = None
        sqi_median = float("nan")
        if sqi is not None:
            sqi_median = median_in_window(
                spec.start_sec,
                spec.end_sec,
                sqi["time"],
                sqi["values"],
                delay_sec=delay,
            )
        if reasons or feature_row is None or eeg_quality.signal is None:
            rejected_rows.append(
                {
                    "case_id": case_id,
                    "window_index": spec.index,
                    "window_start_sec": spec.start_sec,
                    "window_end_sec": spec.end_sec,
                    "reason": ";".join(reasons) if reasons else INVALID_FEATURES,
                    "missing_fraction": eeg_quality.missing_fraction,
                    "n_missing": eeg_quality.n_missing,
                    "peak_abs_uv": eeg_quality.peak_abs_uv,
                    "eeg_std_uv": eeg_quality.std_uv,
                    "bis_n": None if label is None else label["bis_n"],
                    "bis_value": None if label is None else label["BIS_target"],
                }
            )
            continue
        row: dict[str, Any] = {
            "case_id": case_id,
            "window_index": spec.index,
            "window_start_sec": spec.start_sec,
            "window_end_sec": spec.end_sec,
            "n_samples": int(eeg_quality.signal.size),
            "BIS_target": label["BIS_target"],
            "bis_delay_sec": delay,
            "missing_fraction": eeg_quality.missing_fraction,
            "n_missing": eeg_quality.n_missing,
            "peak_abs_uv": eeg_quality.peak_abs_uv,
            "bis_n": label["bis_n"],
            "bis_first_sec": label["bis_first_sec"],
            "bis_last_sec": label["bis_last_sec"],
            "sqi_median": sqi_median,
            "sampling_rate_hz": fs,
        }
        row.update(feature_row)
        _assert_feature_row(row)
        accepted_rows.append(row)
        accepted_signals.append(np.asarray(eeg_quality.signal, dtype=np.float32))
    return accepted_rows, rejected_rows, accepted_signals


def _window_archive(
    case_id: int,
    rows: list[dict[str, Any]],
    signals: list[np.ndarray],
    n_samples: int,
    fs: float,
    interval: float,
    low_hz: float,
    high_hz: float,
) -> dict[str, np.ndarray]:
    if signals:
        eeg = np.stack(signals).astype(np.float32, copy=False)
        window_index = np.asarray([row["window_index"] for row in rows], dtype=np.int32)
        start = np.asarray([row["window_start_sec"] for row in rows], dtype=np.float64)
        end = np.asarray([row["window_end_sec"] for row in rows], dtype=np.float64)
        target = np.asarray([row["BIS_target"] for row in rows], dtype=np.float64)
    else:
        eeg = np.zeros((0, n_samples), dtype=np.float32)
        window_index = np.zeros(0, dtype=np.int32)
        start = np.zeros(0, dtype=np.float64)
        end = np.zeros(0, dtype=np.float64)
        target = np.zeros(0, dtype=np.float64)
    return {
        "eeg": eeg,
        "window_index": window_index,
        "window_start_sec": start,
        "window_end_sec": end,
        "bis_target": target,
        "case_id": np.int32(case_id),
        "sampling_rate_hz": np.float64(fs),
        "sample_interval_sec": np.float64(interval),
        "bandpass_low_hz": np.float64(low_hz),
        "bandpass_high_hz": np.float64(high_hz),
    }


def _reason_counts(rejected: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    if rejected.empty:
        return counts
    for reason in rejected["reason"].astype(str):
        primary = reason.split(";", 1)[0]
        counts[primary] = counts.get(primary, 0) + 1
    return counts


def _case_summary(
    case_id: int,
    raw: np.ndarray,
    filtered: FilterResult,
    n_windows: int,
    accepted: pd.DataFrame,
    rejected: pd.DataFrame,
    fs: float,
    n_samples: int,
    hop: int,
) -> dict[str, Any]:
    raw_missing = int(np.count_nonzero(~np.isfinite(raw)))
    accepted_missing = accepted["missing_fraction"] if not accepted.empty else pd.Series(dtype=float)
    return {
        "case_id": case_id,
        "sampling_rate_hz": fs,
        "n_eeg_samples": int(raw.size),
        "n_missing_raw": raw_missing,
        "missing_fraction_raw": float(raw_missing / raw.size) if raw.size else 1.0,
        "n_missing_filtered": filtered.n_missing_output,
        "missing_fraction_filtered": float(filtered.n_missing_output / raw.size) if raw.size else 1.0,
        "n_long_gaps": filtered.n_long_gaps,
        "n_short_gaps_bridged": filtered.n_short_gaps_bridged,
        "n_edge_gaps": filtered.n_edge_gaps,
        "n_segments_filtered": filtered.n_segments_filtered,
        "n_segments_skipped": filtered.n_segments_skipped,
        "pad_edge_samples": filtered.pad_edge_samples,
        "window_samples": n_samples,
        "hop_samples": hop,
        "n_windows": n_windows,
        "n_accepted": int(len(accepted)),
        "n_rejected": int(len(rejected)),
        "primary_rejection_reasons": _reason_counts(rejected),
        "accepted_missing_fraction_mean": (
            None if accepted_missing.empty else float(accepted_missing.mean())
        ),
        "accepted_missing_fraction_max": (
            None if accepted_missing.empty else float(accepted_missing.max())
        ),
        "accepted_windows_with_any_missing": (
            0 if accepted.empty else int((accepted["n_missing"] > 0).sum())
        ),
    }


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _representative(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    pool = frame[(frame["BIS_target"] >= 40.0) & (frame["BIS_target"] <= 60.0)]
    if pool.empty:
        pool = frame
    target = float(pool["peak_abs_uv"].median())
    position = (pool["peak_abs_uv"] - target).abs().idxmin()
    return pool.loc[position]


def _save(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    figure.clf()
    plt = _pyplot()
    plt.close(figure)


def plot_case(
    case_id: int,
    raw: np.ndarray,
    filtered: np.ndarray,
    t_start: float,
    interval: float,
    bis: dict[str, np.ndarray],
    accepted: pd.DataFrame,
    destination: Path,
    config: dict[str, Any],
) -> list[Path]:
    """Write the five Phase 2 figures for one case. Returns the file paths."""
    plt = _pyplot()
    destination.mkdir(parents=True, exist_ok=True)
    fs = 1.0 / interval
    n_window = int(config["windows"]["expected_window_samples"])
    prefix = destination / f"case_{case_id:05d}"
    paths = [
        prefix.with_name(prefix.name + "_raw_vs_filtered.png"),
        prefix.with_name(prefix.name + "_psd.png"),
        prefix.with_name(prefix.name + "_band_power.png"),
        prefix.with_name(prefix.name + "_bis_vs_time.png"),
        prefix.with_name(prefix.name + "_bis_distribution.png"),
    ]
    chosen = _representative(accepted)
    if chosen is None:
        finite = np.flatnonzero(np.isfinite(raw))
        start_index = int(finite[0]) if finite.size else 0
    else:
        start_index = int(round((float(chosen["window_start_sec"]) - t_start) / interval))
    start_index = max(0, min(start_index, max(0, raw.size - n_window)))
    stop_index = min(raw.size, start_index + n_window)
    time = t_start + np.arange(start_index, stop_index) * interval

    figure, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
    axes[0].plot(time, raw[start_index:stop_index], color="#1f4e79", linewidth=0.7)
    axes[0].set_ylabel("Raw EEG (uV)")
    axes[0].set_title(f"Case {case_id}: raw EEG")
    axes[1].plot(time, filtered[start_index:stop_index], color="#0f766e", linewidth=0.7)
    axes[1].set_ylabel("Filtered EEG (uV)")
    axes[1].set_xlabel("Time from case start (s)")
    axes[1].set_title("Same window after the 0.5–45 Hz Butterworth bandpass")
    figure.suptitle("Educational display of monitor data, not a clinical depth reading.", fontsize=10)
    _save(figure, paths[0])

    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    excerpt = filtered[start_index:stop_index]
    finite_excerpt = np.asarray(excerpt, dtype=np.float64)
    if finite_excerpt.size >= int(config["features"]["welch_nperseg"]) and np.all(np.isfinite(finite_excerpt)):
        from src.features import dominant_frequency, welch_psd

        freqs, psd = welch_psd(
            finite_excerpt,
            fs,
            int(config["features"]["welch_nperseg"]),
            int(config["features"]["welch_noverlap"]),
            str(config["features"]["welch_window"]),
        )
        axis.semilogy(freqs, np.maximum(psd, 1e-8), color="#1f4e79", linewidth=1.0)
        peak = dominant_frequency(
            finite_excerpt,
            fs,
            int(config["features"]["welch_nperseg"]),
            int(config["features"]["welch_noverlap"]),
            str(config["features"]["welch_window"]),
        )
        axis.axvline(peak, color="#9a3412", linewidth=1.0, label=f"Peak {peak:.1f} Hz")
        colors = {
            "delta": "#1d4e89",
            "theta": "#0f766e",
            "alpha": "#b45309",
            "beta": "#9a3412",
            "gamma": "#6b21a8",
        }
        for name, (low, high) in config["features"]["bands"].items():
            axis.axvspan(float(low), float(high), color=colors.get(name, "#94a3b8"), alpha=0.12, label=name)
        axis.legend(ncol=3, fontsize=8)
        axis.set_xlim(0, 50)
    else:
        axis.text(0.5, 0.5, "No finite filtered window for a spectrum", ha="center", va="center")
        axis.axis("off")
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("PSD (uV^2/Hz)")
    axis.set_title(f"Case {case_id}: Welch PSD of the filtered window")
    _save(figure, paths[1])

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    if accepted.empty:
        axis.text(0.5, 0.5, "No accepted windows", ha="center", va="center")
        axis.axis("off")
    else:
        relative = [accepted[f"{name}_relative"].to_numpy(dtype=float) for name in BAND_NAMES]
        axis.boxplot(relative, tick_labels=list(BAND_NAMES), showfliers=False)
        axis.set_ylabel("Relative power")
        axis.set_ylim(0, 1)
    axis.set_title(f"Case {case_id}: relative band power of accepted windows")
    _save(figure, paths[2])

    figure, axis = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    axis.plot(bis["time"], bis["values"], color="#9a3412", linewidth=0.8, label="Recorded BIS")
    if not accepted.empty:
        centers = (accepted["window_start_sec"] + accepted["window_end_sec"]) / 2.0
        axis.plot(
            centers,
            accepted["BIS_target"],
            color="#1f4e79",
            linewidth=1.0,
            label="Window median BIS",
        )
    axis.axhspan(40, 60, color="#d9f2d9", alpha=0.7, zorder=0)
    axis.set_ylim(0, 100)
    axis.set_xlabel("Time from case start (s)")
    axis.set_ylabel("BIS")
    axis.set_title(f"Case {case_id}: BIS and the median label inside each accepted EEG window")
    axis.legend(loc="upper right")
    _save(figure, paths[3])

    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    if accepted.empty:
        axis.text(0.5, 0.5, "No accepted windows", ha="center", va="center")
        axis.axis("off")
    else:
        axis.hist(
            accepted["BIS_target"],
            bins=np.arange(0, 102, 2),
            color="#1f4e79",
            edgecolor="white",
        )
        axis.set_xlim(0, 100)
        axis.set_xlabel("BIS target")
        axis.set_ylabel("Accepted windows")
    axis.set_title(f"Case {case_id}: distribution of BIS labels")
    _save(figure, paths[4])
    return paths


def process_case(case_id: int, config: dict[str, Any], figures_dir: Path) -> dict[str, Any]:
    raw_dir = _under_root(config["paths"]["raw_dir"])
    case_dir = raw_dir / f"case_{case_id:05d}"
    eeg_path = case_dir / "eeg_primary.npz"
    bis_path = case_dir / "bis.npz"
    if not eeg_path.exists() or not bis_path.exists():
        raise Phase2Error(
            f"Case {case_id} has no cached EEG/BIS archive in {case_dir}. Run Phase 1 first."
        )
    waveform = load_waveform(eeg_path)
    bis = load_numeric(bis_path)
    sqi = _optional_numeric(case_dir, "BIS__SQI.npz")
    raw = np.asarray(waveform["values"], dtype=np.float64)
    interval = float(waveform["sample_interval_sec"])
    fs = float(1.0 / interval)
    expected_fs = float(config["windows"]["expected_sampling_rate_hz"])
    if abs(fs - expected_fs) / expected_fs > 0.02:
        raise Phase2Error(f"Case {case_id} sampling rate is {fs} Hz, expected about {expected_fs} Hz.")
    pre = config["preprocessing"]
    filtered = bandpass_filter(
        raw,
        fs,
        float(pre["bandpass_low_hz"]),
        float(pre["bandpass_high_hz"]),
        int(pre["butterworth_order"]),
        float(pre["max_interpolate_gap_sec"]),
    )
    n_window, hop, specs = window_specs(
        raw.size,
        float(waveform["t_start_sec"]),
        interval,
        float(config["windows"]["length_sec"]),
        float(config["windows"]["overlap_fraction"]),
    )
    expected_samples = int(config["windows"]["expected_window_samples"])
    if n_window != expected_samples:
        raise Phase2Error(
            f"Case {case_id} windows have {n_window} samples; Phase 2 expects {expected_samples}."
        )
    accepted_rows, rejected_rows, signals = _process_windows(
        case_id,
        filtered.values,
        specs,
        bis,
        sqi,
        config,
        fs,
    )
    accepted = pd.DataFrame(accepted_rows, columns=FEATURE_COLUMNS)
    rejected = pd.DataFrame(rejected_rows, columns=REJECTION_COLUMNS)
    summary = _case_summary(case_id, raw, filtered, len(specs), accepted, rejected, fs, n_window, hop)
    archive = _window_archive(
        case_id,
        accepted_rows,
        signals,
        n_window,
        fs,
        interval,
        float(pre["bandpass_low_hz"]),
        float(pre["bandpass_high_hz"]),
    )
    figures = plot_case(
        case_id,
        raw,
        filtered.values,
        float(waveform["t_start_sec"]),
        interval,
        bis,
        accepted,
        figures_dir,
        config,
    )
    summary["figures"] = [str(path) for path in figures]
    logger.info(
        "Case %s: %s accepted, %s rejected (%s).",
        case_id,
        summary["n_accepted"],
        summary["n_rejected"],
        summary["primary_rejection_reasons"] or "none",
    )
    return {"accepted": accepted, "rejected": rejected, "summary": summary, "archive": archive}


def run_phase2(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Process the configured cases and write the Phase 2 tables, windows, and plots."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    resolved = config or load_config()
    accepted_ids = _accepted_case_ids(resolved)
    case_ids = [int(case_id) for case_id in resolved["phase2"]["case_ids"]]
    if not case_ids:
        raise Phase2Error("phase2.case_ids is empty.")
    figures_dir = _under_root(resolved["phase2"]["figures_dir"])
    windows_dir = _under_root(resolved["phase2"]["windows_dir"])
    windows_dir.mkdir(parents=True, exist_ok=True)

    accepted_frames = []
    rejected_frames = []
    case_summaries = []
    window_files = []
    for case_id in case_ids:
        if accepted_ids is not None and case_id not in accepted_ids:
            raise Phase2Error(f"Case {case_id} is not an accepted Phase 1 case.")
        logger.info("Processing case %s from the Phase 1 cache.", case_id)
        result = process_case(case_id, resolved, figures_dir)
        window_path = windows_dir / f"case_{case_id:05d}.npz"
        save_processed_windows(window_path, result["archive"])
        loaded = load_processed_windows(window_path)
        if loaded["eeg"].shape[0] != result["summary"]["n_accepted"]:
            raise Phase2Error(f"Case {case_id} window archive does not match the accepted-row count.")
        if loaded["eeg"].shape[0] and loaded["eeg"].shape[1] != result["summary"]["window_samples"]:
            raise Phase2Error(f"Case {case_id} saved windows are not the expected length.")
        accepted_frames.append(result["accepted"])
        rejected_frames.append(result["rejected"])
        case_summaries.append(result["summary"])
        window_files.append(
            {
                "case_id": case_id,
                "path": str(window_path),
                "n_windows": int(loaded["eeg"].shape[0]),
                "n_samples": int(loaded["eeg"].shape[1]) if loaded["eeg"].ndim == 2 else 0,
                "sampling_rate_hz": loaded["sampling_rate_hz"],
            }
        )

    features = pd.concat(accepted_frames, ignore_index=True) if accepted_frames else pd.DataFrame(columns=FEATURE_COLUMNS)
    rejections = pd.concat(rejected_frames, ignore_index=True) if rejected_frames else pd.DataFrame(columns=REJECTION_COLUMNS)
    features_path = _under_root(resolved["phase2"]["features_csv"])
    rejections_path = _under_root(resolved["phase2"]["rejections_csv"])
    summary_path = _under_root(resolved["phase2"]["summary_json"])
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(features_path, index=False)
    rejections.to_csv(rejections_path, index=False)

    total_accepted = int(len(features))
    total_rejected = int(len(rejections))
    sample_rows = []
    for case_id in case_ids:
        subset = features[features["case_id"] == case_id]
        if subset.empty:
            continue
        sample_rows.append(_clean(subset.iloc[0].to_dict()))
    summary = {
        "bis_label": (
            "Median of BIS samples in [window_start + delay, window_end + delay) "
            "with values inside the configured BIS range."
        ),
        "bis_delay_sec": float(resolved["quality"]["bis_delay_sec"]),
        "spectral_edge_fraction": float(resolved["features"]["spectral_edge_fraction"]),
        "total_accepted": total_accepted,
        "total_rejected": total_rejected,
        "features_csv": str(features_path),
        "rejections_csv": str(rejections_path),
        "windows": window_files,
        "cases": case_summaries,
        "sample_rows": sample_rows,
    }
    summary_path.write_text(json.dumps(_clean(summary), indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s (%s windows).", features_path, total_accepted)
    return summary


def main() -> None:
    summary = run_phase2()
    print(f"Usable EEG windows: {summary['total_accepted']}")
    print(f"Rejected windows: {summary['total_rejected']}")
    print(f"CSV: {summary['features_csv']}")


if __name__ == "__main__":
    main()
