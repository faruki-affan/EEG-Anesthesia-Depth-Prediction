"""Run waveform-only CNN inference on a locally cached VitalDB case."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.cnn import load_checkpoint, predict_windows
from src.ml_data import resolve_path
from src.phase2 import Phase2Error, load_processed_windows
from src.preprocess import bandpass_filter
from src.vitaldb_loader import load_config, load_numeric, load_waveform
from src.windows import bis_in_window, evaluate_eeg, window_specs


def case_paths(config, case_id):
    name = f"case_{case_id:05d}"
    return (
        resolve_path(config["paths"]["raw_dir"]) / name,
        resolve_path(config["phase2"]["windows_dir"]) / f"{name}.npz",
    )


def filter_wave(wave, config):
    pre = config["preprocessing"]
    return bandpass_filter(
        wave["values"], 1 / wave["sample_interval_sec"], pre["bandpass_low_hz"],
        pre["bandpass_high_hz"], pre["butterworth_order"], pre["max_interpolate_gap_sec"],
    ).values


def predict_case(case_id, config):
    models = resolve_path(config["phase4"]["models_dir"])
    path = models / f"test_case_{case_id:05d}.pt"
    paths = [path] if path.exists() else sorted(models.glob("test_case_*.pt"))
    if not paths:
        raise ValueError("Run python -m src.train_cnn to create CNN checkpoints.")
    checkpoints = [load_checkpoint(path) for path in paths]
    signal_config = checkpoints[0][1]["signal_config"]
    if any(payload["signal_config"] != signal_config for _, payload in checkpoints):
        raise ValueError("CNN checkpoints have conflicting signal-processing configurations.")
    raw_dir, archive_path = case_paths(config, case_id)
    raw_path = raw_dir / "eeg_primary.npz"
    if raw_path.exists():
        frame, eeg, fs = raw_windows(raw_path, raw_dir / "bis.npz", signal_config)
        source = "Raw EEG windowed for inference"
    else:
        archive = load_processed_windows(archive_path)
        for edge in ("low", "high"):
            key = f"bandpass_{edge}_hz"
            if archive[key] != signal_config["preprocessing"][key]:
                raise ValueError("Processed window filter differs from the CNN checkpoint.")
        frame = pd.DataFrame({
            "window_index": archive["window_index"],
            "Window end (s)": archive["window_end_sec"],
            "Recorded BIS": archive["bis_target"],
        })
        eeg, fs = archive["eeg"], archive["sampling_rate_hz"]
        source = "Bundled EEG windows with recorded BIS"
    if frame.empty:
        raise ValueError("This recording has no usable EEG windows.")
    frame["Predicted BIS"] = np.mean(
        [predict_windows(checkpoint, eeg, fs) for checkpoint in checkpoints], axis=0,
    )
    model = "Held-out CNN checkpoint" if path.exists() else f"Ensemble of {len(checkpoints)} fold checkpoints"
    return frame.sort_values("Window end (s)"), f"{source} · {model}"


def raw_windows(raw_path, bis_path, config):
    wave = load_waveform(raw_path)
    interval = wave["sample_interval_sec"]
    filtered = filter_wave(wave, config)
    windows = config["windows"]
    samples, _, specs = window_specs(
        filtered.size, wave["t_start_sec"], interval, windows["length_sec"], windows["overlap_fraction"],
    )
    quality = config["quality"]
    bis = load_numeric(bis_path) if bis_path.exists() else None
    records, eeg = [], []
    for spec in specs:
        signal = evaluate_eeg(
            filtered[spec.start_index:spec.stop_index],
            max_missing_fraction=quality["max_window_missing_fraction"],
            flatline_std_uv=quality["flatline_std_uv"],
            artifact_peak_uv=quality["artifact_peak_uv"],
            artifact_std_uv=quality["artifact_std_uv"],
        ).signal
        if signal is None:
            continue
        actual = None
        if bis is not None:
            label = bis_in_window(
                spec.start_sec, spec.end_sec, bis["time"], bis["values"],
                min_samples=quality["min_bis_samples"], bis_min=quality["bis_min"],
                bis_max=quality["bis_max"], delay_sec=quality["bis_delay_sec"],
            )
            actual = None if label is None else label["BIS_target"]
        records.append({
            "window_index": spec.index, "Window end (s)": spec.end_sec, "Recorded BIS": actual,
        })
        eeg.append(signal)
    frame = pd.DataFrame(records, columns=["window_index", "Window end (s)", "Recorded BIS"])
    if not eeg:
        return frame, np.empty((0, samples)), 1 / interval
    return frame, np.asarray(eeg, dtype=np.float32), 1 / interval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id", type=int)
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV path")
    args = parser.parse_args()
    config = load_config()
    torch.set_num_threads(config["phase4"]["cpu_threads"])
    try:
        frame, source = predict_case(args.case_id, config)
    except (OSError, ValueError, Phase2Error) as error:
        parser.exit(1, f"{error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"{source}. Saved {len(frame)} predictions to {args.output}.")


if __name__ == "__main__":
    main()
