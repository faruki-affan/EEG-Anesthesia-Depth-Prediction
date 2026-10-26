"""Load one real VitalDB case and plot its EEG waveform beside the BIS timeline.

Run from the project root after the Phase 1 packages are installed:

    .venv\\Scripts\\python scripts\\verify_phase1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vitaldb_loader import (  # noqa: E402
    load_config,
    load_numeric,
    load_waveform,
    project_path,
    run_phase1,
)


def _excerpt_start(values: np.ndarray, sample_interval: float, excerpt_sec: float, search_sec: float) -> int:
    """Pick a  window with continuous, non-saturated EEG.

    The first minutes of a case are often a disconnected or artifact-heavy
    stretch. ``search_sec`` is ignored when it is shorter than the recording;
    the search covers the whole waveform and keeps the first plausible window.
    """
    del search_sec
    window = max(1, int(round(excerpt_sec / sample_interval)))
    step = max(1, int(round(10.0 / sample_interval)))
    if values.size <= window:
        return 0
    fallback = 0
    fallback_std = -1.0
    for start in range(0, values.size - window, step):
        segment = values[start : start + window]
        finite = np.isfinite(segment)
        if float(finite.mean()) < 0.98:
            continue
        data = segment[finite].astype(np.float64, copy=False)
        std = float(np.std(data))
        peak = float(np.max(np.abs(data)))
        if std > fallback_std and peak < 500.0:
            fallback = start
            fallback_std = std
        if 5.0 <= std <= 40.0 and peak <= 150.0:
            return start
    return fallback


def plot_case(case_dir: Path, caseid: int, destination: Path, excerpt_sec: float, search_sec: float) -> None:
    eeg = load_waveform(case_dir / "eeg_primary.npz")
    bis = load_numeric(case_dir / "bis.npz")
    values = eeg["values"]
    interval = eeg["sample_interval_sec"]
    start = _excerpt_start(values, interval, excerpt_sec, search_sec)
    window = max(1, int(round(excerpt_sec / interval)))
    excerpt = values[start : start + window]
    excerpt_time = eeg["t_start_sec"] + (start + np.arange(excerpt.size)) * interval

    second = max(1, int(round(1.0 / interval)))
    usable = values[: (values.size // second) * second]
    blocks = usable.reshape(-1, second)
    finite = np.isfinite(blocks)
    squared = np.where(finite, blocks.astype(np.float64) ** 2, np.nan)
    rms = np.sqrt(np.nanmean(squared, axis=1))
    rms_time = eeg["t_start_sec"] + (np.arange(rms.size) + 0.5)

    figure, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    axes[0].plot(excerpt_time, excerpt, color="#1f4e79", linewidth=0.6)
    axes[0].set_title(f"Case {caseid}: {excerpt_sec:.0f} s of BIS/EEG1_WAV")
    axes[0].set_ylabel("EEG (uV)")
    axes[0].set_xlabel("Time from case start (s)")

    axes[1].plot(rms_time, rms, color="#5b7c99", linewidth=0.8)
    axes[1].set_title("Full recording, 1-second RMS of the same EEG channel")
    axes[1].set_ylabel("RMS (uV)")
    axes[1].set_xlabel("Time from case start (s)")

    axes[2].plot(bis["time"], bis["values"], color="#9a3412", linewidth=0.9)
    axes[2].set_ylim(0, 100)
    axes[2].axhspan(40, 60, color="#d9f2d9", alpha=0.7, zorder=0)
    axes[2].set_title("Recorded BIS/BIS (green band is the usual 40–60 anesthetic range)")
    axes[2].set_ylabel("BIS")
    axes[2].set_xlabel("Time from case start (s)")

    figure.suptitle(
        "VitalDB recording. Educational display of monitor data, not a clinical depth-of-anesthesia reading.",
        fontsize=10,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=120)
    plt.close(figure)


def main() -> int:
    config = load_config(ROOT / "config.yaml")
    result = run_phase1(config)
    accepted = [record for record in result["records"] if record.accepted]
    if not accepted:
        print("No case passed the EEG/BIS quality checks.")
        for record in result["records"]:
            print(f"  case {record.caseid}: {record.reason}")
        return 1

    case = accepted[0]
    raw_dir = project_path(config, "raw_dir")
    figure_dir = project_path(config, "figures_dir")
    case_dir = raw_dir / f"case_{case.caseid:05d}"
    figure_path = figure_dir / f"case_{case.caseid:05d}_eeg_bis.png"
    plot = config["plot"]
    plot_case(
        case_dir,
        case.caseid,
        figure_path,
        float(plot["excerpt_sec"]),
        float(plot["excerpt_search_sec"]),
    )

    eeg = case.eeg
    bis = case.bis
    assert eeg is not None and bis is not None
    problems = []
    if eeg.sampling_rate_hz is None or abs(eeg.sampling_rate_hz - 128.0) / 128.0 > 0.02:
        problems.append(f"EEG sampling rate is {eeg.sampling_rate_hz}, catalog value is 128 Hz.")
    if bis.median_interval_sec is None or not 0.5 <= bis.median_interval_sec <= 2.0:
        problems.append(f"BIS median interval is {bis.median_interval_sec} s; the device table lists 1 s.")
    if eeg.n_samples < 10_000 or eeg.value_std is None:
        problems.append("EEG recording does not contain a usable waveform.")
    if case.overlap_sec is None or case.overlap_sec < 600:
        problems.append(f"EEG/BIS overlap is {case.overlap_sec} s.")
    if not figure_path.exists() or figure_path.stat().st_size < 10_000:
        problems.append(f"Plot was not written to {figure_path}.")

    discovery = result["discovery"]
    print("VitalDB Phase 1 check")
    print(f"  tracks in index: {discovery['n_tracks']}")
    print(f"  cases with {config['tracks']['eeg_primary']} and {config['tracks']['bis']}: "
          f"{discovery['n_cases_with_primary_eeg_and_bis']}")
    print(f"  accepted case: {case.caseid} (subject {case.subjectid})")
    print(f"  EEG samples: {eeg.n_samples}, rate: {eeg.sampling_rate_hz} Hz, "
          f"missing: {eeg.missing_fraction:.2%}, duration: {eeg.duration_sec} s")
    print(f"  EEG header times: {eeg.finite_time_header}")
    print(f"  BIS samples: {bis.n_samples}, median interval: {bis.median_interval_sec} s, "
          f"range: {bis.value_min}–{bis.value_max}, max gap: {bis.max_gap_sec} s")
    print(f"  overlap: {case.overlap_sec} s")
    print(f"  plot: {figure_path}")
    if case.vitalfile_crosscheck:
        print(f"  vital-file cross-check: {case.vitalfile_crosscheck.get('status')}")
    if problems:
        print("Checks failed:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("Checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
