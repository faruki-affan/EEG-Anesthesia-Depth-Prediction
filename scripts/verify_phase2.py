"""Run Phase 2 unit tests, then process cached cases 1, 2, and 3.

From the project root:

    .venv\\Scripts\\python scripts\\verify_phase2.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import BAND_NAMES  # noqa: E402
from src.phase2 import Phase2Error, run_phase2  # noqa: E402
from src.vitaldb_loader import load_config  # noqa: E402


def run_tests() -> bool:
    suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_phase2")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()


def _problems(summary: dict, config: dict) -> list[str]:
    problems = []
    features_path = Path(summary["features_csv"])
    rejections_path = Path(summary["rejections_csv"])
    if not features_path.exists() or features_path.stat().st_size == 0:
        problems.append(f"Feature CSV was not written: {features_path}")
        return problems
    if not rejections_path.exists():
        problems.append(f"Rejection CSV was not written: {rejections_path}")

    import pandas as pd

    features = pd.read_csv(features_path)
    if len(features) != int(summary["total_accepted"]):
        problems.append("Feature CSV row count does not match the summary.")
    if features.empty:
        problems.append("No EEG window was accepted.")
        return problems
    required = [
        "case_id",
        "window_start_sec",
        "window_end_sec",
        "BIS_target",
        "spectral_entropy",
        "spectral_edge_hz",
        "mean",
        "std",
        "rms",
        "variance",
        "hjorth_mobility",
        "hjorth_complexity",
        "missing_fraction",
    ]
    for name in BAND_NAMES:
        required.extend([f"{name}_power", f"{name}_relative"])
    missing = [column for column in required if column not in features.columns]
    if missing:
        problems.append(f"Feature CSV is missing columns: {missing}")
        return problems
    relative = features[[f"{name}_relative" for name in BAND_NAMES]].sum(axis=1)
    if float((relative - 1.0).abs().max()) > 1e-6:
        problems.append("Relative band powers do not sum to 1.")
    duration = features["window_end_sec"] - features["window_start_sec"]
    if float((duration - 10.0).abs().max()) > 1e-6:
        problems.append("Accepted windows are not 10 seconds long.")
    if int(features["n_samples"].nunique()) != 1 or int(features["n_samples"].iloc[0]) != 1280:
        problems.append("Accepted windows do not have 1,280 samples.")
    if features["BIS_target"].lt(0).any() or features["BIS_target"].gt(100).any():
        problems.append("A BIS target is outside 0–100.")
    edge = features["spectral_edge_hz"]
    low = float(config["preprocessing"]["bandpass_low_hz"])
    high = float(config["preprocessing"]["bandpass_high_hz"])
    if edge.lt(low - 1e-6).any() or edge.gt(high + 1e-6).any():
        problems.append("A spectral edge falls outside the analysis band.")
    numeric = [column for column in features.columns if column != "sqi_median"]
    if features[numeric].isna().any().any():
        problems.append("An accepted feature other than SQI is missing.")
    for case in summary["cases"]:
        case_rows = features[features["case_id"] == case["case_id"]]
        if len(case_rows) != case["n_accepted"]:
            problems.append(f"Case {case['case_id']} accepted-row count does not match the CSV.")
        ordered = case_rows.sort_values("window_start_sec")
        if len(ordered) >= 2:
            hops = ordered["window_start_sec"].diff().dropna().to_numpy(dtype=float)
            # Rejected windows leave gaps, so hops are positive multiples of 5 s.
            steps = hops / 5.0
            if float(np.max(np.abs(steps - np.round(steps)))) > 1e-6 or float(hops.min()) < 5.0 - 1e-6:
                problems.append(f"Case {case['case_id']} window starts are not on the 5-second hop.")
    for window in summary["windows"]:
        path = Path(window["path"])
        if not path.exists():
            problems.append(f"Window archive is missing: {path}")
        elif window["n_windows"] <= 0:
            problems.append(f"Case {window['case_id']} saved no CNN windows.")
    for case in summary["cases"]:
        for figure in case["figures"]:
            path = Path(figure)
            if not path.exists() or path.stat().st_size < 5_000:
                problems.append(f"Plot was not written: {path}")
    return problems


def _print_report(summary: dict) -> None:
    print("VitalDB Phase 2 check")
    print(f"  feature CSV: {summary['features_csv']}")
    print(f"  rejection CSV: {summary['rejections_csv']}")
    print(f"  usable EEG windows: {summary['total_accepted']}")
    print(f"  rejected windows: {summary['total_rejected']}")
    print(f"  BIS label: {summary['bis_label']}")
    print(f"  BIS delay: {summary['bis_delay_sec']} s")
    print("  missing data and window counts:")
    for case in summary["cases"]:
        print(
            f"    case {case['case_id']}: accepted {case['n_accepted']}, "
            f"rejected {case['n_rejected']}, reasons {case['primary_rejection_reasons'] or '{}'}"
        )
        print(
            f"      raw missing {case['missing_fraction_raw']:.4%} "
            f"({case['n_missing_raw']} samples), "
            f"filtered missing {case['missing_fraction_filtered']:.4%} "
            f"({case['n_missing_filtered']} samples)"
        )
        print(
            f"      long gaps {case['n_long_gaps']}, short gaps bridged {case['n_short_gaps_bridged']}, "
            f"segments filtered {case['n_segments_filtered']}, "
            f"segments skipped {case['n_segments_skipped']}"
        )
        print(
            f"      accepted-window missing mean {case['accepted_missing_fraction_mean']}, "
            f"max {case['accepted_missing_fraction_max']}, "
            f"windows with any missing {case['accepted_windows_with_any_missing']}"
        )
    print("  sample accepted window from each case:")
    keys = [
        "case_id",
        "window_start_sec",
        "window_end_sec",
        "delta_power",
        "theta_power",
        "alpha_power",
        "beta_power",
        "gamma_power",
        "delta_relative",
        "theta_relative",
        "alpha_relative",
        "beta_relative",
        "gamma_relative",
        "spectral_entropy",
        "spectral_edge_hz",
        "mean",
        "std",
        "rms",
        "variance",
        "hjorth_mobility",
        "hjorth_complexity",
        "BIS_target",
        "missing_fraction",
        "sqi_median",
    ]
    for row in summary["sample_rows"]:
        print(f"    case {row['case_id']} window {row['window_index']} at {row['window_start_sec']} s")
        for key in keys:
            if key in {"case_id", "window_start_sec", "window_end_sec"}:
                continue
            print(f"      {key}: {row.get(key)}")


def main() -> int:
    print("Phase 2 unit tests")
    tests_ok = run_tests()
    print(f"Unit tests: {'passed' if tests_ok else 'failed'}")
    config = load_config(ROOT / "config.yaml")
    try:
        summary = run_phase2(config)
    except Phase2Error as exc:
        print(f"Phase 2 failed: {exc}")
        return 1
    _print_report(summary)
    problems = _problems(summary, config)
    if problems or not tests_ok:
        print("Checks failed:")
        if not tests_ok:
            print("  - unit tests failed")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("Checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
