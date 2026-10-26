"""Feature tables, leave-one-case-out splits, and the Phase 3 quality report.

SQI is used only to describe signal quality and to run a sensitivity analysis.
It is not a model input. BIS is the regression target, not a feature.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.vitaldb_loader import PROJECT_ROOT, load_numeric
from src.windows import median_in_window

FREQUENCY_FEATURES = (
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
)

TIME_FEATURES = (
    "mean",
    "std",
    "rms",
    "variance",
    "hjorth_mobility",
    "hjorth_complexity",
)

MODEL_FEATURE_ORDER = FREQUENCY_FEATURES + TIME_FEATURES

# Requested in the phase brief, but not stored by Phase 2. They are not invented.
ABSENT_FEATURES = {
    "total_power": (
        "Total spectral power is not a column. The five absolute band powers "
        "already partition the in-band spectrum, so no summed column is added."
    ),
    "skewness": "Skewness was not extracted in Phase 2.",
    "kurtosis": "Kurtosis was not extracted in Phase 2.",
    "hjorth_activity": (
        "Hjorth activity is the signal variance. That value is already stored "
        "as variance, and it is not duplicated under another name."
    ),
}

FORBIDDEN_INPUTS = frozenset(
    {
        "bis",
        "bis_target",
        "sqi",
        "sqi_median",
        "emg",
        "sr",
        "suppression_ratio",
        "case_id",
        "subject_id",
        "window_index",
        "window_start_sec",
        "window_end_sec",
        "bis_delay_sec",
        "bis_n",
        "bis_first_sec",
        "bis_last_sec",
        "sampling_rate_hz",
        "n_samples",
        "missing_fraction",
        "n_missing",
        "peak_abs_uv",
        "timestamp",
        "time",
    }
)

MODEL_KEYS = ("median", "random_forest", "hist_gradient_boosting")
MODEL_LABELS = {
    "median": "Median baseline",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
}

# The three-case plan required for this phase. Leave-one-case-out on exactly
# these ids produces the same folds.
EXPECTED_CASES = (1, 2, 3)


class Phase3DataError(Exception):
    """The processed feature table cannot be used for this phase."""


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def feature_group(name: str) -> str:
    if name in FREQUENCY_FEATURES:
        return "frequency"
    if name in TIME_FEATURES:
        return "time"
    return "other"


def _forbidden_tokens(name: str) -> bool:
    tokens = set(name.lower().replace("/", "_").split("_"))
    return bool(tokens & {"bis", "sqi", "emg", "sr"})


def select_model_features(columns: Sequence[str]) -> list[str]:
    """Return EEG features in training order.

    Raises if a required Phase 2 column is missing. Monitor outputs, BIS,
    timestamps, and case identifiers are never selected.
    """
    available = set(columns)
    blocked = [name for name in MODEL_FEATURE_ORDER if name in FORBIDDEN_INPUTS or _forbidden_tokens(name)]
    if blocked:
        raise Phase3DataError(f"Model feature list contains forbidden columns: {blocked}")
    missing = [name for name in MODEL_FEATURE_ORDER if name not in available]
    if missing:
        raise Phase3DataError(
            "The processed table is missing EEG features required by Phase 3: " + ", ".join(missing)
        )
    return list(MODEL_FEATURE_ORDER)


def leave_one_case_out(case_ids: Sequence[int]) -> list[tuple[int, tuple[int, ...]]]:
    """Each fold holds out one entire case.

    Returns ``(test_case, training_cases)`` in ascending test-case order.
    For cases 1, 2, and 3 that is the required three-experiment plan.
    """
    ids = tuple(sorted({int(case_id) for case_id in case_ids}))
    if not ids:
        raise Phase3DataError("No case ids are available for cross-validation.")
    return [(test_case, tuple(case_id for case_id in ids if case_id != test_case)) for test_case in ids]


def assert_case_separation(train_cases: Sequence[int], test_cases: Sequence[int]) -> None:
    train = {int(case_id) for case_id in train_cases}
    test = {int(case_id) for case_id in test_cases}
    overlap = sorted(train & test)
    if overlap:
        raise Phase3DataError(f"Training and test cases overlap: {overlap}")
    if not test:
        raise Phase3DataError("The test split has no case.")
    if not train:
        raise Phase3DataError("The training split has no case.")


def load_feature_table(path: str | Path) -> pd.DataFrame:
    csv_path = resolve_path(path)
    if not csv_path.exists():
        raise Phase3DataError(f"Processed feature CSV was not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    if "case_id" in frame.columns:
        frame["case_id"] = frame["case_id"].astype(int)
    return frame


def _distribution(values: pd.Series) -> dict[str, Any]:
    series = pd.to_numeric(values, errors="coerce")
    finite = series[np.isfinite(series.to_numpy())]
    if finite.empty:
        return {"n": 0}
    quantiles = finite.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": int(finite.shape[0]),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=0)),
        "min": float(finite.min()),
        "p05": float(quantiles.loc[0.05]),
        "p25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "p95": float(quantiles.loc[0.95]),
        "max": float(finite.max()),
    }


def _bis_histogram(values: pd.Series) -> dict[str, int]:
    bins = [0, 20, 40, 60, 80, 100]
    labels = ["0–20", "20–40", "40–60", "60–80", "80–100"]
    categories = pd.cut(values, bins=bins, labels=labels, include_lowest=True, right=True)
    counts = categories.value_counts().reindex(labels, fill_value=0)
    return {str(label): int(counts.loc[label]) for label in labels}


def _sqi_histogram(values: pd.Series) -> dict[str, int]:
    bins = [0, 20, 50, 80, 100]
    labels = ["0–20", "20–50", "50–80", "80–100"]
    categories = pd.cut(values, bins=bins, labels=labels, include_lowest=True, right=True)
    counts = categories.value_counts().reindex(labels, fill_value=0)
    return {str(label): int(counts.loc[label]) for label in labels}


def _load_case_sr(case_id: int) -> dict[str, np.ndarray] | None:
    path = PROJECT_ROOT / "data" / "raw" / f"case_{int(case_id):05d}" / "BIS__SR.npz"
    if not path.exists():
        return None
    loaded = load_numeric(path)
    return {"time": np.asarray(loaded["time"]), "values": np.asarray(loaded["values"])}


def _classify_zero_bis(
    row: pd.Series,
    case_frame: pd.DataFrame,
    sr_median: float | None,
) -> str:
    """Heuristic label for a BIS = 0 window. This is not a clinical diagnosis.

    Genuine suppression is only suggested when amplitude is low, the monitor
    suppression ratio is high, and SQI is high enough to trust the monitor.
    Low SQI at the start or end of a recording is treated as an unreliable
    BIS reading. The window is still kept for the primary model.
    """
    sqi = float(row["sqi_median"])
    amplitude = float(row["std"])
    start = float(row["window_start_sec"])
    positive = case_frame.loc[case_frame["BIS_target"] > 0, "window_start_sec"]
    first_positive = float(positive.min()) if not positive.empty else None
    earlier = case_frame.loc[case_frame["window_start_sec"] < start - 1e-9, "BIS_target"]
    previous_bis = float(earlier.iloc[-1]) if not earlier.empty else None
    sr_known = sr_median is not None and np.isfinite(sr_median)
    if amplitude < 5.0 and sqi >= 50.0 and sr_known and float(sr_median) >= 50.0:
        return "possible_genuine_suppression"
    if first_positive is not None and start < first_positive and sqi < 50.0:
        return "monitor_startup_unreliable"
    if sqi < 20.0 and amplitude < 5.0 and previous_bis is not None and previous_bis > 60.0:
        return "end_of_recording_or_dropout_unreliable"
    if sqi < 50.0:
        return "unreliable_low_signal_quality"
    return "retained_for_review"


def _zero_bis_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    zeros = frame.loc[frame["BIS_target"] == 0].copy()
    records: list[dict[str, Any]] = []
    sr_cache: dict[int, dict[str, np.ndarray] | None] = {}
    for case_id, case_frame in frame.groupby("case_id", sort=True):
        case_frame = case_frame.sort_values("window_start_sec")
        case_zeros = zeros.loc[zeros["case_id"] == case_id]
        if case_zeros.empty:
            continue
        if int(case_id) not in sr_cache:
            sr_cache[int(case_id)] = _load_case_sr(int(case_id))
        sr = sr_cache[int(case_id)]
        for _, row in case_zeros.sort_values("window_start_sec").iterrows():
            sr_median = None
            if sr is not None:
                sr_median = float(
                    median_in_window(
                        float(row["window_start_sec"]),
                        float(row["window_end_sec"]),
                        sr["time"],
                        sr["values"],
                        0.0,
                    )
                )
                if not np.isfinite(sr_median):
                    sr_median = None
            records.append(
                {
                    "case_id": int(row["case_id"]),
                    "window_index": int(row["window_index"]),
                    "window_start_sec": float(row["window_start_sec"]),
                    "window_end_sec": float(row["window_end_sec"]),
                    "sqi_median": float(row["sqi_median"]),
                    "std_uv": float(row["std"]),
                    "spectral_edge_hz": float(row["spectral_edge_hz"]),
                    "delta_relative": float(row["delta_relative"]),
                    "sr_median": sr_median,
                    "classification": _classify_zero_bis(row, case_frame, sr_median),
                }
            )
    return records


def _timestamp_audit(frame: pd.DataFrame) -> dict[str, Any]:
    issues: list[str] = []
    if frame.duplicated(["case_id", "window_index"]).any():
        issues.append("Duplicate case_id and window_index rows are present.")
    if frame.duplicated(["case_id", "window_start_sec"]).any():
        issues.append("Duplicate case_id and window_start_sec rows are present.")
    gaps = []
    for case_id, case_frame in frame.groupby("case_id", sort=True):
        ordered = case_frame.sort_values(["window_start_sec", "window_index"])
        starts = ordered["window_start_sec"].to_numpy(dtype=float)
        if starts.size > 1 and np.any(np.diff(starts) < -1e-9):
            issues.append(f"Case {int(case_id)} has a decreasing window start time.")
        duration = ordered["window_end_sec"].to_numpy(dtype=float) - starts
        if np.any(np.abs(duration - 10.0) > 1e-6):
            issues.append(f"Case {int(case_id)} has a window whose duration is not 10 seconds.")
        hops = np.diff(starts) if starts.size > 1 else np.asarray([], dtype=float)
        long_hops = hops[hops > 5.0 + 1e-6]
        gaps.append(
            {
                "case_id": int(case_id),
                "n_windows": int(len(ordered)),
                "n_hops_longer_than_5s": int(long_hops.size),
                "max_hop_sec": None if hops.size == 0 else float(np.max(hops)),
            }
        )
    return {
        "issues": issues,
        "gaps_after_rejected_windows": gaps,
        "note": (
            "Accepted windows use a 5-second hop. A longer hop means Phase 2 "
            "rejected one or more intervening windows. Those gaps are not "
            "duplicate or reordered timestamps."
        ),
    }


def build_quality_report(frame: pd.DataFrame, expected_windows: int, expected_cases: Sequence[int]) -> dict[str, Any]:
    """Inspect the processed CSV without dropping rows."""
    critical: list[str] = []
    notes: list[str] = []
    columns = [str(column) for column in frame.columns]
    numeric = frame.select_dtypes(include=[np.number])
    non_finite = {}
    for column in numeric.columns:
        values = numeric[column].to_numpy()
        n_bad = int((~np.isfinite(values)).sum()) if values.size else 0
        if n_bad:
            non_finite[str(column)] = n_bad
    if non_finite:
        critical.append(f"Non-finite values are present: {non_finite}")

    n_rows = int(len(frame))
    if n_rows != int(expected_windows):
        critical.append(f"Expected {expected_windows} accepted windows, found {n_rows}.")

    case_counts = {int(case_id): int(count) for case_id, count in frame["case_id"].value_counts().sort_index().items()} if "case_id" in frame.columns else {}
    expected_case_set = {int(case_id) for case_id in expected_cases}
    if set(case_counts) != expected_case_set:
        critical.append(f"Expected cases {sorted(expected_case_set)}, found {sorted(case_counts)}.")

    try:
        selected = select_model_features(columns)
    except Phase3DataError as exc:
        selected = []
        critical.append(str(exc))

    if "BIS_target" not in frame.columns:
        critical.append("BIS_target is missing.")
        bis = pd.Series(dtype=float)
    else:
        bis = frame["BIS_target"]
        outside = int((~bis.between(0, 100)).sum())
        if outside:
            critical.append(f"{outside} BIS targets are outside 0–100.")

    timestamps = _timestamp_audit(frame) if {"case_id", "window_index", "window_start_sec", "window_end_sec"} <= set(columns) else {"issues": ["Timestamp columns are missing."], "gaps_after_rejected_windows": []}
    critical.extend(timestamps["issues"])

    zero_records = _zero_bis_records(frame) if "BIS_target" in frame.columns and n_rows else []
    zero_counts: dict[str, int] = {}
    for record in zero_records:
        zero_counts[record["classification"]] = zero_counts.get(record["classification"], 0) + 1

    per_case = []
    if "case_id" in frame.columns and "BIS_target" in frame.columns:
        for case_id, case_frame in frame.groupby("case_id", sort=True):
            per_case.append(
                {
                    "case_id": int(case_id),
                    "n_windows": int(len(case_frame)),
                    "bis": _distribution(case_frame["BIS_target"]),
                    "bis_histogram": _bis_histogram(case_frame["BIS_target"]),
                    "sqi": _distribution(case_frame["sqi_median"]) if "sqi_median" in case_frame.columns else {},
                    "sqi_histogram": _sqi_histogram(case_frame["sqi_median"]) if "sqi_median" in case_frame.columns else {},
                    "n_bis_zero": int((case_frame["BIS_target"] == 0).sum()),
                }
            )

    if not zero_records:
        notes.append("No accepted window has BIS = 0.")
    else:
        notes.append(
            f"{len(zero_records)} windows have BIS = 0. They are retained in the primary models. "
            "SQI is not used as an input. Classifications are heuristics for the quality report."
        )
    notes.append("No rows were removed while building this report.")

    return {
        "n_rows": n_rows,
        "n_cases": len(case_counts),
        "case_counts": case_counts,
        "columns": columns,
        "n_model_features": len(selected),
        "model_features": selected,
        "frequency_features": [name for name in selected if name in FREQUENCY_FEATURES],
        "time_features": [name for name in selected if name in TIME_FEATURES],
        "absent_features": ABSENT_FEATURES,
        "forbidden_inputs_present_but_not_used": sorted(
            column for column in columns if column.lower() in FORBIDDEN_INPUTS or _forbidden_tokens(column)
        ),
        "non_finite_counts": non_finite,
        "duplicate_window_keys": int(frame.duplicated(["case_id", "window_index"]).sum()) if "window_index" in frame.columns else None,
        "timestamps": timestamps,
        "bis_overall": _distribution(bis) if len(bis) else {},
        "bis_histogram": _bis_histogram(bis) if len(bis) else {},
        "cases": per_case,
        "bis_zero_windows": zero_records,
        "bis_zero_classification_counts": zero_counts,
        "rows_excluded": 0,
        "exclusion_reasons": [],
        "notes": notes,
        "critical_failures": critical,
        "target": "BIS_target",
    }


def render_quality_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 3 data-quality report",
        "",
        f"- Accepted windows: {report['n_rows']}",
        f"- Cases: {report['n_cases']} ({report['case_counts']})",
        f"- Model features: {report['n_model_features']}",
        f"- Rows excluded: {report['rows_excluded']}",
        "",
        "## Feature columns used as model inputs",
        "",
    ]
    for name in report["model_features"]:
        lines.append(f"- {name} ({feature_group(name)})")
    lines.extend(["", "## Requested features that are not in the CSV", ""])
    for name, reason in report["absent_features"].items():
        lines.append(f"- `{name}`: {reason}")
    lines.extend(
        [
            "",
            "## Columns present but not used as inputs",
            "",
            "BIS, SQI, suppression ratio, EMG, timestamps, and case identifiers stay out of the feature matrix.",
            "",
        ]
    )
    for name in report["forbidden_inputs_present_but_not_used"]:
        lines.append(f"- {name}")
    lines.extend(["", "## BIS by case", ""])
    for case in report["cases"]:
        bis = case["bis"]
        lines.append(
            f"- Case {case['case_id']}: n={case['n_windows']}, "
            f"median={bis.get('median'):.2f}, mean={bis.get('mean'):.2f}, "
            f"min={bis.get('min'):.2f}, max={bis.get('max'):.2f}, "
            f"BIS=0 windows={case['n_bis_zero']}"
        )
        lines.append(f"  - Histogram: {case['bis_histogram']}")
        lines.append(f"  - SQI histogram: {case['sqi_histogram']}")
    lines.extend(["", "## BIS = 0 windows", ""])
    lines.append(
        "These windows are not removed. A low-SQI reading of 0 at the start or end of a case "
        "is treated as an unreliable monitor value. Genuine suppression would need low EEG "
        "amplitude, a high suppression ratio, and SQI high enough to trust the monitor."
    )
    lines.append("")
    if not report["bis_zero_windows"]:
        lines.append("None.")
    else:
        lines.append("| Case | Start (s) | SQI | EEG SD (µV) | SEF95 (Hz) | SR median | Classification |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for row in report["bis_zero_windows"]:
            sr = "" if row["sr_median"] is None else f"{row['sr_median']:.1f}"
            lines.append(
                f"| {row['case_id']} | {row['window_start_sec']:.1f} | {row['sqi_median']:.2f} | "
                f"{row['std_uv']:.2f} | {row['spectral_edge_hz']:.1f} | {sr} | {row['classification']} |"
            )
        lines.append("")
        lines.append(f"Counts: {report['bis_zero_classification_counts']}")
    lines.extend(["", "## Timestamps", "", report["timestamps"].get("note", "")])
    for gap in report["timestamps"].get("gaps_after_rejected_windows", []):
        lines.append(
            f"- Case {gap['case_id']}: {gap['n_hops_longer_than_5s']} hops longer than 5 s "
            f"(max hop {gap['max_hop_sec']} s)."
        )
    lines.extend(["", "## Critical failures", ""])
    if report["critical_failures"]:
        for failure in report["critical_failures"]:
            lines.append(f"- {failure}")
    else:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if value is None or isinstance(value, (str, int)):
        return value
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n", encoding="utf-8")
