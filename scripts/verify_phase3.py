"""Check the completed Phase 3 artifacts.

From the project root, after training:

    .venv\\Scripts\\python scripts\\verify_phase3.py

This script does not retrain the models. It exits with status 1 when a
critical check fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml_data import FORBIDDEN_INPUTS, MODEL_FEATURE_ORDER, load_feature_table  # noqa: E402
from src.ml_metrics import regression_metrics  # noqa: E402
from src.predict import load_bis_model, predict_bis  # noqa: E402
from src.vitaldb_loader import load_config  # noqa: E402

REQUIRED_FIGURES = (
    "case_00001_bis_over_time.png",
    "case_00002_bis_over_time.png",
    "case_00003_bis_over_time.png",
    "pooled_actual_vs_predicted.png",
    "pooled_error_histogram.png",
    "pooled_absolute_error.png",
    "random_forest_feature_importance.png",
    "model_comparison.png",
    "bis_distribution_by_case.png",
    "performance_by_bis_range.png",
    "bland_altman.png",
)

MODEL_KEYS = ("median", "random_forest", "hist_gradient_boosting")
FOLDS = ((1, (2, 3)), (2, (1, 3)), (3, (1, 2)))


def _fail(problems: list[str], message: str) -> None:
    problems.append(message)


def main() -> int:
    problems: list[str] = []
    config = load_config()
    phase3 = config["phase3"]
    features = load_feature_table(phase3["features_csv"])
    if len(features) != int(phase3["expected_windows"]):
        _fail(problems, f"Feature CSV has {len(features)} rows, expected {phase3['expected_windows']}.")
    if "BIS_target" not in features.columns:
        _fail(problems, "BIS_target is missing.")
    missing = [column for column in MODEL_FEATURE_ORDER if column not in features.columns]
    if missing:
        _fail(problems, f"Feature CSV is missing model columns: {missing}")
    if features["BIS_target"].lt(0).any() or features["BIS_target"].gt(100).any():
        _fail(problems, "A BIS target is outside 0–100.")

    reports = ROOT / phase3["reports_dir"]
    models = ROOT / phase3["models_dir"]
    figures = ROOT / phase3["figures_dir"]
    predictions = ROOT / phase3["predictions_dir"]
    quality_path = reports / "data_quality.json"
    metrics_path = reports / "metrics.json"
    if not quality_path.exists() or not metrics_path.exists():
        _fail(problems, "Phase 3 reports are missing. Run python -m src.train_ml first.")
        _report(problems, None)
        return 1

    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if quality.get("critical_failures"):
        _fail(problems, f"Data-quality report has critical failures: {quality['critical_failures']}")
    if int(quality.get("rows_excluded", -1)) != 0:
        _fail(problems, "The primary run excluded rows.")
    used = metrics["dataset"]["model_features"]
    blocked = [name for name in used if name.lower() in FORBIDDEN_INPUTS or name in {"BIS_target", "sqi_median", "case_id"}]
    if blocked:
        _fail(problems, f"Model features include forbidden columns: {blocked}")
    if used != list(MODEL_FEATURE_ORDER):
        _fail(problems, "Saved feature order does not match the Phase 3 feature contract.")

    fold_rows = {(int(row["test_case"]), tuple(row["training_cases"])) for row in metrics["folds"]}
    if fold_rows != set(FOLDS):
        _fail(problems, f"Fold plan is {sorted(fold_rows)}, expected {list(FOLDS)}.")
    for row in metrics["folds"]:
        if int(row["test_case"]) in set(row["training_cases"]):
            _fail(problems, f"Test case {row['test_case']} appears in its training cases.")

    pooled_path = predictions / "oof_predictions.csv"
    if not pooled_path.exists():
        _fail(problems, "Pooled out-of-fold predictions are missing.")
        _report(problems, metrics)
        return 1
    pooled = pd.read_csv(pooled_path)
    if len(pooled) != len(features):
        _fail(problems, "Pooled predictions do not cover every accepted window.")
    if set(pooled["case_id"].astype(int)) != {1, 2, 3}:
        _fail(problems, "Pooled predictions do not contain cases 1, 2, and 3.")

    for key in MODEL_KEYS:
        column = f"pred_{key}"
        values = pooled[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            _fail(problems, f"{column} contains a non-finite prediction.")
        low = float(phase3.get("bis_min", 0.0))
        high = float(phase3.get("bis_max", 100.0))
        if bool(phase3.get("clip_predictions", True)) and (np.any(values < low) or np.any(values > high)):
            _fail(problems, f"{column} is outside the configured BIS range.")
        recomputed = regression_metrics(pooled["BIS_target"], values)
        saved = metrics["pooled_metrics"][key]
        for metric_name in ("mae", "rmse"):
            if abs(float(recomputed[metric_name]) - float(saved[metric_name])) > 1e-6:
                _fail(problems, f"Saved {key} {metric_name} does not match the prediction file.")
        if saved["pearson_r"] is None and recomputed["pearson_r"] is not None:
            _fail(problems, f"{key} Pearson correlation was stored as undefined but can be computed.")

    for test_case, train_cases in FOLDS:
        fold_dir = models / "folds" / f"test_case_{test_case:05d}"
        fold_predictions = pd.read_csv(predictions / f"test_case_{test_case:05d}.csv")
        if set(fold_predictions["case_id"].astype(int)) != {test_case}:
            _fail(problems, f"Prediction file for case {test_case} contains another case.")
        for key in MODEL_KEYS:
            path = fold_dir / f"{key}.joblib"
            if not path.exists():
                _fail(problems, f"Missing model artifact {path}.")
                continue
            model = load_bis_model(path)
            if int(model.metadata["test_case"]) != test_case:
                _fail(problems, f"{path} is labeled with the wrong test case.")
            if tuple(model.metadata["training_cases"]) != train_cases:
                _fail(problems, f"{path} training cases are {model.metadata['training_cases']}.")
            if set(model.metadata["training_cases"]) & {test_case}:
                _fail(problems, f"{path} was trained on its test case.")
            sample = fold_predictions.sample(n=min(25, len(fold_predictions)), random_state=42)
            paired = sample[["case_id", "window_index"]].merge(
                features,
                on=["case_id", "window_index"],
                how="left",
                validate="one_to_one",
            )
            try:
                predicted = predict_bis(model, paired)
            except Exception as exc:
                _fail(problems, f"{path} could not predict a saved feature row: {exc}")
                continue
            if not np.all(np.isfinite(predicted)):
                _fail(problems, f"{path} produced a non-finite prediction.")
            if bool(phase3.get("clip_predictions", True)) and (np.any(predicted < 0) or np.any(predicted > 100)):
                _fail(problems, f"{path} predictions are outside 0–100.")
            if not np.allclose(predicted, sample[f"pred_{key}"].to_numpy(dtype=float), atol=1e-6):
                _fail(problems, f"{path} does not reproduce the saved predictions.")

    all_cases = models / "all_cases" / "metadata.json"
    if all_cases.exists():
        payload = json.loads(all_cases.read_text(encoding="utf-8"))
        if payload.get("unseen_case_evaluation") is not False:
            _fail(problems, "The all-case model is not marked as unsuitable for unseen-case scores.")
    else:
        _fail(problems, "The all-case model metadata is missing.")

    for name in REQUIRED_FIGURES:
        path = figures / name
        if not path.exists() or path.stat().st_size < 1000:
            _fail(problems, f"Figure is missing or empty: {path}")
    for name in ("data_quality.json", "data_quality.md", "metrics.json", "feature_importance.csv", "training_config.json"):
        path = reports / name
        if not path.exists() or path.stat().st_size == 0:
            _fail(problems, f"Report is missing or empty: {path}")

    _report(problems, metrics)
    return 1 if problems else 0


def _report(problems: list[str], metrics: dict | None) -> None:
    print("Phase 3 verification")
    if metrics is not None:
        pooled = metrics["pooled_metrics"]
        print(f"  Windows: {metrics['dataset']['n_rows']} across {metrics['dataset']['n_cases']} cases")
        print(f"  Features: {metrics['dataset']['n_model_features']}")
        for key in MODEL_KEYS:
            row = pooled[key]
            pearson = "undefined" if row["pearson_r"] is None else f"{row['pearson_r']:.3f}"
            print(f"  {key}: MAE {row['mae']:.3f}  RMSE {row['rmse']:.3f}  R2 {row['r2']:.3f}  r {pearson}")
    if problems:
        print("FAILED")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("PASSED")
        print("  Case-level separation, metrics, models, predictions, figures, and reports were checked.")
        print("  Three cases do not establish clinical generalization.")


if __name__ == "__main__":
    sys.exit(main())
