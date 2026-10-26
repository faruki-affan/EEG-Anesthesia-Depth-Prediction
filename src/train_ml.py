"""Train BIS regression models with leave-one-case-out evaluation.

Windows from the same surgery overlap, so they are not split at random.
Each fold trains on two cases and tests on the remaining case. Feature
transforms are fit on the training cases only. Fixed hyperparameters are
used; the held-out case is not used for tuning.

The model trained on all three cases is saved for later use and is not
scored on those same cases.
"""

from __future__ import annotations

import logging
import platform
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.ml_data import (
    EXPECTED_CASES,
    MODEL_KEYS,
    MODEL_LABELS,
    Phase3DataError,
    assert_case_separation,
    build_quality_report,
    feature_group,
    leave_one_case_out,
    load_feature_table,
    render_quality_markdown,
    resolve_path,
    select_model_features,
    write_json,
)
from src.ml_metrics import metrics_by_bis_range, regression_metrics
from src.ml_plots import (
    plot_absolute_error,
    plot_bis_distribution,
    plot_bis_over_time,
    plot_bland_altman,
    plot_error_histogram,
    plot_feature_importance,
    plot_model_comparison,
    plot_performance_by_range,
    plot_scatter,
)
from src.predict import BISModel, logged_feature_names, make_estimator, save_bis_model
from src.vitaldb_loader import load_config

logger = logging.getLogger(__name__)

PREDICTION_COLUMNS = (
    "case_id",
    "window_index",
    "window_start_sec",
    "window_end_sec",
    "BIS_target",
    "sqi_median",
)


def _package_versions() -> dict[str, str]:
    import joblib
    import numpy
    import pandas
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def _set_predict_jobs(estimator: Any, n_jobs: int) -> int | None:
    model = estimator
    if hasattr(estimator, "named_steps") and "model" in estimator.named_steps:
        model = estimator.named_steps["model"]
    if not hasattr(model, "n_jobs"):
        return None
    previous = model.n_jobs
    model.n_jobs = n_jobs
    return previous


def _attach_metrics(actual, predicted, case_ids) -> dict[str, Any]:
    metrics = regression_metrics(actual, predicted)
    metrics["n_cases"] = int(pd.Series(case_ids).nunique())
    return metrics


def _count_clipped(raw: np.ndarray, low: float, high: float) -> int:
    return int(np.sum((raw < low) | (raw > high)))


def train_one_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_names: list[str],
    phase3: dict[str, Any],
    compute_importance: bool = False,
) -> dict[str, Any]:
    """Fit all three models on ``train`` and predict ``test``.

    ``train`` and ``test`` must come from disjoint cases.
    """
    train_cases = tuple(sorted(int(case_id) for case_id in train["case_id"].unique()))
    test_cases = tuple(sorted(int(case_id) for case_id in test["case_id"].unique()))
    assert_case_separation(train_cases, test_cases)
    if len(test_cases) != 1:
        raise Phase3DataError(f"Each fold must hold out one case, got {test_cases}.")

    x_train = train.loc[:, feature_names].to_numpy(dtype=np.float64)
    y_train = train["BIS_target"].to_numpy(dtype=np.float64)
    x_test = test.loc[:, feature_names].to_numpy(dtype=np.float64)
    y_test = test["BIS_target"].to_numpy(dtype=np.float64)
    low = float(phase3.get("bis_min", 0.0))
    high = float(phase3.get("bis_max", 100.0))
    clip = bool(phase3.get("clip_predictions", True))
    seed = int(phase3["random_state"])

    models: dict[str, BISModel] = {}
    predictions: dict[str, np.ndarray] = {}
    raw_predictions: dict[str, np.ndarray] = {}
    fit_seconds: dict[str, float] = {}
    logged: dict[str, list[str]] = {}
    parameters: dict[str, dict] = {}

    for key in MODEL_KEYS:
        estimator = make_estimator(key, phase3)
        started = time.perf_counter()
        estimator.fit(x_train, y_train)
        fit_seconds[key] = float(time.perf_counter() - started)
        raw = np.asarray(estimator.predict(x_test), dtype=np.float64).reshape(-1)
        if raw.shape != (len(test),):
            raise Phase3DataError(f"{key} returned shape {raw.shape}, expected {(len(test),)}.")
        raw_predictions[key] = raw
        predictions[key] = np.clip(raw, low, high) if clip else raw
        names = logged_feature_names(estimator, feature_names)
        logged[key] = names
        fitted_params = {}
        inner = estimator.named_steps["model"] if hasattr(estimator, "named_steps") and "model" in getattr(estimator, "named_steps", {}) else estimator
        if hasattr(inner, "get_params"):
            fitted_params = {name: value for name, value in inner.get_params().items() if not isinstance(value, (dict, list))}
        metadata = {
            "model_key": key,
            "model_label": MODEL_LABELS[key],
            "training_cases": train_cases,
            "test_case": test_cases[0],
            "feature_names": list(feature_names),
            "log1p_columns": names,
            "log1p_skew_threshold": float(phase3["log1p_skew_threshold"]),
            "clip_predictions": clip,
            "bis_min": low,
            "bis_max": high,
            "random_state": seed,
            "parameters": fitted_params,
            "fit_seconds": fit_seconds[key],
            "evaluation_role": "leave_one_case_out_fold",
            "unseen_case_evaluation": True,
        }
        models[key] = BISModel(estimator, list(feature_names), metadata)
        parameters[key] = fitted_params

    importance = None
    if compute_importance:
        importance = _permutation_and_impurity(
            models["random_forest"],
            x_test,
            y_test,
            feature_names,
            seed + int(test_cases[0]),
            int(phase3.get("permutation_repeats", 5)),
            int(phase3.get("permutation_max_windows", 500)),
        )

    return {
        "train_cases": train_cases,
        "test_case": test_cases[0],
        "models": models,
        "predictions": predictions,
        "raw_predictions": raw_predictions,
        "n_clipped": {key: _count_clipped(raw_predictions[key], low, high) if clip else 0 for key in MODEL_KEYS},
        "fit_seconds": fit_seconds,
        "log1p_columns": logged,
        "parameters": parameters,
        "importance": importance,
        "metrics": {
            key: _attach_metrics(y_test, predictions[key], test["case_id"])
            for key in MODEL_KEYS
        },
    }


def _permutation_and_impurity(
    model: BISModel,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    seed: int,
    repeats: int,
    max_windows: int,
) -> dict[str, Any]:
    estimator = model.estimator
    previous_jobs = _set_predict_jobs(estimator, 1)
    rng = np.random.default_rng(seed)
    if x_test.shape[0] > max_windows:
        picked = np.sort(rng.choice(x_test.shape[0], size=max_windows, replace=False))
    else:
        picked = np.arange(x_test.shape[0])
    forest = estimator.named_steps["model"]
    impurity = np.asarray(forest.feature_importances_, dtype=np.float64)
    result = permutation_importance(
        estimator,
        x_test[picked],
        y_test[picked],
        scoring="neg_mean_absolute_error",
        n_repeats=repeats,
        random_state=seed,
        n_jobs=1,
    )
    if previous_jobs is not None:
        _set_predict_jobs(estimator, previous_jobs)
    return {
        "n_windows": int(picked.size),
        "n_repeats": int(repeats),
        "scoring": "neg_mean_absolute_error",
        "note": (
            "Impurity importance is computed on the training fold after log1p. "
            "Permutation importance permutes raw held-out features and measures "
            "the increase in MAE. Neither quantity is a causal effect."
        ),
        "rows": [
            {
                "feature": feature_names[index],
                "group": feature_group(feature_names[index]),
                "impurity": float(impurity[index]),
                "permutation_mean": float(result.importances_mean[index]),
                "permutation_std": float(result.importances_std[index]),
            }
            for index in range(len(feature_names))
        ],
    }


def _prediction_frame(test: pd.DataFrame, fold: dict[str, Any]) -> pd.DataFrame:
    frame = test.loc[:, list(PREDICTION_COLUMNS)].copy()
    frame["test_case"] = int(fold["test_case"])
    frame["training_cases"] = ",".join(str(case_id) for case_id in fold["train_cases"])
    for key in MODEL_KEYS:
        frame[f"pred_{key}"] = fold["predictions"][key]
        frame[f"pred_{key}_raw"] = fold["raw_predictions"][key]
    return frame


def _group_importance(summary: pd.DataFrame) -> dict[str, dict[str, float]]:
    grouped = {}
    for group, rows in summary.groupby("group"):
        grouped[str(group)] = {
            "impurity_mean_sum": float(rows["impurity_mean"].sum()),
            "permutation_mean_sum": float(rows["permutation_mean"].sum()),
            "n_features": int(len(rows)),
        }
    return grouped


def _average_importance(fold_tables: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(fold_tables, ignore_index=True)
    averaged = (
        combined.groupby(["feature", "group"], as_index=False)
        .agg(
            impurity_mean=("impurity", "mean"),
            permutation_mean=("permutation_mean", "mean"),
            permutation_std=("permutation_std", "mean"),
        )
        .sort_values("permutation_mean", ascending=False)
    )
    return averaged


def _range_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in MODEL_KEYS:
        for item in metrics_by_bis_range(predictions["BIS_target"], predictions[f"pred_{key}"], predictions["case_id"]):
            item["model"] = key
            rows.append(item)
    return pd.DataFrame(rows)


def _sensitivity(frame: pd.DataFrame, feature_names: list[str], phase3: dict[str, Any]) -> dict[str, Any]:
    threshold = float(phase3["sqi_sensitivity_min"])
    kept = frame.loc[frame["sqi_median"] >= threshold].copy()
    excluded = int(len(frame) - len(kept))
    folds = []
    pooled_parts = []
    for test_case, train_cases in leave_one_case_out(kept["case_id"]):
        train = kept.loc[kept["case_id"].isin(train_cases)]
        test = kept.loc[kept["case_id"] == test_case]
        if train.empty or test.empty:
            raise Phase3DataError(f"SQI filter left fold test case {test_case} empty.")
        logger.info("SQI sensitivity fold, hold out case %s (%s windows).", test_case, len(test))
        fold = train_one_fold(train, test, feature_names, phase3, compute_importance=False)
        folds.append(
            {
                "test_case": fold["test_case"],
                "training_cases": fold["train_cases"],
                "n_train_windows": int(len(train)),
                "n_test_windows": int(len(test)),
                "metrics": fold["metrics"],
            }
        )
        pooled_parts.append(_prediction_frame(test, fold))
    pooled = pd.concat(pooled_parts, ignore_index=True)
    return {
        "sqi_min": threshold,
        "n_windows_excluded": excluded,
        "n_windows_used": int(len(kept)),
        "n_cases": int(kept["case_id"].nunique()),
        "note": (
            "SQI filters the rows only. It is not a model input. "
            "These scores are a sensitivity analysis, not the primary result."
        ),
        "folds": folds,
        "pooled": {
            key: _attach_metrics(pooled["BIS_target"], pooled[f"pred_{key}"], pooled["case_id"])
            for key in MODEL_KEYS
        },
    }


def _save_fold_models(fold: dict[str, Any], directory) -> None:
    from pathlib import Path

    fold_dir = Path(directory) / f"test_case_{int(fold['test_case']):05d}"
    for key, model in fold["models"].items():
        save_bis_model(model, fold_dir / f"{key}.joblib")
    write_json(
        fold_dir / "metadata.json",
        {
            "test_case": fold["test_case"],
            "training_cases": fold["train_cases"],
            "fit_seconds": fold["fit_seconds"],
            "log1p_columns": fold["log1p_columns"],
            "n_clipped": fold["n_clipped"],
            "metrics": fold["metrics"],
            "parameters": fold["parameters"],
            "evaluation_role": "leave_one_case_out_fold",
        },
    )


def _train_all_cases(frame, feature_names, phase3, directory) -> dict[str, Any]:
    """Fit on every available case after cross-validation. Do not score these cases as unseen."""
    from pathlib import Path

    destination = Path(directory) / "all_cases"
    x_all = frame.loc[:, feature_names].to_numpy(dtype=np.float64)
    y_all = frame["BIS_target"].to_numpy(dtype=np.float64)
    cases = tuple(sorted(int(case_id) for case_id in frame["case_id"].unique()))
    saved = []
    for key in MODEL_KEYS:
        estimator = make_estimator(key, phase3)
        started = time.perf_counter()
        estimator.fit(x_all, y_all)
        elapsed = float(time.perf_counter() - started)
        metadata = {
            "model_key": key,
            "model_label": MODEL_LABELS[key],
            "training_cases": cases,
            "test_case": None,
            "feature_names": list(feature_names),
            "log1p_columns": logged_feature_names(estimator, feature_names),
            "log1p_skew_threshold": float(phase3["log1p_skew_threshold"]),
            "clip_predictions": bool(phase3.get("clip_predictions", True)),
            "bis_min": float(phase3.get("bis_min", 0.0)),
            "bis_max": float(phase3.get("bis_max", 100.0)),
            "random_state": int(phase3["random_state"]),
            "fit_seconds": elapsed,
            "evaluation_role": "trained_on_all_available_cases",
            "unseen_case_evaluation": False,
            "note": (
                "This model saw cases "
                + ", ".join(str(case_id) for case_id in cases)
                + ". Do not report its predictions on those cases as unseen-case performance."
            ),
        }
        model = BISModel(estimator, list(feature_names), metadata)
        path = destination / f"{key}.joblib"
        save_bis_model(model, path)
        saved.append(str(path))
    write_json(destination / "metadata.json", {"models": saved, "training_cases": cases, "unseen_case_evaluation": False})
    return {"directory": str(destination), "training_cases": cases, "unseen_case_evaluation": False}


def _format_metric(value: Any) -> str:
    if value is None:
        return "undefined"
    return f"{float(value):.3f}"


def _print_summary(summary: dict[str, Any]) -> None:
    quality = summary["dataset"]
    print("Dataset summary")
    print(f"  Cases: {quality['n_cases']} {quality['case_counts']}")
    print(f"  Accepted windows: {quality['n_rows']}")
    print(f"  Features: {quality['n_model_features']}")
    bis = quality["bis_overall"]
    print(
        f"  BIS: median {bis['median']:.2f}, mean {bis['mean']:.2f}, "
        f"min {bis['min']:.2f}, max {bis['max']:.2f}"
    )
    print(f"  Rows excluded: {quality['rows_excluded']}")
    print(f"  BIS = 0 windows retained: {len(quality['bis_zero_windows'])} {quality['bis_zero_classification_counts']}")
    print()
    print("Pooled out-of-fold results (window-level; not independent patients)")
    header = f"{'Model':<24} {'MAE':>10} {'RMSE':>10} {'R2':>10} {'Pearson r':>12}"
    print(header)
    for key in MODEL_KEYS:
        metrics = summary["pooled_metrics"][key]
        print(
            f"{MODEL_LABELS[key]:<24} "
            f"{_format_metric(metrics['mae']):>10} "
            f"{_format_metric(metrics['rmse']):>10} "
            f"{_format_metric(metrics['r2']):>10} "
            f"{_format_metric(metrics['pearson_r']):>12}"
        )
    print(f"  Pooled windows: {summary['pooled_metrics']['random_forest']['n_windows']}")
    print(f"  Independent cases: {summary['pooled_metrics']['random_forest']['n_cases']}")
    print()
    print("Held-out case results")
    for fold in summary["folds"]:
        print(f"  Test case {fold['test_case']} (trained on {list(fold['training_cases'])}, n={fold['n_test_windows']})")
        for key in MODEL_KEYS:
            metrics = fold["metrics"][key]
            print(
                f"    {MODEL_LABELS[key]:<22} "
                f"MAE {_format_metric(metrics['mae'])}  "
                f"RMSE {_format_metric(metrics['rmse'])}  "
                f"R2 {_format_metric(metrics['r2'])}  "
                f"r {_format_metric(metrics['pearson_r'])}"
            )
    print()
    print("Random Forest features (permutation importance, mean across cases)")
    for row in summary["feature_importance_top10"]:
        print(
            f"  {row['feature']:<22} {row['group']:<10} "
            f"permutation {row['permutation_mean']:.4f}  impurity {row['impurity_mean']:.4f}"
        )
    print(f"  Group sums: {summary['feature_groups']}")
    print("  Feature importance is not evidence of causation.")
    print()
    print("These three cases do not establish clinical generalization.")
    print(f"Training seconds: {summary['train_seconds']:.1f}")


def run_phase3(config: dict[str, Any] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    config = config or load_config()
    phase3 = config["phase3"]
    reports_dir = resolve_path(phase3["reports_dir"])
    figures_dir = resolve_path(phase3["figures_dir"])
    predictions_dir = resolve_path(phase3["predictions_dir"])
    models_dir = resolve_path(phase3["models_dir"])
    for directory in (reports_dir, figures_dir, predictions_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    frame = load_feature_table(phase3["features_csv"])
    expected_cases = tuple(int(case_id) for case_id in phase3.get("expected_cases", EXPECTED_CASES))
    quality = build_quality_report(frame, int(phase3["expected_windows"]), expected_cases)
    write_json(reports_dir / "data_quality.json", quality)
    (reports_dir / "data_quality.md").write_text(render_quality_markdown(quality), encoding="utf-8")
    if quality["critical_failures"]:
        raise Phase3DataError("Data-quality checks failed:\n- " + "\n- ".join(quality["critical_failures"]))

    feature_names = select_model_features(frame.columns)
    usable = frame.copy()
    exclusions: list[dict[str, Any]] = []
    model_matrix = usable.loc[:, feature_names + ["BIS_target"]]
    finite = np.isfinite(model_matrix.to_numpy(dtype=np.float64)).all(axis=1)
    if not bool(finite.all()):
        n_bad = int((~finite).sum())
        raise Phase3DataError(
            f"{n_bad} rows have non-finite model features or BIS targets. "
            "They were not dropped silently."
        )

    folds_out = []
    prediction_frames = []
    importance_frames = []
    fold_plan = leave_one_case_out(usable["case_id"])
    expected_plan = leave_one_case_out(expected_cases)
    if fold_plan != expected_plan:
        raise Phase3DataError(f"Fold plan {fold_plan} does not match the required plan {expected_plan}.")

    for test_case, train_cases in fold_plan:
        train = usable.loc[usable["case_id"].isin(train_cases)]
        test = usable.loc[usable["case_id"] == test_case]
        logger.info(
            "Fold test case %s, train cases %s (%s train windows, %s test windows).",
            test_case,
            train_cases,
            len(train),
            len(test),
        )
        fold = train_one_fold(train, test, feature_names, phase3, compute_importance=True)
        _save_fold_models(fold, models_dir / "folds")
        predicted = _prediction_frame(test, fold)
        predicted.to_csv(predictions_dir / f"test_case_{test_case:05d}.csv", index=False)
        prediction_frames.append(predicted)
        if fold["importance"] is not None:
            importance_frames.append(pd.DataFrame(fold["importance"]["rows"]).assign(test_case=test_case))
        folds_out.append(
            {
                "test_case": fold["test_case"],
                "training_cases": fold["train_cases"],
                "n_train_windows": int(len(train)),
                "n_test_windows": int(len(test)),
                "fit_seconds": fold["fit_seconds"],
                "log1p_columns": fold["log1p_columns"]["random_forest"],
                "n_clipped": fold["n_clipped"],
                "metrics": fold["metrics"],
                "permutation_windows": None if fold["importance"] is None else fold["importance"]["n_windows"],
            }
        )
        for key in MODEL_KEYS:
            metrics = fold["metrics"][key]
            logger.info(
                "  %s MAE %.3f RMSE %.3f R2 %s r %s",
                key,
                metrics["mae"],
                metrics["rmse"],
                metrics["r2"],
                metrics["pearson_r"],
            )

    pooled = pd.concat(prediction_frames, ignore_index=True)
    pooled.to_csv(predictions_dir / "oof_predictions.csv", index=False)
    pooled_metrics = {
        key: _attach_metrics(pooled["BIS_target"], pooled[f"pred_{key}"], pooled["case_id"])
        for key in MODEL_KEYS
    }
    by_case = []
    for case_id, case_frame in pooled.groupby("case_id", sort=True):
        by_case.append(
            {
                "case_id": int(case_id),
                "n_windows": int(len(case_frame)),
                "metrics": {
                    key: _attach_metrics(case_frame["BIS_target"], case_frame[f"pred_{key}"], case_frame["case_id"])
                    for key in MODEL_KEYS
                },
            }
        )

    importance = _average_importance(importance_frames)
    importance.to_csv(reports_dir / "feature_importance.csv", index=False)
    pd.concat(importance_frames, ignore_index=True).to_csv(reports_dir / "feature_importance_by_fold.csv", index=False)
    groups = _group_importance(importance)
    range_table = _range_frame(pooled)
    range_table.to_csv(reports_dir / "metrics_by_bis_range.csv", index=False)

    plot_bis_distribution(usable, figures_dir / "bis_distribution_by_case.png")
    plot_scatter(pooled, figures_dir / "pooled_actual_vs_predicted.png")
    plot_error_histogram(pooled, figures_dir / "pooled_error_histogram.png")
    plot_absolute_error(pooled, figures_dir / "pooled_absolute_error.png")
    plot_feature_importance(importance, figures_dir / "random_forest_feature_importance.png")
    plot_model_comparison(pooled_metrics, figures_dir / "model_comparison.png")
    plot_performance_by_range(range_table, figures_dir / "performance_by_bis_range.png")
    plot_bland_altman(pooled, figures_dir / "bland_altman.png")
    for case_id in expected_cases:
        plot_bis_over_time(pooled, int(case_id), figures_dir / f"case_{int(case_id):05d}_bis_over_time.png")

    sensitivity = None
    if bool(phase3.get("run_sqi_sensitivity", False)):
        sensitivity = _sensitivity(usable, feature_names, phase3)
        write_json(reports_dir / "sqi_sensitivity.json", sensitivity)

    all_cases = None
    if bool(phase3.get("train_all_cases_model", False)):
        logger.info("Fitting models on all cases for later use. These are not unseen-case scores.")
        all_cases = _train_all_cases(usable, feature_names, phase3, models_dir)

    train_seconds = float(time.perf_counter() - started)
    watched = ["delta_power", "alpha_power", "spectral_entropy", "spectral_edge_hz"]
    watched_rows = importance.loc[importance["feature"].isin(watched)].to_dict(orient="records")
    top10 = importance.head(10).to_dict(orient="records")
    config_record = {
        "random_state": int(phase3["random_state"]),
        "clip_predictions": bool(phase3.get("clip_predictions", True)),
        "log1p_skew_threshold": float(phase3["log1p_skew_threshold"]),
        "random_forest": phase3["random_forest"],
        "hist_gradient_boosting": phase3["hist_gradient_boosting"],
        "permutation_repeats": int(phase3.get("permutation_repeats", 5)),
        "permutation_max_windows": int(phase3.get("permutation_max_windows", 500)),
        "feature_names": feature_names,
        "rows_excluded": exclusions,
        "versions": _package_versions(),
        "split": "leave_one_case_out",
        "folds": [{"test_case": test_case, "training_cases": train_cases} for test_case, train_cases in fold_plan],
        "limitation": "Three surgical cases do not establish clinical generalization.",
    }
    write_json(reports_dir / "training_config.json", config_record)
    write_json(
        models_dir / "feature_config.json",
        {
            "feature_names": feature_names,
            "target": "BIS_target",
            "excluded_from_inputs": quality["forbidden_inputs_present_but_not_used"],
            "absent_features": quality["absent_features"],
        },
    )

    summary = {
        "dataset": {
            "n_cases": quality["n_cases"],
            "case_counts": quality["case_counts"],
            "n_rows": quality["n_rows"],
            "n_model_features": quality["n_model_features"],
            "model_features": feature_names,
            "bis_overall": quality["bis_overall"],
            "bis_histogram": quality["bis_histogram"],
            "rows_excluded": 0,
            "exclusion_reasons": [],
            "bis_zero_windows": quality["bis_zero_windows"],
            "bis_zero_classification_counts": quality["bis_zero_classification_counts"],
        },
        "folds": folds_out,
        "by_case": by_case,
        "pooled_metrics": pooled_metrics,
        "metrics_by_bis_range": range_table.to_dict(orient="records"),
        "feature_importance_top10": top10,
        "spectral_features_of_interest": watched_rows,
        "feature_groups": groups,
        "feature_importance_note": (
            "Impurity importance is a training-set split statistic. "
            "Permutation importance is the held-out increase in MAE after shuffling one raw feature. "
            "Group sums are unweighted means across the three held-out cases. "
            "Neither measure shows that a feature causes BIS."
        ),
        "sqi_sensitivity": sensitivity,
        "all_cases_model": all_cases,
        "train_seconds": train_seconds,
        "rows_excluded": exclusions,
        "artifacts": {
            "models_dir": str(models_dir),
            "predictions": str(predictions_dir / "oof_predictions.csv"),
            "figures_dir": str(figures_dir),
            "reports_dir": str(reports_dir),
        },
    }
    write_json(reports_dir / "metrics.json", summary)
    write_json(models_dir / "cv_results.json", summary)
    _print_summary(summary)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_phase3()


if __name__ == "__main__":
    main()
