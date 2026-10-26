"""Tests for BIS regression, feature selection, and case-level splits."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml_data import (  # noqa: E402
    FORBIDDEN_INPUTS,
    MODEL_FEATURE_ORDER,
    Phase3DataError,
    _classify_zero_bis,
    assert_case_separation,
    leave_one_case_out,
    select_model_features,
)
from src.ml_metrics import regression_metrics  # noqa: E402
from src.predict import (  # noqa: E402
    BISModel,
    FeatureValidationError,
    MedianBaseline,
    NonnegativeLog1p,
    load_bis_model,
    save_bis_model,
)
from src.train_ml import train_one_fold  # noqa: E402
from src.vitaldb_loader import PROJECT_ROOT  # noqa: E402


def _phase3() -> dict:
    return {
        "random_state": 42,
        "log1p_skew_threshold": 1.0,
        "clip_predictions": True,
        "bis_min": 0.0,
        "bis_max": 100.0,
        "permutation_repeats": 2,
        "permutation_max_windows": 30,
        "random_forest": {
            "n_estimators": 8,
            "max_depth": 4,
            "min_samples_leaf": 2,
            "n_jobs": 1,
        },
        "hist_gradient_boosting": {
            "max_iter": 12,
            "learning_rate": 0.1,
            "max_leaf_nodes": 8,
            "early_stopping": False,
        },
    }


def _synthetic(seed: int = 0, n_per_case: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for case_id in (1, 2, 3):
        power = rng.lognormal(mean=1.0, sigma=1.2, size=n_per_case)
        for index in range(n_per_case):
            row = {name: float(rng.normal()) for name in MODEL_FEATURE_ORDER}
            row["delta_power"] = float(power[index])
            row["case_id"] = case_id
            row["window_index"] = index
            row["window_start_sec"] = float(index * 5)
            row["window_end_sec"] = float(index * 5 + 10)
            row["sqi_median"] = 90.0
            row["BIS_target"] = float(np.clip(35.0 + 15.0 * np.log1p(power[index]) + rng.normal(0, 1), 0, 100))
            rows.append(row)
    return pd.DataFrame(rows)


class FeatureSelectionTests(unittest.TestCase):
    def test_selects_only_existing_eeg_features(self):
        columns = list(MODEL_FEATURE_ORDER) + [
            "BIS_target",
            "sqi_median",
            "case_id",
            "window_start_sec",
            "BIS/EMG",
            "suppression_ratio",
        ]
        selected = select_model_features(columns)
        self.assertEqual(selected, list(MODEL_FEATURE_ORDER))
        self.assertNotIn("BIS_target", selected)
        self.assertNotIn("sqi_median", selected)
        self.assertTrue(set(selected).isdisjoint(FORBIDDEN_INPUTS))
        self.assertNotIn("skewness", selected)
        self.assertNotIn("kurtosis", selected)
        self.assertNotIn("hjorth_activity", selected)
        self.assertNotIn("total_power", selected)

    def test_missing_feature_is_rejected(self):
        with self.assertRaises(Phase3DataError):
            select_model_features(["delta_power", "BIS_target"])

    def test_processed_csv_matches_the_feature_contract(self):
        path = PROJECT_ROOT / "data" / "processed" / "eeg_features.csv"
        if not path.exists():
            self.skipTest("Processed feature CSV is not in this checkout.")
        frame = pd.read_csv(path)
        self.assertEqual(len(frame), 5712)
        selected = select_model_features(frame.columns)
        self.assertTrue(set(selected).isdisjoint({"BIS_target", "sqi_median", "case_id"}))
        self.assertEqual(set(frame["case_id"].astype(int)), {1, 2, 3})


class SplitTests(unittest.TestCase):
    def test_leave_one_case_out_matches_the_three_case_plan(self):
        folds = leave_one_case_out([3, 1, 2])
        self.assertEqual(folds, [(1, (2, 3)), (2, (1, 3)), (3, (1, 2))])
        for test_case, train_cases in folds:
            self.assertNotIn(test_case, train_cases)
            assert_case_separation(train_cases, [test_case])
            self.assertTrue(set(train_cases).isdisjoint({test_case}))

    def test_overlapping_cases_are_rejected(self):
        with self.assertRaises(Phase3DataError):
            assert_case_separation([1, 2], [2])


class MetricTests(unittest.TestCase):
    def test_mae_and_rmse(self):
        metrics = regression_metrics([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
        self.assertAlmostEqual(metrics["mae"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["rmse"], (2.0 / 3.0) ** 0.5)
        self.assertAlmostEqual(metrics["r2"], 0.0)
        self.assertIsNone(metrics["pearson_r"])

    def test_constant_target_has_undefined_r2(self):
        metrics = regression_metrics([4.0, 4.0, 4.0], [1.0, 2.0, 3.0])
        self.assertIsNone(metrics["r2"])
        self.assertIsNone(metrics["pearson_r"])

    def test_numerical_noise_in_a_constant_prediction_is_undefined(self):
        predicted = np.full(5, 46.3) + np.array([0.0, 1e-15, -1e-15, 0.0, 1e-15])
        metrics = regression_metrics([40.0, 50.0, 45.0, 47.0, 49.0], predicted)
        self.assertIsNone(metrics["pearson_r"])


class ModelContractTests(unittest.TestCase):
    def test_prediction_shape_and_reproducible_seed(self):
        frame = _synthetic()
        features = list(MODEL_FEATURE_ORDER)
        train = frame.loc[frame["case_id"].isin([2, 3])]
        test = frame.loc[frame["case_id"] == 1]
        first = train_one_fold(train, test, features, _phase3(), compute_importance=True)
        second = train_one_fold(train, test, features, _phase3(), compute_importance=False)
        self.assertEqual(set(first["train_cases"]), {2, 3})
        self.assertEqual(first["test_case"], 1)
        self.assertTrue(set(first["train_cases"]).isdisjoint({first["test_case"]}))
        for key in ("median", "random_forest", "hist_gradient_boosting"):
            self.assertEqual(first["predictions"][key].shape, (len(test),))
            self.assertTrue(np.all(np.isfinite(first["predictions"][key])))
            np.testing.assert_allclose(first["predictions"][key], second["predictions"][key])
        self.assertIn("delta_power", first["log1p_columns"]["random_forest"])
        self.assertTrue(first["importance"]["rows"])

    def test_overlapping_fold_is_rejected(self):
        frame = _synthetic(n_per_case=12)
        features = list(MODEL_FEATURE_ORDER)
        train = frame.loc[frame["case_id"].isin([1, 2])]
        test = frame.loc[frame["case_id"] == 2]
        with self.assertRaises(Phase3DataError):
            train_one_fold(train, test, features, _phase3())

    def test_clipping_bounds(self):
        class _Stub:
            def predict(self, features):
                return np.asarray([150.0, -5.0, 40.0])

        model = BISModel(
            _Stub(),
            ["mean"],
            {"clip_predictions": True, "bis_min": 0.0, "bis_max": 100.0},
        )
        clipped = model.predict(pd.DataFrame({"mean": [1.0, 2.0, 3.0]}))
        raw = model.predict(pd.DataFrame({"mean": [1.0, 2.0, 3.0]}), clip=False)
        np.testing.assert_allclose(clipped, [100.0, 0.0, 40.0])
        np.testing.assert_allclose(raw, [150.0, -5.0, 40.0])

    def test_save_and_load_round_trip(self):
        frame = _synthetic(n_per_case=20)
        features = list(MODEL_FEATURE_ORDER)
        train = frame.loc[frame["case_id"].isin([1, 3])]
        test = frame.loc[frame["case_id"] == 2]
        fold = train_one_fold(train, test, features, _phase3())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "random_forest.joblib"
            save_bis_model(fold["models"]["random_forest"], path)
            loaded = load_bis_model(path)
        predicted = loaded.predict(test)
        np.testing.assert_allclose(predicted, fold["predictions"]["random_forest"])
        self.assertEqual(loaded.metadata["training_cases"], (1, 3))
        self.assertEqual(loaded.feature_names, features)

    def test_missing_and_non_finite_features_are_reported(self):
        model = BISModel(MedianBaseline().fit(np.ones((4, 1)), np.arange(4)), ["delta_power"], {"clip_predictions": False})
        with self.assertRaises(FeatureValidationError) as missing:
            model.predict(pd.DataFrame({"mean": [1.0]}))
        self.assertIn("delta_power", str(missing.exception))
        with self.assertRaises(FeatureValidationError) as invalid:
            model.predict(pd.DataFrame({"delta_power": [np.nan]}))
        self.assertIn("delta_power", str(invalid.exception))
        with self.assertRaises(FeatureValidationError):
            model.predict(np.ones((2, 3)))

    def test_named_columns_are_reordered_to_the_training_order(self):
        estimator = MedianBaseline().fit(np.ones((3, 2)), np.array([10.0, 20.0, 30.0]))
        model = BISModel(estimator, ["alpha_power", "delta_power"], {"clip_predictions": False})
        swapped = pd.DataFrame({"delta_power": [1.0], "alpha_power": [2.0]})
        ordered = pd.DataFrame({"alpha_power": [2.0], "delta_power": [1.0]})
        np.testing.assert_allclose(model.predict(swapped), model.predict(ordered))

    def test_log1p_mask_is_fit_on_the_supplied_matrix(self):
        skewed = np.exp(np.random.default_rng(1).normal(size=(80, 1)))
        symmetric = np.random.default_rng(1).normal(loc=5.0, scale=0.2, size=(80, 1))
        logged = NonnegativeLog1p(1.0).fit(skewed)
        unchanged = NonnegativeLog1p(1.0).fit(symmetric)
        self.assertTrue(bool(logged.log_mask_[0]))
        self.assertFalse(bool(unchanged.log_mask_[0]))
        np.testing.assert_allclose(logged.transform(skewed[:3]), np.log1p(skewed[:3]))


class ZeroBisHeuristicTests(unittest.TestCase):
    def test_startup_and_dropout_are_not_called_suppression(self):
        case = pd.DataFrame(
            {
                "window_start_sec": [50.0, 90.0, 400.0],
                "BIS_target": [0.0, 80.0, 0.0],
                "sqi_median": [2.0, 90.0, 0.0],
                "std": [20.0, 15.0, 1.0],
            }
        )
        startup = _classify_zero_bis(case.iloc[0], case.sort_values("window_start_sec"), None)
        dropout = _classify_zero_bis(case.iloc[2], case.sort_values("window_start_sec"), 0.0)
        self.assertEqual(startup, "monitor_startup_unreliable")
        self.assertEqual(dropout, "end_of_recording_or_dropout_unreliable")


if __name__ == "__main__":
    unittest.main()
