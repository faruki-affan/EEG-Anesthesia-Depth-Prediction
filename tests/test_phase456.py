"""Regression tests for the case split, checkpoints, inference and dashboard."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.cnn import load_checkpoint, predict_windows
from src.infer_cnn import raw_windows
from src.ml_data import resolve_path
from src.train_cnn import case_folds, load_training_cases
from src.vitaldb_loader import VitalDBLoader, load_config


class Phase456Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        torch.set_num_threads(cls.config["phase4"]["cpu_threads"])

    def test_case_roles_are_disjoint_and_balanced(self):
        ids = [1, 2, 3]
        folds = list(case_folds(ids, 42))
        self.assertEqual(folds, list(case_folds(ids[::-1], 42)))
        self.assertEqual(sorted(train[0] for train, _, _ in folds), ids)
        self.assertEqual(sorted(validation for _, validation, _ in folds), ids)
        for train, validation, test in folds:
            self.assertEqual(len(set([*train, validation, test])), 3)
        with self.assertRaises(ValueError):
            list(case_folds([1, 2], 42))

    def test_bundled_features_match_waveform_labels_and_timestamps(self):
        cases = load_training_cases(self.config)
        self.assertEqual(sorted(cases), [1, 2, 3])
        self.assertEqual(sum(len(frame) for frame, _, _ in cases.values()), 5712)

    def test_metadata_download_uses_returned_csv_filename(self):
        def plain_csv(session, url, destination, timeout, refresh):
            path = destination.with_suffix("")
            contents = "caseid,tname,tid\n1,BIS/BIS,example\n" if url.endswith("trks") else "caseid\n1\n"
            path.write_text(contents, encoding="utf-8")
            return path

        with tempfile.TemporaryDirectory() as directory:
            loader = VitalDBLoader(self.config)
            loader.metadata_dir = Path(directory)
            with patch("src.vitaldb_loader.download_file", side_effect=plain_csv):
                tracks, cases = loader.ensure_index()
            self.assertEqual(tracks.tname.tolist(), ["BIS/BIS"])
            self.assertEqual(cases.caseid.tolist(), [1])

    def test_missing_bis_does_not_block_raw_waveform_inference(self):
        # The sinusoid checks the input path only. It is excluded from all experiments.
        fs = self.config["windows"]["expected_sampling_rate_hz"]
        time = np.arange(int(fs * 20)) / fs
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eeg_primary.npz"
            np.savez(path, values=20 * np.sin(2 * np.pi * 10 * time), t_start_sec=5.0,
                     sample_interval_sec=1 / fs, t_end_header_sec=25.0)
            frame, eeg, actual_fs = raw_windows(path, path.with_name("bis.npz"), self.config)
        self.assertEqual(actual_fs, fs)
        self.assertEqual(len(eeg), 3)
        self.assertTrue(frame["Recorded BIS"].isna().all())
        np.testing.assert_allclose(frame["Window end (s)"], [15, 20, 25])

    def test_saved_models_reproduce_held_out_predictions(self):
        import pandas as pd

        phase4 = self.config["phase4"]
        predictions = resolve_path(phase4["reports_dir"]) / "oof_predictions.csv"
        if not predictions.exists():
            self.skipTest("Phase 4 artifacts have not been generated.")
        expected = pd.read_csv(predictions)
        for case_id, (features, eeg, fs) in load_training_cases(self.config).items():
            path = resolve_path(phase4["models_dir"]) / f"test_case_{case_id:05d}.pt"
            checkpoint = load_checkpoint(path)
            _, metadata = checkpoint
            self.assertEqual(case_id, metadata["test_case"])
            self.assertNotIn(case_id, metadata["train_cases"])
            self.assertNotEqual(case_id, metadata["validation_case"])
            values = predict_windows(checkpoint, eeg, fs)
            rows = expected[expected.case_id.eq(case_id)].set_index("window_index")
            np.testing.assert_allclose(values, rows.loc[features.window_index, "pred_cnn"], atol=1e-4)
            with self.assertRaises(ValueError):
                predict_windows(checkpoint, eeg[:, :-1], fs)

    def test_dashboard_pages(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        app.run(timeout=30)
        self.assertFalse(app.exception)
        for page in app.sidebar.radio[0].options[1:]:
            app.sidebar.radio[0].set_value(page).run(timeout=30)
            self.assertFalse(app.exception, page)
            self.assertFalse(app.error, page)
        app.sidebar.radio[0].set_value("BIS Prediction").run(timeout=30)
        app.button[0].click().run(timeout=30)
        self.assertFalse(app.exception)
        checkpoints = resolve_path(self.config["phase4"]["models_dir"])
        if (checkpoints / "test_case_00001.pt").exists():
            self.assertFalse(app.error)
            self.assertTrue(app.metric)
            self.assertEqual(app.metric[0].label, "Predicted BIS")
            self.assertTrue(0 <= float(app.metric[0].value) <= 100)
            app.sidebar.radio[0].set_value("Actual vs Predicted BIS").run(timeout=30)
            self.assertFalse(app.exception)
            app.toggle[0].set_value(True).run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(app.get("plotly_chart"))


if __name__ == "__main__":
    unittest.main()

