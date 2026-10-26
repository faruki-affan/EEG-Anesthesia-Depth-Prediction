"""Tests for filtering, Welch frequency detection, windowing, and BIS alignment."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import dominant_frequency, extract_features, time_features  # noqa: E402
from src.phase2 import load_processed_windows, save_processed_windows  # noqa: E402
from src.preprocess import bandpass_filter, design_bandpass, sos_pad_edge  # noqa: E402
from src.windows import (  # noqa: E402
    EXCESSIVE_MISSING,
    FLATLINE,
    INVALID_BIS,
    SEVERE_ARTIFACT,
    bis_in_window,
    evaluate_eeg,
    window_specs,
)

FS = 128.0
BANDS = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
)


def _tone(frequency_hz: float, seconds: float, amplitude: float = 1.0, offset: float = 0.0) -> np.ndarray:
    time = np.arange(int(round(seconds * FS))) / FS
    return offset + amplitude * np.sin(2.0 * np.pi * frequency_hz * time)


def _features(segment: np.ndarray) -> dict[str, float]:
    return extract_features(
        segment,
        FS,
        BANDS,
        nperseg=256,
        noverlap=128,
        window="hann",
        spectral_edge_fraction=0.95,
    )


class BandpassTests(unittest.TestCase):
    def test_design_uses_second_order_sections(self):
        sos = design_bandpass(4, 0.5, 45.0, FS)
        self.assertEqual(sos.shape[1], 6)
        self.assertGreaterEqual(sos.shape[0], 4)
        self.assertTrue(np.all(np.isfinite(sos)))

    def test_filter_is_finite_on_noise_and_respects_pad_length(self):
        sos = design_bandpass(4, 0.5, 45.0, FS)
        edge = sos_pad_edge(sos)
        short = np.random.default_rng(0).normal(size=edge)
        long = np.random.default_rng(0).normal(scale=50.0, size=max(edge + 1, 4096))
        skipped = bandpass_filter(short, FS, 0.5, 45.0, 4, 1.0)
        kept = bandpass_filter(long, FS, 0.5, 45.0, 4, 1.0)
        self.assertEqual(skipped.n_segments_filtered, 0)
        self.assertTrue(np.all(np.isnan(skipped.values)))
        self.assertEqual(kept.n_segments_filtered, 1)
        self.assertTrue(np.all(np.isfinite(kept.values)))
        self.assertLess(float(np.max(np.abs(kept.values))), 1e6)

    def test_passband_is_kept_and_stopband_is_attenuated(self):
        def middle_rms(frequency_hz: float) -> float:
            raw = _tone(frequency_hz, 16.0, amplitude=10.0)
            filtered = bandpass_filter(raw, FS, 0.5, 45.0, 4, 1.0).values
            middle = filtered[4 * int(FS) : -4 * int(FS)]
            return float(np.sqrt(np.mean(middle**2)))

        raw_rms = float(np.sqrt(np.mean(_tone(10.0, 8.0, amplitude=10.0) ** 2)))
        self.assertGreater(middle_rms(10.0), 0.85 * raw_rms)
        self.assertLess(middle_rms(0.1), 0.05 * raw_rms)
        self.assertLess(middle_rms(60.0), 0.25 * raw_rms)

    def test_direct_current_is_removed(self):
        filtered = bandpass_filter(_tone(10.0, 16.0, amplitude=10.0, offset=100.0), FS, 0.5, 45.0, 4, 1.0)
        middle = filtered.values[4 * int(FS) : -4 * int(FS)]
        self.assertLess(abs(float(np.mean(middle))), 0.5)
        self.assertGreater(float(np.std(middle)), 5.0)

    def test_long_gap_is_not_filtered_across(self):
        left = _tone(10.0, 4.0)
        right = _tone(10.0, 4.0, amplitude=1.0)
        gap = np.full(2 * int(FS), np.nan)
        combined = np.concatenate([left, gap, right])
        filtered = bandpass_filter(combined, FS, 0.5, 45.0, 4, 1.0)
        left_alone = bandpass_filter(left, FS, 0.5, 45.0, 4, 1.0).values
        right_alone = bandpass_filter(right, FS, 0.5, 45.0, 4, 1.0).values
        self.assertEqual(filtered.n_long_gaps, 1)
        self.assertTrue(np.all(np.isnan(filtered.values[left.size : left.size + gap.size])))
        self.assertTrue(np.allclose(filtered.values[: left.size], left_alone, atol=1e-8))
        self.assertTrue(np.allclose(filtered.values[-right.size :], right_alone, atol=1e-8))

    def test_short_blip_between_long_gaps_is_left_missing(self):
        blip = _tone(10.0, 2.0 / FS)
        raw = np.concatenate([np.full(200, np.nan), blip, np.full(200, np.nan)])
        filtered = bandpass_filter(raw, FS, 0.5, 45.0, 4, 1.0)
        self.assertEqual(filtered.n_segments_skipped, 1)
        self.assertTrue(np.all(np.isnan(filtered.values)))


class WelchTests(unittest.TestCase):
    def test_detects_a_ten_hertz_tone(self):
        peak = dominant_frequency(_tone(10.0, 10.0), FS, 256, 128)
        self.assertAlmostEqual(peak, 10.0, delta=0.5)

    def test_band_power_follows_the_tone_and_relatives_sum_to_one(self):
        alpha = _features(_tone(10.0, 10.0))
        delta = _features(_tone(2.0, 10.0))
        noise = _features(np.random.default_rng(1).normal(size=1280))
        self.assertGreater(alpha["alpha_relative"], 0.8)
        self.assertGreater(delta["delta_relative"], 0.8)
        for features in (alpha, delta, noise):
            total = sum(features[f"{name}_relative"] for name, _, _ in BANDS)
            self.assertAlmostEqual(total, 1.0, places=8)
        self.assertGreater(alpha["spectral_edge_hz"], 8.0)
        self.assertLess(alpha["spectral_edge_hz"], 15.0)
        self.assertLess(alpha["spectral_entropy"], noise["spectral_entropy"])

    def test_time_features_of_a_unit_sine(self):
        sine = _tone(10.0, 10.0)
        features = time_features(sine)
        self.assertAlmostEqual(features["mean"], 0.0, places=8)
        self.assertAlmostEqual(features["rms"], float(np.sqrt(0.5)), places=3)
        self.assertAlmostEqual(features["variance"], 0.5, places=3)
        omega = 2.0 * np.pi * 10.0 / FS
        expected_mobility = 2.0 * np.sin(omega / 2.0)
        self.assertAlmostEqual(features["hjorth_mobility"], expected_mobility, delta=0.02)
        self.assertAlmostEqual(features["hjorth_complexity"], 1.0, delta=0.15)


class WindowTests(unittest.TestCase):
    def test_ten_second_windows_overlap_by_half(self):
        n_window, hop, windows = window_specs(3840, 0.0, 1.0 / FS, 10.0, 0.5)
        self.assertEqual(n_window, 1280)
        self.assertEqual(hop, 640)
        self.assertEqual(len(windows), 5)
        self.assertEqual(windows[0].start_index, 0)
        self.assertEqual(windows[0].stop_index, 1280)
        self.assertAlmostEqual(windows[0].end_sec - windows[0].start_sec, 10.0)
        self.assertAlmostEqual(windows[1].start_sec, 5.0)
        self.assertEqual(windows[1].start_index - windows[0].start_index, hop)
        self.assertEqual(windows[0].stop_index - windows[1].start_index, n_window // 2)
        self.assertAlmostEqual(windows[-1].end_sec, 30.0)

    def test_timestamps_follow_the_recording_clock(self):
        _, _, windows = window_specs(1280, 12.5, 1.0 / FS, 10.0, 0.5)
        self.assertEqual(len(windows), 1)
        self.assertAlmostEqual(windows[0].start_sec, 12.5)
        self.assertAlmostEqual(windows[0].end_sec, 22.5)


class BisAlignmentTests(unittest.TestCase):
    def test_median_uses_samples_inside_the_half_open_window(self):
        time = np.arange(0.0, 21.0, 1.0)
        values = time.copy()
        label = bis_in_window(5.0, 15.0, time, values, min_samples=3, bis_min=0.0, bis_max=100.0)
        self.assertIsNotNone(label)
        assert label is not None
        self.assertEqual(label["BIS_target"], 9.5)
        self.assertEqual(label["bis_n"], 10)
        self.assertEqual(label["bis_first_sec"], 5.0)
        self.assertEqual(label["bis_last_sec"], 14.0)

    def test_endpoint_belongs_to_the_next_window(self):
        time = np.array([0.0, 10.0])
        values = np.array([20.0, 80.0])
        first = bis_in_window(0.0, 10.0, time, values, min_samples=1, bis_min=0.0, bis_max=100.0)
        second = bis_in_window(10.0, 20.0, time, values, min_samples=1, bis_min=0.0, bis_max=100.0)
        assert first is not None and second is not None
        self.assertEqual(first["BIS_target"], 20.0)
        self.assertEqual(second["BIS_target"], 80.0)

    def test_out_of_range_and_sparse_bis_are_invalid(self):
        time = np.arange(10.0)
        too_high = bis_in_window(0.0, 10.0, time, np.full(10, 140.0), 3, 0.0, 100.0)
        too_few = bis_in_window(0.0, 10.0, time, np.arange(10.0), 11, 0.0, 100.0)
        self.assertIsNone(too_high)
        self.assertIsNone(too_few)

    def test_delay_shifts_only_the_bis_lookup(self):
        time = np.arange(0.0, 30.0, 1.0)
        baseline = bis_in_window(0.0, 10.0, time, time, 3, 0.0, 100.0, delay_sec=0.0)
        delayed = bis_in_window(0.0, 10.0, time, time, 3, 0.0, 100.0, delay_sec=5.0)
        assert baseline is not None and delayed is not None
        self.assertEqual(baseline["BIS_target"], 4.5)
        self.assertEqual(delayed["BIS_target"], 9.5)


class QualityTests(unittest.TestCase):
    def test_rejects_missing_flat_and_artifact_windows(self):
        missing = np.full(1280, np.nan)
        missing[:100] = _tone(10.0, 100 / FS)
        flat = np.zeros(1280)
        artifact = _tone(10.0, 10.0, amplitude=800.0)
        normal = _tone(10.0, 10.0, amplitude=20.0)
        self.assertIn(EXCESSIVE_MISSING, evaluate_eeg(missing, 0.05, 0.5, 500.0, 150.0).reasons)
        self.assertIn(FLATLINE, evaluate_eeg(flat, 0.05, 0.5, 500.0, 150.0).reasons)
        self.assertIn(SEVERE_ARTIFACT, evaluate_eeg(artifact, 0.05, 0.5, 500.0, 150.0).reasons)
        accepted = evaluate_eeg(normal, 0.05, 0.5, 500.0, 150.0)
        self.assertEqual(accepted.reasons, ())
        self.assertIsNotNone(accepted.signal)
        assert accepted.signal is not None
        self.assertEqual(accepted.signal.size, 1280)

    def test_interior_gap_is_filled_and_an_edge_gap_is_rejected(self):
        interior = _tone(10.0, 10.0, amplitude=20.0)
        interior[100:120] = np.nan
        quality = evaluate_eeg(interior, 0.05, 0.5, 500.0, 150.0)
        self.assertEqual(quality.reasons, ())
        self.assertEqual(quality.n_missing, 20)
        assert quality.signal is not None
        self.assertTrue(np.all(np.isfinite(quality.signal)))

        edge = _tone(10.0, 10.0, amplitude=20.0)
        edge[:10] = np.nan
        rejected = evaluate_eeg(edge, 0.05, 0.5, 500.0, 150.0)
        self.assertIn(EXCESSIVE_MISSING, rejected.reasons)
        self.assertIsNone(rejected.signal)

    def test_filtered_window_keeps_alpha_and_the_bis_label(self):
        raw = _tone(10.0, 30.0, amplitude=20.0)
        filtered = bandpass_filter(raw, FS, 0.5, 45.0, 4, 1.0).values
        _, _, windows = window_specs(raw.size, 0.0, 1.0 / FS, 10.0, 0.5)
        center = windows[2]
        quality = evaluate_eeg(filtered[center.start_index : center.stop_index], 0.05, 0.5, 500.0, 150.0)
        self.assertEqual(quality.reasons, ())
        assert quality.signal is not None
        features = _features(quality.signal)
        self.assertGreater(features["alpha_relative"], 0.8)
        bis_time = np.arange(0.0, 30.0, 1.0)
        label = bis_in_window(center.start_sec, center.end_sec, bis_time, np.full(bis_time.shape, 42.0), 3, 0.0, 100.0)
        assert label is not None
        self.assertEqual(label["BIS_target"], 42.0)
        self.assertIsNone(
            bis_in_window(center.start_sec, center.end_sec, np.array([]), np.array([]), 3, 0.0, 100.0)
        )
        self.assertEqual(INVALID_BIS, "invalid_bis")


class WindowArchiveTests(unittest.TestCase):
    def test_roundtrip_preserves_case_time_label_and_rate(self):
        payload = {
            "eeg": np.ones((2, 8), dtype=np.float32),
            "window_index": np.array([0, 1], dtype=np.int32),
            "window_start_sec": np.array([0.0, 5.0]),
            "window_end_sec": np.array([10.0, 15.0]),
            "bis_target": np.array([40.0, 55.0]),
            "case_id": np.int32(3),
            "sampling_rate_hz": np.float64(FS),
            "sample_interval_sec": np.float64(1.0 / FS),
            "bandpass_low_hz": np.float64(0.5),
            "bandpass_high_hz": np.float64(45.0),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case_00003.npz"
            save_processed_windows(path, payload)
            loaded = load_processed_windows(path)
        self.assertEqual(loaded["case_id"], 3)
        self.assertEqual(loaded["eeg"].shape, (2, 8))
        self.assertEqual(loaded["sampling_rate_hz"], FS)
        self.assertTrue(np.allclose(loaded["bis_target"], [40.0, 55.0]))
        self.assertTrue(np.allclose(loaded["window_start_sec"], [0.0, 5.0]))


if __name__ == "__main__":
    unittest.main()
