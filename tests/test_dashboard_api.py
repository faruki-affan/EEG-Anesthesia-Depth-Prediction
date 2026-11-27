"""Regression tests for dashboard input bounds and optional artifacts."""

from copy import deepcopy

from fastapi import HTTPException
import numpy as np
import pytest

from dashboard import api
from src.phase2 import save_processed_windows


@pytest.fixture
def isolated_dashboard(monkeypatch, tmp_path):
    config = deepcopy(api.CONFIG)
    config["paths"]["raw_dir"] = str(tmp_path / "raw")
    config["phase2"]["windows_dir"] = str(tmp_path / "pilot")
    config["temporal"]["cohort_dir"] = str(tmp_path / "expanded")
    config["temporal"]["models_dir"] = str(tmp_path)
    monkeypatch.setattr(api, "CONFIG", config)
    return tmp_path


def test_expanded_archive_is_discoverable_without_raw_or_pilot_data(isolated_dashboard):
    path = isolated_dashboard / "expanded" / "windows" / "case_00004.npz"
    save_processed_windows(path, {
        "eeg": np.zeros((1, 1280), dtype=np.float32),
        "window_index": np.array([0]),
        "window_start_sec": np.array([0.0]),
        "window_end_sec": np.array([10.0]),
        "bis_target": np.array([45.0]),
        "case_id": np.int32(4),
        "sampling_rate_hz": np.float64(128),
        "sample_interval_sec": np.float64(1 / 128),
    })
    assert api.cases() == [4]
    assert api.recording(4)["recorded"]["value"] == [45.0]
    assert api.signal(4, 0.0, 10.0)["raw"] is None


def test_large_signal_coordinates_and_corrupt_temporal_checkpoint(isolated_dashboard, monkeypatch):
    raw = isolated_dashboard / "raw" / "case_00001" / "eeg_primary.npz"
    raw.parent.mkdir(parents=True)
    raw.touch()
    samples = np.ones(1280)
    monkeypatch.setattr(api, "raw_signal", lambda *args: ({
        "values": samples, "sample_interval_sec": 1 / 128, "t_start_sec": 0.0,
    }, samples))
    with pytest.raises(HTTPException) as failure:
        api.signal(1, 1e308, 1.1e308)
    assert failure.value.status_code == 422
    (isolated_dashboard / "temporal.pt").write_bytes(b"corrupt")
    recording = api.recording(1)
    assert not recording["temporal_available"]
    assert recording["temporal_error"]
    with pytest.raises(HTTPException) as prediction:
        api.predict(1, "temporal")
    assert prediction.value.status_code == 422


def test_download_only_includes_files_and_features_for_selected_case(isolated_dashboard):
    from pathlib import Path
    from zipfile import ZipFile

    raw = isolated_dashboard / "raw" / "case_00004"
    raw.mkdir(parents=True)
    (raw / "eeg_primary.npz").write_bytes(b"case 4 waveform")
    (raw / "notes.txt").write_text("Excluded", encoding="utf-8")
    expanded = isolated_dashboard / "expanded"
    windows = expanded / "windows"
    windows.mkdir(parents=True)
    (windows / "case_00004.npz").write_bytes(b"case 4 processed windows")
    (windows / "case_00005.npz").write_bytes(b"case 5 processed windows")
    (expanded / "eeg_features.csv").write_text("case_id,rms\n4,1.0\n5,2.0\n", encoding="utf-8")

    response = api.download_case_dataset(4)
    try:
        assert "vitaldb_case_00004.zip" in response.headers["content-disposition"]
        with ZipFile(response.path) as bundle:
            assert set(bundle.namelist()) == {
                "case_00004/raw/eeg_primary.npz",
                "case_00004/processed/eeg_windows.npz",
                "case_00004/processed/eeg_features.csv",
                "case_00004/README.txt",
            }
            assert bundle.read("case_00004/processed/eeg_features.csv") == b"case_id,rms\n4,1.0\n"
    finally:
        Path(response.path).unlink()
