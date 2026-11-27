"""Verify temporal history boundaries and the baseline checkpoint contract."""

import numpy as np
import torch

from src.cnn import BISConvNet
from src.train_temporal import TemporalReadout, history_indices


def test_history_restarts_at_rejected_window_and_clock_gap():
    index = np.array([0, 1, 3, 4, 5])
    end = np.array([10, 15, 25, 30, 40])
    np.testing.assert_array_equal(history_indices(index, end, 3, 5), [
        [-1, -1, 0], [-1, 0, 1], [-1, -1, 2], [-1, 2, 3], [-1, -1, 4],
    ])


def test_temporal_head_starts_at_exact_single_window_estimate():
    readout = TemporalReadout().eval()
    batch = torch.randn(3, 11, 66)
    torch.testing.assert_close(readout(batch), batch[:, -1, -1])


def test_refactored_encoder_preserves_original_prediction():
    model = BISConvNet().eval()
    eeg = torch.randn(3, 1, 1280)
    with torch.no_grad():
        torch.testing.assert_close(model(eeg), model.network(eeg).squeeze(-1))

def test_temporal_dashboard_integration():
    from pathlib import Path

    import pandas as pd
    import pytest
    from streamlit.testing.v1 import AppTest

    from src.ml_data import resolve_path
    from src.temporal import load_temporal_checkpoint, predict_temporal_case, temporal_checkpoint_path
    from src.vitaldb_loader import load_config

    config = load_config()
    checkpoint_path = temporal_checkpoint_path(config)
    reference_path = resolve_path(config["temporal"]["reports_dir"]) / "test_predictions.csv"
    if not checkpoint_path.exists() or not reference_path.exists():
        pytest.skip("Expanded study artifacts have not been generated.")

    checkpoint = load_temporal_checkpoint(checkpoint_path)
    torch.set_num_threads(config["phase4"]["cpu_threads"])
    case = checkpoint["test_cases"][0]
    frame, source = predict_temporal_case(case, config, checkpoint)
    assert "Held-out test case" in source
    expected = pd.read_csv(reference_path).query("case_id == @case").set_index("window_index")
    np.testing.assert_allclose(frame["Predicted BIS"], expected.loc[frame.window_index, "temporal"], atol=1e-4)

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
    app.selectbox(key="case_picker").set_value(case).run(timeout=30)
    assert app.selectbox(key=f"model_{case}").value == "Temporal CNN"
    assert app.selectbox(key="evaluation_study").value == "Expanded temporal study"
    app.button[0].click().run(timeout=30)
    assert not app.exception and not app.error
    assert any(metric.label == "Predicted BIS" for metric in app.metric)

    seen = checkpoint["encoder"]["train_cases"][0]
    app.selectbox(key="case_picker").set_value(seen).run(timeout=30)
    assert app.selectbox(key=f"model_{seen}").value == "Single-window CNN"
    app.selectbox(key=f"model_{seen}").set_value("Temporal CNN").run(timeout=30)
    assert any("contributed to model development" in warning.value for warning in app.warning)
