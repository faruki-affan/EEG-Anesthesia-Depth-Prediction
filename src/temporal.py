"""Temporal EEG encoder, history assembly, and inference."""

import pickle

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.cnn import BISConvNet, normalize
from src.infer_cnn import cached_case_archive, case_paths, raw_windows
from src.ml_data import resolve_path
from src.phase2 import load_processed_windows, validate_window_config


class TemporalReadout(nn.Module):
    """Predict a correction to the single-window estimate using past EEG."""

    def __init__(self):
        super().__init__()
        self.history = nn.Sequential(
            nn.Conv1d(65, 24, 3, padding=2),
            nn.ReLU(),
            nn.Conv1d(24, 16, 3, dilation=2, padding=4),
            nn.ReLU(),
            nn.Conv1d(16, 16, 3, dilation=2, padding=4),
            nn.ReLU(),
        )
        self.correction = nn.Linear(16, 1)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, batch):
        lag_axis = batch[:, :, :-1].transpose(1, 2)
        # The rightmost output on the original time axis uses present and past only.
        correction = self.correction(self.history(lag_axis)[:, :, lag_axis.shape[-1] - 1])
        return batch[:, -1, -1] + correction.squeeze(-1)


def history_indices(index, end, count, hop):
    """Return positions of available past windows, padding with -1 at gaps."""
    if count < 1 or len(index) != len(end):
        raise ValueError("History length and window coordinates must be valid.")
    positions = np.arange(len(index))
    starts = np.ones(len(index), dtype=bool)
    starts[1:] = (np.diff(index) != 1) | ~np.isclose(np.diff(end), hop, atol=1e-6, rtol=0)
    first = np.maximum.accumulate(np.where(starts, positions, 0))
    history = positions[:, None] - np.arange(count - 1, -1, -1)
    return np.where(history >= first[:, None], history, -1)


@torch.inference_mode()
def encode_cases(model, checkpoint, cases):
    model.eval()
    device = next(model.parameters()).device
    return {case: encode_windows(model, checkpoint, eeg, device)
            for case, (_, eeg, _) in cases.items()}


@torch.inference_mode()
def encode_windows(model, checkpoint, eeg, device):
    inputs = normalize(eeg, checkpoint["mean"], checkpoint["scale"])
    vectors = torch.cat([model.encode(batch.to(device)).cpu() for batch in inputs.split(256)])
    scores = model.readout(vectors.to(device)).cpu()
    return vectors.numpy(), scores.numpy()


def case_inputs(case, cases, encoded, count, hop):
    frame = cases[case][0]
    inputs = temporal_inputs(frame.window_index.to_numpy(), frame.window_end_sec.to_numpy(), encoded[case], count, hop)
    return inputs, torch.tensor(frame.BIS_target.to_numpy() / 100, dtype=torch.float32)


def temporal_inputs(window_index, window_end, encoded, count, hop):
    vectors, scores = encoded
    index = history_indices(window_index, window_end, count, hop)
    present = index >= 0
    inputs = np.zeros((*index.shape, 66), dtype=np.float32)
    inputs[present, :64] = vectors[index[present]]
    inputs[:, :, 64] = present
    inputs[:, -1, 65] = scores
    return torch.from_numpy(inputs)


def temporal_checkpoint_path(config):
    return resolve_path(config["temporal"]["models_dir"]) / "temporal.pt"


def load_temporal_checkpoint(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["version"] != 1:
            raise ValueError("Unsupported temporal CNN checkpoint version.")
        encoder = payload["encoder"]
        for field in ("train_cases", "validation_case", "signal_config", "weights",
                      "batch_size", "samples", "sampling_rate_hz", "mean", "scale"):
            encoder[field]
        for field in ("test_cases", "head", "history_windows", "hop_sec"):
            payload[field]
    except (EOFError, KeyError, TypeError, RuntimeError, pickle.UnpicklingError) as error:
        raise ValueError("The temporal CNN checkpoint is unreadable. Regenerate it with src.train_temporal.") from error
    return payload


def temporal_case_role(checkpoint, case):
    encoder = checkpoint["encoder"]
    if case in encoder["train_cases"]:
        return "Training case"
    if case == encoder["validation_case"]:
        return "Validation case"
    if case in checkpoint["test_cases"]:
        return "Held-out test case"
    return "Case outside the experimental cohort"


@torch.inference_mode()
def predict_temporal_case(case, config, checkpoint):
    encoder = checkpoint["encoder"]
    signal_config = encoder["signal_config"]
    raw_dir, pilot_archive = case_paths(config, case)
    archive_path = cached_case_archive(config, case, prefer_expanded=True)
    raw_path = raw_dir / "eeg_primary.npz"
    # Use the evaluated windows when the experiment cache exists.
    if archive_path != pilot_archive or not raw_path.exists():
        archive = load_processed_windows(archive_path)
        if archive["case_id"] != case:
            raise ValueError(f"Processed window archive belongs to a different case than {case}.")
        validate_window_config(archive, signal_config)
        order = np.argsort(archive["window_index"])
        frame = pd.DataFrame({
            "window_index": archive["window_index"][order],
            "Window end (s)": archive["window_end_sec"][order],
            "Recorded BIS": archive["bis_target"][order],
        })
        eeg, fs = archive["eeg"][order], archive["sampling_rate_hz"]
        source = "Cached EEG windows"
    else:
        frame, eeg, fs = raw_windows(raw_path, raw_dir / "bis.npz", signal_config)
        source = "Raw EEG windowed for inference"
    if frame.empty:
        raise ValueError("This recording has no usable EEG windows.")
    if eeg.shape != (len(frame), encoder["samples"]) or not np.isfinite(eeg).all() or not np.isclose(
        fs, encoder["sampling_rate_hz"], rtol=0.02
    ):
        raise ValueError("EEG windows do not match the temporal checkpoint's input contract.")

    model, head = BISConvNet().eval(), TemporalReadout().eval()
    model.load_state_dict(encoder["weights"])
    head.load_state_dict(checkpoint["head"])
    encoded = encode_windows(model, encoder, eeg, torch.device("cpu"))
    inputs = temporal_inputs(frame["window_index"].to_numpy(), frame["Window end (s)"].to_numpy(),
                             encoded, checkpoint["history_windows"], checkpoint["hop_sec"])
    frame["Predicted BIS"] = torch.cat(
        [head(batch) for batch in inputs.split(encoder["batch_size"])]
    ).mul(100).clamp(0, 100).numpy()
    return frame, f"{source} · Temporal CNN · {temporal_case_role(checkpoint, case)}"
