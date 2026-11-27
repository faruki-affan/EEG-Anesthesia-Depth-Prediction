"""Waveform-only CNN and portable inference checkpoints."""

import numpy as np
import torch
from torch import nn


def conv_block(in_channels, out_channels, kernel):
    return [
        nn.Conv1d(in_channels, out_channels, kernel, padding=kernel // 2),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.MaxPool1d(4),
    ]


class BISConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            *conv_block(1, 16, 15),
            *conv_block(16, 32, 9),
            *conv_block(32, 64, 5),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
        )

    def forward(self, eeg):
        return self.network(eeg).squeeze(-1)


def normalize(eeg, mean, scale):
    return torch.from_numpy(((eeg - mean) / scale).astype(np.float32))[:, None, :]


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["version"] != 1:
        raise ValueError("Unsupported CNN checkpoint version.")
    model = BISConvNet()
    model.load_state_dict(payload["weights"])
    return model.eval(), payload


@torch.inference_mode()
def predict_windows(checkpoint, eeg, sampling_rate_hz):
    model, payload = checkpoint
    if (
        eeg.ndim != 2
        or eeg.shape[1] != payload["samples"]
        or not np.isclose(sampling_rate_hz, payload["sampling_rate_hz"], rtol=0.02)
        or not np.isfinite(eeg).all()
    ):
        raise ValueError("EEG windows do not match the CNN checkpoint's input contract.")
    if not len(eeg):
        return np.empty(0)
    inputs = normalize(eeg, payload["mean"], payload["scale"])
    values = torch.cat([model(batch) for batch in inputs.split(payload["batch_size"])])
    return values.mul(100).clamp(0, 100).numpy()

