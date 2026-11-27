"""FastAPI backend for the local EEG research dashboard."""

from functools import lru_cache
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import pandas as pd
import torch
from starlette.background import BackgroundTask

from dashboard.components import available_cases
from src.features import band_power, spectral_entropy, welch_psd
from src.infer_cnn import cached_case_archive, case_paths, filter_wave, predict_case
from src.ml_data import resolve_path
from src.phase2 import Phase2Error, load_processed_windows
from src.preprocess import true_runs
from src.temporal import load_temporal_checkpoint, predict_temporal_case, temporal_case_role, temporal_checkpoint_path
from src.vitaldb_loader import load_config, load_numeric, load_waveform

CONFIG = load_config()
torch.set_num_threads(CONFIG["phase4"]["cpu_threads"])
app = FastAPI(title="EEG Research Dashboard")


@lru_cache(maxsize=3)
def raw_signal(path: Path, modified_ns: int):
    wave = load_waveform(path)
    return wave, filter_wave(wave, CONFIG)


def nullable(values):
    return [float(value) if np.isfinite(value) else None for value in values]


def paths_for(case):
    if case not in available_cases(CONFIG):
        raise HTTPException(404, "This case is unavailable locally.")
    raw, _ = case_paths(CONFIG, case)
    return raw, cached_case_archive(CONFIG, case)


@lru_cache(maxsize=2)
def cached_checkpoint(path: Path, modified_ns: int):
    return load_temporal_checkpoint(path)


def temporal_payload():
    path = temporal_checkpoint_path(CONFIG)
    try:
        return cached_checkpoint(path, path.stat().st_mtime_ns)
    except FileNotFoundError:
        return None


@app.get("/api/cases")
def cases():
    return available_cases(CONFIG)


@app.get("/api/cases/{case}")
def recording(case: int):
    raw_dir, archive = paths_for(case)
    path = raw_dir / "eeg_primary.npz"
    if path.exists():
        wave, _ = raw_signal(path, path.stat().st_mtime_ns)
        values, interval = wave["values"], wave["sample_interval_sec"]
        start = wave["t_start_sec"]
        stop = start + len(values) * interval
        missing = float((~np.isfinite(values)).mean() * 100)
        search = values[:max(1, round(600 / interval))]
        usable = np.flatnonzero(np.isfinite(search))
        initial = start + int(usable[0]) * interval if usable.size else start
    else:
        cached = load_processed_windows(archive)
        if not len(cached["eeg"]):
            raise HTTPException(422, "No accepted EEG windows are available.")
        start, stop = cached["window_start_sec"][0], cached["window_end_sec"][-1]
        interval, missing, initial = cached["sample_interval_sec"], None, start
    checkpoint_error = None
    try:
        checkpoint = temporal_payload()
    except (OSError, ValueError) as error:
        checkpoint, checkpoint_error = None, str(error)
    role = temporal_case_role(checkpoint, case) if checkpoint else None
    bis_path = raw_dir / "bis.npz"
    if bis_path.exists():
        bis = load_numeric(bis_path)
        valid = np.isfinite(bis["time"]) & (bis["values"] >= 0) & (bis["values"] <= 100)
        time, values = bis["time"][valid], bis["values"][valid]
    elif archive.exists():
        cached = load_processed_windows(archive)
        time, values = cached["window_end_sec"], cached["bis_target"]
    else:
        time, values = np.array([]), np.array([])
    return dict(start=float(start), stop=float(stop), initial=float(initial), sampling_rate_hz=1 / interval,
                missing_percent=missing, has_raw=path.exists(), recorded=dict(time=time.tolist(), value=nullable(values)),
                temporal_role=role, temporal_available=checkpoint is not None,
                temporal_error=checkpoint_error)


@app.get("/api/cases/{case}/signal")
def signal(case: int, start: float = Query(allow_inf_nan=False), end: float = Query(allow_inf_nan=False)):
    raw_dir, archive = paths_for(case)
    if end <= start:
        raise HTTPException(422, "Select a time interval with a positive duration.")
    path = raw_dir / "eeg_primary.npz"
    if path.exists():
        wave, filtered = raw_signal(path, path.stat().st_mtime_ns)
        origin, interval = wave["t_start_sec"], wave["sample_interval_sec"]
        start = max(origin, start)
        end = min(origin + len(filtered) * interval, end)
        if end <= start:
            raise HTTPException(422, "The time interval is outside the EEG recording.")
        first = max(0, int(np.floor((start - origin) / interval)))
        stop = min(len(filtered), int(np.ceil((end - origin) / interval)))
        time = origin + np.arange(first, stop) * interval
        raw = wave["values"][first:stop]
        filtered = filtered[first:stop]
    else:
        cached = load_processed_windows(archive)
        if not len(cached["eeg"]):
            raise HTTPException(422, "No accepted EEG windows are available.")
        beginning = float(cached["window_start_sec"].min())
        ending = float(cached["window_end_sec"].max())
        if min(end, ending) <= max(start, beginning):
            raise HTTPException(422, "The time interval is outside the EEG recording.")
        first = int(np.argmin(np.abs(cached["window_start_sec"] - max(start, beginning))))
        filtered, interval = cached["eeg"][first], cached["sample_interval_sec"]
        time = cached["window_start_sec"][first] + np.arange(len(filtered)) * interval
        raw = None

    stride = max(1, int(np.ceil(len(time) / CONFIG["dashboard"]["max_plot_points"])))
    result = dict(time=time[::stride].tolist(), raw=nullable(raw[::stride]) if raw is not None else None,
                  filtered=nullable(filtered[::stride]), spectrum=None)
    runs = true_runs(np.isfinite(filtered))
    if not runs:
        return result
    left, right = max(runs, key=lambda run: run[1] - run[0])
    if right - left < 4:
        return result
    cfg = CONFIG["features"]
    size = min(cfg["welch_nperseg"], right - left)
    frequency, power = welch_psd(filtered[left:right], 1 / interval, size,
                                 min(size // 2, cfg["welch_noverlap"]), cfg["welch_window"])
    bands = [{"name": name.title(), "low": low, "high": high, "power": band_power(frequency, power, low, high)}
             for name, (low, high) in cfg["bands"].items()]
    in_band = (frequency >= min(band["low"] for band in bands)) & (frequency <= max(band["high"] for band in bands))
    entropy = spectral_entropy(power[in_band])
    result["spectrum"] = dict(frequency=frequency.tolist(), power=nullable(power), bands=bands,
                              entropy=float(entropy) if np.isfinite(entropy) else None,
                              contiguous_seconds=(right - left) * interval)
    return result


@app.post("/api/cases/{case}/predict")
def predict(case: int, model: Literal["temporal", "single_window"]):
    paths_for(case)
    try:
        if model == "temporal":
            checkpoint = temporal_payload()
            if checkpoint is None:
                raise HTTPException(404, "Run the temporal training experiment to create its checkpoint.")
            frame, source = predict_temporal_case(case, CONFIG, checkpoint)
            hop = checkpoint["hop_sec"]
        else:
            frame, source = predict_case(case, CONFIG)
            window = CONFIG["windows"]
            hop = window["length_sec"] * (1 - window["overlap_fraction"])
    except (OSError, ValueError, Phase2Error) as error:
        raise HTTPException(422, str(error)) from error
    return dict(time=frame["Window end (s)"].tolist(), recorded=nullable(frame["Recorded BIS"].astype(float)),
                predicted=nullable(frame["Predicted BIS"]), source=source, hop_sec=hop)


@app.get("/api/cases/{case}/dataset")
def download_case_dataset(case: int):
    raw, _ = paths_for(case)
    _, pilot = case_paths(CONFIG, case)
    archive = cached_case_archive(CONFIG, case, prefer_expanded=True)
    raw_files = []
    if raw.is_dir():
        raw_files = sorted(
            path for path in raw.iterdir()
            if path.is_file() and not path.is_symlink()
            and path.name.lower().endswith((".csv", ".csv.gz", ".npz"))
        )
    if not raw_files and not archive.is_file():
        raise HTTPException(404, "No dataset files are available for this case.")

    name = f"case_{case:05d}"
    with NamedTemporaryFile(suffix=".zip", prefix=f"vitaldb_{name}_", delete=False) as file:
        temporary = Path(file.name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=1) as bundle:
            for path in raw_files:
                compression = ZIP_STORED if path.suffix in (".gz", ".npz") else ZIP_DEFLATED
                bundle.write(path, f"{name}/raw/{path.name}", compress_type=compression)
            if archive.is_file():
                bundle.write(archive, f"{name}/processed/eeg_windows.npz", compress_type=ZIP_STORED)
                features = (archive.parent.parent / "eeg_features.csv" if archive != pilot
                            else resolve_path(CONFIG["phase2"]["features_csv"]))
                if features.is_file():
                    table = pd.read_csv(features)
                    selected = table.loc[table["case_id"].eq(case)]
                    if not selected.empty:
                        bundle.writestr(f"{name}/processed/eeg_features.csv", selected.to_csv(index=False, lineterminator="\n"))
            bundle.writestr(f"{name}/README.txt", (
                f"VitalDB case {case}\n"
                "This export contains locally available files for this case.\n"
                "raw/ contains waveform and numeric monitor tracks as available.\n"
                "processed/ contains filtered EEG windows and per-window features as available.\n"
                "In processed data, BIS is the recorded reference target.\n"
                "Source: VitalDB, Lee et al. (2022), https://doi.org/10.1038/s41597-022-01411-5\n"
                "Dataset: https://physionet.org/content/vitaldb/\n"
            ))
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return FileResponse(temporary, media_type="application/zip", filename=f"vitaldb_{name}.zip",
                        background=BackgroundTask(temporary.unlink, missing_ok=True))


@app.get("/api/studies")
def studies():
    result = {}
    for study, section in (("expanded", "temporal"), ("pilot", "phase4")):
        directory = resolve_path(CONFIG[section]["reports_dir"])
        path = directory / "metrics.json"
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(summary, dict) or not isinstance(summary.get("models"), dict):
            continue
        for filename, key in (("metrics_by_bis_range.csv", "ranges"), ("transitions.csv", "transitions")):
            report = directory / filename
            try:
                summary[key] = json.loads(pd.read_csv(report).to_json(orient="records"))
            except (OSError, ValueError):
                continue
        result[study] = summary
    return result


dist = resolve_path("web/dist")
if dist.is_dir():
    app.mount("/", StaticFiles(directory=dist, html=True), name="dashboard")
