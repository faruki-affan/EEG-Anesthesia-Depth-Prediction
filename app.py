"""Streamlit demo for recorded VitalDB EEG and BIS research."""

import json
import math
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from dashboard.components import available_cases, educational_range
from src.features import band_power, welch_psd
from src.infer_cnn import cached_case_archive, case_paths, filter_wave, predict_case
from src.ml_data import MODEL_LABELS, resolve_path
from src.phase2 import Phase2Error, load_processed_windows
from src.preprocess import true_runs
from src.temporal import load_temporal_checkpoint, predict_temporal_case, temporal_case_role, temporal_checkpoint_path
from src.vitaldb_loader import load_config, load_numeric, load_waveform

CONFIG = load_config()
UI = CONFIG["dashboard"]
torch.set_num_threads(CONFIG["phase4"]["cpu_threads"])
st.set_page_config(page_title="EEG & BIS Research", page_icon="📈", layout="wide")


@st.cache_data(max_entries=3, show_spinner=False)
def cached_waveform(path: Path, modified_ns: int):
    return load_waveform(path)


@st.cache_data(max_entries=3, show_spinner=False)
def cached_filtered(path: Path, modified_ns: int, config):
    return filter_wave(cached_waveform(path, modified_ns), config)


def read_waveform(path):
    return cached_waveform(path, path.stat().st_mtime_ns)


def line(x, y, x_label, y_label, title, key):
    chart = go.Figure(go.Scattergl(x=x, y=y, mode="lines", connectgaps=False))
    chart.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, height=300)
    st.plotly_chart(chart, width="stretch", key=key)


def overview():
    st.title("EEG & Anesthesia Depth")
    st.write(
        "Explore a real surgical recording from VitalDB. EEG measures electrical activity at the scalp. "
        "The Bispectral Index (BIS) is a monitor-derived score from 0 to 100. "
        "This project uses short EEG segments to estimate the BIS score recorded during surgery."
    )
    st.caption("Research and education only. This dashboard is for offline analysis of recorded data.")
    st.markdown(
        "[Recording](#recording) · [EEG explorer](#eeg-explorer) · "
        "[BIS prediction](#bis-prediction) · [Model performance](#model-performance)"
    )


def dataset(case):
    st.header("Recording", anchor="recording")
    st.caption("Sampling rate is the number of EEG measurements per second. Missing EEG is the share of expected samples with no usable value.")
    raw_dir, _ = case_paths(CONFIG, case)
    archive_path = cached_case_archive(CONFIG, case)
    raw_path = raw_dir / "eeg_primary.npz"
    if raw_path.exists():
        waveform = read_waveform(raw_path)
        values = waveform["values"]
        interval = waveform["sample_interval_sec"]
        duration, fs = values.size * interval, 1 / interval
        missing = f"{100 * (~np.isfinite(values)).mean():.1f}%"
    else:
        archive = load_processed_windows(archive_path)
        duration = None
        fs, missing = archive["sampling_rate_hz"], "Unavailable"
        st.info("The local archive contains accepted filtered windows. Download the original case to inspect raw-track coverage.")
    duration_card, fs_card, missing_card = st.columns(3)
    duration_card.metric("EEG recording duration", f"{duration / 60:.1f} min" if duration is not None else "Unavailable")
    fs_card.metric("Sampling rate", f"{fs:.1f} Hz")
    missing_card.metric("Raw EEG missing", missing)
    catalog = CONFIG["tracks"]
    tracks = [
        (catalog["eeg_primary"], raw_path),
        (catalog["eeg_secondary"], raw_dir / "eeg_secondary.npz"),
        (catalog["bis"], raw_dir / "bis.npz"),
    ]
    st.dataframe(pd.DataFrame([
        {"Track": name, "Local data": "Raw track cached" if file.exists() else
         "Available in accepted windows" if archive_path.exists() and index != 1 else "Unavailable"}
        for index, (name, file) in enumerate(tracks)
    ]), hide_index=True)
    summary = resolve_path(CONFIG["phase2"]["summary_json"])
    if summary.exists():
        cases = json.loads(summary.read_text(encoding="utf-8"))["cases"]
        entry = next((item for item in cases if item["case_id"] == case), None)
        if entry:
            with st.expander("Phase 2 data quality and rejection counts"):
                st.caption("Filtering prepares the signal. The quality checks reject windows with missing data, flatlines, large artifacts, or insufficient valid BIS labels.")
                st.json(entry)


def signals(case):
    st.header("EEG explorer", anchor="eeg-explorer")
    st.write("Choose a time interval and explore how EEG changes after filtering, or see which frequencies carry the most power.")
    display = st.radio("View", ["Waveform", "Frequency spectrum", "Power spectral density", "Band power"], horizontal=True, key="eeg_view")
    raw_dir, _ = case_paths(CONFIG, case)
    archive_path = cached_case_archive(CONFIG, case)
    raw_path = raw_dir / "eeg_primary.npz"
    if raw_path.exists():
        waveform = read_waveform(raw_path)
        interval = waveform["sample_interval_sec"]
        fs = 1 / interval
        origin = waveform["t_start_sec"]
        raw = waveform["values"]
        end = origin + len(raw) * interval
        width = min(UI["excerpt_sec"], end - origin)
        finite = np.flatnonzero(np.isfinite(raw))
        initial = min(end - width, origin + int(finite[0]) * interval) if finite.size else origin
        left, right = st.slider("Recording time (s)", float(origin), float(end),
                                (float(initial), float(initial + width)), key=f"signal_range_{case}")
        start = max(0, round((left - origin) * fs))
        stop = min(len(raw), round((right - origin) * fs))
        filtered = cached_filtered(raw_path, raw_path.stat().st_mtime_ns, CONFIG)[start:stop]
        stride = max(1, math.ceil((stop - start) / UI["max_plot_points"]))
        time = origin + np.arange(start, stop, stride) * interval
        if display == "Waveform":
            st.caption("Raw EEG shows recorded voltage in microvolts (µV). The 0.5–45 Hz filter in the default configuration reduces slow drift and high-frequency components.")
            chart = go.Figure()
            chart.add_scattergl(x=time, y=raw[start:stop:stride], mode="lines", name="Raw EEG", opacity=0.55)
            chart.add_scattergl(x=time, y=filtered[::stride], mode="lines", name="Filtered EEG")
            chart.update_layout(xaxis_title="Recording time (s)", yaxis_title="EEG (µV)", height=320,
                                hovermode="x unified", legend_orientation="h")
            st.plotly_chart(chart, width="stretch", key="raw_filtered_eeg")
            return
    else:
        st.info("Raw EEG is unavailable. Showing filtered EEG from the bundled windows.")
        archive = load_processed_windows(archive_path)
        if len(archive["eeg"]) == 0:
            st.warning("No accepted EEG windows are available.")
            return
        windows = archive["window_start_sec"]
        index = st.select_slider("Window start (s)", options=range(len(windows)),
                                 format_func=lambda item: f"{windows[item]:.2f}")
        filtered = archive["eeg"][index]
        fs = archive["sampling_rate_hz"]
        time = windows[index] + np.arange(len(filtered)) / fs
        if display == "Waveform":
            st.caption("Filtered EEG shows voltage after reducing components outside the configured frequency band.")
            line(time, filtered, "Recording time (s)", "EEG (µV)", "Filtered EEG", "filtered_eeg")
            return
    runs = true_runs(np.isfinite(filtered))
    if not runs or max(stop - start for start, stop in runs) < 4:
        st.info("Select an interval with at least four contiguous EEG samples to view its spectrum.")
        return
    start, stop = max(runs, key=lambda run: run[1] - run[0])
    segment = filtered[start:stop]
    if len(segment) < len(filtered):
        st.caption("Spectral estimates use the longest contiguous valid interval.")
    if display == "Frequency spectrum":
        st.caption("This spectrum shows the amplitude of each rhythm. A frequency of 10 Hz means ten cycles per second.")
        spectrum = np.abs(np.fft.rfft(segment - segment.mean())) * (2 / len(segment))
        spectrum[0] /= 2
        if len(segment) % 2 == 0:
            spectrum[-1] /= 2
        spectrum_stride = max(1, math.ceil(len(spectrum) / UI["max_plot_points"]))
        line(np.fft.rfftfreq(len(segment), 1 / fs)[::spectrum_stride], spectrum[::spectrum_stride], "Frequency (Hz)", "Amplitude (µV)", "Frequency spectrum", "spectrum")
        return
    features = CONFIG["features"]
    size = min(len(segment), features["welch_nperseg"])
    frequency, psd = welch_psd(segment, fs, size, min(size // 2, features["welch_noverlap"]),
                               features["welch_window"])
    if display == "Power spectral density":
        st.caption("Power spectral density shows where signal power is concentrated. Welch's method averages estimates from overlapping segments to smooth the curve.")
        line(frequency, psd, "Frequency (Hz)", "Power density (µV²/Hz)", "Welch power spectral density", "welch_psd")
        return
    st.caption("Band power adds up the power within each frequency range. Relative power shows the share of the total across these bands.")
    bands = {label: band_power(frequency, psd, *edges) for label, edges in features["bands"].items()}
    scale = st.radio("Band power scale", ["Absolute", "Relative"], horizontal=True, key="band_scale")
    total = sum(bands.values())
    values = [100 * value / total if total else 0 for value in bands.values()] if scale == "Relative" else list(bands.values())
    labels = [f"{band.title()} ({low:g}–{high:g} Hz)" for band, (low, high) in features["bands"].items()]
    unit = "Share of band power (%)" if scale == "Relative" else "Power (µV²)"
    st.plotly_chart(px.bar(x=labels, y=values, labels={"x": "EEG band", "y": unit}),
                    width="stretch", key="band_power")


def prediction(case):
    st.header("BIS prediction", anchor="bis-prediction")
    st.write(
        "The temporal CNN uses recent EEG history. The single-window CNN uses one EEG segment at a time. "
        "Recorded BIS comes from the VitalDB monitor track."
    )
    checkpoint_path = temporal_checkpoint_path(CONFIG)
    try:
        checkpoint = load_temporal_checkpoint(checkpoint_path) if checkpoint_path.exists() else None
    except (OSError, ValueError):
        checkpoint = None
        st.warning("The temporal checkpoint could not be loaded. The single-window CNN is still available.")
    options = ["Single-window CNN", "Temporal CNN"] if checkpoint else ["Single-window CNN"]
    role = temporal_case_role(checkpoint, case) if checkpoint else None
    default = 1 if checkpoint and role not in ("Training case", "Validation case") else 0
    model = st.selectbox("Prediction model", options, index=default, key=f"model_{case}")
    use_temporal = model == "Temporal CNN"
    if use_temporal:
        seconds = checkpoint["encoder"]["signal_config"]["windows"]["length_sec"]
        seconds += (checkpoint["history_windows"] - 1) * checkpoint["hop_sec"]
        st.caption(f"Up to {seconds:g} seconds of EEG history. History restarts after gaps. {role}.")
        if role in ("Training case", "Validation case"):
            st.warning("This case contributed to model development. Use its predictions for exploration.")
    elif role in ("Training case", "Validation case"):
        st.caption(f"The single-window model is selected because this was a {role.lower()} for the temporal CNN.")
    model_key = "temporal" if use_temporal else "single_window"
    key = f"inference_{case}_{model_key}"
    if st.button("Run CNN inference", type="primary"):
        with st.spinner("Preparing EEG windows and estimating BIS"):
            st.session_state[key] = (
                predict_temporal_case(case, CONFIG, checkpoint) if use_temporal else predict_case(case, CONFIG)
            )
    if key not in st.session_state:
        st.info("Run inference to view model estimates for this recording.")
        return
    frame, source = st.session_state[key]
    st.caption(source)
    if frame.empty:
        st.info("This case has no usable windows to display.")
        return
    times = frame["Window end (s)"].tolist()
    selected = st.select_slider("Inspect a prediction at window end (s)", options=times,
                                format_func=lambda value: f"{value:.2f}", key=f"prediction_time_{case}",
                                help="The CNN uses the EEG leading up to this time. Recorded BIS here is the median of valid monitor readings in the same window.")
    row = frame.loc[frame["Window end (s)"] == selected].iloc[0]
    estimate, actual, error = st.columns(3)
    estimate.metric("Predicted BIS", f"{row['Predicted BIS']:.1f}")
    has_reference = pd.notna(row["Recorded BIS"])
    actual.metric("Recorded BIS", f"{row['Recorded BIS']:.1f}" if has_reference else "Unavailable")
    error.metric("Prediction error", f"{row['Predicted BIS'] - row['Recorded BIS']:+.1f}" if has_reference else "Unavailable",
                 help="Predicted minus recorded BIS in points. Positive means the estimate is higher.")
    st.caption(f"Approximate educational interpretation · {educational_range(row['Predicted BIS'])}.")
    st.download_button("Download case predictions (CSV)", frame.to_csv(index=False), f"case_{case:05d}_{model_key}_bis.csv", "text/csv")
    playback(case, frame, selected)


def playback(case, frame, selected):
    st.subheader("Compare and replay")
    st.caption("Hover over points for exact values. Zoom to inspect part of the case or use Play recording to reveal the timeline.")
    raw_dir, _ = case_paths(CONFIG, case)
    waveform_path = raw_dir / "eeg_primary.npz"
    recorded_path = raw_dir / "bis.npz"
    first = float(frame["Window end (s)"].min())
    last = float(frame["Window end (s)"].max())
    if waveform_path.exists():
        wave = read_waveform(waveform_path)
        first = wave["t_start_sec"]
        last = first + len(wave["values"]) * wave["sample_interval_sec"]
    if recorded_path.exists():
        recorded = load_numeric(recorded_path)
        valid = np.isfinite(recorded["time"]) & (recorded["values"] >= 0) & (recorded["values"] <= 100)
        actual_time, actual_values = recorded["time"][valid], recorded["values"][valid]
        if len(actual_time):
            first = min(first, float(actual_time.min()))
            last = max(last, float(actual_time.max()))
        st.caption("Offline replay. Recorded BIS uses monitor timestamps and predictions appear at EEG window ends.")
    else:
        actual_time = frame["Window end (s)"].to_numpy()
        actual_values = frame["Recorded BIS"].astype(float).to_numpy()
        st.caption("Offline replay of available EEG windows. Each point represents one window.")
    speed = st.select_slider("Replay speed", options=[1, 10, 30, 60, 120, 300],
                             value=UI["playback_default_speed"], format_func=lambda value: f"{value}×", key=f"speed_{case}")
    autoplay = st.toggle("Play recording", key=f"play_{case}")
    origin = monotonic()

    @st.fragment(run_every=UI["playback_refresh_sec"] if autoplay else None)
    def view():
        if autoplay:
            end = min(last, first + (monotonic() - origin) * speed)
        elif first < last:
            end = st.slider("Show through (s)", first, last, last, key=f"replay_end_{case}")
        else:
            end = last
        st.caption(f"Recording time {end:.1f} s")
        shown = frame.loc[frame["Window end (s)"] <= end]
        chart = go.Figure()
        visible = actual_time <= end
        chart.add_trace(go.Scattergl(x=actual_time[visible], y=actual_values[visible],
                                     mode="markers", marker={"size": 3}, name="Recorded BIS"))
        chart.add_trace(go.Scattergl(x=shown["Window end (s)"], y=shown["Predicted BIS"],
                                     mode="markers", marker={"size": 4}, name="Predicted BIS"))
        if selected <= end:
            chart.add_vline(x=selected, line_dash="dot", line_color="#667085")
        chart.update_layout(xaxis_title="Recording time (s)", yaxis_title="BIS",
                            yaxis_range=[0, 100], xaxis_range=[first - 1, last + 1])
        st.plotly_chart(chart, width="stretch", key=f"bis_timeline_{case}")

    view()


def show_report_table(path, labels, description="This breakdown"):
    try:
        table = pd.read_csv(path)
        table["model"] = table["model"].map(labels)
    except (OSError, ValueError, KeyError):
        st.info(f"{description} is unavailable. Rerun training to regenerate the report.")
    else:
        st.dataframe(table, hide_index=True)


def performance():
    st.header("Model performance", anchor="model-performance")
    pilot = resolve_path(CONFIG["phase4"]["reports_dir"]) / "metrics.json"
    expanded = resolve_path(CONFIG["temporal"]["reports_dir"]) / "metrics.json"
    reports = {"Expanded temporal study": expanded, "Original Phase 4 pilot": pilot}
    available = [name for name, path in reports.items() if path.exists()]
    if not available:
        st.info("Run python -m src.train_cnn or python -m src.train_temporal to generate evaluation results.")
        return
    study = st.selectbox("Evaluation study", available, key="evaluation_study")
    path = reports[study]
    is_expanded = path == expanded
    summary = json.loads(path.read_text(encoding="utf-8"))
    labels = {**MODEL_LABELS, "cnn": "1D CNN", "single_window": "Single-window CNN", "temporal": "Temporal CNN"}
    metrics = pd.DataFrame(summary["models"]).T.rename(index=labels)
    fields = {"mae": "MAE", "rmse": "RMSE", "r2": "R²", "pearson_r": "Pearson r"}
    if is_expanded:
        fields = {"case_averaged_mae": "Case-averaged MAE", **fields}
        st.caption("All three models used the same expanded study split. Inference-model selection above does not change these saved test results.")
    else:
        st.caption("Each test case was excluded from model training and CNN early stopping. All models used the same training cases in each fold.")
    st.dataframe(metrics[list(fields)].rename(columns=fields))
    descriptions = {
        "Case-averaged MAE": "Calculate MAE for each test case, then average across cases with equal weight. Lower is better.",
        "MAE": "Mean absolute error is the average miss in BIS points. Lower is better.",
        "RMSE": "Root mean squared error is in BIS points and gives larger misses more weight. Lower is better.",
        "R²": "R² compares squared error with predicting the test set mean. One is perfect, zero matches that reference, and negative values indicate worse performance.",
        "Pearson r": "Pearson r measures how closely estimates and recorded values move together linearly. A value near +1 indicates strong positive association. Estimates can still be systematically high or low.",
    }
    measure = st.selectbox("Comparison metric", list(fields.values()))
    st.caption(descriptions[measure])
    st.plotly_chart(px.bar(metrics.rename(columns=fields), y=measure, labels={"index": "Model"}), width="stretch")
    if is_expanded:
        st.write(f"{len(summary['test_cases'])} held-out cases · {summary['n_test_windows']:,} test windows · "
                 f"{summary['history_seconds']:g} s of EEG history selected on validation")
        with st.expander("How the expanded models were tested"):
            st.write(f"Training cases {summary['training_cases']} · Validation case {summary['validation_case']} · "
                     f"Test cases {summary['test_cases']}")
        for filename, title in (
            ("metrics_by_case.csv", "Per-case metrics"),
            ("metrics_by_bis_range.csv", "BIS ranges"),
            ("transitions.csv", "Transition tracking"),
        ):
            with st.expander(title):
                if filename == "transitions.csv":
                    st.caption("Lag is reported for matched same-direction crossings of BIS 60 within 30 seconds. Check matched counts alongside lag.")
                show_report_table(path.with_name(filename), labels)
        return
    st.write(f"{summary['n_cases']} held-out cases · {summary['n_windows']:,} windows · "
             f"{summary['usable_eeg_seconds'] / 3600:.2f} hours of usable EEG")
    if summary["rejected_windows"] is not None:
        st.write(f"{summary['rejected_windows']:,} rejected windows")
    with st.expander("How the models were tested"):
        st.write(
            "Each fold is a separate experiment. The training case fits the model, the validation case selects "
            "the CNN checkpoint, and the test case measures performance on a recording excluded from both steps. "
            "The median baseline predicts one typical training-set value. The two tree models use summary "
            "features, while the CNN uses the filtered waveform."
        )
        st.dataframe(pd.DataFrame(summary["folds"]), hide_index=True)
    with st.expander("Per-case metrics and BIS ranges"):
        per_case = pd.DataFrame(summary["by_case"])
        per_case["model"] = per_case["model"].map(labels)
        st.dataframe(per_case[["model", "case_id", "n_windows", *fields]].rename(columns=fields), hide_index=True)
        show_report_table(path.with_name("metrics_by_bis_range.csv"), labels, "The BIS range breakdown")
    figures = resolve_path(CONFIG["phase4"]["figures_dir"])
    with st.expander("Evaluation figures"):
        for name in ("loss.png", "held_out_agreement.png", *[f"timeline_case_{case:05d}.png" for case in summary["test_cases"]]):
            figure = figures / name
            if figure.exists():
                st.image(str(figure))


overview()
cases = available_cases(CONFIG)
if cases:
    case = st.selectbox("VitalDB surgical case", cases, format_func=lambda value: f"Case {value:05d}",
                        key="case_picker", help="This selection updates the recording, EEG explorer, prediction, and replay together.")
    for section in (dataset, signals, prediction):
        try:
            section(case)
        except (OSError, ValueError, Phase2Error) as error:
            st.error(str(error))
else:
    st.info("No local EEG recordings found. Run Phase 1 or add the bundled Phase 2 windows.")

performance()
st.caption("Recordings and derived windows are from VitalDB.")
st.markdown("[VitalDB · Lee et al., Scientific Data (2022)](https://doi.org/10.1038/s41597-022-01411-5)")
