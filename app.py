"""Streamlit demo for recorded VitalDB EEG and BIS research."""

import json
import math
from time import monotonic

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from dashboard.components import available_cases, educational_range
from src.features import band_power, welch_psd
from src.infer_cnn import case_paths, filter_wave, predict_case
from src.ml_data import MODEL_LABELS, resolve_path
from src.phase2 import Phase2Error, load_processed_windows
from src.preprocess import true_runs
from src.vitaldb_loader import load_config, load_numeric, load_waveform

CONFIG = load_config()
UI = CONFIG["dashboard"]
torch.set_num_threads(CONFIG["phase4"]["cpu_threads"])
st.set_page_config(page_title="EEG & BIS Research", page_icon="📈", layout="wide")


def line(x, y, x_label, y_label, title):
    chart = go.Figure(go.Scattergl(x=x, y=y, mode="lines", connectgaps=False))
    chart.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, height=300)
    st.plotly_chart(chart, width="stretch")


def overview():
    st.title("EEG-Based Depth of Anesthesia Estimation")
    st.write(
        "Anesthesia changes the brain's electrical activity. EEG records that activity at the scalp. "
        "The Bispectral Index (BIS) is a processed EEG measure recorded by the anesthesia monitor. "
        "This project estimates recorded BIS from EEG segments in VitalDB surgical recordings."
    )
    st.subheader("Workflow")
    st.write("VitalDB recording → EEG filtering and windowing → feature regression or 1D CNN → BIS comparison")
    st.subheader("Explore a case")
    st.write("Choose a recording in the sidebar to inspect its signals, run CNN inference, or replay its BIS timeline.")
    st.caption("Dataset and attribution")
    st.markdown("[VitalDB · Lee et al., Scientific Data (2022)](https://doi.org/10.1038/s41597-022-01411-5)")


def dataset(case):
    st.title("VitalDB Dataset Explorer")
    raw_dir, archive_path = case_paths(CONFIG, case)
    raw_path = raw_dir / "eeg_primary.npz"
    if raw_path.exists():
        waveform = load_waveform(raw_path)
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
                st.json(entry)


def signals(case):
    st.title("EEG Signal Visualization")
    raw_dir, archive_path = case_paths(CONFIG, case)
    raw_path = raw_dir / "eeg_primary.npz"
    if raw_path.exists():
        waveform = load_waveform(raw_path)
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
        filtered = filter_wave(waveform, CONFIG)[start:stop]
        stride = max(1, math.ceil((stop - start) / UI["max_plot_points"]))
        time = origin + np.arange(start, stop, stride) * interval
        line(time, raw[start:stop:stride], "Recording time (s)", "EEG (µV)", "Raw EEG")
        line(time, filtered[::stride], "Recording time (s)", "EEG (µV)", "Filtered EEG")
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
        line(time, filtered, "Recording time (s)", "EEG (µV)", "Filtered EEG")
    runs = true_runs(np.isfinite(filtered))
    if not runs or max(stop - start for start, stop in runs) < 4:
        st.info("Select an interval with at least four contiguous EEG samples to view its spectrum.")
        return
    start, stop = max(runs, key=lambda run: run[1] - run[0])
    segment = filtered[start:stop]
    if len(segment) < len(filtered):
        st.caption("Spectral estimates use the longest contiguous valid interval.")
    spectrum = np.abs(np.fft.rfft(segment - segment.mean())) * (2 / len(segment))
    spectrum[0] /= 2
    if len(segment) % 2 == 0:
        spectrum[-1] /= 2
    spectrum_stride = max(1, math.ceil(len(spectrum) / UI["max_plot_points"]))
    line(np.fft.rfftfreq(len(segment), 1 / fs)[::spectrum_stride], spectrum[::spectrum_stride], "Frequency (Hz)", "Amplitude (µV)", "Frequency spectrum")
    features = CONFIG["features"]
    size = min(len(segment), features["welch_nperseg"])
    frequency, psd = welch_psd(segment, fs, size, min(size // 2, features["welch_noverlap"]),
                               features["welch_window"])
    line(frequency, psd, "Frequency (Hz)", "Power density (µV²/Hz)", "Welch power spectral density")
    bands = {label: band_power(frequency, psd, *edges) for label, edges in features["bands"].items()}
    st.plotly_chart(px.bar(x=list(bands), y=list(bands.values()), labels={"x": "Band", "y": "Power (µV²)"}),
                    width="stretch")


def prediction(case, replay):
    st.title("Actual vs Predicted BIS" if replay else "BIS Prediction")
    key = f"inference_{case}"
    if st.button("Run CNN inference", type="primary"):
        with st.spinner("Preparing EEG windows and estimating BIS"):
            st.session_state[key] = predict_case(case, CONFIG)
    if key not in st.session_state:
        st.info("Run inference to view model estimates for this recording.")
        return
    frame, source = st.session_state[key]
    st.caption(source)
    if replay:
        playback(case, frame)
    else:
        times = frame["Window end (s)"].tolist()
        selected = st.select_slider("Window end (s)", options=times, format_func=lambda value: f"{value:.2f}")
        row = frame.loc[frame["Window end (s)"] == selected].iloc[0]
        predicted, actual = st.columns(2)
        predicted.metric("Predicted BIS", f"{row['Predicted BIS']:.1f}")
        actual.metric("Recorded BIS", f"{row['Recorded BIS']:.1f}" if pd.notna(row["Recorded BIS"]) else "Unavailable")
        st.caption(f"Educational interpretation · {educational_range(row['Predicted BIS'])}")


def playback(case, frame):
    raw_dir, _ = case_paths(CONFIG, case)
    waveform_path = raw_dir / "eeg_primary.npz"
    recorded_path = raw_dir / "bis.npz"
    first = float(frame["Window end (s)"].min())
    last = float(frame["Window end (s)"].max())
    if waveform_path.exists():
        wave = load_waveform(waveform_path)
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
                             value=UI["playback_default_speed"])
    autoplay = st.toggle("Play recording")
    origin = monotonic()

    @st.fragment(run_every=UI["playback_refresh_sec"] if autoplay else None)
    def view():
        if autoplay:
            end = min(last, first + (monotonic() - origin) * speed)
        elif first < last:
            end = st.slider("Show through (s)", first, last, last)
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
        chart.update_layout(xaxis_title="Recording time (s)", yaxis_title="BIS",
                            yaxis_range=[0, 100], xaxis_range=[first - 1, last + 1])
        st.plotly_chart(chart, width="stretch")

    view()


def performance():
    st.title("Model Performance")
    path = resolve_path(CONFIG["phase4"]["reports_dir"]) / "metrics.json"
    if not path.exists():
        st.info("Run python -m src.train_cnn to generate held-out evaluation results.")
        return
    summary = json.loads(path.read_text(encoding="utf-8"))
    labels = {**MODEL_LABELS, "cnn": "1D CNN"}
    metrics = pd.DataFrame(summary["models"]).T.rename(index=labels)
    fields = {"mae": "MAE", "rmse": "RMSE", "r2": "R²", "pearson_r": "Pearson r"}
    st.caption("Each test case was excluded from model training and CNN early stopping. All models used the same training cases in each fold.")
    st.dataframe(metrics[list(fields)].rename(columns=fields))
    measure = st.selectbox("Comparison metric", list(fields.values()))
    st.plotly_chart(px.bar(metrics.rename(columns=fields), y=measure, labels={"index": "Model"}), width="stretch")
    st.write(f"{summary['n_cases']} held-out cases · {summary['n_windows']:,} windows · "
             f"{summary['usable_eeg_seconds'] / 3600:.2f} hours of usable EEG")
    if summary["rejected_windows"] is not None:
        st.write(f"{summary['rejected_windows']:,} rejected windows")
    st.dataframe(pd.DataFrame(summary["folds"]), hide_index=True)
    with st.expander("Per-case metrics and BIS ranges"):
        per_case = pd.DataFrame(summary["by_case"])
        per_case["model"] = per_case["model"].map(labels)
        st.dataframe(per_case[["model", "case_id", "n_windows", *fields]].rename(columns=fields), hide_index=True)
        ranges = pd.read_csv(path.with_name("metrics_by_bis_range.csv"))
        ranges["model"] = ranges["model"].map(labels)
        st.dataframe(ranges, hide_index=True)
    figures = resolve_path(CONFIG["phase4"]["figures_dir"])
    with st.expander("Evaluation figures"):
        for name in ("loss.png", "held_out_agreement.png", *[f"timeline_case_{case:05d}.png" for case in summary["test_cases"]]):
            figure = figures / name
            if figure.exists():
                st.image(str(figure))


pages = ("Project Overview", "VitalDB Dataset Explorer", "EEG Signal Visualization",
         "BIS Prediction", "Actual vs Predicted BIS", "Model Performance")
page = st.sidebar.radio("Page", pages)
st.sidebar.caption("Research and education only. Clinical use is outside the scope of this project.")
if page == pages[0]:
    overview()
elif page == pages[-1]:
    performance()
else:
    cases = available_cases(CONFIG)
    if not cases:
        st.info("No local EEG recordings found. Run Phase 1 or add the bundled Phase 2 windows.")
        st.stop()
    case = st.sidebar.selectbox("VitalDB case", cases)
    try:
        if page == pages[1]:
            dataset(case)
        elif page == pages[2]:
            signals(case)
        else:
            prediction(case, replay=page == pages[4])
    except (OSError, ValueError, Phase2Error) as error:
        st.error(str(error))

