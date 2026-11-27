# EEG Anesthesia Depth Prediction

Estimate recorded Bispectral Index (BIS) from surgical EEG and explore the results
in an offline bedside-style research dashboard. The main model is a Temporal CNN
that uses up to 30 seconds of recent EEG. The dashboard also supports the
original single-window CNN and displays saved traditional machine-learning results.

## Quick start

Clone the repository and open PowerShell in the project directory. Use Python
3.11 and Node.js 22.12 or newer.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm --prefix web ci
npm --prefix web run build
.\.venv\Scripts\python.exe -m uvicorn dashboard.api:app --host 127.0.0.1 --port 8511
```

Open [EEG Research Monitor](http://127.0.0.1:8511/). Select **Case 00001**,
choose **Temporal CNN**, then click **Run inference**. Scrub or replay the
recording to inspect the estimates. Keep the terminal open while using the app.

The repository includes processed windows for Cases 1–3, CNN checkpoints, and
saved study reports. The dashboard can use the bundled filtered EEG without
downloading the raw tracks. Raw waveform views and raw files in case downloads
become available after the VitalDB loader runs.

## What the dashboard does

- Links the EEG waveform and recorded-versus-estimated BIS with a common playhead.
- Runs either CNN and identifies cases used for training, validation, or held-out testing.
- Shows Welch power spectral density, band power, and spectral entropy for the selected EEG interval.
- Exports the selected case’s available data as a ZIP and model predictions as a separate CSV.
- Compares the original pilot and expanded study using saved evaluation reports.

## Data and architecture

[VitalDB](https://physionet.org/content/vitaldb/1.0.0/) provides de-identified
surgical recordings. The project uses primary EEG (`BIS/EEG1_WAV`, nominally
128 Hz) and the monitor’s recorded BIS (`BIS/BIS`, roughly once per second).
BIS is the regression target.

1. The loader discovers cases with both tracks, verifies timing and recording
   coverage, and caches the tracks.
2. Phase 2 filters EEG at 0.5–45 Hz, creates 10-second windows with 50% overlap,
   checks quality, and assigns the median recorded BIS from each window.
3. Traditional models use 18 engineered features, including Welch band power,
   spectral entropy, and time-domain statistics.
4. The Single-window CNN uses filtered waveform samples. The Temporal CNN adds
   a learned correction from the current window and up to four preceding windows.
5. FastAPI serves the React dashboard, runs inference, and exposes saved reports.

### Reproduce the experiments

Run the stages you need from the repository root:

```powershell
# Download the original pilot raw tracks.
.\.venv\Scripts\python.exe -m src.vitaldb_loader

# Rebuild the original three-case dataset and train its models.
.\.venv\Scripts\python.exe -m src.phase2
.\.venv\Scripts\python.exe -m src.train_ml
.\.venv\Scripts\python.exe -m src.train_cnn --device cpu

# Prepare the separate expanded cohort and rerun the matched comparison.
.\.venv\Scripts\python.exe -m src.train_temporal --prepare --device cpu

# Verify the project.
.\.venv\Scripts\python.exe -m pytest tests -q
```

`config.yaml` defines the track names, quality thresholds, window settings, and
artifact locations. Omit `--prepare` to reuse a previously prepared expanded
cohort. The loader downloads a limited cohort and stores raw files locally.

## Matched held-out results

The expanded study used five training cases, one validation case, and three
test cases (1, 4, and 9). All three models were scored on the same 6,735 windows.

| Model | Case-averaged MAE (BIS points) | Pooled MAE (BIS points) |
| --- | ---: | ---: |
| **Temporal CNN** | **8.94** | **8.62** |
| Single-window CNN | 11.34 | 9.93 |
| Random Forest | 11.67 | 11.80 |

Lower MAE is better. The Temporal CNN reduced case-averaged error by about 21%
against the Single-window CNN in this experiment. The comparison is available in
[the expanded study report](outputs/reports/temporal/metrics.json).
The original Phase 3 and Phase 4 runs used different data splits.

## Project layout

| Path | Purpose |
| --- | --- |
| `src/` | Data acquisition, signal processing, training, and inference |
| `dashboard/api.py` | Local FastAPI routes for cases, signals, studies, predictions, and exports |
| `web/` | React, TypeScript, Vite, and Plotly frontend |
| `app.py` | Streamlit research dashboard |
| `data/processed/` | Bundled pilot windows and feature table |
| `models/` | Saved model checkpoints |
| `outputs/reports/` | Phase 3, Phase 4, and expanded study results |
| `tests/` | Tests for processing, models, dashboard APIs, and case exports |

## Research scope and attribution

This is an offline research and education project. Recorded BIS is a
monitor-derived reference. The model was evaluated on a small number of
independent surgical cases, the windows overlap, and performance varies with
the BIS range. Zero-phase filtering uses pre-recorded context. The current
results do not establish clinical reliability or real-time monitoring performance.

Data source: [VitalDB on PhysioNet](https://physionet.org/content/vitaldb/1.0.0/).
The bundled derived data have been filtered, windowed, and labeled using
recorded BIS. Please cite [Lee et al., Scientific Data, 2022](https://doi.org/10.1038/s41597-022-01411-5)
and follow the [dataset license](https://physionet.org/content/vitaldb/1.0.0/#license).
