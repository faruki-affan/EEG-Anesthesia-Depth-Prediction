# EEG-Based Depth of Anesthesia Estimation

Educational project that predicts the recorded Bispectral Index (BIS) from intraoperative EEG. The current tree contains **Phase 1** (VitalDB discovery, a small real-data download, and a waveform plot), **Phase 2** (bandpass filtering, windowing, Welch spectral features, and BIS alignment), and **Phase 3** (leave-one-case-out regression from those features). Phases 4–6 add 1D CNN training and evaluation, a Streamlit dashboard, and reproducibility tests.

This software is for education and research. It does not make clinical decisions and it does not reproduce the proprietary BIS algorithm.

## Dataset

Signals come from the public [VitalDB](https://vitaldb.net/dataset/?query=overview) open dataset (6,388 surgical cases).

- Lee HC, Park Y, Yoon SB, Yang SM, Park D, Jung CW. VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients. *Scientific Data* 9, 279 (2022). https://doi.org/10.1038/s41597-022-01411-5
- PhysioNet release: https://physionet.org/content/vitaldb/1.0.0/ (Creative Commons Attribution 4.0)

The PhysioNet track catalog names the signals used here:

| Track | Description | Type | Unit |
| --- | --- | --- | --- |
| `BIS/EEG1_WAV` | EEG wave, channel 1 | waveform, 128 Hz | µV |
| `BIS/EEG2_WAV` | EEG wave, channel 2 | waveform, 128 Hz | µV |
| `BIS/BIS` | Bispectral index | numeric | unitless |

The BIS Vista device table in the VitalDB paper lists EEG at 1/128 s and the numeric BIS parameters at 1 s. The loader still measures those intervals from each downloaded file.

Web API endpoints used for discovery and download:

- https://api.vitaldb.net/cases
- https://api.vitaldb.net/trks
- https://api.vitaldb.net/{tid}

The official `vitaldb` package is used as an optional alignment cross-check against the packed case file. Its authors note that the older per-track CSV timestamps can be shifted relative to that file.

## Setup

Python 3.11 is the intended version. Commands below use the interpreter that created `.venv`.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python -m pip install numpy pandas PyYAML requests matplotlib vitaldb scipy scikit-learn joblib pytest
```

`requirements.txt` lists the dependencies for all six phases. Phase 1 needs the packages above except SciPy, scikit-learn, joblib, and pytest. Phase 2 needs SciPy as well. Phase 3 needs scikit-learn and joblib.

## Phase 1 commands

From this directory:

```powershell
.\.venv\Scripts\python -m src.vitaldb_loader
.\.venv\Scripts\python scripts\verify_phase1.py
```

The default configuration downloads 3 cases that contain both `BIS/EEG1_WAV` and `BIS/BIS`, after checking the live track index. Change `download.max_cases` in `config.yaml` to widen the subset. Do not point it at the full cohort on a laptop without planning for the transfer size.

Outputs:

- `data/metadata/trks.csv`, `cases.csv` — cached API tables. The HTTP client stores whatever the server sends after decoding; a genuine gzip body keeps the `.gz` suffix.
- `data/metadata/discovery.json` — track names and case counts
- `data/metadata/selected_cases.json` — accepted and excluded cases
- `data/raw/case_#####/` — gzip CSV tracks and NumPy archives
- `outputs/figures/case_#####_eeg_bis.png` — EEG excerpt, EEG RMS, and BIS timeline

## Phase 2 commands

Phase 2 reads `data/raw/case_#####/eeg_primary.npz` and `bis.npz`. It does not download recordings again.

```powershell
.\.venv\Scripts\python -m src.phase2
.\.venv\Scripts\python scripts\verify_phase2.py
```

The verification script runs the unit tests and then processes the cases listed in `phase2.case_ids` (1, 2, and 3).

What it does:

- Butterworth bandpass from 0.5–45 Hz, applied as second-order sections with zero-phase `sosfiltfilt`. Gaps longer than `preprocessing.max_interpolate_gap_sec` split the recording so the filter does not run across them.
- 10 s windows with 50% overlap (1,280 samples, hop 640, at 128 Hz).
- Welch power spectral density and band power for delta, theta, alpha, beta, and gamma, plus relative power, normalized spectral entropy, and the 95% spectral edge.
- Time-domain mean, standard deviation, RMS, variance, Hjorth mobility, and Hjorth complexity.
- Baseline BIS label: median of valid BIS samples with `window_start <= t < window_end`. Window timestamps and the BIS interval used for the label are stored so a processing delay can be tried later (`quality.bis_delay_sec`, default 0).
- Windows are rejected for excessive missing data, flatlines, severe amplitude artifacts, or an invalid BIS label. The reason is written to the rejection table.

Outputs:

- `data/processed/eeg_features.csv` — accepted windows and features
- `data/processed/window_rejections.csv` — rejected windows and reasons
- `data/processed/phase2_summary.json` — counts and missing-data statistics
- `data/processed/windows/case_#####.npz` — filtered windows for a later 1D CNN, with case id, timestamps, BIS label, and sampling rate
- `outputs/figures/phase2/` — raw versus filtered EEG, Welch spectrum, band-power distribution, BIS versus time, and BIS-label histogram

Accepted CNN windows are fixed length. Interior gaps that still pass the missing-data limit are linearly interpolated in that saved window; `n_missing` counts those samples. This phase does not train a model.

## Phase 3 commands

Phase 3 reads `data/processed/eeg_features.csv`. It does not download recordings or recompute the spectrum.

```powershell
.\.venv\Scripts\python -m src.train_ml
.\.venv\Scripts\python scripts\verify_phase3.py
.\.venv\Scripts\python -m pytest tests -q
```

Evaluation is leave-one-case-out: train on cases 2 and 3 and test on case 1, then rotate. Windows are not split at random. BIS, SQI, EMG, suppression ratio, timestamps, and case ids are not model inputs. SQI is used only in the data-quality report and in a separate sensitivity run. A model fit on all three cases is saved under `models/ml/all_cases/` and is not an unseen-case score.

Outputs:

- `outputs/reports/phase3/` — data-quality report, metrics, and feature importance
- `outputs/predictions/phase3/` — out-of-fold BIS predictions
- `outputs/figures/phase3/` — time series, scatter, error, importance, and Bland–Altman figures
- `models/ml/` — fold models and the all-case model

Three cases are not evidence of clinical generalization.

## Project layout

Phase 1 modules are `src/vitaldb_loader.py` and `scripts/verify_phase1.py`. Phase 2 modules are `src/preprocess.py`, `src/features.py`, `src/windows.py`, `src/phase2.py`, `scripts/verify_phase2.py`, and `tests/test_phase2.py`. Phase 3 modules are `src/train_ml.py`, `src/predict.py`, `src/ml_data.py`, `src/ml_metrics.py`, `src/ml_plots.py`, `scripts/verify_phase3.py`, and `tests/test_train_ml.py`. Phase 4 adds src/train_cnn.py, src/cnn.py, src/infer_cnn.py and src/evaluate.py. Phase 5 adds app.py and dashboard/components.py. Phase 6 adds tests/test_phase456.py.

## Phases 4–6 · CNN, dashboard, and verification

Run the commands from the project root in PowerShell. A 64-bit Python 3.11
environment is recommended. This workflow also runs with Python 3.10 and
compatible package builds. For CPU-only PyTorch, install the CPU wheel before
the remaining requirements.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -r requirements.txt
```

### Prepare the data

The bundled accepted windows and feature table are enough to train the CNN and
open the dashboard offline. Download the three original recordings to enable
raw waveform inspection and playback on the recording timeline. Run Phase 2
again when preparing from scratch or changing signal processing. Rebuild the
training artifacts after changing preprocessing, case membership, or labels.

```powershell
.\.venv\Scripts\python -m src.vitaldb_loader
.\.venv\Scripts\python -m src.phase2
```

The data are derived from VitalDB. Attribute the dataset and identify the
filtering, windowing and label aggregation if redistributing derivatives under
the CC BY 4.0 terms. Cite [the dataset](https://physionet.org/content/vitaldb/)
and [Lee et al., Scientific Data (2022)](https://doi.org/10.1038/s41597-022-01411-5).

### Train and evaluate

```powershell
.\.venv\Scripts\python -m src.train_cnn --device cpu
```

Omit `--device cpu` to select CUDA automatically when available.
`config.yaml` controls the seed, epoch budget, batch size and early stopping.
The trainer validates the Phase 2 waveform archives against the feature table
and prints training and validation loss after each epoch.

Each fold holds out one entire case for testing and reserves a second case for
validation. CNN weights and normalization come from the training cases. The
checkpoint with the lowest validation Smooth L1 loss is evaluated once on that
fold's test case. The median, random forest and histogram gradient boosting
models use the same training cases and test windows as the CNN in each fold.
Phase 3 used both non-test cases for training, so its scores answer a different
comparison.

- `models/cnn/test_case_*.pt` contains the best validation checkpoint and its
  normalization parameters, case assignments and signal configuration.
- `outputs/reports/phase4/` contains pooled and per-case scores, BIS-range
  results, loss history and out-of-fold predictions.
- `outputs/figures/phase4/` contains timelines, error plots, scatter plots,
  training curves and descriptive Bland–Altman limits.

### Run the demo and tests

```powershell
.\.venv\Scripts\python -m streamlit run app.py --global.developmentMode false --server.address 127.0.0.1 --server.port 8501 --server.headless true
.\.venv\Scripts\python -m pytest tests -q
```

Open the URL printed by Streamlit. Choose a case in the sidebar and click
**Run CNN inference** on either prediction page. For the three evaluated cases
the app uses the checkpoint that held out the selected case. For a new local
case it averages the fold checkpoints. Raw recordings can be scored without a
BIS track. With only the bundled windows, the app can show filtered EEG,
spectral plots, and replay across the accepted windows. The test suite checks
case separation, feature/window alignment, waveform handling with missing BIS,
checkpoint inference, and dashboard navigation and replay.

For command-line inference on a cached case, export a CSV with one row per usable EEG window.

```powershell
.\.venv\Scripts\python -m src.infer_cnn 1 --output outputs/predictions/case_00001_cnn.csv
```

### Research limitations

This demo estimates a recorded monitor index. Its research-only scope excludes
clinical use and real-time patient monitoring. The exploration cohort has three
surgical cases, with one training case per fold. Grouping is by VitalDB case ID.
Overlapping windows are correlated, BIS may lag EEG, and recorded targets can
include atypical zero values or low signal quality. Reported Bland–Altman limits
are descriptive because repeated windows are clustered within cases. The
zero-phase filtering and playback use pre-recorded data. Generalization would
require a larger cohort with independent patient-level validation.
