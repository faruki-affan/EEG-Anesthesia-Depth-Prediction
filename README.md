# EEG-Based Depth of Anesthesia Estimation

Educational project that predicts the recorded Bispectral Index (BIS) from intraoperative EEG. The current tree contains **Phase 1** (VitalDB discovery, a small real-data download, and a waveform plot), **Phase 2** (bandpass filtering, windowing, Welch spectral features, and BIS alignment), and **Phase 3** (leave-one-case-out regression from those features). A 1D CNN and the Streamlit dashboard are not implemented yet.

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

`requirements.txt` also lists the libraries planned for later phases. Phase 1 needs the packages above except SciPy, scikit-learn, joblib, and pytest. Phase 2 needs SciPy as well. Phase 3 needs scikit-learn and joblib.

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

Phase 1 modules are `src/vitaldb_loader.py` and `scripts/verify_phase1.py`. Phase 2 modules are `src/preprocess.py`, `src/features.py`, `src/windows.py`, `src/phase2.py`, `scripts/verify_phase2.py`, and `tests/test_phase2.py`. Phase 3 modules are `src/train_ml.py`, `src/predict.py`, `src/ml_data.py`, `src/ml_metrics.py`, `src/ml_plots.py`, `scripts/verify_phase3.py`, and `tests/test_train_ml.py`. The 1D CNN and the Streamlit dashboard are later phases and are not in this tree yet.
