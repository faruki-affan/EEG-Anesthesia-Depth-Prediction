# Estimating recorded BIS from surgical EEG

This project asks whether a short recording of brain activity can estimate the **Bispectral Index (BIS)** reported by a monitor during surgery. It compares traditional machine learning with a convolutional neural network (CNN), then tests whether giving the CNN recent EEG history improves its estimates on previously unseen surgical cases.

**In the expanded study, adding up to 30 seconds of EEG history reduced average error per case from 11.34 to 8.94 BIS points.** The comparison used the same three held-out surgical cases for every model. This is an offline research demonstration, and the results support further evaluation on larger, independent cohorts. [Expanded study results](outputs/reports/temporal/metrics.json)

## What EEG and BIS tell us

**Electroencephalography (EEG)** measures small changes in voltage at the scalp. Anesthetic drugs can change the timing and strength of rhythms in that signal. **BIS** is a monitor-derived summary score on a 0–100 scale. Because both can be recorded during the same operation, they give us a waveform to study and a numerical reference to compare against. The project estimates the monitor’s recorded BIS for a corresponding EEG interval. BIS is a proxy for aspects of anesthetic state, and agreement with it alone cannot establish consciousness, clinical depth of anesthesia, or readiness for clinical use. [EEG signatures during propofol anesthesia](https://pmc.ncbi.nlm.nih.gov/articles/PMC3607036/)

## Data and study design

[VitalDB](https://physionet.org/content/vitaldb/1.0.0/) is a public, de-identified collection of 6,388 surgical recordings from Seoul National University Hospital. A **case** is one surgical recording, and a **track** is one signal within it. The loader looks for cases containing primary EEG (`BIS/EEG1_WAV`, nominally 128 Hz) and recorded BIS (`BIS/BIS`, roughly once per second). BIS supplies the reference label. Auxiliary tracks, such as signal quality and muscle activity, can be inspected separately.

| Analysis | Case allocation | Purpose |
| --- | --- | --- |
| Phase 3 · Feature-based pilot | Cases 1–3, with two training cases and one test case per fold | Establish simple and tree-based regression baselines |
| Phase 4 · Waveform pilot | Cases 1–3, rotating distinct training, validation, and test roles | Compare the single-window CNN with traditional models under a matched split |
| Expanded temporal study | Train on 2, 5, 6, 8, 10. Validate on 3. Test on 1, 4, 9. | Measure the benefit of recent EEG history under the same held-out split |

We split by surgical case because neighboring windows share half their EEG samples. A random window split could place overlapping signal in both training and test sets. The expanded dataset contains **19,960 accepted windows**, including **6,735 held-out test windows**. The test evidence comes from **three independent surgical recordings** in that experiment. [Split code](src/train_temporal.py) · [Test predictions](outputs/reports/temporal/test_predictions.csv)

## Methods

1. **Acquire and verify the recordings.** The loader discovers paired EEG and BIS tracks, measures their sampling and time coverage, checks recording quality, and caches the files. EEG and BIS are paired using timestamps because they are recorded at different rates. [Loader](src/vitaldb_loader.py)
2. **Filter, segment, and label the EEG.** A 0.5–45 Hz Butterworth bandpass attenuates slow baseline drift and higher-frequency components. The pipeline creates 10-second EEG windows with a five-second hop and excludes windows with poor signal or insufficient valid BIS. It labels each accepted window with the median recorded BIS inside the same interval. The current alignment uses zero additional BIS delay. [Phase 2 pipeline](src/phase2.py) · [Signal and window settings](config.yaml)
3. **Build a feature-based baseline.** Welch’s method averages FFT-based power estimates over overlapping, Hann-tapered segments. Absolute and relative band powers, spectral entropy, spectral edge frequency, and time-domain measures give 18 EEG features. A training-set median provides a simple reference. Random Forest and Histogram Gradient Boosting help test whether nonlinear relationships among the features improve estimation. [Feature extraction](src/features.py) · [Traditional training](src/train_ml.py)
4. **Learn directly from the waveform.** The 1D CNN receives a filtered 1,280-sample window and learns a representation of its waveform. In the expanded experiment, the Temporal CNN reuses the frozen waveform encoder and learns a correction from successive window representations. Validation selected 30 seconds of context over 60 seconds. History resets across gaps. [Waveform CNN](src/cnn.py) · [Temporal model](src/temporal.py)

## Results

All models below were evaluated on the **same expanded-study test windows**. Case-averaged mean absolute error (MAE) gives each surgical case equal weight. Pooled MAE gives each window equal weight. Root mean squared error (RMSE) gives large errors more influence.

| Model | Case-averaged MAE ↓ | Pooled MAE ↓ | RMSE ↓ | R² ↑ |
| --- | ---: | ---: | ---: | ---: |
| Single-window CNN | 11.34 | 9.93 | 12.76 | 0.41 |
| **Temporal CNN** | **8.94** | **8.62** | **11.15** | **0.55** |
| Random Forest | 11.67 | 11.80 | 14.91 | 0.19 |

The Temporal CNN reduced case-averaged MAE by about **21%** versus the Single-window CNN in this experiment. Error varied across the BIS range. Its pooled MAE was **5.01** in the 40–60 band and **29.06** in the 0–20 band, which had only **55 test windows**. [Metrics by BIS range](outputs/reports/temporal/metrics_by_bis_range.csv)

Estimation error and transition tracking answer different questions. The Temporal CNN matched **11 of 64** recorded crossings of BIS 60 using same-direction, one-to-one matching within ±30 seconds. Zero-phase filtering uses pre-recorded context, and the current BIS alignment does not account for monitor lag. The small, single-source study and uneven BIS coverage limit generalization. [Transition results](outputs/reports/temporal/transitions.csv)

## Explore the dashboard

The React dashboard uses a shared playhead to explore the EEG waveform, recorded BIS, and model estimates. It displays band power and spectral entropy for the selected interval, distinguishes cases used during model development from official test cases, and exports available case data as a ZIP. Traditional regressors appear in the performance reports. The inference selector runs the two CNNs.

Use **Python 3.11** and **Node.js 22.12 or newer**. From the project root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm --prefix web ci
npm --prefix web run build
.\.venv\Scripts\python.exe -m uvicorn dashboard.api:app --host 127.0.0.1 --port 8511
```

Open [EEG Research Monitor](http://127.0.0.1:8511/) and try **Case 00001** with the **Temporal CNN**. The repository includes processed windows for Cases 1–3, model checkpoints, and saved reports. The loader supplies raw waveforms for cases downloaded locally.

## Reproduce the experiments

Each command runs a distinct stage. Run the expanded experiment with `--prepare` the first time, then omit that flag to reuse the cohort cache.

```powershell
.\.venv\Scripts\python.exe -m src.vitaldb_loader
.\.venv\Scripts\python.exe -m src.phase2
.\.venv\Scripts\python.exe -m src.train_ml
.\.venv\Scripts\python.exe -m src.train_cnn --device cpu
.\.venv\Scripts\python.exe -m src.train_temporal --prepare --device cpu
.\.venv\Scripts\python.exe -m pytest tests -q
```

| Location | What to look for |
| --- | --- |
| [`config.yaml`](config.yaml) | Track names, filtering, windowing, quality limits, and experiment settings |
| [`src/`](src/) | Acquisition, preprocessing, training, and inference |
| [`dashboard/api.py`](dashboard/api.py) and [`web/`](web/) | FastAPI routes and the React dashboard |
| [`data/processed/`](data/processed/) | Pilot feature table and filtered waveform archives |
| [`models/`](models/) and [`outputs/reports/`](outputs/reports/) | Checkpoints, case-level results, and evaluation reports |
| [`tests/`](tests/) | Checks for signal processing, case separation, reproducible predictions, and dataset exports |

## Data attribution

The bundled derived data were filtered, windowed, and paired with the median recorded BIS. Cite **Lee et al., “VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients,” Scientific Data (2022)** and follow the dataset license. [Dataset paper](https://doi.org/10.1038/s41597-022-01411-5) · [VitalDB license](https://physionet.org/content/vitaldb/1.0.0/#license)
