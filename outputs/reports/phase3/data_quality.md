# Phase 3 data-quality report

- Accepted windows: 5712
- Cases: 3 ({1: 2091, 2: 2910, 3: 711})
- Model features: 18
- Rows excluded: 0

## Feature columns used as model inputs

- delta_power (frequency)
- theta_power (frequency)
- alpha_power (frequency)
- beta_power (frequency)
- gamma_power (frequency)
- delta_relative (frequency)
- theta_relative (frequency)
- alpha_relative (frequency)
- beta_relative (frequency)
- gamma_relative (frequency)
- spectral_entropy (frequency)
- spectral_edge_hz (frequency)
- mean (time)
- std (time)
- rms (time)
- variance (time)
- hjorth_mobility (time)
- hjorth_complexity (time)

## Requested features that are not in the CSV

- `total_power`: Total spectral power is not a column. The five absolute band powers already partition the in-band spectrum, so no summed column is added.
- `skewness`: Skewness was not extracted in Phase 2.
- `kurtosis`: Kurtosis was not extracted in Phase 2.
- `hjorth_activity`: Hjorth activity is the signal variance. That value is already stored as variance, and it is not duplicated under another name.

## Columns present but not used as inputs

BIS, SQI, suppression ratio, EMG, timestamps, and case identifiers stay out of the feature matrix.

- BIS_target
- bis_delay_sec
- bis_first_sec
- bis_last_sec
- bis_n
- case_id
- missing_fraction
- n_missing
- n_samples
- peak_abs_uv
- sampling_rate_hz
- sqi_median
- window_end_sec
- window_index
- window_start_sec

## BIS by case

- Case 1: n=2091, median=48.70, mean=47.11, min=0.00, max=97.70, BIS=0 windows=9
  - Histogram: {'0–20': 9, '20–40': 522, '40–60': 1386, '60–80': 136, '80–100': 38}
  - SQI histogram: {'0–20': 21, '20–50': 117, '50–80': 646, '80–100': 1307}
- Case 2: n=2910, median=47.20, mean=50.00, min=0.00, max=97.70, BIS=0 windows=4
  - Histogram: {'0–20': 4, '20–40': 49, '40–60': 2551, '60–80': 260, '80–100': 46}
  - SQI histogram: {'0–20': 6, '20–50': 64, '50–80': 474, '80–100': 2366}
- Case 3: n=711, median=35.85, mean=43.61, min=0.00, max=97.70, BIS=0 windows=1
  - Histogram: {'0–20': 1, '20–40': 470, '40–60': 140, '60–80': 24, '80–100': 76}
  - SQI histogram: {'0–20': 2, '20–50': 40, '50–80': 145, '80–100': 524}

## BIS = 0 windows

These windows are not removed. A low-SQI reading of 0 at the start or end of a case is treated as an unreliable monitor value. Genuine suppression would need low EEG amplitude, a high suppression ratio, and SQI high enough to trust the monitor.

| Case | Start (s) | SQI | EEG SD (µV) | SEF95 (Hz) | SR median | Classification |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 90.0 | 0.00 | 0.78 | 34.0 |  | monitor_startup_unreliable |
| 1 | 110.0 | 0.40 | 30.06 | 17.0 |  | monitor_startup_unreliable |
| 1 | 115.0 | 8.35 | 27.35 | 11.0 |  | monitor_startup_unreliable |
| 1 | 120.0 | 9.20 | 19.90 | 29.5 |  | monitor_startup_unreliable |
| 1 | 125.0 | 10.00 | 26.23 | 28.0 |  | monitor_startup_unreliable |
| 1 | 130.0 | 10.80 | 32.43 | 23.5 |  | monitor_startup_unreliable |
| 1 | 135.0 | 11.70 | 29.75 | 25.5 |  | monitor_startup_unreliable |
| 1 | 140.0 | 11.70 | 31.75 | 23.5 |  | monitor_startup_unreliable |
| 1 | 145.0 | 13.30 | 30.24 | 21.0 | 0.0 | monitor_startup_unreliable |
| 2 | 50.0 | 3.35 | 91.62 | 5.5 |  | monitor_startup_unreliable |
| 2 | 55.0 | 7.90 | 109.82 | 5.0 |  | monitor_startup_unreliable |
| 2 | 60.0 | 9.20 | 121.83 | 4.5 |  | monitor_startup_unreliable |
| 2 | 15135.0 | 0.00 | 1.68 | 19.5 |  | end_of_recording_or_dropout_unreliable |
| 3 | 3720.0 | 0.00 | 1.29 | 29.0 |  | end_of_recording_or_dropout_unreliable |

Counts: {'monitor_startup_unreliable': 12, 'end_of_recording_or_dropout_unreliable': 2}

## Timestamps

Accepted windows use a 5-second hop. A longer hop means Phase 2 rejected one or more intervening windows. Those gaps are not duplicate or reordered timestamps.
- Case 1: 22 hops longer than 5 s (max hop 35.0 s).
- Case 2: 31 hops longer than 5 s (max hop 35.0 s).
- Case 3: 9 hops longer than 5 s (max hop 35.0 s).

## Critical failures

None.
