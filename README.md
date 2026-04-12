# SatelliteAnomalyDetection

An end-to-end telemetry analysis pipeline for the **KISPE Satellite Learning Laboratory (SLL)** CubeSat-class platform. The pipeline parses raw SCOTTI archive files, cleans and decodes the data, runs rule-based anomaly detectors, generates diagnostic plots, injects synthetic faults, and evaluates detection performance with both rule-based and ML-based methods.

---

## Table of Contents

- [Overview](#overview)
- [Platform](#platform)
- [Data Format](#data-format)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Anomaly Detection](#anomaly-detection)
- [Fault Injection](#fault-injection)
- [Detection Evaluation](#detection-evaluation)
- [Experiments](#experiments)
- [Results](#results)
- [Running the Pipeline](#running-the-pipeline)

---

## Overview

The SLL is a ground-based flatsat testbed that mimics a CubeSat's onboard data handling (CDH) and attitude determination and control system (ADCS). Experiments run for several minutes and produce binary archive files readable by KISPE's SCOTTI software, exported as `.txt` files in two formats: **Analysis** (high-precision timestamps, key=value fields) and **Debug** (positional Plot lines, 1-second precision).

This project builds a fully automated pipeline that processes those archives, characterises normal and anomalous telemetry behaviour across 18 experiment sessions, and compares four ML-based detectors against rule-based baselines.

---

## Platform

| Subsystem | Description |
|---|---|
| CDH | 87-field telemetry packet — temperatures, voltages, currents, timing |
| ADCS Sensor | 30-field packet — IMU (accel/gyro/mag), wheel speed, sun sensors, magnetorquers |
| Thermistors | 21-channel rod/bath array (hardware stuck at ~77 °C / ~74 °C) |
| Power rails | SYS_V, PLAT_5V_V, OBC_PWR_V, COMMS_PWR_V, ADCS_5V_V, PAYLOAD_5V_V |
| Reaction wheel | Single-axis, monitored via WHEEL_SPEED and GYRO_Z correlation |

---

## Data Format

Two archive formats are supported with automatic fallback:

**Analysis files** (`*_Analysis.txt`)
- `key = value` pairs, sub-millisecond timestamps
- Supports both `DD/MM/YYYY` and `MM/DD/YYYY` date formats (auto-detected)
- Preferred when non-empty

**Debug files** (`*_Debug.txt`)
- Positional `Plot` lines: `MM/DD/YYYY HH:MM:SS (CDH|ADCS Sensor) Plot <f1> <f2> …`
- Used as fallback when the Analysis file is absent or empty

Several fields are hex-encoded and decoded during cleaning:
- Unsigned uint16: current channels (SYS_I, OBC_PWR_I, COMMS_PWR_I, etc.)
- Signed int16 (two's complement): `WHEEL_SPEED`, `MAGNETORQUER_*`, `BATT_CHR_I`

---

## Project Structure

```
SatelliteAnomalyDetection/
├── main.py                  # Pipeline entry point
├── scripts/
│   ├── parser.py            # SCOTTI Analysis + Debug file parsers
│   ├── cleaner.py           # Hex decoding, type coercion, stuck-sensor detection
│   ├── injector.py          # 16-type fault injection + ML evaluation orchestration
│   ├── evaluator.py         # ML detectors (ZScore, RobustRollingZScore, IsolationForest, Autoencoder)
│   └── plotter.py           # Diagnostic and ML evaluation plot functions
├── data/
│   ├── Baseline_1/
│   ├── AccelerometerTest/
│   │   ├── AccelerometerTest_1/
│   │   └── AccelerometerTest_2-6/
│   ├── GyroTest/
│   │   └── GyroTest_1-6/
│   └── ThermalTest/
│       └── ThermalTest_2-6/   (ThermalTest_1 is an empty stub — skipped)
├── results/
│   └── <experiment>/
│       ├── cdh_clean.csv
│       ├── adcs_clean.csv
│       ├── *.png                  (general telemetry plots)
│       ├── anomaly_detection/     (timing, ramp, correlation plots)
│       └── injected/
│           ├── cdh_injected_<type>_<tier>_v<n>.csv
│           ├── adcs_injected_<type>_<tier>_v<n>.csv
│           ├── labels_<type>_<tier>_v<n>.csv
│           ├── injection_summary.csv
│           ├── ml_evaluation.json
│           ├── E2_ml_metrics_bars.png
│           ├── E3_confusion_matrices.png
│           └── E5_latency_distribution.png
├── results/aggregated/
│   ├── AGG_ml_metrics_by_family.png   (per-family mean±std grouped bars)
│   └── AGG_if_feature_importances.png (IF feature importance heatmap by injection type)
└── paper/
    ├── paper.tex
    ├── references.bib
    └── Makefile
```

---

## Pipeline

`main.py` loops over every qualifying experiment subfolder in `data/` and runs four steps:

### 1. Parse
Reads the Analysis file if present; falls back to the Debug file if the Analysis file is absent or produces no CDH rows. Handles both `DD/MM/YYYY` and `MM/DD/YYYY` date formats automatically.

### 2. Clean
- Decodes hex-encoded uint16 and signed int16 fields
- Coerces all numeric columns to float, drops unparseable rows
- Deduplicates timestamps
- Detects stuck thermistor channels (zero variance, non-zero value)
- Derives elapsed time and estimated per-rail power
- Exports `cdh_clean.csv` and `adcs_clean.csv`

### 3. Plot
Generates up to 20 diagnostic plots: board temperatures, power rails, current draw, timing jitter, IMU sensors, wheel speed, sun sensors, correlation heatmaps, and more. ADCS-specific plots (16–20) are only generated when the relevant sensors have active data.

### 4. Inject + ML Evaluate
Generates 2 variants per fault type (tiered `easy/medium/hard`) and evaluates all four ML detectors against them. Saves `ml_evaluation.json` and three per-experiment evaluation plots (E2 metrics bar chart, E3 confusion matrices, E5 latency distribution). After all experiments, writes aggregated cross-family comparison plots to `results/aggregated/`.

---

## Anomaly Detection

Four ML detectors defined in [scripts/evaluator.py](scripts/evaluator.py):

| Detector | Method | Architecture |
|---|---|---|
| ZScore | Hybrid z-score + modified z-score (MAD-based robust variant) | Parametric robust statistics |
| RobustRollingZScore | Rolling median/MAD detector with persistence and CUSUM-style drift sensitivity | Robust streaming detector |
| IsolationForest | Rotation-ensemble Isolation Forest (EIF-inspired) with PCA + feature subsampling | Ensemble of Isolation Forests |
| Autoencoder | Multi-scale denoising sequence autoencoder ensemble | MLP reconstruction on overlapping temporal windows |

### Literature-Based Method Updates (April 2026)

The evaluator was updated to align each detector with modern anomaly-detection literature.

`ZScore` / robust statistical baseline updates:
- Added modified z-score (median + MAD) fusion with classic z-score to improve heavy-tail robustness.
- Citations:
    - Iglewicz, B., Hoaglin, D. C. *How to Detect and Handle Outliers*, 1993.
    - Rousseeuw, P. J., Croux, C. *Alternatives to the Median Absolute Deviation*, JASA 1993. DOI: 10.1080/01621459.1993.10476408
    - NIST/SEMATECH e-Handbook: modified z-score and outlier detection guidance.

`RobustRollingZScore` updates:
- Kept rolling median/MAD and persistence rules, and added bounded CUSUM-style persistence boosting for small sustained shifts.
- Citations:
    - Hampel, F. R. *The Influence Curve and Its Role in Robust Estimation*, 1974.
    - Page, E. S. *Continuous Inspection Schemes* (CUSUM), Biometrika 1954. DOI: 10.1093/biomet/41.1-2.100
    - Iglewicz & Hoaglin (1993), robust modified z-score thresholds.

`IsolationForest` updates:
- Implemented a random-rotation ensemble over reduced feature space to reduce axis-alignment artifacts (EIF-inspired).
- Citations:
    - Liu, F. T., Ting, K. M., Zhou, Z.-H. *Isolation Forest*, ICDM 2008. DOI: 10.1109/ICDM.2008.17
    - Hariri, S., Carrasco Kind, M., Brunner, R. J. *Extended Isolation Forest*, TKDE 2021. DOI: 10.1109/TKDE.2019.2947676

`Autoencoder` updates:
- Switched to multi-scale temporal window ensembling with denoising training noise to improve robustness across anomaly durations.
- Citations:
    - Malhotra, P. et al. *LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection*, 2016. arXiv:1607.00148
    - Hundman, K. et al. *Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding*, KDD 2018. DOI: 10.1145/3219819.3219845
    - Zhang, C. et al. *MSCRED: Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series*, 2018. arXiv:1811.08055

All detectors are trained on the first 60% of the clean session, with the score threshold set at the 99th percentile of training scores. Features include raw telemetry channels, first-order differences, and rolling mean/std (when session ≥ 45 rows).

Key findings from baseline characterisation:
- **Universal OBDH thermal ramp**: all 18 sessions show 0.7–2.0 °C/min with R² > 0.79 — a systemic enclosure heating effect
- **21 stuck thermistors**: rod array at ~77.18 °C, bath at ~74.31 °C — confirmed hardware freeze
- **ADCS all-zero in non-ADCS experiments**: ADCS powered off during Baseline and ThermalTest sessions; active only in AccelerometerTest and GyroTest
- **Voltage rail noise**: SYS_V, PLAT_5V_V, OBC_PWR_V show regular switching transients

---

## Fault Injection

16 synthetic fault types (+2 compound) defined in [scripts/injector.py](scripts/injector.py):

| Class | ID | Description | Target |
|---|---|---|---|
| Thermal | T1_obdh_runaway | Runaway nonlinear thermal ramp (contextual/collective) | CDH |
| Thermal | T2_board_thermal_shock | Step-hold-decay thermal transient | CDH |
| Thermal | T3_thermal_bias_divergence | Bias divergence between correlated thermal channels | CDH |
| Power | P1_bus_sag_recovery | Bus sag with coupled current response and recovery | CDH |
| Power | P2_switching_noise_burst | High-frequency ripple + impulsive switching noise | CDH |
| Power | P3_rail_latchup | Rail latch-up / stuck-high regime | CDH |
| ADCS | A1_accel_packet_dropout | Intermittent accelerometer sample dropout | ADCS |
| ADCS | A2_gyro_clipping | Axis clipping/saturation with contextual bias | ADCS |
| ADCS | A3_mag_field_inversion | Magnetometer polarity inversion + offset | ADCS |
| ADCS | A4_imu_correlation_break | Cross-sensor temporal/correlation consistency break | ADCS |
| ADCS | A5_sun_sensor_blinding | Partial/full sun-sensor blinding and saturation | ADCS |
| Wheel | W1_wheel_runaway | Wheel runaway ramp with oscillatory instability | ADCS |
| Wheel | W2_wheel_stiction_stop | Abrupt deceleration, stiction hold, degraded recovery | ADCS |
| Timing | C1_packet_gap_jitter | Packet removal plus boundary jitter in elapsed time | CDH |
| Comms | C2_rssi_fade | Sustained RSSI fading random walk | CDH |
| Comms | C3_frame_error_avalanche | Accelerating frame-error burst process | CDH |
| Compound | COMPOUND_thermal_power_fault | T1 + P1 simultaneously | CDH |
| Compound | COMPOUND_adcs_power_fault | A4 + P3 simultaneously | CDH + ADCS |

Each injection picks a random contextual window (seeded for reproducibility), supports `easy/medium/hard` tiers, and produces a label record containing `anomaly_type`, `start_idx`, `end_idx`, `severity`, and `affected_channels`.

The redesign is literature-driven. See `documentation/Tutorials/LiteratureDrivenInjection.md` for citations and mapping from paper guidance to implemented anomaly morphology.

---

## Detection Evaluation

ML detectors are trained on the first 60% of the clean session and scored against the full injected dataset. The anomaly threshold is set at the 99th percentile of training-set scores (calibrated on clean data only). Performance is measured at this threshold (Accuracy, FPR, Recall, F1) plus AUC-ROC.

---

## Experiments

Configured window: runs 2-6 (5 runs per family) across four experiment families:

| Family | Sessions | CDH packets | ADCS active |
|---|---|---|---|
| AccelerometerTest | 5 (runs 2-6) | 300–390 | Yes |
| GyroTest | 5 (runs 2-6) | 180–370 | Yes |
| ReactionWheelTest | 5 (runs 2-6) | 200–330 | Yes |
| ThermalTest | 5 (runs 2–6) | 3000–5000 | No |

Baseline_2-6 are not present in the current dataset, so Baseline is excluded when
enforcing the 2-6 run window.

---

## Results

### ML Detector Performance (latest full rerun, tiered injections; four detector families)

| Detector | Accuracy | FPR | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| ZScore | 0.733 | 0.057 | 0.109 | 0.105 | 0.585 |
| RobustRollingZScore | 0.629 | 0.254 | 0.282 | 0.140 | 0.533 |
| IsolationForest | 0.734 | 0.125 | 0.225 | 0.190 | 0.571 |
| **Autoencoder** | **0.743** | 0.183 | **0.485** | **0.356** | **0.683** |

Tier-level mean accuracy across detectors in the latest rerun:

| Tier | Mean Accuracy |
|---|---|
| Easy | 0.795 |
| Medium | 0.704 |
| Hard | 0.630 |

Family-level easy-tier mean accuracy:

| Family | Easy Accuracy |
|---|---|
| AccelerometerTest | 0.855 |
| GyroTest | 0.819 |
| ReactionWheelTest | 0.860 |
| ThermalTest | 0.648 |

Autoencoder currently has the strongest overall recall/F1, while ZScore and IsolationForest provide stronger easy-tier accuracy in the non-thermal families. Thermal remains the hardest family due to long sessions and subtle drift-like faults.

All results are written to `results/<experiment>/`. Each experiment produces `cdh_clean.csv`, `adcs_clean.csv`, up to 20 `.png` plots, and an `injected/` directory with 32 labelled variants, `ml_evaluation.json`, and three evaluation plots (E2 metrics bar chart, E3 confusion matrices, E5 latency distribution). Cross-family aggregated plots are written to `results/aggregated/`.

---

## Running the Pipeline

**Requirements**: Python 3.11+, numpy, pandas, scipy, matplotlib, seaborn, scikit-learn

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy matplotlib seaborn scikit-learn

# Run the full pipeline
python main.py
```

The pipeline auto-discovers all qualifying experiment subfolders under `data/`, enforces the 2-6 run window, aggregates by family, and writes all results to `results/`.
