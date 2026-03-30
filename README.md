# SatelliteAnomalyDetection

An end-to-end telemetry analysis pipeline for the **KISPE Satellite Learning Laboratory (SLL)** CubeSat-class platform. The pipeline parses raw SCOTTI archive files, cleans and decodes the data, runs anomaly detectors, generates diagnostic plots, and evaluates synthetic fault injection against those detectors.

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
- [Paper](#paper)

---

## Overview

The SLL is a ground-based flatsat testbed that mimics a CubeSat's onboard data handling (CDH) and attitude determination and control system (ADCS). Experiments run for several minutes and produce binary archive files readable by KISPE's SCOTTI software, exported as `.txt` files in two formats: **Analysis** (high-precision timestamps, key=value fields) and **Debug** (positional Plot lines, 1-second precision).

This project builds a fully automated pipeline that processes those archives and characterises normal and anomalous telemetry behaviour across 10 experiment sessions.

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
- `key = value` pairs, sub-millisecond timestamps, date format `DD/MM/YYYY`
- Preferred when non-empty

**Debug files** (`*_Debug.txt`)
- Positional `Plot` lines: `MM/DD/YYYY HH:MM:SS (CDH|ADCS Sensor) Plot <f1> <f2> …`
- Used as fallback when the Analysis file is absent or empty
- Some sessions contain duplicate timestamps (deduplicated by keeping the last record)

Several fields are hex-encoded and decoded during cleaning:
- Unsigned uint16: current channels (scaled by 0.1 mA)
- Signed int16 (two's complement): `WHEEL_SPEED`, `MAGNETORQUER_*`, `BATT_CHR_I`

---

## Project Structure

```
SatelliteAnomalyDetection/
├── main.py                  # Pipeline entry point
├── scripts/
│   ├── parser.py            # SCOTTI Analysis + Debug file parsers
│   ├── cleaner.py           # Hex decoding, type coercion, stuck-sensor detection
│   ├── detector.py          # D1–D5 anomaly detectors
│   ├── plotter.py           # 19 diagnostic plot functions
│   └── injector.py          # 16-type fault injection + detection evaluation
├── data/
│   ├── Accelerometer_1/
│   ├── AttitudeDeterminationMagnetometer_1/
│   ├── AttitudeDetermination_Control/
│   ├── Baseline_1/
│   ├── CameraTest_1/
│   ├── Gyro_1/
│   ├── Magnetorquer_1/
│   ├── ReactionWheel_1/
│   ├── SunSensorTest_1/
│   └── ThermalTest_1/
├── results/
│   └── <experiment>/
│       ├── cdh_clean.csv
│       ├── adcs_clean.csv
│       ├── detection_report.json
│       ├── *.png               (11 general telemetry plots)
│       ├── anomaly_detection/  (8 detector-specific plots)
│       │   ├── 02_obdh_thermal_ramp.png
│       │   ├── 04_thermistor_anomaly.png
│       │   ├── 05_temperature_correlation.png
│       │   ├── 06_residual_rail_voltages.png
│       │   ├── 11_obdh_rolling_stats.png
│       │   ├── 12_packet_timing_jitter.png
│       │   ├── 13_temperature_rate_of_change.png
│       │   └── 14_voltage_anomaly_overlay.png
│       └── injected/
│           ├── cdh_injected_<type>_v<n>.csv
│           ├── adcs_injected_<type>_v<n>.csv
│           ├── labels_<type>_v<n>.csv
│           ├── cdh_eval_clean_v<n>.csv
│           ├── adcs_eval_clean_v<n>.csv
│           ├── injection_summary.csv
│           └── injection_eval.json
└── paper/
    ├── paper.tex
    ├── references.bib
    └── Makefile
```

---

## Pipeline

`main.py` loops over every subfolder in `data/` and runs five steps:

### 1. Parse
Reads the Analysis file if present; falls back to the Debug file if the Analysis file is absent or produces no CDH rows. Outputs raw CDH and ADCS DataFrames.

### 2. Clean
- Decodes hex-encoded uint16 and signed int16 fields
- Coerces all numeric columns to float, drops unparseable rows
- Detects stuck thermistor channels (zero variance, non-zero value)
- Deduplicates timestamps (Debug sessions)
- Exports `cdh_clean.csv` and `adcs_clean.csv`

### 3. Detect
Runs all five detectors and saves `detection_report.json`.

### 4. Plot
Generates 19 diagnostic plots covering temperatures, voltages, currents, IMU sensors, wheel speed, sun sensors, magnetorquers, correlation heatmaps, timing jitter, and more.

### 5. Inject + Evaluate
Generates 2 variants per fault type (32 labelled datasets), then evaluates each variant with the full detector suite and saves accuracy results.

---

## Anomaly Detection

Five detectors are defined in [scripts/detector.py](scripts/detector.py):

| ID | Name | Method | Channels |
|---|---|---|---|
| D1 | Thermal trend | Linear regression (scipy.linregress) | TEMP_OBDH |
| D2 | Z-score | \|z\| > 2.5σ | 7 CDH power/temp channels |
| D3 | Correlation breakdown | 30-sample rolling Pearson \|r\| < 0.5 | 4 board-temp pairs + wheel↔gyro |
| D4 | ADCS stuck / all-zero | Zero variance + non-zero value, or all-IMU-zero | All ADCS numeric columns |
| D5 | Packet timing | max inter-packet gap vs baseline | CDH timestamps |

Key findings from baseline characterisation:
- **Universal OBDH thermal ramp**: all 10 sessions show 0.80–2.04 °C/min with R² > 0.88 — a systemic enclosure heating effect
- **21 stuck thermistors**: rod array at 77.18 °C, bath at 74.31 °C — confirmed hardware freeze
- **ADCS all-zero in most sessions**: ADCS powered off during non-ADCS experiments; active only in `AttitudeDetermination_*`, `Gyro_1`, `Magnetorquer_1`, `ReactionWheel_1`
- **Voltage rail noise**: SYS_V, PLAT_5V_V, OBC_PWR_V show 17–19 z-score flags per session due to switching transients

---

## Fault Injection

16 synthetic fault types defined in [scripts/injector.py](scripts/injector.py):

| Class | ID | Description | Target |
|---|---|---|---|
| Thermal | T1 | OBDH ramp acceleration (4×) | CDH |
| Thermal | T2 | Board temperature step + exponential decay | CDH |
| Thermal | T3 | Growing linear bias between two correlated sensors | CDH |
| Power | P1 | Gaussian SYS_V sag | CDH |
| Power | P2 | OBC_PWR_V random spikes | CDH |
| Power | P3 | Rail forced to stuck high value | CDH |
| ADCS | A1 | Accelerometer dropout (all axes zeroed) | ADCS |
| ADCS | A2 | Gyro axis saturation at observed max | ADCS |
| ADCS | A3 | Magnetometer sign flip | ADCS |
| ADCS | A4 | Full IMU blackout (all 9 axes zeroed) | ADCS |
| ADCS | A5 | Sun sensor saturation at 1023 | ADCS |
| Wheel | W1 | Wheel overspeed ramp to 1800 RPM | ADCS |
| Wheel | W2 | Sudden wheel stop at peak speed | ADCS |
| Timing | C1 | Packet gap (~10 s block removed) | CDH |
| Compound | CP1 | T1 + P1 simultaneously | CDH |
| Compound | CP2 | A4 (IMU blackout) + P3 (rail stuck) | CDH + ADCS |

Each injection picks a random window within the session (seeded for reproducibility) and produces a label record containing `anomaly_type`, `start_idx`, `end_idx`, `severity`, and `affected_channels`.

---

## Detection Evaluation

After injection, each variant is reloaded and all five detectors re-run against it. Detection is defined as any detector flagging at least one sample inside the `[start_idx, end_idx]` window:

- **D1**: injected OBDH slope > 1.5× baseline or > 3.0 °C/min absolute
- **D2**: any z-score flag in the anomaly window
- **D3**: rolling correlation drops below threshold inside the window
- **D4**: stuck-channel list non-empty or ADCS all-zero True
- **D5**: max CDH gap > 2× baseline or > 5 s absolute

Results are saved per-experiment to:
- `injected/injection_eval.json` — per-variant JSON with per-detector flags
- `injected/injection_summary.csv` — per-type detection counts and rates

**Overall detection rate: 97–100% across all 10 experiments** (the single miss is a slow-growing T3 divergence in the short Accelerometer_1 session that stays within the correlation window threshold).

---

## Experiments

| Session | CDH Packets | ADCS Active | Notes |
|---|---|---|---|
| Baseline_1 | ~360 | No | Thermal characterisation baseline |
| Accelerometer_1 | ~360 | Yes | IMU accel validation |
| AttitudeDeterminationMagnetometer_1 | ~360 | Yes | Magnetometer AD test |
| AttitudeDetermination_Control | ~360 | Yes | Full ADCS closed-loop |
| CameraTest_1 | ~360 | No | Camera payload active |
| Gyro_1 | ~360 | Yes | Gyroscope characterisation |
| Magnetorquer_1 | ~360 | Yes | Magnetorquer actuation |
| ReactionWheel_1 | ~360 | Yes | Reaction wheel spin-up |
| SunSensorTest_1 | ~360 | No | Sun sensor calibration |
| ThermalTest_1 | ~360 | No | Extended thermal soak |

---

## Results

All results are written to `results/<experiment>/`. Each experiment produces:

- `cdh_clean.csv` / `adcs_clean.csv` — cleaned telemetry
- `detection_report.json` — structured detection output (trends, z-scores, stuck channels, timing)
- 19 `.png` plots
- `injected/` — 32 labelled anomalous variants + evaluation JSON

---

## Running the Pipeline

**Requirements**: Python 3.11+, numpy, pandas, scipy, matplotlib, seaborn

```bash
# Create and activate virtual environment (if not already done)
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy matplotlib seaborn

# Run the full pipeline
python main.py
```

The pipeline auto-discovers all experiment subfolders under `data/`, processes them in alphabetical order, and writes results to `results/`.
