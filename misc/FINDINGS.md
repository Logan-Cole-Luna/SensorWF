# Satellite Learning Laboratory — Telemetry Analysis Findings

**Platform:** KISPE Satellite Learning Laboratory (SLL)
**Archive:** `Archive_2026_03_02_112454`
**Session:** 03 February 2026 | 11:24:54 → 11:27:41 (~2 min 47 s)
**Analyst:** Automated Pipeline (`process_telemetry.py`)
**Report generated:** 2026-03-09

---

## 1. Dataset Overview

| Item | Detail |
|------|--------|
| Source file | `Data/Archive_2026_03_02_112454_Analysis.txt` (42 363 lines) |
| Debug file | `Data/Archive_2026_03_02_112454_Debug.txt` (1 338 lines) |
| CDH packets parsed | **333** |
| ADCS Sensor packets parsed | **335** |
| Approximate sample rate | ~2 Hz per stream |
| Subsystems active | Platform (OBC, EPS, COMMS) |
| Subsystems off | ADCS, Payload |
| Power source | Battery only (indoor ambient light only — no solar charging) |

---

## 2. Data Cleaning

The following cleaning steps were applied before analysis:

- **Timestamp filtering** — rows with missing or unparseable timestamps removed (0 dropped).
- **Duplicate timestamps** — de-duplicated keeping last record per timestamp (0 dropped).
- **Type coercion** — all telemetry fields converted to `float64`; hex-only strings retained as-is.
- **Elapsed time column** — seconds from session start added for convenience (`elapsed_s`).
- **Stuck sensor flagging** — channels with zero variance throughout the session were tagged automatically.

Final clean dataset: **333 CDH rows × 88 columns**, **335 ADCS rows × 31 columns**.
Cleaned CSVs exported to `Output/cdh_clean.csv` and `Output/adcs_clean.csv`.

---

## 3. Anomalies Detected

### 3.1 OBDH Thermal Ramp ⚠️ HIGH PRIORITY

The On-Board Data Handling board (STM32 OBC) exhibited a **sustained monotonic temperature rise** throughout the entire session.

| Metric | Value |
|--------|-------|
| Start temperature | 23.191 °C |
| End temperature | 27.675 °C |
| Total delta | **+4.484 °C** |
| Rate of rise | **1.523 °C/min** |
| Linear fit R² | **0.9747** (near-perfect linear ramp) |

The extremely high R² confirms this is not noise — it is a steady accumulation of OBC processor heat.
All other boards (EPS, Battery, Backplane, ADCS, Wheel, Comms, Payload) remained within ±0.3 °C of ambient (~21 °C).

**Implication:** If the session had continued, the OBDH would reach ~50 °C within ~18 minutes. Long-duration tests should monitor this channel against the board's thermal limit. Ensure adequate ventilation or consider active thermal management.

> Plot: [Output/02_obdh_thermal_ramp.png](Output/02_obdh_thermal_ramp.png)

---

### 3.2 Stuck Thermistor Channels ⚠️ HIGH PRIORITY

**20 thermistor channels** read a completely static, non-physical value for the entire session:

| Group | Channels | Value | Physical? |
|-------|----------|-------|-----------|
| `THERMISTOR_ROD_1` – `_16` | 16 channels | **77.180115 °C** | No — ambient is ~21 °C |
| `THERMISTOR_BATH_1` – `_3` | 3 channels | **74.309649 °C** | No |
| `THERMISTOR_EXP3` | 1 channel | **74.309649 °C** | No |

These values are identical and constant across all 333 frames. This is characteristic of a **disconnected or open-circuit thermistor** — the ADC reads a fixed rail voltage rather than a resistance-derived temperature. The thermal experiment module (rods/bath) does not appear to be connected or is unpowered.

**Implication:** All thermal experiment data from this session is invalid and must be discarded. Hardware connection and wiring should be verified before the next session.

> Plot: [Output/04_thermistor_anomaly.png](Output/04_thermistor_anomaly.png)

---

### 3.3 Residual Voltages on Powered-Off Rails ⚠️ MEDIUM PRIORITY

Both ADCS and Payload subsystems were commanded off (`ADCS_PWR_STATUS = 0`, `PAYLOAD_PWR_STATUS = 0`), yet their 5 V supply rails showed non-zero voltages:

| Rail | Expected | Measured (mean) | Delta |
|------|----------|-----------------|-------|
| `ADCS_5V_V` | 0.000 V | **0.106 V** | +0.106 V |
| `PAYLOAD_5V_V` | 0.000 V | **0.686 V** | +0.686 V |

The Payload rail is particularly notable at ~0.69 V. Possible causes:
- ADC input impedance / measurement offset at the EPS board
- Backplane leakage through protection diodes
- Capacitive hold-up after a prior power-on cycle

These are not hazardous levels but represent measurement inaccuracies that should be baselined and, if possible, corrected via calibration offsets in firmware.

> Plot: [Output/06_residual_rail_voltages.png](Output/06_residual_rail_voltages.png)

---

### 3.4 Z-Score Anomalies on Power Channels ℹ️ LOW PRIORITY

Statistical z-score analysis (threshold = 2.5σ) on key telemetry channels:

| Channel | Anomalous Samples | Notes |
|---------|-------------------|-------|
| `OBC_PWR_V` | **9** | Transient voltage spikes on the OBC 5 V supply |
| `SYS_V` | 3 | Minor system bus fluctuations |
| `PLAT_5V_V` | 3 | Minor platform 5 V fluctuations |
| `TEMP_OBDH` | 0 | Ramp is smooth — no sudden jumps |
| `TEMP_COMMS`, `TEMP_SOLAR` | 0 | Stable |
| `COMMS_PWR_V` | 0 | Stable |

The `OBC_PWR_V` spikes (9 samples flagged) warrant investigation — they could be switching noise from the OBC processor during computation bursts. The magnitude is small but repeated.

---

### 3.5 ADCS Sensors — All Zero ℹ️ EXPECTED

All ADCS sensor channels (`ACCEL_X/Y/Z`, `GYRO_X/Y/Z`, `MAG_X/Y/Z`, sun sensors, wheel speed, magnetorquers) read exactly zero throughout the session.

**This is expected** — `ADCS_PWR_STATUS = 0`, meaning the ADCS board is unpowered. The STM32 ADCS processor returns zeroes as the default null state. No anomaly.

---

## 4. Power System Summary

| Parameter | Value | Assessment |
|-----------|-------|------------|
| Battery voltage (`BATT_CHR_V`) | 7.415 V | Healthy (2S Li-Ion nominal: 7.4 V) |
| System bus voltage (`SYS_V`) | 7.303–7.310 V | Stable |
| Solar voltage (`SOLAR_V`) | 1.927 V (constant) | Indoor — no meaningful charging |
| Battery charge current | `0xFF23` (negative raw) | **Battery discharging** throughout |
| Platform 5 V | 4.986–4.987 V | Nominal |
| OBC supply | 4.956–4.960 V | Nominal (minor spikes noted) |
| Comms supply | 4.980–4.981 V | Nominal |
| Comms frequency | 910.250 MHz | US 915 MHz band |
| RSSI | 0 | No downlink signal received (ground test) |
| Frame errors | 0 | Clean comms |

---

## 5. Thermal Correlation Analysis

The temperature correlation matrix ([Output/05_temperature_correlation.png](Output/05_temperature_correlation.png)) reveals:

- **OBDH** is largely decorrelated from the other boards — it is the only board with a significant thermal source (OBC processor). It heats independently.
- **EPS, Battery, Backplane, Solar, Comms, Payload** are highly correlated (r > 0.85) — they are all tracking the same slowly drifting ambient temperature. This is consistent with a passive ground test environment.
- **ADCS and Wheel** are moderately correlated with the ambient group and with each other.

---

## 6. Plots Generated

### Original Set

| File | Description |
|------|-------------|
| [Output/01_board_temperatures.png](Output/01_board_temperatures.png) | All 9 board sensor temperatures vs elapsed time |
| [Output/02_obdh_thermal_ramp.png](Output/02_obdh_thermal_ramp.png) | OBDH temperature with linear regression fit (R²=0.9747) |
| [Output/03_power_rails.png](Output/03_power_rails.png) | All power rail voltages vs elapsed time |
| [Output/04_thermistor_anomaly.png](Output/04_thermistor_anomaly.png) | Stuck thermistor channels vs real board sensors |
| [Output/05_temperature_correlation.png](Output/05_temperature_correlation.png) | Temperature correlation heatmap |
| [Output/06_residual_rail_voltages.png](Output/06_residual_rail_voltages.png) | Residual voltages on powered-off ADCS/Payload rails |
| [Output/07_summary_dashboard.png](Output/07_summary_dashboard.png) | 6-panel summary dashboard |
| [Output/08_temperature_distributions.png](Output/08_temperature_distributions.png) | Violin plot of temperature distributions across all boards |

### Extended Set

| File | Description | Key Insight |
|------|-------------|-------------|
| [Output/09_current_draw.png](Output/09_current_draw.png) | Subsystem current draw (ADC counts) + signed battery current | Battery discharging (BATT_CHR_I negative) throughout; OBC and COMMS draw stable currents |
| [Output/10_temperature_deltas.png](Output/10_temperature_deltas.png) | ΔTemp from session-start baseline for every board | OBDH stands alone at +4.5°C; all others drift <0.4°C, confirming processor as sole heat source |
| [Output/11_obdh_rolling_stats.png](Output/11_obdh_rolling_stats.png) | OBDH rolling mean ± 1σ band (window=15) | Ramp is smooth and consistent; σ is tiny, confirming steady heat accumulation not noise |
| [Output/12_packet_timing_jitter.png](Output/12_packet_timing_jitter.png) | CDH packet inter-arrival histogram + sequential view | Mean interval 0.506 s, std 0.140 s; max gap 1.152 s (~2× nominal); no dropped packet bursts |
| [Output/13_temperature_rate_of_change.png](Output/13_temperature_rate_of_change.png) | dT/dt (°C/s) per board, smoothed rolling mean | OBDH shows a consistent positive dT/dt of ~+0.025 °C/s; all other boards near zero |
| [Output/14_voltage_anomaly_overlay.png](Output/14_voltage_anomaly_overlay.png) | OBC_PWR_V and SYS_V with z-score anomaly points overlaid | 9 OBC voltage spikes clustered in time — likely OBC compute bursts causing switching noise |
| [Output/15_temperature_pairplot.png](Output/15_temperature_pairplot.png) | Seaborn pairplot of 5 key temperature channels | OBDH vs others show fan-shaped scatter (OBDH drifting); Comms correlates moderately with OBDH |

---

## 7. Recommended Actions

| Priority | Action |
|----------|--------|
| 🔴 High | Investigate thermal experiment module wiring — all 20 thermistor channels return static non-physical values |
| 🔴 High | Monitor OBDH temperature in longer sessions; 1.52 °C/min is substantial — establish thermal limit thresholds |
| 🟡 Medium | Characterise residual rail voltages (ADCS: 0.11 V, Payload: 0.69 V) — apply calibration offsets or identify hardware cause |
| 🟡 Medium | Investigate 9 z-score anomalies on `OBC_PWR_V` — may indicate switching noise |
| 🟢 Low | Run a session with solar panels under direct illumination to validate MPPT charging behaviour |
| 🟢 Low | Enable ADCS subsystem in a future session to validate IMU, magnetometer, and sun sensor readings |
| 🟢 Low | Test with payload powered to characterise camera and imaging pipeline telemetry |

---

## 8. Processing Artefacts

### Pipeline Structure

```
main.py                  ← entry point (parse → clean → detect → plot)
scripts/
  __init__.py
  parser.py              ← SCOTTI archive parser → raw CDH / ADCS DataFrames
  cleaner.py             ← dedup, hex current conversion, elapsed time, power estimation
  detector.py            ← z-score, rolling-change, linear trend, packet timing
  plotter.py             ← all 15 plot functions + run_all_plots()
Output/
  cdh_clean.csv          ← 333 rows × 88 columns (cleaned CDH telemetry)
  adcs_clean.csv         ← 335 rows × 31 columns (cleaned ADCS telemetry)
  01_*.png … 15_*.png    ← all generated figures
.venv/                   ← Python environment (pandas, numpy, seaborn, matplotlib, scipy)
```

### Notable Cleaner Improvements over v1
- **Hex current conversion** — `SYS_I`, `OBC_PWR_I`, `PLAT_5V_I`, `COMMS_PWR_I` etc. are now decoded from raw hex strings to numeric ADC counts rather than being dropped as NaN
- **Signed current** — `BATT_CHR_I` decoded as signed 16-bit two's complement (negative = discharging)
- **Estimated power columns** — `PWR_SYS`, `PWR_PLAT_5V`, `PWR_OBC_PWR`, `PWR_COMMS_PWR` added (V × I, relative mW)
