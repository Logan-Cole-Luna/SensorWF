# SLL Telemetry — Findings & Anomaly Injection Strategy

**Platform:** KISPE Satellite Learning Laboratory (SLL)
**Pipeline version:** v2 (multi-experiment, results/ subfolders)
**Report date:** 2026-03-19
**Experiments processed:** 5 of 7 (Magnetorquer_1 and AttitudeDeterminationMagnetometer_1 pending re-run)

---

## 1. Experiment Inventory

| Experiment | Date | Duration | CDH rows | ADCS rows | ADCS live? |
|---|---|---|---|---|---|
| Baseline_1 | 2026-03-02 | 168 s | 333 | 335 | No (powered off) |
| SunSensorTest_1 | 2026-03-10 | 181 s | 361 | 361 | Yes |
| Accelerometer_1 | 2026-03-12 | 180 s | 360 | 363 | Yes |
| Gyro_1 | 2026-03-12 | 133 s | 267 | 264 | Yes |
| ReactionWheel_1 | 2026-03-12 | 180 s | 357 | 359 | Yes |
| Magnetorquer_1 | 2026-03-12 | — | — | — | Empty analysis file |
| AttitudeDeterminationMagnetometer_1 | 2026-03-10 | — | — | — | Not yet processed |

All sessions run at approximately **2 Hz per telemetry stream** (CDH interval mean ≈ 0.500 s, std ≈ 0.137 s).

---

## 2. Cross-Experiment Findings

### 2.1 OBDH Thermal Ramp — Present in Every Session

The OBC processor (STM32 OBDH board) exhibits a **sustained, near-linear temperature rise** in all experiments without exception. This is the most prominent and reproducible signal in the dataset.

| Experiment | Start (°C) | End (°C) | Δ (°C) | Slope (°C/min) | R² |
|---|---|---|---|---|---|
| Baseline_1 | 23.19 | 27.68 | +4.49 | 1.523 | 0.9747 |
| SunSensorTest_1 | 27.88 | 32.12 | +4.24 | 1.257 | 0.9436 |
| Accelerometer_1 | 22.77 | 28.86 | +6.09 | **1.920** | 0.9445 |
| Gyro_1 | 24.44 | 29.30 | +4.86 | **2.042** | 0.9475 |
| ReactionWheel_1 | 24.68 | 29.91 | +5.23 | 1.557 | 0.9427 |

**Key observations:**
- All sessions start from different initial temperatures (ambient varies ±2 °C session to session), confirming sessions are separated by cool-down periods.
- Slope is steepest during Gyro_1 and Accelerometer_1 — both involved physically rotating the satellite, potentially inducing higher OBC compute load (ADCS sensor fusion, attitude estimation).
- R² > 0.94 in all cases confirms this is a deterministic ramp, not noise.
- At ~1.5–2 °C/min, the OBC would approach thermal limits (~70 °C) within 25–45 minutes of continuous operation. Long-duration sessions require monitoring.
- All other boards (EPS, Battery, Backplane, Comms, Payload, ADCS, Wheel, Solar) drift with ambient (±0.5 °C) and are **decorrelated** from the OBDH ramp.

### 2.2 Stuck Thermistors — Persistent Hardware Fault

**21 thermistor channels** have returned non-physical, fully static values across all 5 processed sessions. This fault has not changed between 2026-03-02 and 2026-03-12.

| Group | Channels | Stuck value | Physical? |
|---|---|---|---|
| `THERMISTOR_ROD_1` – `_16` | 16 | **77.180115 °C** | No — ambient ~22–25 °C |
| `THERMISTOR_BATH_1` – `_3` | 3 | **74.309649 °C** | No |
| `THERMISTOR_EXP3` | 1 | **74.309649 °C** | No |
| `THERMISTOR_HEATER` | 1 | 0 | Unclear |

The exact constant values (77.180115 and 74.309649) are ADC rail-voltage artefacts from open-circuit inputs — the thermal experiment module is physically disconnected. This module should be connected before thermal expansion experiments are attempted.

### 2.3 Residual Voltages on Powered-Off Rails

Both the ADCS and Payload 5 V rails read non-zero even when their subsystems are commanded off. Consistent across all experiments:

| Rail | Expected | Typical measured | Likely cause |
|---|---|---|---|
| `ADCS_5V_V` | 0.000 V | ~0.09–0.11 V | ADC measurement floor / backplane leakage |
| `PAYLOAD_5V_V` | 0.000 V | ~0.68–0.69 V | Capacitive hold-up or protection diode drop |

The PAYLOAD rail at ~0.68 V is notable. This is repeatable and consistent so appears to be a hardware measurement offset rather than a true residual charge.

### 2.4 Battery Discharge — All Sessions

The `BATT_CHR_I` field (signed 16-bit, two's complement) is negative throughout every session, confirming the battery is discharging during all lab tests. Solar panels provide minimal charging under indoor ambient light.

| Experiment | `BATT_CHR_V` range | `SYS_V` range | Notes |
|---|---|---|---|
| Baseline_1 | 7.415 V (stable) | 7.296–7.310 V | Flat — light load |
| SunSensorTest_1 | 7.295–7.388 V | 7.161–7.281 V | Wider — ADCS active |
| Accelerometer_1 | 7.321–7.360 V | 7.142–7.263 V | ADCS + motion |
| Gyro_1 | 7.314–7.324 V | 7.130–7.251 V | ADCS active |
| **ReactionWheel_1** | **7.144–7.334 V** | **6.985–7.226 V** | **SYS_V drops to 6.985 V — motor load visible** |

The ReactionWheel experiment shows the largest voltage sag on the system bus (SYS_V reaching 6.985 V), a direct signature of the motor drawing current.

### 2.5 CDH Packet Timing

| Experiment | Mean interval (s) | Std (s) | Max gap (s) |
|---|---|---|---|
| Baseline_1 | 0.506 | 0.140 | 1.152 |
| SunSensorTest_1 | 0.501 | 0.133 | 0.802 |
| Accelerometer_1 | 0.503 | 0.137 | 1.000 |
| Gyro_1 | 0.500 | 0.137 | 0.705 |
| ReactionWheel_1 | 0.504 | 0.145 | 1.154 |

Timing is consistently ~0.5 s mean with ~0.14 s jitter. Occasional gaps up to ~1.15 s (≈ 2× nominal) — consistent with OS scheduling jitter, not packet loss.

---

## 3. Per-Experiment ADCS Observations

### 3.1 Baseline_1 — ADCS Off

All ADCS sensor channels (`ACCEL_*`, `GYRO_*`, `MAG_*`, sun sensors, wheel speed) are exactly zero. `ADCS_PWR_STATUS = 0`. This session provides the **clean power/thermal baseline** with no actuator or sensor noise.

### 3.2 SunSensorTest_1 — Full-Range Sun Sensor Sweep

The widest sensor dynamic range in the dataset. Sun sensors were swept through direct illumination:

| Channel | Min | Max | Notes |
|---|---|---|---|
| `SUN_SENSOR_1a/1b` | 10 | 1014 | Near ADC full-scale — direct sun source |
| `SUN_SENSOR_2a` | 26 | 894 | |
| `SUN_SENSOR_3b` | 42 | 962 | |
| `SUN_SENSOR_4a` | 10 | 1012 | |
| `SUN_ANGLE` | 10° | 348° | Full 0–360° sweep |

IMU data was live but small in magnitude (accel ≈ −1g on Z = gravity; gyro X/Y/Z up to ±0.8°/s — minimal rotation). Magnetometer swept ±32 µT.

### 3.3 Accelerometer_1 — Manual Rotation Test

The satellite was physically rotated/shaken. This is the highest-motion ADCS dataset:

| Sensor | Range | Physical interpretation |
|---|---|---|
| `ACCEL_X` | −1.07 to +1.03 g | Full ±1g — rotation through all axes |
| `ACCEL_Y` | −1.06 to +1.03 g | |
| `ACCEL_Z` | −1.03 to +0.04 g | Z changes sign — inverted at times |
| `GYRO_X` | −70.4 to +71.4 °/s | Rapid rotation |
| `GYRO_Y` | −102.9 to +148.9 °/s | Fastest axis |
| `GYRO_Z` | −56.3 to +27.6 °/s | |
| `MAG_X/Y/Z` | ±33 µT | Rotating through Earth's field |

Wheel speed residual: −6 to −1 counts (essentially zero — bearing drag only). OBDH slope steepest here (1.92 °C/min).

### 3.4 Gyro_1 — Gyroscope-Focused Rotation

Lower-amplitude motion than Accelerometer_1 (accelerometer mostly near 1g gravity vector), but gyro axes active:

| Sensor | Range | Notes |
|---|---|---|
| `ACCEL_Z` | −1.03 to 0.00 g | Mostly upright |
| `GYRO_Z` | −62.8 to +102.2 °/s | Primary rotation axis |
| `GYRO_Y` | −28.7 to +14.8 °/s | Secondary |
| `MAG_*` | ±33 µT | Full range rotation |

ADCS slope 2.04 °C/min — steepest of all experiments.

### 3.5 ReactionWheel_1 — Bidirectional Wheel Spin

The most operationally complex experiment. The reaction wheel was driven in both directions:

| Wheel state | Speed range (counts) | Physical interpretation |
|---|---|---|
| Reverse spin | −218 to −96 | Negative torque |
| Near-zero / transition | −19 to +26 | Spin-up / spin-down transitions |
| Forward spin | +193 to +991 | Positive torque — reaching ~1000 RPM equivalent |

Gyro axes active (GYRO_Z up to ±100 °/s) — angular momentum of wheel produces satellite body rotation. SYS_V sag to 6.985 V during peak motor drive.

---

## 4. Anomaly Injection Strategy

The goal is to produce a **labelled dataset** for training anomaly detection models — each injected anomaly is timestamped, typed, and of known severity. Injections must be realistic: within sensor physics, consistent with spacecraft operating constraints, and indistinguishable from real faults to a naive classifier.

All injection should be implemented as a separate `scripts/injector.py` module that loads a clean CSV, applies a transformation, and exports a labelled copy. The label schema should record: `anomaly_type`, `start_idx`, `end_idx`, `severity`, `affected_channels`.

---

### 4.1 Thermal Anomalies

#### T-1: OBDH Ramp Acceleration
Multiply the OBDH slope by a factor (e.g., 3×, 5×) starting at a random mid-session timestamp.
- **Baseline normal:** 1.5–2 °C/min
- **Injected:** 6–10 °C/min
- **Target channel:** `TEMP_OBDH`
- **Realistic range:** up to ~60 °C before thermal cutoff
- **Detection hook:** linear trend detector; z-score on rolling dT/dt

#### T-2: Sudden Board Step Change
Inject a step-function temperature increase on any non-OBDH board (e.g., `TEMP_COMMS` jumps +8 °C in one sample, then settles slowly).
- Models a latch-up or transient fault condition
- **Detection hook:** rolling-change spike detector

#### T-3: Board Temperature Divergence
Two boards normally correlated (r > 0.85, e.g. EPS + Battery) suddenly diverge: one continues its trajectory, the other is offset by an injected bias.
- **Detection hook:** correlation breakdown detector (not yet implemented)

#### T-4: Thermistor Un-Sticking (plausible false positive)
Replace a stuck thermistor's constant 77.18 °C with a physically reasonable temperature trajectory (e.g., ramp from 25 °C → 30 °C). Tests whether stuck-sensor detection algorithm false-positives on real data.

---

### 4.2 Power / Voltage Anomalies

#### P-1: System Bus Voltage Sag
Step-drop `SYS_V` by 0.5–1.5 V for N seconds, then recover. Simulates a motor load spike or short-circuit on a rail.
- **Normal range:** 6.985–7.310 V across experiments
- **Injected:** drop to 6.0–6.5 V
- **Correlated injection:** also sag `BATT_CHR_V` slightly (coupled)
- **Detection hook:** z-score on `SYS_V`

#### P-2: OBC Power Spike Cluster
Inject a burst of 5–10 samples where `OBC_PWR_V` spikes up by 0.3–0.8 V above its local baseline.
- Consistent with observed real z-score anomalies (9 flagged in Baseline_1) but at higher amplitude
- **Detection hook:** z-score + rolling-change detector on `OBC_PWR_V`

#### P-3: Rail Stuck High
Force a powered-off rail (`ADCS_5V_V`) to read a plausible but anomalous value (e.g., 3.3 V instead of ~0.1 V). Simulates a stuck relay or power sequencing fault.

#### P-4: Battery Voltage Collapse
Ramp `BATT_CHR_V` downward faster than observed (normal: slow drift). Inject an accelerated drain curve that reaches 6.5 V within the session window.

---

### 4.3 ADCS / IMU Anomalies

#### A-1: Accelerometer Axis Dropout
Force one or more axes (`ACCEL_X`, `ACCEL_Y`, or `ACCEL_Z`) to zero for a contiguous window.
- Simulates sensor fault / I²C timeout
- Testable against Accelerometer_1 and ReactionWheel_1 where motion is significant
- **Detection hook:** stuck-sensor extension to ADCS channels; vector magnitude drop

#### A-2: Gyroscope Saturation
Peg one gyro axis at its maximum observed value (e.g., `GYRO_Y` = 148.9 °/s) for N samples.
- Simulates gimbal lock or out-of-range angular rate
- **Detection hook:** rolling-change spike; stuck-axis detector

#### A-3: Magnetometer Flip
Invert the sign of all three MAG axes simultaneously for a short window.
- Simulates sensor calibration corruption or field anomaly
- Easy to detect: MAG vector direction changes 180°
- **Detection hook:** magnetic heading jump > 90° in one sample

#### A-4: IMU All-Zero Dropout
Force all 9 IMU channels to zero during a live ADCS session (Accelerometer_1, Gyro_1, etc.).
- Simulates ADCS board power cut or SPI bus fault mid-session
- **Detection hook:** `adcs_all_zero` flag; vector magnitude drop

#### A-5: Sun Sensor Saturation
Set all 8 sun sensor channels to their ADC maximum (1023) simultaneously.
- Simulates an intense light source or sensor latch-up
- **Detection hook:** threshold violation + inter-channel correlation collapse

---

### 4.4 Reaction Wheel Anomalies

#### W-1: Wheel Speed Overshoot
Extend the wheel speed ramp beyond the observed maximum (~991 counts) to 1500–2000 counts.
- Models a controller runaway or torque command fault
- **Correlated injection:** increase `GYRO_Z` amplitude proportionally
- **Detection hook:** threshold on `WHEEL_SPEED`; cross-channel correlation (wheel speed ↔ gyro)

#### W-2: Sudden Wheel Stop
At peak speed (~991 counts), drop `WHEEL_SPEED` to 0 in one sample.
- Models bearing seizure or emergency stop
- **Correlated injection:** gyro spike from angular momentum transfer
- **Detection hook:** rolling-change spike on `WHEEL_SPEED`

#### W-3: Wheel Speed Oscillation
Inject a sinusoidal oscillation around zero (±50 counts, period 5–10 samples).
- Models a stuck controller or hunting behaviour
- **Detection hook:** FFT / spectral anomaly (not yet implemented)

#### W-4: SYS_V Sag on Wheel Spin-Up
Extend the natural SYS_V dip (real: 6.985 V) to a more severe sag (6.0 V) during simulated wheel spin-up. Cross-channel anomaly spanning EPS and ADCS telemetry.

---

### 4.5 Communication / Timing Anomalies

#### C-1: Packet Gap Injection
Remove N consecutive CDH records to simulate a communications blackout or packet loss event.
- Real max gap: 1.154 s (2× nominal). Inject gaps of 5–30 s.
- **Detection hook:** `packet_timing` max_gap_s; sequence number gaps

#### C-2: Timestamp Jitter Amplification
Add Gaussian noise to inter-packet timestamps (std = 500 ms vs normal 140 ms).
- Models a clock drift or RTOS scheduling fault
- **Detection hook:** timing jitter z-score

#### C-3: Duplicate Packet Burst
Inject 5–10 duplicate timestamps with identical telemetry values.
- Models a firmware re-transmission bug
- **Detection hook:** duplicate-timestamp check in cleaner

---

### 4.6 Multi-System Compound Anomalies

For training a robust detector these are the most valuable injection targets — they require cross-channel correlation to distinguish from single-channel faults.

| Label | Description | Channels |
|---|---|---|
| `FAULT_THERMAL_POWER` | OBDH ramp + SYS_V sag simultaneously | `TEMP_OBDH`, `SYS_V`, `BATT_CHR_V` |
| `FAULT_ADCS_POWER` | IMU dropout + ADCS_5V_V anomaly | `ACCEL_*`, `GYRO_*`, `ADCS_5V_V` |
| `FAULT_WHEEL_BUS` | Wheel overspeed + SYS_V collapse | `WHEEL_SPEED`, `SYS_V`, `GYRO_Z` |
| `FAULT_SENSOR_CASCADE` | Thermistor un-stick + magnetometer flip | `THERMISTOR_*`, `MAG_*` |

---

## 5. Recommended Next Steps

| Priority | Action |
|---|---|
| 🔴 High | Build `scripts/injector.py` — load clean CSVs, apply typed injections, export with labels |
| 🔴 High | Re-run pipeline for `AttitudeDeterminationMagnetometer_1` and `Magnetorquer_1` after debugging empty analysis file |
| 🟡 Medium | Add ADCS stuck-sensor detection in `detector.py` (extend to live ADCS channels, not just thermistors) |
| 🟡 Medium | Add cross-channel correlation breakdown detector (EPS/Battery divergence, wheel↔gyro decoupling) |
| 🟡 Medium | Fix thermal experiment module wiring — 21 thermistors have been stuck across all sessions |
| 🟡 Medium | Implement `W-3` wheel oscillation detector via FFT on `WHEEL_SPEED` |
| 🟢 Low | Collect a ReactionWheel session with forward spin only to get a cleaner wheel speed signature for baseline |
| 🟢 Low | Run a session with the sun panels under a controlled light source to validate sun sensor calibration |
| 🟢 Low | Investigate OBDH slope increase during rotation tests — may indicate ADCS compute load or vibration coupling |

---

## 6. Pipeline Reference

```
data/
  Baseline_1/                        ← 2026-03-02 reference session
  SunSensorTest_1/                   ← Full sun sensor sweep
  Accelerometer_1/                   ← Manual rotation, high IMU activity
  Gyro_1/                            ← Gyro-focused rotation
  Magnetorquer_1/                    ← Empty analysis file — pending debug
  ReactionWheel_1/                   ← Bidirectional wheel spin
  AttitudeDeterminationMagnetometer_1/ ← Pending processing
results/
  <experiment>/
    cdh_clean.csv                    ← Cleaned CDH telemetry
    adcs_clean.csv                   ← Cleaned ADCS telemetry (hex fields decoded)
    01_board_temperatures.png        ← Plots 01–15: CDH-based
    ...
    16_imu_sensors.png               ← Plots 16–20: ADCS sensor plots (new)
    ...
scripts/
  parser.py                          ← SCOTTI archive → raw DataFrames
  cleaner.py                         ← Hex decode, dedup, elapsed time, power calc
  detector.py                        ← Z-score, rolling-change, trend, timing
  plotter.py                         ← 20 plot functions (15 CDH + 5 ADCS)
  injector.py                        ← (planned) Labelled anomaly injection
main.py                              ← Pipeline entry point
models.md                            ← TinyML model reference for onboard IDS
```
