# Project Memory — Satellite Anomaly Detection

## Project Overview
- **Platform:** KISPE Satellite Learning Laboratory (SLL) — educational CubeSat-form-factor satellite
- **Goal:** Anomaly detection on telemetry extracted from the SLL
- **Working dir:** `/Users/loganluna/VSCode/SatelliteAnomalyDetection/`

## Key Files
- `Data/Archive_2026_03_02_112454_Analysis.txt` — verbose parsed telemetry (42k lines, CDH + ADCS packets)
- `Data/Archive_2026_03_02_112454_Debug.txt` — compact raw/plot debug stream (1.3k lines)
- `process_telemetry.py` — full parsing → cleaning → anomaly detection → seaborn plotting pipeline
- `FINDINGS.md` — markdown report of findings
- `Output/` — all plots (PNGs) + cleaned CSVs (`cdh_clean.csv`, `adcs_clean.csv`)
- `.venv/` — Python venv; use `.venv/bin/python` to run scripts

## Data Structure
- **Two packet types** interleaved at ~2 Hz: CDH (EPS/thermal/comms) and ADCS Sensor
- **333 CDH packets**, **335 ADCS packets**, session duration ~2m 47s
- Debug file uses compact single-line format; Analysis file has labeled `FIELD = VALUE` per line

## Hardware Subsystems (from datasheet)
- OBC (STM32), EPS (7.4V 5000mAh Li-Ion + MPPT solar), ADCS (STM32, 3-axis mag/torquers, reaction wheel, IMU), COMMS (UHF 915MHz), Payload (OV2620 camera), CAN backplane
- SW versions in telemetry: OBC=0x0100, PWR=0x0101, COMMS=0x0102, ADCS=0x0000

## Known Anomalies Found (Session 2026-03-02)
1. **OBDH temp ramp** — 1.523 °C/min, R²=0.9747, +4.48°C over session (highest priority)
2. **20 stuck thermistor channels** — THERMISTOR_ROD_1–16 all at 77.18°C; BATH_1–3 + EXP3 at 74.31°C (disconnected/open-circuit)
3. **Residual rail voltages** — ADCS_5V=0.106V, PAYLOAD_5V=0.686V when subsystems are off
4. **Z-score anomalies** — OBC_PWR_V: 9 samples, SYS_V: 3, PLAT_5V_V: 3
5. **ADCS all-zero** — expected (ADCS unpowered this session)

## User Preferences
- Use `.venv` for Python (always create venv first before pip install)
- Use seaborn for plots
- Document findings in markdown
