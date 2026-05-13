"""
satellite.py -- SensorWF M1 adapter for KISPE SATLL SCOTTI v2 telemetry.

Wraps the bespoke SCOTTI archive parser and CDH/ADCS cleaner so the
satellite domain plugs into the generic M2-M5 pipeline (pipeline_core,
build_generic_feature_matrix, provenance_recorder) exactly like ECG and
Climate adapters do.

The SATLL produces two separate telemetry streams per session:
  CDH  -- on-board data handling (power, thermal, comms, timing; ~87 cols)
  ADCS -- attitude & control (IMU, wheels, sun sensors; ~30 cols)

load() merges both streams into a single DataFrame on the CDH timestamp
timeline (merge_asof, 2 s tolerance).  The separate CDH and ADCS channel
lists are preserved as instance attributes (cdh_channels / adcs_channels)
so callers that need the split (e.g. sat_anomaly.py's fault injector) can
call split() to recover the two DataFrames.

References
----------
KISPE Satellite Learning Laboratory (SATLL), KISPE Space Systems Ltd.
SCOTTI v2 telemetry format: Analysis and Debug archive files.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from .base import DomainAdapter


_ONTOLOGY_PATH = os.path.join("results", "ontologies", "satellitesystem.owl")

# ── SCOTTI archive parser (M1) ─────────────────────────────────────────────────

_TS_RE    = re.compile(r'^(\d{2}:\d{2}:\d{2}\.\d+) (\d{2}/\d{2}/\d{4})\s+\(([\w ]+)\)')
_FIELD_RE = re.compile(r'^\t\t(\w+)\s+=\s+(.+)$')
_DBG_RE   = re.compile(
    r'^(\d{2}/\d{2}/\d{4}) (\d{2}:\d{2}:\d{2}) (CDH|ADCS Sensor) Plot (.+)$'
)

_CDH_FIELDS = [
    "FSW_DB_VER","DAY","HOUR","MINUTE","SECOND","SC_ID",
    "CMD_RCV_CNT","CMD_ACC_CNT","CMD_REJ_CNT","CMD_INC_CNT",
    "PAYLOAD_STATUS","PLATFORM_PWR_STATUS","ADCS_PWR_STATUS",
    "PAYLOAD_PWR_STATUS","SPR_PWR_STATUS",
    "SOLAR_V","SOLAR_S","EXT_V","EXT_S",
    "BATT_CHR_V","BATT_CHR_I","BATT_CHR_STATUS",
    "SYS_V","SYS_I","SPR_V","SPR_I",
    "PLAT_5V_V","PLAT_5V_I","ADCS_5V_V","ADCS_5V_I",
    "PAYLOAD_5V_V","PAYLOAD_5V_I","OBC_PWR_V","OBC_PWR_I",
    "ADCS_PROC_PWR_V","ADCS_PROC_PWR_I","ADCS_PER_PWR_V","ADCS_PER_PWR_I",
    "COMMS_PWR_V","COMMS_PWR_I",
    "TEMP_SOLAR","TEMP_EPS","TEMP_BATTERY","TEMP_BACKPLANE","TEMP_OBDH",
    "TEMP_ADCS","TEMP_WHEEL","TEMP_COMMS","TEMP_PAYLOAD",
    "TEMP_EPS_L","TEMP_OBDH_L","TEMP_ADCS_L","TEMP_WHEEL_L",
    "TEMP_COMMS_L","TEMP_PAYLOAD_L",
    "FREQ","RSSI","TX_POWER","FRAME_ERR","TEMP_THERM_EXP_L",
    "THERMISTOR_ROD_1","THERMISTOR_ROD_2","THERMISTOR_ROD_3","THERMISTOR_ROD_4",
    "THERMISTOR_ROD_5","THERMISTOR_ROD_6","THERMISTOR_ROD_7","THERMISTOR_ROD_8",
    "THERMISTOR_ROD_9","THERMISTOR_ROD_10","THERMISTOR_ROD_11","THERMISTOR_ROD_12",
    "THERMISTOR_ROD_13","THERMISTOR_ROD_14","THERMISTOR_ROD_15","THERMISTOR_ROD_16",
    "THERMISTOR_BATH_1","THERMISTOR_BATH_2","THERMISTOR_BATH_3",
    "THERMISTOR_EXP3","THERMISTOR_HEATER","THERMAL_EXP_SW_VERSION",
    "OBC_SW_VERSION","PWR_SW_VERSION","COMMS_SW_VERSION",
    "ADCS_SW_VERSION","ACTUATOR_SW_VERSION",
]

_ADCS_FIELDS = [
    "ADCS_MODE",
    "ACCEL_X","ACCEL_Y","ACCEL_Z",
    "GYRO_X","GYRO_Y","GYRO_Z",
    "MAG_X","MAG_Y","MAG_Z",
    "SUN_SENSOR_1a","SUN_SENSOR_1b",
    "SUN_SENSOR_2a","SUN_SENSOR_2b",
    "SUN_SENSOR_3a","SUN_SENSOR_3b",
    "SUN_SENSOR_4a","SUN_SENSOR_4b",
    "SUN_ANGLE","WHEEL_SPEED",
    "MAGNETORQUER_X","MAGNETORQUER_Y","MAGNETORQUER_Z",
    "KP_VALUE","KD_VALUE","ADCS_VERSION",
    "ADCS_ACC_CNT","ADCS_REJ_CNT","ADCS_INC_CNT","MAG_DEPLOYMENT_STATUS",
]


def _flush_record(record, pkt_type, cdh_list, adcs_list):
    if record is None:
        return
    if pkt_type == "CDH":
        cdh_list.append(record)
    elif pkt_type == "ADCS Sensor":
        adcs_list.append(record)


def _parse_analysis_file(filepath: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cdh_records: list[dict]  = []
    adcs_records: list[dict] = []
    current_record: dict | None = None
    current_type: str | None    = None

    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip()
            m = _TS_RE.match(line)
            if m:
                _flush_record(current_record, current_type, cdh_records, adcs_records)
                time_str, date_str, pkt_type = m.group(1), m.group(2), m.group(3)
                ts = None
                dt_str = f"{date_str} {time_str}"
                for fmt in ("%d/%m/%Y %H:%M:%S.%f", "%m/%d/%Y %H:%M:%S.%f"):
                    try:
                        ts = pd.to_datetime(dt_str, format=fmt)
                        break
                    except Exception:
                        pass
                current_record = {"timestamp": ts}
                current_type   = pkt_type.strip()
                continue

            m2 = _FIELD_RE.match(line)
            if m2 and current_record is not None:
                name, raw = m2.group(1), m2.group(2).strip()
                try:
                    current_record[name] = float(raw)
                except ValueError:
                    current_record[name] = raw

    _flush_record(current_record, current_type, cdh_records, adcs_records)

    cdh  = pd.DataFrame(cdh_records)
    adcs = pd.DataFrame(adcs_records)
    if not cdh.empty:
        cdh  = cdh.sort_values("timestamp").reset_index(drop=True)
    if not adcs.empty:
        adcs = adcs.sort_values("timestamp").reset_index(drop=True)
    return cdh, adcs


def _parse_debug_file(filepath: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cdh_records: list[dict]  = []
    adcs_records: list[dict] = []

    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip()
            m = _DBG_RE.match(line)
            if not m:
                continue
            date_str, time_str, pkt_type, values_str = (
                m.group(1), m.group(2), m.group(3), m.group(4)
            )
            try:
                ts = pd.to_datetime(f"{date_str} {time_str}", format="%m/%d/%Y %H:%M:%S")
            except Exception:
                ts = None

            tokens = values_str.split()
            if pkt_type == "CDH":
                fields, records = _CDH_FIELDS, cdh_records
            else:
                fields, records = _ADCS_FIELDS, adcs_records

            record: dict = {"timestamp": ts}
            for name, raw in zip(fields, tokens):
                try:
                    record[name] = float(raw)
                except ValueError:
                    record[name] = raw
            records.append(record)

    cdh  = pd.DataFrame(cdh_records)
    adcs = pd.DataFrame(adcs_records)
    if not cdh.empty:
        cdh  = cdh.sort_values("timestamp").reset_index(drop=True)
    if not adcs.empty:
        adcs = adcs.sort_values("timestamp").reset_index(drop=True)
    return cdh, adcs


# ── CDH / ADCS cleaner (M1) ────────────────────────────────────────────────────

_HEX_UINT_COLS = [
    "SYS_I", "OBC_PWR_I", "COMMS_PWR_I", "PLAT_5V_I",
    "ADCS_5V_I", "PAYLOAD_5V_I", "SPR_I", "EXT_I",
    "ADCS_PROC_PWR_I", "ADCS_PER_PWR_I",
]
_HEX_INT16_COLS = ["BATT_CHR_I"]
_ADCS_HEX_UINT_COLS = [
    "ADCS_MODE",
    "SUN_SENSOR_1a", "SUN_SENSOR_1b",
    "SUN_SENSOR_2a", "SUN_SENSOR_2b",
    "SUN_SENSOR_3a", "SUN_SENSOR_3b",
    "SUN_SENSOR_4a", "SUN_SENSOR_4b",
    "SUN_ANGLE", "KP_VALUE", "KD_VALUE", "ADCS_VERSION",
    "ADCS_ACC_CNT", "ADCS_REJ_CNT", "ADCS_INC_CNT", "MAG_DEPLOYMENT_STATUS",
]
_ADCS_HEX_INT16_COLS = [
    "WHEEL_SPEED",
    "MAGNETORQUER_X", "MAGNETORQUER_Y", "MAGNETORQUER_Z",
]
_POWER_PAIRS = [
    ("SYS_V",       "SYS_I"),
    ("PLAT_5V_V",   "PLAT_5V_I"),
    ("OBC_PWR_V",   "OBC_PWR_I"),
    ("COMMS_PWR_V", "COMMS_PWR_I"),
]
_CURRENT_SCALE = 1.0


def _hex_to_uint(val) -> float:
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(int(str(val), 16))
    except (ValueError, TypeError):
        return np.nan


def _hex_to_int16(val) -> float:
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    try:
        n = int(str(val), 16)
        if n >= 0x8000:
            n -= 0x10000
        return float(n)
    except (ValueError, TypeError):
        return np.nan


def _clean_cdh(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    df = df.dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset="timestamp", keep="last")

    for col in _HEX_UINT_COLS:
        if col in df.columns:
            df[col] = np.array([_hex_to_uint(v) for v in df[col]], dtype=float) * _CURRENT_SCALE
    for col in _HEX_INT16_COLS:
        if col in df.columns:
            df[col] = np.array([_hex_to_int16(v) for v in df[col]], dtype=float) * _CURRENT_SCALE

    for col in df.columns:
        if col == "timestamp":
            continue
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    t0 = df["timestamp"].iloc[0]
    df["elapsed_s"] = (df["timestamp"] - t0).dt.total_seconds()

    for v_col, i_col in _POWER_PAIRS:
        if v_col in df.columns and i_col in df.columns:
            df[f"PWR_{v_col.replace('_V', '')}"] = df[v_col] * df[i_col]

    therm_cols = [c for c in df.columns if c.startswith("THERMISTOR_")]
    stuck_sensors = [
        c for c in therm_cols
        if c in df.columns and df[c].nunique(dropna=True) == 1
    ]
    df["_has_stuck_thermistor"] = bool(stuck_sensors)
    return df, stuck_sensors


def _clean_adcs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset="timestamp", keep="last")

    for col in _ADCS_HEX_UINT_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_hex_to_uint)
    for col in _ADCS_HEX_INT16_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_hex_to_int16)

    for col in df.columns:
        if col == "timestamp":
            continue
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    t0 = df["timestamp"].iloc[0]
    df["elapsed_s"] = (df["timestamp"] - t0).dt.total_seconds()
    return df


# ── SatelliteAdapter ───────────────────────────────────────────────────────────

class SatelliteAdapter(DomainAdapter):
    name        = "Satellite Telemetry (KISPE SATLL SCOTTI v2)"
    description = (
        "CDH (power, thermal, comms) and ADCS (IMU, wheels, sun sensors) "
        "telemetry at ~1 Hz from KISPE SATLL CubeSat experiment families."
    )
    native_hz = 1.0

    # Populated dynamically by load(); empty until first call.
    channels:      list[str] = []
    cdh_channels:  list[str] = []
    adcs_channels: list[str] = []

    # ------------------------------------------------------------------ M1 --

    def load(self, path: str, **kwargs) -> pd.DataFrame:
        """
        Parse a SCOTTI archive and return a merged CDH+ADCS DataFrame.

        Parameters
        ----------
        path       : path to an Analysis .txt file (preferred format)
        debug_path : keyword arg — Debug .txt file used as fallback when
                     path is absent or produces an empty CDH frame.

        Returns
        -------
        DataFrame with columns: timestamp, elapsed_s, <all CDH channels>,
        <all ADCS channels>.  CDH timeline is authoritative; ADCS values are
        forward-filled into gaps up to 2 s via merge_asof.
        """
        debug_path = kwargs.get("debug_path")

        cdh_raw = adcs_raw = None
        if path and os.path.isfile(path):
            cdh_raw, adcs_raw = _parse_analysis_file(path)
        if (cdh_raw is None or cdh_raw.empty) and debug_path and os.path.isfile(debug_path):
            cdh_raw, adcs_raw = _parse_debug_file(debug_path)

        if cdh_raw is None or cdh_raw.empty:
            raise ValueError(
                f"SatelliteAdapter.load: no parseable CDH data from {path!r}"
            )

        cdh, stuck = _clean_cdh(cdh_raw)
        self._stuck_sensors = stuck

        adcs = pd.DataFrame()
        if adcs_raw is not None and not adcs_raw.empty:
            adcs = _clean_adcs(adcs_raw)

        self.cdh_channels = [c for c in cdh.columns if c not in {"timestamp", "elapsed_s"}]
        self.adcs_channels = (
            [c for c in adcs.columns if c not in {"timestamp", "elapsed_s"}]
            if not adcs.empty else []
        )

        if not adcs.empty:
            adcs_cols_only = [c for c in adcs.columns if c != "elapsed_s"]
            combined = pd.merge_asof(
                cdh.sort_values("timestamp"),
                adcs[adcs_cols_only].sort_values("timestamp"),
                on="timestamp",
                tolerance=pd.Timedelta("2s"),
                direction="nearest",
            )
            for col in self.adcs_channels:
                if col in combined.columns:
                    combined[col] = combined[col].ffill().bfill()
        else:
            combined = cdh.copy()

        combined = combined.sort_values("timestamp").reset_index(drop=True)
        t0 = combined["timestamp"].iloc[0]
        combined["elapsed_s"] = (combined["timestamp"] - t0).dt.total_seconds()

        self.channels = [c for c in combined.columns if c not in {"timestamp", "elapsed_s"}]
        return combined[["timestamp", "elapsed_s"] + self.channels].copy()

    def split(self, combined_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split a combined DataFrame back into separate CDH and ADCS DataFrames.
        """
        base = ["timestamp", "elapsed_s"]
        cdh_cols  = base + [c for c in self.cdh_channels  if c in combined_df.columns]
        adcs_cols = base + [c for c in self.adcs_channels if c in combined_df.columns]
        cdh  = combined_df[cdh_cols].copy()  if self.cdh_channels  else pd.DataFrame()
        adcs = combined_df[adcs_cols].copy() if self.adcs_channels else pd.DataFrame()
        return cdh, adcs

    def get_stuck_sensors(self) -> list[str]:
        return getattr(self, "_stuck_sensors", [])

    # ----------------------------------------------------------------- M2 --

    def get_quality_config(self) -> dict:
        return {
            "stuck_unique_max": 3,
            "zscore_threshold": 3.0,
            "expected_dt_s":    1.0,
            "gap_multiplier":   5.0,
            "trend_channels":   ["TEMP_OBDH", "TEMP_EPS", "TEMP_BATTERY"],
        }

    # ----------------------------------------------------------------- M3 --

    def get_feature_config(self) -> dict:
        return {
            "channels": None,   # use all channels discovered by load()
            "window":   15,     # ~15 s at 1 Hz
        }

    # ----------------------------------------------------------------- M4 --

    def get_ontology_path(self) -> str:
        return _ONTOLOGY_PATH

    def get_ontology_iri(self) -> str:
        return "https://sensorwf.org/ontologies/satellite"

    def get_class_map(self) -> dict[str, str]:
        _BASE = "https://sensorwf.org/ontologies/satellite#"
        return {
            "ACCEL_":    f"{_BASE}AttitudeDeterminationAndControlSubsystem",
            "GYRO_":     f"{_BASE}AttitudeDeterminationAndControlSubsystem",
            "MAG_":      f"{_BASE}AttitudeDeterminationAndControlSubsystem",
            "WHEEL_":    f"{_BASE}AttitudeDeterminationAndControlSubsystem",
            "SUN_":      f"{_BASE}AttitudeDeterminationAndControlSubsystem",
            "SYS_V":     f"{_BASE}ElectricalPowerSubsystem",
            "SYS_I":     f"{_BASE}ElectricalPowerSubsystem",
            "PLAT_5V":   f"{_BASE}ElectricalPowerSubsystem",
            "OBC_PWR":   f"{_BASE}ElectricalPowerSubsystem",
            "COMMS_PWR": f"{_BASE}ElectricalPowerSubsystem",
            "BATT_":     f"{_BASE}ElectricalPowerSubsystem",
            "SOLAR_":    f"{_BASE}ElectricalPowerSubsystem",
            "EXT_":      f"{_BASE}ElectricalPowerSubsystem",
            "PWR_":      f"{_BASE}ElectricalPowerSubsystem",
            "TEMP_SOLAR":     f"{_BASE}ThermalControlSubsystem",
            "TEMP_EPS":       f"{_BASE}ThermalControlSubsystem",
            "TEMP_BATTERY":   f"{_BASE}ThermalControlSubsystem",
            "TEMP_ADCS":      f"{_BASE}ThermalControlSubsystem",
            "TEMP_WHEEL":     f"{_BASE}ThermalControlSubsystem",
            "TEMP_COMMS":     f"{_BASE}ThermalControlSubsystem",
            "TEMP_PAYLOAD":   f"{_BASE}ThermalControlSubsystem",
            "THERMISTOR_":    f"{_BASE}ThermalControlSubsystem",
            "TEMP_OBDH":      f"{_BASE}OnBoardDataHandling",
            "TEMP_BACKPLANE": f"{_BASE}OnBoardDataHandling",
            "CMD_":           f"{_BASE}OnBoardDataHandling",
            "FSW_":           f"{_BASE}OnBoardDataHandling",
            "COMMS_":    f"{_BASE}CommunicationsSubsystem",
            "RSSI":      f"{_BASE}CommunicationsSubsystem",
            "FRAME_ERR": f"{_BASE}CommunicationsSubsystem",
            "FREQ":      f"{_BASE}CommunicationsSubsystem",
            "TX_":       f"{_BASE}CommunicationsSubsystem",
        }

    def get_plot_groups(self) -> dict[str, list[str]]:
        return {
            "Board Temperatures": [
                "TEMP_SOLAR", "TEMP_EPS", "TEMP_BATTERY", "TEMP_BACKPLANE",
                "TEMP_OBDH", "TEMP_ADCS", "TEMP_WHEEL", "TEMP_COMMS", "TEMP_PAYLOAD",
            ],
            "Power Rails (Voltage)": [
                "SYS_V", "PLAT_5V_V", "OBC_PWR_V", "COMMS_PWR_V",
                "SOLAR_V", "EXT_V", "BATT_CHR_V",
            ],
            "Current Draw": [
                "SYS_I", "OBC_PWR_I", "COMMS_PWR_I", "PLAT_5V_I",
                "ADCS_5V_I", "BATT_CHR_I",
            ],
            "ADCS — IMU": [
                "ACCEL_X", "ACCEL_Y", "ACCEL_Z",
                "GYRO_X",  "GYRO_Y",  "GYRO_Z",
                "MAG_X",   "MAG_Y",   "MAG_Z",
            ],
            "ADCS — Wheel & Sun": [
                "WHEEL_SPEED",
                "SUN_SENSOR_1a", "SUN_SENSOR_1b",
                "SUN_SENSOR_2a", "SUN_SENSOR_2b",
                "SUN_ANGLE",
            ],
        }

    # ----------------------------------------------------------------- E1 --

    def get_fault_types(self) -> list[dict]:
        return [
            {"tag": "T1_obdh_runaway",             "description": "OBDH thermal runaway (sustained ramp)",         "target": "cdh"},
            {"tag": "T2_board_thermal_shock",       "description": "Board thermal shock (rapid spike + recovery)",  "target": "cdh"},
            {"tag": "T3_thermal_bias_divergence",   "description": "Thermal bias divergence across sensors",        "target": "cdh"},
            {"tag": "P1_bus_sag_recovery",          "description": "Bus voltage sag and partial recovery",          "target": "cdh"},
            {"tag": "P2_switching_noise_burst",      "description": "Switching noise burst on power rails",          "target": "cdh"},
            {"tag": "P3_rail_latchup",               "description": "Rail latchup (sudden drop to near-zero)",       "target": "cdh"},
            {"tag": "C1_packet_gap_jitter",         "description": "Packet timing jitter / gap anomaly",            "target": "cdh"},
            {"tag": "C2_rssi_fade",                 "description": "RSSI signal fade",                              "target": "cdh"},
            {"tag": "C3_frame_error_avalanche",     "description": "Frame error count avalanche",                   "target": "cdh"},
            {"tag": "A1_accel_packet_dropout",      "description": "Accelerometer packet dropout (zeros burst)",    "target": "adcs"},
            {"tag": "A2_gyro_clipping",             "description": "Gyro output clipping at ADC rail",              "target": "adcs"},
            {"tag": "A3_mag_field_inversion",       "description": "Magnetometer polarity inversion",               "target": "adcs"},
            {"tag": "A4_imu_correlation_break",     "description": "IMU cross-sensor correlation breakdown",        "target": "adcs"},
            {"tag": "A5_sun_sensor_blinding",       "description": "Sun sensor blinding (step saturation)",         "target": "adcs"},
            {"tag": "W1_wheel_runaway",             "description": "Reaction wheel runaway (unbounded spin-up)",    "target": "adcs"},
            {"tag": "W2_wheel_stiction_stop",       "description": "Reaction wheel stiction / stop",                "target": "adcs"},
            {"tag": "COMPOUND_thermal_power_fault", "description": "Simultaneous thermal + power rail fault",       "target": "cdh"},
        ]
