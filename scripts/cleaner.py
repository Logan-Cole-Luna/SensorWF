"""
cleaner.py — Data cleaning and preprocessing for SLL telemetry.

Handles:
  - Timestamp validation and deduplication
  - Numeric type coercion (including unsigned & signed hex fields)
  - Elapsed-time column derivation
  - Stuck-sensor detection
  - Estimated power computation per rail
"""

import pandas as pd
import numpy as np

# Fields stored as raw unsigned hex strings in the archive
_HEX_UINT_COLS = [
    "SYS_I", "OBC_PWR_I", "COMMS_PWR_I", "PLAT_5V_I",
    "ADCS_5V_I", "PAYLOAD_5V_I", "SPR_I", "EXT_I",
    "ADCS_PROC_PWR_I", "ADCS_PER_PWR_I",
]

# Fields stored as raw signed 16-bit hex (two's complement)
_HEX_INT16_COLS = ["BATT_CHR_I"]

# ADCS Sensor packet hex fields — unsigned 16-bit
_ADCS_HEX_UINT_COLS = [
    "ADCS_MODE",
    "SUN_SENSOR_1a", "SUN_SENSOR_1b",
    "SUN_SENSOR_2a", "SUN_SENSOR_2b",
    "SUN_SENSOR_3a", "SUN_SENSOR_3b",
    "SUN_SENSOR_4a", "SUN_SENSOR_4b",
    "SUN_ANGLE",
    "KP_VALUE", "KD_VALUE",
    "ADCS_VERSION",
    "ADCS_ACC_CNT", "ADCS_REJ_CNT", "ADCS_INC_CNT",
    "MAG_DEPLOYMENT_STATUS",
]

# ADCS Sensor packet hex fields — signed 16-bit (direction-aware)
_ADCS_HEX_INT16_COLS = [
    "WHEEL_SPEED",
    "MAGNETORQUER_X", "MAGNETORQUER_Y", "MAGNETORQUER_Z",
]

# (voltage_col, current_col) pairs for power estimation
_POWER_PAIRS = [
    ("SYS_V",       "SYS_I"),
    ("PLAT_5V_V",   "PLAT_5V_I"),
    ("OBC_PWR_V",   "OBC_PWR_I"),
    ("COMMS_PWR_V", "COMMS_PWR_I"),
]

# Arbitrary EPS ADC scale factor (raw count → milliamps).
# Without the full calibration document, we use 1 mA/count as a relative proxy.
_CURRENT_SCALE = 1.0


def _hex_to_uint(val) -> float:
    """Convert a hex string or numeric to an unsigned integer."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(int(str(val), 16))
    except (ValueError, TypeError):
        return np.nan


def _hex_to_int16(val) -> float:
    """Convert a hex string to a signed 16-bit integer."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        # Already decoded (e.g. from a mixed-format Debug file)
        return float(val)
    try:
        n = int(str(val), 16)
        if n >= 0x8000:
            n -= 0x10000
        return float(n)
    except (ValueError, TypeError):
        return np.nan


def clean_cdh(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Clean a raw CDH DataFrame.

    Returns
    -------
    df_clean : pd.DataFrame
    stuck_sensors : list[str]
        Column names where every non-null value is identical (sensor fault).
    """
    df = df.copy()
    df = df.dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset="timestamp", keep="last")

    # Hex current fields — explicit float64 cast guards against ArrowDtype backends
    for col in _HEX_UINT_COLS:
        if col in df.columns:
            df[col] = np.array([_hex_to_uint(v) for v in df[col]], dtype=float) * _CURRENT_SCALE
    for col in _HEX_INT16_COLS:
        if col in df.columns:
            df[col] = np.array([_hex_to_int16(v) for v in df[col]], dtype=float) * _CURRENT_SCALE

    # Remaining columns → float
    for col in df.columns:
        if col == "timestamp":
            continue
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Elapsed time
    t0 = df["timestamp"].iloc[0]
    df["elapsed_s"] = (df["timestamp"] - t0).dt.total_seconds()

    # Estimated power per rail (mW, relative)
    for v_col, i_col in _POWER_PAIRS:
        if v_col in df.columns and i_col in df.columns:
            df[f"PWR_{v_col.replace('_V', '')}"] = df[v_col] * df[i_col]

    # Stuck sensor detection
    therm_cols = [c for c in df.columns if c.startswith("THERMISTOR_")]
    stuck_sensors = [
        c for c in therm_cols
        if c in df.columns and df[c].nunique(dropna=True) == 1
    ]
    df["_has_stuck_thermistor"] = bool(stuck_sensors)

    return df, stuck_sensors


def clean_adcs(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw ADCS Sensor DataFrame."""
    df = df.copy()
    df = df.dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset="timestamp", keep="last")

    # Decode known hex fields before generic coercion
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
