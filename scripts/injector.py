"""
injector.py — Synthetic anomaly injection for SLL telemetry.

Loads clean CSV telemetry (cdh_clean.csv / adcs_clean.csv produced by the
pipeline), injects synthetic anomalies, and exports labelled copies.

Each injection function signature:
    inject_<name>(df, rng, **kwargs) -> tuple[pd.DataFrame, dict]

The returned dict (label_record) contains:
    anomaly_type      : str
    start_idx         : int
    end_idx           : int
    severity          : str   ("low" | "medium" | "high")
    affected_channels : list[str]

The returned DataFrame always has:
    anomaly       : bool   (True for injected rows)
    anomaly_type  : str    (empty string for normal rows)
"""

from __future__ import annotations

import ast
import json
import os
import math
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CDH_TEMP_COLS = [
    "TEMP_SOLAR", "TEMP_EPS", "TEMP_BATTERY", "TEMP_BACKPLANE",
    "TEMP_OBDH", "TEMP_ADCS", "TEMP_WHEEL", "TEMP_COMMS", "TEMP_PAYLOAD",
]

_CDH_BOARD_TEMP_COLS = [
    c for c in _CDH_TEMP_COLS
    if c != "TEMP_OBDH" and not c.endswith("_L")
]

_GYRO_COLS  = ["GYRO_X", "GYRO_Y", "GYRO_Z"]
_ACCEL_COLS = ["ACCEL_X", "ACCEL_Y", "ACCEL_Z"]
_MAG_COLS   = ["MAG_X", "MAG_Y", "MAG_Z"]
_IMU_COLS   = _ACCEL_COLS + _GYRO_COLS + _MAG_COLS

_SUN_SENSOR_PREFIXES = ["SUN_SENSOR_"]


def _ensure_label_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add anomaly / anomaly_type columns if they do not already exist."""
    if "anomaly" not in df.columns:
        df["anomaly"] = False
    if "anomaly_type" not in df.columns:
        df["anomaly_type"] = ""
    return df


def _mark_rows(df: pd.DataFrame, start: int, end: int, atype: str) -> pd.DataFrame:
    """Mark rows [start, end) as anomalous."""
    df = _ensure_label_cols(df)
    df.loc[start:end - 1, "anomaly"] = True
    df.loc[start:end - 1, "anomaly_type"] = atype
    return df


def _mid_split(df: pd.DataFrame, rng: np.random.Generator) -> int:
    """Return a random row index in the middle third of the DataFrame."""
    n = len(df)
    lo = n // 3
    hi = (2 * n) // 3
    if lo >= hi:
        lo, hi = 1, max(2, n - 1)
    return int(rng.integers(lo, hi))


def _window(df: pd.DataFrame, frac: float, rng: np.random.Generator) -> tuple[int, int]:
    """
    Pick a random contiguous window of length ~ frac * len(df).
    Returns (start_row_idx, end_row_idx) where end is exclusive.
    """
    n = len(df)
    length = max(2, int(round(frac * n)))
    length = min(length, n)
    start = int(rng.integers(0, n - length + 1))
    return start, start + length


# ---------------------------------------------------------------------------
# ── Thermal injections (CDH) ───────────────────────────────────────────────
# ---------------------------------------------------------------------------

def inject_obdh_ramp_accel(
    df: pd.DataFrame,
    rng: np.random.Generator,
    factor: float = 4.0,
) -> tuple[pd.DataFrame, dict]:
    """T1: Accelerate TEMP_OBDH slope by *factor* from a random mid-session row."""
    df = df.copy()
    col = "TEMP_OBDH"
    if col not in df.columns:
        df = _ensure_label_cols(df)
        return df, {}

    n = len(df)
    split = _mid_split(df, rng)

    # Estimate normal slope from rows before split
    elapsed = df["elapsed_s"].values
    temps   = df[col].values

    if split < 2:
        split = 2

    # Linear fit on pre-split segment to get slope (°C/s)
    t_pre = elapsed[:split]
    y_pre = temps[:split]
    if len(t_pre) >= 2:
        coeffs = np.polyfit(t_pre, y_pre, 1)
        slope_per_s = coeffs[0]
    else:
        slope_per_s = 0.0

    last_normal  = temps[split - 1]
    elapsed_split = elapsed[split - 1]

    new_temps = temps.copy().astype(float)
    for i in range(split, n):
        dt = elapsed[i] - elapsed_split
        new_temps[i] = last_normal + dt * slope_per_s * factor

    df[col] = new_temps
    df = _mark_rows(df, split, n, "T1_obdh_ramp_accel")

    severity = "high" if factor >= 4.0 else "medium"
    label = {
        "anomaly_type":      "T1_obdh_ramp_accel",
        "start_idx":         split,
        "end_idx":           n - 1,
        "severity":          severity,
        "affected_channels": [col],
    }
    return df, label


def inject_board_step(
    df: pd.DataFrame,
    rng: np.random.Generator,
    col: Optional[str] = None,
    delta: float = 8.0,
) -> tuple[pd.DataFrame, dict]:
    """T2: Step + exponential decay in a random CDH temp column (not OBDH, not _L)."""
    df = df.copy()

    candidates = [
        c for c in _CDH_BOARD_TEMP_COLS
        if c in df.columns and not c.endswith("_L")
    ]
    if not candidates:
        df = _ensure_label_cols(df)
        return df, {}

    if col is None or col not in df.columns:
        col = str(rng.choice(candidates))

    n    = len(df)
    step = _mid_split(df, rng)
    tau  = 30  # samples

    vals = df[col].values.copy().astype(float)
    for i in range(step, n):
        decay      = math.exp(-(i - step) / tau)
        vals[i]   += delta * decay

    df[col] = vals
    df = _mark_rows(df, step, n, "T2_board_step")

    if abs(delta) >= 10.0:
        severity = "high"
    elif abs(delta) >= 5.0:
        severity = "medium"
    else:
        severity = "low"

    label = {
        "anomaly_type":      "T2_board_step",
        "start_idx":         step,
        "end_idx":           n - 1,
        "severity":          severity,
        "affected_channels": [col],
    }
    return df, label


def inject_board_divergence(
    df: pd.DataFrame,
    rng: np.random.Generator,
    col_a: str = "TEMP_EPS",
    col_b: str = "TEMP_BATTERY",
    bias: float = 5.0,
) -> tuple[pd.DataFrame, dict]:
    """T3: Growing linear bias on col_a from a random mid-session row; col_b unchanged."""
    df = df.copy()

    if col_a not in df.columns:
        df = _ensure_label_cols(df)
        return df, {}

    n     = len(df)
    split = _mid_split(df, rng)
    window_len = n - split

    vals = df[col_a].values.copy().astype(float)
    for i in range(split, n):
        frac     = (i - split) / max(window_len - 1, 1)
        vals[i] += bias * frac

    df[col_a] = vals
    df = _mark_rows(df, split, n, "T3_board_divergence")

    severity = "high" if bias >= 8.0 else ("medium" if bias >= 4.0 else "low")
    channels = [col_a]
    if col_b in df.columns:
        channels.append(col_b)

    label = {
        "anomaly_type":      "T3_board_divergence",
        "start_idx":         split,
        "end_idx":           n - 1,
        "severity":          severity,
        "affected_channels": channels,
    }
    return df, label


# ---------------------------------------------------------------------------
# ── Power injections (CDH) ────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def inject_sysv_sag(
    df: pd.DataFrame,
    rng: np.random.Generator,
    drop: float = 1.2,
    duration_frac: float = 0.15,
) -> tuple[pd.DataFrame, dict]:
    """P1: Gaussian-shaped dip in SYS_V; also dip BATT_CHR_V by drop*0.3."""
    df = df.copy()

    if "SYS_V" not in df.columns:
        df = _ensure_label_cols(df)
        return df, {}

    n = len(df)
    width = max(4, int(round(duration_frac * n)))
    centre = int(rng.integers(width // 2, n - width // 2))

    idxs = np.arange(n)
    sigma = width / 4.0
    gaussian = np.exp(-0.5 * ((idxs - centre) / sigma) ** 2)

    df["SYS_V"] = df["SYS_V"].values.astype(float) - drop * gaussian

    if "BATT_CHR_V" in df.columns:
        df["BATT_CHR_V"] = df["BATT_CHR_V"].values.astype(float) - drop * 0.3 * gaussian

    start = max(0, centre - width // 2)
    end   = min(n, centre + width // 2)
    df = _mark_rows(df, start, end, "P1_sysv_sag")

    severity = "high" if drop >= 1.5 else ("medium" if drop >= 0.8 else "low")
    channels = ["SYS_V"]
    if "BATT_CHR_V" in df.columns:
        channels.append("BATT_CHR_V")

    label = {
        "anomaly_type":      "P1_sysv_sag",
        "start_idx":         start,
        "end_idx":           end - 1,
        "severity":          severity,
        "affected_channels": channels,
    }
    return df, label


def inject_obc_voltage_spike(
    df: pd.DataFrame,
    rng: np.random.Generator,
    n_spikes: int = 8,
    amplitude: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    """P2: Inject n_spikes random OBC_PWR_V spikes within a 20-sample window."""
    df = df.copy()

    col = "OBC_PWR_V"
    if col not in df.columns:
        df = _ensure_label_cols(df)
        return df, {}

    n = len(df)
    window_size = min(20, n)
    win_start = int(rng.integers(0, n - window_size + 1))
    win_end   = win_start + window_size

    spike_positions = rng.integers(win_start, win_end, size=n_spikes)

    vals = df[col].values.copy().astype(float)
    for pos in spike_positions:
        sign = rng.choice([-1, 1])
        vals[pos] += sign * amplitude

    df[col] = vals
    df = _mark_rows(df, win_start, win_end, "P2_obc_voltage_spike")

    severity = "high" if amplitude >= 1.0 else ("medium" if amplitude >= 0.3 else "low")
    label = {
        "anomaly_type":      "P2_obc_voltage_spike",
        "start_idx":         win_start,
        "end_idx":           win_end - 1,
        "severity":          severity,
        "affected_channels": [col],
    }
    return df, label


def inject_rail_stuck_high(
    df: pd.DataFrame,
    rng: np.random.Generator,
    col: str = "ADCS_5V_V",
    value: float = 3.3,
) -> tuple[pd.DataFrame, dict]:
    """P3: Force col to value from a random row for the rest of the session."""
    df = df.copy()

    if col not in df.columns:
        df = _ensure_label_cols(df)
        return df, {}

    n     = len(df)
    start = _mid_split(df, rng)

    df.loc[start:, col] = value
    df = _mark_rows(df, start, n, "P3_rail_stuck_high")

    label = {
        "anomaly_type":      "P3_rail_stuck_high",
        "start_idx":         start,
        "end_idx":           n - 1,
        "severity":          "high",
        "affected_channels": [col],
    }
    return df, label


# ---------------------------------------------------------------------------
# ── ADCS injections ───────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def inject_accel_dropout(
    df: pd.DataFrame,
    rng: np.random.Generator,
    axes: Optional[list] = None,
    duration_frac: float = 0.2,
) -> tuple[pd.DataFrame, dict]:
    """A1: Zero out ACCEL_X/Y/Z (or subset) for a contiguous window."""
    df = df.copy()

    if axes is None:
        axes = _ACCEL_COLS

    available = [c for c in axes if c in df.columns]
    if not available:
        df = _ensure_label_cols(df)
        return df, {}

    start, end = _window(df, duration_frac, rng)
    for col in available:
        df.loc[start:end - 1, col] = 0.0

    df = _mark_rows(df, start, end, "A1_accel_dropout")

    label = {
        "anomaly_type":      "A1_accel_dropout",
        "start_idx":         start,
        "end_idx":           end - 1,
        "severity":          "medium",
        "affected_channels": available,
    }
    return df, label


def inject_gyro_saturation(
    df: pd.DataFrame,
    rng: np.random.Generator,
    col: Optional[str] = None,
    sat_value: Optional[float] = None,
    duration_frac: float = 0.2,
) -> tuple[pd.DataFrame, dict]:
    """A2: Peg a random GYRO_* axis at its observed max for a window."""
    df = df.copy()

    available = [c for c in _GYRO_COLS if c in df.columns]
    if not available:
        df = _ensure_label_cols(df)
        return df, {}

    if col is None or col not in df.columns:
        col = str(rng.choice(available))

    if sat_value is None:
        sat_value = float(df[col].abs().max())
        if sat_value == 0.0:
            sat_value = 100.0  # fallback if all-zero column

    start, end = _window(df, duration_frac, rng)
    df.loc[start:end - 1, col] = sat_value

    df = _mark_rows(df, start, end, "A2_gyro_saturation")

    label = {
        "anomaly_type":      "A2_gyro_saturation",
        "start_idx":         start,
        "end_idx":           end - 1,
        "severity":          "high",
        "affected_channels": [col],
    }
    return df, label


def inject_mag_flip(
    df: pd.DataFrame,
    rng: np.random.Generator,
    duration_frac: float = 0.1,
) -> tuple[pd.DataFrame, dict]:
    """A3: Invert sign of all MAG_X/Y/Z for a short window."""
    df = df.copy()

    available = [c for c in _MAG_COLS if c in df.columns]
    if not available:
        df = _ensure_label_cols(df)
        return df, {}

    start, end = _window(df, duration_frac, rng)
    for col in available:
        df.loc[start:end - 1, col] = -df.loc[start:end - 1, col]

    df = _mark_rows(df, start, end, "A3_mag_flip")

    label = {
        "anomaly_type":      "A3_mag_flip",
        "start_idx":         start,
        "end_idx":           end - 1,
        "severity":          "high",
        "affected_channels": available,
    }
    return df, label


def inject_imu_allzero(
    df: pd.DataFrame,
    rng: np.random.Generator,
    duration_frac: float = 0.25,
) -> tuple[pd.DataFrame, dict]:
    """A4: Zero all 9 IMU axes for a contiguous window."""
    df = df.copy()

    available = [c for c in _IMU_COLS if c in df.columns]
    if not available:
        df = _ensure_label_cols(df)
        return df, {}

    start, end = _window(df, duration_frac, rng)
    for col in available:
        df.loc[start:end - 1, col] = 0.0

    df = _mark_rows(df, start, end, "A4_imu_allzero")

    label = {
        "anomaly_type":      "A4_imu_allzero",
        "start_idx":         start,
        "end_idx":           end - 1,
        "severity":          "high",
        "affected_channels": available,
    }
    return df, label


def inject_sun_saturation(
    df: pd.DataFrame,
    rng: np.random.Generator,
    duration_frac: float = 0.1,
) -> tuple[pd.DataFrame, dict]:
    """A5: Set all SUN_SENSOR_* channels to 1023 for a window."""
    df = df.copy()

    sun_cols = [c for c in df.columns if c.startswith("SUN_SENSOR_")]
    if not sun_cols:
        df = _ensure_label_cols(df)
        return df, {}

    start, end = _window(df, duration_frac, rng)
    for col in sun_cols:
        df.loc[start:end - 1, col] = 1023.0

    df = _mark_rows(df, start, end, "A5_sun_saturation")

    label = {
        "anomaly_type":      "A5_sun_saturation",
        "start_idx":         start,
        "end_idx":           end - 1,
        "severity":          "medium",
        "affected_channels": sun_cols,
    }
    return df, label


# ---------------------------------------------------------------------------
# ── Wheel injections (ADCS) ───────────────────────────────────────────────
# ---------------------------------------------------------------------------

def inject_wheel_overspeed(
    df: pd.DataFrame,
    rng: np.random.Generator,
    target: float = 1800.0,
) -> tuple[pd.DataFrame, dict]:
    """W1: From the point of max abs wheel speed, linearly ramp to target and hold."""
    df = df.copy()

    col = "WHEEL_SPEED"
    if col not in df.columns:
        df = _ensure_label_cols(df)
        return df, {}

    vals = df[col].values.copy().astype(float)
    n    = len(vals)

    peak_idx = int(np.argmax(np.abs(vals)))
    ramp_len = max(2, (n - peak_idx) // 3)
    ramp_end = min(n, peak_idx + ramp_len)

    start_val = vals[peak_idx]
    for i in range(peak_idx, ramp_end):
        frac     = (i - peak_idx) / max(ramp_len - 1, 1)
        vals[i]  = start_val + frac * (target - start_val)

    vals[ramp_end:] = target

    df[col] = vals
    df = _mark_rows(df, peak_idx, n, "W1_wheel_overspeed")

    label = {
        "anomaly_type":      "W1_wheel_overspeed",
        "start_idx":         peak_idx,
        "end_idx":           n - 1,
        "severity":          "high",
        "affected_channels": [col],
    }
    return df, label


def inject_wheel_sudden_stop(
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """W2: At peak wheel speed, zero WHEEL_SPEED in one sample and hold zero."""
    df = df.copy()

    col = "WHEEL_SPEED"
    if col not in df.columns:
        df = _ensure_label_cols(df)
        return df, {}

    vals = df[col].values.copy().astype(float)
    n    = len(vals)

    peak_idx = int(np.argmax(np.abs(vals)))
    stop_idx = min(peak_idx + 1, n - 1)

    vals[stop_idx:] = 0.0
    df[col] = vals

    df = _mark_rows(df, stop_idx, n, "W2_wheel_sudden_stop")

    label = {
        "anomaly_type":      "W2_wheel_sudden_stop",
        "start_idx":         stop_idx,
        "end_idx":           n - 1,
        "severity":          "high",
        "affected_channels": [col],
    }
    return df, label


# ---------------------------------------------------------------------------
# ── Timing injections (CDH) ───────────────────────────────────────────────
# ---------------------------------------------------------------------------

def inject_packet_gap(
    df: pd.DataFrame,
    rng: np.random.Generator,
    gap_s: float = 10.0,
) -> tuple[pd.DataFrame, dict]:
    """C1: Remove a contiguous block totalling gap_s seconds; recompute elapsed_s."""
    df = df.copy()

    if "elapsed_s" not in df.columns or len(df) < 4:
        df = _ensure_label_cols(df)
        return df, {}

    elapsed = df["elapsed_s"].values
    total_s = elapsed[-1] - elapsed[0]
    if total_s <= 0:
        df = _ensure_label_cols(df)
        return df, {}

    # Estimate nominal sample rate
    dt_median = float(np.median(np.diff(elapsed)))
    if dt_median <= 0:
        dt_median = 1.0
    gap_rows  = max(1, int(round(gap_s / dt_median)))

    n = len(df)
    split = _mid_split(df, rng)
    gap_end = min(split + gap_rows, n)

    # Record the gap so we can put it in the label
    start_idx = split
    end_idx   = gap_end - 1

    df_before = df.iloc[:split].copy()
    df_after  = df.iloc[gap_end:].copy()

    # Shift elapsed_s in the after segment
    if len(df_after) > 0 and len(df_before) > 0:
        time_before = df_before["elapsed_s"].iloc[-1]
        time_after_orig = df_after["elapsed_s"].iloc[0]
        shift = time_after_orig - time_before - dt_median
        df_after["elapsed_s"] = df_after["elapsed_s"] - shift

    df_out = pd.concat([df_before, df_after], ignore_index=True)
    df_out = _ensure_label_cols(df_out)
    # Mark the rows just before and just after the gap boundary
    # (the gap itself is gone — mark surrounding context)
    mark_start = max(0, split - 1)
    mark_end   = min(len(df_out), split + 1)
    df_out = _mark_rows(df_out, mark_start, mark_end, "C1_packet_gap")

    label = {
        "anomaly_type":      "C1_packet_gap",
        "start_idx":         start_idx,
        "end_idx":           end_idx,
        "severity":          "medium" if gap_s >= 5.0 else "low",
        "affected_channels": ["elapsed_s"],
    }
    return df_out, label


# ---------------------------------------------------------------------------
# ── Compound injections ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def inject_thermal_power_fault(
    df_cdh: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    """Compound: T1 (OBDH ramp accel) + P1 (SYS_V sag) on CDH df."""
    df, label_t1 = inject_obdh_ramp_accel(df_cdh, rng)
    df, label_p1 = inject_sysv_sag(df, rng)

    combined_label = {
        "anomaly_type":      "COMPOUND_thermal_power_fault",
        "start_idx":         min(
            label_t1.get("start_idx", 0) if label_t1 else 0,
            label_p1.get("start_idx", 0) if label_p1 else 0,
        ),
        "end_idx": max(
            label_t1.get("end_idx", 0) if label_t1 else 0,
            label_p1.get("end_idx", 0) if label_p1 else 0,
        ),
        "severity":          "high",
        "affected_channels": list({
            ch
            for lbl in [label_t1, label_p1]
            if lbl
            for ch in lbl.get("affected_channels", [])
        }),
    }
    return df, combined_label


def inject_adcs_power_fault(
    df_cdh: pd.DataFrame,
    df_adcs: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Compound: A4 (IMU all-zero on ADCS) + P3 (rail stuck high on CDH)."""
    df_adcs_out, label_a4 = inject_imu_allzero(df_adcs, rng)
    df_cdh_out,  label_p3 = inject_rail_stuck_high(df_cdh, rng)

    combined_label = {
        "anomaly_type":      "COMPOUND_adcs_power_fault",
        "start_idx":         min(
            label_a4.get("start_idx", 0) if label_a4 else 0,
            label_p3.get("start_idx", 0) if label_p3 else 0,
        ),
        "end_idx": max(
            label_a4.get("end_idx", 0) if label_a4 else 0,
            label_p3.get("end_idx", 0) if label_p3 else 0,
        ),
        "severity":          "high",
        "affected_channels": list({
            ch
            for lbl in [label_a4, label_p3]
            if lbl
            for ch in lbl.get("affected_channels", [])
        }),
    }
    return df_cdh_out, df_adcs_out, combined_label


# ---------------------------------------------------------------------------
# ── Registry of single-df injection types ────────────────────────────────
# ---------------------------------------------------------------------------

# Each entry: (tag, target, fn)
#   target = "cdh" | "adcs"
#   fn     = callable(df, rng) -> (df, label)

_SINGLE_INJECTIONS: list[tuple[str, str, object]] = [
    # Thermal
    ("T1_obdh_ramp_accel",   "cdh",  inject_obdh_ramp_accel),
    ("T2_board_step",        "cdh",  inject_board_step),
    ("T3_board_divergence",  "cdh",  inject_board_divergence),
    # Power
    ("P1_sysv_sag",          "cdh",  inject_sysv_sag),
    ("P2_obc_voltage_spike", "cdh",  inject_obc_voltage_spike),
    ("P3_rail_stuck_high",   "cdh",  inject_rail_stuck_high),
    # ADCS
    ("A1_accel_dropout",     "adcs", inject_accel_dropout),
    ("A2_gyro_saturation",   "adcs", inject_gyro_saturation),
    ("A3_mag_flip",          "adcs", inject_mag_flip),
    ("A4_imu_allzero",       "adcs", inject_imu_allzero),
    ("A5_sun_saturation",    "adcs", inject_sun_saturation),
    # Wheel
    ("W1_wheel_overspeed",   "adcs", inject_wheel_overspeed),
    ("W2_wheel_sudden_stop", "adcs", inject_wheel_sudden_stop),
    # Timing
    ("C1_packet_gap",        "cdh",  inject_packet_gap),
]

# Compound injection tags handled separately
_COMPOUND_TAGS = ["COMPOUND_thermal_power_fault", "COMPOUND_adcs_power_fault"]


# ---------------------------------------------------------------------------
# ── Detection evaluation ──────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _parse_channel_list(raw) -> list[str]:
    """Parse affected_channels from a CSV-loaded value (string or list)."""
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str):
        return []
    raw = raw.strip()
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    # fallback: comma-split, strip quotes/brackets
    return [c.strip().strip("'\"[]") for c in raw.split(",") if c.strip().strip("'\"[]")]


def evaluate_injections(
    injected_dir: str,
    files_generated: list[tuple[str, str, int]],
    cdh_clean: pd.DataFrame,
    adcs_clean: pd.DataFrame,
    baseline_report: dict,
) -> list[dict]:
    """
    For each generated injection variant, reload the injected CSV and its label,
    re-run all four detectors, and check whether any detector flagged the
    injected anomaly window.

    Detection rules
    ---------------
    D1 — thermal trend  : injected OBDH slope > 1.5× baseline OR absolute slope
                          > 3.0 °C/min (catches accelerated ramps)
    D2 — z-score        : any z-score flag in the [start_idx, end_idx] window
                          across all monitored CDH channels
    D3 — correlation    : rolling Pearson breakdown in the injected window
                          (board-temp pairs + wheel↔gyro when active)
    D4 — ADCS stuck     : stuck-channel list non-empty OR adcs_all_zero True
    D5 — timing         : CDH max_gap_s > 2× baseline gap OR > 5 s absolute

    Returns a list of result dicts, one per (tag, variant) pair.
    """
    from scripts.detector import run_detection  # local import avoids circular dep

    results: list[dict] = []

    baseline_obdh_slope  = baseline_report.get("obdh_trend",  {}).get("slope_per_min", 0.0)
    baseline_timing_max  = baseline_report.get("cdh_timing",  {}).get("max_gap_s",    0.0)

    for tag, target, v in files_generated:
        label_path = os.path.join(injected_dir, f"labels_{tag}_v{v}.csv")
        if not os.path.exists(label_path):
            continue

        label_row  = pd.read_csv(label_path).iloc[0].to_dict()
        start_idx  = int(label_row.get("start_idx", 0))
        end_idx    = int(label_row.get("end_idx",   0))
        affected   = _parse_channel_list(label_row.get("affected_channels", []))

        # Load injected CDH (present for every type)
        cdh_path  = os.path.join(injected_dir, f"cdh_injected_{tag}_v{v}.csv")
        adcs_path = os.path.join(injected_dir, f"adcs_injected_{tag}_v{v}.csv")

        cdh_inj  = pd.read_csv(cdh_path)  if os.path.exists(cdh_path)  else cdh_clean.copy()
        adcs_inj = pd.read_csv(adcs_path) if os.path.exists(adcs_path) else adcs_clean.copy()

        # Restore datetime dtype lost during CSV round-trip
        for df_tmp in (cdh_inj, adcs_inj):
            if "timestamp" in df_tmp.columns:
                df_tmp["timestamp"] = pd.to_datetime(df_tmp["timestamp"])

        try:
            inj_report = run_detection(cdh_inj, adcs_inj, stuck_sensors=[])
        except Exception as exc:
            results.append({
                "tag": tag, "variant": v, "target": target,
                "detected": False, "error": str(exc),
                "D1_trend": False, "D2_zscore": False,
                "D3_correlation": False, "D4_stuck": False, "D5_timing": False,
            })
            continue

        # ── D2: z-score flag inside injected window ──────────────────────
        d2 = False
        for col, mask in inj_report.get("zscore_masks", {}).items():
            try:
                if mask.iloc[start_idx : end_idx + 1].any():
                    d2 = True
                    break
            except Exception:
                pass

        # ── D1: OBDH slope ratio vs baseline ────────────────────────────
        inj_slope   = float(inj_report.get("obdh_trend", {}).get("slope_per_min", 0.0))
        if baseline_obdh_slope != 0.0:
            slope_ratio = inj_slope / baseline_obdh_slope
        else:
            slope_ratio = abs(inj_slope)
        d1 = bool(slope_ratio > 1.5 or inj_slope > 3.0)

        # ── D3: correlation breakdown inside injected window ─────────────
        d3 = False
        for corr_dict in (
            inj_report.get("correlation_breakdown", {}),
            inj_report.get("wheel_gyro_correlation", {}),
        ):
            for _, corr_mask in corr_dict.items():
                try:
                    if corr_mask.iloc[start_idx : end_idx + 1].any():
                        d3 = True
                        break
                except Exception:
                    pass
            if d3:
                break

        # ── D4: ADCS stuck / all-zero ────────────────────────────────────
        d4 = bool(inj_report.get("adcs_stuck_channels")) or bool(
            inj_report.get("adcs_all_zero", False)
        )

        # ── D5: CDH timing gap ───────────────────────────────────────────
        inj_max_gap = inj_report.get("cdh_timing", {}).get("max_gap_s", 0.0)
        d5 = inj_max_gap > max(baseline_timing_max * 2.0, 5.0)

        detected = bool(d1 or d2 or d3 or d4 or d5)

        results.append({
            "tag":            tag,
            "variant":        v,
            "target":         target,
            "start_idx":      start_idx,
            "end_idx":        end_idx,
            "detected":       detected,
            "D1_trend":       bool(d1),
            "D2_zscore":      bool(d2),
            "D3_correlation": bool(d3),
            "D4_stuck":       bool(d4),
            "D5_timing":      bool(d5),
        })

    return results


def _print_accuracy_table(results: list[dict]) -> None:
    """Print per-type and overall detection accuracy."""
    if not results:
        return

    # Per-type aggregation
    from collections import defaultdict
    by_type: dict = defaultdict(lambda: {"total": 0, "detected": 0})
    det_counts = {"D1_trend": 0, "D2_zscore": 0, "D3_correlation": 0,
                  "D4_stuck": 0, "D5_timing": 0}

    for r in results:
        if "error" in r:
            continue
        t = r["tag"]
        by_type[t]["total"]    += 1
        by_type[t]["detected"] += int(r["detected"])
        for det in det_counts:
            det_counts[det] += int(r.get(det, False))

    total   = sum(v["total"]    for v in by_type.values())
    correct = sum(v["detected"] for v in by_type.values())
    overall = correct / total if total else 0.0

    print("\n" + "=" * 70)
    print("Injection detection accuracy")
    print("=" * 70)
    print(f"  {'Type':<40} {'Detected':>10}  {'Rate':>6}")
    print("  " + "-" * 58)
    for tag in sorted(by_type):
        info = by_type[tag]
        rate = info["detected"] / info["total"] if info["total"] else 0.0
        print(f"  {tag:<40} {info['detected']:>4}/{info['total']:<4}  {rate:>5.0%}")
    print("  " + "-" * 58)
    print(f"  {'OVERALL':<40} {correct:>4}/{total:<4}  {overall:>5.0%}")
    print("-" * 70)
    print("  Detector breakdown (variants caught by each detector):")
    for det, cnt in det_counts.items():
        print(f"    {det:<20}: {cnt}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# ── Top-level entry point ─────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def run_injections(
    cdh: pd.DataFrame,
    adcs: pd.DataFrame,
    output_dir: str,
    seed: int = 42,
    n_variants: int = 3,
    baseline_report: Optional[dict] = None,
) -> None:
    """
    For each injection type generate n_variants labelled variants plus
    n_variants clean (non-anomalous) evaluation copies.  Then re-runs
    all four detectors against each variant and computes detection accuracy.

    Exports:
        injected/cdh_injected_<type>_v<n>.csv  (anomalous data)
        injected/adcs_injected_<type>_v<n>.csv
        injected/labels_<type>_v<n>.csv         (per-injection label record)
        injected/cdh_eval_clean_v<n>.csv        (clean labelled evaluation copies)
        injected/adcs_eval_clean_v<n>.csv
        injected/injection_summary.csv          (one row per variant, with detection result)
        injected/injection_eval.json            (full per-variant evaluation detail)

    Prints injection and accuracy tables at the end.
    """
    injected_dir = os.path.join(output_dir, "injected")
    os.makedirs(injected_dir, exist_ok=True)

    files_generated: list[tuple[str, str, int]] = []  # (type_tag, target, variant)

    base_rng = np.random.default_rng(seed)

    # ── Single-df injections ──────────────────────────────────────────────
    for tag, target, fn in _SINGLE_INJECTIONS:
        for v in range(1, n_variants + 1):
            variant_seed = int(base_rng.integers(0, 2**31))
            rng = np.random.default_rng(variant_seed)

            source_df = cdh if target == "cdh" else adcs

            try:
                out_df, label = fn(source_df, rng)
            except Exception as exc:
                print(f"  [WARN] {tag} v{v} failed: {exc}")
                continue

            if not label:
                continue

            # Data file
            data_fname = f"{target}_injected_{tag}_v{v}.csv"
            data_path  = os.path.join(injected_dir, data_fname)
            out_df.to_csv(data_path, index=False)

            # Label file
            label_df   = pd.DataFrame([label])
            label_fname = f"labels_{tag}_v{v}.csv"
            label_path  = os.path.join(injected_dir, label_fname)
            label_df.to_csv(label_path, index=False)

            files_generated.append((tag, target, v))

    # ── Compound: thermal_power_fault (CDH only) ──────────────────────────
    tag = "COMPOUND_thermal_power_fault"
    for v in range(1, n_variants + 1):
        variant_seed = int(base_rng.integers(0, 2**31))
        rng = np.random.default_rng(variant_seed)
        try:
            out_df, label = inject_thermal_power_fault(cdh, rng)
        except Exception as exc:
            print(f"  [WARN] {tag} v{v} failed: {exc}")
            continue

        data_fname  = f"cdh_injected_{tag}_v{v}.csv"
        label_fname = f"labels_{tag}_v{v}.csv"
        out_df.to_csv(os.path.join(injected_dir, data_fname), index=False)
        pd.DataFrame([label]).to_csv(os.path.join(injected_dir, label_fname), index=False)
        files_generated.append((tag, "cdh", v))

    # ── Compound: adcs_power_fault (CDH + ADCS) ──────────────────────────
    tag = "COMPOUND_adcs_power_fault"
    for v in range(1, n_variants + 1):
        variant_seed = int(base_rng.integers(0, 2**31))
        rng = np.random.default_rng(variant_seed)
        try:
            out_cdh, out_adcs, label = inject_adcs_power_fault(cdh, adcs, rng)
        except Exception as exc:
            print(f"  [WARN] {tag} v{v} failed: {exc}")
            continue

        out_cdh.to_csv(
            os.path.join(injected_dir, f"cdh_injected_{tag}_v{v}.csv"), index=False
        )
        out_adcs.to_csv(
            os.path.join(injected_dir, f"adcs_injected_{tag}_v{v}.csv"), index=False
        )
        pd.DataFrame([label]).to_csv(
            os.path.join(injected_dir, f"labels_{tag}_v{v}.csv"), index=False
        )
        files_generated.append((tag, "cdh+adcs", v))

    # ── Clean evaluation copies ───────────────────────────────────────────
    cdh_clean_eval  = _ensure_label_cols(cdh.copy())
    adcs_clean_eval = _ensure_label_cols(adcs.copy()) if not adcs.empty else adcs.copy()

    for v in range(1, n_variants + 1):
        cdh_clean_eval.to_csv(
            os.path.join(injected_dir, f"cdh_eval_clean_v{v}.csv"), index=False
        )
        if not adcs.empty:
            adcs_clean_eval.to_csv(
                os.path.join(injected_dir, f"adcs_eval_clean_v{v}.csv"), index=False
            )

    # ── Detection evaluation ──────────────────────────────────────────────
    eval_results: list[dict] = []
    if baseline_report is not None:
        print("        Evaluating detection accuracy …")
        eval_results = evaluate_injections(
            injected_dir,
            files_generated,
            cdh_clean=cdh,
            adcs_clean=adcs,
            baseline_report=baseline_report,
        )
        # Save full evaluation detail
        eval_path = os.path.join(injected_dir, "injection_eval.json")
        with open(eval_path, "w") as fh:
            json.dump(eval_results, fh, indent=2, default=str)
        print(f"        Eval saved     : {eval_path}")

    # ── injection_summary.csv (with detection result if available) ────────
    from collections import defaultdict

    # Build a lookup: (tag, variant) → detected bool
    eval_lookup: dict[tuple[str, int], bool] = {
        (r["tag"], r["variant"]): r.get("detected", None)
        for r in eval_results
    }

    counts: dict = defaultdict(lambda: {"target": "", "n": 0, "detected": 0})
    for tag, target, v in files_generated:
        counts[tag]["target"] = target
        counts[tag]["n"] += 1
        det = eval_lookup.get((tag, v))
        if det:
            counts[tag]["detected"] += 1

    summary_rows = []
    for t, info in sorted(counts.items()):
        row: dict = {"type": t, "target": info["target"], "variants": info["n"]}
        if eval_results:
            row["detected"] = info["detected"]
            row["detection_rate"] = (
                round(info["detected"] / info["n"], 3) if info["n"] else 0.0
            )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(injected_dir, "injection_summary.csv"), index=False)

    # ── Print injection table ─────────────────────────────────────────────
    total_data_files  = sum(2 if t == "cdh+adcs" else 1 for _, t, _ in files_generated)
    total_label_files = len(files_generated)

    print("\n" + "=" * 64)
    print(f"Injection summary — output: {injected_dir}/")
    print("=" * 64)
    print(f"  {'Type':<40} {'Target':<10} {'Variants':>8}")
    print("  " + "-" * 60)
    for row in summary_rows:
        print(f"  {row['type']:<40} {row['target']:<10} {row['variants']:>8}")
    print("  " + "-" * 60)
    print(f"  Total data files   : {total_data_files}")
    print(f"  Total label files  : {total_label_files}")
    print("=" * 64)

    # ── Print accuracy table ──────────────────────────────────────────────
    if eval_results:
        _print_accuracy_table(eval_results)
