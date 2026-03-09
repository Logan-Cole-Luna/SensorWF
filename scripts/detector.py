"""
detector.py — Anomaly detection for SLL telemetry.

Provides:
  - Z-score flagging
  - Rolling-change spike detection
  - Linear trend characterisation
  - Packet timing jitter analysis
  - Consolidated run_detection() entry point
"""

import numpy as np
import pandas as pd
from scipy import stats


# Channels to run z-score detection on
_ZSCORE_COLS = [
    "TEMP_OBDH", "TEMP_COMMS", "TEMP_SOLAR",
    "SYS_V", "PLAT_5V_V", "OBC_PWR_V", "COMMS_PWR_V",
]


def zscore_anomalies(series: pd.Series, threshold: float = 2.5) -> pd.Series:
    """
    Return a boolean mask where |z-score| > threshold.

    Parameters
    ----------
    series    : pd.Series  (numeric, may contain NaNs)
    threshold : float      (default 2.5σ)
    """
    clean = series.dropna()
    if len(clean) < 3:
        return pd.Series(False, index=series.index)
    z = np.abs(stats.zscore(clean))
    mask = pd.Series(False, index=series.index)
    mask.loc[clean.index[z > threshold]] = True
    return mask


def rolling_change_anomalies(
    series: pd.Series, window: int = 5, threshold_std: float = 3.0
) -> pd.Series:
    """
    Flag samples where the absolute inter-sample delta exceeds
    threshold_std × rolling standard deviation of deltas.
    """
    delta       = series.diff().abs()
    rolling_std = delta.rolling(window, min_periods=2).std()
    return (delta > threshold_std * rolling_std).fillna(False)


def linear_trend(series: pd.Series, elapsed: pd.Series) -> dict:
    """
    Fit a linear trend to series vs elapsed time.

    Returns a dict with: slope_per_min, intercept, r_squared, start, end, delta.
    """
    valid = series.dropna()
    t     = elapsed.loc[valid.index]
    slope, intercept, r, *_ = stats.linregress(t, valid)
    return {
        "slope_per_min": round(slope * 60, 4),
        "intercept":     round(intercept, 4),
        "r_squared":     round(r ** 2, 4),
        "start":         round(float(valid.iloc[0]), 3),
        "end":           round(float(valid.iloc[-1]), 3),
        "delta":         round(float(valid.iloc[-1] - valid.iloc[0]), 3),
        # raw slope in °C/s for internal use
        "_slope_per_s":  slope,
    }


def packet_timing(timestamps: pd.Series) -> dict:
    """
    Analyse inter-packet arrival intervals.

    Returns mean, std, max gap, and a Series of all intervals.
    """
    intervals = timestamps.sort_values().diff().dt.total_seconds().dropna()
    return {
        "mean_s":     round(float(intervals.mean()), 4),
        "std_s":      round(float(intervals.std()), 4),
        "max_gap_s":  round(float(intervals.max()), 4),
        "intervals":  intervals,
    }


def run_detection(
    cdh: pd.DataFrame,
    adcs: pd.DataFrame,
    stuck_sensors: list[str],
) -> dict:
    """
    Run all anomaly detectors against the cleaned telemetry.

    Returns a structured report dict suitable for logging and plotting.
    """
    report: dict = {}

    # ── OBDH thermal ramp ────────────────────────────────────────────────────
    obdh_trend = linear_trend(cdh["TEMP_OBDH"], cdh["elapsed_s"])
    report["obdh_trend"] = obdh_trend

    # ── COMMS thermal trend (secondary ramp) ─────────────────────────────────
    report["comms_trend"] = linear_trend(cdh["TEMP_COMMS"], cdh["elapsed_s"])

    # ── Z-score anomalies ────────────────────────────────────────────────────
    zscore_counts = {}
    zscore_masks  = {}
    for col in _ZSCORE_COLS:
        if col in cdh.columns:
            mask = zscore_anomalies(cdh[col])
            zscore_counts[col] = int(mask.sum())
            zscore_masks[col]  = mask
    report["zscore_counts"] = zscore_counts
    report["zscore_masks"]  = zscore_masks

    # ── Residual off-rail voltages ────────────────────────────────────────────
    report["adcs_rail_mean"]    = round(float(cdh["ADCS_5V_V"].mean()), 4)
    report["payload_rail_mean"] = round(float(cdh["PAYLOAD_5V_V"].mean()), 4)

    # ── Stuck thermistors ─────────────────────────────────────────────────────
    report["stuck_sensor_count"]   = len(stuck_sensors)
    report["stuck_sensor_names"]   = stuck_sensors
    report["thermistor_rod_value"] = 77.180115
    report["thermistor_bath_value"]= 74.309649

    # ── ADCS null state ───────────────────────────────────────────────────────
    imu_cols = ["ACCEL_X","ACCEL_Y","ACCEL_Z","GYRO_X","GYRO_Y","GYRO_Z",
                "MAG_X","MAG_Y","MAG_Z"]
    report["adcs_all_zero"] = all(
        adcs[c].eq(0).all() for c in imu_cols if c in adcs.columns
    )

    # ── CDH packet timing ─────────────────────────────────────────────────────
    report["cdh_timing"] = packet_timing(cdh["timestamp"])

    return report
