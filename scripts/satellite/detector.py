"""
detector.py — Statistical anomaly detection for cleaned telemetry.

This module implements the quality-assessment / statistical detection
layer (Workflow Module M2 / early-stage detection), separate from the
ML-based evaluation in evaluator.py.

run_detection()
    • Per-channel Z-score flags (3-sigma threshold).
    • OBDH thermal trend (linear regression slope + R²).
    • ADCS all-zero check and stuck-channel detection.
    • Cross-sensor Pearson correlation breakdown windows.
    • CDH packet-interval timing statistics.

Returns a report dict consumed by main.py and the plotter.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Column groups ──────────────────────────────────────────────────────────────
_TEMP_COLS = [
    "TEMP_SOLAR", "TEMP_EPS", "TEMP_BATTERY", "TEMP_BACKPLANE",
    "TEMP_OBDH", "TEMP_ADCS", "TEMP_WHEEL", "TEMP_COMMS", "TEMP_PAYLOAD",
]
_VOLT_COLS = ["SYS_V", "PLAT_5V_V", "OBC_PWR_V", "COMMS_PWR_V"]
_ADCS_MOTION_COLS = ["ACCEL_X", "ACCEL_Y", "ACCEL_Z",
                     "GYRO_X",  "GYRO_Y",  "GYRO_Z",
                     "MAG_X",   "MAG_Y",   "MAG_Z"]

_ZSCORE_THRESHOLD = 3.0
_STUCK_UNIQUE_MAX = 3          # channels with ≤ this many unique values are flagged
_CORRELATION_PAIRS = [
    ("TEMP_EPS",     "TEMP_OBDH"),
    ("TEMP_BATTERY", "TEMP_EPS"),
    ("TEMP_ADCS",    "TEMP_WHEEL"),
    ("GYRO_X",       "GYRO_Y"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_detection(
    cdh: pd.DataFrame,
    adcs: pd.DataFrame,
    stuck_sensors: list[str],
) -> dict:
    """
    Run the statistical detection layer on cleaned CDH and ADCS DataFrames.

    Parameters
    ----------
    cdh           : cleaned CDH DataFrame (output of clean_cdh)
    adcs          : cleaned ADCS DataFrame (output of clean_adcs)
    stuck_sensors : list of channel names flagged as stuck by the cleaner

    Returns
    -------
    dict with keys:
        obdh_trend           : dict (slope_per_min, r_squared, delta)
        zscore_counts        : dict[str, int] channels → number of 3σ flags
        zscore_masks         : dict[str, np.ndarray] boolean flag arrays
        adcs_all_zero        : bool
        adcs_stuck_channels  : list[str]
        correlation_breakdown: dict[str, np.ndarray] pair → boolean mask
        wheel_gyro_correlation: float
        cdh_timing           : dict (mean_s, std_s, max_gap_s)
    """
    report: dict = {}

    # 1. OBDH thermal trend
    report["obdh_trend"] = _obdh_trend(cdh)

    # 2. Per-channel Z-score flags
    counts, masks = _zscore_flags(cdh)
    report["zscore_counts"] = counts
    report["zscore_masks"]  = masks

    # 3. ADCS checks
    report["adcs_all_zero"]       = _adcs_all_zero(adcs)
    report["adcs_stuck_channels"] = _adcs_stuck(adcs)

    # 4. Cross-sensor correlation breakdown
    report["correlation_breakdown"] = _correlation_breakdown(cdh, adcs)

    # 5. Wheel-gyro correlation
    report["wheel_gyro_correlation"] = _wheel_gyro_corr(adcs)

    # 6. CDH packet-interval timing
    report["cdh_timing"] = _timing_stats(cdh)

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _obdh_trend(cdh: pd.DataFrame) -> dict:
    """Linear regression of TEMP_OBDH over elapsed_s."""
    if "TEMP_OBDH" not in cdh.columns or "elapsed_s" not in cdh.columns:
        return {"slope_per_min": 0.0, "r_squared": 0.0, "delta": 0.0}

    t = pd.to_numeric(cdh["elapsed_s"], errors="coerce").dropna()
    y = pd.to_numeric(cdh["TEMP_OBDH"], errors="coerce").dropna()
    n = min(len(t), len(y))
    if n < 4:
        return {"slope_per_min": 0.0, "r_squared": 0.0, "delta": 0.0}

    t, y = t.values[:n], y.values[:n]
    try:
        slope, intercept, r, *_ = sp_stats.linregress(t, y)
    except Exception:
        return {"slope_per_min": 0.0, "r_squared": 0.0, "delta": 0.0}

    delta = float(y[-1] - y[0]) if n > 1 else 0.0
    return {
        "slope_per_min": round(float(slope) * 60.0, 6),
        "r_squared":     round(float(r ** 2),        6),
        "delta":         round(delta,                 4),
    }


def _zscore_flags(cdh: pd.DataFrame) -> tuple[dict, dict]:
    """Flag samples exceeding _ZSCORE_THRESHOLD standard deviations per channel."""
    counts: dict[str, int]          = {}
    masks:  dict[str, np.ndarray]   = {}

    for col in _TEMP_COLS + _VOLT_COLS:
        if col not in cdh.columns:
            continue
        x = pd.to_numeric(cdh[col], errors="coerce").ffill().bfill().fillna(0.0).values
        std = float(x.std())
        if std < 1e-9:
            continue
        z = np.abs((x - x.mean()) / std)
        mask = z > _ZSCORE_THRESHOLD
        if mask.any():
            counts[col] = int(mask.sum())
            masks[col]  = mask

    return counts, masks


def _adcs_all_zero(adcs: pd.DataFrame) -> bool:
    """Return True if all ADCS motion channels are zero or missing."""
    cols = [c for c in _ADCS_MOTION_COLS if c in adcs.columns]
    if not cols:
        return True
    sub = adcs[cols].apply(pd.to_numeric, errors="coerce")
    return bool((sub.abs().max(axis=None) == 0.0) or sub.isnull().all(axis=None))


def _adcs_stuck(adcs: pd.DataFrame) -> list[str]:
    """Return ADCS channels with unusually few unique values (stuck sensors)."""
    stuck: list[str] = []
    for col in _ADCS_MOTION_COLS:
        if col not in adcs.columns:
            continue
        n_unique = adcs[col].nunique(dropna=True)
        if 0 < n_unique <= _STUCK_UNIQUE_MAX:
            stuck.append(col)
    return stuck


def _correlation_breakdown(
    cdh: pd.DataFrame,
    adcs: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """
    Detect windows where a correlated pair shows abnormally low correlation.
    Returns a boolean mask (length = CDH rows) per pair.
    """
    result: dict[str, np.ndarray] = {}
    n = len(cdh)
    window = max(20, n // 10)

    def _get(df: pd.DataFrame, col: str) -> np.ndarray | None:
        src = cdh if col in cdh.columns else (adcs if col in adcs.columns else None)
        if src is None:
            return None
        return pd.to_numeric(src[col], errors="coerce").ffill().bfill().fillna(0.0).values[:n]

    for col_a, col_b in _CORRELATION_PAIRS:
        a = _get(cdh, col_a)
        b = _get(cdh, col_b)
        if a is None or b is None:
            continue

        mask = np.zeros(n, dtype=bool)
        for i in range(0, n - window, window // 2):
            j = min(i + window, n)
            sa, sb = a[i:j], b[i:j]
            if np.std(sa) < 1e-9 or np.std(sb) < 1e-9:
                continue
            r = float(np.corrcoef(sa, sb)[0, 1])
            if not np.isfinite(r):
                continue
            # Baseline correlation from full series
            base_r = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-9 and np.std(b) > 1e-9 else 0.0
            if abs(r - base_r) > 0.35:
                mask[i:j] = True

        if mask.any():
            result[f"{col_a}_{col_b}"] = mask

    return result


def _wheel_gyro_corr(adcs: pd.DataFrame) -> float:
    """Pearson correlation between WHEEL_SPEED magnitude and mean gyro magnitude."""
    cols_g = [c for c in ("GYRO_X", "GYRO_Y", "GYRO_Z") if c in adcs.columns]
    if "WHEEL_SPEED" not in adcs.columns or not cols_g:
        return float("nan")
    ws = pd.to_numeric(adcs["WHEEL_SPEED"], errors="coerce").abs().ffill().bfill().fillna(0.0)
    gm = adcs[cols_g].apply(pd.to_numeric, errors="coerce").abs().mean(axis=1).ffill().bfill().fillna(0.0)
    n = min(len(ws), len(gm))
    if n < 5 or ws.values[:n].std() < 1e-9 or gm.values[:n].std() < 1e-9:
        return float("nan")
    return round(float(np.corrcoef(ws.values[:n], gm.values[:n])[0, 1]), 4)


def _timing_stats(cdh: pd.DataFrame) -> dict:
    """Packet-interval statistics from elapsed_s."""
    if "elapsed_s" not in cdh.columns or len(cdh) < 3:
        return {"mean_s": float("nan"), "std_s": float("nan"), "max_gap_s": float("nan")}
    t = pd.to_numeric(cdh["elapsed_s"], errors="coerce").dropna()
    dt = t.diff().dropna()
    dt = dt[dt > 0]
    if dt.empty:
        return {"mean_s": float("nan"), "std_s": float("nan"), "max_gap_s": float("nan")}
    return {
        "mean_s":    round(float(dt.mean()),  4),
        "std_s":     round(float(dt.std()),   4),
        "max_gap_s": round(float(dt.max()),   4),
    }
