"""
pipeline_core.py -- Domain-agnostic M1-M3 pipeline functions for SensorWF.

Works with any time-series DataFrame that has:
  - timestamp  : datetime column
  - elapsed_s  : float seconds from session start
  - N numeric channels (any names)

Used by run_ecg.py, run_climate.py, and (via adapters) main.py.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# M2 -- Domain-agnostic quality assessment
# ---------------------------------------------------------------------------

_DEFAULT_QUALITY_CFG: dict[str, Any] = {
    "stuck_unique_max":   3,       # channels with <= this many unique values are stuck
    "zscore_threshold":   3.0,     # sigma threshold for per-channel z-score flagging
    "expected_dt_s":      None,    # expected sample interval (None = auto-estimate)
    "gap_multiplier":     5.0,     # gap > expected_dt * this is flagged
    "trend_channels":     [],      # channel names to run linear trend detection on
}


def run_quality_assessment(
    df: pd.DataFrame,
    channels: list[str],
    config: dict | None = None,
) -> dict:
    """
    Generic M2 quality assessment on any multi-channel time-series DataFrame.

    Parameters
    ----------
    df       : DataFrame with timestamp, elapsed_s, and sensor channels
    channels : list of numeric channel names to assess
    config   : optional dict overriding _DEFAULT_QUALITY_CFG keys

    Returns
    -------
    dict with keys: nan_rates, stuck_channels, timing, zscore_flags, trends
    """
    cfg = {**_DEFAULT_QUALITY_CFG, **(config or {})}
    report: dict[str, Any] = {}

    # 1. NaN rates per channel
    nan_rates: dict[str, float] = {}
    for col in channels:
        if col in df.columns:
            nan_rates[col] = round(float(df[col].isna().mean()), 4)
    report["nan_rates"] = nan_rates

    # 2. Stuck-channel detection
    stuck: list[str] = []
    for col in channels:
        if col not in df.columns:
            continue
        n_unique = df[col].nunique(dropna=True)
        if 0 < n_unique <= cfg["stuck_unique_max"]:
            stuck.append(col)
    report["stuck_channels"] = stuck

    # 3. Timing statistics
    if "elapsed_s" in df.columns:
        t = pd.to_numeric(df["elapsed_s"], errors="coerce").dropna()
        dt = t.diff().dropna()
        dt_pos = dt[dt > 0]
        if not dt_pos.empty:
            expected_dt = cfg["expected_dt_s"] or float(dt_pos.median())
            max_gap = float(dt_pos.max())
            gap_threshold = expected_dt * cfg["gap_multiplier"]
            n_gaps = int((dt_pos > gap_threshold).sum())
            report["timing"] = {
                "mean_dt_s":      round(float(dt_pos.mean()), 4),
                "std_dt_s":       round(float(dt_pos.std()),  4),
                "max_gap_s":      round(max_gap, 4),
                "expected_dt_s":  round(expected_dt, 4),
                "n_large_gaps":   n_gaps,
            }
        else:
            report["timing"] = {}
    else:
        report["timing"] = {}

    # 4. Per-channel Z-score flags
    zscore_flags: dict[str, int] = {}
    thr = float(cfg["zscore_threshold"])
    for col in channels:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce").ffill().bfill().fillna(0.0).to_numpy()
        std = x.std()
        if std < 1e-9:
            continue
        z = np.abs((x - x.mean()) / std)
        n_flagged = int((z > thr).sum())
        if n_flagged > 0:
            zscore_flags[col] = n_flagged
    report["zscore_flags"] = zscore_flags

    # 5. Linear trend detection on designated channels
    trends: dict[str, dict] = {}
    if "elapsed_s" in df.columns:
        t_all = pd.to_numeric(df["elapsed_s"], errors="coerce").ffill().bfill().values
        for col in cfg.get("trend_channels", []):
            if col not in df.columns:
                continue
            y = pd.to_numeric(df[col], errors="coerce").ffill().bfill().fillna(0.0).values
            n = min(len(t_all), len(y))
            if n < 4:
                continue
            try:
                slope, _, r, *_ = sp_stats.linregress(t_all[:n], y[:n])
                trends[col] = {
                    "slope_per_min": round(float(slope) * 60.0, 6),
                    "r_squared":     round(float(r ** 2), 6),
                    "delta":         round(float(y[n-1] - y[0]), 4),
                }
            except Exception:
                pass
    report["trends"] = trends

    # Summary counts
    n_total = len(df)
    n_good_channels = sum(1 for col in channels
                          if col in df.columns and df[col].notna().any())
    report["summary"] = {
        "n_rows":         n_total,
        "n_channels":     len(channels),
        "n_good_channels": n_good_channels,
        "n_stuck":        len(stuck),
        "n_zscore_flagged": len(zscore_flags),
    }

    return report


# ---------------------------------------------------------------------------
# M3 -- Domain-agnostic feature engineering
# ---------------------------------------------------------------------------

def build_generic_features(
    df: pd.DataFrame,
    channels: list[str],
    window: int = 15,
) -> tuple[np.ndarray | None, list[str]]:
    """
    Build a generic feature matrix from any set of numeric channels.

    Features per channel
    --------------------
    1. Raw value
    2. First-order difference (rate of change)
    3. Rolling mean (window)  -- only when session >= 3× window
    4. Rolling std  (window)  -- only when session >= 3× window

    Timing feature
    --------------
    dt_sample : elapsed_s first difference (packet-interval / sample-gap)

    Returns
    -------
    (X, feature_names) : (n_samples, n_features) float64 array + name list
    Returns (None, []) if no usable channels found.
    """
    cols = [c for c in channels if c in df.columns and df[c].notna().any()]
    if not cols:
        return None, []

    raw = df[cols].copy().astype(float)
    raw = raw.ffill().bfill().fillna(0.0)

    arrays: list[np.ndarray] = [raw.to_numpy()]
    names:  list[str]        = list(cols)

    # First-order differences
    diff = raw.diff().fillna(0.0).to_numpy()
    arrays.append(diff)
    names += [f"d_{c}" for c in cols]

    # Timing feature
    if "elapsed_s" in df.columns:
        elapsed = pd.to_numeric(df["elapsed_s"], errors="coerce").ffill().bfill().fillna(0.0)
        dt = elapsed.diff().fillna(0.0)
        arrays.append(dt.to_numpy()[:, None])
        names.append("dt_sample")

    # Rolling statistics
    if len(df) >= window * 3:
        rm = raw.rolling(window, min_periods=1).mean().to_numpy()
        rs = raw.rolling(window, min_periods=1).std().fillna(0.0).to_numpy()
        arrays.append(rm)
        arrays.append(rs)
        names += [f"rm_{c}" for c in cols]
        names += [f"rs_{c}" for c in cols]

    X = np.concatenate(arrays, axis=1).astype(np.float64)
    return X, names


# ---------------------------------------------------------------------------
# Helpers used by domain runners
# ---------------------------------------------------------------------------

def save_quality_report(report: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)


def make_elapsed(df: pd.DataFrame, dt_s: float) -> pd.DataFrame:
    """Add elapsed_s column from row index when timestamps unavailable."""
    df = df.copy()
    df["elapsed_s"] = np.arange(len(df), dtype=float) * dt_s
    return df
