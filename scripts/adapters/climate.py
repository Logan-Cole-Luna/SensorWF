"""
climate.py -- SensorWF M1 adapter for the Jena Climate Dataset.

14 meteorological features sampled every 10 minutes at the Max Planck
Institute for Biogeochemistry weather station in Jena, Germany.
Covers 01 Jan 2009 -- 31 Dec 2016 (~420,550 rows).

Reference
---------
Jena Climate Data, Max Planck Institute for Biogeochemistry.
https://www.bgc-jena.mpg.de/wetter/
Dataset widely used in time-series ML benchmarks (Chollet 2021).
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np

from .base import DomainAdapter

_INTERVAL_MINUTES = 10
_INTERVAL_S       = _INTERVAL_MINUTES * 60.0      # 600 s

# Map CSV column names → clean snake_case names
_COL_MAP = {
    "p (mbar)":       "p_mbar",
    "T (degC)":       "T_degC",
    "Tpot (K)":       "Tpot_K",
    "Tdew (degC)":    "Tdew_degC",
    "rh (%)":         "rh_pct",
    "VPmax (mbar)":   "VPmax_mbar",
    "VPact (mbar)":   "VPact_mbar",
    "VPdef (mbar)":   "VPdef_mbar",
    "sh (g/kg)":      "sh_gkg",
    "H2OC (mmol/mol)": "H2OC_mmolmol",
    "rho (g/m**3)":   "rho_gm3",
    "wv (m/s)":       "wv_ms",
    "max. wv (m/s)":  "max_wv_ms",
    "wd (deg)":       "wd_deg",
}

# Channels used for quality + features (all 14 physical channels)
_CHANNELS = list(_COL_MAP.values())

# Wind speed sentinel value: -9999 m/s means sensor malfunction → NaN
_WV_SENTINEL = -9999.0


class ClimateAdapter(DomainAdapter):
    name        = "Atmospheric Climate (Jena Climate Dataset)"
    description = ("14 meteorological features at 10-min resolution from Jena, Germany "
                   "(2009-2016). Canonical time-series ML benchmark (420,551 rows).")
    channels    = _CHANNELS
    native_hz   = 1.0 / _INTERVAL_S     # ~0.00167 Hz

    def __init__(self, year: int = 2011, months: tuple[int, int] = (1, 6)):
        """
        Parameters
        ----------
        year   : calendar year to extract
        months : (start_month, end_month) inclusive
        """
        self.year   = int(year)
        self.months = months

    # ------------------------------------------------------------------
    def load(self, path: str, **kwargs) -> pd.DataFrame:
        """
        Load Jena Climate CSV and extract a single-year slice.

        Parameters
        ----------
        path : path to jena_climate_2009_2016.csv

        Returns
        -------
        DataFrame with timestamp, elapsed_s, and 14 meteorological channels.
        """
        df = pd.read_csv(path, parse_dates=["Date Time"], dayfirst=True)
        df = df.rename(columns={"Date Time": "timestamp"})
        df = df.rename(columns=_COL_MAP)

        # Clean wind speed sentinel
        for wv_col in ["wv_ms", "max_wv_ms"]:
            if wv_col in df.columns:
                df[wv_col] = df[wv_col].replace(_WV_SENTINEL, np.nan)

        # Filter to requested year / months
        mask = (
            (df["timestamp"].dt.year  == self.year) &
            (df["timestamp"].dt.month >= self.months[0]) &
            (df["timestamp"].dt.month <= self.months[1])
        )
        df = df[mask].reset_index(drop=True)

        if df.empty:
            raise ValueError(f"No data found for year={self.year} months={self.months}")

        # elapsed_s from session start
        t0 = df["timestamp"].iloc[0]
        df["elapsed_s"] = (df["timestamp"] - t0).dt.total_seconds()

        # Forward-fill then backward-fill residual NaN gaps
        for ch in _CHANNELS:
            if ch in df.columns:
                df[ch] = pd.to_numeric(df[ch], errors="coerce").ffill().bfill()

        return df[["timestamp", "elapsed_s"] + _CHANNELS].copy()

    # ------------------------------------------------------------------
    def get_quality_config(self) -> dict:
        return {
            "stuck_unique_max": 3,
            "zscore_threshold": 3.5,
            "expected_dt_s":    _INTERVAL_S,
            "gap_multiplier":   3.0,
            "trend_channels":   ["T_degC", "p_mbar"],
        }

    def get_feature_config(self) -> dict:
        return {
            "channels": _CHANNELS,
            "window":   24,     # 4 hours at 10-min resolution
        }

    def get_ontology_path(self) -> str:
        return os.path.join("results", "ontologies", "climate.owl")

    def get_fault_types(self) -> list[dict]:
        return [
            {"tag": "clim_sensor_drift",   "description": "Slow polynomial drift (sensor aging / fouling)"},
            {"tag": "clim_stuck_sensor",   "description": "Channel frozen at pre-fault value (sensor failure)"},
            {"tag": "clim_spike_burst",    "description": "Burst of amplitude spikes (electrical interference)"},
            {"tag": "clim_dropout_zero",   "description": "Channel drops to zero (power loss to sensor)"},
            {"tag": "clim_scale_error",    "description": "Multiplicative calibration shift (recalibration error)"},
            {"tag": "clim_sign_flip",      "description": "Channel sign inversion (wiring reversal)"},
        ]
