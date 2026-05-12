"""
ecg.py -- SensorWF M1 adapter for the MIT-BIH Arrhythmia Database.

Each MIT-BIH record is a 30-minute, 2-lead ECG at 360 Hz.
This adapter:
  1. Reads the pre-converted CSV (columns: 'sample #', 'MLII', 'V5').
  2. Optionally downsamples to target_hz (default 50 Hz) for computational efficiency.
  3. Z-score normalises each lead relative to its own mean/std.
  4. Parses the annotation file for ground-truth beat labels.
  5. Returns a clean DataFrame with timestamp, elapsed_s, MLII, V5.

Reference
---------
Moody GB, Mark RG. "The impact of the MIT-BIH Arrhythmia Database."
IEEE Eng Med Biol Mag 20(3): 45-50 (May-June 2001). PMID: 11446209.
https://physionet.org/content/mitdb/1.0.0/
"""

from __future__ import annotations

import os
import re
import pandas as pd
import numpy as np

from .base import DomainAdapter

_NATIVE_HZ = 360.0
_TARGET_HZ  = 50.0
_DECIMATE   = int(round(_NATIVE_HZ / _TARGET_HZ))   # = 7, giving ~51.4 Hz

# Beat annotation types considered abnormal (non-normal)
_ABNORMAL_TYPES = frozenset("LRAaJSVF!ejE/fxQ|".split())


class ECGAdapter(DomainAdapter):
    name        = "Biomedical ECG (MIT-BIH Arrhythmia Database)"
    description = ("2-lead ambulatory ECG at 360 Hz. 48 30-minute records annotated "
                   "by cardiologists for beat type. Standard benchmark for arrhythmia detection.")
    channels    = ["MLII", "V5"]   # default; overridden by load() based on actual columns
    native_hz   = _NATIVE_HZ

    def __init__(self, target_hz: float = _TARGET_HZ, session_minutes: float = 5.0):
        self.target_hz       = float(target_hz)
        self.decimate        = max(1, int(round(_NATIVE_HZ / target_hz)))
        self.session_samples = int(session_minutes * 60 * target_hz)

    # ------------------------------------------------------------------
    def load(self, path: str, **kwargs) -> pd.DataFrame:
        """
        Load a MIT-BIH CSV file.

        Parameters
        ----------
        path : path to <record>.csv  (columns: 'sample #', 'MLII', 'V5')

        Returns
        -------
        DataFrame with timestamp, elapsed_s, MLII, V5 (Z-score normalised).
        """
        df = pd.read_csv(path)
        # Standardise column names (strip quotes / spaces)
        df.columns = [c.strip().strip("'") for c in df.columns]
        col_map = {c: c for c in df.columns}
        # 'sample #' → sample_idx
        for c in df.columns:
            if "sample" in c.lower():
                col_map[c] = "sample_idx"
        df = df.rename(columns=col_map)

        if "sample_idx" not in df.columns:
            df.insert(0, "sample_idx", np.arange(len(df)))

        # Auto-detect ECG channel names (MLII + V1/V2/.../V5)
        ecg_cols = [c for c in df.columns
                    if c.upper().startswith(("MLII", "V1", "V2", "V3", "V4", "V5",
                                              "ECG", "LEAD"))]
        if ecg_cols:
            self.channels = ecg_cols[:2]   # take first two leads

        # Downsample
        df = df.iloc[:: self.decimate].reset_index(drop=True)

        # Truncate to session length
        df = df.iloc[: self.session_samples].reset_index(drop=True)

        # Build elapsed_s
        dt_s = self.decimate / _NATIVE_HZ
        df["elapsed_s"] = np.arange(len(df), dtype=float) * dt_s

        # Synthetic timestamp (start from T0)
        t0 = pd.Timestamp("2000-01-01")
        df["timestamp"] = [t0 + pd.Timedelta(seconds=float(e)) for e in df["elapsed_s"]]

        # Z-score normalise ECG leads
        for ch in self.channels:
            if ch in df.columns:
                x = df[ch].astype(float)
                mu, sigma = x.mean(), x.std()
                if sigma > 1e-9:
                    df[ch] = (x - mu) / sigma
                else:
                    df[ch] = 0.0

        # Keep only detected channels that actually exist
        keep_channels = [c for c in self.channels if c in df.columns]
        if not keep_channels:
            raise KeyError(f"No ECG signal columns found. Got: {list(df.columns)}")
        self.channels = keep_channels
        return df[["timestamp", "elapsed_s"] + self.channels].copy()

    # ------------------------------------------------------------------
    def load_annotations(self, ann_path: str) -> pd.DataFrame:
        """
        Parse a MIT-BIH annotations text file.

        Returns DataFrame with columns:
          sample_native : sample index at 360 Hz
          beat_type     : single-char annotation type
          is_abnormal   : bool (True for non-normal beats)
        """
        rows = []
        with open(ann_path, encoding="utf-8") as fh:
            lines = fh.readlines()

        # Skip header line(s)
        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                sample = int(parts[1])
                btype  = parts[2]
                rows.append({
                    "sample_native": sample,
                    "beat_type":     btype,
                    "is_abnormal":   any(c in _ABNORMAL_TYPES for c in btype),
                })
            except (ValueError, IndexError):
                continue

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def get_quality_config(self) -> dict:
        return {
            "stuck_unique_max": 3,
            "zscore_threshold": 4.0,    # ECG has sharper spikes than satellite
            "expected_dt_s":    self.decimate / _NATIVE_HZ,
            "gap_multiplier":   10.0,
            "trend_channels":   [],
        }

    def get_feature_config(self) -> dict:
        return {
            "channels": self.channels,
            "window":   100,            # 2 seconds at 50 Hz
        }

    def get_ontology_path(self) -> str:
        return os.path.join("results", "ontologies", "ecg.owl")

    def get_fault_types(self) -> list[dict]:
        return [
            {"tag": "ecg_baseline_wander",  "description": "Sinusoidal baseline drift (respiration artifact)"},
            {"tag": "ecg_electrode_dropout", "description": "Lead zero-clamp (electrode disconnection)"},
            {"tag": "ecg_emg_burst",         "description": "High-frequency EMG noise burst (muscle artifact)"},
            {"tag": "ecg_powerline_noise",   "description": "60 Hz sinusoidal interference (power line)"},
            {"tag": "ecg_amplitude_scale",   "description": "Multiplicative amplitude scaling (amplifier error)"},
            {"tag": "ecg_lead_inversion",    "description": "Sign inversion of one lead (lead reversal)"},
        ]
