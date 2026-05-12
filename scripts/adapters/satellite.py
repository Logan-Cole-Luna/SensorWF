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
import pandas as pd

from .base import DomainAdapter
from scripts.satellite.parser import parse_analysis_file, parse_debug_file
from scripts.satellite.cleaner import clean_cdh, clean_adcs


_ONTOLOGY_PATH = os.path.join("results", "ontologies", "satellitesystem.owl")


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
            cdh_raw, adcs_raw = parse_analysis_file(path)
        if (cdh_raw is None or cdh_raw.empty) and debug_path and os.path.isfile(debug_path):
            cdh_raw, adcs_raw = parse_debug_file(debug_path)

        if cdh_raw is None or cdh_raw.empty:
            raise ValueError(
                f"SatelliteAdapter.load: no parseable CDH data from {path!r}"
            )

        cdh, stuck = clean_cdh(cdh_raw)
        self._stuck_sensors = stuck

        adcs = pd.DataFrame()
        if adcs_raw is not None and not adcs_raw.empty:
            adcs = clean_adcs(adcs_raw)

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

        Use this when downstream code (e.g. fault injector) requires the two
        streams separately.  Both returned DataFrames include timestamp and
        elapsed_s columns.
        """
        base = ["timestamp", "elapsed_s"]
        cdh_cols  = base + [c for c in self.cdh_channels  if c in combined_df.columns]
        adcs_cols = base + [c for c in self.adcs_channels if c in combined_df.columns]
        cdh  = combined_df[cdh_cols].copy()  if self.cdh_channels  else pd.DataFrame()
        adcs = combined_df[adcs_cols].copy() if self.adcs_channels else pd.DataFrame()
        return cdh, adcs

    def get_stuck_sensors(self) -> list[str]:
        """Stuck sensor channels detected during the last load() call."""
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
        """Return path to the satellite OWL ontology (generated on first run)."""
        return _ONTOLOGY_PATH

    # ----------------------------------------------------------------- E1 --

    def get_fault_types(self) -> list[dict]:
        """
        18 satellite fault morphologies (16 single-channel + 2 compound).
        Tags match those used in scripts/utils/injector.py.
        """
        return [
            # CDH — thermal
            {"tag": "T1_obdh_runaway",            "description": "OBDH thermal runaway (sustained ramp)",          "target": "cdh"},
            {"tag": "T2_board_thermal_shock",      "description": "Board thermal shock (rapid spike + recovery)",   "target": "cdh"},
            {"tag": "T3_thermal_bias_divergence",  "description": "Thermal bias divergence across sensors",         "target": "cdh"},
            # CDH — power
            {"tag": "P1_bus_sag_recovery",         "description": "Bus voltage sag and partial recovery",           "target": "cdh"},
            {"tag": "P2_switching_noise_burst",     "description": "Switching noise burst on power rails",           "target": "cdh"},
            {"tag": "P3_rail_latchup",              "description": "Rail latchup (sudden drop to near-zero)",        "target": "cdh"},
            # CDH — comms
            {"tag": "C1_packet_gap_jitter",        "description": "Packet timing jitter / gap anomaly",             "target": "cdh"},
            {"tag": "C2_rssi_fade",                "description": "RSSI signal fade",                               "target": "cdh"},
            {"tag": "C3_frame_error_avalanche",    "description": "Frame error count avalanche",                    "target": "cdh"},
            # ADCS — IMU
            {"tag": "A1_accel_packet_dropout",     "description": "Accelerometer packet dropout (zeros burst)",     "target": "adcs"},
            {"tag": "A2_gyro_clipping",            "description": "Gyro output clipping at ADC rail",               "target": "adcs"},
            {"tag": "A3_mag_field_inversion",      "description": "Magnetometer polarity inversion",                "target": "adcs"},
            {"tag": "A4_imu_correlation_break",    "description": "IMU cross-sensor correlation breakdown",         "target": "adcs"},
            {"tag": "A5_sun_sensor_blinding",      "description": "Sun sensor blinding (step saturation)",          "target": "adcs"},
            # ADCS — reaction wheels
            {"tag": "W1_wheel_runaway",            "description": "Reaction wheel runaway (unbounded spin-up)",     "target": "adcs"},
            {"tag": "W2_wheel_stiction_stop",      "description": "Reaction wheel stiction / stop",                 "target": "adcs"},
            # Compound
            {"tag": "COMPOUND_thermal_power_fault","description": "Simultaneous thermal + power rail fault",        "target": "cdh"},
        ]
