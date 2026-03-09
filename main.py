"""
main.py — KISPE Satellite Learning Laboratory Telemetry Pipeline
Entry point. Orchestrates parsing → cleaning → detection → plotting → export.
"""

import os
import sys

import pandas as pd

from scripts.parser   import parse_analysis_file
from scripts.cleaner  import clean_cdh, clean_adcs
from scripts.detector import run_detection
from scripts.plotter  import run_all_plots

# ── Configuration ─────────────────────────────────────────────────────────────
ANALYSIS_FILE = os.path.join("Data", "Archive_2026_03_02_112454_Analysis.txt")
OUTPUT_DIR    = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Parse
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("KISPE SLL Telemetry Pipeline")
print("=" * 60)

print("\n[1/4] Parsing archive …")
cdh_raw, adcs_raw = parse_analysis_file(ANALYSIS_FILE)
print(f"      CDH  packets : {len(cdh_raw)}")
print(f"      ADCS packets : {len(adcs_raw)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Clean
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/4] Cleaning data …")
cdh, stuck_sensors = clean_cdh(cdh_raw)
adcs               = clean_adcs(adcs_raw)
print(f"      CDH  rows after cleaning : {len(cdh)}")
print(f"      ADCS rows after cleaning : {len(adcs)}")
print(f"      Stuck thermistor channels: {len(stuck_sensors)}")

# Export cleaned CSVs
cdh.to_csv(os.path.join(OUTPUT_DIR, "cdh_clean.csv"), index=False)
adcs.to_csv(os.path.join(OUTPUT_DIR, "adcs_clean.csv"), index=False)
print("      Exported cdh_clean.csv and adcs_clean.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Detect
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/4] Running anomaly detection …")
report = run_detection(cdh, adcs, stuck_sensors)

obdh = report["obdh_trend"]
print(f"      OBDH slope  : {obdh['slope_per_min']:.3f} °C/min")
print(f"      OBDH R²     : {obdh['r_squared']:.4f}")
print(f"      OBDH delta  : {obdh['delta']:.3f} °C")
print(f"      Z-score flags: {report['zscore_counts']}")
print(f"      ADCS all-zero: {report['adcs_all_zero']}")
ct = report["cdh_timing"]
print(f"      CDH interval: mean={ct['mean_s']:.3f}s  std={ct['std_s']:.3f}s  max_gap={ct['max_gap_s']:.3f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Plot
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/4] Generating plots …")
saved = run_all_plots(cdh, adcs, report, OUTPUT_DIR)
print(f"\n      {len(saved)} plots saved to {OUTPUT_DIR}/")

print("\n" + "=" * 60)
print("Pipeline complete.")
print("=" * 60)
