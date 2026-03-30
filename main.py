"""
main.py — KISPE Satellite Learning Laboratory Telemetry Pipeline
Entry point. Orchestrates parsing → cleaning → detection → plotting → export.
Loops over all experiment subfolders found in data/.
Falls back to Debug file parsing when an Analysis file is empty.
"""

import json
import os
import glob

import pandas as pd

from scripts.parser   import parse_analysis_file, parse_debug_file
from scripts.cleaner  import clean_cdh, clean_adcs
from scripts.detector import run_detection
from scripts.plotter  import run_all_plots
from scripts.injector import run_injections

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR    = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Collect all experiment folders — prefer Analysis file, fall back to Debug
experiments = []
for entry in sorted(os.scandir(DATA_DIR), key=lambda e: e.name):
    if not entry.is_dir():
        continue
    analysis_files = sorted(glob.glob(os.path.join(entry.path, "*_Analysis.txt")))
    debug_files    = sorted(glob.glob(os.path.join(entry.path, "*_Debug.txt")))
    if analysis_files:
        experiments.append((entry.name, analysis_files[0], debug_files[0] if debug_files else None))
    elif debug_files:
        experiments.append((entry.name, None, debug_files[0]))

if not experiments:
    print(f"No experiments found in {DATA_DIR}/")
    raise SystemExit(1)

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("KISPE SLL Telemetry Pipeline")
print(f"Found {len(experiments)} experiment(s)")
print("=" * 60)

for exp_name, analysis_file, debug_file in experiments:
    output_dir = os.path.join(RESULTS_DIR, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'─' * 60}")
    print(f"Experiment : {exp_name}")
    print(f"Output     : {output_dir}/")
    print(f"{'─' * 60}")

    # ── 1. Parse ──────────────────────────────────────────────────────────────
    print("\n  [1/5] Parsing archive …")
    cdh_raw = adcs_raw = None
    parse_source = None
    try:
        if analysis_file:
            cdh_raw, adcs_raw = parse_analysis_file(analysis_file)
            parse_source = f"Analysis  {analysis_file}"
        # Fall back to Debug file if Analysis produced no CDH rows
        if (cdh_raw is None or cdh_raw.empty) and debug_file:
            cdh_raw, adcs_raw = parse_debug_file(debug_file)
            parse_source = f"Debug     {debug_file}"
    except Exception as exc:
        print(f"  ERROR parsing {exp_name}: {exc} — skipping.")
        continue

    if cdh_raw is None or cdh_raw.empty:
        print(f"  No parseable data found for {exp_name} — skipping.")
        continue

    print(f"        Source       : {parse_source}")
    print(f"        CDH  packets : {len(cdh_raw)}")
    print(f"        ADCS packets : {len(adcs_raw)}")

    # ── 2. Clean ──────────────────────────────────────────────────────────────
    print("\n  [2/5] Cleaning data …")
    try:
        cdh, stuck_sensors = clean_cdh(cdh_raw)
        adcs = clean_adcs(adcs_raw) if not adcs_raw.empty else adcs_raw
    except Exception as exc:
        print(f"  ERROR cleaning {exp_name}: {exc} — skipping.")
        continue
    print(f"        CDH  rows after cleaning : {len(cdh)}")
    print(f"        ADCS rows after cleaning : {len(adcs)}")
    print(f"        Stuck thermistor channels: {len(stuck_sensors)}")

    cdh.to_csv(os.path.join(output_dir, "cdh_clean.csv"), index=False)
    adcs.to_csv(os.path.join(output_dir, "adcs_clean.csv"), index=False)
    print("        Exported cdh_clean.csv and adcs_clean.csv")

    # ── 3. Detect ─────────────────────────────────────────────────────────────
    print("\n  [3/5] Running anomaly detection …")
    try:
        report = run_detection(cdh, adcs, stuck_sensors)
    except Exception as exc:
        print(f"  ERROR in detection for {exp_name}: {exc} — skipping.")
        continue

    obdh = report["obdh_trend"]
    print(f"        OBDH slope   : {obdh['slope_per_min']:.3f} °C/min")
    print(f"        OBDH R²      : {obdh['r_squared']:.4f}")
    print(f"        OBDH delta   : {obdh['delta']:.3f} °C")
    print(f"        Z-score flags: {report['zscore_counts']}")
    print(f"        ADCS all-zero: {report['adcs_all_zero']}")
    if report["adcs_stuck_channels"]:
        print(f"        ADCS stuck   : {report['adcs_stuck_channels']}")
    cb = report["correlation_breakdown"]
    broken = [pair for pair, mask in cb.items() if mask.sum() > 5]
    if broken:
        print(f"        Corr broken  : {broken}")
    ct = report["cdh_timing"]
    print(f"        CDH interval : mean={ct['mean_s']:.3f}s  std={ct['std_s']:.3f}s  max_gap={ct['max_gap_s']:.3f}s")

    # Serialise report to JSON (drop non-scalar objects: Series masks, intervals)
    def _serialisable(report: dict) -> dict:
        skip_keys = {"zscore_masks", "correlation_breakdown", "wheel_gyro_correlation"}
        out = {}
        for k, v in report.items():
            if k in skip_keys:
                continue
            if isinstance(v, dict):
                out[k] = {ik: (iv if not hasattr(iv, "tolist") else iv) for ik, iv in v.items()
                          if not hasattr(iv, "iloc")}
            elif hasattr(v, "iloc"):   # pandas Series — skip
                continue
            else:
                out[k] = v
        return out

    report_path = os.path.join(output_dir, "detection_report.json")
    with open(report_path, "w") as fh:
        json.dump(_serialisable(report), fh, indent=2, default=str)
    print(f"        Report saved  : {report_path}")

    # ── 4. Plot ───────────────────────────────────────────────────────────────
    print("\n  [4/5] Generating plots …")
    try:
        saved = run_all_plots(cdh, adcs, report, output_dir)
        anom_count = sum(1 for p in saved if "anomaly_detection" in p)
        gen_count  = len(saved) - anom_count
        print(f"\n        {gen_count} general plots  → {output_dir}/")
        print(f"        {anom_count} detection plots → {output_dir}/anomaly_detection/")
    except Exception as exc:
        print(f"  ERROR plotting {exp_name}: {exc}")

    # ── 5. Inject anomalies ───────────────────────────────────────────────────
    print("\n  [5/5] Generating injected anomaly variants …")
    try:
        run_injections(cdh, adcs, output_dir, seed=42, n_variants=2,
                       baseline_report=report)
    except Exception as exc:
        print(f"  ERROR injecting {exp_name}: {exc}")

print("\n" + "=" * 60)
print(f"Pipeline complete. Results in {RESULTS_DIR}/")
print("=" * 60)
