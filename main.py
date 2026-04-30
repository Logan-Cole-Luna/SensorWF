"""
main.py — TelWF: CubeSat Telemetry Analysis Core Pipeline
==========================================================
Automated pipeline from raw SCOTTI archive exports to analysis-ready
clean data, quality reports, and feature matrices.

Pipeline stages
───────────────
  [P1] TelemetryIngestion  — discover and parse SCOTTI archive files
  [P2] QualityAssessment   — clean, hex-decode, derive quality flags
  [P3] FeatureEngineering  — build numeric feature matrices

Use-case extensions (fault injection, anomaly detection, semantic KG,
provenance export) are in use_case.py.

Data discovery strategy
────────────────────────
  data/<Family>/<Session>/Archive_*_Analysis.txt   ← preferred
  data/<Family>/<Session>/Archive_*_Debug.txt      ← fallback
  data/<Session>/Archive_*_Analysis.txt            ← flat layout (legacy)

  For each Family, the session with the most CDH lines is selected as
  the representative clean-baseline session.

Outputs per family
──────────────────
  results/<Family>/
    cdh_clean.csv  adcs_clean.csv  detection_report.json  *.png
  results/
    workflow_spec.ttl
"""

from __future__ import annotations

import argparse
import json
import os
import glob
from datetime import datetime, timezone

import pandas as pd

from scripts.parser   import parse_analysis_file, parse_debug_file
from scripts.cleaner  import clean_cdh, clean_adcs
from scripts.detector import run_detection
from scripts.plotter  import run_all_plots
from scripts.workflow import save_workflow_spec
from scripts.evaluator import build_feature_matrix, build_combined_feature_matrix

# ── CLI arguments ──────────────────────────────────────────────────────────────
_ap = argparse.ArgumentParser(description="TelWF core pipeline: ingest → quality → features")
_ap.add_argument("--families", nargs="*", metavar="NAME", help="Restrict to named experiment families")
_ARGS = _ap.parse_args()

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR    = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Emit static workflow specification (ProvONE) ───────────────────────────────
spec_path = os.path.join(RESULTS_DIR, "workflow_spec.ttl")
save_workflow_spec(spec_path)

# ── Collect experiment families and their representative session ───────────────

def _pick_best_session(sessions: list[tuple[str, str]]) -> tuple[str, str]:
    """Return (analysis_path, debug_path) from the session with the most rows."""
    if len(sessions) == 1:
        return sessions[0]
    best, best_size = sessions[0], -1
    for s in sessions:
        try:
            sz = os.path.getsize(s[0]) if s[0] else 0
        except OSError:
            sz = 0
        if sz > best_size:
            best, best_size = s, sz
    return best


def _discover_experiments(data_dir: str) -> list[tuple[str, str, str | None]]:
    """
    Walk data_dir and return [(family_name, analysis_file, debug_file), ...].

    Handles:
      data/<Family>/<Session>/Archive_*_Analysis.txt  (nested, primary)
      data/<Session>/Archive_*_Analysis.txt           (flat, legacy)
    """
    found: list[tuple[str, str, str | None]] = []
    skip_families = {"misc"}

    for family_entry in sorted(os.scandir(data_dir), key=lambda e: e.name):
        if not family_entry.is_dir():
            continue
        if family_entry.name in skip_families:
            continue

        sessions: list[tuple[str, str]] = []
        for session_entry in sorted(os.scandir(family_entry.path), key=lambda e: e.name):
            if not session_entry.is_dir():
                continue
            a_files = sorted(glob.glob(os.path.join(session_entry.path, "*_Analysis.txt")))
            d_files = sorted(glob.glob(os.path.join(session_entry.path, "*_Debug.txt")))
            if a_files:
                sessions.append((a_files[0], d_files[0] if d_files else ""))
            elif d_files:
                sessions.append(("", d_files[0]))

        if sessions:
            best = _pick_best_session(sessions)
            a = best[0] if best[0] else None
            d = best[1] if best[1] else None
            found.append((family_entry.name, a, d))
            continue

        a_files = sorted(glob.glob(os.path.join(family_entry.path, "*_Analysis.txt")))
        d_files = sorted(glob.glob(os.path.join(family_entry.path, "*_Debug.txt")))
        if a_files:
            found.append((family_entry.name, a_files[0], d_files[0] if d_files else None))
        elif d_files:
            found.append((family_entry.name, None, d_files[0]))

    return found


experiments = _discover_experiments(DATA_DIR)
if _ARGS.families:
    allowed = {f.lower() for f in _ARGS.families}
    experiments = [e for e in experiments if e[0].lower() in allowed]

if not experiments:
    print(f"No experiments found in {DATA_DIR}/")
    raise SystemExit(1)

# ─────────────────────────────────────────────────────────────────────────────
run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
print("=" * 70)
print("TelWF — CubeSat Telemetry Analysis Core Pipeline")
print(f"Found {len(experiments)} experiment family/families   |   Run: {run_ts}")
print("=" * 70)
for name, af, df in experiments:
    src = af or df or "?"
    print(f"  {name:<30} ← {os.path.relpath(src)}")

for exp_name, analysis_file, debug_file in experiments:
    output_dir = os.path.join(RESULTS_DIR, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'─' * 70}")
    print(f"Experiment : {exp_name}")
    print(f"Output     : {output_dir}/")
    print(f"{'─' * 70}")

    # ── P1: TelemetryIngestion ─────────────────────────────────────────────────
    print("\n  [P1] TelemetryIngestion — parsing archive …")
    cdh_raw = adcs_raw = None
    parse_source = None
    try:
        if analysis_file:
            cdh_raw, adcs_raw = parse_analysis_file(analysis_file)
            parse_source = analysis_file
        if (cdh_raw is None or cdh_raw.empty) and debug_file:
            cdh_raw, adcs_raw = parse_debug_file(debug_file)
            parse_source = debug_file
    except Exception as exc:
        print(f"  ERROR in P1 for {exp_name}: {exc} — skipping.")
        continue

    if cdh_raw is None or cdh_raw.empty:
        print(f"  No parseable data for {exp_name} — skipping.")
        continue

    print(f"        Source       : {parse_source}")
    print(f"        CDH  packets : {len(cdh_raw)}")
    print(f"        ADCS packets : {len(adcs_raw)}")

    # ── P2: QualityAssessment ──────────────────────────────────────────────────
    print("\n  [P2] QualityAssessment — cleaning data …")
    try:
        cdh, stuck_sensors = clean_cdh(cdh_raw)
        adcs = clean_adcs(adcs_raw) if not adcs_raw.empty else adcs_raw
    except Exception as exc:
        print(f"  ERROR in P2 for {exp_name}: {exc} — skipping.")
        continue

    print(f"        CDH  rows after cleaning : {len(cdh)}")
    print(f"        ADCS rows after cleaning : {len(adcs)}")
    print(f"        Stuck thermistor channels: {len(stuck_sensors)}")

    cdh_path  = os.path.join(output_dir, "cdh_clean.csv")
    adcs_path = os.path.join(output_dir, "adcs_clean.csv")
    cdh.to_csv(cdh_path,  index=False)
    adcs.to_csv(adcs_path, index=False)
    print(f"        Exported: cdh_clean.csv  adcs_clean.csv")

    # Statistical detection (per-channel flags, OBDH trend, timing)
    print("\n  [P2b] StatisticalDetection — computing per-channel flags …")
    try:
        report = run_detection(cdh, adcs, stuck_sensors)
    except Exception as exc:
        print(f"  ERROR in StatisticalDetection for {exp_name}: {exc}")
        report = {
            "obdh_trend": {"slope_per_min": 0.0, "r_squared": 0.0, "delta": 0.0},
            "zscore_counts": {}, "zscore_masks": {},
            "adcs_all_zero": False, "adcs_stuck_channels": [],
            "correlation_breakdown": {}, "wheel_gyro_correlation": float("nan"),
            "cdh_timing": {"mean_s": float("nan"), "std_s": float("nan"), "max_gap_s": float("nan")},
        }

    obdh = report["obdh_trend"]
    print(f"        OBDH slope   : {obdh['slope_per_min']:.3f} °C/min")
    print(f"        OBDH R²      : {obdh['r_squared']:.4f}")
    print(f"        Z-score flags: {report['zscore_counts']}")
    ct = report["cdh_timing"]
    print(f"        CDH interval : mean={ct['mean_s']:.3f}s  std={ct['std_s']:.3f}s  max_gap={ct['max_gap_s']:.3f}s")

    def _serialisable(r: dict) -> dict:
        skip_keys = {"zscore_masks", "correlation_breakdown", "wheel_gyro_correlation"}
        out = {}
        for k, v in r.items():
            if k in skip_keys:
                continue
            if isinstance(v, dict):
                out[k] = {ik: iv for ik, iv in v.items() if not hasattr(iv, "iloc")}
            elif hasattr(v, "iloc"):
                continue
            else:
                out[k] = v
        return out

    report_path = os.path.join(output_dir, "detection_report.json")
    with open(report_path, "w") as fh:
        json.dump(_serialisable(report), fh, indent=2, default=str)
    print(f"        Report saved  : {report_path}")

    # ── P3: FeatureEngineering ─────────────────────────────────────────────────
    print("\n  [P3] FeatureEngineering — deriving feature matrices …")
    X_cdh,   names_cdh   = build_feature_matrix(cdh,  "cdh")
    X_adcs,  names_adcs  = build_feature_matrix(adcs, "adcs")
    X_combo, names_combo = build_combined_feature_matrix(cdh, adcs)
    print(f"        CDH  features : {X_cdh.shape[1]  if X_cdh  is not None else 0}")
    print(f"        ADCS features : {X_adcs.shape[1] if X_adcs is not None else 0}")
    print(f"        Combined      : {X_combo.shape[1] if X_combo is not None else 0}")

    # ── Telemetry visualisations ───────────────────────────────────────────────
    print("\n  [plots] Generating telemetry visualisations …")
    try:
        saved = run_all_plots(cdh, adcs, output_dir)
        print(f"        {len(saved)} figures → {output_dir}/")
    except Exception as exc:
        print(f"  ERROR plotting {exp_name}: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"TelWF core pipeline complete.  Results → {RESULTS_DIR}/")
print(f"  Workflow spec  : {spec_path}")
print(f"  Experiments    : {len(experiments)}")
print(f"  Next step      : python use_case.py   (fault injection → detection → KG → provenance)")
print("=" * 70)
