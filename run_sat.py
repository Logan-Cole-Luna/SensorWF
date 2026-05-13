"""
run_sat.py — SensorWF: Satellite Telemetry Core Pipeline (M1-M5)
================================================================
Runs the five core SensorWF modules on KISPE SATLL SCOTTI v2 telemetry
using the same domain-agnostic implementations as the ECG and Climate runners.

Pipeline stages
───────────────
  M1  DataIngestion      — SatelliteAdapter.load()
  M2  QualityAssessment  — pipeline_core.run_quality_assessment()
  M3  FeatureEngineering — evaluator.build_generic_feature_matrix()
  M4  SemanticAnnotation — pipeline_core.build_semantic_kg() via adapter.get_class_map()
  M5  ProvenanceExport   — ProvenanceRecorder (PROV-O/ProvONE, SHA-256)

Data discovery
──────────────
  data/<Family>/<Session>/Archive_*_Analysis.txt   ← preferred
  data/<Family>/<Session>/Archive_*_Debug.txt      ← fallback
  data/<Session>/Archive_*_Analysis.txt            ← flat layout (legacy)

Anomaly detection use case (E1 + E2)
──────────────────────────────────────
  python use_cases/sat_anomaly.py [--families AccelerometerTest ...]

Outputs
───────
  results/satellite/<Family>/
    cdh_clean.csv         CDH channels (for sat_anomaly.py)
    adcs_clean.csv        ADCS channels (for sat_anomaly.py)
    quality_report.json   M2 quality report
    *.png                 channel overview figures
    semantic/             M4: KG artefacts
  results/satellite/
    workflow_spec.ttl     M5: static ProvONE workflow spec
    provenance.ttl        M5: runtime PROV-O trace

Usage
─────
  python run_sat.py [--families AccelerometerTest GyroTest ...]
                    [--ontology results/ontologies/satellitesystem.owl]
"""

from __future__ import annotations

import argparse
import json
import os
import glob
import time
from datetime import datetime, timezone

import pandas as pd

from scripts.adapters.satellite import SatelliteAdapter
from scripts.pipeline_core import (
    run_quality_assessment,
    save_quality_report,
    build_semantic_kg,
    plot_channels,
)
from scripts.evaluator import build_generic_feature_matrix
from scripts.workflow import save_workflow_spec, M1_INGEST, M2_QUALITY, M3_FEATURES, M4_SEMANTIC, M5_PROVENANCE
from scripts.provenance_recorder import ProvenanceRecorder

# ── CLI ────────────────────────────────────────────────────────────────────────
_ap = argparse.ArgumentParser(
    description="SensorWF satellite domain pipeline (M1-M5)"
)
_ap.add_argument("--families", nargs="*", metavar="NAME")
_ap.add_argument("--seed",     type=int, default=42)
_ap.add_argument("--ontology", default=os.path.join("results", "ontologies", "satellitesystem.owl"))
_ARGS = _ap.parse_args()

DATA_DIR      = "data"
RESULTS_DIR   = os.path.join("results", "satellite")
ONTOLOGY_PATH = _ARGS.ontology
os.makedirs(RESULTS_DIR, exist_ok=True)

_stage_times: dict[str, float] = {}

spec_path = os.path.join(RESULTS_DIR, "workflow_spec.ttl")
save_workflow_spec(spec_path)

run_ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
recorder = ProvenanceRecorder(workflow_run_id=f"sensorwf_sat_{run_ts}")


# ── File discovery ─────────────────────────────────────────────────────────────

def _pick_best_session(sessions: list[tuple[str, str]]) -> tuple[str, str]:
    best, best_size = sessions[0], -1
    for s in sessions:
        try:
            sz = os.path.getsize(s[0]) if s[0] else 0
        except OSError:
            sz = 0
        if sz > best_size:
            best, best_size = s, sz
    return best


def _discover_experiments(data_dir: str) -> list[tuple[str, str | None, str | None]]:
    found: list[tuple[str, str | None, str | None]] = []
    for entry in sorted(os.scandir(data_dir), key=lambda e: e.name):
        if not entry.is_dir() or entry.name == "misc":
            continue
        sessions: list[tuple[str, str]] = []
        for ses in sorted(os.scandir(entry.path), key=lambda e: e.name):
            if not ses.is_dir():
                continue
            a = sorted(glob.glob(os.path.join(ses.path, "*_Analysis.txt")))
            d = sorted(glob.glob(os.path.join(ses.path, "*_Debug.txt")))
            if a:
                sessions.append((a[0], d[0] if d else ""))
            elif d:
                sessions.append(("", d[0]))
        if sessions:
            best = _pick_best_session(sessions)
            found.append((entry.name, best[0] or None, best[1] or None))
            continue
        # Flat layout
        a = sorted(glob.glob(os.path.join(entry.path, "*_Analysis.txt")))
        d = sorted(glob.glob(os.path.join(entry.path, "*_Debug.txt")))
        if a:
            found.append((entry.name, a[0], d[0] if d else None))
        elif d:
            found.append((entry.name, None, d[0]))
    return found


experiments = _discover_experiments(DATA_DIR)
if _ARGS.families:
    allowed = {f.lower() for f in _ARGS.families}
    experiments = [e for e in experiments if e[0].lower() in allowed]

if not experiments:
    print(f"No experiments found in {DATA_DIR}/")
    raise SystemExit(1)

print("=" * 70)
print("SensorWF — Satellite Telemetry Domain Pipeline (M1-M5)")
print(f"Found {len(experiments)} experiment family/families   |   Run: {run_ts}")
print("=" * 70)
for name, af, df_file in experiments:
    print(f"  {name:<30} ← {os.path.relpath(af or df_file or '?')}")


for exp_name, analysis_file, debug_file in experiments:
    output_dir = os.path.join(RESULTS_DIR, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    sem_dir = os.path.join(output_dir, "semantic")

    print(f"\n{'─' * 70}")
    print(f"Experiment : {exp_name}")
    print(f"{'─' * 70}")

    # ── M1: DataIngestion ─────────────────────────────────────────────────────
    print("\n  [M1] DataIngestion …")
    _t0 = time.perf_counter()
    adapter = SatelliteAdapter()
    try:
        combined = adapter.load(analysis_file or "", debug_path=debug_file)
    except Exception as exc:
        print(f"  ERROR in M1 for {exp_name}: {exc} — skipping.")
        continue
    _stage_times["M1"] = _stage_times.get("M1", 0.0) + (time.perf_counter() - _t0)
    print(f"        Rows: {len(combined)}  CDH channels: {len(adapter.cdh_channels)}  "
          f"ADCS channels: {len(adapter.adcs_channels)}")

    m1_start = datetime.now(timezone.utc)
    cdh, adcs = adapter.split(combined)
    cdh_path  = os.path.join(output_dir, "cdh_clean.csv")
    adcs_path = os.path.join(output_dir, "adcs_clean.csv")
    cdh.to_csv(cdh_path,  index=False)
    adcs.to_csv(adcs_path, index=False)
    m1_end = datetime.now(timezone.utc)

    recorder.record(M1_INGEST,
        inputs  = {"analysis_file": {"path": analysis_file or ""},
                   "debug_file":    {"path": debug_file    or ""}},
        outputs = {"cdh_clean":  {"path": cdh_path,  "rows": len(cdh)},
                   "adcs_clean": {"path": adcs_path, "rows": len(adcs)}},
        start_time=m1_start, end_time=m1_end, experiment=exp_name,
    )

    # ── M2: QualityAssessment ─────────────────────────────────────────────────
    print("\n  [M2] QualityAssessment …")
    _t0 = time.perf_counter()
    m2_start = datetime.now(timezone.utc)
    quality_report = run_quality_assessment(combined, adapter.channels, adapter.get_quality_config())
    quality_path   = os.path.join(output_dir, "quality_report.json")
    save_quality_report(quality_report, quality_path)
    _stage_times["M2"] = _stage_times.get("M2", 0.0) + (time.perf_counter() - _t0)
    m2_end = datetime.now(timezone.utc)
    print(f"        Stuck: {quality_report.get('stuck_channels', [])}  "
          f"Z-flags: {len(quality_report.get('zscore_flags', {}))}")

    recorder.record(M2_QUALITY,
        inputs  = {"combined_df":    {"rows": len(combined)}},
        outputs = {"quality_report": {"path": quality_path}},
        start_time=m2_start, end_time=m2_end, experiment=exp_name,
    )

    # ── M3: FeatureEngineering ────────────────────────────────────────────────
    print("\n  [M3] FeatureEngineering …")
    _t0 = time.perf_counter()
    m3_start = datetime.now(timezone.utc)
    feat_cfg  = adapter.get_feature_config()
    X, feat_names = build_generic_feature_matrix(
        combined, feat_cfg["channels"] or adapter.channels, window=feat_cfg["window"]
    )
    _stage_times["M3"] = _stage_times.get("M3", 0.0) + (time.perf_counter() - _t0)
    m3_end = datetime.now(timezone.utc)
    if X is not None:
        print(f"        Feature matrix: {X.shape[0]} × {X.shape[1]}")
        recorder.record(M3_FEATURES,
            inputs  = {"combined_df": {"rows": len(combined)}},
            outputs = {"feature_matrix": {"shape": list(X.shape)}},
            start_time=m3_start, end_time=m3_end, experiment=exp_name,
        )

    # ── Plots: channel overview ────────────────────────────────────────────────
    print("\n  [plots] Generating channel overview figures …")
    try:
        saved = plot_channels(combined, adapter.get_plot_groups(), output_dir,
                              title_prefix=exp_name)
        print(f"        {len(saved)} figures → {output_dir}/")
    except Exception as exc:
        print(f"  WARNING: plot_channels failed: {exc}")

    ontology_path = adapter.ensure_ontology()


# ── M4: SemanticAnnotation ─────────────────────────────────────────────────────
print(f"\n  [M4] SemanticAnnotation — building knowledge graph …")
print(f"        (Core mode: feature-to-class evidence only; no ML importances.)")
print(f"        Run use_cases/sat_anomaly.py for the ML-weighted enhanced KG.")
semantic_out_dir = os.path.join(RESULTS_DIR, "semantic")
m4_start = datetime.now(timezone.utc)
_t0 = time.perf_counter()
m4_artifacts: dict = {}
try:
    # Use the last experiment's feature names as representative (all families share schema)
    if feat_names:
        m4_artifacts = build_semantic_kg(
            feature_names  = feat_names,
            class_map      = adapter.get_class_map(),
            output_dir     = semantic_out_dir,
            ontology_path  = ontology_path,
            file_prefix    = "sat_",
            domain_uri     = "https://sensorwf.org/ontologies/satellite",
        )
        for k, v in m4_artifacts.items():
            print(f"          {k}: {v}")
except Exception as exc:
    print(f"  WARNING: M4 KG failed: {exc}")
_stage_times["M4"] = time.perf_counter() - _t0
m4_end = datetime.now(timezone.utc)

if m4_artifacts:
    recorder.record(M4_SEMANTIC,
        inputs  = {"ontology_owl": {"path": ontology_path}},
        outputs = {k: {"path": str(v)} for k, v in m4_artifacts.items()},
        start_time=m4_start, end_time=m4_end,
    )

# ── M5: ProvenanceExport ───────────────────────────────────────────────────────
print(f"\n  [M5] ProvenanceExport …")
_t0 = time.perf_counter()
prov_path = os.path.join(RESULTS_DIR, "provenance.ttl")
recorder.save(prov_path)
_stage_times["M5"] = time.perf_counter() - _t0

# ── Runtime summary ────────────────────────────────────────────────────────────
total_s = sum(_stage_times.values())
print("\n" + "=" * 70)
print(f"SensorWF satellite core pipeline complete.  Results → {RESULTS_DIR}/")
print(f"  Workflow spec : {spec_path}")
print(f"  Provenance    : {prov_path}")
print(f"  Experiments   : {len(experiments)}")
print(f"\n  Module times (summed across {len(experiments)} families):")
for stage, secs in _stage_times.items():
    print(f"    {stage:<8} {secs:6.1f}s")
print(f"    {'TOTAL':<8} {total_s:6.1f}s")
print(f"\nTo run anomaly detection (E1+E2):")
print(f"  python use_cases/sat_anomaly.py")
print("=" * 70)
