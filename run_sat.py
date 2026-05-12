"""
run_sat.py — SensorWF: Satellite Telemetry Core Pipeline (M1-M5)
================================================================
Runs the five core SensorWF modules on KISPE SATLL SCOTTI v2 telemetry.
The satellite domain is ingested via SatelliteAdapter (DomainAdapter), so
M2-M5 use the same domain-agnostic implementations as ECG and Climate.

Pipeline stages
───────────────
  M1  DataIngestion      — SatelliteAdapter: parse SCOTTI, merge CDH+ADCS
  M2  QualityAssessment  — pipeline_core.run_quality_assessment (generic)
  M3  FeatureEngineering — evaluator.build_generic_feature_matrix (generic)
  M4  SemanticAnnotation — scripts/satellite/semantic_kg.py
  M5  ProvenanceExport   — ProvenanceRecorder (PROV-O/ProvONE)

Data discovery
──────────────
  data/<Family>/<Session>/Archive_*_Analysis.txt   ← preferred
  data/<Family>/<Session>/Archive_*_Debug.txt      ← fallback
  data/<Session>/Archive_*_Analysis.txt            ← flat layout (legacy)

  The session with the largest archive file per family is used.

Satellite-specific supplementary checks
─────────────────────────────────────────
  In addition to generic M2, run_detection() from scripts/satellite/detector.py
  produces a domain-specific detection_report.json (OBDH trend, wheel-gyro
  correlation, cross-sensor correlation breakdown).  These enrich M4 but are
  not part of the reusable core.

Anomaly detection use case (E1 + E2)
──────────────────────────────────────
  python use_cases/sat_anomaly.py [--families AccelerometerTest ...]

Outputs
───────
  results/satellite/<Family>/
    cdh_clean.csv         M1 output: CDH channels only
    adcs_clean.csv        M1 output: ADCS channels only
    quality_report.json   M2 generic quality report
    detection_report.json M2 satellite-specific statistical checks
    *.png                 telemetry visualisations
  results/satellite/
    workflow_spec.ttl     M5: static ProvONE workflow spec
    provenance.ttl        M5: runtime PROV-O trace with SHA-256 checksums
    semantic/             M4: knowledge graph artefacts

Usage
─────
  python run_sat.py [--families AccelerometerTest GyroTest ...]
                    [--seed 42]
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
from scripts.satellite.detector import run_detection
from scripts.satellite.plotter  import run_all_plots
from scripts.satellite.ontology import generate_satellite_ontology
from scripts.satellite.semantic_kg import build_semantic_graph
from scripts.pipeline_core import run_quality_assessment, save_quality_report
from scripts.evaluator import build_generic_feature_matrix
from scripts.workflow import save_workflow_spec, M1_INGEST, M2_QUALITY, M3_FEATURES, M4_SEMANTIC, M5_PROVENANCE
from scripts.provenance_recorder import ProvenanceRecorder

# ── CLI arguments ──────────────────────────────────────────────────────────────
_ap = argparse.ArgumentParser(
    description="SensorWF satellite domain pipeline: ingest → quality → features → KG → provenance"
)
_ap.add_argument("--families", nargs="*", metavar="NAME",
                 help="Restrict to named experiment families")
_ap.add_argument("--seed", type=int, default=42,
                 help="Reserved for reproducibility (not used by core modules)")
_ap.add_argument("--ontology", default=os.path.join("results", "ontologies", "satellitesystem.owl"),
                 help="Path to OWL ontology file (default: results/ontologies/satellitesystem.owl)")
_ARGS = _ap.parse_args()

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR      = "data"
RESULTS_DIR   = os.path.join("results", "satellite")
ONTOLOGY_PATH = _ARGS.ontology

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── M1 prerequisite: ensure satellite OWL ontology exists ─────────────────────
if not os.path.isfile(ONTOLOGY_PATH):
    print(f"  [M4-pre] Generating satellite ontology → {ONTOLOGY_PATH}")
    generate_satellite_ontology(ONTOLOGY_PATH)

# Per-module wall-clock timing accumulator
_stage_times: dict[str, float] = {}

# ── Emit static workflow specification (ProvONE) ───────────────────────────────
spec_path = os.path.join(RESULTS_DIR, "workflow_spec.ttl")
save_workflow_spec(spec_path)

# ── Initialise provenance recorder ────────────────────────────────────────────
run_ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
recorder = ProvenanceRecorder(workflow_run_id=f"sensorwf_sat_{run_ts}")


# ── File discovery ─────────────────────────────────────────────────────────────

def _pick_best_session(sessions: list[tuple[str, str]]) -> tuple[str, str]:
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
    found: list[tuple[str, str, str | None]] = []
    skip_families = {"misc"}

    for family_entry in sorted(os.scandir(data_dir), key=lambda e: e.name):
        if not family_entry.is_dir() or family_entry.name in skip_families:
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
            found.append((family_entry.name, best[0] or None, best[1] or None))
            continue

        # Flat layout (legacy)
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
print("=" * 70)
print("SensorWF — Satellite Telemetry Domain Pipeline (M1-M5)")
print(f"Found {len(experiments)} experiment family/families   |   Run: {run_ts}")
print("=" * 70)
for name, af, df_file in experiments:
    src = af or df_file or "?"
    print(f"  {name:<30} ← {os.path.relpath(src)}")

for exp_name, analysis_file, debug_file in experiments:
    output_dir = os.path.join(RESULTS_DIR, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'─' * 70}")
    print(f"Experiment : {exp_name}")
    print(f"Output     : {output_dir}/")
    print(f"{'─' * 70}")

    # ── M1: DataIngestion ─────────────────────────────────────────────────────
    print("\n  [M1] DataIngestion — parsing SCOTTI archive …")
    _t0 = time.perf_counter()
    adapter = SatelliteAdapter()
    try:
        combined = adapter.load(
            analysis_file or "",
            debug_path=debug_file,
        )
    except Exception as exc:
        print(f"  ERROR in M1 for {exp_name}: {exc} — skipping.")
        continue
    _stage_times["M1_ingestion"] = _stage_times.get("M1_ingestion", 0.0) + (time.perf_counter() - _t0)

    print(f"        Combined rows    : {len(combined)}")
    print(f"        CDH channels     : {len(adapter.cdh_channels)}")
    print(f"        ADCS channels    : {len(adapter.adcs_channels)}")
    print(f"        Total channels   : {len(adapter.channels)}")

    m1_start = datetime.now(timezone.utc)

    # Split back into CDH/ADCS for the anomaly use case and satellite-specific checks
    cdh, adcs = adapter.split(combined)

    # Save cleaned CSVs (consumed by sat_anomaly.py / injector)
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

    # ── M2: QualityAssessment (generic) ───────────────────────────────────────
    print("\n  [M2] QualityAssessment — generic quality checks …")
    _t0 = time.perf_counter()
    m2_start = datetime.now(timezone.utc)
    quality_cfg    = adapter.get_quality_config()
    quality_report = run_quality_assessment(combined, adapter.channels, quality_cfg)
    quality_path   = os.path.join(output_dir, "quality_report.json")
    save_quality_report(quality_report, quality_path)
    _stage_times["M2_quality"] = _stage_times.get("M2_quality", 0.0) + (time.perf_counter() - _t0)
    m2_end = datetime.now(timezone.utc)

    print(f"        Stuck channels   : {quality_report.get('stuck_channels', [])}")
    print(f"        Z-score flags    : {len(quality_report.get('zscore_flags', {}))}")
    print(f"        Stuck sensors    : {adapter.get_stuck_sensors()}")

    recorder.record(M2_QUALITY,
        inputs  = {"combined_df":    {"rows": len(combined)}},
        outputs = {"quality_report": {"path": quality_path}},
        start_time=m2_start, end_time=m2_end, experiment=exp_name,
    )

    # Satellite-specific supplementary detection (OBDH trend, wheel-gyro corr, etc.)
    print("\n  [M2+] StatisticalDetection — satellite-specific checks …")
    try:
        det_report = run_detection(cdh, adcs, adapter.get_stuck_sensors())
        obdh = det_report["obdh_trend"]
        ct   = det_report["cdh_timing"]
        print(f"        OBDH slope       : {obdh['slope_per_min']:.3f} °C/min  R²={obdh['r_squared']:.4f}")
        print(f"        CDH interval     : mean={ct['mean_s']:.3f}s  max_gap={ct['max_gap_s']:.3f}s")

        def _serialisable(r: dict) -> dict:
            skip_keys = {"zscore_masks", "correlation_breakdown", "wheel_gyro_correlation"}
            out: dict = {}
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

        det_path = os.path.join(output_dir, "detection_report.json")
        with open(det_path, "w") as fh:
            json.dump(_serialisable(det_report), fh, indent=2, default=str)
    except Exception as exc:
        print(f"  ERROR in supplementary detection for {exp_name}: {exc}")

    # ── M3: FeatureEngineering (generic) ──────────────────────────────────────
    print("\n  [M3] FeatureEngineering — building feature matrix …")
    _t0 = time.perf_counter()
    m3_start = datetime.now(timezone.utc)
    feat_cfg = adapter.get_feature_config()
    window   = feat_cfg["window"]
    channels = feat_cfg["channels"] or adapter.channels
    X, feat_names = build_generic_feature_matrix(combined, channels, window=window)
    _stage_times["M3_features"] = _stage_times.get("M3_features", 0.0) + (time.perf_counter() - _t0)
    m3_end = datetime.now(timezone.utc)

    if X is not None:
        print(f"        Feature matrix   : {X.shape[0]} × {X.shape[1]} ({len(feat_names)} features)")
    else:
        print(f"        Feature matrix   : skipped (insufficient data)")

    if X is not None:
        recorder.record(M3_FEATURES,
            inputs  = {"combined_df":    {"rows": len(combined)}},
            outputs = {"feature_matrix": {"shape": list(X.shape), "n_features": len(feat_names)}},
            start_time=m3_start, end_time=m3_end, experiment=exp_name,
        )

    # ── Telemetry visualisations ───────────────────────────────────────────────
    print("\n  [plots] Generating telemetry visualisations …")
    try:
        saved = run_all_plots(cdh, adcs, output_dir)
        print(f"        {len(saved)} figures → {output_dir}/")
    except Exception as exc:
        print(f"  ERROR plotting {exp_name}: {exc}")

    # Regenerate ontology with actual channel lists so M4 KG uses data-grounded classes
    generate_satellite_ontology(ONTOLOGY_PATH, adapter.cdh_channels, adapter.adcs_channels)


# ── M4: SemanticAnnotation (runs across all families at once) ──────────────────
print(f"\n  [M4] SemanticAnnotation — building ontology-linked knowledge graph …")
print(f"        Note: core mode uses feature-based class mapping (no ML importances).")
print(f"        Run use_cases/sat_anomaly.py to produce the ML-weighted enhanced KG.")
semantic_out_dir = os.path.join(RESULTS_DIR, "semantic")
m4_start = datetime.now(timezone.utc)
_t0 = time.perf_counter()
m4_artifacts: dict = {}
try:
    m4_artifacts = build_semantic_graph(
        results_dir   = RESULTS_DIR,
        ontology_path = ONTOLOGY_PATH,
        output_dir    = semantic_out_dir,
    )
    print(f"        KG artifacts → {semantic_out_dir}/")
    for k, v in m4_artifacts.items():
        print(f"          {k}: {v}")
except Exception as exc:
    print(f"  INFO: M4 KG skipped (requires ML results — run sat_anomaly.py): {exc}")
_stage_times["M4_semantic"] = time.perf_counter() - _t0
m4_end = datetime.now(timezone.utc)

if m4_artifacts:
    recorder.record(M4_SEMANTIC,
        inputs  = {"ontology_owl": {"path": ONTOLOGY_PATH}},
        outputs = {k: {"path": str(v)} for k, v in m4_artifacts.items()},
        start_time=m4_start, end_time=m4_end,
    )

# ── M5: ProvenanceExport ───────────────────────────────────────────────────────
print(f"\n  [M5] ProvenanceExport — writing PROV-O / ProvONE trace …")
_t0 = time.perf_counter()
prov_path = os.path.join(RESULTS_DIR, "provenance.ttl")
recorder.save(prov_path)
_stage_times["M5_provenance"] = time.perf_counter() - _t0

# ── Runtime summary ────────────────────────────────────────────────────────────
total_s = sum(_stage_times.values())
print("\n" + "=" * 70)
print(f"SensorWF satellite core pipeline complete.  Results → {RESULTS_DIR}/")
print(f"  Workflow spec  : {spec_path}")
print(f"  Provenance     : {prov_path}")
print(f"  Semantic KG    : {semantic_out_dir}/")
print(f"  Experiments    : {len(experiments)}")
print(f"\n  Module wall-clock times (summed across {len(experiments)} families):")
for stage, secs in _stage_times.items():
    print(f"    {stage:<20} {secs:6.1f}s")
print(f"    {'TOTAL':<20} {total_s:6.1f}s")
print(f"\nTo run anomaly detection use case (E1 + E2):")
print(f"  python use_cases/sat_anomaly.py")
print("=" * 70)
