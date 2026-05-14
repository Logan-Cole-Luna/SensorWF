"""
run_ecg.py -- SensorWF core pipeline for the MIT-BIH Arrhythmia Database.

Core modules executed
---------------------
M1  DataIngestion      : ECGAdapter -- load, downsample (50 Hz), Z-score normalise
M2  QualityAssessment  : generic quality report (NaN, stuck, timing, z-score flags)
M3  FeatureEngineering : raw + diff + rolling stats (window=100 @ 50 Hz)
M4  SemanticAnnotation : map feature names → ecg.owl ontology classes
M5  ProvenanceExport   : PROV-O/ProvONE trace to results/ECG/provenance.ttl

Records processed (20 records spanning diverse arrhythmia types)
-----------------------------------------------------------------
Normal sinus rhythm:  100, 101, 103, 112, 113, 115
Bundle branch block:  106, 108, 109, 111, 118
PVC / bigeminy:       105, 119, 200, 205, 215
AF / complex:         201, 208, 213, 221

Anomaly detection use case (E1 + E2)
--------------------------------------
  Run after the core pipeline to add fault injection and ML evaluation:
    python use_cases/ecg_anomaly.py [--records 100 106 ...]

Usage
-----
  python run_ecg.py [--records 100 106 108] [--data-dir data/ECG]
                    [--results-dir results/ECG] [--seed 42]
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import time
from typing import Any

import numpy as np
import pandas as pd

from scripts.adapters.ecg import ECGAdapter
from scripts.pipeline_core import (
    run_quality_assessment,
    save_quality_report,
    build_semantic_kg,
    plot_channels,
)
from scripts.evaluator import build_generic_feature_matrix
from scripts.provenance_recorder import ProvenanceRecorder

_DEFAULT_RECORDS  = [
    # Normal sinus rhythm
    "100", "101", "103", "112", "113", "115",
    # Bundle branch block
    "106", "108", "109", "111", "118",
    # PVC / bigeminy / trigiminy
    "105", "119", "200", "205", "215",
    # Atrial fibrillation / complex ventricular
    "201", "208", "213", "221",
]
_DATA_DIR         = os.path.join("data", "ECG")
_RESULTS_DIR      = os.path.join("results", "ECG", "mitdb")
_SEED             = 42
_SESSION_MINUTES  = 5.0    # ECG session duration


# ===========================================================================
# Per-record core pipeline
# ===========================================================================

def run_ecg_record(
    record_id: str,
    data_dir: str,
    results_dir: str,
    recorder: ProvenanceRecorder | None,
    rerun: bool = False,
) -> dict:
    """Run M1-M5 on one MIT-BIH record. Returns summary dict."""

    record_dir   = os.path.join(results_dir, f"record_{record_id}")
    sem_dir      = os.path.join(record_dir, "semantic")
    summary_path = os.path.join(record_dir, "run_summary.json")

    if not rerun and os.path.isfile(summary_path):
        with open(summary_path) as fh:
            cached = json.load(fh)
        print(f"  [CACHE] record {record_id} — skipping (pass --rerun to force)")
        return cached

    os.makedirs(record_dir, exist_ok=True)
    os.makedirs(sem_dir, exist_ok=True)
    os.makedirs(os.path.join(record_dir, "plots"), exist_ok=True)

    # ── M1: Data Ingestion ──────────────────────────────────────────────
    t0 = time.time()
    adapter  = ECGAdapter(session_minutes=_SESSION_MINUTES)
    csv_path = os.path.join(data_dir, f"{record_id}.csv")
    ann_path = os.path.join(data_dir, f"{record_id}annotations.txt")

    if not os.path.exists(csv_path):
        print(f"  [M1] SKIP record {record_id}: {csv_path} not found")
        return {}

    df = adapter.load(csv_path)
    channels = adapter.channels
    feat_cfg = adapter.get_feature_config()
    print(f"  [M1] Record {record_id}: {len(df)} rows × {len(channels)} channels "
          f"@ ~{adapter.target_hz:.0f} Hz")

    ann_summary: dict[str, Any] = {}
    if os.path.exists(ann_path):
        ann_df = adapter.load_annotations(ann_path)
        ann_summary = {
            "n_beats_total":    len(ann_df),
            "n_normal":         int((~ann_df["is_abnormal"]).sum()),
            "n_abnormal":       int(ann_df["is_abnormal"].sum()),
            "beat_type_counts": ann_df["beat_type"].value_counts().to_dict(),
        }

    t_m1 = time.time() - t0
    if recorder:
        recorder.record_activity(
            "M1_ECG_DataIngestion",
            inputs={"csv_path": csv_path},
            outputs={"n_rows": len(df), "channels": channels},
            params={"record_id": record_id, "target_hz": adapter.target_hz,
                    "session_minutes": _SESSION_MINUTES},
        )

    # ── M2: Quality Assessment ──────────────────────────────────────────
    t0 = time.time()
    quality_cfg    = adapter.get_quality_config()
    quality_report = run_quality_assessment(df, channels, quality_cfg)
    quality_report["annotations"] = ann_summary
    save_quality_report(quality_report, os.path.join(record_dir, "quality_report.json"))
    t_m2 = time.time() - t0

    if recorder:
        recorder.record_activity(
            "M2_ECG_QualityAssessment",
            inputs={"n_rows": len(df)},
            outputs={"quality_report": os.path.join(record_dir, "quality_report.json")},
            params=quality_cfg,
        )

    clean_csv = os.path.join(record_dir, "signal_clean.csv")
    df.to_csv(clean_csv, index=False)

    # ── Overview plot ────────────────────────────────────────────────────
    plot_channels(df, adapter.get_plot_groups(),
                  os.path.join(record_dir, "plots"),
                  title_prefix=f"MIT-BIH {record_id}")

    # ── M3: Feature Engineering ─────────────────────────────────────────
    t0 = time.time()
    X_clean, feat_names = build_generic_feature_matrix(df, channels, window=feat_cfg["window"])
    t_m3 = time.time() - t0
    if X_clean is None:
        print(f"  [M3] SKIP record {record_id}: no usable features")
        return {}
    print(f"  [M3] Feature matrix: {X_clean.shape} ({len(feat_names)} features)")

    if recorder:
        recorder.record_activity(
            "M3_ECG_FeatureEngineering",
            inputs={"n_rows": len(df)},
            outputs={"X_shape": list(X_clean.shape), "n_features": len(feat_names)},
            params={"window": feat_cfg["window"]},
        )

    # ── M4: Semantic Annotation ─────────────────────────────────────────
    t_m4_start = time.time()
    m4_arts = build_semantic_kg(
        feature_names = feat_names,
        class_map     = adapter.get_class_map(),
        output_dir    = sem_dir,
        ontology_path = adapter.ensure_ontology(),
        file_prefix   = "ecg_",
        domain_uri    = "https://sensorwf.org/ontologies/ecg",
    )
    t_m4 = time.time() - t_m4_start
    print(f"  [M4] ECG KG: {len(m4_arts)} artefacts → {sem_dir}")

    if recorder:
        recorder.record_activity(
            "M4_ECG_SemanticAnnotation",
            inputs={"ontology": adapter.ensure_ontology()},
            outputs={k: v for k, v in m4_arts.items()},
            params={},
        )

    summary = {
        "record_id":  record_id,
        "n_rows":     len(df),
        "n_features": len(feat_names),
        "timing_s":   {"m1": t_m1, "m2": t_m2, "m3": t_m3, "m4": t_m4},
    }
    with open(os.path.join(record_dir, "run_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    return summary


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SensorWF ECG core pipeline (M1-M5)")
    parser.add_argument("--records",     nargs="+", default=_DEFAULT_RECORDS)
    parser.add_argument("--data-dir",    default=_DATA_DIR)
    parser.add_argument("--results-dir", default=_RESULTS_DIR)
    parser.add_argument("--seed",        type=int, default=_SEED,
                        help="Reserved for reproducibility (not used by core modules)")
    parser.add_argument("--rerun",       action="store_true",
                        help="Ignore cached run_summary.json and reprocess every record")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    prov_path = os.path.join(args.results_dir, "provenance.ttl")
    recorder  = ProvenanceRecorder(workflow_uri="sensorwf:ECGWorkflow",
                                   domain="biomedical_ecg")

    print("=" * 64)
    print("SensorWF -- ECG Core Pipeline (M1-M5)")
    print(f"Records: {args.records}")
    print(f"Session: {_SESSION_MINUTES:.0f} min @ 50 Hz = "
          f"{int(_SESSION_MINUTES * 60 * 50):,} samples")
    print("=" * 64)

    all_summaries = []
    for rec_id in args.records:
        print(f"\n── Record {rec_id} {'─' * 40}")
        summary = run_ecg_record(
            record_id=rec_id,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            recorder=recorder,
            rerun=args.rerun,
        )
        if summary:
            all_summaries.append(summary)

    # ── M5: Provenance Export ───────────────────────────────────────────
    recorder.save(prov_path)
    print(f"\n[M5] Provenance trace → {prov_path}")

    with open(os.path.join(args.results_dir, "run_summaries.json"), "w") as fh:
        json.dump(all_summaries, fh, indent=2)

    print(f"\nDone. Results in: {args.results_dir}")
    print(f"  Records processed: {len(all_summaries)}/{len(args.records)}")
    print(f"\nTo run all phases (core + CWL + report + anomaly):")
    print(f"  python use_case.py --domain ecg")
    print(f"To run anomaly detection only:")
    print(f"  python use_cases/ecg_anomaly.py [--records {' '.join(args.records[:3])} ...]")


if __name__ == "__main__":
    main()
