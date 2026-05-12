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

import argparse
import json
import os
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.adapters.ecg import ECGAdapter
from scripts.pipeline_core import run_quality_assessment, save_quality_report
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
_WINDOW           = 100    # rolling window in samples (2 s at 50 Hz)
_SESSION_MINUTES  = 5.0    # ECG session duration


# ===========================================================================
# M4 -- ECG semantic annotation (lightweight KG builder)
# ===========================================================================

def _build_ecg_kg(
    feature_names: list[str],
    ontology_path: str,
    out_dir: str,
    ml_results: list[dict] | None = None,
) -> None:
    """Build a minimal ontology-linked KG for ECG domain.

    ml_results: optional M5 output; if provided, IF importances weight edges.
    Outputs nodes_csv, edges_csv, and a summary Turtle stub.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ECG class → feature prefix mapping
    class_map = {
        "MLII":                  "ecg:MLII",
        "V5":                    "ecg:V5",
        "d_MLII":                "ecg:MLII",
        "d_V5":                  "ecg:V5",
        "rm_MLII":               "ecg:MLII",
        "rs_MLII":               "ecg:MLII",
        "rm_V5":                 "ecg:V5",
        "rs_V5":                 "ecg:V5",
        "BaselineWander":        "ecg:BaselineWander",
        "ElectrodeDropout":      "ecg:ElectrodeDropout",
        "EMGArtifact":           "ecg:EMGArtifact",
        "PowerLineInterference": "ecg:PowerLineInterference",
    }

    importances: dict[str, float] = {}
    if ml_results:
        for row in ml_results:
            if row.get("detector") == "IsolationForest" and "if_importances" in row:
                imp_list = row["if_importances"]
                fn       = row.get("if_feature_names", feature_names)
                for feat, imp in zip(fn, imp_list):
                    importances[feat] = max(importances.get(feat, 0.0), float(imp))

    nodes = []
    edges = []

    for feat in feature_names:
        nodes.append({"node_id": feat, "node_type": "feature", "label": feat})

    seen_classes: set[str] = set()
    for feat, cls in class_map.items():
        if feat in feature_names and cls not in seen_classes:
            local = cls.split(":")[-1]
            nodes.append({"node_id": cls, "node_type": "ontologyclass", "label": local})
            seen_classes.add(cls)
            edges.append({
                "source":   feat,
                "target":   cls,
                "relation": "evidenceForClass",
                "weight":   importances.get(feat, 0.001),
            })

    if ml_results:
        fault_tags = set(r["tag"] for r in ml_results if "tag" in r)
        for tag in fault_tags:
            nodes.append({"node_id": tag, "node_type": "anomalytag", "label": tag})
            for feat in feature_names[:3]:
                edges.append({
                    "source":   feat,
                    "target":   tag,
                    "relation": "relatedFault",
                    "weight":   importances.get(feat, 0.001),
                })

    pd.DataFrame(nodes).to_csv(os.path.join(out_dir, "ecg_kg_nodes.csv"), index=False)
    pd.DataFrame(edges).to_csv(os.path.join(out_dir, "ecg_kg_edges.csv"), index=False)

    ttl = (
        "@prefix ecg: <https://example.org/sensorwf/ecg#> .\n"
        "@prefix if:  <https://example.org/sensorwf/if#> .\n\n"
    )
    for _, row in pd.DataFrame(edges).iterrows():
        ttl += (f"<{row['source']}> if:evidenceForClass <{row['target']}> ;\n"
                f"    if:featureImportance {row['weight']:.6f} .\n")
    with open(os.path.join(out_dir, "ecg_kg.ttl"), "w") as fh:
        fh.write(ttl)

    print(f"  [M4] ECG KG: {len(nodes)} nodes, {len(edges)} edges → {out_dir}")


# ===========================================================================
# Per-record core pipeline
# ===========================================================================

def run_ecg_record(
    record_id: str,
    data_dir: str,
    results_dir: str,
    recorder: ProvenanceRecorder | None,
) -> dict:
    """Run M1-M5 on one MIT-BIH record. Returns summary dict."""

    record_dir = os.path.join(results_dir, f"record_{record_id}")
    sem_dir    = os.path.join(record_dir, "semantic")
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
    fig, axes = plt.subplots(len(channels), 1, figsize=(10, 3 * len(channels)),
                             sharex=True)
    if len(channels) == 1:
        axes = [axes]
    for ax, ch in zip(axes, channels):
        ax.plot(df["elapsed_s"].values, df[ch].values, lw=0.5, color="#1565C0")
        ax.set_ylabel(ch, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("Elapsed time (s)", fontsize=9)
    fig.suptitle(f"MIT-BIH Record {record_id} -- {_SESSION_MINUTES:.0f}-min ECG Session "
                 f"(Z-score normalised, {adapter.target_hz:.0f} Hz)", fontsize=10)
    fig.savefig(os.path.join(record_dir, "plots", "01_signal_overview.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── M3: Feature Engineering ─────────────────────────────────────────
    t0 = time.time()
    X_clean, feat_names = build_generic_feature_matrix(df, channels, window=_WINDOW)
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
            params={"window": _WINDOW},
        )

    # ── M4: Semantic Annotation ─────────────────────────────────────────
    t0 = time.time()
    _build_ecg_kg(feat_names, adapter.get_ontology_path(), sem_dir)
    t_m6 = time.time() - t0

    if recorder:
        recorder.record_activity(
            "M4_ECG_SemanticAnnotation",
            inputs={"ontology": adapter.get_ontology_path()},
            outputs={"kg_dir": sem_dir},
            params={},
        )

    summary = {
        "record_id":  record_id,
        "n_rows":     len(df),
        "n_features": len(feat_names),
        "timing_s":   {"m1": t_m1, "m2": t_m2, "m3": t_m3, "m6": t_m6},
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
    print(f"\nTo run anomaly detection use case:")
    print(f"  python use_cases/ecg_anomaly.py [--records {' '.join(args.records[:3])} ...]")


if __name__ == "__main__":
    main()
