"""
run_ecg.py -- SensorWF use-case pipeline for the MIT-BIH Arrhythmia Database.

Modules executed
----------------
M1  DataIngestion      : ECGAdapter -- load, downsample (50 Hz), Z-score normalise
M2  QualityAssessment  : generic quality report (NaN, stuck, timing, z-score flags)
M3  FeatureEngineering : raw + diff + rolling stats (window=100 @ 50 Hz)
M4  FaultInjection     : 6 ECG artifact types × 3 tiers × 2 variants = 36 per record
M5  AnomalyDetection   : ZScore, RobustRollingZScore, IsolationForest, Autoencoder
M6  SemanticAnnotation : map IF importances → ecg.owl classes
M7  ProvenanceExport   : PROV-O/ProvONE trace to results/ECG/provenance.ttl

Records processed
-----------------
100: Normal sinus rhythm + occasional PVC
106: Right bundle branch block
108: Right bundle branch block + PVC
200: Very frequent PVCs (bigeminy)
201: Atrial fibrillation + PVCs

Usage
-----
  python run_ecg.py [--records 100 106 108] [--data-dir data/ECG]
                    [--results-dir results/ECG] [--seed 42] [--fast]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from scripts.adapters.ecg import ECGAdapter
from scripts.pipeline_core import run_quality_assessment, save_quality_report
from scripts.evaluator import (
    build_generic_feature_matrix,
    run_generic_ml_evaluation,
    print_ml_summary as print_metrics_summary,
)
from scripts.provenance_recorder import ProvenanceRecorder

# ── Tier parameters (shared with satellite injector) ───────────────────────
_TIERS = ["easy", "medium", "hard"]
_TIER_AMP  = {"easy": 1.45, "medium": 1.05, "hard": 0.72}
_TIER_DUR  = {"easy": 0.30, "medium": 0.95, "hard": 1.55}

_DEFAULT_RECORDS  = ["100", "106", "108", "200", "201"]
_DATA_DIR         = os.path.join("data", "ECG")
_RESULTS_DIR      = os.path.join("results", "ECG", "mitdb")
_N_VARIANTS       = 2
_SEED             = 42
_WINDOW           = 100    # rolling window in samples (2 s at 50 Hz)
_SESSION_MINUTES  = 5.0    # ECG session duration


# ===========================================================================
# M4 -- ECG fault injection
# ===========================================================================

def _pick_window(n: int, rng: np.random.Generator, base_frac: float, dur_scale: float):
    width = max(10, int(round(n * base_frac * dur_scale)))
    width = min(width, max(10, n - 2))
    if n <= width + 1:
        return 0, n
    lo = max(1, n // 8)
    max_start = n - width
    if max_start <= lo:
        start = max(0, max_start // 2)
    else:
        start = int(rng.integers(lo, max_start))
    return start, start + width


def _signal_std(df: pd.DataFrame, channels: list[str]) -> float:
    vals = []
    for ch in channels:
        if ch in df.columns:
            s = pd.to_numeric(df[ch], errors="coerce").dropna()
            if len(s) > 1:
                vals.append(float(s.std()))
    return float(np.mean(vals)) if vals else 1.0


def inject_ecg_fault(
    df: pd.DataFrame,
    tag: str,
    tier: str,
    rng: np.random.Generator,
    channels: list[str],
    sample_hz: float = 50.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply one ECG artifact type to a clean DataFrame.

    Returns modified DataFrame and a label dict
    {start_idx, end_idx, severity, affected_channels}.
    """
    out   = df.copy()
    n     = len(out)
    amp   = _TIER_AMP[tier]
    dur   = _TIER_DUR[tier]
    sig_s = _signal_std(df, channels)
    sev   = {"easy": "low", "medium": "medium", "hard": "high"}[tier]

    start, end = _pick_window(n, rng, base_frac=0.40, dur_scale=dur)
    affected   = list(channels)  # default: all channels

    if tag == "ecg_baseline_wander":
        # Sinusoidal drift 0.15-0.4 Hz (breathing rate)
        freq = rng.uniform(0.15, 0.40)
        t    = np.arange(end - start, dtype=float) / sample_hz
        drift = amp * sig_s * 1.5 * np.sin(2 * np.pi * freq * t)
        for ch in channels:
            if ch in out.columns:
                out.iloc[start:end, out.columns.get_loc(ch)] += drift

    elif tag == "ecg_electrode_dropout":
        # Zero-clamp one lead
        ch = rng.choice(channels)
        affected = [str(ch)]
        out.iloc[start:end, out.columns.get_loc(ch)] = 0.0

    elif tag == "ecg_emg_burst":
        # Bandpass-like noise: white noise * window
        noise_amp = amp * sig_s * 2.0
        for ch in channels:
            if ch in out.columns:
                noise = rng.normal(0, noise_amp, end - start)
                out.iloc[start:end, out.columns.get_loc(ch)] += noise

    elif tag == "ecg_powerline_noise":
        # 60 Hz sinusoidal (powerline)
        t = np.arange(end - start, dtype=float) / sample_hz
        pl = amp * sig_s * 0.8 * np.sin(2 * np.pi * 60.0 * t)
        for ch in channels:
            if ch in out.columns:
                out.iloc[start:end, out.columns.get_loc(ch)] += pl

    elif tag == "ecg_amplitude_scale":
        # Multiplicative scaling (amplifier error)
        scale = 1.0 + amp * rng.choice([-1, 1]) * rng.uniform(0.4, 0.9)
        ch = rng.choice(channels)
        affected = [str(ch)]
        out.iloc[start:end, out.columns.get_loc(ch)] *= scale

    elif tag == "ecg_lead_inversion":
        # Sign inversion (lead reversal)
        ch = rng.choice(channels)
        affected = [str(ch)]
        out.iloc[start:end, out.columns.get_loc(ch)] *= -1.0

    else:
        raise ValueError(f"Unknown ECG fault tag: {tag}")

    label = {
        "anomaly_type": tag,
        "start_idx":    int(start),
        "end_idx":      int(end - 1),
        "severity":     sev,
        "affected_channels": affected,
    }
    return out, label


# ===========================================================================
# M6 -- ECG semantic annotation (lightweight KG builder)
# ===========================================================================

def _build_ecg_kg(
    ml_results: list[dict],
    feature_names: list[str],
    ontology_path: str,
    out_dir: str,
) -> None:
    """
    Build a minimal ontology-linked KG for ECG domain.
    Outputs nodes_csv, edges_csv, and a summary Turtle stub.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ECG class → feature prefix mapping
    class_map = {
        "MLII":               "ecg:MLII",
        "V5":                 "ecg:V5",
        "d_MLII":             "ecg:MLII",
        "d_V5":               "ecg:V5",
        "rm_MLII":            "ecg:MLII",
        "rs_MLII":            "ecg:MLII",
        "rm_V5":              "ecg:V5",
        "rs_V5":              "ecg:V5",
        "BaselineWander":     "ecg:BaselineWander",
        "ElectrodeDropout":   "ecg:ElectrodeDropout",
        "EMGArtifact":        "ecg:EMGArtifact",
        "PowerLineInterference": "ecg:PowerLineInterference",
    }

    # Collect IF importances from results
    importances: dict[str, float] = {}
    for row in ml_results:
        if row.get("detector") == "IsolationForest" and "if_importances" in row:
            imp_list = row["if_importances"]
            fn       = row.get("if_feature_names", feature_names)
            for feat, imp in zip(fn, imp_list):
                importances[feat] = max(importances.get(feat, 0.0), float(imp))

    nodes = []
    edges = []

    # Feature nodes
    for feat in feature_names:
        nodes.append({"node_id": feat, "node_type": "feature", "label": feat})

    # Ontology class nodes
    seen_classes: set[str] = set()
    for feat, cls in class_map.items():
        if feat in feature_names and cls not in seen_classes:
            local = cls.split(":")[-1]
            nodes.append({"node_id": cls, "node_type": "ontologyclass", "label": local})
            seen_classes.add(cls)
            edges.append({
                "source": feat,
                "target": cls,
                "relation": "evidenceForClass",
                "weight": importances.get(feat, 0.001),
            })

    # Fault tag nodes
    fault_tags = set(r["tag"] for r in ml_results if "tag" in r)
    for tag in fault_tags:
        nodes.append({"node_id": tag, "node_type": "anomalytag", "label": tag})
        for feat in feature_names[:3]:
            edges.append({
                "source": feat,
                "target": tag,
                "relation": "relatedFault",
                "weight": importances.get(feat, 0.001),
            })

    pd.DataFrame(nodes).to_csv(os.path.join(out_dir, "ecg_kg_nodes.csv"), index=False)
    pd.DataFrame(edges).to_csv(os.path.join(out_dir, "ecg_kg_edges.csv"), index=False)

    # Minimal Turtle stub
    ttl = (
        "@prefix ecg: <https://example.org/sensorwf/ecg#> .\n"
        "@prefix if:  <https://example.org/sensorwf/if#> .\n\n"
    )
    for _, row in pd.DataFrame(edges).iterrows():
        ttl += (f"<{row['source']}> if:evidenceForClass <{row['target']}> ;\n"
                f"    if:featureImportance {row['weight']:.6f} .\n")
    with open(os.path.join(out_dir, "ecg_kg.ttl"), "w") as fh:
        fh.write(ttl)

    print(f"  [M6] ECG KG: {len(nodes)} nodes, {len(edges)} edges → {out_dir}")


# ===========================================================================
# Per-record pipeline
# ===========================================================================

def run_ecg_record(
    record_id: str,
    data_dir: str,
    results_dir: str,
    seed: int,
    n_variants: int,
    fast: bool,
    recorder: ProvenanceRecorder | None,
) -> dict:
    """Run M1-M7 on one MIT-BIH record. Returns summary dict."""

    record_dir = os.path.join(results_dir, f"record_{record_id}")
    inj_dir    = os.path.join(record_dir, "injected")
    sem_dir    = os.path.join(record_dir, "semantic")
    os.makedirs(inj_dir, exist_ok=True)
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

    # Load annotations summary if available
    ann_summary: dict[str, Any] = {}
    if os.path.exists(ann_path):
        ann_df = adapter.load_annotations(ann_path)
        ann_summary = {
            "n_beats_total":   len(ann_df),
            "n_normal":        int((~ann_df["is_abnormal"]).sum()),
            "n_abnormal":      int(ann_df["is_abnormal"].sum()),
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
    quality_cfg = adapter.get_quality_config()
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

    # Save clean signal
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
    feat_cfg = adapter.get_feature_config()
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

    # ── M4: Fault Injection ─────────────────────────────────────────────
    t0 = time.time()
    fault_types = adapter.get_fault_types()
    files_generated: list[tuple] = []
    injection_rows: list[dict]   = []
    master_rng = np.random.default_rng(seed)

    tiers = [_TIERS[0]] if fast else _TIERS

    for fault in fault_types:
        tag = fault["tag"]
        for tier in tiers:
            for v in range(1, n_variants + 1):
                rng = np.random.default_rng(
                    int(master_rng.integers(0, 2**31)))
                try:
                    df_inj, label = inject_ecg_fault(
                        df, tag, tier, rng, channels,
                        sample_hz=adapter.target_hz,
                    )
                except Exception as exc:
                    print(f"  [M4] ERROR {tag}/{tier}/v{v}: {exc}")
                    continue

                inj_csv   = os.path.join(inj_dir, f"signal_injected_{tag}_{tier}_v{v}.csv")
                label_csv = os.path.join(inj_dir, f"labels_{tag}_{tier}_v{v}.csv")
                df_inj.to_csv(inj_csv, index=False)
                pd.DataFrame([label]).to_csv(label_csv, index=False)

                files_generated.append((tag, tier, v))
                injection_rows.append({
                    "tag": tag, "tier": tier, "variant": v,
                    "description": fault["description"],
                    **label,
                })

    pd.DataFrame(injection_rows).to_csv(
        os.path.join(inj_dir, "injection_summary.csv"), index=False)
    t_m4 = time.time() - t0
    print(f"  [M4] Injected {len(files_generated)} variants in {t_m4:.1f}s")

    if recorder:
        recorder.record_activity(
            "M4_ECG_FaultInjection",
            inputs={"clean_csv": clean_csv},
            outputs={"n_injected": len(files_generated),
                     "injected_dir": inj_dir},
            params={"n_fault_types": len(fault_types), "tiers": tiers,
                    "n_variants": n_variants, "seed": seed},
        )

    # ── M5: Anomaly Detection ───────────────────────────────────────────
    t0 = time.time()
    ml_results = run_generic_ml_evaluation(
        injected_dir=inj_dir,
        files_generated=files_generated,
        clean_df=df,
        channels=channels,
        window=_WINDOW,
        family=f"ECG_record{record_id}",
    )
    t_m5 = time.time() - t0
    print(f"  [M5] {len(ml_results)} detector evaluations in {t_m5:.1f}s")

    if ml_results:
        ml_df = pd.DataFrame(ml_results)

        # Save full results
        ml_df.to_csv(os.path.join(inj_dir, "ml_results.csv"), index=False)

        # Aggregate by tier + detector
        metrics_cols = ["accuracy", "fpr", "recall", "f1", "auc_roc"]
        avail = [c for c in metrics_cols if c in ml_df.columns]
        tier_rows: list[dict] = []
        for tier in sorted(ml_df["tier"].unique()):
            for det in sorted(ml_df["detector"].unique()):
                sub = ml_df[(ml_df["tier"] == tier) & (ml_df["detector"] == det)]
                if not sub.empty:
                    row = {"tier": tier, "detector": det}
                    row.update(sub[avail].mean().round(4).to_dict())
                    tier_rows.append(row)
        # Total row
        for det in sorted(ml_df["detector"].unique()):
            sub = ml_df[ml_df["detector"] == det]
            row = {"tier": "total", "detector": det}
            row.update(sub[avail].mean().round(4).to_dict())
            tier_rows.append(row)
        tier_df = pd.DataFrame(tier_rows)
        tier_df.to_csv(os.path.join(inj_dir, "ml_metrics_by_tier.csv"), index=False)
        print_metrics_summary(ml_results)

    if recorder:
        recorder.record_activity(
            "M5_ECG_AnomalyDetection",
            inputs={"injected_dir": inj_dir, "n_variants": len(files_generated)},
            outputs={"n_eval_rows": len(ml_results),
                     "metrics_csv": os.path.join(inj_dir, "ml_metrics_by_tier.csv")},
            params={"window": _WINDOW, "train_frac": 0.60},
        )

    # ── M6: Semantic Annotation ─────────────────────────────────────────
    t0 = time.time()
    _build_ecg_kg(ml_results, feat_names, adapter.get_ontology_path(), sem_dir)
    t_m6 = time.time() - t0

    if recorder:
        recorder.record_activity(
            "M6_ECG_SemanticAnnotation",
            inputs={"ontology": adapter.get_ontology_path()},
            outputs={"kg_dir": sem_dir},
            params={},
        )

    summary = {
        "record_id": record_id,
        "n_rows":     len(df),
        "n_features": len(feat_names),
        "n_injected": len(files_generated),
        "n_ml_evals": len(ml_results),
        "timing_s":   {"m1": t_m1, "m2": t_m2, "m3": t_m3,
                       "m4": t_m4, "m5": t_m5, "m6": t_m6},
    }
    with open(os.path.join(record_dir, "run_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    return summary


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="SensorWF ECG domain pipeline")
    parser.add_argument("--records",     nargs="+", default=_DEFAULT_RECORDS)
    parser.add_argument("--data-dir",    default=_DATA_DIR)
    parser.add_argument("--results-dir", default=_RESULTS_DIR)
    parser.add_argument("--seed",        type=int, default=_SEED)
    parser.add_argument("--n-variants",  type=int, default=_N_VARIANTS)
    parser.add_argument("--fast",        action="store_true",
                        help="Run easy tier only (faster for testing)")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    prov_path = os.path.join(args.results_dir, "provenance.ttl")
    recorder  = ProvenanceRecorder(workflow_uri="sensorwf:ECGWorkflow",
                                   domain="biomedical_ecg")

    print("=" * 64)
    print("SensorWF -- ECG Domain Pipeline (MIT-BIH Arrhythmia Database)")
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
            seed=args.seed,
            n_variants=args.n_variants,
            fast=args.fast,
            recorder=recorder,
        )
        if summary:
            all_summaries.append(summary)

    # ── M7: Provenance Export ───────────────────────────────────────────
    recorder.save(prov_path)
    print(f"\n[M7] Provenance trace → {prov_path}")

    with open(os.path.join(args.results_dir, "run_summaries.json"), "w") as fh:
        json.dump(all_summaries, fh, indent=2)

    print(f"\nDone. Results in: {args.results_dir}")
    print(f"  Records processed: {len(all_summaries)}/{len(args.records)}")


if __name__ == "__main__":
    main()
