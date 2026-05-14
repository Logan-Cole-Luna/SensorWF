"""
sat_anomaly.py — Anomaly Detection Use Case for SensorWF Satellite Domain
==========================================================================
Extends the SensorWF core pipeline (M1-M5) with synthetic fault
injection (E1) and anomaly detection evaluation (E2) for KISPE SATLL telemetry.

This script is a use-case extension, not part of the reusable core framework.
It demonstrates that the core pipeline's clean CDH/ADCS CSVs are directly
analysis-ready for downstream anomaly detection tasks.

Prerequisites
─────────────
  Run the core pipeline first (or let use_case.py handle it):
    python use_case.py --domain satellite --skip-anomaly
    python use_cases/run_sat.py [--families AccelerometerTest ...]

  Core outputs consumed:
    results/satellite/<Family>/cdh_clean.csv
    results/satellite/<Family>/adcs_clean.csv

Outputs
───────
  results/satellite/<Family>/injected/
    injection_summary.csv    -- E1 fault manifest
    signal_injected_*.csv    -- injected signal variants
    labels_*.csv             -- ground-truth anomaly labels
    ml_evaluation.json       -- per-variant per-detector metrics (E2)
    ml_metrics_by_tier.csv   -- tier-aggregated metrics (E2)

  results/satellite/semantic/ -- enhanced M4 KG with ML importances
  results/satellite/anomaly_provenance.ttl -- M5 provenance trace

Usage
─────
  python use_cases/sat_anomaly.py [--families AccelerometerTest GyroTest]
                                   [--seed 42] [--n-variants 2] [--fast]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is in Python path for relative imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from scripts.utils.injector import run_injections
from scripts.adapters.satellite import SatelliteAdapter
from scripts.pipeline_core import build_semantic_kg
from scripts.provenance_recorder import ProvenanceRecorder
from scripts.workflow import E1_FAULT_INJECTION, E2_DETECTION, M4_SEMANTIC

_RESULTS_DIR   = os.path.join("results", "satellite")
_ONTOLOGY_PATH = os.path.join("results", "ontologies", "satellitesystem.owl")
_ALL_FAMILIES  = [
    "AccelerometerTest", "GyroTest", "ReactionWheelTest", "ThermalTest",
]


def _load_ml_importances(results_dir: str) -> dict[str, float]:
    """Aggregate IsolationForest feature importances across all family ml_evaluation.json files."""
    acc: dict[str, list[float]] = {}
    for family in sorted(os.listdir(results_dir)):
        ml_json = os.path.join(results_dir, family, "injected", "ml_evaluation.json")
        if not os.path.isfile(ml_json):
            continue
        try:
            data = json.loads(open(ml_json, encoding="utf-8").read())
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for r in data:
            if r.get("detector") != "IsolationForest":
                continue
            imps  = r.get("feature_importances") or []
            names = r.get("feature_names") or []
            for feat, imp in zip(names, imps):
                acc.setdefault(str(feat), []).append(float(imp))
    return {k: float(sum(v) / len(v)) for k, v in acc.items() if v}


def run_sat_anomaly(
    families: list[str],
    results_dir: str,
    seed: int,
    n_variants: int,
    tiers: list[str] | None,
    recorder: ProvenanceRecorder,
    rerun: bool = False,
) -> None:
    """Run E1 + E2 for each satellite family, then rebuild M4 KG with ML importances."""
    adapter = SatelliteAdapter()

    for family in families:
        output_dir   = os.path.join(results_dir, family)
        injected_dir = os.path.join(output_dir, "injected")
        cdh_path     = os.path.join(output_dir, "cdh_clean.csv")
        adcs_path    = os.path.join(output_dir, "adcs_clean.csv")

        if not os.path.isfile(cdh_path):
            print(f"  SKIP {family}: {cdh_path} not found — run run_sat.py first")
            continue

        ml_cache = os.path.join(injected_dir, "ml_evaluation.json")
        if not rerun and os.path.isfile(ml_cache):
            print(f"  [CACHE] {family} — skipping E1+E2 (pass --rerun to force)")
            continue

        print(f"\n── {family} {'─' * (60 - len(family))}")
        cdh  = pd.read_csv(cdh_path,  low_memory=False)
        adcs = pd.read_csv(adcs_path, low_memory=False) if os.path.isfile(adcs_path) else pd.DataFrame()

        adapter.cdh_channels  = [c for c in cdh.columns  if c not in {"timestamp", "elapsed_s"}]
        adapter.adcs_channels = [c for c in adcs.columns if c not in {"timestamp", "elapsed_s"}]
        ontology_path = adapter.get_ontology_path()

        e1_start = datetime.now(timezone.utc)
        try:
            run_injections(cdh, adcs, output_dir,
                           seed=seed, n_variants=n_variants, tiers=tiers)
        except Exception as exc:
            print(f"  ERROR in E1/E2 for {family}: {exc}")
        e1_end = datetime.now(timezone.utc)

        ml_json_path = os.path.join(injected_dir, "ml_evaluation.json")

        recorder.record(E1_FAULT_INJECTION,
            inputs  = {"cdh_clean":  {"path": cdh_path,  "rows": len(cdh)},
                       "adcs_clean": {"path": adcs_path, "rows": len(adcs)},
                       "seed": seed, "n_variants": n_variants},
            outputs = {"injected_variants": {"path": injected_dir},
                       "injection_summary": {"path": os.path.join(injected_dir, "injection_summary.csv")}},
            start_time=e1_start, end_time=e1_end, experiment=family,
        )
        recorder.record(E2_DETECTION,
            inputs  = {"injected_variants": {"path": injected_dir},
                       "cdh_clean":  {"rows": len(cdh)},
                       "adcs_clean": {"rows": len(adcs)}},
            outputs = {"ml_evaluation_json": {"path": ml_json_path},
                       "ml_metrics_csv": {"path": os.path.join(injected_dir, "ml_metrics_by_tier.csv")}},
            start_time=e1_start, end_time=e1_end, experiment=family,
        )

    # ── Enhanced M4: rebuild KG with aggregated ML importances ─────────────
    print(f"\n  [M4] Building enhanced semantic KG with ML importances ...")
    semantic_out_dir = os.path.join(results_dir, "semantic")
    m4_start = datetime.now(timezone.utc)
    m4_artifacts: dict = {}
    try:
        ml_importances = _load_ml_importances(results_dir)
        feat_names = list(ml_importances.keys())
        if feat_names:
            m4_artifacts = build_semantic_kg(
                feature_names  = feat_names,
                class_map      = adapter.get_class_map(),
                output_dir     = semantic_out_dir,
                ontology_path  = ontology_path,
                ml_importances = ml_importances,
                file_prefix    = "sat_enhanced_",
                domain_uri     = "https://sensorwf.org/ontologies/satellite",
            )
            for k, v in m4_artifacts.items():
                print(f"    {k}: {v}")
        else:
            print("    No ML importances found — skipping enhanced KG")
    except Exception as exc:
        print(f"  WARNING: M4 enhanced KG failed: {exc}")
    m4_end = datetime.now(timezone.utc)

    if m4_artifacts:
        recorder.record(M4_SEMANTIC,
            inputs  = {"ontology_owl": {"path": ontology_path}},
            outputs = {k: {"path": str(v)} for k, v in m4_artifacts.items()},
            start_time=m4_start, end_time=m4_end,
        )


def main():
    parser = argparse.ArgumentParser(
        description="SensorWF satellite anomaly detection use case (E1 + E2)")
    parser.add_argument("--families",    nargs="*", metavar="NAME",
                        help="Experiment families (default: all 4)")
    parser.add_argument("--results-dir", default=_RESULTS_DIR)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--n-variants",  type=int, default=2)
    parser.add_argument("--fast",        action="store_true",
                        help="Easy tier only")
    parser.add_argument("--rerun",       action="store_true",
                        help="Ignore cached ml_evaluation.json and rerun E1+E2")
    args = parser.parse_args()

    families = args.families if args.families else _ALL_FAMILIES
    tiers    = ["easy"] if args.fast else None

    run_ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    recorder = ProvenanceRecorder(workflow_run_id=f"sensorwf_sat_anomaly_{run_ts}")

    print("=" * 64)
    print("SensorWF -- Satellite Anomaly Detection Use Case (E1 + E2)")
    print(f"Families : {families}")
    print(f"Results  : {args.results_dir}")
    print("=" * 64)

    run_sat_anomaly(
        families=families,
        results_dir=args.results_dir,
        seed=args.seed,
        n_variants=args.n_variants,
        tiers=tiers,
        recorder=recorder,
        rerun=args.rerun,
    )

    prov_path = os.path.join(args.results_dir, "anomaly_provenance.ttl")
    recorder.save(prov_path)
    print(f"\n[M5] Anomaly use-case provenance → {prov_path}")
    print(f"\nDone. Results in: {args.results_dir}/*/injected/")


if __name__ == "__main__":
    main()
