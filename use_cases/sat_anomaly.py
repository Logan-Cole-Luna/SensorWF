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
  Run the core pipeline first:
    python run_sat.py [--families AccelerometerTest ...]

  Core outputs consumed:
    results/satellite/<Family>/cdh_clean.csv
    results/satellite/<Family>/adcs_clean.csv

Outputs
───────
  results/satellite/<Family>/injected/
    injection_summary.csv    -- M4 fault manifest
    signal_injected_*.csv    -- injected signal variants
    labels_*.csv             -- ground-truth anomaly labels
    ml_evaluation.json       -- per-variant per-detector metrics (M5)
    ml_metrics_by_tier.csv   -- tier-aggregated metrics (M5)

  results/satellite/semantic/ -- enhanced M6 KG with ML importances
  results/satellite/provenance.ttl -- M7 provenance (appended)

Usage
─────
  python use_cases/sat_anomaly.py [--families AccelerometerTest GyroTest]
                                   [--seed 42] [--n-variants 2] [--fast]
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import pandas as pd

from scripts.utils.injector import run_injections
from scripts.satellite.semantic_kg import build_semantic_graph
from scripts.satellite.ontology import generate_satellite_ontology
from scripts.provenance_recorder import ProvenanceRecorder
from scripts.workflow import E1_FAULT_INJECTION, E2_DETECTION, M4_SEMANTIC

_RESULTS_DIR  = os.path.join("results", "satellite")
_ONTOLOGY_PATH = os.path.join("results", "ontologies", "satellitesystem.owl")
_ALL_FAMILIES = [
    "AccelerometerTest", "GyroTest", "ReactionWheelTest", "ThermalTest",
]


def run_sat_anomaly(
    families: list[str],
    results_dir: str,
    ontology_path: str,
    seed: int,
    n_variants: int,
    tiers: list[str] | None,
    recorder: ProvenanceRecorder,
) -> None:
    """Run M4 + M5 for each satellite family, consuming core pipeline outputs."""
    for family in families:
        output_dir   = os.path.join(results_dir, family)
        injected_dir = os.path.join(output_dir, "injected")
        cdh_path     = os.path.join(output_dir, "cdh_clean.csv")
        adcs_path    = os.path.join(output_dir, "adcs_clean.csv")

        if not os.path.isfile(cdh_path):
            print(f"  SKIP {family}: {cdh_path} not found — run run_sat.py first")
            continue

        print(f"\n── {family} {'─' * (60 - len(family))}")
        cdh  = pd.read_csv(cdh_path,  low_memory=False)
        adcs = pd.read_csv(adcs_path, low_memory=False) if os.path.isfile(adcs_path) else pd.DataFrame()

        cdh_cols  = [c for c in cdh.columns  if c not in {"timestamp", "elapsed_s"}]
        adcs_cols = [c for c in adcs.columns if c not in {"timestamp", "elapsed_s"}]

        generate_satellite_ontology(ontology_path, cdh_cols, adcs_cols)

        m4_start = datetime.now(timezone.utc)
        try:
            run_injections(cdh, adcs, output_dir,
                           seed=seed, n_variants=n_variants, tiers=tiers)
        except Exception as exc:
            print(f"  ERROR in E1/E2 for {family}: {exc}")
        m4_end = datetime.now(timezone.utc)

        ml_json_path = os.path.join(injected_dir, "ml_evaluation.json")

        recorder.record(E1_FAULT_INJECTION,
            inputs  = {"cdh_clean": {"path": cdh_path,  "rows": len(cdh)},
                       "adcs_clean": {"path": adcs_path, "rows": len(adcs)},
                       "seed": seed, "n_variants": n_variants},
            outputs = {"injected_variants": {"path": injected_dir},
                       "injection_summary": {"path": os.path.join(injected_dir, "injection_summary.csv")}},
            start_time=m4_start, end_time=m4_end, experiment=family,
        )
        recorder.record(E2_DETECTION,
            inputs  = {"injected_variants": {"path": injected_dir},
                       "cdh_clean": {"rows": len(cdh)},
                       "adcs_clean": {"rows": len(adcs)}},
            outputs = {"ml_evaluation_json": {"path": ml_json_path},
                       "ml_metrics_csv": {"path": os.path.join(injected_dir, "ml_metrics_by_tier.csv")}},
            start_time=m4_start, end_time=m4_end, experiment=family,
        )

    # ── Enhanced M6: rebuild KG with ML importances ─────────────────────────
    print(f"\n  [M4] Building enhanced semantic KG with ML importances ...")
    semantic_out_dir = os.path.join(results_dir, "semantic")
    m6_start = datetime.now(timezone.utc)
    m6_artifacts: dict = {}
    try:
        m6_artifacts = build_semantic_graph(
            results_dir=results_dir,
            ontology_path=ontology_path,
            output_dir=semantic_out_dir,
        )
        for k, v in m6_artifacts.items():
            print(f"    {k}: {v}")
    except Exception as exc:
        print(f"  WARNING: M6 enhanced KG failed: {exc}")
    m6_end = datetime.now(timezone.utc)

    if m6_artifacts:
        recorder.record(M4_SEMANTIC,
            inputs  = {"ml_evaluation_json": {"path": os.path.join(results_dir, "**", "injected", "ml_evaluation.json")},
                       "ontology_owl": {"path": ontology_path}},
            outputs = {k: {"path": str(v)} for k, v in m6_artifacts.items()},
            start_time=m6_start, end_time=m6_end,
        )


def main():
    parser = argparse.ArgumentParser(
        description="SensorWF satellite anomaly detection use case (E1 + E2)")
    parser.add_argument("--families",    nargs="*", metavar="NAME",
                        help="Experiment families (default: all 4)")
    parser.add_argument("--results-dir", default=_RESULTS_DIR,
                        help="Core pipeline results dir (default: results/satellite)")
    parser.add_argument("--ontology",    default=_ONTOLOGY_PATH)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--n-variants",  type=int, default=2)
    parser.add_argument("--fast",        action="store_true",
                        help="Easy tier only")
    args = parser.parse_args()

    families = args.families if args.families else _ALL_FAMILIES
    tiers    = ["easy"] if args.fast else None

    if not os.path.isfile(args.ontology):
        generate_satellite_ontology(args.ontology)

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
        ontology_path=args.ontology,
        seed=args.seed,
        n_variants=args.n_variants,
        tiers=tiers,
        recorder=recorder,
    )

    prov_path = os.path.join(args.results_dir, "anomaly_provenance.ttl")
    recorder.save(prov_path)
    print(f"\n[M5] Anomaly use-case provenance → {prov_path}")
    print(f"\nDone. Results in: {args.results_dir}/*/injected/")


if __name__ == "__main__":
    main()
