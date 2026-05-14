"""
run_climate.py -- SensorWF core pipeline for the Jena Climate Dataset.

Core modules executed
---------------------
M1  DataIngestion      : ClimateAdapter -- load year slice, rename columns, fill gaps
M2  QualityAssessment  : generic quality report (NaN, stuck, timing, trend detection)
M3  FeatureEngineering : raw + diff + rolling stats (window=24, 4 hours @ 10-min)
M4  SemanticAnnotation : map feature names → climate.owl ontology classes
M5  ProvenanceExport   : PROV-O/ProvONE trace to results/Climate/provenance.ttl

Periods processed (7 years: Jan-Jun slices from 2009-2015)
------------------------------------------------------------
  2009H1 - 2015H1: ~26,280 rows each (10-min resolution, 14 channels)
  Covers cold/warm seasons, inter-annual variability, instrument drift epochs.

Anomaly detection use case (E1 + E2)
--------------------------------------
  Run after the core pipeline to add fault injection and ML evaluation:
    python use_cases/climate_anomaly.py [--years 2009 2010 ...]

Usage
-----
  python run_climate.py [--years 2011 2012] [--months 1 6]
                        [--data-path data/Climate/jena_climate_2009_2016.csv]
                        [--results-dir results/Climate/jena] [--seed 42]
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from scripts.adapters.climate import ClimateAdapter
from scripts.pipeline_core import (
    run_quality_assessment,
    save_quality_report,
    build_semantic_kg,
    plot_channels,
)
from scripts.evaluator import build_generic_feature_matrix
from scripts.provenance_recorder import ProvenanceRecorder

_DEFAULT_YEARS    = [2009, 2010, 2011, 2012, 2013, 2014, 2015]
_DEFAULT_MONTHS   = (1, 6)
_DATA_PATH        = os.path.join("data", "Climate", "jena_climate_2009_2016.csv")
_RESULTS_DIR      = os.path.join("results", "Climate", "jena")
_SEED             = 42


# ===========================================================================
# Per-period core pipeline
# ===========================================================================

def run_climate_period(
    year: int,
    months: tuple[int, int],
    data_path: str,
    results_dir: str,
    recorder: ProvenanceRecorder | None,
    rerun: bool = False,
) -> dict:
    """Run M1-M4 on one climate period. Returns summary dict."""
    period_label = f"{year}_M{months[0]:02d}_M{months[1]:02d}"
    period_dir   = os.path.join(results_dir, period_label)
    sem_dir      = os.path.join(period_dir, "semantic")

    summary_path = os.path.join(period_dir, "run_summary.json")
    if not rerun and os.path.isfile(summary_path):
        with open(summary_path) as fh:
            cached = json.load(fh)
        print(f"  [CACHE] {period_label} — skipping (pass --rerun to force)")
        return cached

    os.makedirs(period_dir, exist_ok=True)
    os.makedirs(sem_dir, exist_ok=True)
    os.makedirs(os.path.join(period_dir, "plots"), exist_ok=True)

    # ── M1: Data Ingestion ──────────────────────────────────────────────
    t0      = time.time()
    adapter  = ClimateAdapter(year=year, months=months)
    df       = adapter.load(data_path)
    channels = adapter.channels
    feat_cfg = adapter.get_feature_config()
    print(f"  [M1] {period_label}: {len(df):,} rows × {len(channels)} channels "
          f"(10-min resolution, {df['elapsed_s'].max()/3600:.0f} hours)")
    t_m1 = time.time() - t0

    if len(df) < 100:
        print(f"  [M1] SKIP {period_label}: too few rows ({len(df)})")
        return {}

    if recorder:
        recorder.record_activity(
            f"M1_Climate_DataIngestion_{period_label}",
            inputs={"data_path": data_path},
            outputs={"n_rows": len(df), "channels": channels},
            params={"year": year, "months": list(months)},
        )

    # ── M2: Quality Assessment ──────────────────────────────────────────
    t0 = time.time()
    quality_cfg    = adapter.get_quality_config()
    quality_report = run_quality_assessment(df, channels, quality_cfg)
    save_quality_report(quality_report, os.path.join(period_dir, "quality_report.json"))
    t_m2 = time.time() - t0

    clean_csv = os.path.join(period_dir, "signal_clean.csv")
    df.to_csv(clean_csv, index=False)

    # ── Overview plots ────────────────────────────────────────────────────
    plot_channels(df, adapter.get_plot_groups(),
                  os.path.join(period_dir, "plots"),
                  title_prefix=f"Jena Climate {period_label}")

    # ── M3: Feature Engineering ─────────────────────────────────────────
    t0 = time.time()
    X_clean, feat_names = build_generic_feature_matrix(df, channels, window=feat_cfg["window"])
    t_m3 = time.time() - t0
    if X_clean is None:
        print(f"  [M3] SKIP {period_label}: no usable features")
        return {}
    print(f"  [M3] Feature matrix: {X_clean.shape}")

    if recorder:
        recorder.record_activity(
            f"M3_Climate_FeatureEngineering_{period_label}",
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
        file_prefix   = "climate_",
        domain_uri    = "https://sensorwf.org/ontologies/climate",
    )
    t_m4 = time.time() - t_m4_start
    print(f"  [M4] Climate KG: {len(m4_arts)} artefacts → {sem_dir}")

    if recorder:
        recorder.record_activity(
            f"M4_Climate_SemanticAnnotation_{period_label}",
            inputs={"ontology": adapter.ensure_ontology()},
            outputs={k: v for k, v in m4_arts.items()},
            params={},
        )

    summary = {
        "period":     period_label,
        "year":       year,
        "n_rows":     len(df),
        "n_features": len(feat_names),
        "timing_s":   {"m1": t_m1, "m2": t_m2, "m3": t_m3, "m4": t_m4},
    }
    with open(os.path.join(period_dir, "run_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SensorWF Climate core pipeline (M1-M5)")
    parser.add_argument("--years",       nargs="+", type=int, default=_DEFAULT_YEARS)
    parser.add_argument("--months",      nargs=2,   type=int, default=list(_DEFAULT_MONTHS))
    parser.add_argument("--data-path",   default=_DATA_PATH)
    parser.add_argument("--results-dir", default=_RESULTS_DIR)
    parser.add_argument("--seed",        type=int, default=_SEED,
                        help="Reserved for reproducibility (not used by core modules)")
    parser.add_argument("--rerun",       action="store_true",
                        help="Ignore cached run_summary.json and reprocess every period")
    args = parser.parse_args()

    months = tuple(args.months)
    os.makedirs(args.results_dir, exist_ok=True)
    prov_path = os.path.join(args.results_dir, "provenance.ttl")
    recorder  = ProvenanceRecorder(workflow_uri="sensorwf:ClimateWorkflow",
                                   domain="atmospheric_climate")

    print("=" * 64)
    print("SensorWF -- Climate Core Pipeline (M1-M5)")
    print(f"Years: {args.years}  Months: {months[0]}-{months[1]}")
    print("=" * 64)

    all_summaries = []
    for year in args.years:
        print(f"\n── Year {year} {'─' * 45}")
        summary = run_climate_period(
            year=year, months=months,
            data_path=args.data_path,
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
    print(f"\nTo run all phases (core + CWL + report + anomaly):")
    print(f"  python use_case.py --domain climate")
    print(f"To run anomaly detection only:")
    print(f"  python use_cases/climate_anomaly.py [--years {' '.join(str(y) for y in args.years[:3])} ...]")


if __name__ == "__main__":
    main()
