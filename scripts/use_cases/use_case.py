"""
use_case.py — TelWF Anomaly Detection Use Case
===============================================
Demonstrates the anomaly detection use case on top of the core TelWF
pipeline outputs.  Requires main.py to have been run first so that
clean CDH/ADCS CSVs exist under results/<Family>/.

Use-case stages
───────────────
  [P4] FaultInjection      — generate labelled synthetic fault variants
  [P5] AnomalyDetection    — train and evaluate four ML detectors
  [P6] SemanticAnnotation  — ontology-linked knowledge graph
  [P7] ProvenanceExport    — emit PROV-O / ProvONE provenance trace

Usage
─────
  # Run core pipeline first (if not already done)
  python main.py

  # Then run the use-case extension
  python use_case.py [--results-dir results] [--families AccelerometerTest ...]
                     [--seed 42] [--n-variants 2] [--fast]
                     [--ontology satellitesystem.owl]
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import pandas as pd

from scripts.injector        import run_injections
from scripts.semantic_if_kg  import build_semantic_graph
from scripts.workflow        import (
    save_workflow_spec,
    M4_FAULT_INJECTION, M5_DETECTION, M6_SEMANTIC,
)
from scripts.provenance_recorder import ProvenanceRecorder

# ── CLI arguments ──────────────────────────────────────────────────────────────
_ap = argparse.ArgumentParser(
    description="TelWF use case: fault injection → anomaly detection → semantic KG → provenance"
)
_ap.add_argument("--results-dir", default="results",
                 help="Directory written by main.py (default: results)")
_ap.add_argument("--families", nargs="*", metavar="NAME",
                 help="Restrict to named experiment families")
_ap.add_argument("--seed", type=int, default=42,
                 help="Random seed for fault injection (default: 42)")
_ap.add_argument("--n-variants", type=int, default=2,
                 help="Injected variants per fault × tier (default: 2)")
_ap.add_argument("--fast", action="store_true",
                 help="Quick mode: 1 variant, easy tier only")
_ap.add_argument("--ontology", default="satellitesystem.owl",
                 help="Path to OWL ontology file (default: satellitesystem.owl)")
_ARGS = _ap.parse_args()

RESULTS_DIR   = _ARGS.results_dir
ONTOLOGY_PATH = _ARGS.ontology
N_VARIANTS    = 1 if _ARGS.fast else _ARGS.n_variants
TIERS         = ["easy"] if _ARGS.fast else None  # None → all three tiers

# ── Discover experiment families from core pipeline output ─────────────────────

def _discover_families(results_dir: str) -> list[tuple[str, str, str]]:
    """Return [(family_name, cdh_csv_path, adcs_csv_path), ...] for families
    that have both cdh_clean.csv and adcs_clean.csv written by main.py."""
    found = []
    for entry in sorted(os.scandir(results_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        cdh_path  = os.path.join(entry.path, "cdh_clean.csv")
        adcs_path = os.path.join(entry.path, "adcs_clean.csv")
        if os.path.isfile(cdh_path) and os.path.isfile(adcs_path):
            found.append((entry.name, cdh_path, adcs_path))
    return found


families = _discover_families(RESULTS_DIR)
if _ARGS.families:
    allowed = {f.lower() for f in _ARGS.families}
    families = [f for f in families if f[0].lower() in allowed]

if not families:
    print(f"No core-pipeline outputs found in {RESULTS_DIR}/")
    print("Run  python main.py  first to generate cdh_clean.csv / adcs_clean.csv.")
    raise SystemExit(1)

# ── Initialise provenance recorder (covers P4–P7) ─────────────────────────────
run_ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
recorder = ProvenanceRecorder(workflow_run_id=f"telwf_usecase_{run_ts}")

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("TelWF — Anomaly Detection Use Case")
print(f"Found {len(families)} family/families   |   Run: {run_ts}")
print("=" * 70)
for name, cdh_p, adcs_p in families:
    print(f"  {name:<30} ← {os.path.relpath(cdh_p)}")

for exp_name, cdh_path, adcs_path in families:
    output_dir   = os.path.join(RESULTS_DIR, exp_name)
    injected_dir = os.path.join(output_dir, "injected")

    print(f"\n{'─' * 70}")
    print(f"Experiment : {exp_name}")
    print(f"{'─' * 70}")

    # Load clean data written by the core pipeline
    cdh  = pd.read_csv(cdh_path)
    adcs = pd.read_csv(adcs_path)
    print(f"  Loaded CDH  : {len(cdh)} rows   ({os.path.relpath(cdh_path)})")
    print(f"  Loaded ADCS : {len(adcs)} rows   ({os.path.relpath(adcs_path)})")

    # ── P4: FaultInjection + P5: AnomalyDetection ─────────────────────────────
    print(f"\n  [P4] FaultInjection — generating synthetic fault variants …")
    print(f"  [P5] AnomalyDetection — training and evaluating detectors …")
    m4_start = datetime.now(timezone.utc)
    try:
        run_injections(cdh, adcs, output_dir,
                       seed=_ARGS.seed, n_variants=N_VARIANTS, tiers=TIERS)
    except Exception as exc:
        print(f"  ERROR in P4/P5 for {exp_name}: {exc}")
    m4_end = datetime.now(timezone.utc)

    ml_json_path = os.path.join(injected_dir, "ml_evaluation.json")

    recorder.record(M4_FAULT_INJECTION,
        inputs  = {
            "cdh_clean":  {"path": cdh_path,  "rows": len(cdh)},
            "adcs_clean": {"path": adcs_path, "rows": len(adcs)},
            "seed":       _ARGS.seed,
            "n_variants": N_VARIANTS,
        },
        outputs = {
            "injected_variants": {"path": injected_dir},
            "injection_summary": {"path": os.path.join(injected_dir, "injection_summary.csv")},
        },
        start_time = m4_start,
        end_time   = m4_end,
        experiment = exp_name,
    )

    recorder.record(M5_DETECTION,
        inputs  = {
            "injected_variants": {"path": injected_dir},
            "cdh_clean":         {"rows": len(cdh)},
            "adcs_clean":        {"rows": len(adcs)},
        },
        outputs = {
            "ml_evaluation_json": {"path": ml_json_path},
            "ml_metrics_csv":     {"path": os.path.join(injected_dir, "ml_metrics_by_tier.csv")},
        },
        start_time = m4_start,
        end_time   = m4_end,
        experiment = exp_name,
    )

# ── P6: SemanticAnnotation (runs across all families at once) ──────────────────
print(f"\n  [P6] SemanticAnnotation — building ontology-linked knowledge graph …")
semantic_out_dir = os.path.join(RESULTS_DIR, "semantic")
m6_start = datetime.now(timezone.utc)
m6_artifacts: dict = {}
try:
    m6_artifacts = build_semantic_graph(
        results_dir  = RESULTS_DIR,
        ontology_path= ONTOLOGY_PATH,
        output_dir   = semantic_out_dir,
    )
    print(f"        KG artifacts → {semantic_out_dir}/")
    for k, v in m6_artifacts.items():
        print(f"          {k}: {v}")
except Exception as exc:
    print(f"  ERROR in P6: {exc}")
m6_end = datetime.now(timezone.utc)

recorder.record(M6_SEMANTIC,
    inputs  = {
        "ml_evaluation_json": {"path": os.path.join(RESULTS_DIR, "**", "injected", "ml_evaluation.json")},
        "ontology_owl":       {"path": ONTOLOGY_PATH},
    },
    outputs = {k: {"path": str(v)} for k, v in m6_artifacts.items()},
    start_time = m6_start,
    end_time   = m6_end,
)

# ── P7: ProvenanceExport ───────────────────────────────────────────────────────
print(f"\n  [P7] ProvenanceExport — writing PROV-O / ProvONE trace …")
prov_path = os.path.join(RESULTS_DIR, "provenance.ttl")
recorder.save(prov_path)

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"TelWF use case complete.  Results → {RESULTS_DIR}/")
print(f"  Provenance     : {prov_path}")
print(f"  Semantic KG    : {semantic_out_dir}/")
print(f"  Families       : {len(families)}")
print("=" * 70)
