"""
use_case.py — SensorWF Core Framework Runner
=============================================
Dispatches to the satellite, ECG, or climate core pipeline (M1-M3, M6, M7).

The core framework provides reusable modules for data ingestion, quality
assessment, feature engineering, semantic annotation, and provenance export.
The anomaly detection use case (M4 fault injection + M5 ML evaluation) is
implemented as a separate extension -- see use_cases/ directory.

Usage
─────
  python use_case.py                          # run all three domain cores
  python use_case.py --domain satellite       # satellite core only
  python use_case.py --domain ecg             # ECG core only
  python use_case.py --domain climate         # climate core only

  # Pass domain-specific args after --domain <name>:
  python use_case.py --domain satellite --families AccelerometerTest
  python use_case.py --domain ecg --records 100 106
  python use_case.py --domain climate --years 2011

Anomaly detection use case (M4 + M5)
──────────────────────────────────────
  Run after the core framework to add fault injection and ML evaluation:

    python use_cases/sat_anomaly.py [--families ...]
    python use_cases/ecg_anomaly.py [--records ...]
    python use_cases/climate_anomaly.py [--years ...]

Core pipeline outputs
──────────────────────
  results/satellite/<Family>/
    cdh_clean.csv  adcs_clean.csv  quality_report.json  *.png
    provenance.ttl  workflow_spec.ttl  semantic/

  results/ECG/mitdb/record_<id>/
    signal_clean.csv  quality_report.json  semantic/

  results/Climate/jena/<period>/
    signal_clean.csv  quality_report.json  semantic/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import os


# ── CLI ─────────────────────────────────────────────────────────────────────────

_ap = argparse.ArgumentParser(
    description="SensorWF core framework runner (satellite | ecg | climate | all)",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
    add_help=True,
)
_ap.add_argument(
    "--domain",
    choices=["satellite", "ecg", "climate", "all"],
    default="all",
    help="Which domain core to run (default: all)",
)
_known, _extra = _ap.parse_known_args()

DOMAIN      = _known.domain
EXTRA_ARGS  = _extra


# ── Runner dispatch ──────────────────────────────────────────────────────────────

RUNNERS = {
    "satellite": "run_sat.py",
    "ecg":       "run_ecg.py",
    "climate":   "run_climate.py",
}

DOMAIN_LABELS = {
    "satellite": "CubeSat Telemetry (core: M1-M3, M6, M7)",
    "ecg":       "Biomedical ECG (core: M1-M3, M6, M7)",
    "climate":   "Atmospheric Climate (core: M1-M3, M6, M7)",
}


def run_domain(domain: str, extra: list[str]) -> int:
    """Run one domain's core pipeline as a subprocess. Returns the exit code."""
    script = RUNNERS[domain]
    label  = DOMAIN_LABELS[domain]
    cmd    = [sys.executable, script] + extra
    print(f"\n{'=' * 70}")
    print(f"SensorWF — {label}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70)
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    if DOMAIN == "all":
        domains_to_run = ["satellite", "ecg", "climate"]
    else:
        domains_to_run = [DOMAIN]

    exit_codes: dict[str, int] = {}
    for domain in domains_to_run:
        rc = run_domain(domain, EXTRA_ARGS)
        exit_codes[domain] = rc
        if rc != 0:
            print(f"\nWARNING: {domain} runner exited with code {rc}")

    print("\n" + "=" * 70)
    print("SensorWF — Core framework complete")
    for domain, rc in exit_codes.items():
        status = "OK" if rc == 0 else f"FAILED (exit {rc})"
        label  = DOMAIN_LABELS[domain].split("(")[0].strip()
        print(f"  {label:<28} : {status}")
    print("=" * 70)
    print("\nTo run anomaly detection use case on top of core outputs:")
    print("  python use_cases/sat_anomaly.py")
    print("  python use_cases/ecg_anomaly.py")
    print("  python use_cases/climate_anomaly.py")

    if any(rc != 0 for rc in exit_codes.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
