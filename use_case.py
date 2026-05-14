"""
use_case.py — SensorWF Full Orchestrator
=========================================
Single entry point that runs all workflow stages in sequence:

  Phase 1 — Core pipelines      M1-M5 per domain (data ingestion → provenance)
  Phase 2 — CWL export          Sync components.json → cwl/ tools
  Phase 3 — HTML run reports    One self-contained HTML report per domain
  Phase 4 — Anomaly detection   E1 fault injection + E2 ML evaluation (run last)

Each phase is independent.  Individual scripts can still be called directly:

  Core runners  : use_cases/run_{sat,ecg,climate}.py
  Anomaly       : use_cases/{sat,ecg,climate}_anomaly.py
  CWL export    : python -m scripts.utils.cwl_export
  HTML report   : python -m scripts.utils.html_report --results-dir results/<domain>

Usage
─────
  python use_case.py                         # all domains, all phases
  python use_case.py --domain satellite      # satellite only, all phases
  python use_case.py --domain ecg            # ECG only, all phases
  python use_case.py --domain climate        # climate only, all phases

  python use_case.py --skip-anomaly          # skip Phase 4 (much faster)
  python use_case.py --skip-report           # skip HTML generation
  python use_case.py --skip-cwl              # skip CWL regeneration

  # Domain-specific args are forwarded to the relevant core runner and anomaly script:
  python use_case.py --domain satellite --families AccelerometerTest
  python use_case.py --domain ecg --records 100 106 --rerun
  python use_case.py --domain climate --years 2011 2012
  python use_case.py --seed 42 --n-variants 2 --fast
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Script paths ──────────────────────────────────────────────────────────────

CORE_RUNNERS = {
    "satellite": os.path.join("use_cases", "run_sat.py"),
    "ecg":       os.path.join("use_cases", "run_ecg.py"),
    "climate":   os.path.join("use_cases", "run_climate.py"),
}

ANOMALY_RUNNERS = {
    "satellite": os.path.join("use_cases", "sat_anomaly.py"),
    "ecg":       os.path.join("use_cases", "ecg_anomaly.py"),
    "climate":   os.path.join("use_cases", "climate_anomaly.py"),
}

REPORT_ARGS: dict[str, list[str]] = {
    "satellite": ["--results-dir", os.path.join("results", "satellite"), "--domain", "satellite"],
    "ecg":       ["--results-dir", os.path.join("results", "ECG"),       "--domain", "ecg"],
    "climate":   ["--results-dir", os.path.join("results", "Climate"),   "--domain", "climate"],
}

DOMAIN_LABELS = {
    "satellite": "CubeSat Telemetry",
    "ecg":       "Biomedical ECG",
    "climate":   "Atmospheric Climate",
}

ALL_DOMAINS = ["satellite", "ecg", "climate"]


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    ap = argparse.ArgumentParser(
        description="SensorWF full orchestrator — runs all workflow stages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--domain",
        choices=["satellite", "ecg", "climate", "all"],
        default="all",
        help="Which domain(s) to process (default: all)",
    )
    ap.add_argument("--skip-cwl",     action="store_true",
                    help="Skip Phase 2: CWL regeneration")
    ap.add_argument("--skip-report",  action="store_true",
                    help="Skip Phase 3: HTML report generation")
    ap.add_argument("--skip-anomaly", action="store_true",
                    help="Skip Phase 4: anomaly detection (E1+E2)")
    return ap.parse_known_args()


# ── Subprocess helper ─────────────────────────────────────────────────────────

def _run(cmd: list[str], label: str) -> int:
    print(f"\n  {label}")
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=_ROOT)
    if result.returncode != 0:
        print(f"  WARNING: '{label}' exited with code {result.returncode}")
    return result.returncode


# ── Phase 1: Core pipelines ───────────────────────────────────────────────────

def phase_core(domains: list[str], extra: list[str]) -> dict[str, int]:
    print(f"\n{'=' * 70}")
    print("  PHASE 1 — Core Pipelines (M1-M5)")
    print(f"{'=' * 70}")
    codes: dict[str, int] = {}
    for domain in domains:
        codes[domain] = _run(
            [sys.executable, CORE_RUNNERS[domain]] + extra,
            f"{DOMAIN_LABELS[domain]} core (M1-M5)",
        )
    return codes


# ── Phase 2: CWL export ───────────────────────────────────────────────────────

def phase_cwl() -> int:
    print(f"\n{'=' * 70}")
    print("  PHASE 2 — CWL Export")
    print(f"{'=' * 70}")
    return _run(
        [sys.executable, "-m", "scripts.utils.cwl_export",
         "--components", "components.json", "--output", "cwl"],
        "Regenerate CWL tools from components.json",
    )


# ── Phase 3: HTML reports ─────────────────────────────────────────────────────

def phase_reports(domains: list[str]) -> dict[str, int]:
    print(f"\n{'=' * 70}")
    print("  PHASE 3 — HTML Run Reports")
    print(f"{'=' * 70}")
    codes: dict[str, int] = {}
    for domain in domains:
        codes[domain] = _run(
            [sys.executable, "-m", "scripts.utils.html_report", "--no-browser"]
            + REPORT_ARGS[domain],
            f"{DOMAIN_LABELS[domain]} HTML report",
        )
    return codes


# ── Phase 4: Anomaly detection ────────────────────────────────────────────────

def phase_anomaly(domains: list[str], extra: list[str]) -> dict[str, int]:
    print(f"\n{'=' * 70}")
    print("  PHASE 4 — Anomaly Detection (E1 fault injection + E2 ML evaluation)")
    print(f"{'=' * 70}")
    codes: dict[str, int] = {}
    for domain in domains:
        codes[domain] = _run(
            [sys.executable, ANOMALY_RUNNERS[domain]] + extra,
            f"{DOMAIN_LABELS[domain]} anomaly detection (E1+E2)",
        )
    return codes


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(
    phase_results: dict[str, dict[str, int]],
    skipped: list[str],
) -> bool:
    any_failed = False
    print(f"\n{'=' * 70}")
    print("  SensorWF — Run Complete")
    print(f"{'=' * 70}")
    for phase, codes in phase_results.items():
        for key, rc in codes.items():
            label  = f"{phase} / {DOMAIN_LABELS.get(key, key)}"
            status = "OK" if rc == 0 else f"FAILED (exit {rc})"
            flag   = " *" if rc != 0 else ""
            print(f"  {label:<50} {status}{flag}")
            if rc != 0:
                any_failed = True
    for name in skipped:
        print(f"  {name:<50} skipped")
    print(f"{'=' * 70}")
    if any_failed:
        print("  (*) One or more stages failed — check output above.")
    return any_failed


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args, extra = _parse_args()
    domains = ALL_DOMAINS if args.domain == "all" else [args.domain]

    all_results: dict[str, dict[str, int]] = {}
    skipped: list[str] = []

    # Phase 1 — always runs
    all_results["Core (M1-M5)"] = phase_core(domains, extra)

    # Phase 2 — CWL export (once, domain-agnostic)
    if args.skip_cwl:
        skipped.append("CWL export")
    else:
        rc = phase_cwl()
        all_results["CWL export"] = {"registry": rc}

    # Phase 3 — HTML reports
    if args.skip_report:
        skipped.append("HTML report")
    else:
        all_results["HTML report"] = phase_reports(domains)

    # Phase 4 — anomaly detection (last — longest stage)
    if args.skip_anomaly:
        skipped.append("Anomaly (E1+E2)")
    else:
        all_results["Anomaly (E1+E2)"] = phase_anomaly(domains, extra)

    any_failed = _print_summary(all_results, skipped)
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
