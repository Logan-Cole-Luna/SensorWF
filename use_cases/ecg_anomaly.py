"""
ecg_anomaly.py — Anomaly Detection Use Case for SensorWF ECG Domain
=====================================================================
Extends the SensorWF core pipeline (M1-M3, M6, M7) with synthetic fault
injection (M4) and anomaly detection evaluation (M5) for MIT-BIH ECG records.

This script is a use-case extension, not part of the reusable core framework.
It demonstrates that the core pipeline's clean signal CSVs and feature matrices
are directly analysis-ready for downstream anomaly detection tasks.

Prerequisites
─────────────
  Run the core pipeline first (or let use_case.py handle it):
    python use_case.py --domain ecg --skip-anomaly
    python use_cases/run_ecg.py [--records 100 106 ...]

  Core outputs consumed:
    results/ECG/mitdb/record_<id>/signal_clean.csv

Outputs
───────
  results/ECG/mitdb/record_<id>/injected/
    injection_summary.csv    -- M4 fault manifest
    signal_injected_*.csv    -- injected signal variants
    labels_*.csv             -- ground-truth anomaly labels
    ml_results.csv           -- per-variant per-detector metrics (M5)
    ml_metrics_by_tier.csv   -- tier-aggregated metrics (M5)

Usage
─────
  python use_cases/ecg_anomaly.py [--records 100 106 108]
                                   [--results-dir results/ECG/mitdb]
                                   [--seed 42] [--n-variants 2] [--fast]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure project root is in Python path for relative imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from scripts.adapters.ecg import ECGAdapter
from scripts.evaluator import build_generic_feature_matrix, run_generic_ml_evaluation

# ── Tier / injection parameters ────────────────────────────────────────────────
_TIERS    = ["easy", "medium", "hard"]
_TIER_AMP = {"easy": 1.45, "medium": 1.05, "hard": 0.72}
_TIER_DUR = {"easy": 0.30, "medium": 0.95, "hard": 1.55}

_DEFAULT_RECORDS = [
    "100", "101", "103", "105", "106", "108", "109", "111",
    "112", "113", "115", "118", "119", "200", "201", "205",
    "208", "213", "215", "221",
]
_RESULTS_DIR     = os.path.join("results", "ECG", "mitdb")
_WINDOW          = 100
_SESSION_MINUTES = 5.0


# ===========================================================================
# M4 — ECG fault injection (use-case only)
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
    """Apply one ECG artifact morphology to a clean DataFrame."""
    out   = df.copy()
    n     = len(out)
    amp   = _TIER_AMP[tier]
    dur   = _TIER_DUR[tier]
    sig_s = _signal_std(df, channels)
    sev   = {"easy": "low", "medium": "medium", "hard": "high"}[tier]

    start, end = _pick_window(n, rng, base_frac=0.40, dur_scale=dur)
    affected   = list(channels)

    if tag == "ecg_baseline_wander":
        freq  = rng.uniform(0.15, 0.40)
        t     = np.arange(end - start, dtype=float) / sample_hz
        drift = amp * sig_s * 1.5 * np.sin(2 * np.pi * freq * t)
        for ch in channels:
            if ch in out.columns:
                out.iloc[start:end, out.columns.get_loc(ch)] += drift

    elif tag == "ecg_electrode_dropout":
        ch = rng.choice(channels)
        affected = [str(ch)]
        out.iloc[start:end, out.columns.get_loc(ch)] = 0.0

    elif tag == "ecg_emg_burst":
        noise_amp = amp * sig_s * 2.0
        for ch in channels:
            if ch in out.columns:
                noise = rng.normal(0, noise_amp, end - start)
                out.iloc[start:end, out.columns.get_loc(ch)] += noise

    elif tag == "ecg_powerline_noise":
        # 10 Hz = alias of 60 Hz mains after 360→50 Hz decimation
        t  = np.arange(end - start, dtype=float) / sample_hz
        pl = amp * sig_s * 0.8 * np.sin(2 * np.pi * 10.0 * t)
        for ch in channels:
            if ch in out.columns:
                out.iloc[start:end, out.columns.get_loc(ch)] += pl

    elif tag == "ecg_amplitude_scale":
        scale = 1.0 + amp * rng.choice([-1, 1]) * rng.uniform(0.4, 0.9)
        ch = rng.choice(channels)
        affected = [str(ch)]
        out.iloc[start:end, out.columns.get_loc(ch)] *= scale

    elif tag == "ecg_lead_inversion":
        ch = rng.choice(channels)
        affected = [str(ch)]
        out.iloc[start:end, out.columns.get_loc(ch)] *= -1.0

    else:
        raise ValueError(f"Unknown ECG fault tag: {tag}")

    label = {
        "anomaly_type":      tag,
        "start_idx":         int(start),
        "end_idx":           int(end - 1),
        "severity":          sev,
        "affected_channels": affected,
    }
    return out, label


# ===========================================================================
# Per-record use-case runner
# ===========================================================================

def run_ecg_anomaly(
    record_id: str,
    results_dir: str,
    seed: int,
    n_variants: int,
    fast: bool,
    rerun: bool = False,
) -> dict:
    """Run M4 + M5 for one ECG record, consuming core pipeline outputs."""
    record_dir = os.path.join(results_dir, f"record_{record_id}")
    clean_csv  = os.path.join(record_dir, "signal_clean.csv")
    inj_dir    = os.path.join(record_dir, "injected")

    if not os.path.isfile(clean_csv):
        print(f"  SKIP {record_id}: {clean_csv} not found — run run_ecg.py first")
        return {}

    ml_cache = os.path.join(inj_dir, "ml_results.csv")
    if not rerun and os.path.isfile(ml_cache):
        print(f"  [CACHE] record {record_id} — skipping M5 (pass --rerun to force)")
        return {"record_id": record_id, "n_injected": 0, "n_ml_evals": 0, "timing_s": {}}

    os.makedirs(inj_dir, exist_ok=True)

    df = pd.read_csv(clean_csv, low_memory=False)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    channels = [
        c for c in df.columns
        if c not in {"elapsed_s", "timestamp"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    adapter     = ECGAdapter(session_minutes=_SESSION_MINUTES)
    fault_types = adapter.get_fault_types()
    tiers       = [_TIERS[0]] if fast else _TIERS

    # ── M4: Fault Injection ─────────────────────────────────────────────────
    t0             = time.time()
    files_generated: list[tuple] = []
    injection_rows: list[dict]   = []
    master_rng = np.random.default_rng(seed)

    for fault in fault_types:
        tag = fault["tag"]
        for tier in tiers:
            for v in range(1, n_variants + 1):
                rng = np.random.default_rng(int(master_rng.integers(0, 2**31)))
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
                    "description": fault["description"], **label,
                })

    pd.DataFrame(injection_rows).to_csv(
        os.path.join(inj_dir, "injection_summary.csv"), index=False)
    t_m4 = time.time() - t0
    print(f"  [M4] {len(files_generated)} variants in {t_m4:.1f}s")

    # ── M5: Anomaly Detection ───────────────────────────────────────────────
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
    print(f"  [M5] {len(ml_results)} evaluations in {t_m5:.1f}s")

    if ml_results:
        ml_df = pd.DataFrame(ml_results)
        ml_df.to_csv(os.path.join(inj_dir, "ml_results.csv"), index=False)

        metrics_cols = ["accuracy", "fpr", "recall", "f1", "auc_roc", "auc_pr", "event_rate"]
        avail = [c for c in metrics_cols if c in ml_df.columns]
        tier_rows: list[dict] = []
        for tier in sorted(ml_df["tier"].unique()):
            for det in sorted(ml_df["detector"].unique()):
                sub = ml_df[(ml_df["tier"] == tier) & (ml_df["detector"] == det)]
                if not sub.empty:
                    row = {"tier": tier, "detector": det}
                    row.update(sub[avail].mean().round(4).to_dict())
                    tier_rows.append(row)
        for det in sorted(ml_df["detector"].unique()):
            sub = ml_df[ml_df["detector"] == det]
            row = {"tier": "total", "detector": det}
            row.update(sub[avail].mean().round(4).to_dict())
            tier_rows.append(row)
        pd.DataFrame(tier_rows).to_csv(
            os.path.join(inj_dir, "ml_metrics_by_tier.csv"), index=False)

        by_det: dict[str, list] = {}
        by_det_pr: dict[str, list] = {}
        for r in ml_results:
            if "error" in r:
                continue
            d = r.get("detector", "?")
            by_det.setdefault(d, []).append(r.get("auc_roc", float("nan")))
            by_det_pr.setdefault(d, []).append(r.get("auc_pr", float("nan")))
        parts    = " | ".join(f"{d}: {sum(v)/len(v):.3f}" for d, v in by_det.items())
        parts_pr = " | ".join(f"{d}: {sum(v)/len(v):.3f}" for d, v in by_det_pr.items())
        print(f"  [M5] AUC-ROC: {parts}")
        print(f"  [M5] AUC-PR:  {parts_pr}")

    return {
        "record_id":    record_id,
        "n_injected":  len(files_generated),
        "n_ml_evals":  len(ml_results),
        "timing_s":    {"m4": t_m4, "m5": t_m5},
    }


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SensorWF ECG anomaly detection use case (M4 + M5)")
    parser.add_argument("--records",     nargs="+", default=_DEFAULT_RECORDS,
                        help="MIT-BIH record IDs to process")
    parser.add_argument("--results-dir", default=_RESULTS_DIR,
                        help="Core pipeline results dir (default: results/ECG/mitdb)")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--n-variants",  type=int, default=2,
                        help="Injection variants per fault × tier (default: 2)")
    parser.add_argument("--fast",        action="store_true",
                        help="Easy tier only (faster for testing)")
    parser.add_argument("--rerun",       action="store_true",
                        help="Ignore cached ml_results.csv and rerun M4+M5")
    args = parser.parse_args()

    print("=" * 64)
    print("SensorWF -- ECG Anomaly Detection Use Case (M4 + M5)")
    print(f"Records : {args.records}")
    print(f"Results : {args.results_dir}")
    print("=" * 64)

    results = []
    for rec_id in args.records:
        print(f"\n── Record {rec_id} {'─' * 40}")
        r = run_ecg_anomaly(
            record_id=rec_id,
            results_dir=args.results_dir,
            seed=args.seed,
            n_variants=args.n_variants,
            fast=args.fast,
            rerun=args.rerun,
        )
        if r:
            results.append(r)

    print(f"\nDone. {len(results)}/{len(args.records)} records processed.")
    if results:
        total_inj = sum(r["n_injected"] for r in results)
        total_ml  = sum(r["n_ml_evals"] for r in results)
        print(f"  Total injections : {total_inj}")
        print(f"  Total ML evals   : {total_ml}")


if __name__ == "__main__":
    main()
