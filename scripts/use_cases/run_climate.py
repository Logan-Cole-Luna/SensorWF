"""
run_climate.py -- SensorWF use-case pipeline for the Jena Climate Dataset.

Modules executed
----------------
M1  DataIngestion      : ClimateAdapter -- load year slice, rename columns, fill gaps
M2  QualityAssessment  : generic quality report (NaN, stuck, timing, trend detection)
M3  FeatureEngineering : raw + diff + rolling stats (window=24, 4 hours @ 10-min)
M4  FaultInjection     : 6 meteorological fault types × 3 tiers × 2 variants
M5  AnomalyDetection   : ZScore, RobustRollingZScore, IsolationForest, Autoencoder
M6  SemanticAnnotation : map IF importances → climate.owl classes
M7  ProvenanceExport   : PROV-O/ProvONE trace to results/Climate/provenance.ttl

Periods processed
-----------------
  2011H1 : Jan-Jun 2011 (~26,280 rows)
  2012H1 : Jan-Jun 2012 (~26,280 rows)

Usage
-----
  python run_climate.py [--years 2011 2012] [--months 1 6]
                        [--data-path data/Climate/jena_climate_2009_2016.csv]
                        [--results-dir results/Climate/jena] [--seed 42] [--fast]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from scripts.adapters.climate import ClimateAdapter
from scripts.pipeline_core import run_quality_assessment, save_quality_report
from scripts.evaluator import (
    build_generic_feature_matrix,
    run_generic_ml_evaluation,
    print_ml_summary as print_metrics_summary,
)
from scripts.provenance_recorder import ProvenanceRecorder

# ── Tier parameters ─────────────────────────────────────────────────────────
_TIERS    = ["easy", "medium", "hard"]
_TIER_AMP = {"easy": 1.45, "medium": 1.05, "hard": 0.72}
_TIER_DUR = {"easy": 0.30, "medium": 0.95, "hard": 1.55}

_DEFAULT_YEARS    = [2011, 2012]
_DEFAULT_MONTHS   = (1, 6)
_DATA_PATH        = os.path.join("data", "Climate", "jena_climate_2009_2016.csv")
_RESULTS_DIR      = os.path.join("results", "Climate", "jena")
_N_VARIANTS       = 2
_SEED             = 42
_WINDOW           = 24   # 4 hours at 10-min resolution


# ===========================================================================
# M4 -- Climate fault injection
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


def _channel_std(df: pd.DataFrame, channel: str) -> float:
    if channel not in df.columns:
        return 1.0
    s = pd.to_numeric(df[channel], errors="coerce").dropna()
    v = float(s.std()) if len(s) > 1 else 1.0
    return v if v > 1e-9 else 1.0


def inject_climate_fault(
    df: pd.DataFrame,
    tag: str,
    tier: str,
    rng: np.random.Generator,
    channels: list[str],
) -> tuple[pd.DataFrame, dict]:
    """Apply one meteorological sensor fault to a clean Climate DataFrame."""
    out = df.copy()
    n   = len(out)
    amp = _TIER_AMP[tier]
    dur = _TIER_DUR[tier]
    sev = {"easy": "low", "medium": "medium", "hard": "high"}[tier]

    start, end = _pick_window(n, rng, base_frac=0.40, dur_scale=dur)

    # Pick a primary target channel (physical channels, avoid derived pressure)
    primary_candidates = [c for c in channels if c in df.columns
                          and "T_" in c or "rh" in c or "wv" in c or "p_" in c]
    if not primary_candidates:
        primary_candidates = [c for c in channels if c in df.columns]
    ch_primary = str(rng.choice(primary_candidates))
    affected   = [ch_primary]
    ch_std     = _channel_std(df, ch_primary)

    if tag == "clim_sensor_drift":
        # Quadratic drift on primary channel
        t    = np.linspace(0, 1, end - start)
        drift = amp * ch_std * 3.0 * t ** 2
        if ch_primary in out.columns:
            out.iloc[start:end, out.columns.get_loc(ch_primary)] += drift

    elif tag == "clim_stuck_sensor":
        # Freeze channel at its pre-fault value
        if ch_primary in out.columns:
            stuck_val = float(out.iloc[max(0, start - 1)][ch_primary])
            out.iloc[start:end, out.columns.get_loc(ch_primary)] = stuck_val

    elif tag == "clim_spike_burst":
        # Random spikes of large amplitude
        n_spikes  = max(3, int((end - start) * 0.1))
        spike_idx = rng.choice(np.arange(start, end), size=n_spikes, replace=False)
        spike_amp = amp * ch_std * rng.uniform(4.0, 7.0, size=n_spikes)
        spike_amp *= rng.choice([-1, 1], size=n_spikes)
        if ch_primary in out.columns:
            col_loc = out.columns.get_loc(ch_primary)
            for idx, sa in zip(spike_idx, spike_amp):
                out.iloc[idx, col_loc] += sa

    elif tag == "clim_dropout_zero":
        # Channel drops to zero
        if ch_primary in out.columns:
            out.iloc[start:end, out.columns.get_loc(ch_primary)] = 0.0

    elif tag == "clim_scale_error":
        # Multiplicative calibration error
        scale = 1.0 + amp * rng.choice([-1, 1]) * rng.uniform(0.3, 0.7)
        if ch_primary in out.columns:
            out.iloc[start:end, out.columns.get_loc(ch_primary)] *= scale

    elif tag == "clim_sign_flip":
        # Sign inversion (wiring reversal)
        if ch_primary in out.columns:
            out.iloc[start:end, out.columns.get_loc(ch_primary)] *= -1.0

    else:
        raise ValueError(f"Unknown climate fault tag: {tag}")

    label = {
        "anomaly_type":     tag,
        "start_idx":        int(start),
        "end_idx":          int(end - 1),
        "severity":         sev,
        "affected_channels": affected,
    }
    return out, label


# ===========================================================================
# M6 -- Climate semantic annotation
# ===========================================================================

# Map channel prefixes to OWL classes in climate.owl
_CLIM_CLASS_MAP = {
    "T_degC":    "clim:Temperature",
    "Tpot_K":    "clim:PotentialTemperature",
    "Tdew_degC": "clim:DewPointTemperature",
    "p_mbar":    "clim:AtmosphericPressure",
    "rh_pct":    "clim:RelativeHumidity",
    "sh_gkg":    "clim:SpecificHumidity",
    "H2OC_mmolmol": "clim:WaterVapourConcentration",
    "VPmax_mbar":  "clim:SaturationVapourPressure",
    "VPact_mbar":  "clim:ActualVapourPressure",
    "VPdef_mbar":  "clim:VapourPressureDeficit",
    "rho_gm3":   "clim:AirDensity",
    "wv_ms":     "clim:WindSpeed",
    "max_wv_ms": "clim:MaxWindSpeed",
    "wd_deg":    "clim:WindDirection",
}


def _build_climate_kg(
    ml_results: list[dict],
    feature_names: list[str],
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    importances: dict[str, float] = {}
    for row in ml_results:
        if row.get("detector") == "IsolationForest" and "if_importances" in row:
            for feat, imp in zip(row.get("if_feature_names", feature_names),
                                 row["if_importances"]):
                importances[feat] = max(importances.get(feat, 0.0), float(imp))

    nodes, edges = [], []
    seen_classes: set[str] = set()

    for feat in feature_names:
        nodes.append({"node_id": feat, "node_type": "feature", "label": feat})
        # Match feature to class by base channel name
        base = feat.replace("d_", "").replace("rm_", "").replace("rs_", "")
        cls  = _CLIM_CLASS_MAP.get(base)
        if cls and cls not in seen_classes:
            local = cls.split(":")[-1]
            nodes.append({"node_id": cls, "node_type": "ontologyclass", "label": local})
            seen_classes.add(cls)
        if cls:
            edges.append({
                "source":   feat,
                "target":   cls,
                "relation": "evidenceForClass",
                "weight":   importances.get(feat, 0.001),
            })

    fault_tags = set(r["tag"] for r in ml_results if "tag" in r)
    for tag in fault_tags:
        nodes.append({"node_id": tag, "node_type": "anomalytag", "label": tag})

    pd.DataFrame(nodes).to_csv(os.path.join(out_dir, "climate_kg_nodes.csv"), index=False)
    pd.DataFrame(edges).to_csv(os.path.join(out_dir, "climate_kg_edges.csv"), index=False)
    print(f"  [M6] Climate KG: {len(nodes)} nodes, {len(edges)} edges")


# ===========================================================================
# Per-year pipeline
# ===========================================================================

def run_climate_period(
    year: int,
    months: tuple[int, int],
    data_path: str,
    results_dir: str,
    seed: int,
    n_variants: int,
    fast: bool,
    recorder: ProvenanceRecorder | None,
) -> dict:
    period_label = f"{year}_M{months[0]:02d}_M{months[1]:02d}"
    period_dir   = os.path.join(results_dir, period_label)
    inj_dir      = os.path.join(period_dir, "injected")
    sem_dir      = os.path.join(period_dir, "semantic")
    os.makedirs(inj_dir, exist_ok=True)
    os.makedirs(sem_dir, exist_ok=True)
    os.makedirs(os.path.join(period_dir, "plots"), exist_ok=True)

    # ── M1: Data Ingestion ──────────────────────────────────────────────
    t0      = time.time()
    adapter = ClimateAdapter(year=year, months=months)
    df      = adapter.load(data_path)
    channels = adapter.channels
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

    # ── Overview plot (temperature + pressure + wind) ────────────────────
    plot_channels = [c for c in ["T_degC", "p_mbar", "rh_pct", "wv_ms"] if c in df.columns]
    fig, axes = plt.subplots(len(plot_channels), 1,
                             figsize=(11, 2.5 * len(plot_channels)), sharex=True)
    if len(plot_channels) == 1:
        axes = [axes]
    colors = ["#C62828", "#1565C0", "#2E7D32", "#6A1B9A"]
    for ax, ch, col in zip(axes, plot_channels, colors):
        ax.plot(df["elapsed_s"].values / 3600, df[ch].values, lw=0.4, color=col)
        ax.set_ylabel(ch, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("Elapsed time (hours)", fontsize=9)
    fig.suptitle(f"Jena Climate -- {period_label} ({len(df):,} rows @ 10-min)", fontsize=10)
    fig.savefig(os.path.join(period_dir, "plots", "01_climate_overview.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── M3: Feature Engineering ─────────────────────────────────────────
    t0 = time.time()
    X_clean, feat_names = build_generic_feature_matrix(df, channels, window=_WINDOW)
    t_m3 = time.time() - t0
    if X_clean is None:
        print(f"  [M3] SKIP {period_label}: no usable features")
        return {}
    print(f"  [M3] Feature matrix: {X_clean.shape}")

    # ── M4: Fault Injection ─────────────────────────────────────────────
    t0 = time.time()
    fault_types    = adapter.get_fault_types()
    files_generated: list[tuple]  = []
    injection_rows: list[dict]    = []
    master_rng = np.random.default_rng(seed)
    tiers      = [_TIERS[0]] if fast else _TIERS

    for fault in fault_types:
        tag = fault["tag"]
        for tier in tiers:
            for v in range(1, n_variants + 1):
                rng = np.random.default_rng(int(master_rng.integers(0, 2**31)))
                try:
                    df_inj, label = inject_climate_fault(df, tag, tier, rng, channels)
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
    print(f"  [M4] Injected {len(files_generated)} variants in {t_m4:.1f}s")

    # ── M5: Anomaly Detection ───────────────────────────────────────────
    t0 = time.time()
    ml_results = run_generic_ml_evaluation(
        injected_dir=inj_dir,
        files_generated=files_generated,
        clean_df=df,
        channels=channels,
        window=_WINDOW,
        family=f"Climate_{period_label}",
    )
    t_m5 = time.time() - t0
    print(f"  [M5] {len(ml_results)} evaluations in {t_m5:.1f}s")

    if ml_results:
        ml_df = pd.DataFrame(ml_results)
        ml_df.to_csv(os.path.join(inj_dir, "ml_results.csv"), index=False)
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
        for det in sorted(ml_df["detector"].unique()):
            sub = ml_df[ml_df["detector"] == det]
            row = {"tier": "total", "detector": det}
            row.update(sub[avail].mean().round(4).to_dict())
            tier_rows.append(row)
        pd.DataFrame(tier_rows).to_csv(
            os.path.join(inj_dir, "ml_metrics_by_tier.csv"), index=False)
        print_metrics_summary(ml_results)

    if recorder:
        recorder.record_activity(
            f"M5_Climate_AnomalyDetection_{period_label}",
            inputs={"injected_dir": inj_dir},
            outputs={"n_evals": len(ml_results)},
            params={"window": _WINDOW},
        )

    # ── M6: Semantic Annotation ─────────────────────────────────────────
    _build_climate_kg(ml_results, feat_names, sem_dir)

    if recorder:
        recorder.record_activity(
            f"M6_Climate_SemanticAnnotation_{period_label}",
            inputs={"ontology": adapter.get_ontology_path()},
            outputs={"kg_dir": sem_dir},
            params={},
        )

    summary = {
        "period":     period_label,
        "year":       year,
        "n_rows":     len(df),
        "n_features": len(feat_names),
        "n_injected": len(files_generated),
        "n_ml_evals": len(ml_results),
    }
    with open(os.path.join(period_dir, "run_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="SensorWF Climate domain pipeline")
    parser.add_argument("--years",       nargs="+", type=int, default=_DEFAULT_YEARS)
    parser.add_argument("--months",      nargs=2,   type=int, default=list(_DEFAULT_MONTHS))
    parser.add_argument("--data-path",   default=_DATA_PATH)
    parser.add_argument("--results-dir", default=_RESULTS_DIR)
    parser.add_argument("--seed",        type=int, default=_SEED)
    parser.add_argument("--n-variants",  type=int, default=_N_VARIANTS)
    parser.add_argument("--fast",        action="store_true")
    args = parser.parse_args()

    months = tuple(args.months)
    os.makedirs(args.results_dir, exist_ok=True)
    prov_path = os.path.join(args.results_dir, "provenance.ttl")
    recorder  = ProvenanceRecorder(workflow_uri="sensorwf:ClimateWorkflow",
                                   domain="atmospheric_climate")

    print("=" * 64)
    print("SensorWF -- Climate Domain Pipeline (Jena Climate Dataset)")
    print(f"Years: {args.years}  Months: {months[0]}-{months[1]}")
    print("=" * 64)

    all_summaries = []
    for year in args.years:
        print(f"\n── Year {year} {'─' * 45}")
        summary = run_climate_period(
            year=year, months=months,
            data_path=args.data_path,
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


if __name__ == "__main__":
    main()
