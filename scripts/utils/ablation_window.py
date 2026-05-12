"""
ablation_window.py — M3 rolling-window size ablation for SensorWF.

Evaluates how the choice of rolling window w (15, 30, 50, 100 samples) in M3
feature engineering affects detection AUC-ROC and AUC-PR across all detectors.

Reads existing clean signal CSVs; re-runs M3 feature engineering + M5 evaluation
for each window size using the existing injected variants (no new injection needed).

Usage
-----
  python scripts/utils/ablation_window.py [--domain satellite|ecg|climate]
  Results → results/ablation/window_sensitivity.csv
             results/ablation/fig_ablation_window.png
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
import pathlib
# Ensure project root is on sys.path when run as a standalone script
_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.evaluator import (
    build_generic_feature_matrix,
    _DETECTORS, _TRAIN_FRAC, _threshold_percentile, _if_contamination, _ae_seq_len,
    ZScoreDetector, RobustRollingZScoreDetector, IsolationForestDetector,
    AutoencoderDetector, LOFDetector, evaluate_detector, _fit_threshold,
)

_WINDOWS = [15, 30, 50, 100]
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_OUT_DIR = str(_PROJECT_ROOT / "results" / "ablation")


def _build_detector(det_name: str, family: str, tag: str, contamination: float,
                    seq_len: int) -> object:
    if det_name == "ZScore":
        return ZScoreDetector()
    elif det_name == "RobustRollingZScore":
        return RobustRollingZScoreDetector(family=family)
    elif det_name == "IsolationForest":
        return IsolationForestDetector(contamination=contamination)
    elif det_name == "LOF":
        return LOFDetector(contamination=contamination)
    else:
        return AutoencoderDetector(seq_len=seq_len, denoise_sigma=0.06)


def run_window_ablation_for_session(
    clean_csv: str,
    channels: list[str],
    inj_dir: str,
    family: str,
    windows: list[int] = _WINDOWS,
) -> list[dict]:
    """
    For each window size, rebuild features and re-evaluate all detectors
    on every injected variant found in inj_dir.
    Returns list of {window, detector, tag, tier, auc_roc, auc_pr} dicts.
    """
    clean_df = pd.read_csv(clean_csv, low_memory=False)
    if "timestamp" in clean_df.columns:
        clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"], errors="coerce")

    inj_csvs = [f for f in os.listdir(inj_dir) if f.startswith("signal_injected_")]
    results = []

    for w in windows:
        X_clean, feat_names = build_generic_feature_matrix(clean_df, channels, window=w)
        if X_clean is None or len(X_clean) < 20:
            continue
        n_train = max(10, int(len(X_clean) * _TRAIN_FRAC))
        X_tr = X_clean[:n_train]

        for fname in inj_csvs:
            parts = fname.replace("signal_injected_", "").replace(".csv", "").rsplit("_v", 1)
            if len(parts) != 2:
                continue
            tag_tier = parts[0]
            v_str = parts[1]
            tier_parts = tag_tier.rsplit("_", 1)
            if len(tier_parts) == 2 and tier_parts[1] in ("easy", "medium", "hard"):
                tag, tier = tier_parts[0], tier_parts[1]
            else:
                tag, tier = tag_tier, "unknown"

            inj_path   = os.path.join(inj_dir, fname)
            label_path = os.path.join(inj_dir, f"labels_{tag}_{tier}_v{v_str}.csv")
            if not os.path.isfile(label_path):
                continue

            try:
                df_inj = pd.read_csv(inj_path, low_memory=False)
                label_row = pd.read_csv(label_path).iloc[0]
                start_idx = int(label_row.get("start_idx", 0))
                end_idx   = int(label_row.get("end_idx", 0))
            except Exception:
                continue

            X_inj, _ = build_generic_feature_matrix(df_inj, channels, window=w)
            if X_inj is None:
                continue
            n = min(len(df_inj), len(X_inj))
            labels = np.zeros(n, dtype=bool)
            lo = max(0, min(start_idx, n - 1))
            hi = max(lo, min(end_idx, n - 1))
            labels[lo: hi + 1] = True
            elapsed = (df_inj["elapsed_s"].to_numpy(dtype=float)[:n]
                       if "elapsed_s" in df_inj.columns else np.arange(n, dtype=float))

            n_feats = min(X_tr.shape[1], X_inj.shape[1])
            X_tr_f  = X_tr[:, :n_feats]
            X_inj_f = X_inj[:, :n_feats]
            fnames  = feat_names[:n_feats]

            for det_name in _DETECTORS:
                thr_pct = _threshold_percentile(det_name, family, "signal", tag)
                cont = _if_contamination(family, "signal", tag)
                sl   = _ae_seq_len(family, tag)
                try:
                    det = _build_detector(det_name, family, tag, cont, sl)
                    det.fit(X_tr_f, fnames)
                    use_evt = det_name in ("IsolationForest", "LOF")
                    thr_val = _fit_threshold(det, X_tr_f, thr_pct, use_evt=use_evt)
                    scores  = det.score(X_inj_f)
                    m       = evaluate_detector(scores, labels, elapsed,
                                                train_scores=det.score(X_tr_f),
                                                threshold_pct=thr_pct,
                                                threshold_value=thr_val)
                    results.append({
                        "window": w, "detector": det_name, "tag": tag, "tier": tier,
                        "auc_roc": m["auc_roc"], "auc_pr": m["auc_pr"],
                        "recall": m["recall"], "fpr": m["fpr"],
                    })
                except Exception:
                    pass
    return results


def _find_sessions(domain: str) -> list[tuple[str, list[str], str, str]]:
    """Return list of (clean_csv, channels, inj_dir, family) tuples."""
    root = _PROJECT_ROOT
    sessions = []
    if domain == "satellite":
        for fam in ["AccelerometerTest", "GyroTest", "ReactionWheelTest", "ThermalTest"]:
            base = root / "results" / "satellite" / fam
            cdh_csv = str(base / "cdh_clean.csv")
            inj_dir = str(base / "injected")
            if not os.path.isfile(cdh_csv) or not os.path.isdir(inj_dir):
                continue
            df_tmp = pd.read_csv(cdh_csv, nrows=2)
            channels = [c for c in df_tmp.columns if c not in {"timestamp", "elapsed_s"}]
            sessions.append((cdh_csv, channels, inj_dir, fam))

    elif domain == "ecg":
        base = root / "results" / "ECG" / "mitdb"
        for rec_dir in sorted(os.listdir(base)):
            csv_path = str(base / rec_dir / "signal_clean.csv")
            inj_dir  = str(base / rec_dir / "injected")
            if not os.path.isfile(csv_path) or not os.path.isdir(inj_dir):
                continue
            df_tmp   = pd.read_csv(csv_path, nrows=2)
            _skip = {"elapsed_s", "timestamp"}
            channels = [c for c in df_tmp.columns
                        if c not in _skip and pd.api.types.is_numeric_dtype(df_tmp[c])]
            sessions.append((csv_path, channels, inj_dir, f"ECG_{rec_dir}"))

    elif domain == "climate":
        base = root / "results" / "Climate" / "jena"
        for period_dir in sorted(os.listdir(base)):
            csv_path = str(base / period_dir / "signal_clean.csv")
            inj_dir  = str(base / period_dir / "injected")
            if not os.path.isfile(csv_path) or not os.path.isdir(inj_dir):
                continue
            df_tmp   = pd.read_csv(csv_path, nrows=2)
            _skip = {"elapsed_s", "timestamp"}
            channels = [c for c in df_tmp.columns
                        if c not in _skip and pd.api.types.is_numeric_dtype(df_tmp[c])]
            sessions.append((csv_path, channels, inj_dir, f"Climate_{period_dir}"))

    return sessions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["satellite", "ecg", "climate"], default="ecg")
    parser.add_argument("--max-sessions", type=int, default=5,
                        help="Limit sessions for speed (0 = all)")
    args = parser.parse_args()

    sessions = _find_sessions(args.domain)
    if args.max_sessions > 0:
        sessions = sessions[: args.max_sessions]

    if not sessions:
        print(f"No sessions found for domain={args.domain}")
        return

    print(f"Window ablation: {args.domain}, {len(sessions)} session(s), "
          f"windows={_WINDOWS}")

    all_rows: list[dict] = []
    for clean_csv, channels, inj_dir, family in sessions:
        print(f"  {family} ...")
        rows = run_window_ablation_for_session(clean_csv, channels, inj_dir, family)
        all_rows.extend(rows)

    if not all_rows:
        print("No results produced.")
        return

    os.makedirs(_OUT_DIR, exist_ok=True)
    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(_OUT_DIR, f"window_sensitivity_{args.domain}.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # ── Figure via consolidated figures module ───────────────────────────────
    from scripts.utils.figures import fig_ablation_window
    fig_path = os.path.join(_OUT_DIR, f"fig_ablation_window_{args.domain}.csv")
    out_png  = os.path.join(_OUT_DIR, f"fig_ablation_window_{args.domain}.png")
    fig_ablation_window(csv_path=out_csv, out_path=out_png)

    # Print summary table
    print(f"\nMean AUC-ROC/AUC-PR by window × detector:")
    print(f"{'Window':>8} {'Detector':<22} {'AUC-ROC':>8} {'AUC-PR':>8}")
    print("-" * 52)
    for _, row in agg.sort_values(["detector", "window"]).iterrows():
        print(f"{int(row['window']):>8} {row['detector']:<22} {row['auc_roc']:>8.3f} {row['auc_pr']:>8.3f}")


if __name__ == "__main__":
    main()
