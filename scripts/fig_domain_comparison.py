"""
fig_domain_comparison.py
Cross-domain AUC-ROC and Recall comparison across all three SensorWF domains:
  Satellite (AccelerometerTest), Biomedical ECG (MIT-BIH), Atmospheric Climate (Jena).

Reads ml_metrics_by_tier.csv from each domain's results and aggregates the
'total' tier row for each detector across sessions.

Produces: results/figures/fig_domain_comparison.png
"""

from __future__ import annotations
import os, csv, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_PATH = os.path.join("results", "figures", "fig_domain_comparison.png")

# ── Domain definitions ──────────────────────────────────────────────────────

DOMAINS = {
    "Satellite\n(SATLL)": [
        "results/AccelerometerTest/injected/ml_metrics_by_tier.csv",
        "results/GyroTest/injected/ml_metrics_by_tier.csv",
        "results/ReactionWheelTest/injected/ml_metrics_by_tier.csv",
        "results/ThermalTest/injected/ml_metrics_by_tier.csv",
    ],
    "Biomedical ECG\n(MIT-BIH)": [
        "results/ECG/mitdb/record_100/injected/ml_metrics_by_tier.csv",
        "results/ECG/mitdb/record_106/injected/ml_metrics_by_tier.csv",
        "results/ECG/mitdb/record_108/injected/ml_metrics_by_tier.csv",
        "results/ECG/mitdb/record_200/injected/ml_metrics_by_tier.csv",
        "results/ECG/mitdb/record_201/injected/ml_metrics_by_tier.csv",
    ],
    "Atmospheric\nClimate (Jena)": [
        "results/Climate/jena/2011_M01_M06/injected/ml_metrics_by_tier.csv",
        "results/Climate/jena/2012_M01_M06/injected/ml_metrics_by_tier.csv",
    ],
}

DETECTORS = ["Autoencoder", "IsolationForest", "RobustRollingZScore", "ZScore"]
DET_SHORT  = ["AE", "IF", "RRZS", "ZS"]
DET_COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

METRICS = ["recall", "fpr", "f1", "auc_roc"]
MET_LABELS = ["Recall", "FPR", "F1", "AUC-ROC"]


# ── Load aggregated 'total' row per domain × detector ──────────────────────

def load_domain_metrics(csv_paths: list[str]) -> dict[str, dict[str, float]]:
    """Load and average 'total' tier metrics across all sessions for a domain."""
    accumulated: dict[str, list] = {d: {m: [] for m in METRICS} for d in DETECTORS}
    for path in csv_paths:
        if not os.path.isfile(path):
            continue
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("tier") != "total":
                    continue
                det = row.get("detector", "")
                if det not in accumulated:
                    continue
                for m in METRICS:
                    try:
                        accumulated[det][m].append(float(row[m]))
                    except (KeyError, ValueError):
                        pass
    result: dict[str, dict[str, float]] = {}
    for det, mdict in accumulated.items():
        result[det] = {m: float(np.mean(v)) if v else float("nan")
                       for m, v in mdict.items()}
    return result


domain_data: dict[str, dict] = {}
for domain_label, paths in DOMAINS.items():
    domain_data[domain_label] = load_domain_metrics(paths)

domain_labels = list(domain_data.keys())
n_domains = len(domain_labels)

# ── Figure: 2×2 grouped bar charts ─────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
fig.suptitle("SensorWF Cross-Domain Detection Performance\n"
             "(Satellite, Biomedical ECG, Atmospheric Climate)",
             fontsize=11, fontweight="bold")

x = np.arange(n_domains)
bar_w = 0.17
offsets = np.linspace(-(len(DETECTORS)-1)/2, (len(DETECTORS)-1)/2, len(DETECTORS)) * bar_w

ylims = [(0.0, 1.0), (0.4, 1.0), (0.0, 0.6), (0.0, 0.65)]

for ax_idx, (metric, label, ylim) in enumerate(zip(METRICS, MET_LABELS, ylims)):
    ax = axes[ax_idx // 2][ax_idx % 2]
    for di, (det, color, short) in enumerate(zip(DETECTORS, DET_COLORS, DET_SHORT)):
        vals = [domain_data[dl][det].get(metric, float("nan")) for dl in domain_labels]
        ax.bar(x + offsets[di], vals, bar_w, color=color,
               label=short, alpha=0.88, edgecolor="white", linewidth=0.5)
    ax.set_ylabel(label, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(domain_labels, fontsize=8.5)
    ax.set_ylim(ylim)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

patches = [mpatches.Patch(color=c, label=s) for c, s in zip(DET_COLORS, DET_SHORT)]
fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, -0.02))

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_PATH}")

# ── Also produce a simple radar/summary table for the paper text ────────────

print("\nCross-domain AUC-ROC summary (total tier, averaged across sessions):")
print(f"{'Domain':<28} {'AE':>7} {'IF':>7} {'RRZS':>7} {'ZS':>7}")
print("-" * 56)
for dl in domain_labels:
    row_str = f"{dl.replace(chr(10), ' '):<28}"
    for det in DETECTORS:
        val = domain_data[dl][det].get("auc_roc", float("nan"))
        row_str += f" {val:>7.3f}"
    print(row_str)
