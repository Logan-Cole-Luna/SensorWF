"""
fig_cross_family.py
Generate Figure: cross-family aggregated detection metrics heatmap + bar chart.

Reads results/<Family>/injected/ml_metrics_by_tier.csv for all families,
aggregates the "total" tier row for each family, and produces:
  results/figures/fig_cross_family.png
"""

from __future__ import annotations
import os
import csv
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = "results"
OUT_PATH    = os.path.join(RESULTS_DIR, "figures", "fig_cross_family.png")

FAMILIES = ["AccelerometerTest", "GyroTest", "ReactionWheelTest", "ThermalTest"]
FAMILY_LABELS = ["Accel.", "Gyro", "ReactionWheel", "Thermal"]

DETECTORS = ["Autoencoder", "IsolationForest", "RobustRollingZScore", "ZScore"]
DET_SHORT  = ["AE", "IF", "RRZS", "ZS"]

METRICS   = ["recall", "fpr", "f1", "auc_roc"]
MET_LABEL = ["Recall", "FPR", "F1", "AUC-ROC"]

COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]  # one per detector


def load_total_row(csv_path: str) -> dict[str, dict[str, float]]:
    """Return {detector: {metric: value}} for the 'total' tier rows."""
    rows: dict[str, dict[str, float]] = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["tier"] == "total":
                det = row["detector"]
                rows[det] = {m: float(row[m]) for m in METRICS}
    return rows


# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
data: dict[str, dict[str, dict[str, float]]] = {}  # family -> detector -> metric -> value
for family in FAMILIES:
    csv_path = os.path.join(RESULTS_DIR, family, "injected", "ml_metrics_by_tier.csv")
    if os.path.isfile(csv_path):
        data[family] = load_total_row(csv_path)
    else:
        print(f"WARNING: missing {csv_path}")

present_families = [f for f in FAMILIES if f in data]
family_labels    = [FAMILY_LABELS[FAMILIES.index(f)] for f in present_families]
n_fam = len(present_families)

# ------------------------------------------------------------------
# Build arrays: shape (n_families, n_detectors) per metric
# ------------------------------------------------------------------
def metric_array(metric: str) -> np.ndarray:
    arr = np.full((n_fam, len(DETECTORS)), np.nan)
    for fi, fam in enumerate(present_families):
        for di, det in enumerate(DETECTORS):
            if det in data[fam]:
                arr[fi, di] = data[fam][det][metric]
    return arr

recall_arr  = metric_array("recall")
auc_arr     = metric_array("auc_roc")
fpr_arr     = metric_array("fpr")
f1_arr      = metric_array("f1")

# ------------------------------------------------------------------
# Figure: 2x2 grouped bar charts (Recall, AUC-ROC, FPR, F1)
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
fig.suptitle("Cross-Family Detection Performance by Detector", fontsize=11, fontweight="bold")

x = np.arange(n_fam)
bar_w = 0.18
offsets = np.linspace(-(len(DETECTORS)-1)/2, (len(DETECTORS)-1)/2, len(DETECTORS)) * bar_w

plot_data = [
    (axes[0, 0], recall_arr, "Recall",   (0.0, 1.0)),
    (axes[0, 1], auc_arr,    "AUC-ROC",  (0.4, 1.0)),
    (axes[1, 0], fpr_arr,    "FPR",      (0.0, 0.7)),
    (axes[1, 1], f1_arr,     "F1",       (0.0, 0.6)),
]

for ax, arr, ylabel, ylim in plot_data:
    for di, (det, color, short) in enumerate(zip(DETECTORS, COLORS, DET_SHORT)):
        vals = arr[:, di]
        bars = ax.bar(x + offsets[di], vals, bar_w, color=color,
                      label=short, alpha=0.88, edgecolor="white", linewidth=0.4)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(family_labels, fontsize=8)
    ax.set_ylim(ylim)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

# Shared legend at bottom
patches = [mpatches.Patch(color=c, label=s) for c, s in zip(COLORS, DET_SHORT)]
fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, -0.02))

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_PATH}")
