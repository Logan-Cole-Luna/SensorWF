"""
scripts/utils/figures.py — SensorWF publication figure generators.

All figures used in the IEEE eScience 2026 paper are defined here as
standalone functions. Run this module directly to regenerate all figures:

    python scripts/utils/figures.py            # all figures
    python scripts/utils/figures.py --only kg  # single figure

Functions
---------
fig_domain_comparison  Cross-domain AUC-ROC / AUC-PR comparison (Fig. 2)
fig_kg_subgraph        M4 semantic KG tripartite subgraph (Fig. 3)
fig_ablation_window    M3 rolling-window sensitivity (Fig. 4)
fig_cross_family       Satellite cross-family heatmap (supplementary)
fig_provenance         PROV-O activity graph (supplementary)
"""

from __future__ import annotations
import csv as csv_mod
import collections
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np

# ── Shared style constants ────────────────────────────────────────────────────

DETECTORS  = ["Autoencoder", "IsolationForest", "LOF", "RobustRollingZScore", "ZScore"]
DET_LABELS = ["AE", "IF", "LOF", "RRZS", "ZS"]
DET_COLORS = ["#1A6FBB", "#E87820", "#C5393A", "#2E9147", "#7B4397"]
DET_MARKERS = ["o", "s", "^", "D", "v"]
DET_LS      = ["-", "-", "-", "--", "--"]

_FIGURES_DIR = os.path.join("results", "figures")
_ABLATION_DIR = os.path.join("results", "ablation")
_DOCS_FIG_DIR = os.path.join("documentation", "EScience", "figures")


def _ensure(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _save(fig: plt.Figure, path: str, symlink_doc: bool = True) -> None:
    _ensure(path)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")
    if symlink_doc:
        _copy_to_doc(path)


def _copy_to_doc(src: str) -> None:
    fname = os.path.basename(src)
    dst = os.path.join(_DOCS_FIG_DIR, fname)
    if not os.path.isdir(_DOCS_FIG_DIR):
        return
    # Skip if src and dst resolve to the same file (symlink case)
    try:
        if os.path.realpath(src) == os.path.realpath(dst):
            return
    except OSError:
        pass
    import shutil
    shutil.copy2(src, dst)
    print(f"  Copied: {dst}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1 (in paper) — Cross-domain detection performance
# ═══════════════════════════════════════════════════════════════════════════════

def fig_domain_comparison(
    out_path: str = os.path.join(_FIGURES_DIR, "fig_domain_comparison.png"),
) -> None:
    """Horizontal grouped bar chart: AUC-ROC and AUC-PR for 5 detectors × 3 domains."""

    _DOMAINS = {
        "Satellite\n(SATLL)": [
            f"results/satellite/{f}/injected/ml_metrics_by_tier.csv"
            for f in ["AccelerometerTest", "GyroTest", "ReactionWheelTest", "ThermalTest"]
        ],
        "Biomedical ECG\n(MIT-BIH)": [
            f"results/ECG/mitdb/record_{r}/injected/ml_metrics_by_tier.csv"
            for r in [100,101,103,105,106,108,109,111,112,113,115,118,119,
                      200,201,205,208,213,215,221]
        ],
        "Atmospheric\nClimate (Jena)": [
            f"results/Climate/jena/{y}_M01_M06/injected/ml_metrics_by_tier.csv"
            for y in range(2009, 2016)
        ],
    }
    METRICS    = ["auc_roc", "auc_pr"]
    MET_LABELS = ["AUC-ROC", "AUC-PR"]
    DOM_COLORS  = ["#4A90D9", "#D45F5F", "#5CAB6D"]
    DOM_HATCHES = ["", "///", "..."]

    def _load(paths):
        acc = {d: {m: [] for m in METRICS} for d in DETECTORS}
        for path in paths:
            if not os.path.isfile(path):
                continue
            with open(path, newline="") as fh:
                for row in csv_mod.DictReader(fh):
                    if row.get("tier") != "total":
                        continue
                    det = row.get("detector", "")
                    if det not in acc:
                        continue
                    for m in METRICS:
                        try:
                            acc[det][m].append(float(row[m]))
                        except (KeyError, ValueError):
                            pass
        return {det: {m: float(np.mean(v)) if v else float("nan")
                      for m, v in mdict.items()}
                for det, mdict in acc.items()}

    domain_keys = list(_DOMAINS.keys())
    data = {dk: _load(paths) for dk, paths in _DOMAINS.items()}

    n_det = len(DETECTORS)
    n_dom = len(domain_keys)
    bar_h = 0.20
    group_gap = 0.12
    y_step = n_dom * bar_h + group_gap
    y_centers = np.arange(n_det) * y_step

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), constrained_layout=True)

    for ax_idx, (metric, met_label) in enumerate(zip(METRICS, MET_LABELS)):
        ax = axes[ax_idx]
        for di, (dk, dcol, dhatch) in enumerate(zip(domain_keys, DOM_COLORS, DOM_HATCHES)):
            y_offset = (di - (n_dom - 1) / 2) * bar_h
            vals = [data[dk][det].get(metric, float("nan")) for det in DETECTORS]
            bars = ax.barh(
                y_centers + y_offset, vals, bar_h * 0.88,
                color=dcol, hatch=dhatch, edgecolor="white",
                linewidth=0.4, alpha=0.90,
                label=dk.replace("\n", " "),
            )
            for bar_patch, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(val + 0.006, bar_patch.get_y() + bar_patch.get_height() / 2,
                            f"{val:.3f}", va="center", ha="left",
                            fontsize=5.5, color="#333333")

        ax.set_yticks(y_centers)
        ax.set_yticklabels(DET_LABELS, fontsize=8.5)
        ax.set_xlabel(met_label, fontsize=9)
        ax.set_xlim(0.0, 1.08)
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax.tick_params(axis="x", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.55, color="#aaaaaa")
        ax.set_axisbelow(True)
        for yc in y_centers[:-1]:
            ax.axhline(yc + y_step / 2, color="#dddddd", linewidth=0.6)

        if ax_idx == 0:
            ax.set_ylabel("Detector", fontsize=9, labelpad=4)

    handles = [mpatches.Patch(color=c, label=dk.replace("\n", " "), hatch=h)
               for c, dk, h in zip(DOM_COLORS, domain_keys, DOM_HATCHES)]
    axes[0].legend(handles=handles, loc="lower right", fontsize=6.5,
                   frameon=True, framealpha=0.92, handlelength=1.4,
                   title="Domain", title_fontsize=7)

    _save(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2 (in paper) — M4 semantic knowledge graph subgraph
# ═══════════════════════════════════════════════════════════════════════════════

def fig_kg_subgraph(
    out_path: str = os.path.join(_FIGURES_DIR, "fig_kg_subgraph.png"),
) -> None:
    """Tripartite layout: OWL ontology | sensor features | anomaly semantics."""

    LEFT_NODES = [
        {"id": "sat:ADCS",      "label": "sat:ADCS",    "group": "subsystem"},
        {"id": "sat:EPS/CDH",   "label": "sat:EPS/CDH", "group": "subsystem"},
        {"id": "sat:COMMS",     "label": "sat:COMMS",   "group": "subsystem"},
        {"id": "Gyroscope",     "label": "Gyroscope",   "group": "owlclass"},
        {"id": "Magnetometer",  "label": "Magnetometer","group": "owlclass"},
        {"id": "ReactionWheel", "label": "Reaction\nWheel", "group": "owlclass"},
        {"id": "PowerSensor",   "label": "Power\nSensor",   "group": "owlclass"},
        {"id": "CommsSensor",   "label": "Comms\nSensor",   "group": "owlclass"},
    ]
    CENTRE_NODES = [
        {"id": "GYRO_X",     "label": "GYRO_X",     "group": "feature"},
        {"id": "MAG_X",      "label": "MAG_X",      "group": "feature"},
        {"id": "TEMP_WHEEL", "label": "TEMP\nWHEEL","group": "feature"},
        {"id": "SYS_I",      "label": "SYS_I",      "group": "feature"},
        {"id": "OBC_PWR_I",  "label": "OBC_PWR_I",  "group": "feature"},
        {"id": "TEMP_COMMS", "label": "TEMP\nCOMMS","group": "feature"},
    ]
    RIGHT_NODES = [
        {"id": "A2_gyro",  "label": "A2 gyro\nclipping", "group": "anomalytag"},
        {"id": "A3_mag",   "label": "A3 mag\ninversion",  "group": "anomalytag"},
        {"id": "W1_wheel", "label": "W1 wheel\nrunaway",  "group": "anomalytag"},
        {"id": "C2_comms", "label": "C2 RSSI\ndrop",      "group": "anomalytag"},
        {"id": "AngVel",   "label": "angular\nvelocity",  "group": "semtype"},
        {"id": "MagField", "label": "magnetic\nfield",    "group": "semtype"},
        {"id": "WheelDyn", "label": "wheel\ndynamics",    "group": "semtype"},
    ]
    EDGES = [
        # Feature → OWL class
        ("GYRO_X",     "Gyroscope",     "mapsTo"),
        ("MAG_X",      "Magnetometer",  "mapsTo"),
        ("TEMP_WHEEL", "ReactionWheel", "mapsTo"),
        ("SYS_I",      "PowerSensor",   "mapsTo"),
        ("OBC_PWR_I",  "PowerSensor",   "mapsTo"),
        ("TEMP_COMMS", "CommsSensor",   "mapsTo"),
        # Feature → Subsystem
        ("GYRO_X",     "sat:ADCS",    "belongsTo"),
        ("MAG_X",      "sat:ADCS",    "belongsTo"),
        ("TEMP_WHEEL", "sat:ADCS",    "belongsTo"),
        ("SYS_I",      "sat:EPS/CDH", "belongsTo"),
        ("OBC_PWR_I",  "sat:EPS/CDH", "belongsTo"),
        ("TEMP_COMMS", "sat:COMMS",   "belongsTo"),
        # OWL class ⊆ Subsystem
        ("Gyroscope",    "sat:ADCS",    "subClassOf"),
        ("Magnetometer", "sat:ADCS",    "subClassOf"),
        ("ReactionWheel","sat:ADCS",    "subClassOf"),
        ("PowerSensor",  "sat:EPS/CDH", "subClassOf"),
        ("CommsSensor",  "sat:COMMS",   "subClassOf"),
        # AnomalyTag → Feature (importantFeature)
        ("A2_gyro",  "GYRO_X",     "important"),
        ("A3_mag",   "MAG_X",      "important"),
        ("W1_wheel", "TEMP_WHEEL", "important"),
        ("C2_comms", "TEMP_COMMS", "important"),
        # Feature → SemanticType
        ("GYRO_X",     "AngVel",   "hasType"),
        ("MAG_X",      "MagField", "hasType"),
        ("TEMP_WHEEL", "WheelDyn", "hasType"),
    ]

    GROUP_COLOR = {
        "subsystem": "#1565C0", "owlclass": "#C62828",
        "feature": "#2E7D32",  "anomalytag": "#6A1B9A", "semtype": "#E65100",
    }
    GROUP_SIZE = {
        "subsystem": 2000, "owlclass": 1600, "feature": 1800,
        "anomalytag": 1600, "semtype": 1400,
    }
    EDGE_STYLES = {
        "mapsTo":    {"color": "#C62828", "ls": "-",  "lw": 1.6, "alpha": 0.80},
        "belongsTo": {"color": "#1565C0", "ls": "-",  "lw": 1.3, "alpha": 0.70},
        "subClassOf":{"color": "#90A4AE", "ls": "--", "lw": 0.9, "alpha": 0.55},
        "important": {"color": "#6A1B9A", "ls": ":",  "lw": 1.5, "alpha": 0.85},
        "hasType":   {"color": "#E65100", "ls": "-.", "lw": 1.2, "alpha": 0.75},
    }

    def _col_pos(nodes, x, y_top, y_step):
        return {n["id"]: (x, y_top - i * y_step) for i, n in enumerate(nodes)}

    Y_STEP_L, Y_STEP_C, Y_STEP_R = 0.80, 1.00, 0.82
    n_l, n_c, n_r = len(LEFT_NODES), len(CENTRE_NODES), len(RIGHT_NODES)
    h_l = (n_l - 1) * Y_STEP_L
    h_c = (n_c - 1) * Y_STEP_C
    h_r = (n_r - 1) * Y_STEP_R
    max_h = max(h_l, h_c, h_r)

    pos = {
        **_col_pos(LEFT_NODES,   0.0, max_h / 2 + h_l / 2, Y_STEP_L),
        **_col_pos(CENTRE_NODES, 1.0, max_h / 2 + h_c / 2, Y_STEP_C),
        **_col_pos(RIGHT_NODES,  2.0, max_h / 2 + h_r / 2, Y_STEP_R),
    }
    node_meta = {n["id"]: n for n in LEFT_NODES + CENTRE_NODES + RIGHT_NODES}

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.set_axis_off()
    ax.set_xlim(-0.55, 2.85)
    ax.set_ylim(-0.60, max_h + 0.60)

    # Column background bands
    for (x0, x1), title, bg in [
        ((-0.45, 0.45), "OWL Ontology\n(SSN/SOSA-aligned)", "#E3F2FD"),
        (( 0.55, 1.45), "Sensor Features\n(M3 output)",     "#E8F5E9"),
        (( 1.55, 2.75), "Semantics\n(M4 / E2 output)",      "#F3E5F5"),
    ]:
        ax.add_patch(plt.Rectangle((x0, -0.45), x1 - x0, max_h + 0.90,
                                   facecolor=bg, edgecolor="#cccccc",
                                   linewidth=0.7, zorder=0))
        ax.text((x0 + x1) / 2, max_h + 0.30, title,
                ha="center", va="center", fontsize=7.5,
                fontweight="bold", color="#444444")

    # Edges
    for src, dst, pred in EDGES:
        if src not in pos or dst not in pos:
            continue
        es = EDGE_STYLES[pred]
        ax.annotate(
            "", xy=pos[dst], xytext=pos[src],
            arrowprops=dict(
                arrowstyle="-|>", color=es["color"], lw=es["lw"],
                linestyle=es["ls"], alpha=es["alpha"],
                mutation_scale=10, shrinkA=14, shrinkB=14,
            ),
        )

    # Nodes + labels
    for nid, meta in node_meta.items():
        if nid not in pos:
            continue
        x, y = pos[nid]
        ax.scatter(x, y, s=GROUP_SIZE[meta["group"]],
                   c=GROUP_COLOR[meta["group"]], zorder=3, alpha=0.92,
                   edgecolors="white", linewidths=1.2)
        ax.text(x, y, meta["label"], ha="center", va="center",
                fontsize=6.8, fontweight="bold", color="white", zorder=4,
                multialignment="center")

    # Legend
    node_lgd = [
        mpatches.Patch(facecolor=GROUP_COLOR[g], label=l)
        for g, l in [("subsystem","Subsystem (sat:)"), ("owlclass","OWL sensor class"),
                     ("feature","Feature node"), ("anomalytag","Anomaly tag (E2)"),
                     ("semtype","Semantic type")]
    ]
    edge_lgd = [
        Line2D([0],[0], color=EDGE_STYLES[p]["color"], lw=1.4,
               ls=EDGE_STYLES[p]["ls"], label=l)
        for p, l in [("mapsTo","mapsToSensorClass"), ("belongsTo","belongsToSubsystem"),
                     ("subClassOf","rdfs:subClassOf"), ("important","if:importantFeature"),
                     ("hasType","if:hasSemanticType")]
    ]
    ax.legend(handles=node_lgd + edge_lgd,
              loc="lower left", fontsize=6.0, frameon=True,
              framealpha=0.94, ncol=1, handlelength=1.4,
              bbox_to_anchor=(-0.55, -0.58), borderaxespad=0)

    ax.text(2.85, -0.52,
            r"Full graph: 869 nodes, 11$\,$762 edges",
            ha="right", va="bottom", fontsize=6.5, color="#555555", style="italic")

    _save(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3 (in paper) — M3 rolling-window sensitivity ablation
# ═══════════════════════════════════════════════════════════════════════════════

def fig_ablation_window(
    csv_path: str = os.path.join(_ABLATION_DIR, "window_sensitivity_ecg.csv"),
    out_path: str = os.path.join(_ABLATION_DIR, "fig_ablation_window_ecg.png"),
) -> None:
    """Line plot of mean AUC-ROC / AUC-PR vs. rolling window size (ECG domain)."""
    import pandas as pd

    if not os.path.isfile(csv_path):
        print(f"  SKIP fig_ablation_window: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    WINDOWS = sorted(df["window"].unique())
    METRICS = ["auc_roc", "auc_pr"]
    MET_LABELS = ["AUC-ROC", "AUC-PR"]

    agg = df.groupby(["window", "detector"])[METRICS].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9), constrained_layout=True)

    for ax_idx, (metric, met_label) in enumerate(zip(METRICS, MET_LABELS)):
        ax = axes[ax_idx]
        for det, label, color, marker, ls in zip(DETECTORS, DET_LABELS, DET_COLORS, DET_MARKERS, DET_LS):
            sub = agg[agg["detector"] == det].sort_values("window")
            if sub.empty:
                continue
            ax.plot(sub["window"], sub[metric],
                    marker=marker, color=color, label=label,
                    linewidth=1.7, markersize=5.5, linestyle=ls,
                    markeredgewidth=0.8, markeredgecolor="white")

        ax.set_xlabel("Rolling window $w$ (samples)", fontsize=9)
        ax.set_ylabel(met_label, fontsize=9)
        ax.set_xticks(WINDOWS)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

        col_data = agg[metric].dropna()
        y_lo = max(0.0, col_data.min() - 0.06)
        y_hi = min(1.0, col_data.max() + 0.06)
        ax.set_ylim(y_lo, y_hi)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.55, color="#aaaaaa")
        ax.set_axisbelow(True)

        # Annotate best at w=100
        best = (agg[agg["window"] == max(WINDOWS)]
                .sort_values(metric, ascending=False).iloc[0])
        bi = DETECTORS.index(best["detector"]) if best["detector"] in DETECTORS else 0
        ax.annotate(
            DET_LABELS[bi],
            xy=(max(WINDOWS), best[metric]),
            xytext=(max(WINDOWS) * 0.88, best[metric] + (y_hi - y_lo) * 0.10),
            fontsize=6.5, color=DET_COLORS[bi], ha="right",
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.7),
        )

        if ax_idx == 0:
            ax.legend(fontsize=7, frameon=True, framealpha=0.92,
                      loc="lower right", handlelength=1.8,
                      title="Detector", title_fontsize=7)

    fig.text(0.50, -0.04,
             "ECG domain (5 MIT-BIH records). Values averaged over all fault types and difficulty tiers.",
             ha="center", fontsize=6.5, color="#555555", style="italic")

    _save(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Supplementary — Satellite cross-family comparison
# ═══════════════════════════════════════════════════════════════════════════════

def fig_cross_family(
    results_dir: str = os.path.join("results", "satellite"),
    out_path: str = os.path.join(_FIGURES_DIR, "fig_cross_family.png"),
) -> None:
    """2×2 grouped bar chart of recall/AUC-ROC/FPR/F1 across 4 satellite families."""

    FAMILIES = ["AccelerometerTest", "GyroTest", "ReactionWheelTest", "ThermalTest"]
    FAM_LABELS = ["Accel.", "Gyro", "ReactionWheel", "Thermal"]
    DETS4 = ["Autoencoder", "IsolationForest", "RobustRollingZScore", "ZScore"]
    SHORT4 = ["AE", "IF", "RRZS", "ZS"]
    COLS4  = ["#1A6FBB", "#E87820", "#2E9147", "#7B4397"]
    METS   = ["recall", "fpr", "f1", "auc_roc"]
    MLBLS  = ["Recall", "FPR", "F1", "AUC-ROC"]

    def _load_total(path):
        rows = {}
        if not os.path.isfile(path):
            return rows
        with open(path, newline="") as fh:
            for row in csv_mod.DictReader(fh):
                if row.get("tier") == "total":
                    det = row.get("detector", "")
                    rows[det] = {m: float(row.get(m, "nan")) for m in METS}
        return rows

    data = {}
    for fam in FAMILIES:
        data[fam] = _load_total(
            os.path.join(results_dir, fam, "injected", "ml_metrics_by_tier.csv"))

    present = [f for f in FAMILIES if data[f]]
    fam_lbls = [FAM_LABELS[FAMILIES.index(f)] for f in present]

    def _arr(met):
        a = np.full((len(present), len(DETS4)), np.nan)
        for fi, fam in enumerate(present):
            for di, det in enumerate(DETS4):
                if det in data[fam]:
                    a[fi, di] = data[fam][det][met]
        return a

    bar_w = 0.18
    x = np.arange(len(present))
    offs = np.linspace(-(len(DETS4)-1)/2, (len(DETS4)-1)/2, len(DETS4)) * bar_w

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.0), constrained_layout=True)

    for ax, met, ylabel, ylim in zip(
        axes.flat, METS, MLBLS,
        [(0.0, 1.0), (0.0, 0.7), (0.0, 0.6), (0.4, 1.0)],
    ):
        arr = _arr(met)
        for di, (det, col, short) in enumerate(zip(DETS4, COLS4, SHORT4)):
            ax.bar(x + offs[di], arr[:, di], bar_w, color=col,
                   label=short, alpha=0.88, edgecolor="white", linewidth=0.4)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(fam_lbls, fontsize=8)
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.55, color="#aaaaaa")
        ax.set_axisbelow(True)

    patches = [mpatches.Patch(color=c, label=s) for c, s in zip(COLS4, SHORT4)]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.03))

    _save(fig, out_path, symlink_doc=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Supplementary — PROV-O provenance activity graph
# ═══════════════════════════════════════════════════════════════════════════════

def fig_provenance(
    ttl_path: str = os.path.join("results", "satellite", "provenance.ttl"),
    out_path: str = os.path.join(_FIGURES_DIR, "fig_provenance.png"),
) -> None:
    """Bipartite PROV-O activity / entity graph parsed from provenance.ttl."""
    import networkx as nx

    if not os.path.isfile(ttl_path):
        print(f"  SKIP fig_provenance: {ttl_path} not found")
        return

    # Parse TTL into subject → {predicate: [objects]}
    triples: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(ttl_path, encoding="utf-8") as fh:
        text = re.sub(r'#[^\n]*', '', fh.read())
    for stmt in re.split(r'\.\s', text):
        stmt = stmt.strip()
        if not stmt or stmt.startswith('@'):
            continue
        parts = re.split(r'\s*;\s*', stmt)
        first = parts[0].split()
        if len(first) < 3:
            continue
        subj, pred = first[0], first[1]
        for obj in re.split(r'\s*,\s*', " ".join(first[2:])):
            if obj.strip():
                triples[subj][pred].append(obj.strip())
        for part in parts[1:]:
            toks = part.split()
            if len(toks) >= 2:
                for obj in re.split(r'\s*,\s*', " ".join(toks[1:])):
                    if obj.strip():
                        triples[toks[0]][toks[1]].append(obj.strip())

    def _has_type(s, t):
        return any(t in x for x in triples[s].get("a", []))

    def _label(s):
        lbls = triples[s].get("rdfs:label", [])
        raw = lbls[0].strip('"') if lbls else s.split(":")[-1]
        parts = raw.split("_")
        return "_".join(parts[:3])[:28]

    activities = [s for s in triples if _has_type(s, "prov:Activity")
                  and not _has_type(s, "provone:Workflow")]
    entities   = [s for s in triples if _has_type(s, "prov:Entity")]

    KW = ("M4", "M5", "E1", "E2", "Semantic", "Provenance", "Injection", "Detection")
    core_acts = [a for a in activities
                 if any(k.lower() in a.lower() or k.lower() in _label(a).lower()
                        for k in KW)] or activities

    linked = set()
    for act in core_acts:
        for e in triples[act].get("prov:used", []):
            e = e.strip(" ,;.")
            if e in entities:
                linked.add(e)
    for ent in entities:
        for ga in triples[ent].get("prov:wasGeneratedBy", []):
            if ga.strip(" ,;.") in core_acts:
                linked.add(ent)

    G = nx.DiGraph()
    for a in core_acts:
        G.add_node(a, kind="activity", label=_label(a))
    for e in list(linked)[:20]:
        G.add_node(e, kind="entity", label=_label(e))
    for act in core_acts:
        for e in triples[act].get("prov:used", []):
            e = e.strip(" ,;.")
            if e in G:
                G.add_edge(e, act, rel="used")
    for ent in entities:
        for ga in triples[ent].get("prov:wasGeneratedBy", []):
            if ga.strip(" ,;.") in core_acts and ent in G:
                G.add_edge(ga.strip(" ,;."), ent, rel="generated")

    G.remove_nodes_from([n for n in G.nodes if G.degree(n) == 0])

    act_nodes = [n for n in G.nodes if G.nodes[n]["kind"] == "activity"]
    ent_nodes = [n for n in G.nodes if G.nodes[n]["kind"] == "entity"]

    if G.number_of_nodes() == 0:
        fig, ax = plt.subplots(figsize=(7, 2.5))
        ax.text(0.5, 0.5, "SensorWF PROV-O Runtime Provenance Trace\n"
                "PROV-O/ProvONE Turtle: activities, entities, used/wasGeneratedBy",
                ha="center", va="center", fontsize=10, transform=ax.transAxes,
                bbox=dict(boxstyle="round", fc="#FFF9C4", ec="#aaaaaa"))
        ax.axis("off")
        _save(fig, out_path, symlink_doc=False)
        return

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = {}
        for i, n in enumerate(act_nodes):
            pos[n] = ((i + 1) * 3.5, 2.0)
        for i, n in enumerate(ent_nodes):
            pos[n] = ((i + 1) * 2.0, 0.0)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_axis_off()
    ACT_COLOR, ENT_COLOR = "#1565C0", "#E65100"
    nx.draw_networkx_nodes(G, pos, nodelist=act_nodes, node_color=ACT_COLOR,
                           node_shape="s", node_size=800, ax=ax, alpha=0.9)
    nx.draw_networkx_nodes(G, pos, nodelist=ent_nodes, node_color=ENT_COLOR,
                           node_shape="o", node_size=450, ax=ax, alpha=0.9)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=12,
                           edge_color="#90A4AE", width=1.2,
                           connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]["label"] for n in act_nodes},
                            font_size=6.5, ax=ax, font_color="white", font_weight="bold")
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]["label"] for n in ent_nodes},
                            font_size=6, ax=ax, font_color="#212121")
    ax.legend(handles=[mpatches.Patch(color=ACT_COLOR, label="prov:Activity"),
                        mpatches.Patch(color=ENT_COLOR, label="prov:Entity")],
              loc="upper right", fontsize=8.5, frameon=True)

    _save(fig, out_path, symlink_doc=False)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def generate_all(only: str | None = None) -> None:
    REGISTRY = {
        "domain":     fig_domain_comparison,
        "kg":         fig_kg_subgraph,
        "ablation":   fig_ablation_window,
        "crossfamily":fig_cross_family,
        "provenance": fig_provenance,
    }
    targets = {only: REGISTRY[only]} if only and only in REGISTRY else REGISTRY
    for name, fn in targets.items():
        print(f"\n── {name} ──")
        try:
            fn()
        except Exception as exc:
            print(f"  ERROR: {exc}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate SensorWF publication figures")
    ap.add_argument("--only", choices=["domain","kg","ablation","crossfamily","provenance"],
                    help="Generate only this figure")
    args = ap.parse_args()
    generate_all(only=args.only)
