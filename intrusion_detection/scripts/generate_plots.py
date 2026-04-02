#!/usr/bin/env python3
"""
generate_plots.py
-----------------
Comprehensive visualization suite for the Satellite Intrusion Detection System paper.

Generates the following figures into results/plots/:
  01_model_comparison.png         - Phase 1: All 5 models across 4 metrics
  02_tree_structure_phase2.png    - Decision tree structure (UNSW-NB15)
  03_tree_structure_can.png       - Decision tree structure (CAN Bus)
  04_confusion_matrices.png       - Confusion matrices for Phase 2, 2b, 4
  05_threshold_sweep.png          - Precision / Recall / F1 vs threshold
  06_quantization_tradeoffs.png   - Float32 vs Int16 vs Int8 comparison
    07_benchmark_can_comparison.png - UNSW->CAN vs NSL->CAN overall comparison
  08_cross_dataset_heatmap.png    - Performance heatmap across datasets
  09_feature_importance.png       - Feature importances (Phase 2 + CAN)
  10_roc_curves.png               - ROC curves (Phase 2, 2b, Phase 4)
"""

import sys, json
sys.path.insert(0, '/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib

from sklearn.tree import plot_tree
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE   = Path('/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection')
RES    = BASE / 'results'
MODELS = BASE / 'models' / 'trained_models'
DATA   = BASE / 'datasets'
PLOTS  = RES / 'plots'
PLOTS.mkdir(exist_ok=True)

# ─── Style ───────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', font_scale=1.15)
PALETTE   = sns.color_palette('deep')
HIGHLIGHT = '#e74c3c'   # red accent for recommended model

LIGHT_BLUE   = PALETTE[0]
ORANGE       = PALETTE[1]
GREEN        = PALETTE[2]
RED          = PALETTE[3]
PURPLE       = PALETTE[4]
BROWN        = PALETTE[5]

MODEL_COLORS = {
    'LightRandomForest': LIGHT_BLUE,
    'CompactExtraTrees': ORANGE,
    'MicroXGBoost':      GREEN,
    'TinyDecisionTree':  RED,
    'TinyXGBoost':       PURPLE,
}

def save(fig, name):
    path = PLOTS / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path.name}")


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 01 — Model Comparison (Phase 1, synthetic)
# ══════════════════════════════════════════════════════════════════════════════
def plot_model_comparison():
    with open(RES / 'evaluation_results.json') as f:
        data = json.load(f)

    models    = [d['model'] for d in data]
    acc       = [d['test_acc'] * 100 for d in data]
    f1        = [d['f1'] * 100 for d in data]
    size_kb   = [d['model_size_kb'] for d in data]
    inf_ms    = [d['inference_time_ms'] for d in data]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Phase 1 – Model Comparison (Synthetic Benchmark)', fontsize=14, fontweight='bold')

    short_names = [m.replace('Light','L.').replace('Compact','C.').replace('Tiny','T.').replace('Micro','M.') for m in models]
    colors = [MODEL_COLORS[m] for m in models]

    def _bar(ax, vals, title, ylabel, fmt='{:.1f}', annotate=True):
        bars = ax.bar(short_names, vals, color=colors, edgecolor='white', linewidth=0.8)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=20)
        if annotate:
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                        fmt.format(v), ha='center', va='bottom', fontsize=9)
        # highlight TinyDecisionTree
        idx = models.index('TinyDecisionTree')
        bars[idx].set_edgecolor(HIGHLIGHT)
        bars[idx].set_linewidth(2.5)
        return ax

    _bar(axes[0,0], acc,     'Test Accuracy',       'Accuracy (%)',       '{:.2f}')
    _bar(axes[0,1], f1,      'F1 Score',             'F1 (%)',             '{:.2f}')
    _bar(axes[1,0], size_kb, 'Model Size',           'Size (KB)',          '{:.2f}')
    _bar(axes[1,1], inf_ms,  'Inference Time',       'Time (ms)',          '{:.4f}')

    legend_patch = mpatches.Patch(facecolor='white', edgecolor=HIGHLIGHT, linewidth=2.5, label='TinyDecisionTree (recommended)')
    fig.legend(handles=[legend_patch], loc='lower center', ncol=1, frameon=True, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save(fig, '01_model_comparison.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 02 & 03 — Decision Tree Structures
# ══════════════════════════════════════════════════════════════════════════════
def plot_tree_structure(model_path, feature_names, title, out_name, figsize=(22, 10)):
    clf = joblib.load(model_path)

    fig, ax = plt.subplots(figsize=figsize)
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=['Normal', 'Attack'],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
        impurity=True,
        proportion=False,
        precision=3,
    )
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    save(fig, out_name)


def plot_trees():
    # Phase 2 – UNSW-NB15
    with open(MODELS / 'phase2_feature_names.json') as f:
        feat2 = json.load(f)
    plot_tree_structure(
        MODELS / 'phase2_decision_tree.joblib',
        feat2,
        'TinyDecisionTree – UNSW-NB15 Network IDS (depth=5, 26 leaves)',
        '02_tree_structure_phase2.png',
        figsize=(24, 11),
    )

    # Phase 4B – CAN from benchmark (prefer NSL model if available)
    model_path = MODELS / 'phase4b_nsl_can_decision_tree.joblib'
    feat_path = MODELS / 'phase4b_nsl_can_feature_names.json'
    if not model_path.exists() or not feat_path.exists():
        model_path = MODELS / 'can_decision_tree.joblib'
        feat_path = MODELS / 'can_feature_names.json'

    with open(feat_path) as f:
        feat_can = json.load(f)
    plot_tree_structure(
        model_path,
        feat_can,
        'TinyDecisionTree – Satellite CAN IDS (benchmark-to-CAN)',
        '03_tree_structure_can.png',
        figsize=(22, 11),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Fig 04 — Confusion Matrices
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrices():
    datasets = [
        ('UNSW-NB15',  'phase2_results.json',         'baseline_threshold'),
        ('NSL-KDD',    'phase2b_nslkdd_results.json',  'baseline_threshold'),
        ('NSL->CAN',   'phase4b_nsl_can_results.json',  'overall'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Confusion Matrices – TinyDecisionTree', fontsize=14, fontweight='bold')

    for ax, (name, fname, key) in zip(axes, datasets):
        with open(RES / fname) as f:
            d = json.load(f)
        m = d[key]
        tn, fp, fn, tp = m['tn'], m['fp'], m['fn'], m['tp']
        cm = np.array([[tn, fp], [fn, tp]])
        total = cm.sum()

        # Percentage labels
        annot = np.array([[f'{tn:,}\n({tn/total:.1%})', f'{fp:,}\n({fp/total:.1%})'],
                          [f'{fn:,}\n({fn/total:.1%})', f'{tp:,}\n({tp/total:.1%})']])

        sns.heatmap(
            cm, annot=annot, fmt='', ax=ax,
            cmap='Blues', linewidths=0.5,
            xticklabels=['Predicted\nNormal', 'Predicted\nAttack'],
            yticklabels=['Actual\nNormal', 'Actual\nAttack'],
            cbar=False, annot_kws={'size': 10},
        )
        acc = (tn + tp) / total
        recall = tp / (tp + fn)
        ax.set_title(f'{name}\nAcc={acc:.2%}  Recall={recall:.2%}', fontweight='bold', fontsize=11)

    fig.tight_layout()
    save(fig, '04_confusion_matrices.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 05 — Threshold Sweep
# ══════════════════════════════════════════════════════════════════════════════
def plot_threshold_sweep():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Threshold Sweep – Precision / Recall / F1', fontsize=14, fontweight='bold')

    sweeps = [
        ('UNSW-NB15', 'phase2_results.json',        axes[0], 0.5,  '#2196F3'),
        ('NSL-KDD',   'phase2b_nslkdd_results.json', axes[1], 0.5,  '#4CAF50'),
    ]

    for name, fname, ax, baseline_t, color in sweeps:
        with open(RES / fname) as f:
            d = json.load(f)
        sweep = d['threshold_sweep']
        ts  = [s['threshold'] for s in sweep]
        pre = [s['precision'] for s in sweep]
        rec = [s['recall']    for s in sweep]
        f1v = [s['f1']        for s in sweep]
        fpr = [s['fpr']       for s in sweep]

        ax.plot(ts, rec, label='Recall',    color='#e74c3c', lw=2)
        ax.plot(ts, pre, label='Precision', color='#3498db', lw=2)
        ax.plot(ts, f1v, label='F1',        color='#2ecc71', lw=2, linestyle='--')
        ax.plot(ts, fpr, label='FPR',       color='#e67e22', lw=1.5, linestyle=':')

        ax.axvline(baseline_t, color='gray', linestyle='--', alpha=0.7, label=f'Threshold={baseline_t}')
        ax.set_xlabel('Decision Threshold')
        ax.set_ylabel('Score')
        ax.set_title(f'{name}', fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)

    fig.tight_layout()
    save(fig, '05_threshold_sweep.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 06 — Quantization Trade-offs
# ══════════════════════════════════════════════════════════════════════════════
def plot_quantization():
    with open(RES / 'phase3_results.json') as f:
        d = json.load(f)
    q = d['quantization_results']

    precisions = ['float32', 'int16', 'int8']
    acc  = [q[p]['accuracy']    * 100 for p in precisions]
    rec  = [q[p]['recall']      * 100 for p in precisions]
    f1v  = [q[p]['f1']          * 100 for p in precisions]
    fpr  = [q[p]['fpr']         * 100 for p in precisions]
    size = [q[p]['size_kb']           for p in precisions]
    inf  = [q[p]['inference_us']      for p in precisions]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Phase 3 – Quantization Trade-offs (UNSW-NB15)', fontsize=14, fontweight='bold')

    q_colors = ['#2196F3', '#FF9800', '#F44336']

    # Panel 1: Grouped metric bars
    x = np.arange(len(precisions))
    w = 0.2
    ax = axes[0]
    ax.bar(x - 1.5*w, acc, w, label='Accuracy',  color='#3498db')
    ax.bar(x - 0.5*w, rec, w, label='Recall',    color='#e74c3c')
    ax.bar(x + 0.5*w, f1v, w, label='F1',        color='#2ecc71')
    ax.bar(x + 1.5*w, fpr, w, label='FPR',       color='#e67e22')
    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in precisions])
    ax.set_ylabel('Score (%)')
    ax.set_title('Metrics by Precision', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 115)
    for i, (a, r, f, fp_) in enumerate(zip(acc, rec, f1v, fpr)):
        for j, (v, offset) in enumerate(zip([a, r, f, fp_], [-1.5, -0.5, 0.5, 1.5])):
            ax.text(i + offset*w, v + 1, f'{v:.0f}', ha='center', fontsize=7)

    # Panel 2: Model size
    axes[1].bar([p.upper() for p in precisions], size, color=q_colors, edgecolor='white')
    axes[1].set_ylabel('Size (KB)')
    axes[1].set_title('Model Size', fontweight='bold')
    for i, v in enumerate(size):
        axes[1].text(i, v + 0.01, f'{v:.3f} KB', ha='center', fontsize=10, fontweight='bold')

    # Panel 3: Inference time
    axes[2].bar([p.upper() for p in precisions], inf, color=q_colors, edgecolor='white')
    axes[2].set_ylabel('Inference Time (µs)')
    axes[2].set_title('Inference Latency', fontweight='bold')
    for i, v in enumerate(inf):
        axes[2].text(i, v + 0.05, f'{v:.2f} µs', ha='center', fontsize=10, fontweight='bold')

    fig.tight_layout()
    save(fig, '06_quantization_tradeoffs.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 07 — Benchmark-to-CAN Comparison
# ══════════════════════════════════════════════════════════════════════════════
def plot_benchmark_can_comparison():
    unsw = _load_json(RES / 'phase4b_unsw_can_results.json')
    nsl = _load_json(RES / 'phase4b_nsl_can_results.json')

    labels = ['UNSW->CAN', 'NSL->CAN']
    metrics = {
        'Accuracy': [unsw['overall']['accuracy'] * 100, nsl['overall']['accuracy'] * 100],
        'Recall': [unsw['overall']['recall'] * 100, nsl['overall']['recall'] * 100],
        'Precision': [unsw['overall']['precision'] * 100, nsl['overall']['precision'] * 100],
        'F1': [unsw['overall']['f1'] * 100, nsl['overall']['f1'] * 100],
        'ROC-AUC': [unsw['overall']['roc_auc'] * 100, nsl['overall']['roc_auc'] * 100],
        'Low FPR': [(1 - unsw['overall']['fpr']) * 100, (1 - nsl['overall']['fpr']) * 100],
    }
    model_kb = [unsw['model_size_kb'], nsl['model_size_kb']]
    inf_ms = [unsw['inference_time_ms'], nsl['inference_time_ms']]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle('Phase 4B – Benchmark to CAN IDS Comparison', fontsize=14, fontweight='bold')

    x = np.arange(len(labels))
    width = 0.12
    metric_names = list(metrics.keys())
    colors = sns.color_palette('Set2', len(metric_names))
    for i, (mname, vals) in enumerate(metrics.items()):
        axes[0].bar(x + (i - (len(metric_names)-1)/2) * width, vals, width, label=mname, color=colors[i])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel('Score (%)')
    axes[0].set_ylim(0, 105)
    axes[0].set_title('Detection Metrics', fontweight='bold')
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].bar(np.array(x) - 0.15, model_kb, 0.3, label='Model size (KB)', color='#5DA5DA')
    axes[1].set_ylabel('Model Size (KB)', color='#2c3e50')
    axes[1].tick_params(axis='y', labelcolor='#2c3e50')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)

    ax2 = axes[1].twinx()
    ax2.bar(np.array(x) + 0.15, inf_ms, 0.3, label='Inference (ms)', color='#FAA43A')
    ax2.set_ylabel('Inference Time (ms)', color='#8e44ad')
    ax2.tick_params(axis='y', labelcolor='#8e44ad')
    axes[1].set_title('Deployment Cost', fontweight='bold')

    for i, v in enumerate(model_kb):
        axes[1].text(i - 0.15, v + 0.03, f'{v:.2f}', ha='center', fontsize=9)
    for i, v in enumerate(inf_ms):
        ax2.text(i + 0.15, v + 0.001, f'{v:.4f}', ha='center', fontsize=9)

    fig.tight_layout()
    save(fig, '07_benchmark_can_comparison.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 08 — Cross-Dataset Performance Heatmap
# ══════════════════════════════════════════════════════════════════════════════
def plot_cross_dataset_heatmap():
    p2 = _load_json(RES / 'phase2_results.json')['baseline_threshold']
    p2b = _load_json(RES / 'phase2b_nslkdd_results.json')['baseline_threshold']
    ucan = _load_json(RES / 'phase4b_unsw_can_results.json')['overall']
    ncan = _load_json(RES / 'phase4b_nsl_can_results.json')['overall']

    rows = {
        'UNSW-NB15\n(Network)': {
            'Accuracy': p2['acc'] * 100,
            'Precision': p2['precision'] * 100,
            'Recall': p2['recall'] * 100,
            'F1': p2['f1'] * 100,
            'FPR': p2['fpr'] * 100,
            'ROC-AUC': _load_json(RES / 'phase2_results.json')['roc_auc'] * 100,
        },
        'NSL-KDD\n(Network)': {
            'Accuracy': p2b['acc'] * 100,
            'Precision': p2b['precision'] * 100,
            'Recall': p2b['recall'] * 100,
            'F1': p2b['f1'] * 100,
            'FPR': p2b['fpr'] * 100,
            'ROC-AUC': _load_json(RES / 'phase2b_nslkdd_results.json')['roc_auc'] * 100,
        },
        'UNSW->CAN\n(Benchmark)': {
            'Accuracy': ucan['accuracy'] * 100,
            'Precision': ucan['precision'] * 100,
            'Recall': ucan['recall'] * 100,
            'F1': ucan['f1'] * 100,
            'FPR': ucan['fpr'] * 100,
            'ROC-AUC': ucan['roc_auc'] * 100,
        },
        'NSL->CAN\n(Benchmark)': {
            'Accuracy': ncan['accuracy'] * 100,
            'Precision': ncan['precision'] * 100,
            'Recall': ncan['recall'] * 100,
            'F1': ncan['f1'] * 100,
            'FPR': ncan['fpr'] * 100,
            'ROC-AUC': ncan['roc_auc'] * 100,
        },
    }
    df = pd.DataFrame(rows).T
    metric_order = ['ROC-AUC', 'Accuracy', 'F1', 'Recall', 'Precision', 'FPR']
    df = df[metric_order]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.heatmap(
        df, annot=True, fmt='.2f', cmap='RdYlGn', linewidths=0.5,
        vmin=0, vmax=100, ax=ax, cbar_kws={'label': 'Score (%)'},
        annot_kws={'size': 12, 'weight': 'bold'},
    )
    ax.set_title('TinyDecisionTree – Cross-Dataset Performance Summary (%)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xticklabels(metric_order, fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10, rotation=0)
    ax.set_xlabel('')

    fig.tight_layout()
    save(fig, '08_cross_dataset_heatmap.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 09 — Feature Importances
# ══════════════════════════════════════════════════════════════════════════════
def plot_feature_importances():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Feature Importances – TinyDecisionTree (Gini)', fontsize=14, fontweight='bold')

    configs = [
        (MODELS / 'phase2_decision_tree.joblib', MODELS / 'phase2_feature_names.json',
         'UNSW-NB15 (34 features)', axes[0], '#3498db'),
        (MODELS / 'can_decision_tree.joblib',    MODELS / 'can_feature_names.json',
         'CAN Bus IDS (15 features)', axes[1], '#e74c3c'),
    ]

    for model_path, feat_path, title, ax, color in configs:
        clf = joblib.load(model_path)
        with open(feat_path) as f:
            feat_names = json.load(f)

        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        # Show top-N (all for CAN, top 15 for UNSW)
        top_n = min(15, len(feat_names))
        top_idx = indices[:top_n]

        feat_labels = [feat_names[i] for i in top_idx]
        feat_vals   = importances[top_idx]

        # Horizontal bar chart
        y_pos = np.arange(top_n)
        palette_grad = sns.color_palette('Blues_r', top_n) if color == '#3498db' else sns.color_palette('Reds_r', top_n)
        ax.barh(y_pos, feat_vals[::-1], color=palette_grad, edgecolor='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feat_labels[::-1], fontsize=9)
        ax.set_xlabel('Gini Importance')
        ax.set_title(title, fontweight='bold')
        ax.set_xlim(0, max(feat_vals) * 1.2)
        for i, v in enumerate(feat_vals[::-1]):
            ax.text(v + max(feat_vals)*0.01, i, f'{v:.3f}', va='center', fontsize=8)

    fig.tight_layout()
    save(fig, '09_feature_importance.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 10 — ROC Curves (load models + datasets)
# ══════════════════════════════════════════════════════════════════════════════
def _load_unswnb15_test():
    """Load and preprocess UNSW-NB15 test set."""
    train_df = pd.read_parquet(DATA / 'UNSW_NB15_training-set.parquet')
    test_df  = pd.read_parquet(DATA / 'UNSW_NB15_testing-set.parquet')

    combined  = pd.concat([train_df, test_df], ignore_index=True)
    split_idx = len(train_df)

    label_col = 'label' if 'label' in combined.columns else 'Attack'
    y = combined[label_col].values

    X = combined.drop(columns=[label_col])
    if 'attack_cat' in X.columns:
        X = X.drop(columns=['attack_cat'])
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0).astype(np.float32)

    scaler = joblib.load(MODELS / 'phase2_scaler.joblib')
    X_scaled = scaler.transform(X.values)
    y = y.astype(int)

    return X_scaled[split_idx:], y[split_idx:]


def _load_nslkdd_test():
    """Load and preprocess NSL-KDD test set."""
    nsl_dir = DATA / 'NSL-KDD'
    train_file = nsl_dir / 'KDDTrain+.txt'
    test_file  = nsl_dir / 'KDDTest+.txt'

    col_names_file = nsl_dir / 'Field Names.csv'
    if col_names_file.exists():
        cols_df = pd.read_csv(col_names_file, header=None)
        cols = cols_df.iloc[:, 0].tolist()
    else:
        # fallback: use standard KDD99 column names
        cols = [
            'duration','protocol_type','service','flag','src_bytes','dst_bytes',
            'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
            'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
            'num_shells','num_access_files','num_outbound_cmds','is_host_login',
            'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
            'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
            'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
            'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
            'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
        ]

    train_df = pd.read_csv(train_file, header=None, names=cols)
    test_df  = pd.read_csv(test_file,  header=None, names=cols)

    for df in [train_df, test_df]:
        if 'difficulty' in df.columns:
            df.drop(columns=['difficulty'], inplace=True)

    combined  = pd.concat([train_df, test_df], ignore_index=True)
    split_idx = len(train_df)

    label_col = 'label'
    combined['binary_label'] = (combined[label_col] != 'normal').astype(int)
    y = combined['binary_label'].values

    X = combined.drop(columns=[label_col, 'binary_label'])
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    X = X.fillna(0).astype(np.float32)

    scaler = joblib.load(MODELS / 'phase2b_scaler.joblib')
    X_scaled = scaler.transform(X.values)

    return X_scaled[split_idx:], y[split_idx:]


def _load_can_test_phase4b(dataset: str):
    """Load benchmark-to-CAN feature test set for phase 4b."""
    test_df = pd.read_csv(DATA / 'CAN_FROM_BENCHMARK' / f'{dataset}_can_test_features.csv')
    y = test_df['label'].astype(int).values
    with open(MODELS / f'phase4b_{dataset}_can_feature_names.json') as f:
        feat_names = json.load(f)
    X = test_df[feat_names].values.astype(np.float32)
    scaler = joblib.load(MODELS / f'phase4b_{dataset}_can_scaler.joblib')
    X_scaled = scaler.transform(X)
    return X_scaled, y


def plot_roc_curves():
    print("  Loading datasets for ROC curves (this may take a moment)...")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title('ROC Curves – TinyDecisionTree Across Datasets', fontsize=13, fontweight='bold')

    datasets = [
        ('UNSW-NB15\n(Network, Real)',  MODELS / 'phase2_decision_tree.joblib',  _load_unswnb15_test, '#2196F3'),
        ('NSL-KDD\n(Network, Novel)',   MODELS / 'phase2b_decision_tree.joblib', _load_nslkdd_test,   '#4CAF50'),
        ('UNSW->CAN\n(Benchmark)',      MODELS / 'phase4b_unsw_can_decision_tree.joblib', lambda: _load_can_test_phase4b('unsw'), '#F44336'),
        ('NSL->CAN\n(Benchmark)',       MODELS / 'phase4b_nsl_can_decision_tree.joblib',  lambda: _load_can_test_phase4b('nsl'),  '#FF9800'),
    ]

    for label, model_path, loader_fn, color in datasets:
        try:
            clf    = joblib.load(model_path)
            X_test, y_test = loader_fn()
            y_prob = clf.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2.2,
                    label=f'{label}  (AUC={roc_auc:.4f})')
        except Exception as e:
            print(f"    Warning: could not load {label}: {e}")

    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='Random classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, '10_roc_curves.png')


# ══════════════════════════════════════════════════════════════════════════════
# Fig 11 — Dataset & Model Overview Summary (spider / radar chart)
# ══════════════════════════════════════════════════════════════════════════════
def plot_radar_summary():
    """Radar chart comparing the three deployment scenarios."""
    categories = ['Recall', 'Precision', 'F1', 'Accuracy', 'Low FPR', 'ROC-AUC']

    p2 = _load_json(RES / 'phase2_results.json')
    p2b = _load_json(RES / 'phase2b_nslkdd_results.json')
    ucan = _load_json(RES / 'phase4b_unsw_can_results.json')
    ncan = _load_json(RES / 'phase4b_nsl_can_results.json')

    # Normalise FPR to "Low FPR" = 1 - FPR (higher is better)
    data_raw = {
        'UNSW-NB15': [
            p2['baseline_threshold']['recall'] * 100,
            p2['baseline_threshold']['precision'] * 100,
            p2['baseline_threshold']['f1'] * 100,
            p2['baseline_threshold']['acc'] * 100,
            (1 - p2['baseline_threshold']['fpr']) * 100,
            p2['roc_auc'] * 100,
        ],
        'NSL-KDD': [
            p2b['baseline_threshold']['recall'] * 100,
            p2b['baseline_threshold']['precision'] * 100,
            p2b['baseline_threshold']['f1'] * 100,
            p2b['baseline_threshold']['acc'] * 100,
            (1 - p2b['baseline_threshold']['fpr']) * 100,
            p2b['roc_auc'] * 100,
        ],
        'UNSW->CAN': [
            ucan['overall']['recall'] * 100,
            ucan['overall']['precision'] * 100,
            ucan['overall']['f1'] * 100,
            ucan['overall']['accuracy'] * 100,
            (1 - ucan['overall']['fpr']) * 100,
            ucan['overall']['roc_auc'] * 100,
        ],
        'NSL->CAN': [
            ncan['overall']['recall'] * 100,
            ncan['overall']['precision'] * 100,
            ncan['overall']['f1'] * 100,
            ncan['overall']['accuracy'] * 100,
            (1 - ncan['overall']['fpr']) * 100,
            ncan['overall']['roc_auc'] * 100,
        ],
    }

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors_r = ['#2196F3', '#4CAF50', '#F44336', '#FF9800']

    for (name, vals), color in zip(data_raw.items(), colors_r):
        vals_norm = [v / 100 for v in vals]
        vals_norm += vals_norm[:1]
        ax.plot(angles, vals_norm, color=color, lw=2.2, label=name)
        ax.fill(angles, vals_norm, color=color, alpha=0.12)

    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)
    ax.set_title('TinyDecisionTree – Performance Radar\n(across all deployment scenarios)',
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=10)

    fig.tight_layout()
    save(fig, '11_radar_summary.png')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f"\nGenerating plots → {PLOTS}\n")

    steps = [
        ('01 Model comparison',        plot_model_comparison),
        ('02+03 Tree structures',      plot_trees),
        ('04 Confusion matrices',      plot_confusion_matrices),
        ('05 Threshold sweep',         plot_threshold_sweep),
        ('06 Quantization trade-offs', plot_quantization),
        ('07 Benchmark-CAN compare',   plot_benchmark_can_comparison),
        ('08 Cross-dataset heatmap',   plot_cross_dataset_heatmap),
        ('09 Feature importances',     plot_feature_importances),
        ('10 ROC curves',              plot_roc_curves),
        ('11 Radar summary',           plot_radar_summary),
    ]

    for name, fn in steps:
        print(f"[{name}]")
        try:
            fn()
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    print(f"\nDone. All plots saved to {PLOTS}")
