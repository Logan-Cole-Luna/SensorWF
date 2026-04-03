#!/usr/bin/env python3
"""
benchmark_all_models.py
=======================
Train and evaluate all five candidate lightweight IDS models on:
  - UNSW-NB15  (175K train / 82K test, 34 network features)
  - NSL-KDD    (125K train / 22K test, 41 features, 17 novel attack types)

Outputs:
  results/all_models_unsw_results.json
  results/all_models_nslkdd_results.json
  results/all_models_comparison.json   ← combined table used in the paper
  results/plots/12_all_models_realworld.png
"""

import sys, time, json, pickle
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
from xgboost import XGBClassifier

BASE       = Path('/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection')
DATASETS   = BASE / 'datasets'
RESULTS    = BASE / 'results'
PLOTS      = RESULTS / 'plots'
RESULTS.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)

# ─── Model catalogue ──────────────────────────────────────────────────────────

def make_models():
    return {
        'TinyDecisionTree': DecisionTreeClassifier(
            max_depth=5, min_samples_split=5, min_samples_leaf=2, random_state=42),
        'TinyXGBoost': XGBClassifier(
            n_estimators=3, max_depth=2, learning_rate=0.5,
            reg_alpha=1.0, reg_lambda=1.0,
            objective='binary:logistic', random_state=42,
            tree_method='hist', device='cpu', verbosity=0),
        'MicroXGBoost': XGBClassifier(
            n_estimators=5, max_depth=3, learning_rate=0.3,
            subsample=0.8, colsample_bytree=0.8,
            objective='binary:logistic', random_state=42,
            tree_method='hist', device='cpu', verbosity=0),
        'LightRandomForest': RandomForestClassifier(
            n_estimators=10, max_depth=7, min_samples_split=5,
            n_jobs=-1, random_state=42),
        'CompactExtraTrees': ExtraTreesClassifier(
            n_estimators=8, max_depth=6, min_samples_split=5,
            n_jobs=-1, random_state=42),
    }

MODEL_ORDER = [
    'TinyDecisionTree', 'TinyXGBoost', 'MicroXGBoost',
    'LightRandomForest', 'CompactExtraTrees',
]

# ─── Data loaders ─────────────────────────────────────────────────────────────

def load_unsw():
    print("  Loading UNSW-NB15 parquet files...")
    train_df = pd.read_parquet(DATASETS / 'UNSW_NB15_training-set.parquet')
    test_df  = pd.read_parquet(DATASETS / 'UNSW_NB15_testing-set.parquet')
    combined  = pd.concat([train_df, test_df], ignore_index=True)
    split_idx = len(train_df)

    label_col = 'label' if 'label' in combined.columns else 'Attack'
    y = combined[label_col].values.astype(int)
    X = combined.drop(columns=[label_col])
    if 'attack_cat' in X.columns:
        X = X.drop(columns=['attack_cat'])
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values[:split_idx])
    Xs_test = scaler.transform(X.values[split_idx:])

    print(f"  Train: {split_idx:,}  Test: {len(test_df):,}  Features: {X.shape[1]}")
    return Xs, Xs_test, y[:split_idx], y[split_idx:]


def load_nsl():
    print("  Loading NSL-KDD...")
    cols = [
        'duration','protocol_type','service','flag','src_bytes','dst_bytes',
        'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
        'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
        'num_shells','num_access_files','num_outbound_cmds','is_host_login',
        'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
        'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
        'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
        'dst_host_same_srv_rate','dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
        'dst_host_serror_rate','dst_host_srv_serror_rate',
        'dst_host_rerror_rate','dst_host_srv_rerror_rate',
        'label','difficulty',
    ]
    train_df = pd.read_csv(DATASETS / 'NSL-KDD' / 'KDDTrain+.txt', header=None, names=cols)
    test_df  = pd.read_csv(DATASETS / 'NSL-KDD' / 'KDDTest+.txt',  header=None, names=cols)

    for df in [train_df, test_df]:
        df.drop(columns=['difficulty'], inplace=True)

    combined  = pd.concat([train_df, test_df], ignore_index=True)
    split_idx = len(train_df)

    combined['_y'] = (combined['label'] != 'normal').astype(int)
    y = combined['_y'].values
    X = combined.drop(columns=['label','_y'])
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)

    scaler = StandardScaler()
    Xs      = scaler.fit_transform(X.values[:split_idx])
    Xs_test = scaler.transform(X.values[split_idx:])

    print(f"  Train: {split_idx:,}  Test: {len(test_df):,}  Features: {X.shape[1]}")
    return Xs, Xs_test, y[:split_idx], y[split_idx:]


# ─── Evaluation helper ────────────────────────────────────────────────────────

def measure_inf_us(clf, X, n=2000):
    s = X[:1]
    t0 = time.perf_counter()
    for _ in range(n):
        clf.predict(s)
    return (time.perf_counter() - t0) / n * 1e6


def evaluate(name, clf, X_train, X_test, y_train, y_test):
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_s = time.perf_counter() - t0

    y_pred = clf.predict(X_test)
    try:
        y_prob = clf.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        auc = float('nan')

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0

    size_kb = len(pickle.dumps(clf)) / 1024
    inf_us  = measure_inf_us(clf, X_test)

    return {
        'model':      name,
        'accuracy':   round(float(accuracy_score(y_test, y_pred)), 4),
        'precision':  round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        'recall':     round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        'f1':         round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        'fpr':        round(float(fpr), 4),
        'roc_auc':    round(auc, 4),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'train_time_s': round(train_s, 4),
        'inf_us':     round(inf_us, 2),
        'size_kb':    round(size_kb, 3),
    }


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_comparison(unsw_results, nsl_results):
    sns.set_theme(style='whitegrid', font_scale=1.1)
    palette = sns.color_palette('deep', 5)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle('All Candidate Models — Real-World Dataset Benchmark', fontsize=14, fontweight='bold')

    metrics = [
        ('accuracy', 'Accuracy', '%'),
        ('recall',   'Recall',   '%'),
        ('f1',       'F1',       '%'),
        ('fpr',      'FPR',      '%'),
    ]

    for col, (metric, label, unit) in enumerate(metrics):
        for row, (dataset_name, results) in enumerate([('UNSW-NB15', unsw_results), ('NSL-KDD', nsl_results)]):
            ax = axes[row][col]
            names = [r['model'].replace('Light','L.').replace('Compact','C.')
                               .replace('Micro','M.').replace('Tiny','T.') for r in results]
            vals  = [r[metric] * 100 for r in results]
            colors = [palette[i] for i in range(len(results))]

            bars = ax.bar(names, vals, color=colors, edgecolor='white', linewidth=0.8)
            ax.set_title(f'{dataset_name} — {label}', fontweight='bold', fontsize=10)
            ax.set_ylabel(f'{label} ({unit})')
            ax.tick_params(axis='x', rotation=25, labelsize=8)
            ax.set_ylim(0, 110)

            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{v:.1f}', ha='center', va='bottom', fontsize=7)

            # Highlight TinyDecisionTree
            idx = [r['model'] for r in results].index('TinyDecisionTree')
            bars[idx].set_edgecolor('#e74c3c')
            bars[idx].set_linewidth(2.5)

    fig.tight_layout()
    out = PLOTS / '12_all_models_realworld.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_dataset(name, X_train, X_test, y_train, y_test):
    print(f"\n{'='*60}")
    print(f"  Dataset: {name}  train={len(X_train):,}  test={len(X_test):,}")
    print(f"{'='*60}")
    results = []
    for mname in MODEL_ORDER:
        clf = make_models()[mname]
        print(f"  [{mname}]", end='', flush=True)
        r = evaluate(mname, clf, X_train, X_test, y_train, y_test)
        results.append(r)
        print(f"  acc={r['accuracy']:.4f}  rec={r['recall']:.4f}  "
              f"f1={r['f1']:.4f}  fpr={r['fpr']:.4f}  "
              f"size={r['size_kb']:.1f}KB  inf={r['inf_us']:.1f}µs  "
              f"auc={r['roc_auc']:.4f}")
    return results


if __name__ == '__main__':
    # ── UNSW-NB15 ─────────────────────────────────────────────────────────
    print("\n[1/2] UNSW-NB15")
    X_tr, X_te, y_tr, y_te = load_unsw()
    unsw_results = run_dataset('UNSW-NB15', X_tr, X_te, y_tr, y_te)
    with open(RESULTS / 'all_models_unsw_results.json', 'w') as f:
        json.dump(unsw_results, f, indent=2)

    # ── NSL-KDD ───────────────────────────────────────────────────────────
    print("\n[2/2] NSL-KDD")
    X_tr, X_te, y_tr, y_te = load_nsl()
    nsl_results = run_dataset('NSL-KDD', X_tr, X_te, y_tr, y_te)
    with open(RESULTS / 'all_models_nslkdd_results.json', 'w') as f:
        json.dump(nsl_results, f, indent=2)

    # ── Combined summary ──────────────────────────────────────────────────
    combined = {'UNSW-NB15': unsw_results, 'NSL-KDD': nsl_results}
    with open(RESULTS / 'all_models_comparison.json', 'w') as f:
        json.dump(combined, f, indent=2)

    # ── Plot ──────────────────────────────────────────────────────────────
    print("\nGenerating comparison plot...")
    plot_comparison(unsw_results, nsl_results)

    # ── Pretty-print summary table ────────────────────────────────────────
    print("\n" + "="*90)
    print(f"{'Model':<22} {'Dataset':<12} {'Acc':>6} {'Rec':>6} {'F1':>6} "
          f"{'FPR':>6} {'AUC':>6} {'KB':>6} {'µs':>7}")
    print("-"*90)
    for ds_name, results in [('UNSW-NB15', unsw_results), ('NSL-KDD', nsl_results)]:
        for r in results:
            print(f"{r['model']:<22} {ds_name:<12} "
                  f"{r['accuracy']*100:>5.1f}% {r['recall']*100:>5.1f}% "
                  f"{r['f1']*100:>5.1f}% {r['fpr']*100:>5.1f}% "
                  f"{r['roc_auc']:>6.4f} {r['size_kb']:>5.1f} {r['inf_us']:>6.1f}")
        print()
    print("="*90)
    print(f"\nResults saved to {RESULTS}/all_models_*.json")
