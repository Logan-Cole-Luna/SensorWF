#!/usr/bin/env python3
"""
Phase 2b: TinyDecisionTree on NSL-KDD — the canonical IDS benchmark.

NSL-KDD is the most-cited IDS dataset in academic literature (~10,000+ papers).
KDDTest+ deliberately includes attack types not seen in KDDTrain+, making it
a strong generalization test — directly relevant to the "stream unknown traffic
through satellite hardware" use case.

Outputs:
  results/phase2b_nslkdd_results.json
  models/trained_models/phase2b_decision_tree.joblib
  models/trained_models/phase2b_scaler.joblib
  models/trained_models/stm32_tree_nslkdd_float32.h
  models/trained_models/stm32_tree_nslkdd_int16.h
"""

import sys
sys.path.insert(0, '/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection')

import numpy as np
import pandas as pd
import json, time, pickle, joblib
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

DATASET_DIR = Path('/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection/datasets/NSL-KDD')
RESULTS_DIR = Path('/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection/results')
MODELS_DIR  = Path('/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection/models/trained_models')
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# NSL-KDD has 41 features + label + difficulty_score
# Columns 1, 2, 3 are categorical (protocol_type, service, flag)
COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'label', 'difficulty_score'
]
CATEGORICAL = ['protocol_type', 'service', 'flag']
NORMAL_LABEL = 'normal'


def load_nslkdd():
    train_df = pd.read_csv(DATASET_DIR / 'KDDTrain+.txt', header=None, names=COLUMNS)
    test_df  = pd.read_csv(DATASET_DIR / 'KDDTest+.txt',  header=None, names=COLUMNS)
    print(f"  Train: {train_df.shape}  Test: {test_df.shape}")

    # Attack types in test but NOT in train (novel attacks — key validity point)
    train_attacks = set(train_df[train_df['label'] != NORMAL_LABEL]['label'].unique())
    test_attacks  = set(test_df[test_df['label'] != NORMAL_LABEL]['label'].unique())
    novel = test_attacks - train_attacks
    print(f"  Attack types in train: {len(train_attacks)}")
    print(f"  Attack types in test:  {len(test_attacks)}")
    print(f"  Novel attacks (test-only): {len(novel)} → {sorted(novel)}")
    return train_df, test_df, novel


def preprocess(train_df, test_df):
    # Binary label: normal=0, attack=1
    def binarize(df):
        return (df['label'] != NORMAL_LABEL).astype(int).values

    y_train = binarize(train_df)
    y_test  = binarize(test_df)

    # Combine for consistent label encoding of categoricals
    combined = pd.concat([train_df, test_df], ignore_index=True)
    n_train  = len(train_df)

    X = combined.drop(columns=['label', 'difficulty_score'])

    encoders = {}
    for col in CATEGORICAL:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    X = X.astype(np.float32)
    feature_names = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    X_train = X_scaled[:n_train]
    X_test  = X_scaled[n_train:]

    print(f"  Features: {X_scaled.shape[1]}")
    print(f"  Train attack ratio: {y_train.mean():.2%}  "
          f"Test attack ratio: {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test, scaler, feature_names


def measure_inf_ms(clf, X_test, n=1000):
    s = X_test[:1]
    t0 = time.perf_counter()
    for _ in range(n):
        clf.predict(s)
    return (time.perf_counter() - t0) / n * 1000


def evaluate_threshold(y_test, y_proba, t):
    y_pred = (y_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return dict(
        threshold=round(t, 2),
        acc=round(float(accuracy_score(y_test, y_pred)), 4),
        precision=round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        recall=round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        f1=round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        fpr=round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0, 4),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def run():
    print("=" * 70)
    print("PHASE 2b — TinyDecisionTree on NSL-KDD (canonical IDS benchmark)")
    print("=" * 70)

    # ── Load ───────────────────────────────────────────────────────────────
    print("\n[1/5] Loading NSL-KDD...")
    train_df, test_df, novel_attacks = load_nslkdd()

    # ── Preprocess ─────────────────────────────────────────────────────────
    print("\n[2/5] Preprocessing...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(train_df, test_df)

    # ── Train ──────────────────────────────────────────────────────────────
    print("\n[3/5] Training TinyDecisionTree (max_depth=5)...")
    clf = DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=2, random_state=42)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    print(f"  Training time : {train_time:.3f}s")
    print(f"  Tree depth    : {clf.get_depth()}")
    print(f"  Leaves        : {clf.get_n_leaves()}")
    print(f"  Nodes         : {clf.tree_.node_count}")
    print(f"  Pickle size   : {len(pickle.dumps(clf)) / 1024:.2f} KB")

    # ── Evaluate ───────────────────────────────────────────────────────────
    print("\n[4/5] Evaluating on KDDTest+ (includes novel attacks)...")
    y_proba  = clf.predict_proba(X_test)[:, 1]
    roc_auc  = roc_auc_score(y_test, y_proba)
    inf_ms   = measure_inf_ms(clf, X_test)
    model_kb = len(pickle.dumps(clf)) / 1024

    baseline = evaluate_threshold(y_test, y_proba, 0.5)

    print(f"\n  Default threshold (0.5):")
    print(f"    Accuracy  : {baseline['acc']:.4f}")
    print(f"    Precision : {baseline['precision']:.4f}")
    print(f"    Recall    : {baseline['recall']:.4f}")
    print(f"    F1-Score  : {baseline['f1']:.4f}")
    print(f"    FPR       : {baseline['fpr']:.4f}")
    print(f"    ROC-AUC   : {roc_auc:.4f}")
    print(f"    Model size: {model_kb:.2f} KB")
    print(f"    Inference : {inf_ms:.4f} ms")

    print(f"\n  Threshold sweep:")
    print(f"  {'Threshold':>10}  {'Acc':>7}  {'Prec':>7}  {'Recall':>7}  {'F1':>7}  {'FPR':>7}")
    print(f"  {'-'*55}")
    sweep = []
    for t in np.linspace(0.0, 1.0, 101):
        r = evaluate_threshold(y_test, y_proba, round(t, 2))
        sweep.append(r)
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        r = evaluate_threshold(y_test, y_proba, t)
        print(f"  {r['threshold']:>10.2f}  {r['acc']:>7.4f}  {r['precision']:>7.4f}  "
              f"{r['recall']:>7.4f}  {r['f1']:>7.4f}  {r['fpr']:>7.4f}")

    viable = [r for r in sweep if r['recall'] >= 0.90]
    best   = min(viable, key=lambda r: r['fpr']) if viable else baseline
    print(f"\n  Best threshold (recall≥90%, min FPR): {best['threshold']:.2f}")
    print(f"    → FPR={best['fpr']:.4f}  Recall={best['recall']:.4f}  F1={best['f1']:.4f}")

    # ── Save model + C headers ─────────────────────────────────────────────
    print("\n[5/5] Saving model and generating C headers...")
    joblib.dump(clf,    MODELS_DIR / 'phase2b_decision_tree.joblib')
    joblib.dump(scaler, MODELS_DIR / 'phase2b_scaler.joblib')
    with open(MODELS_DIR / 'phase2b_feature_names.json', 'w') as f:
        json.dump(feature_names, f)

    # Compact C headers (float32 and int16)
    from scripts.phase3_quantization import QuantizedDecisionTree, generate_c_header
    for dtype_name, dtype in [
        ('float32', QuantizedDecisionTree.DTYPE_FLOAT32),
        ('int16',   QuantizedDecisionTree.DTYPE_INT16),
    ]:
        qt = QuantizedDecisionTree.from_sklearn(clf, X_train, dtype=dtype)
        path = MODELS_DIR / f'stm32_tree_nslkdd_{dtype_name}.h'
        generate_c_header(qt, feature_names, path)
        print(f"  {dtype_name:8} → {qt.size_bytes()} bytes ({qt.size_kb():.3f} KB)")

    # ── Save results ───────────────────────────────────────────────────────
    results = {
        'phase': '2b',
        'dataset': 'NSL-KDD',
        'model': 'TinyDecisionTree',
        'n_train': int(len(X_train)),
        'n_test':  int(len(X_test)),
        'n_features': int(X_train.shape[1]),
        'train_attack_ratio': float(y_train.mean()),
        'test_attack_ratio':  float(y_test.mean()),
        'novel_attack_types_in_test': sorted(novel_attacks),
        'train_time_s': round(train_time, 4),
        'tree_depth':   int(clf.get_depth()),
        'n_leaves':     int(clf.get_n_leaves()),
        'n_nodes':      int(clf.tree_.node_count),
        'model_size_kb': round(model_kb, 3),
        'inference_time_ms': round(inf_ms, 6),
        'roc_auc': round(roc_auc, 4),
        'baseline_threshold': baseline,
        'best_threshold': best,
        'threshold_sweep': sweep,
    }
    out = RESULTS_DIR / 'phase2b_nslkdd_results.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved results → results/phase2b_nslkdd_results.json")

    print("\n" + "=" * 70)
    print("PHASE 2b COMPLETE")
    print("=" * 70)
    return results


if __name__ == '__main__':
    run()
