#!/usr/bin/env python3
"""
Phase 4: CAN Bus IDS for STM32H7 Satellite
-------------------------------------------
Full pipeline:
  1. Generate synthetic satellite CAN bus dataset
  2. Extract 15 CAN-specific features (sliding window)
  3. Train TinyDecisionTree on CAN features
  4. Evaluate: accuracy, recall, FPR per attack type
  5. Export STM32H7 C headers (float32 + int16)

Outputs:
  datasets/CAN_SATELLITE/satellite_can_train.csv
  datasets/CAN_SATELLITE/satellite_can_test.csv
  datasets/CAN_SATELLITE/can_train_features.csv
  datasets/CAN_SATELLITE/can_test_features.csv
  models/trained_models/can_decision_tree.joblib
  models/trained_models/can_scaler.joblib
  models/trained_models/stm32h7_can_ids_float32.h
  models/trained_models/stm32h7_can_ids_int16.h
  results/phase4_can_results.json
"""

import sys
sys.path.insert(0, '/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection')

import numpy as np
import pandas as pd
import json, time, pickle, joblib
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

from scripts.can_dataset_generator import build_dataset, KNOWN_IDS
from scripts.can_feature_engineering import (
    extract_features, build_baseline, FEATURE_NAMES
)
from scripts.phase3_quantization import QuantizedDecisionTree, generate_c_header

DATASET_DIR = Path('/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection/datasets/CAN_SATELLITE')
RESULTS_DIR = Path('/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection/results')
MODELS_DIR  = Path('/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection/models/trained_models')

DATASET_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def measure_inf_ms(clf, X, n=1000):
    s = X[:1]
    t0 = time.perf_counter()
    for _ in range(n):
        clf.predict(s)
    return (time.perf_counter() - t0) / n * 1000


def per_attack_metrics(y_test, y_pred, attack_types):
    """Compute recall and FPR per attack type."""
    results = {}
    for atype in sorted(set(attack_types)):
        if atype == 'normal':
            continue
        mask = (attack_types == atype) | (attack_types == 'normal')
        yt = (attack_types[mask] != 'normal').astype(int)
        yp = y_pred[mask]
        if yt.sum() == 0:
            continue
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        results[atype] = {
            'recall': round(float(tp / (tp + fn)) if (tp + fn) > 0 else 0, 4),
            'fpr':    round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0, 4),
            'n_samples': int(yt.sum()),
        }
    return results


def run():
    print("=" * 70)
    print("PHASE 4 — CAN Bus IDS for STM32H7 Satellite")
    print("=" * 70)

    # ── 1. Generate dataset ────────────────────────────────────────────────
    print("\n[1/6] Generating satellite CAN bus dataset...")
    train_raw = build_dataset(duration_s=600.0,  output_path=DATASET_DIR / 'satellite_can_train.csv')
    test_raw  = build_dataset(duration_s=120.0,  output_path=DATASET_DIR / 'satellite_can_test.csv')

    # ── 2. Build baseline profile from normal training frames ──────────────
    print("\n[2/6] Building CAN ID baseline from normal traffic...")
    normal_train = train_raw[train_raw['label'] == 0]
    baseline = build_baseline(normal_train)
    print(f"  Known CAN IDs: {sorted(hex(k) for k in baseline)}")

    # ── 3. Extract features ────────────────────────────────────────────────
    print("\n[3/6] Extracting CAN bus features (sliding window=50)...")

    t0 = time.perf_counter()
    train_feat = extract_features(train_raw, baseline)
    print(f"  Train: {len(train_feat):,} samples  ({time.perf_counter()-t0:.2f}s)")

    t0 = time.perf_counter()
    test_feat  = extract_features(test_raw,  baseline)
    print(f"  Test : {len(test_feat):,} samples  ({time.perf_counter()-t0:.2f}s)")

    # Save feature CSVs
    train_feat.to_csv(DATASET_DIR / 'can_train_features.csv', index=False)
    test_feat.to_csv( DATASET_DIR / 'can_test_features.csv',  index=False)

    # Preserve attack_type from raw for per-attack eval
    test_attack_types = test_raw['attack_type'].values[:len(test_feat)]

    X_train = train_feat[FEATURE_NAMES].values.astype(np.float32)
    y_train = train_feat['label'].values.astype(int)
    X_test  = test_feat[FEATURE_NAMES].values.astype(np.float32)
    y_test  = test_feat['label'].values.astype(int)

    print(f"  Features per sample: {X_train.shape[1]}")
    print(f"  Train attack ratio: {y_train.mean():.1%}  Test: {y_test.mean():.1%}")

    # ── 4. Scale ───────────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── 5. Train ───────────────────────────────────────────────────────────
    print("\n[4/6] Training TinyDecisionTree (max_depth=5)...")
    clf = DecisionTreeClassifier(max_depth=5, min_samples_split=5,
                                  min_samples_leaf=2, random_state=42)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    print(f"  Training time : {train_time:.3f}s")
    print(f"  Tree depth    : {clf.get_depth()}")
    print(f"  Leaves        : {clf.get_n_leaves()}")
    print(f"  Nodes         : {clf.tree_.node_count}")
    print(f"  Pickle size   : {len(pickle.dumps(clf)) / 1024:.2f} KB")

    # ── 6. Evaluate ────────────────────────────────────────────────────────
    print("\n[5/6] Evaluating...")
    y_pred   = clf.predict(X_test)
    y_proba  = clf.predict_proba(X_test)[:, 1]
    roc_auc  = roc_auc_score(y_test, y_proba)
    inf_ms   = measure_inf_ms(clf, X_test)
    model_kb = len(pickle.dumps(clf)) / 1024

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"\n  Overall:")
    print(f"    Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"    Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"    Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"    F1        : {f1_score(y_test, y_pred):.4f}")
    print(f"    FPR       : {fpr:.4f}")
    print(f"    ROC-AUC   : {roc_auc:.4f}")
    print(f"    Model size: {model_kb:.2f} KB (pickle)")
    print(f"    Inference : {inf_ms:.4f} ms  ({inf_ms*1000:.2f} µs)")

    per_attack = per_attack_metrics(y_test, y_pred, test_attack_types)
    print(f"\n  Per-attack-type breakdown:")
    print(f"  {'Attack':12}  {'Recall':>8}  {'FPR':>8}  {'Samples':>8}")
    print(f"  {'-'*44}")
    for atype, m in per_attack.items():
        print(f"  {atype:12}  {m['recall']:>8.4f}  {m['fpr']:>8.4f}  {m['n_samples']:>8,}")

    # ── 7. C headers ───────────────────────────────────────────────────────
    print("\n[6/6] Generating STM32H7 C headers...")
    for dtype_name, dtype in [
        ('float32', QuantizedDecisionTree.DTYPE_FLOAT32),
        ('int16',   QuantizedDecisionTree.DTYPE_INT16),
    ]:
        qt   = QuantizedDecisionTree.from_sklearn(clf, X_train, dtype=dtype)
        path = MODELS_DIR / f'stm32h7_can_ids_{dtype_name}.h'
        generate_c_header(qt, FEATURE_NAMES, path)
        print(f"  {dtype_name:8} → {qt.size_bytes()} bytes ({qt.size_kb():.3f} KB)")

    # ── Save model + results ───────────────────────────────────────────────
    joblib.dump(clf,    MODELS_DIR / 'can_decision_tree.joblib')
    joblib.dump(scaler, MODELS_DIR / 'can_scaler.joblib')
    with open(MODELS_DIR / 'can_feature_names.json', 'w') as f:
        json.dump(FEATURE_NAMES, f)

    results = {
        'phase': 4,
        'dataset': 'Synthetic Satellite CAN Bus',
        'model': 'TinyDecisionTree',
        'n_features': len(FEATURE_NAMES),
        'feature_names': FEATURE_NAMES,
        'n_train': int(len(X_train)),
        'n_test':  int(len(X_test)),
        'attack_types': ['dos', 'fuzzy', 'spoofing', 'replay'],
        'train_attack_ratio': float(y_train.mean()),
        'test_attack_ratio':  float(y_test.mean()),
        'train_time_s': round(train_time, 4),
        'tree_depth':   int(clf.get_depth()),
        'n_leaves':     int(clf.get_n_leaves()),
        'n_nodes':      int(clf.tree_.node_count),
        'model_size_kb': round(model_kb, 3),
        'inference_time_ms': round(inf_ms, 6),
        'roc_auc': round(roc_auc, 4),
        'overall': {
            'accuracy':  round(float(accuracy_score(y_test, y_pred)), 4),
            'precision': round(float(precision_score(y_test, y_pred)), 4),
            'recall':    round(float(recall_score(y_test, y_pred)), 4),
            'f1':        round(float(f1_score(y_test, y_pred)), 4),
            'fpr':       round(fpr, 4),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        },
        'per_attack_type': per_attack,
    }

    out = RESULTS_DIR / 'phase4_can_results.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved results → results/phase4_can_results.json")

    print("\n" + "=" * 70)
    print("PHASE 4 COMPLETE")
    print("=" * 70)
    return results


if __name__ == '__main__':
    run()
