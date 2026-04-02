#!/usr/bin/env python3
"""
Phase 4B: Train CAN IDS from real benchmark datasets converted to CAN stream.

Flow:
  1) Convert UNSW or NSL tabular train/test to CAN frame CSVs
  2) Extract CAN sliding-window features
  3) Train TinyDecisionTree
  4) Evaluate and export STM32 headers + scaler
"""

import sys
sys.path.insert(0, '/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection')

import argparse
import json
import pickle
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from scripts.benchmark_to_can import CAN_LAYOUT, build_benchmark_can_dataset
from scripts.can_feature_engineering import FEATURE_NAMES, build_baseline, extract_features
from scripts.phase3_quantization import QuantizedDecisionTree, generate_c_header


ROOT = Path('/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection')
DATASET_DIR = ROOT / 'datasets' / 'CAN_FROM_BENCHMARK'
RESULTS_DIR = ROOT / 'results'
MODELS_DIR = ROOT / 'models' / 'trained_models'

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def measure_inf_ms(clf, X, n=1000):
    sample = X[:1]
    t0 = time.perf_counter()
    for _ in range(n):
        clf.predict(sample)
    return (time.perf_counter() - t0) / n * 1000


def per_attack_metrics(y_true, y_pred, attack_types):
    out = {}
    attack_types = np.asarray(attack_types)
    for atype in sorted(set(attack_types.tolist())):
        if atype == 'normal':
            continue
        mask = (attack_types == atype) | (attack_types == 'normal')
        yt = (attack_types[mask] != 'normal').astype(int)
        yp = y_pred[mask]
        if yt.sum() == 0:
            continue
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        out[atype] = {
            'recall': round(float(tp / (tp + fn)) if (tp + fn) > 0 else 0, 4),
            'fpr': round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0, 4),
            'n_samples': int(yt.sum()),
        }
    return out


def run(dataset: str, max_train_rows: int, max_test_rows: int, exclude_meta_frames: bool):
    print('=' * 72)
    print(f'PHASE 4B — Benchmark-to-CAN IDS ({dataset.upper()})')
    print('=' * 72)

    print('\n[1/6] Converting benchmark tabular data to CAN frames...')
    info = build_benchmark_can_dataset(
        dataset=dataset,
        out_dir=DATASET_DIR,
        max_train_rows=max_train_rows,
        max_test_rows=max_test_rows,
    )
    train_raw = pd.read_csv(info['train_path'])
    test_raw = pd.read_csv(info['test_path'])
    print(f"  Train frames: {len(train_raw):,}  Test frames: {len(test_raw):,}")

    if exclude_meta_frames:
        meta_id = CAN_LAYOUT[dataset]['meta_id']
        train_raw = train_raw[train_raw['can_id'] != meta_id].reset_index(drop=True)
        test_raw = test_raw[test_raw['can_id'] != meta_id].reset_index(drop=True)
        print(f"  Excluding meta frames (CAN ID 0x{meta_id:03X})")
        print(f"  Filtered train/test: {len(train_raw):,} / {len(test_raw):,}")

    print('\n[2/6] Building baseline from normal train CAN traffic...')
    baseline = build_baseline(train_raw[train_raw['label'] == 0])
    print(f"  Known IDs: {len(baseline)}")

    print('\n[3/6] Extracting CAN sliding-window features...')
    t0 = time.perf_counter()
    train_feat = extract_features(train_raw, baseline)
    print(f"  Train features: {len(train_feat):,} ({time.perf_counter() - t0:.2f}s)")
    t0 = time.perf_counter()
    test_feat = extract_features(test_raw, baseline)
    print(f"  Test features : {len(test_feat):,} ({time.perf_counter() - t0:.2f}s)")

    train_feat.to_csv(DATASET_DIR / f'{dataset}_can_train_features.csv', index=False)
    test_feat.to_csv(DATASET_DIR / f'{dataset}_can_test_features.csv', index=False)

    X_train = train_feat[FEATURE_NAMES].values.astype(np.float32)
    y_train = train_feat['label'].values.astype(int)
    X_test = test_feat[FEATURE_NAMES].values.astype(np.float32)
    y_test = test_feat['label'].values.astype(int)
    attack_types_test = test_raw['attack_type'].values[:len(test_feat)]

    print('\n[4/6] Scaling + training TinyDecisionTree...')
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=2, random_state=42)
    t0 = time.perf_counter()
    clf.fit(X_train_s, y_train)
    train_time = time.perf_counter() - t0
    print(f"  Train time   : {train_time:.3f}s")
    print(f"  Tree depth   : {clf.get_depth()}")
    print(f"  Leaves       : {clf.get_n_leaves()}")
    print(f"  Pickle size  : {len(pickle.dumps(clf)) / 1024:.2f} KB")

    print('\n[5/6] Evaluating...')
    y_pred = clf.predict(X_test_s)
    y_proba = clf.predict_proba(X_test_s)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    metrics = {
        'accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
        'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        'recall': round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        'f1': round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        'fpr': round(float(fpr), 4),
        'roc_auc': round(float(roc_auc_score(y_test, y_proba)), 4),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }
    inf_ms = measure_inf_ms(clf, X_test_s)
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  FPR      : {metrics['fpr']:.4f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"  Inference: {inf_ms:.6f} ms")

    attack_breakdown = per_attack_metrics(y_test, y_pred, attack_types_test)

    print('\n[6/6] Saving model and generating C headers...')
    joblib.dump(clf, MODELS_DIR / f'phase4b_{dataset}_can_decision_tree.joblib')
    joblib.dump(scaler, MODELS_DIR / f'phase4b_{dataset}_can_scaler.joblib')
    with open(MODELS_DIR / f'phase4b_{dataset}_can_feature_names.json', 'w') as f:
        json.dump(FEATURE_NAMES, f)

    for dtype_name, dtype in [
        ('float32', QuantizedDecisionTree.DTYPE_FLOAT32),
        ('int16', QuantizedDecisionTree.DTYPE_INT16),
    ]:
        qt = QuantizedDecisionTree.from_sklearn(clf, X_train_s, dtype=dtype)
        out_h = MODELS_DIR / f'stm32f373_{dataset}_can_ids_{dtype_name}.h'
        generate_c_header(qt, FEATURE_NAMES, out_h)
        print(f"  {dtype_name:8} -> {qt.size_bytes()} bytes ({qt.size_kb():.3f} KB)")

    results = {
        'phase': '4b',
        'dataset_source': dataset,
        'exclude_meta_frames': bool(exclude_meta_frames),
        'train_tabular_rows': info['train_rows'],
        'test_tabular_rows': info['test_rows'],
        'train_can_frames': info['train_frames'],
        'test_can_frames': info['test_frames'],
        'n_features': len(FEATURE_NAMES),
        'train_time_s': round(train_time, 4),
        'tree_depth': int(clf.get_depth()),
        'n_leaves': int(clf.get_n_leaves()),
        'model_size_kb': round(len(pickle.dumps(clf)) / 1024, 3),
        'inference_time_ms': round(inf_ms, 6),
        'overall': metrics,
        'per_attack_type': attack_breakdown,
        'artifacts': {
            'train_can_csv': info['train_path'],
            'test_can_csv': info['test_path'],
            'encoding_meta': info['meta_path'],
            'model': str(MODELS_DIR / f'phase4b_{dataset}_can_decision_tree.joblib'),
            'scaler': str(MODELS_DIR / f'phase4b_{dataset}_can_scaler.joblib'),
        },
    }

    out_json = RESULTS_DIR / f'phase4b_{dataset}_can_results.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results -> {out_json}")

    print('\n' + '=' * 72)
    print('PHASE 4B COMPLETE')
    print('=' * 72)


def main():
    ap = argparse.ArgumentParser(description='Train CAN IDS from UNSW/NSL converted CAN frames')
    ap.add_argument('--dataset', choices=['unsw', 'nsl'], required=True)
    ap.add_argument('--max-train-rows', type=int, default=120000)
    ap.add_argument('--max-test-rows', type=int, default=50000)
    ap.add_argument(
        '--include-meta-frames',
        action='store_true',
        help='Include meta CAN frames in feature extraction (default: excluded)',
    )
    args = ap.parse_args()
    run(
        dataset=args.dataset,
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
        exclude_meta_frames=not args.include_meta_frames,
    )


if __name__ == '__main__':
    main()
