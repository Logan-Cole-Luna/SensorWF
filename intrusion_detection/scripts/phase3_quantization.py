#!/usr/bin/env python3
"""
Phase 3: Decision Tree Quantization for STM32 Deployment
---------------------------------------------------------
Quantizes the trained TinyDecisionTree (from Phase 2) at three precision levels:
  - Float32  (sklearn baseline)
  - INT16    (thresholds quantized to 16-bit integers)
  - INT8     (thresholds quantized to 8-bit integers)

For each level, reports:
  - Model size in bytes / KB
  - Accuracy, Recall, F1, FPR on UNSW-NB15 test set
  - Inference time in microseconds

Also generates a C header (stm32_tree.h) with the quantized INT8 tree
ready to compile onto an STM32 device.
"""

import sys
sys.path.insert(0, '/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection')

import numpy as np
import pandas as pd
import json
import time
import struct
import joblib
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

DATASET_DIR = Path('/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection/datasets')
RESULTS_DIR = Path('/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection/results')
MODELS_DIR  = Path('/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection/models/trained_models')
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Quantized Decision Tree
# ─────────────────────────────────────────────────────────────────────────────

class QuantizedDecisionTree:
    """
    A portable, serializable decision tree that stores split thresholds
    as either float32, int16, or int8 with per-feature min-max scaling.

    Tree is stored as flat parallel arrays (structure-of-arrays), the same
    layout used in the generated C header.
    """

    DTYPE_FLOAT32 = 'float32'
    DTYPE_INT16   = 'int16'
    DTYPE_INT8    = 'int8'

    def __init__(self, dtype: str = DTYPE_FLOAT32):
        assert dtype in (self.DTYPE_FLOAT32, self.DTYPE_INT16, self.DTYPE_INT8)
        self.dtype = dtype
        # Arrays filled by from_sklearn()
        self.feature    = None  # int16 array  — feature index (-1 = leaf)
        self.threshold  = None  # quantized threshold array
        self.left       = None  # int16 array  — left child index
        self.right      = None  # int16 array  — right child index
        self.leaf_class = None  # int8 array   — class at leaf (-1 = not leaf)
        # Feature quantization params (only used for int modes)
        self.feat_min   = None  # float32 per-feature minimum
        self.feat_scale = None  # float32 per-feature (max-min) / QMAX
        self.n_nodes    = 0
        self.n_features = 0
        self.qmax       = {'float32': None, 'int16': 32767, 'int8': 127}[dtype]

    @classmethod
    def from_sklearn(cls, sk_tree, X_train: np.ndarray, dtype: str = DTYPE_FLOAT32):
        """
        Extract a sklearn DecisionTreeClassifier into the quantized format.
        X_train is used to compute per-feature ranges for quantization.
        """
        obj = cls(dtype)
        tree = sk_tree.tree_
        n    = tree.node_count
        obj.n_nodes    = n
        obj.n_features = sk_tree.n_features_in_

        # Raw sklearn arrays
        raw_feature   = tree.feature.copy()    # -2 for leaves
        raw_threshold = tree.threshold.copy()  # -2.0 for leaves
        raw_left      = tree.children_left     # -1 for leaves
        raw_right     = tree.children_right    # -1 for leaves

        # Majority class at each leaf (from value array)
        values = tree.value  # shape (n_nodes, 1, n_classes)
        leaf_class = np.array([
            int(np.argmax(values[i, 0])) if raw_left[i] == -1 else -1
            for i in range(n)
        ], dtype=np.int8)

        # Per-feature quantization params (computed from X_train)
        if dtype != cls.DTYPE_FLOAT32:
            feat_min   = X_train.min(axis=0).astype(np.float32)
            feat_max   = X_train.max(axis=0).astype(np.float32)
            feat_range = feat_max - feat_min
            feat_range[feat_range == 0] = 1.0  # avoid div-by-zero
            qmax       = obj.qmax
            feat_scale = feat_range / qmax      # float32 per feature

            # Quantize each split threshold using its feature's scale
            q_threshold = np.full(n, 0, dtype=np.int16 if dtype == cls.DTYPE_INT16 else np.int8)
            for i in range(n):
                f = raw_feature[i]
                if f >= 0:  # non-leaf
                    q = (raw_threshold[i] - feat_min[f]) / feat_scale[f]
                    q = np.clip(np.round(q), -qmax, qmax)
                    q_threshold[i] = q.astype(np.int16 if dtype == cls.DTYPE_INT16 else np.int8)
                # else leave as 0 (ignored for leaves)

            obj.feat_min   = feat_min
            obj.feat_scale = feat_scale
            obj.threshold  = q_threshold
        else:
            obj.feat_min   = None
            obj.feat_scale = None
            obj.threshold  = raw_threshold.astype(np.float32)

        obj.feature    = raw_feature.astype(np.int16)
        obj.left       = raw_left.astype(np.int16)
        obj.right      = raw_right.astype(np.int16)
        obj.leaf_class = leaf_class
        return obj

    # ── Inference ─────────────────────────────────────────────────────────

    def _quantize_sample(self, x: np.ndarray) -> np.ndarray:
        """Quantize a single feature vector (for int modes)."""
        q = (x - self.feat_min) / self.feat_scale
        return np.clip(np.round(q), -self.qmax, self.qmax)

    def predict_one(self, x: np.ndarray) -> int:
        """Traverse tree for a single sample; return class label."""
        if self.dtype != self.DTYPE_FLOAT32:
            xq = self._quantize_sample(x)
        else:
            xq = x

        node = 0
        while self.feature[node] >= 0:          # not a leaf
            f = self.feature[node]
            if xq[f] <= self.threshold[node]:
                node = self.left[node]
            else:
                node = self.right[node]
        return int(self.leaf_class[node])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_one(X[i]) for i in range(len(X))])

    # ── Size ──────────────────────────────────────────────────────────────

    def size_bytes(self) -> int:
        """Exact byte count of the arrays stored on device."""
        n = self.n_nodes
        f = self.n_features

        threshold_bytes = {
            self.DTYPE_FLOAT32: 4,
            self.DTYPE_INT16:   2,
            self.DTYPE_INT8:    1,
        }[self.dtype]

        # Struct-of-arrays:
        #   feature    int16  x n
        #   threshold  T      x n
        #   left       int16  x n
        #   right      int16  x n
        #   leaf_class int8   x n
        tree_bytes = n * (2 + threshold_bytes + 2 + 2 + 1)

        # Quantization params (only for int modes)
        param_bytes = 0
        if self.dtype != self.DTYPE_FLOAT32:
            param_bytes = f * 4 + f * 4   # feat_min float32, feat_scale float32

        return tree_bytes + param_bytes

    def size_kb(self) -> float:
        return self.size_bytes() / 1024


# ─────────────────────────────────────────────────────────────────────────────
# C header generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_c_header(qtree: QuantizedDecisionTree, feature_names: list, path: Path):
    """Generate a portable C header that can be compiled on STM32."""
    n = qtree.n_nodes
    f = qtree.n_features
    is_int8   = qtree.dtype == QuantizedDecisionTree.DTYPE_INT8
    is_int16  = qtree.dtype == QuantizedDecisionTree.DTYPE_INT16
    T_type    = 'int8_t' if is_int8 else ('int16_t' if is_int16 else 'float')

    lines = [
        '/* Auto-generated by phase3_quantization.py — DO NOT EDIT */',
        '#pragma once',
        '#include <stdint.h>',
        '',
        f'#define IDS_N_NODES    {n}',
        f'#define IDS_N_FEATURES {f}',
        f'#define IDS_QMAX       {qtree.qmax if qtree.qmax else "/* float */" }',
        '',
    ]

    # Feature indices
    lines.append(f'static const int16_t ids_feature[{n}] = {{')
    lines.append('    ' + ', '.join(str(v) for v in qtree.feature))
    lines.append('};')
    lines.append('')

    # Thresholds
    lines.append(f'static const {T_type} ids_threshold[{n}] = {{')
    if qtree.dtype == QuantizedDecisionTree.DTYPE_FLOAT32:
        lines.append('    ' + ', '.join(f'{v:.6f}f' for v in qtree.threshold))
    else:
        lines.append('    ' + ', '.join(str(int(v)) for v in qtree.threshold))
    lines.append('};')
    lines.append('')

    # Left / right children
    lines.append(f'static const int16_t ids_left[{n}] = {{')
    lines.append('    ' + ', '.join(str(v) for v in qtree.left))
    lines.append('};')
    lines.append('')
    lines.append(f'static const int16_t ids_right[{n}] = {{')
    lines.append('    ' + ', '.join(str(v) for v in qtree.right))
    lines.append('};')
    lines.append('')

    # Leaf classes
    lines.append(f'static const int8_t ids_leaf_class[{n}] = {{')
    lines.append('    ' + ', '.join(str(int(v)) for v in qtree.leaf_class))
    lines.append('};')
    lines.append('')

    if qtree.dtype != QuantizedDecisionTree.DTYPE_FLOAT32:
        lines.append(f'static const float ids_feat_min[{f}] = {{')
        lines.append('    ' + ', '.join(f'{v:.6f}f' for v in qtree.feat_min))
        lines.append('};')
        lines.append('')
        lines.append(f'static const float ids_feat_scale[{f}] = {{')
        lines.append('    ' + ', '.join(f'{v:.6f}f' for v in qtree.feat_scale))
        lines.append('};')
        lines.append('')

    # Inference function
    if qtree.dtype != QuantizedDecisionTree.DTYPE_FLOAT32:
        lines += [
            '/* Quantized inference — call with a raw float feature vector */',
            'static inline int ids_predict(const float *x) {',
            f'    {T_type} xq[IDS_N_FEATURES];',
            '    for (int i = 0; i < IDS_N_FEATURES; i++) {',
            '        float q = (x[i] - ids_feat_min[i]) / ids_feat_scale[i];',
            f'        if (q > IDS_QMAX) q = IDS_QMAX;',
            f'        if (q < -IDS_QMAX) q = -IDS_QMAX;',
            f'        xq[i] = ({T_type})q;',
            '    }',
            '    int16_t node = 0;',
            '    while (ids_feature[node] >= 0) {',
            '        int16_t f = ids_feature[node];',
            '        if (xq[f] <= ids_threshold[node])',
            '            node = ids_left[node];',
            '        else',
            '            node = ids_right[node];',
            '    }',
            '    return ids_leaf_class[node];',
            '}',
        ]
    else:
        lines += [
            '/* Float32 inference */',
            'static inline int ids_predict(const float *x) {',
            '    int16_t node = 0;',
            '    while (ids_feature[node] >= 0) {',
            '        int16_t f = ids_feature[node];',
            '        if (x[f] <= ids_threshold[node])',
            '            node = ids_left[node];',
            '        else',
            '            node = ids_right[node];',
            '    }',
            '    return ids_leaf_class[node];',
            '}',
        ]

    path.write_text('\n'.join(lines) + '\n')
    try:
        rel = path.relative_to(Path('/Users/loganluna/VSCode/SatelliteAnomalyDetection'))
    except ValueError:
        rel = path
    print(f"  Saved C header → {rel}")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(qtree, X_test, y_test, n_timing=500):
    y_pred = qtree.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Inference time (average over n single-sample predictions)
    t0 = time.perf_counter()
    for _ in range(n_timing):
        qtree.predict_one(X_test[0])
    inf_us = (time.perf_counter() - t0) / n_timing * 1e6

    return {
        'accuracy':  round(float(accuracy_score(y_test, y_pred)), 4),
        'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        'recall':    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        'f1':        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        'fpr':       round(fpr, 4),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'size_bytes': qtree.size_bytes(),
        'size_kb':    round(qtree.size_kb(), 3),
        'inference_us': round(inf_us, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def load_test_data():
    """Reload UNSW-NB15 test set using the saved scaler from Phase 2."""
    train_df = pd.read_parquet(DATASET_DIR / 'UNSW_NB15_training-set.parquet')
    test_df  = pd.read_parquet(DATASET_DIR / 'UNSW_NB15_testing-set.parquet')

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
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0).astype(np.float32)

    scaler = joblib.load(MODELS_DIR / 'phase2_scaler.joblib')
    X_scaled = scaler.transform(X.values)

    X_train = X_scaled[:split_idx]
    X_test  = X_scaled[split_idx:]
    y_test  = y[split_idx:]
    return X_train, X_test, y_test


def run_phase3():
    print("=" * 70)
    print("PHASE 3 — Decision Tree Quantization")
    print("=" * 70)

    # ── Load trained model + data ──────────────────────────────────────────
    print("\n[1/4] Loading Phase 2 model + UNSW-NB15 test data...")
    sk_tree = joblib.load(MODELS_DIR / 'phase2_decision_tree.joblib')
    with open(MODELS_DIR / 'phase2_feature_names.json') as f:
        feature_names = json.load(f)
    X_train, X_test, y_test = load_test_data()

    print(f"  Tree depth: {sk_tree.get_depth()}  Leaves: {sk_tree.get_n_leaves()}")
    print(f"  Node count: {sk_tree.tree_.node_count}")
    print(f"  Sklearn pickle size: {len(pickle.dumps(sk_tree)) / 1024:.2f} KB")
    print(f"  Test samples: {len(X_test)}")

    # ── Build quantized variants ───────────────────────────────────────────
    print("\n[2/4] Building quantized tree variants...")

    variants = {}
    for dtype_name, dtype in [
        ('float32', QuantizedDecisionTree.DTYPE_FLOAT32),
        ('int16',   QuantizedDecisionTree.DTYPE_INT16),
        ('int8',    QuantizedDecisionTree.DTYPE_INT8),
    ]:
        print(f"  Building {dtype_name}...", end=' ', flush=True)
        qt = QuantizedDecisionTree.from_sklearn(sk_tree, X_train, dtype=dtype)
        variants[dtype_name] = qt
        print(f"→ {qt.size_bytes()} bytes ({qt.size_kb():.3f} KB)")

    # ── Evaluate all variants ──────────────────────────────────────────────
    print("\n[3/4] Evaluating on UNSW-NB15 test set...")

    all_results = {}
    for name, qt in variants.items():
        print(f"  {name:8}...", end=' ', flush=True)
        r = evaluate(qt, X_test, y_test)
        all_results[name] = r
        print(f"Acc={r['accuracy']:.4f}  F1={r['f1']:.4f}  FPR={r['fpr']:.4f}  "
              f"{r['size_bytes']}B  {r['inference_us']:.2f}µs")

    # ── Print comparison table ─────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("QUANTIZATION COMPARISON")
    print("=" * 90)
    baseline = all_results['float32']
    header = f"{'Precision':>10}  {'Size(B)':>8}  {'Size(KB)':>8}  {'Acc':>7}  {'F1':>7}  {'Recall':>7}  {'FPR':>7}  {'Inf(µs)':>8}"
    print(header)
    print("-" * 90)
    for name, r in all_results.items():
        acc_delta = r['accuracy'] - baseline['accuracy']
        f1_delta  = r['f1'] - baseline['f1']
        flag = ''
        if name != 'float32':
            flag = f"  (acc Δ={acc_delta:+.4f}, F1 Δ={f1_delta:+.4f})"
        print(f"  {name:>8}  {r['size_bytes']:>8}  {r['size_kb']:>8.3f}  "
              f"{r['accuracy']:>7.4f}  {r['f1']:>7.4f}  {r['recall']:>7.4f}  "
              f"{r['fpr']:>7.4f}  {r['inference_us']:>8.2f}{flag}")

    print(f"\n  Target: < 5 KB → {'ACHIEVED' if all_results['int8']['size_kb'] < 5 else 'NOT MET'} "
          f"(INT8 = {all_results['int8']['size_kb']:.3f} KB)")

    # ── Generate C headers for all variants ───────────────────────────────
    print("\n[4/4] Generating STM32 C headers...")
    generate_c_header(variants['float32'], feature_names, MODELS_DIR / 'stm32_tree_float32.h')
    generate_c_header(variants['int16'],   feature_names, MODELS_DIR / 'stm32_tree_int16.h')
    generate_c_header(variants['int8'],    feature_names, MODELS_DIR / 'stm32_tree_int8.h')

    # ── Save results ───────────────────────────────────────────────────────
    results = {
        'phase': 3,
        'model': 'TinyDecisionTree',
        'dataset': 'UNSW-NB15',
        'n_nodes': int(sk_tree.tree_.node_count),
        'tree_depth': int(sk_tree.get_depth()),
        'n_leaves': int(sk_tree.get_n_leaves()),
        'sklearn_pickle_kb': round(len(pickle.dumps(sk_tree)) / 1024, 3),
        'quantization_results': all_results,
        'target_5kb_achieved': all_results['int8']['size_kb'] < 5,
    }
    out = RESULTS_DIR / 'phase3_results.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results → results/phase3_results.json")

    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE")
    print("=" * 70)
    return results


if __name__ == '__main__':
    run_phase3()
