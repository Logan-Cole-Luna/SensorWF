#!/usr/bin/env python3
"""
Phase 2: Real-World Validation - TinyDecisionTree on UNSW-NB15
Retrains and evaluates TinyDecisionTree with detailed FPR analysis.
Saves the trained model + scaler for Phase 3 quantization.
"""

import sys
sys.path.insert(0, '/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection')

import pandas as pd
import numpy as np
import time
import json
import joblib
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

DATASET_DIR = Path('/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection/datasets')
RESULTS_DIR = Path('/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection/results')
MODELS_DIR = Path('/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection/models/trained_models')

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def load_unswnb15():
    train_df = pd.read_parquet(DATASET_DIR / 'UNSW_NB15_training-set.parquet')
    test_df  = pd.read_parquet(DATASET_DIR / 'UNSW_NB15_testing-set.parquet')
    print(f"  Train: {train_df.shape}  Test: {test_df.shape}")
    return train_df, test_df


def preprocess(train_df, test_df):
    combined = pd.concat([train_df, test_df], ignore_index=True)
    split_idx = len(train_df)

    label_col = 'label' if 'label' in combined.columns else 'Attack'
    y = combined[label_col].values

    X = combined.drop(columns=[label_col])

    # Drop the attack_cat column (string, high-cardinality, leaks label info)
    if 'attack_cat' in X.columns:
        X = X.drop(columns=['attack_cat'])

    # Encode remaining categoricals
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Coerce and fill
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0).astype(np.float32)

    feature_names = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    y = y.astype(int)

    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"  Features: {X_scaled.shape[1]}  Train samples: {len(X_train)}  Test samples: {len(X_test)}")
    print(f"  Train attack ratio: {y_train.mean():.2%}  Test attack ratio: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test, scaler, feature_names


def measure_inference_time(model, X_test, n=1000):
    """Average over n single-sample predictions (ms)."""
    sample = X_test[:1]
    start = time.perf_counter()
    for _ in range(n):
        model.predict(sample)
    return (time.perf_counter() - start) / n * 1000


def evaluate_at_threshold(y_test, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    return dict(threshold=threshold, acc=acc, precision=prec, recall=tpr,
                f1=f1, fpr=fpr, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def run_phase2():
    print("=" * 70)
    print("PHASE 2 — Real-World Validation: TinyDecisionTree on UNSW-NB15")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading UNSW-NB15...")
    train_df, test_df = load_unswnb15()

    # ── Preprocess ─────────────────────────────────────────────────────────
    print("\n[2/4] Preprocessing...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(train_df, test_df)

    # ── Train TinyDecisionTree ─────────────────────────────────────────────
    print("\n[3/4] Training TinyDecisionTree (max_depth=5)...")
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=2, random_state=42)

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    print(f"  Training time: {train_time:.3f}s")
    print(f"  Actual tree depth: {clf.get_depth()}")
    print(f"  Number of leaves: {clf.get_n_leaves()}")

    # ── Evaluate ───────────────────────────────────────────────────────────
    print("\n[4/4] Evaluating...")

    y_proba   = clf.predict_proba(X_test)[:, 1]
    roc_auc   = roc_auc_score(y_test, y_proba)
    inf_ms    = measure_inference_time(clf, X_test)
    model_kb  = len(pickle.dumps(clf)) / 1024

    # Default threshold = 0.5
    baseline = evaluate_at_threshold(y_test, y_proba, 0.5)

    print(f"\n  Default threshold (0.5):")
    print(f"    Accuracy  : {baseline['acc']:.4f}")
    print(f"    Precision : {baseline['precision']:.4f}")
    print(f"    Recall    : {baseline['recall']:.4f}")
    print(f"    F1-Score  : {baseline['f1']:.4f}")
    print(f"    FPR       : {baseline['fpr']:.4f}")
    print(f"    ROC-AUC   : {roc_auc:.4f}")
    print(f"    Model size: {model_kb:.2f} KB")
    print(f"    Inference : {inf_ms:.6f} ms")

    # ── Threshold sweep for FPR analysis ──────────────────────────────────
    print("\n  Threshold sweep (FPR targets: 1%, 5%, 10%, 20%):")
    fpr_curve, tpr_curve, thresholds_roc = roc_curve(y_test, y_proba)

    sweep_results = []
    for t in np.linspace(0.0, 1.0, 101):
        sweep_results.append(evaluate_at_threshold(y_test, y_proba, round(t, 2)))

    print(f"\n  {'Threshold':>10}  {'Acc':>7}  {'Prec':>7}  {'Recall':>7}  {'F1':>7}  {'FPR':>7}")
    print(f"  {'-'*55}")
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        r = evaluate_at_threshold(y_test, y_proba, t)
        print(f"  {r['threshold']:>10.2f}  {r['acc']:>7.4f}  {r['precision']:>7.4f}  "
              f"{r['recall']:>7.4f}  {r['f1']:>7.4f}  {r['fpr']:>7.4f}")

    # Find threshold that minimises FPR while keeping recall >= 0.90
    viable = [r for r in sweep_results if r['recall'] >= 0.90]
    best = min(viable, key=lambda r: r['fpr']) if viable else baseline
    print(f"\n  Best threshold (recall>=90%, min FPR): {best['threshold']:.2f}")
    print(f"    → FPR={best['fpr']:.4f}  Recall={best['recall']:.4f}  F1={best['f1']:.4f}")

    # ── Save model + scaler ────────────────────────────────────────────────
    joblib.dump(clf,    MODELS_DIR / 'phase2_decision_tree.joblib')
    joblib.dump(scaler, MODELS_DIR / 'phase2_scaler.joblib')
    with open(MODELS_DIR / 'phase2_feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    print(f"\n  Saved model  → models/trained_models/phase2_decision_tree.joblib")
    print(f"  Saved scaler → models/trained_models/phase2_scaler.joblib")

    # ── Save results ───────────────────────────────────────────────────────
    results = {
        'phase': 2,
        'dataset': 'UNSW-NB15',
        'model': 'TinyDecisionTree',
        'n_train': int(len(X_train)),
        'n_test':  int(len(X_test)),
        'n_features': int(X_train.shape[1]),
        'train_attack_ratio': float(y_train.mean()),
        'test_attack_ratio':  float(y_test.mean()),
        'train_time_s': round(train_time, 4),
        'tree_depth': int(clf.get_depth()),
        'n_leaves': int(clf.get_n_leaves()),
        'model_size_kb': round(model_kb, 3),
        'inference_time_ms': round(inf_ms, 6),
        'roc_auc': round(roc_auc, 4),
        'baseline_threshold': baseline,
        'best_threshold': best,
        'threshold_sweep': sweep_results,
    }
    out = RESULTS_DIR / 'phase2_results.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results → results/phase2_results.json")

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)
    return results


if __name__ == '__main__':
    run_phase2()
