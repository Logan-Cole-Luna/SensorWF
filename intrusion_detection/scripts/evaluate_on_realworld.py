#!/usr/bin/env python3
"""
Evaluate lightweight IDS models on real-world UNSW-NB15 dataset
"""

import sys
sys.path.insert(0, '/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection')

import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import pickle

from models.lightweight_ids_models import MODELS, get_model

class RealWorldIDSEvaluator:
    def __init__(self, output_dir="results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
    
    def load_unswnb15(self):
        """Load UNSW-NB15 dataset from parquet files"""
        print("[*] Loading UNSW-NB15 dataset...")
        
        # Load training and testing sets
        train_df = pd.read_parquet(
            '/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection/datasets/UNSW_NB15_training-set.parquet'
        )
        test_df = pd.read_parquet(
            '/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection/datasets/UNSW_NB15_testing-set.parquet'
        )
        
        print(f"  Training set: {train_df.shape}")
        print(f"  Testing set: {test_df.shape}")
        print(f"  Columns: {train_df.columns.tolist()}")
        
        return train_df, test_df
    
    def preprocess_unswnb15(self, train_df, test_df):
        """Preprocess UNSW-NB15 data"""
        print("[*] Preprocessing data...")
        
        # Combine for preprocessing
        combined = pd.concat([train_df, test_df], ignore_index=True)
        
        # The label column is typically 'label' or 'Attack'
        label_col = 'label' if 'label' in combined.columns else 'Attack'
        print(f"  Using label column: {label_col}")
        
        # Separate features and labels
        y = combined[label_col].values
        X = combined.drop(columns=[label_col])
        
        # Print stats
        print(f"  Original shape: {X.shape}")
        print(f"  Classes: {np.unique(y)}")
        print(f"  Class distribution: {np.bincount(y.astype(int))}")
        
        # Handle categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"  Encoding categorical columns: {categorical_cols.tolist()}")
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        # Handle missing values and invalid strings
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(0)

        # Convert to numpy
        X = X.values.astype(np.float32)
        y = y.astype(int)
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        print(f"  Final shape: {X.shape}")
        print(f"  Feature range: [{X.min():.2f}, {X.max():.2f}]")
        
        # Split into train/test
        split_idx = len(train_df)
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"  Train attack ratio: {y_train.mean():.2%}")
        print(f"  Test attack ratio: {y_test.mean():.2%}")
        
        return X_train, X_test, y_train, y_test, scaler
    
    def train_model(self, model, X_train, y_train):
        """Train a model and measure time"""
        print(f"\n  Training {model.name}...", end=" ", flush=True)
        start_time = time.time()
        model.model.fit(X_train, y_train)
        train_time = time.time() - start_time
        print(f"✓ ({train_time:.3f}s)")
        return train_time
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test, train_time):
        """Evaluate model on real data"""
        print(f"    Evaluating...", end=" ", flush=True)
        
        # Predictions
        y_pred_train = model.model.predict(X_train)
        y_pred_test = model.model.predict(X_test)
        
        # Probability predictions
        if hasattr(model.model, 'predict_proba'):
            y_proba_test = model.model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba_test)
        else:
            roc_auc = None
        
        # Inference time (100 predictions)
        start_time = time.time()
        for _ in range(100):
            model.model.predict(X_test[:1])
        inference_time_ms = (time.time() - start_time) / 10
        
        # Model size
        model_size_kb = len(pickle.dumps(model.model)) / 1024
        
        # Metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
        
        metrics = {
            'model': model.name,
            'train_acc': accuracy_score(y_train, y_pred_train),
            'test_acc': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test),
            'recall': recall_score(y_test, y_pred_test),
            'f1': f1_score(y_test, y_pred_test),
            'roc_auc': roc_auc,
            'train_time_s': train_time,
            'inference_time_ms': inference_time_ms,
            'model_size_kb': model_size_kb,
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'fpr': float(fp) / (fp + tn) if (fp + tn) > 0 else 0,
            'dataset': 'UNSW-NB15'
        }
        
        print("✓")
        return metrics
    
    def run_evaluation(self):
        """Run complete evaluation on real data"""
        print("=" * 70)
        print("Real-World Intrusion Detection Evaluation - UNSW-NB15")
        print("=" * 70)
        
        # Load and preprocess data
        train_df, test_df = self.load_unswnb15()
        X_train, X_test, y_train, y_test, scaler = self.preprocess_unswnb15(train_df, test_df)
        
        # Train and evaluate each model
        print("\nTraining models on UNSW-NB15...")
        for model_name in ['tree', 'xgb', 'tiny_xgb', 'rf', 'et']:
            try:
                model = get_model(model_name)
                train_time = self.train_model(model, X_train, y_train)
                metrics = self.evaluate_model(model, X_train, X_test, y_train, y_test, train_time)
                self.results.append(metrics)
            except Exception as e:
                print(f"  ✗ Error training {model_name}: {e}")
        
        return self.results
    
    def print_results(self):
        """Print results"""
        print("\n" + "=" * 90)
        print("REAL-WORLD RESULTS (UNSW-NB15)")
        print("=" * 90)
        
        # Sort by F1 score
        self.results.sort(key=lambda x: x['f1'], reverse=True)
        
        print(f"\n{'Model':<18} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'FPR':<8} {'Size':<10} {'Speed(ms)':<10}")
        print("-" * 90)
        
        for r in self.results:
            print(f"{r['model']:<18} {r['test_acc']:.4f}  {r['precision']:.4f}  {r['recall']:.4f}  "
                  f"{r['f1']:.4f}  {r['fpr']:.4f}  {r['model_size_kb']:>8.2f}  {r['inference_time_ms']:>9.4f}")
        
        # Detailed breakdown
        print("\n" + "=" * 70)
        print("DETAILED METRICS")
        print("=" * 70)
        
        for r in self.results:
            print(f"\n{r['model']}:")
            print(f"  Accuracy (Train/Test): {r['train_acc']:.4f} / {r['test_acc']:.4f}")
            print(f"  Precision: {r['precision']:.4f}")
            print(f"  Recall: {r['recall']:.4f}")
            print(f"  F1-Score: {r['f1']:.4f}")
            if r['roc_auc']:
                print(f"  ROC-AUC: {r['roc_auc']:.4f}")
            print(f"  False Positive Rate: {r['fpr']:.4f}")
            print(f"  Model Size: {r['model_size_kb']:.2f} KB")
            print(f"  Inference Time: {r['inference_time_ms']:.6f} ms")
            print(f"  Training Time: {r['train_time_s']:.3f} s")
            print(f"  Confusion Matrix: TN={r['tn']}, FP={r['fp']}, FN={r['fn']}, TP={r['tp']}")
    
    def save_results_json(self):
        """Save results to JSON"""
        output_file = self.output_dir / "realworld_results.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to {output_file}")

if __name__ == "__main__":
    evaluator = RealWorldIDSEvaluator(
        output_dir="/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection/results"
    )
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    # Print results
    evaluator.print_results()
    
    # Save results
    evaluator.save_results_json()
    
    print("\n✓ Real-world evaluation complete!")
