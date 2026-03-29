#!/usr/bin/env python3
"""
Train and evaluate lightweight IDS models
Focuses on performance metrics relevant to embedded systems
"""

import sys
sys.path.insert(0, '/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection')

import pandas as pd
import numpy as np
import time
from pathlib import Path
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import pickle

from models.lightweight_ids_models import MODELS, get_model

class IDSEvaluator:
    def __init__(self, output_dir="results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
    
    def load_data(self, data_path):
        """Load dataset"""
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        # Prepare features and labels
        X = df.drop(['label', 'attack_type'], axis=1).values
        y = df['label'].values
        
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Attack ratio: {y.mean():.1%}")
        
        return X_train, X_test, y_train, y_test, scaler
    
    def train_model(self, model, X_train, y_train):
        """Train a model and measure time"""
        print(f"\nTraining {model.name}...", end=" ", flush=True)
        start_time = time.time()
        model.model.fit(X_train, y_train)
        train_time = time.time() - start_time
        print(f"✓ ({train_time:.3f}s)")
        return train_time
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test, train_time):
        """Evaluate model on multiple metrics"""
        print(f"  Evaluating {model.name}...", end=" ", flush=True)
        
        # Predictions
        y_pred_train = model.model.predict(X_train)
        y_pred_test = model.model.predict(X_test)
        
        # Probability predictions for ROC-AUC
        if hasattr(model.model, 'predict_proba'):
            y_proba_test = model.model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba_test)
        else:
            roc_auc = None
        
        # Inference time
        start_time = time.time()
        for _ in range(1000):
            model.model.predict(X_test[:1])
        inference_time_ms = (time.time() - start_time) / 10  # per prediction in ms
        
        # Model size
        model_size_kb = len(pickle.dumps(model.model)) / 1024
        
        # Metrics
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
            'tn': confusion_matrix(y_test, y_pred_test)[0, 0],
            'fp': confusion_matrix(y_test, y_pred_test)[0, 1],
            'fn': confusion_matrix(y_test, y_pred_test)[1, 0],
            'tp': confusion_matrix(y_test, y_pred_test)[1, 1],
        }
        
        # False positive rate
        metrics['fpr'] = metrics['fp'] / (metrics['fp'] + metrics['tn'])
        
        print("✓")
        return metrics
    
    def run_evaluation(self, data_path):
        """Run complete evaluation"""
        print("=" * 70)
        print("Lightweight IDS Model Evaluation")
        print("=" * 70)
        
        # Load data
        X_train, X_test, y_train, y_test, scaler = self.load_data(data_path)
        
        # Train and evaluate each model
        for model_name in ['tree', 'rf', 'et', 'xgb', 'tiny_xgb']:
            try:
                model = get_model(model_name)
                train_time = self.train_model(model, X_train, y_train)
                metrics = self.evaluate_model(model, X_train, X_test, y_train, y_test, train_time)
                self.results.append(metrics)
            except Exception as e:
                print(f"  ✗ Error training {model_name}: {e}")
        
        return self.results
    
    def print_results(self):
        """Print results table"""
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        
        # Sort by F1 score
        self.results.sort(key=lambda x: x['f1'], reverse=True)
        
        print(f"\n{'Model':<18} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Size(KB)':<10} {'Inf(ms)':<10}")
        print("-" * 80)
        
        for r in self.results:
            print(f"{r['model']:<18} {r['test_acc']:.4f}  {r['precision']:.4f}  {r['recall']:.4f}  "
                  f"{r['f1']:.4f}  {r['model_size_kb']:.2f}  {r['inference_time_ms']:.4f}")
        
        # Print detailed metrics
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
            print(f"  Inference Time: {r['inference_time_ms']:.4f} ms")
            print(f"  Training Time: {r['train_time_s']:.3f} s")
            print(f"  Confusion Matrix:")
            print(f"    TN={r['tn']}, FP={r['fp']}, FN={r['fn']}, TP={r['tp']}")
    
    def save_results_json(self):
        """Save results to JSON"""
        output_file = self.output_dir / "evaluation_results.json"

        # Convert to serializable format
        results_json = []
        for r in self.results:
            r_copy = r.copy()
            if r_copy['roc_auc'] is None:
                r_copy['roc_auc'] = 0
            # Convert numpy types to Python native types
            for key, value in r_copy.items():
                if isinstance(value, (np.integer, np.floating)):
                    r_copy[key] = float(value) if isinstance(value, np.floating) else int(value)
            results_json.append(r_copy)

        with open(output_file, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"\n✓ Results saved to {output_file}")

if __name__ == "__main__":
    evaluator = IDSEvaluator(output_dir="/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection/results")
    
    # Use synthetic data
    data_path = "/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection/datasets/SAMPLE/synthetic_train.csv"
    
    # Run evaluation
    results = evaluator.run_evaluation(data_path)
    
    # Print results
    evaluator.print_results()
    
    # Save results
    evaluator.save_results_json()
    
    print("\n✓ Evaluation complete!")
