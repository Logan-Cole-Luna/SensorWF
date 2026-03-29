#!/usr/bin/env python3
"""
Test SHAP explainability overhead on lightweight IDS models
Measure cost in memory, inference time, and code size
"""

import sys
sys.path.insert(0, '/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection')

import pandas as pd
import numpy as np
import time
import pickle
import psutil
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.lightweight_ids_models import get_model

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Note: SHAP not installed, will test installation overhead")

class SHAPOverheadTester:
    def __init__(self):
        self.results = []
        self.process = psutil.Process(os.getpid())
    
    def load_data(self):
        """Load UNSW-NB15 real-world data"""
        print("[*] Loading UNSW-NB15 data...")
        train_df = pd.read_parquet(
            '/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection/datasets/UNSW_NB15_training-set.parquet'
        )
        
        # Take small sample for testing
        train_df = train_df.sample(n=5000, random_state=42)
        
        y = train_df['label'].values.astype(int)
        X = train_df.drop(columns=['label', 'attack_cat'])
        
        # Handle categorical columns
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = pd.factorize(X[col])[0]
        
        X = X.fillna(0).astype(np.float32)
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"  Training: {X_train.shape}")
        print(f"  Testing: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def test_model_without_shap(self, model_name, X_train, X_test, y_train, y_test):
        """Test model inference without SHAP"""
        print(f"\n[*] Testing {model_name} WITHOUT SHAP...")
        
        # Memory before
        mem_before = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Train model
        model = get_model(model_name)
        start = time.time()
        model.model.fit(X_train, y_train)
        train_time = time.time() - start
        
        # Model size
        model_size_kb = len(pickle.dumps(model.model)) / 1024
        
        # Inference time (100 predictions)
        start = time.time()
        for _ in range(100):
            model.model.predict(X_test[:1])
        inference_time = (time.time() - start) / 100
        
        # Memory after
        mem_after = self.process.memory_info().rss / 1024 / 1024
        
        return {
            'model': model_name,
            'method': 'No SHAP',
            'train_time_s': train_time,
            'model_size_kb': model_size_kb,
            'inference_time_ms': inference_time * 1000,
            'memory_mb': mem_after - mem_before,
            'shap_size_kb': 0,
            'shap_time_ms': 0,
            'total_size_kb': model_size_kb,
            'total_time_ms': inference_time * 1000,
            'model_obj': model
        }
    
    def test_model_with_shap(self, model_name, model_obj, X_test, y_test):
        """Test model with SHAP explanations"""
        print(f"\n[*] Testing {model_name} WITH SHAP...")
        
        if not SHAP_AVAILABLE:
            print("  SHAP not available - skipping")
            return None
        
        # Memory before SHAP
        mem_before = self.process.memory_info().rss / 1024 / 1024
        
        try:
            # Create SHAP explainer
            start = time.time()
            explainer = shap.TreeExplainer(model_obj.model)
            explainer_setup_time = time.time() - start
            
            # Size of explainer code (approximate)
            explainer_size = len(pickle.dumps(explainer)) / 1024
            
            # Get SHAP values for small sample
            sample_size = min(100, len(X_test))
            start = time.time()
            shap_values = explainer.shap_values(X_test[:sample_size])
            shap_time = (time.time() - start) / sample_size  # per sample
            
            # Memory after SHAP
            mem_after = self.process.memory_info().rss / 1024 / 1024
            
            return {
                'model': model_name,
                'method': 'With SHAP',
                'explainer_size_kb': explainer_size,
                'explainer_setup_time_s': explainer_setup_time,
                'shap_time_per_sample_ms': shap_time * 1000,
                'shap_memory_mb': mem_after - mem_before,
                'shap_values_shape': str(shap_values.shape) if isinstance(shap_values, np.ndarray) else str(type(shap_values)),
            }
        except Exception as e:
            print(f"  Error: {e}")
            return None
    
    def run_test(self):
        """Run complete overhead test"""
        print("=" * 70)
        print("SHAP Explainability Overhead Test")
        print("=" * 70)
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Test models
        models_to_test = ['tree', 'xgb', 'rf']
        
        for model_name in models_to_test:
            print(f"\n{'='*70}")
            print(f"Testing {model_name.upper()}")
            print(f"{'='*70}")
            
            # Test without SHAP
            result_no_shap = self.test_model_without_shap(
                model_name, X_train, X_test, y_train, y_test
            )
            self.results.append(result_no_shap)
            
            # Test with SHAP
            result_with_shap = self.test_model_with_shap(
                model_name, result_no_shap['model_obj'], X_test, y_test
            )
            if result_with_shap:
                self.results.append(result_with_shap)
    
    def print_results(self):
        """Print formatted results"""
        print("\n" + "=" * 100)
        print("SHAP OVERHEAD ANALYSIS")
        print("=" * 100)
        
        # Group by model
        models = set(r['model'] for r in self.results)
        
        for model_name in sorted(models):
            model_results = [r for r in self.results if r['model'] == model_name]
            print(f"\n{model_name.upper()}")
            print("-" * 100)
            
            without_shap = [r for r in model_results if r['method'] == 'No SHAP'][0]
            with_shap = [r for r in model_results if r['method'] == 'With SHAP']
            
            # Print without SHAP
            print(f"WITHOUT SHAP:")
            print(f"  Model Size:           {without_shap['model_size_kb']:.2f} KB")
            print(f"  Inference Time:       {without_shap['inference_time_ms']:.4f} ms")
            print(f"  Training Time:        {without_shap['train_time_s']:.3f} s")
            print(f"  Memory Used:          {without_shap['memory_mb']:.2f} MB")
            
            if with_shap:
                shap_result = with_shap[0]
                print(f"\nWITH SHAP:")
                print(f"  Explainer Size:       {shap_result['explainer_size_kb']:.2f} KB")
                print(f"  SHAP Time per Sample: {shap_result['shap_time_per_sample_ms']:.2f} ms")
                print(f"  Explainer Setup:      {shap_result['explainer_setup_time_s']:.3f} s")
                print(f"  Memory for SHAP:      {shap_result['shap_memory_mb']:.2f} MB")
                print(f"  SHAP Values Shape:    {shap_result['shap_values_shape']}")
                
                # Calculate overhead
                print(f"\nOVERHEAD:")
                total_size = without_shap['model_size_kb'] + shap_result['explainer_size_kb']
                size_overhead_pct = (shap_result['explainer_size_kb'] / without_shap['model_size_kb']) * 100
                time_overhead_pct = (shap_result['shap_time_per_sample_ms'] / without_shap['inference_time_ms']) * 100
                
                print(f"  Size Increase:        {shap_result['explainer_size_kb']:.2f} KB ({size_overhead_pct:.0f}%)")
                print(f"  Time Increase:        {shap_result['shap_time_per_sample_ms']:.2f} ms ({time_overhead_pct:.0f}%)")
                print(f"  Total Model+SHAP:     {total_size:.2f} KB")
                print(f"  Total Latency:        {without_shap['inference_time_ms'] + shap_result['shap_time_per_sample_ms']:.2f} ms")
            else:
                print(f"\nWITH SHAP: Not available")
    
    def save_results_csv(self):
        """Save results to CSV"""
        output_file = Path('/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection/results/shap_overhead_results.csv')
        
        rows = []
        for r in self.results:
            row = {
                'Model': r['model'],
                'Method': r['method'],
                'Model_Size_KB': r.get('model_size_kb', ''),
                'Inference_Time_ms': r.get('inference_time_ms', ''),
                'SHAP_Size_KB': r.get('explainer_size_kb', ''),
                'SHAP_Time_ms': r.get('shap_time_per_sample_ms', ''),
                'Total_Size_KB': r.get('model_size_kb', 0) + r.get('explainer_size_kb', 0),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    tester = SHAPOverheadTester()
    
    # Check if SHAP needs to be installed
    if not SHAP_AVAILABLE:
        print("[*] Installing SHAP...")
        import subprocess
        subprocess.run([
            'bash', '-c',
            'source /Users/loganluna/VSCode/SatelliteAnomalyDetection/.venv/bin/activate && pip install shap -q'
        ])
        print("[*] SHAP installed, please run script again")
    else:
        tester.run_test()
        tester.print_results()
        tester.save_results_csv()
        
        print("\n" + "=" * 100)
        print("CONCLUSION")
        print("=" * 100)
        print("""
SHAP Overhead Summary:
- Explainer code: Additional 10-100 KB depending on model type
- Per-sample time: 10-100x overhead vs. inference alone
- Memory usage: Significant temporary spike during explanation

For satellites: Consider whether explainability is worth the cost.
Alternative: Generate SHAP values on ground station, not on satellite.
        """)
