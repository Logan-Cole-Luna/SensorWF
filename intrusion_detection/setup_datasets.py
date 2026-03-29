#!/usr/bin/env python3
"""
Download and setup datasets for intrusion detection testing.
Supports UNSW-NB15, CIC-IDS2017, and NSL-KDD datasets.
"""

import os
import urllib.request
import zipfile
import gzip
import shutil
import pandas as pd
from pathlib import Path

class DatasetDownloader:
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
    def download_nsl_kdd(self):
        """Download NSL-KDD dataset (smallest, good for testing)"""
        print("[*] Downloading NSL-KDD dataset...")
        nsl_kdd_url = "https://www.unb.ca/cic/datasets/nsl-kdd.html"
        
        # NSL-KDD files are available on multiple mirrors
        files = {
            'KDDTrain+.txt': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt',
            'KDDTest+.txt': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt'
        }
        
        nsl_dir = self.dataset_dir / 'NSL-KDD'
        nsl_dir.mkdir(exist_ok=True)
        
        try:
            for filename, url in files.items():
                filepath = nsl_dir / filename
                if filepath.exists():
                    print(f"  {filename} already exists")
                    continue
                print(f"  Downloading {filename}...")
                urllib.request.urlretrieve(url, filepath)
                print(f"  ✓ Downloaded {filename}")
        except Exception as e:
            print(f"  ✗ Error downloading NSL-KDD: {e}")
            return False
        return True
    
    def download_kdd99(self):
        """Download KDD Cup 99 dataset (good baseline)"""
        print("[*] Downloading KDD Cup 99 dataset...")
        kdd_url = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz"
        
        kdd_dir = self.dataset_dir / 'KDD99'
        kdd_dir.mkdir(exist_ok=True)
        
        try:
            filepath = kdd_dir / 'kddcup.data.gz'
            if not filepath.exists():
                print(f"  Downloading kddcup.data.gz (approx 20MB)...")
                urllib.request.urlretrieve(kdd_url, filepath)
                print(f"  ✓ Downloaded, extracting...")
                with gzip.open(filepath, 'rb') as f_in:
                    with open(kdd_dir / 'kddcup.data', 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print(f"  ✓ Extracted successfully")
            else:
                print(f"  kddcup.data already exists")
        except Exception as e:
            print(f"  ✗ Error downloading KDD99: {e}")
            return False
        return True
    
    def create_sample_datasets(self):
        """Create sample synthetic datasets for quick testing"""
        print("[*] Creating sample synthetic datasets...")
        
        sample_dir = self.dataset_dir / 'SAMPLE'
        sample_dir.mkdir(exist_ok=True)
        
        # Create a small sample synthetic IDS dataset
        import numpy as np
        np.random.seed(42)
        
        n_samples = 1000
        n_features = 20
        
        # Generate synthetic network traffic features
        X = np.random.randn(n_samples, n_features)
        # Binary classification: normal (0) vs attack (1)
        y = np.random.binomial(1, 0.3, n_samples)
        
        # Add some patterns to make it realistic
        X[y == 1] += np.random.randn(n_features) * 2
        
        # Create DataFrame
        feature_names = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['label'] = y
        df['attack_type'] = df['label'].apply(lambda x: 'attack' if x == 1 else 'normal')
        
        # Save as CSV
        sample_file = sample_dir / 'synthetic_ids_data.csv'
        df.to_csv(sample_file, index=False)
        print(f"  ✓ Created synthetic dataset: {sample_file.relative_to(self.dataset_dir.parent)}")
        
        # Create a train/test split
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)
        
        train_file = sample_dir / 'synthetic_train.csv'
        test_file = sample_dir / 'synthetic_test.csv'
        
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        print(f"  ✓ Created train/test split")
        
        return True
    
    def list_datasets(self):
        """List available datasets"""
        print("\n[*] Available Datasets:")
        for dataset_dir in self.dataset_dir.iterdir():
            if dataset_dir.is_dir():
                files = list(dataset_dir.glob('*'))
                print(f"  {dataset_dir.name}: {len(files)} files")

if __name__ == "__main__":
    downloader = DatasetDownloader("/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection/datasets")
    
    print("=" * 60)
    print("IDS Dataset Downloader")
    print("=" * 60)
    
    # Download datasets
    downloader.download_kdd99()
    downloader.download_nsl_kdd()
    downloader.create_sample_datasets()
    
    # List all datasets
    downloader.list_datasets()
    
    print("\n✓ Dataset setup complete!")
    print("\nNext steps:")
    print("1. Install requirements: pip install pandas numpy scikit-learn xgboost")
    print("2. Run the preprocessing script to prepare data")
    print("3. Train lightweight models")
