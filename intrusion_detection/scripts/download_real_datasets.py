#!/usr/bin/env python3
"""
Download real-world intrusion detection datasets from various sources.
Includes NSL-KDD, KDD99, and instructions for Kaggle datasets.
"""

import os
import urllib.request
import gzip
import shutil
import subprocess
from pathlib import Path

class RealDatasetDownloader:
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
    
    def download_nsl_kdd_github(self):
        """Download NSL-KDD from GitHub mirror (alternative method)"""
        print("[*] Attempting to download NSL-KDD dataset...")
        nsl_dir = self.dataset_dir / 'NSL-KDD'
        nsl_dir.mkdir(exist_ok=True)
        
        # Try downloading from GitHub raw content
        files = {
            'KDDTrain+.txt': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt',
            'KDDTest+.txt': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt',
            'KDDTrain+_20Percent.txt': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B_20Percent.txt'
        }
        
        for filename, url in files.items():
            filepath = nsl_dir / filename
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024*1024)
                print(f"  ✓ {filename} already exists ({size_mb:.1f} MB)")
                continue
            
            try:
                print(f"  Downloading {filename}...", end=" ", flush=True)
                urllib.request.urlretrieve(url, filepath)
                size_mb = filepath.stat().st_size / (1024*1024)
                print(f"✓ ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"✗ Error: {e}")
                if filepath.exists():
                    filepath.unlink()
        
        return True
    
    def download_kdd99(self):
        """Download KDD Cup 99 dataset"""
        print("[*] Attempting to download KDD Cup 99 dataset...")
        kdd_dir = self.dataset_dir / 'KDD99'
        kdd_dir.mkdir(exist_ok=True)
        
        urls = {
            'kddcup.data.gz': 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz',
            'kddcup.data_10_percent.gz': 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz'
        }
        
        for filename, url in urls.items():
            filepath = kdd_dir / filename
            output_file = kdd_dir / filename.replace('.gz', '')
            
            if output_file.exists():
                size_mb = output_file.stat().st_size / (1024*1024)
                print(f"  ✓ {output_file.name} already exists ({size_mb:.1f} MB)")
                continue
            
            try:
                print(f"  Downloading {filename} (may take a minute)...", end=" ", flush=True)
                urllib.request.urlretrieve(url, filepath, reporthook=self._download_hook)
                
                print(f"\n    Extracting...", end=" ", flush=True)
                with gzip.open(filepath, 'rb') as f_in:
                    with open(output_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                filepath.unlink()
                size_mb = output_file.stat().st_size / (1024*1024)
                print(f"✓ ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"✗ Error: {e}")
                if filepath.exists():
                    filepath.unlink()
    
    def download_unsw_nb15_instructions(self):
        """Print instructions for downloading UNSW-NB15"""
        print("[*] UNSW-NB15 Dataset Download Instructions")
        print("  This dataset is available through:")
        print("  1. Official UNSW repository:")
        print("     https://research.unsw.edu.au/projects/unsw-nb15-dataset")
        print("  2. Kaggle (faster, no registration):")
        print("     https://www.kaggle.com/datasets/dhoogla/unswnb15")
        print("")
        print("  To download from Kaggle using CLI:")
        print("    # Install: pip install kaggle")
        print("    # Setup: Configure ~/.kaggle/kaggle.json with API token")
        print("    # Download: kaggle datasets download -d dhoogla/unswnb15")
        print("")
        print("  Files will be automatically extracted to: datasets/UNSW-NB15/")
    
    def download_cicids_instructions(self):
        """Print instructions for CIC-IDS2017/2018"""
        print("[*] CIC-IDS2017/2018 Dataset Download Instructions")
        print("  Available from Canadian Institute for Cybersecurity:")
        print("  https://www.unb.ca/cic/datasets/")
        print("")
        print("  Direct download (if still available):")
        print("    CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html")
        print("    CIC-IDS2018: https://www.unb.ca/cic/datasets/ids-2018.html")
        print("")
        print("  Also available on Kaggle:")
        print("    https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset")
    
    def _download_hook(self, block_num, block_size, total_size):
        """Progress hook for downloads"""
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded / total_size) * 100)
            if percent % 25 == 0 and downloaded % (block_size * 4) == 0:
                mb = downloaded / (1024*1024)
                total_mb = total_size / (1024*1024)
                print(f"{percent:.0f}%...", end=" ", flush=True)
    
    def check_kaggle_api(self):
        """Check if Kaggle API is installed"""
        try:
            import kaggle
            return True
        except ImportError:
            return False
    
    def print_summary(self):
        """Print available datasets"""
        print("\n" + "=" * 70)
        print("AVAILABLE INTRUSION DETECTION DATASETS")
        print("=" * 70)
        
        for dataset_dir in sorted(self.dataset_dir.iterdir()):
            if dataset_dir.is_dir():
                files = list(dataset_dir.glob('*'))
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                size_mb = total_size / (1024*1024)
                print(f"\n{dataset_dir.name}/")
                print(f"  Files: {len(files)}")
                if size_mb > 0:
                    print(f"  Total Size: {size_mb:.1f} MB")
                for f in files[:5]:  # Show first 5 files
                    f_size = f.stat().st_size / (1024*1024)
                    print(f"    - {f.name} ({f_size:.1f} MB)")
                if len(files) > 5:
                    print(f"    ... and {len(files)-5} more files")

if __name__ == "__main__":
    dataset_dir = "/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection/datasets"
    downloader = RealDatasetDownloader(dataset_dir)
    
    print("=" * 70)
    print("Real-World Intrusion Detection Dataset Downloader")
    print("=" * 70)
    print("")
    
    # Try to download freely available datasets
    downloader.download_nsl_kdd_github()
    downloader.download_kdd99()
    
    # Print instructions for other datasets
    print("")
    downloader.download_unsw_nb15_instructions()
    print("")
    downloader.download_cicids_instructions()
    
    # Check for Kaggle API
    print("\n[*] Kaggle API Status:")
    if downloader.check_kaggle_api():
        print("  ✓ Kaggle API is installed")
        print("  You can download datasets directly from command line")
    else:
        print("  ✗ Kaggle API not installed")
        print("  Install with: pip install kaggle")
    
    # Print summary
    downloader.print_summary()
    
    print("\n" + "=" * 70)
    print("Dataset Download Status Complete")
    print("=" * 70)
    print("\nNote: Some datasets require manual download or Kaggle API authentication")
    print("See instructions above for downloading UNSW-NB15 and CIC-IDS datasets")
