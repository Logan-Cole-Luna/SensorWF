# Lightweight Intrusion Detection for Satellites

Ultra-lightweight machine learning models for network intrusion detection on embedded systems (STM32 and similar microcontrollers).

This repository now supports two IDS paths:
- Network IDS benchmarks (UNSW-NB15, NSL-KDD)
- Benchmark-to-CAN conversion for satellite streaming and embedded CAN IDS training

## Latest Benchmark-to-CAN Results (Phase 4B)

TinyDecisionTree (depth 5), benchmark rows converted into CAN frame streams, then trained via CAN sliding-window features.

| Source | Accuracy | Precision | Recall | F1 | FPR | ROC-AUC | Model Size | Inference |
|-------|----------|-----------|--------|----|-----|---------|------------|-----------|
| UNSW->CAN | 58.52% | 58.74% | 82.87% | 68.75% | 71.33% | 0.5802 | 5.87 KB | 0.02922 ms |
| NSL->CAN  | 65.40% | 89.33% | 44.53% | 59.44% | 7.03%  | 0.7454 | 6.02 KB | 0.02983 ms |

Interpretation for deployment:
- UNSW->CAN favors high recall but is too noisy (very high false positives).
- NSL->CAN is conservative (low FPR) and therefore safer for always-on satellite operation, but misses more attacks.

Recommended on-board streaming dataset right now: NSL->CAN.

## Quick Stats (Phase 2 Real-World Network IDS)

| Model | Size | Latency | Recall | F1-Score | Best For |
|-------|------|---------|--------|----------|----------|
| TinyDecisionTree | 5.08 KB | 0.3 us | 99.43% | 85.23% | RECOMMENDED |
| MicroXGBoost | 9.74 KB | 2.6 us | 99.23% | 82.89% | Balance |
| TinyXGBoost | 6.62 KB | 2.5 us | 99.53% | 81.52% | Speed |
| LightRandomForest | 108.59 KB | 142 us | 99.62% | 85.17% | More resources |
| CompactExtraTrees | 39.91 KB | 147 us | 96.60% | 79.30% | Ensemble | |

## Getting Started

### 1. Setup Environment
```bash
cd intrusion_detection
source ../.venv/bin/activate
pip install pandas numpy scikit-learn xgboost
```

### 2. Run Model Evaluation
```bash
python scripts/train_and_evaluate.py
```

### 3. Convert Benchmark Data to CAN and Train (Phase 4B)
```bash
python scripts/phase4_benchmark_can_ids.py --dataset unsw
python scripts/phase4_benchmark_can_ids.py --dataset nsl
```

Important (quality fix):
- By default, Phase 4B now excludes protocol/meta CAN frames from feature extraction.
- This reduces training on bookkeeping traffic and improves benchmark-to-CAN model stability.
- To include them for ablation only:
```bash
python scripts/phase4_benchmark_can_ids.py --dataset nsl --include-meta-frames
```

Optional row caps (faster iteration):
```bash
python scripts/phase4_benchmark_can_ids.py --dataset nsl --max-train-rows 80000 --max-test-rows 20000
```

### 4. View Results
```bash
cat results/evaluation_results.json
cat results/phase4b_unsw_can_results.json
cat results/phase4b_nsl_can_results.json
```

### 5. Generate Plots
```bash
python scripts/generate_plots.py
```

### 6. Deploy Trained Benchmark-to-CAN Model to Firmware Headers

Generate dataset-specific scaler header:
```bash
python scripts/generate_scaler_header.py \
   --scaler models/trained_models/phase4b_nsl_can_scaler.joblib \
   --features models/trained_models/phase4b_nsl_can_feature_names.json \
   --out models/trained_models/stm32f373_nsl_can_ids_scaler.h
```

Deploy NSL model + scaler into firmware include names used by EmbeddedBabel:
```bash
python scripts/deploy_phase4b_model_to_firmware.py --dataset nsl
```

## Project Structure

```
intrusion_detection/
├── datasets/
│   ├── SAMPLE/                      # Synthetic test data
│   ├── CAN_SATELLITE/               # Synthetic CAN pipeline data
│   └── CAN_FROM_BENCHMARK/          # UNSW/NSL converted CAN datasets
├── models/
│   └── lightweight_ids_models.py    # 5 models defined
├── scripts/
│   ├── train_and_evaluate.py        # Train & benchmark models
│   ├── phase4_benchmark_can_ids.py  # Benchmark->CAN train/eval/export
│   ├── benchmark_to_can.py          # Convert UNSW/NSL to CAN frames
│   ├── deploy_phase4b_model_to_firmware.py
│   ├── generate_plots.py            # Paper/report plots
│   └── host_inference_runner.py     # Stream test features to MCU
├── results/
│   ├── evaluation_results.json
│   ├── phase4b_unsw_can_results.json
│   ├── phase4b_nsl_can_results.json
│   └── plots/
├── INTRUSION_DETECTION_REPORT.md    # Full research report
└── README.md                         # This file
```

## Model Details

### TinyDecisionTree
- **Size:** 5.08 KB (smallest real-world)
- **Speed:** 0.3 microseconds (fastest)
- **Recall:** 99.43% on real attacks
- **Target:** STM32F0+ (8KB+ flash)
- **Use when:** Smallest footprint needed, excellent recall

### MicroXGBoost
- **Size:** 9.74 KB (real-world)
- **Speed:** 2.6 us
- **Recall:** 99.23% (real-world)
- **Target:** STM32L0+ (32KB+ flash)
- **Use when:** Balance of speed, accuracy, and size needed

### LightRandomForest
- **Size:** 11.55 KB
- **Speed:** 1.49 ms
- **Accuracy:** 100%
- **Target:** STM32L4+
- **Use when:** More flash available, ensemble preferred

### CompactExtraTrees
- **Size:** 25.37 KB
- **Speed:** 1.51 ms
- **Accuracy:** 100%
- **Target:** STM32H7 or higher
- **Use when:** Maximum reliability required

### TinyXGBoost
- **Size:** 6.62 KB
- **Speed:** 0.029 ms
- **Accuracy:** 99.58%
- **Use when:** Size critical, ~0.4% accuracy loss acceptable

## Real-World Datasets

For validation with real intrusion detection data:

### UNSW-NB15 (Recommended)
- 49 network features
- 9 attack types
- 2.5M records
- Download: [Kaggle](https://www.kaggle.com/datasets/dhoogla/unswnb15)

```bash
pip install kaggle
kaggle datasets download -d dhoogla/unswnb15
unzip unswnb15.zip -d datasets/UNSW-NB15/
```

### CIC-IDS2017/2018
- Modern network traffic
- Real attack scenarios
- Download: [Kaggle](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)

```bash
kaggle datasets download -d chethuhn/network-intrusion-dataset
unzip network-intrusion-dataset.zip -d datasets/CIC-IDS/
```

### NSL-KDD
- Classical benchmark
- Well-studied
- Download: [Kaggle](https://www.kaggle.com/datasets/hassan06/nslkdd)

```bash
kaggle datasets download -d hassan06/nslkdd
unzip nslkdd.zip -d datasets/NSL-KDD/
```

## STM32 Hardware Requirements

| Series | Flash | RAM | Recommended Model |
|--------|-------|-----|------------------|
| STM32F0 | 8-64 KB | 4-8 KB | TinyDecisionTree |
| STM32L0 | 32-192 KB | 8-20 KB | MicroXGBoost |
| STM32L4 | 64-1024 KB | 64-320 KB | LightRandomForest |
| STM32H7 | 448-2048 KB | 512+ KB | CompactExtraTrees |

## Using the Models

### Train on Custom Data
```python
from models.lightweight_ids_models import get_model
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_csv('your_dataset.csv')
X = df.drop('label', axis=1).values
y = df['label'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Get and train model
model = get_model('xgb')
model.model.fit(X_train, y_train)

# Evaluate
score = model.model.score(X_test, y_test)
print(f"Accuracy: {score:.4f}")
print(f"Model Size: {model.get_model_size():.2f} KB")

# Save
model.save_model('my_ids_model.joblib')
```

### Load Trained Model
```python
import joblib
from models.lightweight_ids_models import LightweightIDS

model_obj = joblib.load('my_ids_model.joblib')
predictions = model_obj.predict(X_test)
```

### Inference on New Data
```python
import numpy as np

# Single sample prediction
x_sample = np.array([[...]])  # Your features
prediction = model.model.predict(x_sample)
confidence = model.model.predict_proba(x_sample)[0]

print(f"Attack detected: {bool(prediction[0])}")
print(f"Confidence: {max(confidence):.4f}")
```

## Performance Characteristics

### CPU Usage
- **Decision Tree:** ~1000 cycles (2 µs @ 48 MHz)
- **XGBoost:** ~2000-5000 cycles (4-10 µs @ 48 MHz)
- **Random Forest:** ~20000 cycles (41 µs @ 48 MHz)

### Memory Usage
- **Flash (Program):** 1.65-25.37 KB
- **RAM (Runtime):** ~1-5 KB
- **Feature Buffer:** Depends on input size

### Power Consumption (Estimated)
At 100 Hz detection rate with STM32L4:
- Inference: ~100 µA·ms = 10 µW
- Total (with 1 ms feature extraction): ~1 mW

## Evaluation Protocol (Recommended)

For each candidate model/dataset pair, evaluate in this order:

1. Offline detection quality
- Primary: FPR, Recall, ROC-AUC
- Secondary: Accuracy, F1

2. On-board suitability
- Model flash size
- Inference latency (ms)
- Stability under streaming workload

3. Operational fit
- For autonomous onboard alarms, prioritize low FPR.
- For forensic/offline review, prioritize high recall.

Current decision for satellite streaming:
- Use NSL->CAN model for primary flight-like run (lower FPR).
- Keep UNSW->CAN as a high-recall secondary comparison/baseline.

## Stream Through Satellite (Host Runner)

Dry-run with benchmark-converted CAN features:
```bash
python scripts/host_inference_runner.py \
   --dry-run \
   --features-csv datasets/CAN_FROM_BENCHMARK/nsl_can_test_features.csv \
   --attack-source-csv datasets/CAN_FROM_BENCHMARK/nsl_can_test.csv \
   --model-path models/trained_models/phase4b_nsl_can_decision_tree.joblib \
   --scaler-path models/trained_models/phase4b_nsl_can_scaler.joblib
```

With hardware connected:
```bash
python scripts/host_inference_runner.py \
   --port /dev/tty.usbmodemXXXX \
   --features-csv datasets/CAN_FROM_BENCHMARK/nsl_can_test_features.csv \
   --attack-source-csv datasets/CAN_FROM_BENCHMARK/nsl_can_test.csv
```

Recommended full run order (satellite inference path):
1. Train NSL->CAN model (meta frames excluded by default):
```bash
python scripts/phase4_benchmark_can_ids.py --dataset nsl
```
2. Generate NSL scaler header:
```bash
python scripts/generate_scaler_header.py \
   --scaler models/trained_models/phase4b_nsl_can_scaler.joblib \
   --features models/trained_models/phase4b_nsl_can_feature_names.json \
   --out models/trained_models/stm32f373_nsl_can_ids_scaler.h
```
3. Deploy active firmware headers:
```bash
python scripts/deploy_phase4b_model_to_firmware.py --dataset nsl
```
4. Build/flash EmbeddedBabel firmware.
5. Stream benchmark-converted test traffic via host runner command above.

## Next Steps

1. **Threshold tuning for NSL->CAN**
   - raise recall while keeping FPR in acceptable mission range
   - add precision-recall threshold sweep in phase4_benchmark_can_ids.py

2. **Feature packetization refinement**
   - test alternative CAN chunk layouts and timing jitter models
   - evaluate impact on sliding-window feature stability

3. **Hybrid dataset strategy**
   - combine NSL->CAN low-FPR profile with selective UNSW->CAN high-recall examples
   - retrain and compare ROC/FPR frontier

4. **End-to-end EmbeddedBabel validation**
   - stream benchmark-converted features through MCU link
   - validate latency/resource reports under sustained run

## Key Findings

- All models fit in STM32 flash memory (5-109 KB)
- Sub-microsecond inference times (0.3-147 us)
- 96-99% attack detection rate on real data
- Can retrain on-device in <1 second
- Requires only 1-5 KB RAM for inference
- Smallest model (TinyDecisionTree) fits in STM32F0

## References

- Research Report: [INTRUSION_DETECTION_REPORT.md](./INTRUSION_DETECTION_REPORT.md)
- Model Code: [models/lightweight_ids_models.py](./models/lightweight_ids_models.py)
- Training Script: [scripts/train_and_evaluate.py](./scripts/train_and_evaluate.py)

## Status

PHASE 4B COMPLETE - Benchmark-to-CAN Pipeline:
- UNSW/NSL tabular datasets converted into CAN frame streams
- CAN feature extraction + TinyDecisionTree training completed
- STM32 headers exported for both UNSW->CAN and NSL->CAN models
- Plotting updated to include benchmark-to-CAN comparisons

---

**Created:** 2026-03-27
**Branch:** intrusion_detection
**Status:** Phase 2 Complete - Real-World Validated
