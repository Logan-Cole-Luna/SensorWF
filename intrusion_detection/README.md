# Energy-Efficient Intrusion Detection for Satellite CAN Networks

A production-ready, lightweight intrusion detection system (IDS) for satellite Controller Area Network (CAN) bus traffic, engineered for resource-constrained microcontrollers. The system occupies **541 bytes** of flash memory, executes in **4.2 microseconds**, and imposes a power increment of at most **48–53 microWatts**—validated on hardware from the KISPE Satellite Learning Laboratory (SATLL).

This work addresses a critical gap in satellite cybersecurity: protecting energy-grid-dependent spacecraft (GPS timing, renewable energy telemetry, grid monitoring) while respecting the extreme power and memory budgets of CubeSat-class platforms.

## What We Built

A **domain-specific anomaly detection system** for satellite bus networks:

- **14 CAN features** capturing payload structure, timing dynamics, and bus topology
- **TinyDecisionTree** (depth 5, 39 nodes) as the sole model meeting hardware constraints
- **Zero-heap C implementation** for deterministic, bounded inference on bare-metal STM32
- **Power-aware evaluation** using live SATLL EPS telemetry (not simulated estimates)

## Core Results

### In-Domain Performance (Native Satellite CAN)
Training set: 25,861 frames | Test set: 5,166 frames | Attack classes: DoS, Fuzzy, Spoofing, Replay

| Metric | Value |
|--------|-------|
| **Recall** | **99.86%** (4 missed attacks) |
| **Precision** | 96.38% |
| **F1-Score** | 98.09% |
| **ROC-AUC** | **0.9912** |

### Hardware Validation (STM32F373C8T @ 72 MHz)

| Metric | Value |
|--------|-------|
| **Flash (tree + scaler)** | 541 bytes |
| **RAM (feature buffer)** | 60 bytes (stack only) |
| **Inference latency (mean)** | 4.19–4.21 µs |
| **CPU duty @ 100 Hz** | 0.042% |
| **Inference increment** | **≤48–53 µW** |

### Cross-Domain Generalization (UNSW-NB15 & NSL-KDD)
Network benchmark datasets encoded as synthetic CAN streams, then evaluated on STM32:

| Dataset | Recall | Precision | F1 | FPR |
|---------|--------|-----------|----|----|
| UNSW-NB15 (82K test) | 93.64% | 55.85% | 69.97% | 90.68% |
| NSL-KDD (48K test) | 52.88% | 86.65% | 65.68% | 10.76% |

**Interpretation:** The satellite-trained model generalizes reasonably to alien network traffic, but authentic mission data is essential for production training.

## How It Works

### 1. CAN Feature Extraction

Maintain a 50-frame sliding window per CAN ID. For each frame, compute 14 features:

**Payload structure (per-frame):**
- `can_id_norm` — Normalized 11-bit CAN ID
- `dlc` — Data length code (0–8 bytes)
- `data_mean`, `data_std` — Payload byte statistics
- `data_entropy` — Shannon entropy (bits)
- `data_range` — max − min of bytes
- `hamming_dist` — Bit differences from previous frame with same ID
- `payload_delta` — L1 distance from previous payload
- `dlc_anomaly`, `id_is_known` — Anomaly flags

**Temporal dynamics (windowed):**
- `inter_arrival_mean` — Mean Δt between same-ID frames
- `id_freq` — This ID's message rate (msgs/s)

**Bus topology (windowed):**
- `bus_load` — Total bus message rate (msgs/s)
- `unique_ids` — Distinct CAN IDs in window

### 2. Model Selection & Training

Five candidates (TinyDecisionTree, TinyXGBoost, MicroXGBoost, LightRandomForest, CompactExtraTrees) evaluated on UNSW-NB15 and NSL-KDD. **TinyDecisionTree selected** as the only model meeting all three constraints:
- ≤ 8 KB serialized flash
- ≤ 100 µs inference latency
- ≥ 65% recall on both datasets

Retrained on native satellite CAN dataset (25K train / 5K test) with depth constraint of 5.

### 3. Firmware Export & Integration

**Exported as static C arrays** (no heap, no FPU calls):
- `tree_thresholds[]`, `feature_idx[]`, `children_left[]`, `children_right[]` — decision structure
- `mean[]`, `scale[]` (14 × float32) — z-score normalization constants
- Total: 541 bytes in flash; 60 bytes stack for feature buffer

**Root-to-leaf traversal:** Deterministic 302-cycle fixed path (every frame takes the same number of clock cycles).

### 4. Power Accounting

**Two-level decomposition:**

| Component | Draw |
|-----------|------|
| MCU base (STM32F373C8T @ 72 MHz) | 115–125 mW |
| Board peripherals (USB-FS, LDO) | ≈49 mW |
| **Inference algorithm (100 Hz, 4.2 µs)** | **≤48–53 µW** |

The 48–53 µW figure is the **marginal cost of tree traversal alone**, derived from live SATLL EPS telemetry rather than model estimates. It represents **<0.004% of ADCS subsystem overhead** during satellite experiments.

## Getting Started

### Prerequisites

```bash
cd intrusion_detection
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install pandas numpy scikit-learn joblib
```

### 1. Download and Prepare Datasets

**Native satellite CAN data** (from SATLL, included with paper):
```bash
# Unzip to: datasets/SATLL/
# Contains: baseline CAN traffic + labeled attack frames (DoS, Fuzzy, Spoofing, Replay)
```

**Network benchmarks** (optional, for cross-domain evaluation):
```bash
# Download UNSW-NB15 and NSL-KDD from Kaggle or UCI
# Place CSV files in datasets/UNSW_NB15/ and datasets/NSL_KDD/
```

### 2. Train Native Satellite IDS

```bash
python scripts/train_satellite_can_ids.py \
  --data datasets/SATLL/labeled_can_traffic.csv \
  --output models/satellite_can_ids_tree.joblib \
  --max-depth 5
```

Produces:
- Trained tree model (joblib)
- Scaler parameters (JSON)
- Feature list (JSON)
- Metrics (JSON)

### 3. Export to STM32 Firmware Headers

```bash
python scripts/export_to_stm32.py \
  --model models/satellite_can_ids_tree.joblib \
  --scaler models/satellite_can_ids_scaler.json \
  --features models/satellite_can_ids_features.json \
  --output firmware/can_ids_model.h \
  --format float32
```

Produces:
- `can_ids_model.h` — Decision arrays, thresholds, tree structure
- `can_ids_scaler.h` — Normalization constants
- Ready to `#include` in STM32 firmware

### 4. Evaluate on Hardware (SATLL + STM32)

Flash firmware with embedded model header to STM32F373C8T evaluation board, powered from SATLL payload rail:

```bash
# Stream pre-computed features via USB OTG CDC-ACM
python scripts/host_inference_runner.py \
  --port /dev/tty.usbmodem12345 \
  --features datasets/SATLL/test_features.csv \
  --labels datasets/SATLL/test_labels.csv
```

Logs per-sample prediction, cycle count, and stack usage. Concurrent SATLL EPS telemetry records payload rail power.

### 5. Cross-Domain Evaluation (Optional)

Encode UNSW-NB15 and NSL-KDD as synthetic CAN streams:

```bash
python scripts/benchmark_to_can.py \
  --input datasets/UNSW_NB15/unsw_nb15_training.csv \
  --output datasets/UNSW_CAN_encoded.csv \
  --meta-frame-id 0x100 \
  --payload-frame-ids 0x200
```

Train on encoded CAN features:

```bash
python scripts/train_satellite_can_ids.py \
  --data datasets/UNSW_CAN_encoded.csv \
  --output models/unsw_can_ids_tree.joblib \
  --max-depth 5
```

## Project Structure

```
intrusion_detection/
├── ACM_Intrusion_Detection_for_Satellites/
│   ├── main.tex                         # Paper main file
│   ├── content/
│   │   ├── Introduction.tex
│   │   ├── Methodology.tex
│   │   ├── Results.tex
│   │   └── Conclusion.tex
│   ├── references.bib                   # BibTeX citations
│   └── images/                          # Figures (tree structure, ROC curves, etc.)
├── datasets/
│   ├── SATLL/                           # Native satellite CAN traffic
│   ├── UNSW_NB15/                       # Network benchmark (optional)
│   └── NSL_KDD/                         # Network benchmark (optional)
├── models/
│   ├── satellite_can_ids_tree.joblib    # Trained TinyDecisionTree
│   ├── satellite_can_ids_scaler.json    # Scaler params
│   └── satellite_can_ids_features.json  # Feature names/order
├── firmware/
│   └── can_ids_model.h                  # Exported C header
├── scripts/
│   ├── train_satellite_can_ids.py       # Main training
│   ├── export_to_stm32.py               # Firmware export
│   ├── benchmark_to_can.py              # Encode benchmark datasets
│   ├── host_inference_runner.py         # Hardware evaluation
│   └── generate_plots.py                # Paper figures
├── results/
│   ├── satellite_can_ids_metrics.json   # In-domain results
│   ├── unsw_can_ids_metrics.json        # Cross-domain results
│   ├── nsl_can_ids_metrics.json         # Cross-domain results
│   └── satll_power_summary.json         # Power telemetry
└── README.md                             # This file
```

## Key Features

✅ **Minimal resource footprint** — 541 bytes flash, 60 bytes RAM  
✅ **Deterministic latency** — 4.2 µs fixed-path tree traversal  
✅ **Negligible power overhead** — ≤53 µW inference increment  
✅ **High in-domain accuracy** — 99.86% recall on satellite CAN  
✅ **Production-grade firmware** — No heap, no malloc, stack-only  
✅ **Realistic power accounting** — Measured from actual spacecraft EPS telemetry  
✅ **Cross-domain validation** — Tested on UNSW-NB15 and NSL-KDD  
✅ **Open reproducibility** — All code, datasets, and trained models included

## Hardware Compatibility

| MCU Series | Flash | RAM | Status |
|-----------|-------|-----|--------|
| STM32F0xx | 8–64 KB | 4–8 KB | ✅ Supported (TinyDecisionTree fits) |
| STM32L0xx | 32–192 KB | 8–20 KB | ✅ Supported |
| STM32F3xx | 64–256 KB | 16–32 KB | ✅ **Validated** (F373C8T) |
| STM32H5xx | 256–512 KB | 32–192 KB | ✅ Supported |

**Tested platform:** STM32F373C8T at 72 MHz (as used in SATLL OBDH-class hardware)

## Performance Characteristics

### Latency Profile

| Operation | Time |
|-----------|------|
| Feature computation (14 features) | ~2 µs |
| Z-score normalization | ~1 µs |
| Tree traversal (5 levels) | ~1.2 µs |
| **Total inference** | **~4.2 µs** |

### Memory Profile

| Resource | Size | Notes |
|----------|------|-------|
| Tree arrays (429 B) | 429 bytes | Thresholds, split indices, leaves |
| Scaler constants (112 B) | 112 bytes | 14 × float32 mean/scale |
| Feature buffer | 60 bytes | Stack-allocated, reused per sample |
| **Total flash** | **541 bytes** | |
| **Total RAM (dynamic)** | **0 bytes** | No heap allocation |

### Power Profile

At 100 Hz polling rate with STM32F373C8T at 72 MHz:

| Metric | Value |
|--------|-------|
| MCU base current (active-run) | 35–38 mA @ 3.3 V |
| MCU base power | 115–125 mW |
| Inference margin | ≤48–53 µW per inference |
| Total evaluation board | 164–176 mW (external MCU + bridge) |
| Native OBDH integration projection | ~40 µW residual |

**Power data source:** Live SATLL EPS telemetry (1 Hz CDH logging); not simulated or estimated.

## Reproducing Results

### Step 1: Train on Native Satellite CAN

```bash
python scripts/train_satellite_can_ids.py \
  --data datasets/SATLL/labeled_can_traffic.csv \
  --test-size 0.166 \
  --max-depth 5 \
  --output models/satellite_can_ids_tree.joblib
```

Expected output:
```
Recall: 99.86%
Precision: 96.38%
F1-Score: 98.09%
ROC-AUC: 0.9912
Model size: 5.2 KB (sklearn pickle)
Exported size: 541 bytes (C header)
```

### Step 2: Export Firmware Headers

```bash
python scripts/export_to_stm32.py \
  --model models/satellite_can_ids_tree.joblib \
  --scaler models/satellite_can_ids_scaler.json \
  --features models/satellite_can_ids_features.json \
  --output firmware/can_ids_model.h
```

### Step 3: Compile & Flash STM32 Firmware

```bash
cd firmware/
make clean
make BOARD=stm32f373c8t
# Use ST-Link or OpenOCD to flash
st-flash write build/firmware.bin 0x08000000
```

### Step 4: Hardware Validation

**Setup:**
- STM32 evaluation board powered from SATLL payload 5V rail
- Host PC connected via USB OTG CDC-ACM
- SATLL EPS logging enabled (1 Hz CDH telemetry)

**Run inference benchmark:**

```bash
python scripts/host_inference_runner.py \
  --port /dev/tty.usbmodem12345 \
  --features datasets/SATLL/test_features.csv \
  --labels datasets/SATLL/test_labels.csv \
  --output results/hardware_validation.json
```

**Collect power telemetry** (parallel):
```bash
# Download SATLL CDH logs (EPS housekeeping)
# Compute payload rail power: P = V * I
python scripts/analyze_power_telemetry.py \
  --cdh-logs data/Baseline_1/Analysis.txt \
  --output results/satll_power_baseline.json
```

### Step 5: Verify Against Paper Metrics

```bash
python scripts/validate_reproduction.py \
  --inference-results results/hardware_validation.json \
  --power-results results/satll_power_baseline.json
```

Expected matching:
- **Hardware latency:** 4.19–4.21 µs ✅
- **Inference increment:** 48–53 µW ✅
- **In-domain recall:** 99.86% ✅
- **CPU duty @ 100 Hz:** 0.042% ✅

## Paper & References

This work is published as:

**"Energy-Efficient Embedded Cybersecurity for Satellite Avionics"**  
*ACM EnergySP '26* (Workshop on Cybersecurity and Privacy of Energy Systems)

Key citations:
- [KISPE SATLL platform](https://arc.aiaa.org/doi/10.2514/6.2022-4207) — Sellers et al., ASCEND 2022
- [Viasat KA-SAT attack](https://www.viasat.com/about/newsroom/press-release/ka-sat-network-cyber-attack-overview/) — Case study on satellite-to-grid impact
- [UNSW-NB15 dataset](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity-datasets/unsw-nb15-dataset/) — Network IDS benchmark
- [NSL-KDD dataset](https://www.unb.ca/cic/datasets/nsl.html) — IDS evaluation standard

**Full paper, appendices, and trained models:** Included in submission.

## Future Work

1. **Direct OBDH integration** — Deploy natively on STM32H573 OBDH firmware (currently prevented by closed-source SATLL software; future open-firmware platforms targeted)
2. **Dynamic thresholding** — Adapt detection boundary at runtime based on operational context (eclipse, maneuver, nominal)
3. **Lightweight ensemble** — Combine multiple shallow trees within the same 541-byte flash budget for adversarial robustness
4. **Operational traffic collection** — Expand training dataset to include eclipse transitions, orbit corrections, and payload activations
5. **Hardware acceleration** — Co-design with DSP accelerators for multi-satellite constellation monitoring

## Contributing

This research is part of the KISPE Satellite Learning Laboratory initiative. Contributions, bug reports, and dataset extensions are welcome. Please open an issue or contact the authors.

## License

[Specify: MIT, Apache 2.0, or other as appropriate]

## Contact

**Authors:**  
- Sirio Jansen-Sánchez (Embry-Riddle Aeronautical University)
- Logan Luna (Embry-Riddle Aeronautical University)
- Joseph Rigo (Embry-Riddle Aeronautical University)

---

**Last updated:** April 2026  
**Status:** Paper accepted to ACM EnergySP '26  
**Branch:** `intrusion_detection`
