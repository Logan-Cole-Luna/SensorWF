# CubeSat Onboard IDS: Lightweight Models and Rule-Based Approaches

A reference guide for intrusion detection system design targeting STM32-class (ARM Cortex-M) hardware, where compute budget must be shared with primary satellite functions.

---

## ML-Based IDS Models (TinyML-Compatible)

### Tier 1: Genuinely Microcontroller-Viable
*Fits STM32F4 class and below, under 512KB flash*

#### 1. Random Forest / Extra Trees via emlearn

Train in scikit-learn, export to a pure C header with no runtime needed and no dynamic allocation. This is the most battle-tested path for MCU deployment. Supports anomaly detection via `GaussianMixture` and `EllipticEnvelope` as well. Inference times on STM32-class hardware are typically under 1ms for small forests.

- Repo: https://github.com/emlearn/emlearn
- Targets: ARM Cortex-M (including STM32), AVR8, RISC-V

#### 2. One-Class SVM (Novelty / Anomaly Detection) via micromlgen

Trains a boundary around "normal" behavior, which is well suited for unsupervised onboard IDS where all attack types cannot be enumerated ahead of time. Exports scikit-learn `OneClassSVM` directly to C++ code for Arduino/STM32. Memory footprint is very low when using 10-20 engineered features from telemetry.

- Repo: https://github.com/eloquentarduino/micromlgen

#### 3. Quantized Autoencoder (QAE)

Reconstruction-error-based anomaly detection, quantized to INT8 (QAE-u8) or FP16 (QAE-f16). Train on nominal satellite behavior and flag high reconstruction error as a potential intrusion. Uses pruning, clustering, and integer quantization to make autoencoders deployable on resource-constrained edge devices. Deployable via TensorFlow Lite for Microcontrollers (TFLM) or STM32Cube.AI.

- Reference paper: Sharmila & Nagapadma, *Cybersecurity* 2023
  https://cybersecurity.springeropen.com/articles/10.1186/s42400-023-00178-5

#### 4. Decision Tree (Pruned)

Single decision trees can be extremely compact. A depth-8 tree trained on telemetry features takes only kilobytes of flash. Export via micromlgen or emlearn. Fast inference (microseconds), fully deterministic, and easy to audit for space qualification. A good baseline to compare against more complex models.

#### 5. XGBoost (Pruned and Quantized)

An optimized XGBoost model enhanced with pruning, quantization, and feature selection has been demonstrated as a TinyML IDS on edge devices with minimal computational overhead. Larger than a single decision tree but significantly more accurate, and still fits on STM32H7-class hardware.

- Conversion tools: https://github.com/dmlc/treelite or `m2cgen` library

---

### Tier 2: Requires a More Capable MCU
*STM32H7, Cortex-M7, or an OBC with dedicated flash*

#### 6. Small 1D-CNN (TFLite for Microcontrollers)

A quantized 1D-CNN can serve as a lightweight spatial feature extractor to identify intrusions in real time. A 3-4 layer 1D-CNN over sliding windows of telemetry runs well on STM32H7 with STM32Cube.AI. STM32Cube.AI will profile memory and compute usage before committing to hardware.

- Framework: https://github.com/tensorflow/tflite-micro
- ST tooling: https://www.st.com/en/embedded-software/x-cube-ai.html

#### 7. Shallow MLP Autoencoder (TFLM)

A 3-layer fully-connected autoencoder (for example, 23 to 8 to 23 features, INT8 quantized) fits on STM32F7/H7. All ops run in INT8 after quantization. Silicon Labs' MLTK toolkit provides a ready reference implementation.

- Reference: https://siliconlabs.github.io/mltk/docs/python_api/models/tinyml/anomaly_detection.html

#### 8. Isolation Forest (iForest)

A tree-ensemble anomaly detector that works without labeled attack data. Well suited for detecting command flooding or storage exhaustion attacks, which are precisely the adversarial scenarios present in the CuCD-ID dataset (see below). Can be exported to C via emlearn or custom serialization.

- TinyML-targeted paper: TiWS-iForest (Isolation Forest in Weakly Supervised and TinyML scenarios)

---

## Relevant Dataset: CubeSat-Specific

### CuCD-ID (CubeSat Cybersecurity Dataset for Intrusion Detection)

This dataset contains labeled command and telemetry data generated using NASA's NOS3/cFS simulator. It covers five scenarios: one nominal case and four adversarial tactics from the SPARTA framework, specifically command flooding, false data injection, storage exhaustion, and defence impairment. Contains 25,000 records with 31 features in the raw version, and an augmented noised version with 22,465 records and 23 features.

- Published: February 2026
- Source: https://www.sciencedirect.com/science/article/pii/S2352340926001514

This is the most directly relevant training and evaluation dataset available for this application.

---

## TinyML Deployment Frameworks

| Framework | Best For | MCU Support |
|---|---|---|
| emlearn | scikit-learn to C header, zero runtime | Any C-capable MCU including STM32 |
| TFLite Micro (TFLM) | Keras/TF neural network models | ARM Cortex-M, RISC-V, ESP32 |
| STM32Cube.AI | ST-specific optimization and profiling | STM32 family only |
| micromlgen | SVM, DT, RF to Arduino C++ | STM32, Arduino |
| NanoEdge AI Studio | ST drag-and-drop anomaly detection tool | STM32 only |
| EloquentTinyML | TFLM wrapper for Arduino and STM32 | STM32, ESP32 |

---

## Rule-Based Methods (Comparison Baseline)

These tools do not run directly on the STM32 but are valuable for ground station processing, dataset generation, and as a reference for detection logic that can be distilled into simpler onboard rules.

### Snort 3

The gold standard signature-based IDS. Not MCU-runnable (requires Linux), but useful for generating ground-truth labeled datasets and as a reference for detection logic that can be simplified into onboard threshold rules.

- https://www.snort.org
- https://github.com/snort3/snort3

### Suricata

Developed by the Open Information Security Foundation (OISF). Uses a rule set to specify suspicious activity within packet payloads, targeting known attack signatures and protocol vulnerabilities. More flexible than Snort with multi-threading and deeper application-layer protocol inspection. Can consume Snort rules with minor adjustments.

- https://github.com/OISF/suricata

### Zeek (formerly Bro)

Script-based anomaly-and-signature hybrid. Rather than traditional IDS signatures, Zeek uses scripts to analyze traffic, which allows for highly automated workflows and decisions more granular than simple pass/drop logic. Best suited for telemetry analysis at the ground station.

- https://github.com/zeek/zeek

### OSSEC (Host-Based)

Log and file integrity monitoring. If the OBC runs embedded Linux (Raspberry Pi-class or similar), OSSEC can monitor file checksums and log anomalies with low overhead.

- https://github.com/ossec/ossec-hids

### Custom Threshold and State-Machine Rules (MCU-viable)

For onboard use, the most practical rule-based approach is a hand-coded state machine in C that monitors the following:

- Command rate (flood detection)
- Unexpected subsystem state transitions
- Telemetry value bounds (data injection detection)
- Uplink and downlink frequency anomalies

This approach runs in under 1KB of flash with negligible CPU overhead and is the most common method found in actual flight software today. It also works effectively as a first filter layer before invoking any ML model, avoiding unnecessary compute on clearly nominal traffic.

---

## Recommended Architecture for a CubeSat IDS

Given typical CubeSat compute constraints, a two-layer design is practical and well-supported by existing research.

### Layer 1: Always-On Rules (MCU, C state machine)

Lightweight threshold and state-machine rules implemented directly in C. Zero ML overhead. Catches blatant attacks such as command flooding and out-of-bounds telemetry injection with microsecond latency. Runs continuously without impacting primary mission compute.

### Layer 2: Periodic ML Inference (MCU, triggered)

A quantized Random Forest or One-Class SVM deployed via emlearn or micromlgen. Activated on a periodic schedule (for example, every N seconds) or when Layer 1 raises a soft flag. Catches subtle behavioral anomalies that threshold rules would miss.

The CuCD-ID dataset is the recommended starting point for training both layers, as it was purpose-built for CubeSat intrusion detection using realistic NASA simulator telemetry.

---

## Hugging Face and GitHub Resources Summary

| Resource | Type | Link |
|---|---|---|
| emlearn | GitHub - MCU ML inference engine | https://github.com/emlearn/emlearn |
| micromlgen | GitHub - sklearn to C++ for MCU | https://github.com/eloquentarduino/micromlgen |
| EloquentTinyML | GitHub - TFLM wrapper | https://github.com/eloquentarduino/EloquentTinyML |
| tinyml-example-anomaly-detection | GitHub - TinyML anomaly detection demo | https://github.com/ShawnHymel/tinyml-example-anomaly-detection |
| Anomaly-Detection-using-TinyML | GitHub - TFLM IDS for Arduino | https://github.com/DeepthiSudharsan/Anomaly-Detection-using-TinyML |
| embeddedml | GitHub - Embedded ML notes and resources | https://github.com/jonnor/embeddedml |
| awesome-tinyml | GitHub - Curated TinyML resource list | https://github.com/gauravfs-14/awesome-tinyml |
| tinyml-papers-and-projects | GitHub - TinyML paper and project list | https://github.com/gigwegbe/tinyml-papers-and-projects |
| snort3 | GitHub - Snort IDS/IPS | https://github.com/snort3/snort3 |
| suricata | GitHub - Suricata IDS/IPS | https://github.com/OISF/suricata |
| zeek | GitHub - Zeek network analysis | https://github.com/zeek/zeek |
| ossec-hids | GitHub - Host-based IDS | https://github.com/ossec/ossec-hids |
| dalton | GitHub - Snort/Suricata rule testing | https://github.com/secureworks/dalton |