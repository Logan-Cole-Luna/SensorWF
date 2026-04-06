# Intrusion Detection Project Memory

Last updated: 2026-04-02

## 1) Scope and evolution

This project started as phase-based intrusion detection work using a lightweight decision tree, then expanded into:
- real-world benchmark validation
- quantization and C header export for MCU deployment
- CAN-native IDS dataset/model pipeline
- host + firmware benchmarking for resource and energy estimation
- STM32 target correction from H7 assumptions to STM32F373C8T
- EmbeddedBabel project integration using existing hooks

Primary direction at the end of current work: integrate and run IDS inference through EmbeddedBabel hooks without requiring missing UART HAL driver files.

## 2) Completed phases and key outcomes

### Phase 2: real-world validation (UNSW-NB15)
- Script: scripts/phase2_realworld_validation.py
- Model: TinyDecisionTree (depth 5)
- Outcome: strong recall but elevated false positive rate on threshold sweep.

### Phase 3: quantization/compact deployment
- Script: scripts/phase3_quantization.py
- Outputs: float32/int16/int8 compact tree formats and STM32-ready headers.
- Outcome:
  - compact float32 retained performance with significantly lower footprint
  - int16 showed moderate degradation
  - int8 showed substantial degradation
- Fix applied: brittle output path print bug (parents index) removed.

### Phase 2b: canonical benchmark validation (NSL-KDD)
- Script: scripts/phase2b_nslkdd_validation.py
- Outcome: cross-benchmark result set added to strengthen external validity narrative.

### Phase 4: CAN-based IDS pivot
- Scripts:
  - scripts/can_feature_engineering.py
  - scripts/can_dataset_generator.py
  - scripts/phase4_can_ids.py
- Outcome: synthetic satellite CAN dataset + TinyDecisionTree CAN IDS + generated headers for embedded use.
- Fix applied: replay attack injection bounds bug in CAN dataset generation.

## 3) On-board benchmarking framework

### Added tooling
- scripts/generate_scaler_header.py
- scripts/host_inference_runner.py
- firmware/stm32h7_can_ids_benchmark.c/.h (initial)
- firmware/stm32f373_can_ids_benchmark.c/.h (retargeted)

### Measurement approach
- model-focused timing/resource estimation path
- DWT cycle timing
- stack canary usage
- duty-cycle and energy estimation in host reports
- dry-run mode support for report generation without hardware execution

### Host runner stabilization
Multiple fixes were made for:
- struct packing/unpacking alignment
- field naming consistency
- report key mismatches
- dry-run summary/report output robustness

## 4) Target/platform decisions

### RTOS
- Decision: no FreeRTOS requirement for benchmark path.
- Bare-metal approach used instead.

### MCU correction
- Correct hardware target: STM32F373C8T.
- Constants, assumptions, and naming updated accordingly in benchmark and host-side logic.

### EmbeddedBabel integration constraints
- EmbeddedBabel HAL package does not include UART HAL driver files (e.g., stm32f3xx_hal_uart.h).
- UART-dependent benchmark integration inside EmbeddedBabel was therefore not viable without adding missing drivers.
- Integration direction changed to avoid UART dependency and use existing hook-based inference path.

## 5) EmbeddedBabel integration state (latest)

### Header availability for hook path
- Present in intrusion_detection/firmware:
  - stm32h7_can_ids_float32.h
  - stm32h7_can_ids_scaler.h

### Hook wiring decisions
- main loop calls nn_runner() via runner.h.
- nn_runner implementation moved to EmbeddedBabel source hook file.
- Duplicate nn_runner definition removed from main.c to avoid symbol conflicts.

### Build path fix
- EmbeddedBabel .cproject updated so source/ is included as a source path in Debug and Release.
- This ensures source/runner.c is compiled, not just indexed.

### Current diagnostic caveat
- Editor diagnostics still reported cannot open stm32f3xx_hal_uart.h from main translation unit at one point.
- HAL UART module define is commented in Core/Inc/stm32f3xx_hal_conf.h.
- This appeared to be stale index/include-path state vs current intended configuration.
- Recommended validation: clean/reindex + full Debug/Release builds in STM32CubeIDE.

## 6) Important scripts and outputs

### Key scripts (intrusion_detection/scripts)
- phase2_realworld_validation.py
- phase3_quantization.py
- phase2b_nslkdd_validation.py
- can_feature_engineering.py
- can_dataset_generator.py
- phase4_can_ids.py
- generate_scaler_header.py
- host_inference_runner.py

### Key model headers (intrusion_detection/models/trained_models and firmware)
- stm32_tree_float32.h
- stm32_tree_int16.h
- stm32_tree_int8.h
- stm32_tree_nslkdd_float32.h
- stm32_tree_nslkdd_int16.h
- stm32h7_can_ids_float32.h
- stm32h7_can_ids_int16.h
- stm32h7_can_ids_scaler.h

### Key result artifacts (intrusion_detection/results)
- phase2_results.json
- phase3_results.json
- phase2b_nslkdd_results.json
- phase4_can_results.json
- benchmark report/json/csv outputs for stm32h7 and stm32f373 runs

### Report
- INTRUSION_DETECTION_REPORT.md was updated across phases to capture results and narrative.

## 7) Known technical risks / open items

1. EmbeddedBabel compile validation still required after latest hook/source-path adjustments.
2. Need to verify include resolution and generated make artifacts in both Debug and Release builds.
3. Current nn_runner path uses placeholder features; real CAN frame to feature-window extraction pipeline should be connected for true runtime IDS behavior.
4. Ensure no stale benchmark include/dependency remains in EmbeddedBabel generated files.

## 8) Suggested immediate validation checklist

1. Clean project in STM32CubeIDE for Debug and Release.
2. Rebuild both configs and confirm source/runner.c is compiled.
3. Confirm no duplicate nn_runner symbols at link time.
4. Confirm model headers resolve from configured include paths.
5. Run target/hardware smoke test: nn_runner executes ids_scale_features + ids_predict path.

## 9) Practical notes

- Use project venv python when running intrusion_detection scripts.
- Dataset downloader had SSL issues in one flow; manual download was previously used as fallback.
- Data folder telemetry logs do not provide explicit CPU/heap metrics; only power and subsystem telemetry proxies are available there.
