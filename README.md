# SensorWF — Generalizable FAIR Sensor Analytics Workflow

A reproducible, FAIR-compliant workflow framework for multi-domain scientific sensor time-series analysis. The core framework (M1–M5) is domain-agnostic and reusable across any sensor domain via a pluggable adapter pattern. This repository also includes an anomaly detection use case (extensions E1–E2) validated on three domains.

**Submitted to:** IEEE eScience 2026, Naples, September 28–October 2, 2026.

---

## Architecture

```
Core framework (M1–M5)              Use-case extensions (E1–E2)
─────────────────────────           ────────────────────────────
M1  DataIngestion                   E1  FaultInjection
M2  QualityAssessment          →    E2  AnomalyDetection
M3  FeatureEngineering
M4  SemanticAnnotation
M5  ProvenanceExport
```

The five core modules are reusable across any sensor domain. Extensions E1 and E2 implement synthetic fault injection and ML-based anomaly detection for demonstration purposes — they are not part of the reusable core and are run separately via `use_cases/`.

---

## Validated Domains

| Domain | Dataset | Sessions | Best AUC-ROC / AUC-PR |
|--------|---------|----------|----------------------|
| Satellite telemetry | KISPE SATLL (4 families) | 4 | AE = 0.704 / 0.498 |
| Biomedical ECG | MIT-BIH Arrhythmia DB (20 records) | 20 | LOF = 0.884 / 0.788 |
| Atmospheric climate | Jena Climate 2009–2015 (7 years) | 7 | AE = 0.801 / 0.658 |

---

## Quick Start

### Prerequisites

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run everything (recommended)

`use_case.py` is the single entry point. It runs all four phases in order:

```bash
python use_case.py                    # all domains — core + CWL + reports + anomaly
python use_case.py --domain satellite # satellite only, all phases
python use_case.py --domain ecg       # ECG only, all phases
python use_case.py --domain climate   # climate only, all phases
python use_case.py --skip-anomaly     # skip Phase 4 (much faster for dev)
```

The four phases and their individual scripts (can also be called directly):

| Phase | What it does | Individual script |
|-------|-------------|-------------------|
| 1 — Core (M1-M5) | Data ingestion → provenance | `use_cases/run_{sat,ecg,climate}.py` |
| 2 — CWL export | Sync `components.json` → `cwl/` | `python -m scripts.utils.cwl_export` |
| 3 — HTML reports | Self-contained run report per domain | `python -m scripts.utils.html_report` |
| 4 — Anomaly (E1+E2) | Fault injection + ML evaluation | `use_cases/{sat,ecg,climate}_anomaly.py` |

### Validate an adapter before running

```bash
python -m scripts.adapters.validator --adapter ecg     --path data/ECG/100.csv
python -m scripts.adapters.validator --adapter climate --path data/Climate/jena_climate_2009_2016.csv
```

### CLI options

Domain-specific args are forwarded automatically by `use_case.py`:

```bash
# Satellite: restrict to specific experiment families
python use_case.py --domain satellite --families AccelerometerTest GyroTest

# ECG: restrict to specific records
python use_case.py --domain ecg --records 100 106 108

# Climate: restrict to specific years
python use_case.py --domain climate --years 2011 2012 2013

# Anomaly use case: fast mode / fewer variants
python use_case.py --fast
python use_case.py --n-variants 1 --seed 0
```

Or invoke individual runners directly:

```bash
python use_cases/run_sat.py --families AccelerometerTest GyroTest
python use_cases/run_ecg.py --records 100 106 108
python use_cases/run_climate.py --years 2011 2012 2013
python use_cases/sat_anomaly.py --fast
```

---

## Output Structure

```
results/
  satellite/
    AccelerometerTest/
      cdh_clean.csv          # M1 output: cleaned telemetry
      adcs_clean.csv
      detection_report.json  # M2 quality report
      *.png                  # telemetry visualizations
      injected/              # E1+E2 outputs
        injection_summary.csv
        ml_evaluation.json
        ml_metrics_by_tier.csv
    semantic/                # M4 knowledge graph
      if_ontology_graph.ttl
      if_ontology_graph.graphml
      if_ontology_nodes.csv
      if_ontology_edges.csv
      if_interpretability_report.md
    workflow_spec.ttl        # M5: static ProvONE workflow spec
    provenance.ttl           # M5: runtime PROV-O trace with SHA-256 checksums
    run_report.html          # self-contained HTML run report

  ECG/mitdb/record_100/
    signal_clean.csv
    quality_report.json
    semantic/

  Climate/jena/2011_M01_M06/
    signal_clean.csv
    quality_report.json
    semantic/

  figures/                   # cross-domain publication figures
  ablation/                  # M3 window-sensitivity ablation results
  ontologies/
    satellitesystem.owl
    ecg.owl
    climate.owl

cwl/                         # CWL export
  workflow.cwl
  tools/M1.cwl ... tools/E2.cwl
```

---

## Adapting to a New Domain

To apply SensorWF to a new sensor domain, implement a `DomainAdapter` subclass:

```python
# scripts/adapters/my_domain.py
from scripts.adapters.base import DomainAdapter
import pandas as pd

class MyDomainAdapter(DomainAdapter):
    name     = "My Domain"
    channels = ["channel_a", "channel_b", "channel_c"]
    native_hz = 1.0

    def load(self, path: str, **kwargs) -> pd.DataFrame:
        # Return DataFrame with columns: timestamp, elapsed_s, <channels>
        ...

    def get_quality_config(self) -> dict:
        return {
            "stuck_unique_max": 3,
            "zscore_threshold": 3.0,
            "expected_dt_s": 1.0,
            "gap_multiplier": 5.0,
            "trend_channels": ["channel_a"],
        }

    def get_feature_config(self) -> dict:
        return {"window": 15, "channels": None}

    def get_ontology_path(self) -> str:
        return "results/ontologies/my_domain.owl"

    def get_fault_types(self) -> list[dict]:
        return [{"tag": "spike", "description": "Amplitude Spike", ...}]
```

Validate before running:

```python
from scripts.adapters.validator import validate_adapter
ok, report = validate_adapter(MyDomainAdapter(), sample_path="path/to/data.csv")
```

Then create a `use_cases/run_my_domain.py` entry point following the pattern in `use_cases/run_ecg.py` or `use_cases/run_climate.py`.

Three adapters are already provided:

| Adapter | Module | Domain |
|---------|--------|--------|
| `SatelliteAdapter` | `scripts/adapters/satellite.py` | KISPE SATLL SCOTTI v2 (merges CDH + ADCS) |
| `ECGAdapter` | `scripts/adapters/ecg.py` | MIT-BIH Arrhythmia Database |
| `ClimateAdapter` | `scripts/adapters/climate.py` | Jena Climate Dataset |

---

## Module Reference

### Core modules

| ID | Name | Script | Description |
|----|------|--------|-------------|
| M1 | DataIngestion | `scripts/adapters/` | Parse raw sensor archive via DomainAdapter |
| M2 | QualityAssessment | `scripts/pipeline_core.py` | NaN, stuck-channel, timing, z-score audit |
| M3 | FeatureEngineering | `scripts/pipeline_core.py` | 9-feature-family matrix per channel (raw, diff, rolling mean/std/skew/kurt, ZCR, spectral entropy, dominant frequency) |
| M4 | SemanticAnnotation | `scripts/satellite/semantic_kg.py` | Ontology-linked knowledge graph (RDF/Turtle + GraphML); multi-detector performance nodes |
| M5 | ProvenanceExport | `scripts/provenance_recorder.py` | PROV-O/ProvONE trace with SHA-256 file checksums |

### Extension modules (anomaly detection use case only)

| ID | Name | Script | Description |
|----|------|--------|-------------|
| E1 | FaultInjection | `scripts/utils/injector.py` | Synthetic fault injection, 3-tier difficulty |
| E2 | AnomalyDetection | `scripts/evaluator.py` | 5 ML detectors with EVT threshold calibration |

**E2 detectors:** ZScore, RobustRollingZScore, IsolationForest (rotation ensemble), Autoencoder (multi-scale MLP), LOF (density-based novelty with EVT thresholding).

---

## Feature Engineering (M3)

When a session has at least `3 × window` samples, M3 computes nine feature families per channel:

| # | Feature | Symbol | Notes |
|---|---------|--------|-------|
| 1 | Raw value | `{c}` | |
| 2 | First-order difference | `d_{c}` | Rate of change |
| 3 | Rolling mean | `rm_{c}` | Window = domain-specific |
| 4 | Rolling std | `rs_{c}` | |
| 5 | Rolling skewness | `sk_{c}` | Distribution asymmetry |
| 6 | Rolling excess kurtosis | `ku_{c}` | Tail heaviness |
| 7 | Zero-crossing rate | `zcr_{c}` | Relative to channel mean |
| 8 | Spectral entropy | `se_{c}` | Shannon entropy of power spectrum |
| 9 | Dominant frequency (Hz) | `df_{c}` | Peak frequency excluding DC |

Plus one timing feature: `dt_sample` (sample interval in seconds).

Default window sizes: ECG = 100 samples, Climate = 24 samples, Satellite = 15 samples.

---

## Provenance and FAIR Compliance

Every run produces two provenance artifacts:

- **`workflow_spec.ttl`** — static ProvONE program graph describing module ports, assumptions, and data flows.
- **`provenance.ttl`** — runtime PROV-O trace with per-module wall-clock times, row counts, file paths, and SHA-256 checksums for all output files.

The component registry (`components.json`) lists all modules with their semantic contracts, enabling automated discovery and reuse. A CWL export (`cwl/workflow.cwl`) provides interoperability with CWL-compatible workflow engines.

---

## Repository Structure

```
use_case.py         Full orchestrator: all four phases, all domains
use_cases/
  run_sat.py        Core pipeline: satellite (M1-M5)
  run_ecg.py        Core pipeline: ECG (M1-M5)
  run_climate.py    Core pipeline: climate (M1-M5)
  sat_anomaly.py    Anomaly detection: satellite (E1+E2)
  ecg_anomaly.py    Anomaly detection: ECG (E1+E2)
  climate_anomaly.py Anomaly detection: climate (E1+E2)
scripts/
  pipeline_core.py        Domain-agnostic M2/M3 logic (shared feature helpers)
  evaluator.py            Feature engineering + 5 ML detectors (M3, E2)
  provenance_recorder.py  Runtime PROV-O recorder with SHA-256 (M5)
  workflow.py             Static module registry and ProvONE spec emitter
  adapters/               DomainAdapter base class + all three domain adapters
    base.py               Abstract DomainAdapter interface
    satellite.py          KISPE SATLL adapter (merges CDH+ADCS; M1 for satellite)
    ecg.py                MIT-BIH ECG adapter
    climate.py            Jena Climate adapter
    validator.py          Pre-run schema validation harness
  satellite/              Satellite-specific internals (used by SatelliteAdapter + E1/E2)
    parser.py             SCOTTI archive parser (internal to SatelliteAdapter.load)
    cleaner.py            CDH/ADCS hex-decoding and cleaning (internal to SatelliteAdapter.load)
    detector.py           Satellite-specific supplementary quality checks (beyond generic M2)
    plotter.py            Telemetry visualization utility
    ontology.py           Satellite OWL ontology generator (runtime, channel-grounded)
    semantic_kg.py        M4 knowledge graph with multi-detector aggregation (post-E2)
    plot_kg.py            Knowledge graph visualization utility
  utils/                  Framework-level utilities (domain-agnostic)
    injector.py           Fault injection logic (E1)
    ablation_window.py    M3 window-size ablation study
    figures.py            Publication figure generators
    html_report.py        Academic-style HTML run report generator
    cwl_export.py         CWL workflow export from components.json
components.json     Machine-readable module registry (FAIR)
cwl/                CWL export (generated)
data/               Raw sensor archives (not tracked in git)
results/            Pipeline outputs (not tracked in git)
documentation/
  EScienceV2/paper.tex  Paper source (IEEE eScience 2026)
```
