"""
workflow.py — FAIR-annotated computational module registry for SensorWF.

Core modules (M1–M5): domain-agnostic, reusable across satellite, ECG, climate.
Extension modules (E1–E2): use-case-specific, loaded only by use_cases/ scripts.

  M1  DataIngestion      — parse raw sensor archives into standardised DataFrames
  M2  QualityAssessment  — NaN/stuck/timing/trend quality audit
  M3  FeatureEngineering — raw + diff + rolling feature matrix
  M4  SemanticAnnotation — ontology-linked knowledge graph (RDF/Turtle, GraphML)
  M5  ProvenanceExport   — PROV-O/ProvONE trace

  E1  FaultInjection     — synthetic fault injection (anomaly detection use case)
  E2  AnomalyDetection   — ML detector training and evaluation (anomaly detection use case)

Ontology prefixes used throughout
───────────────────────────────────
  prov:     http://www.w3.org/ns/prov#            (W3C PROV-O)
  provone:  http://purl.dataone.org/provone/2015/01/15/ontology#  (ProvONE)
  xsd:      http://www.w3.org/2001/XMLSchema#
  rdfs:     http://www.w3.org/2000/01/rdf-schema#
  sensorwf: https://example.org/sensorwf/workflow# (this workflow's namespace)
  dc:       http://purl.org/dc/terms/

Usage
─────
  from scripts.workflow import CORE_MODULES, emit_workflow_ttl
  ttl = emit_workflow_ttl()          # static workflow specification
  print(ttl)

  # At runtime, use ProvenanceRecorder (provenance_recorder.py) to capture
  # actual execution events (timestamps, row counts, file paths).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Ontology prefix map (shared with ProvenanceRecorder)
# ─────────────────────────────────────────────────────────────────────────────

PREFIXES: dict[str, str] = {
    "prov":      "http://www.w3.org/ns/prov#",
    "provone":   "http://purl.dataone.org/provone/2015/01/15/ontology#",
    "xsd":       "http://www.w3.org/2001/XMLSchema#",
    "rdfs":      "http://www.w3.org/2000/01/rdf-schema#",
    "owl":       "http://www.w3.org/2002/07/owl#",
    "dc":        "http://purl.org/dc/terms/",
    "sensorwf":  "https://example.org/sensorwf/workflow#",
    "telwf":     "https://example.org/telwf/workflow#",   # kept for backwards-compat in generated Turtle
}


def _ttl_prefix_block() -> str:
    return "\n".join(f"@prefix {k}: <{v}> ." for k, v in PREFIXES.items()) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Port and module data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PortSpec:
    """Describes one input or output port of a computational module."""
    name:        str
    label:       str
    rdf_type:    str
    description: str


@dataclass
class WorkflowModule:
    """One annotated computational module in the SensorWF framework.

    Fields
    ------
    module_id    : unique short identifier, used as RDF local name
    label        : human-readable name
    description  : short prose description of the module's purpose
    script_path  : relative path to the implementing Python module
    inputs       : list of PortSpec (in-ports)
    outputs      : list of PortSpec (out-ports)
    assumptions  : scientific/operational assumptions baked into this module
    extension    : True for use-case extensions (E1, E2); False for core (M1-M5)
    """
    module_id:   str
    label:       str
    description: str
    script_path: str
    inputs:      list[PortSpec] = field(default_factory=list)
    outputs:     list[PortSpec] = field(default_factory=list)
    assumptions: list[str]      = field(default_factory=list)
    extension:   bool           = False

    def to_turtle(self) -> str:
        """Emit this module as a ProvONE Program with PROV-O annotations."""
        lines: list[str] = []
        mid = f"sensorwf:{self.module_id}"

        lines.append(f"{mid}")
        lines.append(f"    a provone:Program, prov:Activity ;")
        lines.append(f'    rdfs:label "{self.label}" ;')
        lines.append(f'    dc:description "{_esc(self.description)}" ;')
        lines.append(f'    sensorwf:implementedBy "{self.script_path}" ;')
        if self.extension:
            lines.append(f'    sensorwf:isExtension true ;')

        for i, port in enumerate(self.inputs):
            pid = f"sensorwf:{self.module_id}_in_{port.name}"
            sep = "," if (i < len(self.inputs) - 1 or self.outputs or self.assumptions) else ""
            lines.append(f"    provone:hasInPort {pid}{sep}")

        for i, port in enumerate(self.outputs):
            pid = f"sensorwf:{self.module_id}_out_{port.name}"
            sep = "," if (i < len(self.outputs) - 1 or self.assumptions) else ""
            lines.append(f"    provone:hasOutPort {pid}{sep}")

        for i, assumption in enumerate(self.assumptions):
            sep = ";" if i < len(self.assumptions) - 1 else ""
            lines.append(f'    sensorwf:hasAssumption "{_esc(assumption)}"{"," if sep else ""}')

        lines.append(".")
        lines.append("")

        for port in self.inputs:
            pid = f"sensorwf:{self.module_id}_in_{port.name}"
            lines += _port_turtle(pid, port, "in")

        for port in self.outputs:
            pid = f"sensorwf:{self.module_id}_out_{port.name}"
            lines += _port_turtle(pid, port, "out")

        return "\n".join(lines)


def _esc(s: str) -> str:
    """Escape double-quotes and backslashes for Turtle string literals."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _port_turtle(pid: str, port: PortSpec, direction: str) -> list[str]:
    dtype = "provone:InPort" if direction == "in" else "provone:OutPort"
    return [
        f"{pid}",
        f"    a {dtype} ;",
        f'    rdfs:label "{_esc(port.label)}" ;',
        f'    dc:description "{_esc(port.description)}" ;',
        f"    sensorwf:dataType {port.rdf_type} .",
        "",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Core module definitions (M1–M5)
# ─────────────────────────────────────────────────────────────────────────────

M1_INGEST = WorkflowModule(
    module_id   = "M1_DataIngestion",
    label       = "Data Ingestion",
    description = (
        "Domain-adaptive ingestion that normalises raw sensor archives into a "
        "standardised time-series DataFrame (timestamp, elapsed_s, numeric channels). "
        "Each domain is handled by a pluggable DomainAdapter implementing the shared "
        "interface. Supports SCOTTI v2 telemetry, MIT-BIH ECG CSV, and Jena Climate CSV."
    ),
    script_path = "scripts/adapters/",
    inputs = [
        PortSpec(
            name        = "raw_files",
            label       = "Raw Sensor Archive",
            rdf_type    = "xsd:anyURI",
            description = "Raw sensor archive (SCOTTI v2 txt, MIT-BIH CSV, Jena Climate CSV).",
        ),
        PortSpec(
            name        = "adapter_config",
            label       = "Adapter Configuration",
            rdf_type    = "xsd:string",
            description = "Domain adapter parameters (target_hz, session_minutes, year, etc.).",
        ),
    ],
    outputs = [
        PortSpec(
            name        = "clean_df",
            label       = "Standardised Time-Series DataFrame",
            rdf_type    = "sensorwf:DataFrame",
            description = "Standardised time-series with timestamp, elapsed_s, and sensor channels.",
        ),
        PortSpec(
            name        = "adapter_meta",
            label       = "Adapter Metadata",
            rdf_type    = "xsd:string",
            description = "Adapter name, channels, native_hz, session length.",
        ),
    ],
    assumptions = [
        "Each domain has a unique raw format handled by its DomainAdapter.",
        "All adapters return a DataFrame with timestamp (datetime) and elapsed_s (float).",
        "Downsampling is domain-specific and configured via the adapter.",
    ],
)


M2_QUALITY = WorkflowModule(
    module_id   = "M2_QualityAssessment",
    label       = "Quality Assessment",
    description = (
        "Domain-agnostic sensor quality audit: NaN rates, stuck-channel detection, "
        "timing regularity, per-channel z-score flagging, and linear trend detection "
        "on selected channels. Outputs a machine-readable JSON quality report."
    ),
    script_path = "scripts/pipeline_core.py",
    inputs = [
        PortSpec(
            name        = "clean_df",
            label       = "Standardised Time-Series",
            rdf_type    = "sensorwf:DataFrame",
            description = "Standardised time-series from M1.",
        ),
        PortSpec(
            name        = "channels",
            label       = "Channel List",
            rdf_type    = "xsd:string",
            description = "Sensor channels to assess.",
        ),
        PortSpec(
            name        = "config",
            label       = "Quality Config",
            rdf_type    = "xsd:string",
            description = "Domain quality-assessment config (thresholds, expected dt, trend channels).",
        ),
    ],
    outputs = [
        PortSpec(
            name        = "quality_report",
            label       = "Quality Report",
            rdf_type    = "sensorwf:QualityReport",
            description = "Per-channel quality metrics and summary flags.",
        ),
    ],
    assumptions = [
        "Stuck sensor: <= 3 unique values in a channel across the full session.",
        "Z-score flag threshold: configurable per domain (default 3.0 sigma).",
        "Timing gap threshold: expected_dt * gap_multiplier (domain-specific).",
    ],
)


M3_FEATURES = WorkflowModule(
    module_id   = "M3_FeatureEngineering",
    label       = "Feature Engineering",
    description = (
        "Builds a multi-dimensional feature matrix from the clean time-series. "
        "Nine feature families per channel (when session >= 3x window): raw value, "
        "first-order difference, rolling mean, rolling std, rolling skewness, rolling "
        "excess kurtosis, zero-crossing rate, spectral entropy, and dominant frequency (Hz). "
        "A sample-interval timing feature (dt_sample) is always included."
    ),
    script_path = "scripts/evaluator.py",
    inputs = [
        PortSpec(
            name        = "clean_df",
            label       = "Standardised Time-Series",
            rdf_type    = "sensorwf:DataFrame",
            description = "Standardised time-series from M1.",
        ),
        PortSpec(
            name        = "channels",
            label       = "Channel List",
            rdf_type    = "xsd:string",
            description = "Sensor channels to engineer features over.",
        ),
        PortSpec(
            name        = "window",
            label       = "Rolling Window Size",
            rdf_type    = "xsd:integer",
            description = "Rolling window size in samples.",
        ),
    ],
    outputs = [
        PortSpec(
            name        = "feature_matrix",
            label       = "Feature Matrix",
            rdf_type    = "sensorwf:NumpyArray",
            description = "Numeric (n_samples, n_features) float64 array.",
        ),
        PortSpec(
            name        = "feature_names",
            label       = "Feature Name List",
            rdf_type    = "xsd:string",
            description = "Ordered list of feature column names matching the feature matrix.",
        ),
    ],
    assumptions = [
        "Rolling statistics are only computed when session length >= 3 * window.",
        "NaN values are forward-filled then backward-filled before feature extraction.",
        "Window size should be tuned per domain (ECG: 100, Climate: 24, Satellite: 15 samples).",
    ],
)


M4_SEMANTIC = WorkflowModule(
    module_id   = "M4_SemanticAnnotation",
    label       = "Semantic Annotation and Knowledge Graph Construction",
    description = (
        "Maps feature names and (optionally) ML detector evidence to domain OWL ontology "
        "classes to produce an ontology-linked knowledge graph (RDF/Turtle, GraphML). "
        "Enables SPARQL-queryable, subsystem-grounded explainability and cross-domain "
        "semantic comparison. In core mode, edges encode feature-class membership; "
        "in use-case mode, edges are additionally weighted by IsolationForest importances."
    ),
    script_path = "scripts/utils/semantic_if_kg.py",
    inputs = [
        PortSpec(
            name        = "feature_names",
            label       = "Feature Name List",
            rdf_type    = "xsd:string",
            description = "Feature names from M3.",
        ),
        PortSpec(
            name        = "ontology_owl",
            label       = "Domain Ontology",
            rdf_type    = "xsd:anyURI",
            description = "Domain OWL ontology file.",
        ),
        PortSpec(
            name        = "ml_results",
            label       = "ML Evaluation Results (optional)",
            rdf_type    = "xsd:anyURI",
            description = "Optional E2 output; if provided, IF importances weight KG edges.",
        ),
    ],
    outputs = [
        PortSpec(
            name        = "kg_turtle",
            label       = "Knowledge Graph (Turtle)",
            rdf_type    = "xsd:anyURI",
            description = "RDF/Turtle serialisation of the semantic KG.",
        ),
        PortSpec(
            name        = "kg_nodes_csv",
            label       = "Knowledge Graph Nodes CSV",
            rdf_type    = "xsd:anyURI",
            description = "Node table for GraphML/NetworkX analysis.",
        ),
        PortSpec(
            name        = "kg_edges_csv",
            label       = "Knowledge Graph Edges CSV",
            rdf_type    = "xsd:anyURI",
            description = "Edge table with semantic relation types and importance weights.",
        ),
    ],
    assumptions = [
        "Feature names are matched to OWL class local names by prefix substring.",
        "IsolationForest feature_importances_ are used as edge weights (extension mode only).",
        "Ontology local names follow the domain-specific naming convention.",
    ],
)


M5_PROVENANCE = WorkflowModule(
    module_id   = "M5_ProvenanceExport",
    label       = "Provenance Export",
    description = (
        "Records one prov:Activity per M1-M4 module execution with wall-clock timestamps, "
        "input row counts, output file paths, and parameter values. Serialises the "
        "complete PROV-O/ProvONE trace to RDF/Turtle."
    ),
    script_path = "scripts/provenance_recorder.py",
    inputs = [
        PortSpec(
            name        = "execution_log",
            label       = "Execution Log",
            rdf_type    = "sensorwf:ExecutionLog",
            description = "In-memory list of module execution records collected during the pipeline run.",
        ),
    ],
    outputs = [
        PortSpec(
            name        = "provenance_ttl",
            label       = "Provenance Trace (Turtle)",
            rdf_type    = "xsd:anyURI",
            description = "PROV-O/ProvONE serialisation of the full run.",
        ),
    ],
    assumptions = [
        "Timestamps are UTC ISO 8601 strings.",
        "SHA-256 checksums are computed automatically for all file-path entities and stored as telwf:sha256 in the provenance trace.",
        "Agent information is limited to the host machine; no multi-user attribution is tracked.",
    ],
)


# ─────────────────────────────────────────────────────────────────────────────
# Extension module definitions (E1–E2, anomaly detection use case only)
# ─────────────────────────────────────────────────────────────────────────────

E1_FAULT_INJECTION = WorkflowModule(
    module_id   = "E1_FaultInjection",
    label       = "Synthetic Fault Injection",
    description = (
        "Generates labelled sensor variants by injecting domain-specific fault morphologies "
        "across three difficulty tiers (easy / medium / hard). Each tier co-varies "
        "amplitude, duration, and channel spread following benchmark design guidelines "
        "from Hundman et al. (2018) and Wu & Keogh (2022)."
    ),
    script_path = "scripts/utils/injector.py",
    extension   = True,
    inputs = [
        PortSpec(
            name        = "clean_df",
            label       = "Clean Session DataFrame",
            rdf_type    = "sensorwf:DataFrame",
            description = "Output of M2: fault-free session.",
        ),
        PortSpec(
            name        = "seed",
            label       = "Random Seed",
            rdf_type    = "xsd:integer",
            description = "Integer seed for reproducible variant generation.",
        ),
        PortSpec(
            name        = "n_variants",
            label       = "Variant Count",
            rdf_type    = "xsd:integer",
            description = "Number of stochastic realisations per fault type per tier.",
        ),
    ],
    outputs = [
        PortSpec(
            name        = "injected_variants",
            label       = "Injected CSV Variants",
            rdf_type    = "xsd:anyURI",
            description = "Directory of injected CSV files, one per fault/tier/variant.",
        ),
        PortSpec(
            name        = "label_files",
            label       = "Ground-Truth Label Files",
            rdf_type    = "xsd:anyURI",
            description = "Companion labels_*.csv files with start_idx, end_idx, severity.",
        ),
        PortSpec(
            name        = "injection_summary",
            label       = "Injection Summary CSV",
            rdf_type    = "xsd:anyURI",
            description = "Aggregated injection_summary.csv listing fault types and counts.",
        ),
    ],
    assumptions = [
        "Tier profiles (easy/medium/hard) simultaneously scale amplitude, duration, and channel spread.",
        "Random seed 42 is the default canonical seed for reproducibility.",
        "Anomaly windows occupy 10-50% of a session; threshold calibration uses the clean 60% prefix.",
    ],
)


E2_DETECTION = WorkflowModule(
    module_id   = "E2_AnomalyDetection",
    label       = "Anomaly Detection",
    description = (
        "Trains and evaluates five anomaly detectors (ZScore, RobustRollingZScore, "
        "IsolationForest with rotation ensemble, Autoencoder multi-scale MLP, LOF) "
        "on injected sensor variants. Threshold strategy: EVT/POT GPD calibration for "
        "density detectors; 99th-percentile for statistical detectors."
    ),
    script_path = "scripts/evaluator.py",
    extension   = True,
    inputs = [
        PortSpec(
            name        = "injected_variants",
            label       = "Injected Variants Directory",
            rdf_type    = "xsd:anyURI",
            description = "Output of E1: directory containing injected CSV files.",
        ),
        PortSpec(
            name        = "label_files",
            label       = "Ground-Truth Label Files",
            rdf_type    = "xsd:anyURI",
            description = "Output of E1: labels_*.csv files with anomaly window indices.",
        ),
        PortSpec(
            name        = "clean_df",
            label       = "Clean Session DataFrame",
            rdf_type    = "sensorwf:DataFrame",
            description = "Used to build the clean training prefix for detector fitting.",
        ),
    ],
    outputs = [
        PortSpec(
            name        = "ml_evaluation_json",
            label       = "ML Evaluation JSON",
            rdf_type    = "xsd:anyURI",
            description = "Per-(tag, tier, variant, detector) result dicts.",
        ),
        PortSpec(
            name        = "ml_metrics_csv",
            label       = "Tier Metrics CSV",
            rdf_type    = "xsd:anyURI",
            description = "Aggregated metrics per tier and detector.",
        ),
        PortSpec(
            name        = "feature_importances",
            label       = "IsolationForest Feature Importances",
            rdf_type    = "sensorwf:NumpyArray",
            description = "Mean split-frequency importances across all IF trees.",
        ),
    ],
    assumptions = [
        "Training split is 60% of the clean session prefix; the remaining 40% is used for threshold validation.",
        "Decision threshold is the 99th percentile of clean training scores, targeting ~1% nominal FPR.",
        "IsolationForest uses a rotation ensemble of 3 members to reduce axis-alignment bias.",
        "Autoencoder uses multi-scale windowing (seq_len 8 and 16) with denoising (sigma=0.06).",
        "LOF uses novelty=True mode — fitted on clean data, evaluated on injected data.",
        "EVT calibration requires >=10 exceedances above the 90th-percentile pre-threshold.",
    ],
)


# ─────────────────────────────────────────────────────────────────────────────
# Workflow channel declarations (module-to-module data flows, core only)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WorkflowChannel:
    """A provone:Channel connecting an out-port to an in-port."""
    channel_id:  str
    from_module: str
    out_port:    str
    to_module:   str
    in_port:     str

    def to_turtle(self) -> str:
        cid = f"sensorwf:{self.channel_id}"
        src = f"sensorwf:{self.from_module}_out_{self.out_port}"
        dst = f"sensorwf:{self.to_module}_in_{self.in_port}"
        return (
            f"{cid}\n"
            f"    a provone:Channel ;\n"
            f"    provone:connectsTo {dst} ;\n"
            f"    provone:connectsFrom {src} .\n"
        )


CHANNELS: list[WorkflowChannel] = [
    WorkflowChannel("ch01", "M1_DataIngestion",     "clean_df",       "M2_QualityAssessment",  "clean_df"),
    WorkflowChannel("ch02", "M1_DataIngestion",     "adapter_meta",   "M2_QualityAssessment",  "channels"),
    WorkflowChannel("ch03", "M2_QualityAssessment", "quality_report", "M3_FeatureEngineering", "clean_df"),
    WorkflowChannel("ch04", "M3_FeatureEngineering","feature_names",  "M4_SemanticAnnotation", "feature_names"),
    WorkflowChannel("ch05", "M3_FeatureEngineering","feature_matrix", "M4_SemanticAnnotation", "ml_results"),
    WorkflowChannel("ch06", "M4_SemanticAnnotation","kg_turtle",      "M5_ProvenanceExport",   "execution_log"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Module registries
# ─────────────────────────────────────────────────────────────────────────────

CORE_MODULES: list[WorkflowModule] = [
    M1_INGEST,
    M2_QUALITY,
    M3_FEATURES,
    M4_SEMANTIC,
    M5_PROVENANCE,
]

EXTENSION_MODULES: list[WorkflowModule] = [
    E1_FAULT_INJECTION,
    E2_DETECTION,
]

MODULE_REGISTRY: list[WorkflowModule] = CORE_MODULES + EXTENSION_MODULES

MODULE_BY_ID: dict[str, WorkflowModule] = {m.module_id: m for m in MODULE_REGISTRY}


# ─────────────────────────────────────────────────────────────────────────────
# Static workflow specification emission
# ─────────────────────────────────────────────────────────────────────────────

_WORKFLOW_COMMENT = """\
# SensorWF — Generalizable FAIR Sensor Analytics Workflow
# Static specification: module ports, assumptions, and data-flow channels.
# Runtime execution provenance is appended by scripts/provenance_recorder.py.
#
# Core modules:      M1 DataIngestion, M2 QualityAssessment, M3 FeatureEngineering,
#                    M4 SemanticAnnotation, M5 ProvenanceExport
# Extension modules: E1 FaultInjection, E2 AnomalyDetection (use_cases/ only)
#
# Ontology alignment:
#   provone:Program  — computational module
#   provone:InPort   — input data/parameter slot
#   provone:OutPort  — output artifact slot
#   provone:Channel  — data-flow edge between ports
#   prov:Activity    — a single module execution event (runtime)
#   prov:Entity      — a data artifact (runtime)
#
# Reference:
#   Missier et al. (2013) ProvONE. DataONE Project.
#   W3C PROV-O Recommendation (2013). https://www.w3.org/TR/prov-o/
"""

_WORKFLOW_HEADER = """\
sensorwf:SensorWFWorkflow
    a provone:Workflow, prov:Plan ;
    rdfs:label "SensorWF: Generalizable FAIR Sensor Analytics Workflow" ;
    dc:description "A FAIR-annotated, reproducible workflow for multi-domain sensor "
        "time-series analysis: data ingestion, quality assessment, feature engineering, "
        "semantic annotation, and provenance export. Supports satellite telemetry, "
        "biomedical ECG, and atmospheric climate domains." ;
    dc:creator "Logan Luna" ;
    sensorwf:version "1.0" ;
"""


def emit_workflow_ttl() -> str:
    """Emit the complete static workflow specification as a Turtle string."""
    parts: list[str] = [
        _WORKFLOW_COMMENT,
        _ttl_prefix_block(),
        "",
        _WORKFLOW_HEADER,
    ]

    # Wire workflow → core modules
    parts.append("    provone:hasSubProgram")
    for i, mod in enumerate(CORE_MODULES):
        sep = "," if i < len(CORE_MODULES) - 1 else " ."
        parts.append(f"        sensorwf:{mod.module_id}{sep}")
    parts.append("")

    # Individual module descriptions (core only in static spec)
    for mod in CORE_MODULES:
        parts.append(f"# ── {mod.label} ──────────────────────────────────────────")
        parts.append(mod.to_turtle())

    # Data-flow channels
    parts.append("# ── Data-flow channels ──────────────────────────────────────────")
    for ch in CHANNELS:
        parts.append(ch.to_turtle())

    return "\n".join(parts)


def save_workflow_spec(path: str) -> None:
    """Write the static workflow specification to *path* (Turtle format)."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(emit_workflow_ttl())
    print(f"  [workflow] Specification written to {path}")
