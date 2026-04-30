"""
workflow.py — FAIR-annotated computational module registry for the
              CubeSat Telemetry Analysis Workflow (TelWF).

Each WorkflowModule instance is the canonical machine-readable description
of one processing stage:  its input/output ports, scientific assumptions,
and PROV-O / ProvONE identifiers needed to emit a reproducible provenance
trace at runtime.

Ontology prefixes used throughout
───────────────────────────────────
  prov:    http://www.w3.org/ns/prov#            (W3C PROV-O)
  provone: http://purl.dataone.org/provone/2015/01/15/ontology#  (ProvONE)
  xsd:     http://www.w3.org/2001/XMLSchema#
  rdfs:    http://www.w3.org/2000/01/rdf-schema#
  sat:     https://example.org/telwf/satellite#  (domain vocabulary)
  telwf:   https://example.org/telwf/workflow#   (this workflow's namespace)

Usage
─────
  from scripts.workflow import MODULE_REGISTRY, emit_workflow_ttl
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
    "prov":    "http://www.w3.org/ns/prov#",
    "provone": "http://purl.dataone.org/provone/2015/01/15/ontology#",
    "xsd":     "http://www.w3.org/2001/XMLSchema#",
    "rdfs":    "http://www.w3.org/2000/01/rdf-schema#",
    "owl":     "http://www.w3.org/2002/07/owl#",
    "sat":     "https://example.org/telwf/satellite#",
    "telwf":   "https://example.org/telwf/workflow#",
    "dc":        "http://purl.org/dc/terms/",
    "sensorwf":  "https://example.org/sensorwf/workflow#",
}


def _ttl_prefix_block() -> str:
    return "\n".join(f"@prefix {k}: <{v}> ." for k, v in PREFIXES.items()) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Port and module data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PortSpec:
    """Describes one input or output port of a computational module.

    Attributes
    ----------
    name        : machine-readable identifier (used in RDF local name)
    label       : human-readable label
    rdf_type    : XSD or PROV-O type string, e.g. "xsd:string", "prov:Entity"
    description : one-sentence description
    """
    name:        str
    label:       str
    rdf_type:    str
    description: str


@dataclass
class WorkflowModule:
    """One annotated computational module in the telemetry analysis workflow.

    Fields
    ------
    module_id    : unique short identifier, used as RDF local name
    label        : human-readable name
    description  : short prose description of the module's purpose
    script_path  : relative path to the implementing Python module
    inputs       : list of PortSpec (in-ports)
    outputs      : list of PortSpec (out-ports)
    assumptions  : scientific/operational assumptions baked into this module
    """
    module_id:   str
    label:       str
    description: str
    script_path: str
    inputs:      list[PortSpec] = field(default_factory=list)
    outputs:     list[PortSpec] = field(default_factory=list)
    assumptions: list[str]      = field(default_factory=list)

    # ── RDF serialisation ────────────────────────────────────────────────────

    def to_turtle(self) -> str:
        """Emit this module as a ProvONE Program with PROV-O annotations."""
        lines: list[str] = []
        mid = f"telwf:{self.module_id}"

        lines.append(f"{mid}")
        lines.append(f"    a provone:Program, prov:Activity ;")
        lines.append(f'    rdfs:label "{self.label}" ;')
        lines.append(f'    dc:description "{_esc(self.description)}" ;')
        lines.append(f'    telwf:implementedBy "{self.script_path}" ;')

        for i, port in enumerate(self.inputs):
            pid = f"telwf:{self.module_id}_in_{port.name}"
            sep = "," if (i < len(self.inputs) - 1 or self.outputs or self.assumptions) else ""
            lines.append(f"    provone:hasInPort {pid}{sep}")

        for i, port in enumerate(self.outputs):
            pid = f"telwf:{self.module_id}_out_{port.name}"
            sep = "," if (i < len(self.outputs) - 1 or self.assumptions) else ""
            lines.append(f"    provone:hasOutPort {pid}{sep}")

        for i, assumption in enumerate(self.assumptions):
            sep = ";" if i < len(self.assumptions) - 1 else ""
            lines.append(f'    sat:hasScientificAssumption "{_esc(assumption)}"{"," if sep else ""}')

        lines.append(".")
        lines.append("")

        # Emit port descriptions
        for port in self.inputs:
            pid = f"telwf:{self.module_id}_in_{port.name}"
            lines += _port_turtle(pid, port, "in")

        for port in self.outputs:
            pid = f"telwf:{self.module_id}_out_{port.name}"
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
        f"    telwf:dataType {port.rdf_type} .",
        "",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Module definitions — one per processing stage
# ─────────────────────────────────────────────────────────────────────────────

M1_INGEST = WorkflowModule(
    module_id   = "M1_TelemetryIngestion",
    label       = "Telemetry Ingestion",
    description = (
        "Parses raw SCOTTI archive exports (.txt) into structured CDH and ADCS "
        "DataFrames. Supports both Analysis-format and Debug-format files, with "
        "automatic fallback. Extracts 87-field CDH packets and 30-field ADCS "
        "Sensor packets and assigns ISO 8601 timestamps."
    ),
    script_path = "scripts/parser.py",
    inputs = [
        PortSpec(
            name        = "analysis_file",
            label       = "SCOTTI Analysis File",
            rdf_type    = "xsd:anyURI",
            description = "Path to a SCOTTI *_Analysis.txt archive export.",
        ),
        PortSpec(
            name        = "debug_file",
            label       = "SCOTTI Debug File",
            rdf_type    = "xsd:anyURI",
            description = "Optional path to *_Debug.txt; used if Analysis file is empty.",
        ),
    ],
    outputs = [
        PortSpec(
            name        = "cdh_raw",
            label       = "Raw CDH DataFrame",
            rdf_type    = "telwf:DataFrame",
            description = "Un-cleaned CDH packet table with string-typed measurement columns.",
        ),
        PortSpec(
            name        = "adcs_raw",
            label       = "Raw ADCS DataFrame",
            rdf_type    = "telwf:DataFrame",
            description = "Un-cleaned ADCS Sensor packet table with string-typed columns.",
        ),
    ],
    assumptions = [
        "Archive files follow the SCOTTI v2 protocol with fixed-column CDH and ADCS Plot lines.",
        "Nominal telemetry sampling rate is 1 Hz; sub-second timestamps are available in Analysis format.",
        "Hex-encoded fields use unsigned 16-bit representation unless documented as signed (BATT_CHR_I, WHEEL_SPEED).",
    ],
)


M2_QUALITY = WorkflowModule(
    module_id   = "M2_QualityAssessment",
    label       = "Telemetry Quality Assessment",
    description = (
        "Cleans raw CDH and ADCS DataFrames by decoding hex fields, coercing "
        "types, deduplicating timestamps, deriving elapsed-time and power "
        "estimation columns, and flagging stuck-sensor channels. Produces "
        "analysis-ready DataFrames and a structured quality report."
    ),
    script_path = "scripts/cleaner.py",
    inputs = [
        PortSpec(
            name        = "cdh_raw",
            label       = "Raw CDH DataFrame",
            rdf_type    = "telwf:DataFrame",
            description = "Output of M1: un-cleaned CDH packet table.",
        ),
        PortSpec(
            name        = "adcs_raw",
            label       = "Raw ADCS DataFrame",
            rdf_type    = "telwf:DataFrame",
            description = "Output of M1: un-cleaned ADCS packet table.",
        ),
    ],
    outputs = [
        PortSpec(
            name        = "cdh_clean",
            label       = "Cleaned CDH DataFrame",
            rdf_type    = "telwf:DataFrame",
            description = "Type-coerced, deduplicated CDH table with elapsed_s and power columns.",
        ),
        PortSpec(
            name        = "adcs_clean",
            label       = "Cleaned ADCS DataFrame",
            rdf_type    = "telwf:DataFrame",
            description = "Type-coerced ADCS table with signed direction-aware numeric fields.",
        ),
        PortSpec(
            name        = "quality_report",
            label       = "Quality Report",
            rdf_type    = "telwf:QualityReport",
            description = "List of stuck-sensor channel names and timing statistics.",
        ),
    ],
    assumptions = [
        "Current ADC scale factor is 1.0 mA/count (relative proxy; full calibration document not available).",
        "Stuck-sensor detection uses a unique-value threshold of 3 or fewer distinct values over the session.",
        "Timestamps that are identical to the previous row are dropped as duplicates.",
    ],
)


M3_FEATURES = WorkflowModule(
    module_id   = "M3_FeatureEngineering",
    label       = "Feature Engineering",
    description = (
        "Constructs a numeric feature matrix from cleaned telemetry by combining "
        "raw channel values, first-order differences (rate-of-change), packet-interval "
        "features from elapsed_s, and rolling mean/std descriptors. Supports CDH-only, "
        "ADCS-only, and combined CDH+ADCS feature spaces."
    ),
    script_path = "scripts/evaluator.py",
    inputs = [
        PortSpec(
            name        = "cdh_clean",
            label       = "Cleaned CDH DataFrame",
            rdf_type    = "telwf:DataFrame",
            description = "Output of M2.",
        ),
        PortSpec(
            name        = "adcs_clean",
            label       = "Cleaned ADCS DataFrame",
            rdf_type    = "telwf:DataFrame",
            description = "Output of M2.",
        ),
    ],
    outputs = [
        PortSpec(
            name        = "feature_matrix",
            label       = "Feature Matrix",
            rdf_type    = "telwf:NumpyArray",
            description = "Numeric (n_samples, n_features) float64 array ready for ML detectors.",
        ),
        PortSpec(
            name        = "feature_names",
            label       = "Feature Name List",
            rdf_type    = "xsd:string",
            description = "Ordered list of feature column names matching the feature matrix.",
        ),
    ],
    assumptions = [
        "Rolling statistics window is 15 samples; features are only added when session length >= 45 rows.",
        "Missing values are forward-filled then backward-filled then zero-filled before feature extraction.",
        "First-order differences at index 0 are set to 0 to avoid NaN propagation.",
    ],
)


M4_FAULT_INJECTION = WorkflowModule(
    module_id   = "M4_FaultInjection",
    label       = "Synthetic Fault Injection",
    description = (
        "Generates labelled telemetry variants by injecting 16 literature-informed "
        "single-channel fault morphologies plus 2 compound multi-subsystem faults "
        "across three difficulty tiers (easy / medium / hard). Each tier co-varies "
        "amplitude, duration, and channel spread following benchmark design guidelines "
        "from Hundman et al. (2018) and Wu & Keogh (2022)."
    ),
    script_path = "scripts/injector.py",
    inputs = [
        PortSpec(
            name        = "cdh_clean",
            label       = "Cleaned CDH DataFrame",
            rdf_type    = "telwf:DataFrame",
            description = "Output of M2: fault-free CDH telemetry session.",
        ),
        PortSpec(
            name        = "adcs_clean",
            label       = "Cleaned ADCS DataFrame",
            rdf_type    = "telwf:DataFrame",
            description = "Output of M2: fault-free ADCS telemetry session.",
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
            description = "Directory of *_injected_*.csv files, one per fault/tier/variant.",
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
        "Fault morphology parameters are sampled from literature (KDD 2018, NAB 2015, TKDE 2022).",
        "Tier profiles (easy/medium/hard) simultaneously scale amplitude, duration, and channel spread.",
        "Random seed 42 is the default canonical seed for reproducibility; changing it is documented.",
        "Anomaly windows occupy 10-50% of a session; detector threshold calibration uses the clean 60% prefix.",
    ],
)


M5_DETECTION = WorkflowModule(
    module_id   = "M5_AnomalyDetection",
    label       = "Anomaly Detection",
    description = (
        "Trains and evaluates four anomaly detectors (ZScore, RobustRollingZScore, "
        "IsolationForest with rotation ensemble, multi-scale denoising Autoencoder) "
        "on injected telemetry variants. Each detector is trained on the first 60% "
        "of the clean session and scored on the full injected stream. Decision "
        "thresholds are calibrated at the 99th percentile of clean training scores."
    ),
    script_path = "scripts/evaluator.py",
    inputs = [
        PortSpec(
            name        = "injected_variants",
            label       = "Injected Variants Directory",
            rdf_type    = "xsd:anyURI",
            description = "Output of M4: directory containing injected CSV files.",
        ),
        PortSpec(
            name        = "label_files",
            label       = "Ground-Truth Label Files",
            rdf_type    = "xsd:anyURI",
            description = "Output of M4: labels_*.csv files with anomaly window indices.",
        ),
        PortSpec(
            name        = "cdh_clean",
            label       = "Clean CDH DataFrame",
            rdf_type    = "telwf:DataFrame",
            description = "Used to build the clean training prefix for detector fitting.",
        ),
        PortSpec(
            name        = "adcs_clean",
            label       = "Clean ADCS DataFrame",
            rdf_type    = "telwf:DataFrame",
            description = "Used to build the clean training prefix for detector fitting.",
        ),
    ],
    outputs = [
        PortSpec(
            name        = "ml_evaluation_json",
            label       = "ML Evaluation JSON",
            rdf_type    = "xsd:anyURI",
            description = "ml_evaluation.json: per-(tag, tier, variant, detector) result dicts.",
        ),
        PortSpec(
            name        = "ml_metrics_csv",
            label       = "Tier Metrics CSV",
            rdf_type    = "xsd:anyURI",
            description = "ml_metrics_by_tier.csv: aggregated metrics per tier and detector.",
        ),
        PortSpec(
            name        = "feature_importances",
            label       = "IsolationForest Feature Importances",
            rdf_type    = "telwf:NumpyArray",
            description = "Mean split-frequency importances across all IF trees and ensemble members.",
        ),
    ],
    assumptions = [
        "Training split is 60% of the clean telemetry prefix; the remaining 40% is used for threshold validation.",
        "Decision threshold is the 99th percentile of clean training scores, targeting ~1% nominal FPR.",
        "IsolationForest uses a rotation ensemble of 3 members to reduce axis-alignment bias.",
        "Autoencoder uses multi-scale windowing (seq_len 8 and 16) with denoising (sigma=0.06).",
    ],
)


M6_SEMANTIC = WorkflowModule(
    module_id   = "M6_SemanticAnnotation",
    label       = "Semantic Annotation and Knowledge Graph Construction",
    description = (
        "Maps detector evidence (feature importances, tree split rules, per-instance "
        "local contributions) to ontology classes from satellitesystem.owl to produce "
        "an ontology-linked knowledge graph (RDF/Turtle, GraphML). Enables "
        "SPARQL-queryable, subsystem-grounded explainability and cross-family "
        "semantic comparison."
    ),
    script_path = "scripts/semantic_if_kg.py",
    inputs = [
        PortSpec(
            name        = "ml_evaluation_json",
            label       = "ML Evaluation JSON",
            rdf_type    = "xsd:anyURI",
            description = "Output of M5: feature importances and per-instance scores.",
        ),
        PortSpec(
            name        = "ontology_owl",
            label       = "Satellite System Ontology",
            rdf_type    = "xsd:anyURI",
            description = "satellitesystem.owl: OWL ontology covering CDH, ADCS, EPS, COMMS subsystems.",
        ),
        PortSpec(
            name        = "clean_data_dir",
            label       = "Clean Data Directory",
            rdf_type    = "xsd:anyURI",
            description = "Directory containing cdh_clean.csv and adcs_clean.csv per experiment.",
        ),
    ],
    outputs = [
        PortSpec(
            name        = "kg_turtle",
            label       = "Knowledge Graph (Turtle)",
            rdf_type    = "xsd:anyURI",
            description = "if_ontology_graph.ttl: RDF/Turtle serialisation of the semantic KG.",
        ),
        PortSpec(
            name        = "kg_nodes_csv",
            label       = "Knowledge Graph Nodes CSV",
            rdf_type    = "xsd:anyURI",
            description = "if_ontology_nodes.csv: node table for GraphML/NetworkX analysis.",
        ),
        PortSpec(
            name        = "kg_edges_csv",
            label       = "Knowledge Graph Edges CSV",
            rdf_type    = "xsd:anyURI",
            description = "if_ontology_edges.csv: edge table with semantic relation types.",
        ),
        PortSpec(
            name        = "interpretability_report",
            label       = "Interpretability Report",
            rdf_type    = "xsd:anyURI",
            description = "if_interpretability_report.md: human-readable semantic findings.",
        ),
    ],
    assumptions = [
        "Ontology local names match the feature-name prefixes used in the feature engineering step.",
        "IsolationForest is re-fit in no-PCA, no-rotation mode for rule extraction to preserve feature-space interpretability.",
        "Local contribution signals are from tree-path depth proxies; SHAP is used if available, else depth-based fallback.",
    ],
)


M7_PROVENANCE = WorkflowModule(
    module_id   = "M7_ProvenanceExport",
    label       = "Provenance Export",
    description = (
        "Collects runtime execution records from all upstream modules and serialises "
        "them as a PROV-O / ProvONE Turtle document. Records prov:Activity instances "
        "with start/end times, prov:Entity instances for all data artifacts with row "
        "counts and file paths, and prov:wasDerivedFrom / prov:wasGeneratedBy "
        "provenance chains linking every output back to its source inputs."
    ),
    script_path = "scripts/provenance_recorder.py",
    inputs = [
        PortSpec(
            name        = "execution_log",
            label       = "Execution Log",
            rdf_type    = "telwf:ExecutionLog",
            description = "In-memory list of ModuleExecution records collected during the pipeline run.",
        ),
    ],
    outputs = [
        PortSpec(
            name        = "provenance_ttl",
            label       = "Provenance Trace (Turtle)",
            rdf_type    = "xsd:anyURI",
            description = "results/provenance.ttl: PROV-O/ProvONE serialisation of the full run.",
        ),
    ],
    assumptions = [
        "Timestamps are UTC ISO 8601 strings.",
        "File sizes and row counts are recorded but not checksummed; SHA-256 checksums can be added.",
        "Agent information is limited to the host machine; no multi-user attribution is tracked.",
    ],
)


# ─────────────────────────────────────────────────────────────────────────────
# Workflow channel declarations (module-to-module data flows)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WorkflowChannel:
    """A provone:Channel connecting an out-port to an in-port."""
    channel_id:  str
    from_module: str   # module_id
    out_port:    str   # port name
    to_module:   str   # module_id
    in_port:     str   # port name

    def to_turtle(self) -> str:
        cid = f"telwf:{self.channel_id}"
        src = f"telwf:{self.from_module}_out_{self.out_port}"
        dst = f"telwf:{self.to_module}_in_{self.in_port}"
        return (
            f"{cid}\n"
            f"    a provone:Channel ;\n"
            f"    provone:connectsTo {dst} ;\n"
            f"    provone:connectsFrom {src} .\n"
        )


CHANNELS: list[WorkflowChannel] = [
    WorkflowChannel("ch01", "M1_TelemetryIngestion",  "cdh_raw",             "M2_QualityAssessment",  "cdh_raw"),
    WorkflowChannel("ch02", "M1_TelemetryIngestion",  "adcs_raw",            "M2_QualityAssessment",  "adcs_raw"),
    WorkflowChannel("ch03", "M2_QualityAssessment",   "cdh_clean",           "M3_FeatureEngineering", "cdh_clean"),
    WorkflowChannel("ch04", "M2_QualityAssessment",   "adcs_clean",          "M3_FeatureEngineering", "adcs_clean"),
    WorkflowChannel("ch05", "M2_QualityAssessment",   "cdh_clean",           "M4_FaultInjection",     "cdh_clean"),
    WorkflowChannel("ch06", "M2_QualityAssessment",   "adcs_clean",          "M4_FaultInjection",     "adcs_clean"),
    WorkflowChannel("ch07", "M2_QualityAssessment",   "cdh_clean",           "M5_AnomalyDetection",   "cdh_clean"),
    WorkflowChannel("ch08", "M2_QualityAssessment",   "adcs_clean",          "M5_AnomalyDetection",   "adcs_clean"),
    WorkflowChannel("ch09", "M4_FaultInjection",      "injected_variants",   "M5_AnomalyDetection",   "injected_variants"),
    WorkflowChannel("ch10", "M4_FaultInjection",      "label_files",         "M5_AnomalyDetection",   "label_files"),
    WorkflowChannel("ch11", "M5_AnomalyDetection",    "ml_evaluation_json",  "M6_SemanticAnnotation", "ml_evaluation_json"),
    WorkflowChannel("ch12", "M5_AnomalyDetection",    "ml_evaluation_json",  "M7_ProvenanceExport",   "execution_log"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Module registry — ordered by execution stage
# ─────────────────────────────────────────────────────────────────────────────

MODULE_REGISTRY: list[WorkflowModule] = [
    M1_INGEST,
    M2_QUALITY,
    M3_FEATURES,
    M4_FAULT_INJECTION,
    M5_DETECTION,
    M6_SEMANTIC,
    M7_PROVENANCE,
]

MODULE_BY_ID: dict[str, WorkflowModule] = {m.module_id: m for m in MODULE_REGISTRY}


# ─────────────────────────────────────────────────────────────────────────────
# Static workflow specification emission
# ─────────────────────────────────────────────────────────────────────────────

_WORKFLOW_COMMENT = """\
# TelWF — CubeSat Telemetry Analysis Workflow
# Static specification: module ports, assumptions, and data-flow channels.
# Runtime execution provenance is appended by scripts/provenance_recorder.py.
#
# Ontology alignment:
#   provone:Program  — computational module
#   provone:InPort   — input data/parameter slot
#   provone:OutPort  — output artifact slot
#   provone:Channel  — data-flow edge between ports
#   sat:hasScientificAssumption — domain assumption annotation
#   prov:Activity    — a single module execution event (runtime)
#   prov:Entity      — a data artifact (runtime)
#
# Reference:
#   Missier et al. (2013) ProvONE. DataONE Project.
#   W3C PROV-O Recommendation (2013). https://www.w3.org/TR/prov-o/
"""

_WORKFLOW_HEADER = """\
telwf:TelemetryAnalysisWorkflow
    a provone:Workflow, prov:Plan ;
    rdfs:label "TelWF: CubeSat Telemetry Analysis Workflow" ;
    dc:description "A FAIR-annotated, reproducible workflow for CubeSat telemetry "
        "ingestion, quality assessment, feature engineering, synthetic fault injection, "
        "multi-detector anomaly scoring, and ontology-linked semantic interpretation." ;
    dc:creator "Logan Luna" ;
    telwf:version "1.0" ;
"""


def emit_workflow_ttl() -> str:
    """Emit the complete static workflow specification as a Turtle string."""
    parts: list[str] = [
        _WORKFLOW_COMMENT,
        _ttl_prefix_block(),
        "",
        _WORKFLOW_HEADER,
    ]

    # Wire workflow → modules
    parts.append("    provone:hasSubProgram")
    for i, mod in enumerate(MODULE_REGISTRY):
        sep = "," if i < len(MODULE_REGISTRY) - 1 else " ."
        parts.append(f"        telwf:{mod.module_id}{sep}")
    parts.append("")

    # Individual module descriptions
    for mod in MODULE_REGISTRY:
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
