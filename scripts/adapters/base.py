"""
base.py -- Abstract base class for SensorWF domain adapters.

Each adapter implements M1 (data ingestion) for a specific domain and
returns a standardised DataFrame compatible with pipeline_core M2-M5.

The base class also provides adaptive M4 ontology generation: calling
ensure_ontology() builds a domain-specific OWL file from the adapter's
own get_class_map() and get_fault_types() — no hand-written OWL required.
Any new DomainAdapter subclass automatically gets a conformant ontology.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod

import pandas as pd


# ---------------------------------------------------------------------------
# OWL generation helpers (module-level, shared across all adapters)
# ---------------------------------------------------------------------------

def _camel_to_label(name: str) -> str:
    """Convert CamelCase identifier to a human-readable label string."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)


def _tag_to_classname(tag: str) -> str:
    """Convert a SensorWF fault tag (e.g. 'T1_obdh_runaway') to CamelCase."""
    return "".join(w.capitalize() for w in tag.split("_"))


# ---------------------------------------------------------------------------
# Abstract base adapter
# ---------------------------------------------------------------------------

class DomainAdapter(ABC):
    """
    Abstract M1 adapter: loads raw domain data and returns a clean,
    standardised DataFrame for downstream SensorWF modules.

    Required columns in returned DataFrame:
      timestamp : pd.Timestamp  (datetime index)
      elapsed_s : float          (seconds from session start)
      <channel> : float          (one column per sensor channel)

    Subclasses must implement:
      load(), get_quality_config(), get_feature_config(),
      get_ontology_path(), get_fault_types()

    Subclasses should override (optional):
      get_class_map(), get_plot_groups(), get_ontology_iri()
    """

    name: str = "AbstractDomain"
    description: str = ""
    channels: list[str] = []
    native_hz: float = 1.0

    @abstractmethod
    def load(self, path: str, **kwargs) -> pd.DataFrame:
        """Load raw file(s) → normalised DataFrame."""

    @abstractmethod
    def get_quality_config(self) -> dict:
        """Return quality-assessment config dict for pipeline_core.run_quality_assessment."""

    @abstractmethod
    def get_feature_config(self) -> dict:
        """Return feature-engineering config (window, channels subset, etc.)."""

    @abstractmethod
    def get_ontology_path(self) -> str:
        """Return path to domain OWL ontology file (written by ensure_ontology)."""

    @abstractmethod
    def get_fault_types(self) -> list[dict]:
        """Return list of fault-type dicts understood by the domain injector."""

    # ------------------------------------------------------------------
    # Optional overrides

    def get_class_map(self) -> dict[str, str]:
        """
        Return a {channel_name_or_prefix: ontology_class_uri} mapping for M4.
        Used by pipeline_core.build_semantic_kg to link features to OWL classes.
        Default: empty dict (no semantic annotation).
        """
        return {}

    def get_plot_groups(self) -> dict[str, list[str]]:
        """
        Return {group_name: [channel_names]} for generic channel overview plots.
        Used by pipeline_core.plot_channels.
        Default: all channels in a single group.
        """
        return {"Channels": list(self.channels)}

    # ------------------------------------------------------------------
    # M4 ontology: adaptive generation

    def get_ontology_iri(self) -> str:
        """
        Return the ontology IRI (without trailing #).
        Derived from get_ontology_path() stem by default.
        Override if the IRI should differ from the file name.
        """
        stem = os.path.splitext(os.path.basename(self.get_ontology_path()))[0]
        return f"https://sensorwf.org/ontologies/{stem}"

    def generate_owl_content(self) -> str:
        """
        Build a conformant RDF/XML OWL ontology from adapter metadata.

        Generates:
          - owl:Class for each unique URI in get_class_map(), as subClassOf SensorObservable
          - owl:Class for each fault type in get_fault_types(), as subClassOf SensorFault
          - featureImportance DatatypeProperty

        The result is consistent and reproducible for every domain: adding a
        new DomainAdapter subclass automatically gives it a valid ontology.
        """
        iri  = self.get_ontology_iri()
        ns   = iri + "#"
        stem = iri.rsplit("/", 1)[-1]
        pfx  = re.sub(r"[^a-z0-9]", "", stem.lower())[:8] or "domain"

        # Collect unique class local-names + which channel keys map to them
        class_channels: dict[str, list[str]] = {}
        for ch_key, cls_uri in self.get_class_map().items():
            local = cls_uri.split("#")[-1] if "#" in cls_uri else cls_uri.rsplit("/", 1)[-1]
            class_channels.setdefault(local, []).append(ch_key)

        lines: list[str] = [
            '<?xml version="1.0"?>',
            f'<rdf:RDF xmlns="{ns}"',
            f'         xml:base="{iri}"',
            f'         xmlns:{pfx}="{ns}"',
            '         xmlns:owl="http://www.w3.org/2002/07/owl#"',
            '         xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"',
            '         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"',
            '         xmlns:xsd="http://www.w3.org/2001/XMLSchema#"',
            '         xmlns:ssn="http://www.w3.org/ns/ssn/"',
            '         xmlns:sosa="http://www.w3.org/ns/sosa/">',
            '',
            f'  <owl:Ontology rdf:about="{iri}">',
            f'    <rdfs:label>SensorWF {self.name} Ontology</rdfs:label>',
            f'    <rdfs:comment>Domain ontology auto-generated by SensorWF for {self.name}. '
            f'Classes link M3 features to domain concepts via the M4 knowledge graph.</rdfs:comment>',
            '  </owl:Ontology>',
            '',
            f'  <owl:Class rdf:about="{pfx}:SensorObservable">',
            '    <rdfs:label>Sensor Observable</rdfs:label>',
            '    <rdfs:comment>A physical quantity or subsystem observable by domain sensors.</rdfs:comment>',
            '  </owl:Class>',
            '',
        ]

        for local, ch_keys in class_channels.items():
            label  = _camel_to_label(local)
            sample = ", ".join(sorted(ch_keys)[:6])
            if len(ch_keys) > 6:
                sample += f", ... ({len(ch_keys)} keys total)"
            lines += [
                f'  <owl:Class rdf:about="{pfx}:{local}">',
                f'    <rdfs:label>{label}</rdfs:label>',
                f'    <rdfs:subClassOf rdf:resource="{pfx}:SensorObservable"/>',
                f'    <rdfs:comment>Channel keys: {sample}.</rdfs:comment>',
                '  </owl:Class>',
                '',
            ]

        fault_types = self.get_fault_types()
        if fault_types:
            lines += [
                f'  <owl:Class rdf:about="{pfx}:SensorFault">',
                '    <rdfs:label>Sensor Fault</rdfs:label>',
                '    <rdfs:comment>A failure or anomaly mode injectable into domain sensor data.</rdfs:comment>',
                '  </owl:Class>',
                '',
            ]
            for ft in fault_types:
                tag  = ft.get("tag", "UnknownFault")
                desc = ft.get("description", tag)
                cls  = _tag_to_classname(tag)
                lines += [
                    f'  <owl:Class rdf:about="{pfx}:{cls}">',
                    f'    <rdfs:label>{desc}</rdfs:label>',
                    f'    <rdfs:subClassOf rdf:resource="{pfx}:SensorFault"/>',
                    f'    <rdfs:comment>SensorWF fault tag: {tag}.</rdfs:comment>',
                    '  </owl:Class>',
                    '',
                ]

        lines += [
            f'  <owl:DatatypeProperty rdf:about="{pfx}:featureImportance">',
            '    <rdfs:label>feature importance</rdfs:label>',
            '    <rdfs:range rdf:resource="xsd:double"/>',
            '    <rdfs:comment>IsolationForest mean split frequency for this feature.</rdfs:comment>',
            '  </owl:DatatypeProperty>',
            '',
            '</rdf:RDF>',
        ]
        return "\n".join(lines) + "\n"

    def ensure_ontology(self, channels: list[str] | None = None) -> str:
        """
        Ensure the domain OWL ontology exists at get_ontology_path() and return the path.

        If the file is absent, generate_owl_content() is called and the result written.
        Because content is derived from the adapter's own metadata, this is reproducible
        across runs and domain-agnostic — no per-domain OWL files need to be committed.
        """
        from scripts.pipeline_core import write_owl_if_missing
        return write_owl_if_missing(self.get_ontology_path(), self.generate_owl_content())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(channels={self.channels})"
