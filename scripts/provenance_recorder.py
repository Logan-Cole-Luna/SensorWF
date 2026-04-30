"""
provenance_recorder.py — Runtime PROV-O / ProvONE provenance recording for TelWF.

Usage pattern
─────────────
  from scripts.provenance_recorder import ProvenanceRecorder
  from scripts.workflow import M1_INGEST, MODULE_BY_ID

  rec = ProvenanceRecorder(workflow_run_id="run_20260415_001")

  # Before a module runs:
  exec_id = rec.begin_module(M1_INGEST, inputs={
      "analysis_file": "/data/Exp1/Exp1_Analysis.txt",
  })

  # ... run the module ...

  # After the module completes:
  rec.end_module(exec_id, outputs={
      "cdh_raw": {"path": None, "rows": len(cdh_raw), "columns": list(cdh_raw.columns)},
      "adcs_raw": {"path": None, "rows": len(adcs_raw)},
  })

  # At the end of the pipeline:
  rec.save("results/provenance.ttl")

Ontology alignment
──────────────────
  prov:Activity       — a module execution instance
  prov:Entity         — a data artifact (file or in-memory table)
  prov:Agent          — the software / operator that ran the workflow
  prov:wasGeneratedBy — links output Entity to its generating Activity
  prov:used           — links Activity to its input Entity
  prov:wasDerivedFrom — direct derivation chain between output and input
  prov:startedAtTime  — UTC ISO 8601 start timestamp
  prov:endedAtTime    — UTC ISO 8601 end timestamp
  provone:wasPartOf   — links Activity to the parent Workflow execution

References
──────────
  W3C PROV-O Recommendation (2013). https://www.w3.org/TR/prov-o/
  Missier et al. (2013). ProvONE: A PROV Extension for Scientific Workflows.
    DataONE Project. http://purl.dataone.org/provone-v1-dev
"""

from __future__ import annotations

import os
import socket
import platform
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from scripts.workflow import MODULE_BY_ID, PREFIXES, WorkflowModule


# ─────────────────────────────────────────────────────────────────────────────
# Execution record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModuleExecution:
    """Records one execution of a WorkflowModule."""
    exec_id:     str                  # unique URI local-name
    module_id:   str                  # WorkflowModule.module_id
    start_time:  datetime
    end_time:    datetime | None      = None
    inputs:      dict[str, Any]       = field(default_factory=dict)
    outputs:     dict[str, Any]       = field(default_factory=dict)
    status:      str                  = "running"   # running | success | error
    error_msg:   str | None           = None
    experiment:  str                  = ""          # e.g. "AccelerometerTest"


# ─────────────────────────────────────────────────────────────────────────────
# Provenance recorder
# ─────────────────────────────────────────────────────────────────────────────

class ProvenanceRecorder:
    """Collect runtime events and emit PROV-O / ProvONE Turtle."""

    def __init__(self, workflow_run_id: str | None = None,
                 workflow_uri: str | None = None, domain: str | None = None):
        self.run_id = workflow_run_id or f"run_{_now().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.domain        = domain or "generic"
        self.workflow_uri  = workflow_uri or f"sensorwf:{self.run_id}"
        self._executions: list[ModuleExecution] = []
        self._active: dict[str, ModuleExecution] = {}  # exec_id → record
        self._run_start: datetime = _now()
        self._generic_activities: list[dict] = []  # for record_activity()

    # ── Public API ────────────────────────────────────────────────────────────

    def begin_module(
        self,
        module: WorkflowModule | str,
        inputs: dict[str, Any] | None = None,
        experiment: str = "",
    ) -> str:
        """Register the start of a module execution; return exec_id."""
        if isinstance(module, str):
            module = MODULE_BY_ID[module]
        exec_id = f"{module.module_id}_{uuid.uuid4().hex[:8]}"
        rec = ModuleExecution(
            exec_id    = exec_id,
            module_id  = module.module_id,
            start_time = _now(),
            inputs     = inputs or {},
            experiment = experiment,
        )
        self._executions.append(rec)
        self._active[exec_id] = rec
        return exec_id

    def end_module(
        self,
        exec_id: str,
        outputs: dict[str, Any] | None = None,
        status: str = "success",
        error_msg: str | None = None,
    ) -> None:
        """Mark a module execution as complete."""
        rec = self._active.pop(exec_id, None)
        if rec is None:
            return
        rec.end_time = _now()
        rec.outputs  = outputs or {}
        rec.status   = status
        rec.error_msg = error_msg

    def record_activity(
        self,
        activity_name: str,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> str:
        """
        Lightweight provenance record for generic domain runners.
        Does not require a WorkflowModule -- records as a free-form activity.
        """
        exec_id = f"{activity_name}_{uuid.uuid4().hex[:8]}"
        now = _now()
        self._generic_activities.append({
            "exec_id":    exec_id,
            "name":       activity_name,
            "start_time": _fmt(now),
            "end_time":   _fmt(now),
            "inputs":     inputs or {},
            "outputs":    outputs or {},
            "params":     params or {},
            "domain":     self.domain,
        })
        return exec_id

    def record(
        self,
        module: WorkflowModule | str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        start_time: datetime,
        end_time: datetime,
        experiment: str = "",
        status: str = "success",
    ) -> str:
        """Convenience: record a completed execution in one call."""
        if isinstance(module, str):
            module = MODULE_BY_ID[module]
        exec_id = f"{module.module_id}_{uuid.uuid4().hex[:8]}"
        rec = ModuleExecution(
            exec_id    = exec_id,
            module_id  = module.module_id,
            start_time = start_time,
            end_time   = end_time,
            inputs     = inputs,
            outputs    = outputs,
            status     = status,
            experiment = experiment,
        )
        self._executions.append(rec)
        return exec_id

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_turtle(self) -> str:
        """Emit the full provenance trace as a Turtle string."""
        lines: list[str] = [_PROV_HEADER]

        # Prefix block
        for k, v in PREFIXES.items():
            lines.append(f"@prefix {k}: <{v}> .")
        lines.append("")

        # Workflow run
        run_iri = f"telwf:{self.run_id}"
        lines += [
            f"# ── Workflow run ──────────────────────────────────────────────",
            f"{run_iri}",
            f"    a provone:Execution, prov:Activity ;",
            f'    rdfs:label "TelWF pipeline run: {self.run_id}" ;',
            f"    prov:startedAtTime \"{_fmt(self._run_start)}\"^^xsd:dateTime ;",
        ]
        if self._executions:
            end_times = [r.end_time for r in self._executions if r.end_time]
            if end_times:
                run_end = max(end_times)
                lines.append(f"    prov:endedAtTime \"{_fmt(run_end)}\"^^xsd:dateTime ;")
        lines.append(f"    prov:wasAssociatedWith telwf:TelWFAgent .")
        lines.append("")

        # Software agent
        lines += [
            f"telwf:TelWFAgent",
            f"    a prov:SoftwareAgent ;",
            f'    rdfs:label "TelWF Pipeline" ;',
            f'    telwf:hostname "{socket.gethostname()}" ;',
            f'    telwf:platform "{_esc(platform.platform())}" .',
            "",
        ]

        # One prov:Activity per module execution
        entity_map: dict[str, str] = {}  # artifact key → entity IRI

        lines.append("# ── Module executions ────────────────────────────────────────")
        for rec in self._executions:
            lines += self._activity_turtle(rec, run_iri, entity_map)

        # wasDerivedFrom chains
        lines.append("# ── Derivation chains ────────────────────────────────────────")
        lines += self._derivation_turtle(entity_map)

        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Write the provenance trace to *path*."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        ttl = self.to_turtle()
        # Append generic_activities as prov:Activity stubs
        if self._generic_activities:
            ttl += "\n# ── Generic domain activities ──────────────────────────────\n"
            for act in self._generic_activities:
                aid = act["exec_id"].replace("-", "_")
                ttl += (
                    f"sensorwf:{aid}\n"
                    f'    a prov:Activity ;\n'
                    f'    rdfs:label "{act["name"]}" ;\n'
                    f'    prov:startedAtTime "{act["start_time"]}"^^xsd:dateTime ;\n'
                    f'    prov:endedAtTime "{act["end_time"]}"^^xsd:dateTime ;\n'
                    f'    sensorwf:domain "{act["domain"]}" .\n\n'
                )
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(ttl)
        n_total = len(self._executions) + len(self._generic_activities)
        print(f"  [provenance] Trace written → {path}  ({n_total} activities)")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _activity_turtle(
        self,
        rec: ModuleExecution,
        run_iri: str,
        entity_map: dict[str, str],
    ) -> list[str]:
        lines: list[str] = []
        act_iri = f"telwf:{rec.exec_id}"
        exp_tag = f"_{rec.experiment}" if rec.experiment else ""

        # Activity
        lines += [
            f"# {rec.module_id} ({rec.experiment or 'global'})",
            f"{act_iri}",
            f"    a prov:Activity, provone:Execution ;",
            f'    rdfs:label "{rec.module_id}{exp_tag}" ;',
            f"    provone:wasPartOf {run_iri} ;",
            f"    prov:startedAtTime \"{_fmt(rec.start_time)}\"^^xsd:dateTime ;",
        ]
        if rec.end_time:
            lines.append(f"    prov:endedAtTime \"{_fmt(rec.end_time)}\"^^xsd:dateTime ;")
        lines.append(f"    telwf:status \"{rec.status}\" ;")
        if rec.error_msg:
            lines.append(f"    telwf:errorMessage \"{_esc(rec.error_msg)}\" ;")

        # Link inputs as prov:used
        input_iris: list[str] = []
        for port_name, val in rec.inputs.items():
            ent_iri = self._entity_iri(rec.exec_id, port_name, "in")
            entity_map[f"{rec.exec_id}::in::{port_name}"] = ent_iri
            input_iris.append(ent_iri)

        if input_iris:
            usage_list = ", ".join(input_iris)
            lines.append(f"    prov:used {usage_list} ;")

        # Close activity
        output_iris: list[str] = []
        for port_name in rec.outputs:
            ent_iri = self._entity_iri(rec.exec_id, port_name, "out")
            entity_map[f"{rec.exec_id}::out::{port_name}"] = ent_iri
            output_iris.append(ent_iri)

        lines.append("    prov:wasAssociatedWith telwf:TelWFAgent .")
        lines.append("")

        # Input entities
        for port_name, val in rec.inputs.items():
            ent_iri = entity_map[f"{rec.exec_id}::in::{port_name}"]
            lines += self._entity_turtle(ent_iri, port_name, val, act_iri, "in")

        # Output entities
        for port_name, val in rec.outputs.items():
            ent_iri = entity_map[f"{rec.exec_id}::out::{port_name}"]
            lines += self._entity_turtle(ent_iri, port_name, val, act_iri, "out")

        return lines

    def _entity_turtle(
        self,
        ent_iri: str,
        port_name: str,
        val: Any,
        act_iri: str,
        direction: str,
    ) -> list[str]:
        lines: list[str] = [
            f"{ent_iri}",
            f"    a prov:Entity ;",
            f'    rdfs:label "{port_name}" ;',
        ]

        if isinstance(val, dict):
            for k, v in val.items():
                if v is None:
                    continue
                if isinstance(v, str):
                    lines.append(f'    telwf:{k} "{_esc(str(v))}" ;')
                elif isinstance(v, (int, float)):
                    xtype = "xsd:integer" if isinstance(v, int) else "xsd:double"
                    lines.append(f'    telwf:{k} "{v}"^^{xtype} ;')
                elif isinstance(v, list):
                    s = ", ".join(str(x) for x in v[:10])
                    lines.append(f'    telwf:{k} "{_esc(s)}" ;')
        elif isinstance(val, str):
            lines.append(f'    telwf:value "{_esc(val)}" ;')
        elif isinstance(val, (int, float)):
            xtype = "xsd:integer" if isinstance(val, int) else "xsd:double"
            lines.append(f'    telwf:value "{val}"^^{xtype} ;')

        if direction == "out":
            lines.append(f"    prov:wasGeneratedBy {act_iri} .")
        else:
            lines.append("    .")

        lines.append("")
        return lines

    @staticmethod
    def _entity_iri(exec_id: str, port_name: str, direction: str) -> str:
        safe = port_name.replace(" ", "_")
        return f"telwf:{exec_id}_{direction}_{safe}"

    def _derivation_turtle(self, entity_map: dict[str, str]) -> list[str]:
        """Emit prov:wasDerivedFrom triples for output→input cross-execution chains."""
        lines: list[str] = []
        # Group by experiment: match output port names to input port names of next modules
        # Simple heuristic: if two entities share the same port_name, link them.
        by_port: dict[str, list[tuple[str, str]]] = {}  # port_name → [(exec_id, direction), ...]
        for key, iri in entity_map.items():
            exec_id, direction, port_name = key.split("::", 2)
            by_port.setdefault(port_name, []).append((direction, iri))

        for port_name, entries in by_port.items():
            outs = [iri for d, iri in entries if d == "out"]
            ins  = [iri for d, iri in entries if d == "in"]
            for out_iri in outs:
                for in_iri in ins:
                    if out_iri == in_iri:
                        continue
                    lines.append(f"{in_iri} prov:wasDerivedFrom {out_iri} .")

        lines.append("")
        return lines


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _esc(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


_PROV_HEADER = """\
# TelWF Runtime Provenance Trace
# Format: RDF Turtle, aligned to PROV-O + ProvONE
# Generated by scripts/provenance_recorder.py
#
# Entities:     data artifacts (DataFrames, CSV files, JSON results)
# Activities:   module executions (begin/end time, inputs used, outputs generated)
# Agents:       TelWFAgent (the pipeline software)
#
# References:
#   PROV-O (W3C 2013)   https://www.w3.org/TR/prov-o/
#   ProvONE (DataONE)   http://purl.dataone.org/provone-v1-dev
"""
