"""
semantic_if_kg.py

Build an ontology-linked interpretability view for IsolationForest by combining:
1) Learned model evidence (feature importances from ml_evaluation.json)
2) Interpretable tree split/path rules from an IF model fit in no-PCA/no-rotation mode
3) Class/property structure from satellitesystem.owl

Outputs are written to results/semantic/:
- if_ontology_nodes.csv
- if_ontology_edges.csv
- if_ontology_graph.ttl
- if_interpretability_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import networkx as nx

try:
    import shap
except Exception:
    shap = None

from scripts.evaluator import IsolationForestDetector, build_combined_feature_matrix, build_feature_matrix


@dataclass
class OntologyInfo:
    classes: set[str]
    object_properties: set[str]
    subclass_edges: list[tuple[str, str]]
    domain_edges: list[tuple[str, str]]
    range_edges: list[tuple[str, str]]


def _local_name(iri: str) -> str:
    if not isinstance(iri, str):
        return ""
    if "#" in iri:
        return iri.rsplit("#", 1)[-1]
    if "/" in iri:
        return iri.rstrip("/").rsplit("/", 1)[-1]
    return iri


def parse_ontology(owl_path: str) -> OntologyInfo:
    ns = {
        "owl": "http://www.w3.org/2002/07/owl#",
    }
    root = ET.parse(owl_path).getroot()

    classes: set[str] = set()
    obj_props: set[str] = set()
    subclass_edges: list[tuple[str, str]] = []
    domain_edges: list[tuple[str, str]] = []
    range_edges: list[tuple[str, str]] = []

    for c in root.findall(".//owl:Class", ns):
        iri = c.attrib.get("IRI", "")
        nm = _local_name(iri)
        if nm:
            classes.add(nm)

    for p in root.findall(".//owl:ObjectProperty", ns):
        iri = p.attrib.get("IRI", "")
        nm = _local_name(iri)
        if nm:
            obj_props.add(nm)

    for ax in root.findall(".//owl:SubClassOf", ns):
        kids = ax.findall("owl:Class", ns)
        if len(kids) >= 2:
            a = _local_name(kids[0].attrib.get("IRI", ""))
            b = _local_name(kids[1].attrib.get("IRI", ""))
            if a and b:
                subclass_edges.append((a, b))

    for ax in root.findall(".//owl:ObjectPropertyDomain", ns):
        p = ax.find("owl:ObjectProperty", ns)
        c = ax.find("owl:Class", ns)
        if p is None or c is None:
            continue
        pn = _local_name(p.attrib.get("IRI", ""))
        cn = _local_name(c.attrib.get("IRI", ""))
        if pn and cn:
            domain_edges.append((pn, cn))

    for ax in root.findall(".//owl:ObjectPropertyRange", ns):
        p = ax.find("owl:ObjectProperty", ns)
        c = ax.find("owl:Class", ns)
        if p is None or c is None:
            continue
        pn = _local_name(p.attrib.get("IRI", ""))
        cn = _local_name(c.attrib.get("IRI", ""))
        if pn and cn:
            range_edges.append((pn, cn))

    return OntologyInfo(
        classes=classes,
        object_properties=obj_props,
        subclass_edges=subclass_edges,
        domain_edges=domain_edges,
        range_edges=range_edges,
    )


def _normalize_feature_name(name: str) -> str:
    s = str(name)
    for p in ("cdh::", "adcs::"):
        if s.startswith(p):
            s = s[len(p):]
    changed = True
    while changed:
        changed = False
        for p in ("d_", "rm_", "rs_"):
            if s.startswith(p):
                s = s[len(p):]
                changed = True
    return s


def _feature_semantic_map(feature: str, available_classes: set[str]) -> tuple[str, str, str, str]:
    f = _normalize_feature_name(feature).upper()

    if "ACCEL" in f:
        sensor, meas, subsystem, sem = "Accelerometer", "AccelerationMeas", "ADCS", "inertial_acceleration"
    elif "GYRO" in f:
        sensor, meas, subsystem, sem = "Gyroscope", "AngularVelMeas", "ADCS", "inertial_angular_velocity"
    elif "MAG_" in f:
        sensor, meas, subsystem, sem = "Magnetometer", "MagneticFieldMeas", "ADCS", "magnetic_field"
    elif "SUN" in f:
        sensor, meas, subsystem, sem = "SunSensor", "SunAngleMeas", "ADCS", "sun_geometry"
    elif "WHEEL" in f:
        sensor, meas, subsystem, sem = "ReactionWheel", "WheelSpdMeas", "ADCS", "wheel_dynamics"
    elif any(k in f for k in ["FRAME_ERR", "CMD_REJ", "REJECT", "CNT"]):
        sensor = "Sensor"
        meas = "Counter" if "Counter" in available_classes else "Measurement"
        subsystem = "COMMS"
        sem = "counter_diagnostics"
    elif any(k in f for k in ["RSSI", "COMMS", "DT_PACKET", "PACKET", "LATENCY"]):
        sensor, meas, subsystem, sem = "Sensor", "Measurement", "COMMS", "link_quality_timing"
    elif any(k in f for k in ["TEMP_ADCS", "TEMP_WHEEL"]):
        sensor, meas, subsystem, sem = "Sensor", "Measurement", "ADCS", "thermal_adcs"
    elif any(k in f for k in ["TEMP", "V", "I", "PWR", "BATTERY", "SYS"]):
        sensor, meas, subsystem, sem = "Sensor", "Measurement", "EPS_Thermal_CDH", "power_thermal_housekeeping"
    else:
        sensor, meas, subsystem, sem = "Sensor", "Measurement", "Unknown", "untyped"

    if sensor not in available_classes:
        sensor = "Sensor" if "Sensor" in available_classes else sensor
    if meas not in available_classes:
        meas = "Measurement" if "Measurement" in available_classes else meas

    return sensor, meas, subsystem, sem


def _transform_for_model(detector: IsolationForestDetector, X: np.ndarray) -> np.ndarray:
    if detector._feature_idx is None:
        return X
    Xt = X[:, detector._feature_idx]
    Xt = detector._scaler.transform(Xt)
    if detector._pca is not None:
        Xt = detector._pca.transform(Xt)
    return Xt


def _list_families(results_dir: str) -> list[str]:
    out: list[str] = []
    for entry in sorted(os.listdir(results_dir)):
        if entry == "aggregated" or entry == "semantic":
            continue
        p = os.path.join(results_dir, entry)
        if os.path.isdir(p):
            out.append(entry)
    return out


def load_isolationforest_rows(results_dir: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fam in _list_families(results_dir):
        p = os.path.join(results_dir, fam, "injected", "ml_evaluation.json")
        if not os.path.exists(p):
            continue
        try:
            data = json.loads(open(p, "r", encoding="utf-8").read())
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for r in data:
            if r.get("detector") != "IsolationForest":
                continue
            if r.get("error"):
                continue
            imps = r.get("feature_importances")
            names = r.get("feature_names")
            if not isinstance(imps, list) or not isinstance(names, list):
                continue
            if not imps or not names:
                continue
            n = min(len(imps), len(names))
            if n <= 0:
                continue
            for i in range(n):
                rows.append(
                    {
                        "family": fam,
                        "tag": str(r.get("tag", "unknown")),
                        "tier": str(r.get("tier", "unknown")),
                        "feature": str(names[i]),
                        "importance": float(imps[i]),
                        "f1": float(r.get("f1", np.nan)),
                        "recall": float(r.get("recall", np.nan)),
                        "fpr": float(r.get("fpr", np.nan)),
                    }
                )
    return pd.DataFrame(rows)


def _parse_target_from_tag(tag: str) -> str:
    t = str(tag)
    if t.startswith("COMPOUND_adcs_power_fault"):
        return "cdh+adcs"
    if t.startswith("A") or t.startswith("W"):
        return "adcs"
    return "cdh"


def _load_injected_case(injected_dir: str, tag: str) -> tuple[str, pd.DataFrame | None, pd.DataFrame | None, tuple[int, int] | None]:
    # Prefer medium v1, then easy v1, then hard v1.
    for tier in ["medium", "easy", "hard"]:
        label_p = os.path.join(injected_dir, f"labels_{tag}_{tier}_v1.csv")
        if not os.path.exists(label_p):
            continue
        try:
            lbl = pd.read_csv(label_p).iloc[0]
            start_idx = int(lbl.get("start_idx", 0))
            end_idx = int(lbl.get("end_idx", 0))
        except Exception:
            start_idx, end_idx = 0, 0

        cdh_p = os.path.join(injected_dir, f"cdh_injected_{tag}_{tier}_v1.csv")
        adcs_p = os.path.join(injected_dir, f"adcs_injected_{tag}_{tier}_v1.csv")
        cdh_df = pd.read_csv(cdh_p, low_memory=False) if os.path.exists(cdh_p) else None
        adcs_df = pd.read_csv(adcs_p, low_memory=False) if os.path.exists(adcs_p) else None

        if cdh_df is not None and adcs_df is not None:
            return "cdh+adcs", cdh_df, adcs_df, (start_idx, end_idx)
        if adcs_df is not None:
            return "adcs", None, adcs_df, (start_idx, end_idx)
        if cdh_df is not None:
            return "cdh", cdh_df, None, (start_idx, end_idx)
    return "", None, None, None


def _extract_tree_path_rules(detector: IsolationForestDetector, sample: np.ndarray, top_k: int = 8) -> list[tuple[str, int]]:
    if not detector._models:
        return []

    model = detector._models[0]
    rot = detector._rotations[0]
    xs = sample.reshape(1, -1)
    if detector._feature_idx is None:
        return []
    xs = xs[:, detector._feature_idx]
    xs = detector._scaler.transform(xs)
    if detector._pca is not None:
        xs = detector._pca.transform(xs)
    rules = Counter()

    for est in model.estimators_:
        tree = est.tree_
        x_rot = xs @ rot
        node_path = est.decision_path(x_rot).indices
        for node_id in node_path:
            feat = int(tree.feature[node_id])
            thr = float(tree.threshold[node_id])
            if feat < 0:
                continue
            if feat >= x_rot.shape[1]:
                continue
            fname = detector.feature_names_[feat] if feat < len(detector.feature_names_) else f"f{feat}"
            direction = "<=" if float(x_rot[0, feat]) <= thr else ">"
            key = f"{fname} {direction} {thr:.3f}"
            rules[key] += 1

    return rules.most_common(top_k)


def _collect_split_importance(detector: IsolationForestDetector, top_k: int = 10) -> list[tuple[str, int]]:
    if not detector._models:
        return []

    counts = Counter()
    for model in detector._models:
        for est in model.estimators_:
            tree = est.tree_
            for feat in tree.feature:
                f = int(feat)
                if f < 0:
                    continue
                fname = detector.feature_names_[f] if f < len(detector.feature_names_) else f"f{f}"
                counts[fname] += 1
    return counts.most_common(top_k)


def _local_shap_contribs(
    detector: IsolationForestDetector,
    X_train: np.ndarray,
    x_instance: np.ndarray,
    top_k: int = 12,
) -> list[dict[str, float | str]]:
    """Return top local per-feature contributions for one instance.

    Uses TreeSHAP when available. Returns empty list if SHAP is unavailable
    or computation fails.
    """
    if shap is None or not detector._models:
        return []
    try:
        model = detector._models[0]
        rot = detector._rotations[0]
        Xt_train = _transform_for_model(detector, X_train)
        Xt_inst = _transform_for_model(detector, x_instance.reshape(1, -1))
        Xt_train_rot = Xt_train @ rot
        Xt_inst_rot = Xt_inst @ rot

        explainer = shap.TreeExplainer(model, data=Xt_train_rot)
        sv = explainer.shap_values(Xt_inst_rot)
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[0]
        if sv.ndim == 2:
            sv = sv[0]
        if sv.ndim != 1:
            return []

        feats = detector.feature_names_
        n = min(len(feats), len(sv))
        if n == 0:
            return []
        abs_ord = np.argsort(np.abs(sv[:n]))[::-1][:top_k]
        out: list[dict[str, float | str]] = []
        for i in abs_ord:
            out.append(
                {
                    "feature": str(feats[i]),
                    "shap_value": float(sv[i]),
                    "abs_shap_value": float(abs(sv[i])),
                }
            )
        return out
    except Exception:
        return []


def fit_interpretable_if_rules(results_dir: str, max_tags_per_family: int = 12) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for fam in _list_families(results_dir):
        fam_dir = os.path.join(results_dir, fam)
        injected_dir = os.path.join(fam_dir, "injected")
        cdh_clean_p = os.path.join(fam_dir, "cdh_clean.csv")
        adcs_clean_p = os.path.join(fam_dir, "adcs_clean.csv")
        if not (os.path.exists(cdh_clean_p) and os.path.exists(adcs_clean_p) and os.path.isdir(injected_dir)):
            continue

        cdh_clean = pd.read_csv(cdh_clean_p, low_memory=False)
        adcs_clean = pd.read_csv(adcs_clean_p, low_memory=False)

        label_files = sorted(
            f for f in os.listdir(injected_dir)
            if f.startswith("labels_") and f.endswith("_medium_v1.csv")
        )
        tags = []
        for lf in label_files:
            m = re.match(r"labels_(.+)_medium_v1\.csv$", lf)
            if not m:
                continue
            tags.append(m.group(1))
        tags = sorted(set(tags))[:max_tags_per_family]

        for tag in tags:
            target, cdh_inj, adcs_inj, win = _load_injected_case(injected_dir, tag)
            if not target:
                continue

            if target == "cdh+adcs":
                X_clean, fnames = build_combined_feature_matrix(cdh_clean, adcs_clean)
                X_inj, _ = build_combined_feature_matrix(cdh_inj, adcs_inj)
            elif target == "adcs":
                X_clean, fnames = build_feature_matrix(adcs_clean, "adcs")
                X_inj, _ = build_feature_matrix(adcs_inj, "adcs")
            else:
                X_clean, fnames = build_feature_matrix(cdh_clean, "cdh")
                X_inj, _ = build_feature_matrix(cdh_inj, "cdh")

            if X_clean is None or X_inj is None:
                continue
            if len(X_clean) < 20 or len(X_inj) < 10:
                continue

            n_train = max(20, int(0.6 * len(X_clean)))
            X_train = X_clean[:n_train]

            det = IsolationForestDetector(
                contamination=0.03,
                n_estimators=250,
                use_pca=False,
                use_rotation=False,
                n_ensemble=1,
            )
            det.fit(X_train, fnames)

            scores = det.score(X_inj)
            thr = float(np.percentile(det.score(X_train), 99.0))
            preds = scores >= thr

            sample_idx = int(np.argmax(scores))
            if win is not None:
                lo = max(0, min(win[0], len(scores) - 1))
                hi = max(lo, min(win[1], len(scores) - 1))
                cand = np.where(preds[lo:hi + 1])[0]
                if len(cand) > 0:
                    sample_idx = lo + int(cand[0])

            top_rules = _extract_tree_path_rules(det, X_inj[sample_idx], top_k=8)
            top_splits = _collect_split_importance(det, top_k=10)
            local_contribs = _local_shap_contribs(det, X_train, X_inj[sample_idx], top_k=12)

            rows.append(
                {
                    "family": fam,
                    "tag": tag,
                    "target": target,
                    "sample_idx": sample_idx,
                    "top_rules": top_rules,
                    "top_splits": top_splits,
                    "local_contribs": local_contribs,
                }
            )

    return rows


def build_semantic_graph(
    results_dir: str,
    ontology_path: str,
    output_dir: str,
    max_tags_per_family: int = 12,
) -> dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    onto = parse_ontology(ontology_path)
    df_if = load_isolationforest_rows(results_dir)
    rule_rows = fit_interpretable_if_rules(results_dir, max_tags_per_family=max_tags_per_family)

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def add_node(node_id: str, node_type: str, label: str, source: str) -> None:
        if node_id not in nodes:
            nodes[node_id] = {
                "node_id": node_id,
                "node_type": node_type,
                "label": label,
                "source": source,
            }

    def add_edge(src: str, pred: str, dst: str, weight: float | None = None, context: str = "") -> None:
        edges.append(
            {
                "src": src,
                "predicate": pred,
                "dst": dst,
                "weight": np.nan if weight is None else float(weight),
                "context": context,
            }
        )

    for c in sorted(onto.classes):
        add_node(f"ont:class:{c}", "OntologyClass", c, "satellitesystem.owl")
    for p in sorted(onto.object_properties):
        add_node(f"ont:property:{p}", "OntologyProperty", p, "satellitesystem.owl")

    for child, parent in onto.subclass_edges:
        add_edge(f"ont:class:{child}", "rdfs:subClassOf", f"ont:class:{parent}")
    for prop, dom in onto.domain_edges:
        add_edge(f"ont:property:{prop}", "rdfs:domain", f"ont:class:{dom}")
    for prop, ran in onto.range_edges:
        add_edge(f"ont:property:{prop}", "rdfs:range", f"ont:class:{ran}")

    if not df_if.empty:
        agg = (
            df_if.groupby(["family", "tag", "feature"], as_index=False)
            .agg(mean_importance=("importance", "mean"), mean_f1=("f1", "mean"), mean_recall=("recall", "mean"), mean_fpr=("fpr", "mean"))
        )

        for _, r in agg.iterrows():
            fam = str(r["family"])
            tag = str(r["tag"])
            feat = str(r["feature"])
            imp = float(r["mean_importance"])

            fam_id = f"if:family:{fam}"
            tag_id = f"if:tag:{tag}"
            feat_id = f"if:feature:{feat}"

            add_node(fam_id, "Family", fam, "ml_evaluation.json")
            add_node(tag_id, "AnomalyTag", tag, "ml_evaluation.json")
            add_node(feat_id, "Feature", feat, "ml_evaluation.json")

            add_edge(fam_id, "if:hasTag", tag_id)
            add_edge(tag_id, "if:importantFeature", feat_id, weight=imp, context=fam)

            sensor_cls, meas_cls, subsystem, sem_type = _feature_semantic_map(feat, onto.classes)
            subsys_id = f"if:subsystem:{subsystem}"
            sem_id = f"if:semanticType:{sem_type}"
            add_node(subsys_id, "Subsystem", subsystem, "heuristic")
            add_node(sem_id, "SemanticType", sem_type, "heuristic")
            add_node(f"ont:class:{sensor_cls}", "OntologyClass", sensor_cls, "satellitesystem.owl")
            add_node(f"ont:class:{meas_cls}", "OntologyClass", meas_cls, "satellitesystem.owl")

            add_edge(feat_id, "if:mapsToSensorClass", f"ont:class:{sensor_cls}")
            add_edge(feat_id, "if:mapsToMeasurementClass", f"ont:class:{meas_cls}")
            add_edge(feat_id, "if:belongsToSubsystem", subsys_id)
            add_edge(feat_id, "if:hasSemanticType", sem_id)

    for rr in rule_rows:
        fam = rr["family"]
        tag = rr["tag"]
        fam_id = f"if:family:{fam}"
        tag_id = f"if:tag:{tag}"
        add_node(fam_id, "Family", fam, "interpretable_if_fit")
        add_node(tag_id, "AnomalyTag", tag, "interpretable_if_fit")
        add_edge(fam_id, "if:hasTag", tag_id)

        for feat, cnt in rr.get("top_splits", []):
            feat_id = f"if:feature:{feat}"
            add_node(feat_id, "Feature", feat, "interpretable_if_fit")
            add_edge(tag_id, "if:treeSplitFeature", feat_id, weight=float(cnt), context=fam)

            sensor_cls, meas_cls, subsystem, sem_type = _feature_semantic_map(feat, onto.classes)
            subsys_id = f"if:subsystem:{subsystem}"
            sem_id = f"if:semanticType:{sem_type}"
            add_node(subsys_id, "Subsystem", subsystem, "heuristic")
            add_node(sem_id, "SemanticType", sem_type, "heuristic")
            add_edge(feat_id, "if:belongsToSubsystem", subsys_id)
            add_edge(feat_id, "if:mapsToSensorClass", f"ont:class:{sensor_cls}")
            add_edge(feat_id, "if:mapsToMeasurementClass", f"ont:class:{meas_cls}")
            add_edge(feat_id, "if:hasSemanticType", sem_id)

        for rule, cnt in rr.get("top_rules", []):
            rule_id = f"if:rule:{fam}:{tag}:{abs(hash(rule))}"
            add_node(rule_id, "TreeRule", rule, "interpretable_if_fit")
            add_edge(tag_id, "if:explainedByRule", rule_id, weight=float(cnt), context=fam)
            m = re.match(r"(.+)\s(<=|>)\s[-0-9.]+$", rule)
            if m:
                feat = m.group(1)
                feat_id = f"if:feature:{feat}"
                add_node(feat_id, "Feature", feat, "interpretable_if_fit")
                add_edge(rule_id, "if:usesFeature", feat_id)

        instance_id = f"if:instance:{fam}:{tag}:{rr.get('sample_idx', 0)}"
        add_node(instance_id, "AnomalyInstance", f"{fam}:{tag}:{rr.get('sample_idx', 0)}", "local_shap")
        add_edge(tag_id, "if:hasInstance", instance_id, context=fam)
        for lc in rr.get("local_contribs", []):
            feat = str(lc.get("feature", ""))
            if not feat:
                continue
            feat_id = f"if:feature:{feat}"
            add_node(feat_id, "Feature", feat, "local_shap")
            add_edge(
                instance_id,
                "if:localContributionAbs",
                feat_id,
                weight=float(lc.get("abs_shap_value", 0.0)),
                context=fam,
            )
            add_edge(
                instance_id,
                "if:localContributionSigned",
                feat_id,
                weight=float(lc.get("shap_value", 0.0)),
                context=fam,
            )

    nodes_path = os.path.join(output_dir, "if_ontology_nodes.csv")
    edges_path = os.path.join(output_dir, "if_ontology_edges.csv")

    with open(nodes_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["node_id", "node_type", "label", "source"])
        writer.writeheader()
        for n in sorted(nodes.values(), key=lambda x: x["node_id"]):
            writer.writerow(n)

    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["src", "predicate", "dst", "weight", "context"])
        writer.writeheader()
        for e in edges:
            writer.writerow(e)

    ttl_path = os.path.join(output_dir, "if_ontology_graph.ttl")
    with open(ttl_path, "w", encoding="utf-8") as f:
        f.write("@prefix if: <http://example.org/ifkg#> .\n")
        f.write("@prefix ont: <http://www.semanticweb.org/satellite-system/> .\n")
        f.write("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\n")

        for n in sorted(nodes.values(), key=lambda x: x["node_id"]):
            nid = n["node_id"].replace(" ", "_")
            ntype = n["node_type"].replace(" ", "_")
            label = str(n["label"]).replace('"', "'")
            f.write(f"if:{abs(hash(nid))} a if:{ntype} ; rdfs:label \"{label}\" .\n")

        f.write("\n")
        node_hash = {k: f"if:{abs(hash(k.replace(' ', '_')))}" for k in nodes}
        for e in edges:
            s = node_hash.get(e["src"])
            o = node_hash.get(e["dst"])
            if not s or not o:
                continue
            pred = str(e["predicate"]).replace(":", "_")
            f.write(f"{s} if:{pred} {o} .\n")

    graphml_path = os.path.join(output_dir, "if_ontology_graph.graphml")
    g = nx.DiGraph()
    for n in nodes.values():
        g.add_node(
            n["node_id"],
            node_type=n["node_type"],
            label=n["label"],
            source=n["source"],
        )
    for i, e in enumerate(edges):
        g.add_edge(
            e["src"],
            e["dst"],
            key=f"e{i}",
            predicate=e["predicate"],
            weight=e["weight"],
            context=e["context"],
        )
    nx.write_graphml(g, graphml_path)

    report_path = os.path.join(output_dir, "if_interpretability_report.md")
    _write_report(report_path, df_if, rule_rows)

    return {
        "nodes": nodes_path,
        "edges": edges_path,
        "ttl": ttl_path,
        "graphml": graphml_path,
        "report": report_path,
    }


def _expected_subsystem(tag: str) -> str:
    t = str(tag)
    if t.startswith(("A", "W")) or "adcs" in t:
        return "ADCS"
    if t.startswith("C"):
        return "COMMS"
    if t.startswith(("T", "P")):
        return "EPS_Thermal_CDH"
    return "Unknown"


def _write_report(path: str, df_if: pd.DataFrame, rule_rows: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# IsolationForest + Ontology Interpretability Report")
    lines.append("")
    lines.append("This report links IsolationForest evidence (feature importances and split/path rules) to ontology classes in satellitesystem.owl.")
    lines.append("")

    if df_if.empty:
        lines.append("No IsolationForest rows found in ml_evaluation artifacts.")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return

    agg = (
        df_if.groupby(["family", "tag", "feature"], as_index=False)
        .agg(mean_importance=("importance", "mean"))
    )

    lines.append("## Top Features by Anomaly Tag")
    lines.append("")
    for (fam, tag), g in agg.groupby(["family", "tag"]):
        top = g.sort_values("mean_importance", ascending=False).head(5)
        lines.append(f"### {fam} / {tag}")
        for _, r in top.iterrows():
            lines.append(f"- {r['feature']}: mean importance {float(r['mean_importance']):.4f}")
        lines.append("")

    lines.append("## Interpretable Tree-Path Rules (No PCA/No Rotation IF)")
    lines.append("")
    if not rule_rows:
        lines.append("No tree-path rules were generated.")
    else:
        for rr in rule_rows:
            fam = rr["family"]
            tag = rr["tag"]
            lines.append(f"### {fam} / {tag}")
            lines.append("Top split features:")
            for feat, cnt in rr.get("top_splits", [])[:6]:
                lines.append(f"- {feat}: split count {cnt}")
            lines.append("Top path rules:")
            for rule, cnt in rr.get("top_rules", [])[:6]:
                lines.append(f"- {rule} (count {cnt})")
            lines.append("")

    lines.append("## Ontology-Consistency Comparison")
    lines.append("")
    lines.append("Heuristic comparison: expected subsystem by anomaly tag prefix vs observed dominant subsystem from top features.")
    lines.append("")

    by_tag = agg.groupby(["tag", "feature"], as_index=False)["mean_importance"].mean()
    for tag, g in by_tag.groupby("tag"):
        scores = defaultdict(float)
        for _, r in g.iterrows():
            feat = str(r["feature"])
            _, _, sub, _ = _feature_semantic_map(feat, {"Sensor", "Measurement"})
            scores[sub] += float(r["mean_importance"])
        if not scores:
            continue
        dominant = sorted(scores.items(), key=lambda x: x[1], reverse=True)[0][0]
        expected = _expected_subsystem(tag)
        ok = "yes" if (expected == "Unknown" or expected == dominant) else "partial/no"
        lines.append(f"- {tag}: expected={expected}, dominant={dominant}, consistency={ok}")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    if rule_rows:
        top_local = Counter()
        for rr in rule_rows:
            for lc in rr.get("local_contribs", [])[:5]:
                top_local[str(lc.get("feature", ""))] += float(lc.get("abs_shap_value", 0.0))
        if top_local:
            lines.append("Top local (per-instance) contribution features from TreeSHAP-like analysis:")
            for feat, v in top_local.most_common(8):
                lines.append(f"- {feat}: aggregated abs contribution {v:.4f}")
            lines.append("")

    lines.append("- The production detector uses PCA and random rotations for performance/robustness.")
    lines.append("- For rule extraction, this report fits an interpretable IF mode (no PCA, no rotation) to recover direct feature split rules.")
    lines.append("- This yields human-readable rules that can be attached to ontology classes and subsystem concepts.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build ontology-linked IF interpretability knowledge graph.")
    ap.add_argument("--results-dir", default="results", help="Path to results directory")
    ap.add_argument("--ontology", default="satellitesystem.owl", help="Path to ontology OWL file")
    ap.add_argument("--output-dir", default=os.path.join("results", "semantic"), help="Output directory")
    ap.add_argument("--max-tags-per-family", type=int, default=12, help="Max tags per family for path-rule extraction")
    args = ap.parse_args()

    out = build_semantic_graph(
        results_dir=args.results_dir,
        ontology_path=args.ontology,
        output_dir=args.output_dir,
        max_tags_per_family=args.max_tags_per_family,
    )

    print("Semantic IF-KG artifacts:")
    for k, v in out.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
