"""
html_report.py — Lightweight academic-style HTML run report for SensorWF.

Reads existing result artifacts from the results directory and assembles a
single self-contained HTML file.  Plots are included selectively (summary
dashboard + at most two supplemental figures per family) to keep file size
reasonable.  Opens the file in the default browser.

Usage
-----
  python -m scripts.utils.html_report [--results-dir results/satellite]
                                      [--output report.html]
                                      [--no-browser]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _b64_png(path: str) -> str | None:
    try:
        with open(path, "rb") as fh:
            return "data:image/png;base64," + base64.b64encode(fh.read()).decode()
    except OSError:
        return None


def _plot_label(stem: str) -> str:
    s = re.sub(r"^\d+_", "", stem)
    s = s.replace("_", " ").replace("-", " ")
    return s.title()


def _img_tag(path: str) -> str:
    uri = _b64_png(path)
    if not uri:
        return f'<p class="na">Figure not found: {Path(path).name}</p>'
    label = _plot_label(Path(path).stem)
    return (
        f'<figure>'
        f'<img src="{uri}" alt="{label}">'
        f'<figcaption>{label}</figcaption>'
        f'</figure>'
    )


def _read_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _read_csv(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _fmt(v: Any, decimals: int = 3) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def _df_to_html(df: pd.DataFrame, max_rows: int = 25) -> str:
    if df is None or df.empty:
        return "<p class='na'>No data available.</p>"
    df = df.head(max_rows).round(4)
    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{_fmt(v)}</td>" for v in row) + "</tr>"
        for row in df.itertuples(index=False)
    )
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>"


# ── Plot selection ────────────────────────────────────────────────────────────

_PREFERRED_PLOTS = [
    "07_summary_dashboard",
    "01_board_temperatures",
    "03_power_rails",
    "08_temperature_distributions",
    "16_imu_sensors",
    "17_imu_magnitudes",
]

def _select_plots(fam_dir: str, max_plots: int = 3) -> list[Path]:
    try:
        all_pngs = {p.stem: p for p in Path(fam_dir).glob("*.png")}
    except OSError:
        return []
    chosen: list[Path] = []
    for pref in _PREFERRED_PLOTS:
        if pref in all_pngs:
            chosen.append(all_pngs[pref])
        if len(chosen) >= max_plots:
            break
    if not chosen:
        chosen = sorted(all_pngs.values())[:max_plots]
    return chosen


# ── KG helpers ────────────────────────────────────────────────────────────────

def _find_kg_csvs(results_dir: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Search results_dir and its first two levels of subdirs for *_kg_nodes/edges CSVs."""
    all_nodes: list[pd.DataFrame] = []
    all_edges: list[pd.DataFrame] = []

    def _scan(directory: str) -> bool:
        found = False
        try:
            for fname in sorted(os.listdir(directory)):
                fpath = os.path.join(directory, fname)
                if fname.endswith("_kg_nodes.csv"):
                    df = _read_csv(fpath)
                    if df is not None and not df.empty:
                        all_nodes.append(df)
                        found = True
                elif fname.endswith("_kg_edges.csv"):
                    df = _read_csv(fpath)
                    if df is not None and not df.empty:
                        all_edges.append(df)
                        found = True
        except OSError:
            pass
        return found

    # Try results_dir/semantic/ first
    _scan(os.path.join(results_dir, "semantic"))

    # Try results_dir/ itself
    if not all_nodes:
        _scan(results_dir)

    # Recurse one level into subdirs, then into their semantic/ dirs
    if not all_nodes:
        try:
            subdirs = sorted(
                e.path for e in os.scandir(results_dir)
                if e.is_dir() and e.name not in {"semantic", "injected", "plots"}
            )
        except OSError:
            subdirs = []
        for sub in subdirs:
            _scan(os.path.join(sub, "semantic")) or _scan(sub)
            if all_nodes:
                break  # Use first sub-family's KG as representative

    if not all_nodes:
        return None, None

    merged_nodes = pd.concat(all_nodes, ignore_index=True)
    if "node_id" in merged_nodes.columns:
        merged_nodes = merged_nodes.drop_duplicates(subset=["node_id"])

    merged_edges = pd.concat(all_edges, ignore_index=True) if all_edges else pd.DataFrame()
    return merged_nodes, merged_edges


def _owl_kg_fallback(owl_path: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Build a minimal class-only KG from OWL when no CSV files are found."""
    if not os.path.isfile(owl_path):
        return None, None
    try:
        import xml.etree.ElementTree as ET
        root = ET.parse(owl_path).getroot()
    except Exception:
        return None, None

    OWL  = "http://www.w3.org/2002/07/owl#"
    RDF  = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    RDFS = "http://www.w3.org/2000/01/rdf-schema#"

    def _local(uri: str) -> str:
        if "#" in uri: return uri.split("#")[-1]
        return uri.rsplit(":", 1)[-1] if ":" in uri else uri

    nodes, edges = [], []
    for cls_el in root.findall(f".//{{{OWL}}}Class"):
        about = cls_el.get(f"{{{RDF}}}about", "")
        local = _local(about)
        if not local or local in ("SensorObservable", "SensorFault", ""):
            continue
        label_el = cls_el.find(f"{{{RDFS}}}label")
        label    = label_el.text if label_el is not None and label_el.text else local
        sub_el   = cls_el.find(f"{{{RDFS}}}subClassOf")
        parent   = _local(sub_el.get(f"{{{RDF}}}resource", "") if sub_el is not None else "")
        ntype    = "fault_class" if "Fault" in parent else "observable_class"
        nodes.append({"node_id": local, "node_type": ntype, "label": label})
        if parent and parent not in ("SensorObservable", "SensorFault"):
            edges.append({"source": local, "target": parent, "relation": "subClassOf", "weight": 1.0})

    if not nodes:
        return None, None
    return pd.DataFrame(nodes), pd.DataFrame(edges) if edges else pd.DataFrame()


def _kg_graph_html(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, canvas_id: str = "kg-cv") -> str:
    """Render a self-contained canvas + vanilla-JS force-directed graph."""
    if nodes_df is None or nodes_df.empty:
        return "<p class='na'>No knowledge graph data found.</p>"

    type_col = "node_type" if "node_type" in nodes_df.columns else None

    def _is_class(ntype: str) -> bool:
        return ntype in ("ontologyclass", "observable_class", "fault_class")

    if type_col:
        class_mask = nodes_df[type_col].apply(_is_class)
    else:
        class_mask = pd.Series([False] * len(nodes_df))

    class_nodes = nodes_df[class_mask].copy()
    feat_nodes  = nodes_df[~class_mask].copy()

    # Limit feature nodes to top-40 by edge weight to avoid clutter
    if not feat_nodes.empty and edges_df is not None and not edges_df.empty:
        weight_col = "weight" if "weight" in edges_df.columns else None
        src_col    = "source" if "source" in edges_df.columns else None
        if weight_col and src_col:
            top = edges_df.groupby(src_col)[weight_col].sum().nlargest(40).index.tolist()
            feat_nodes = feat_nodes[feat_nodes["node_id"].isin(top)]
        else:
            feat_nodes = feat_nodes.head(40)
    elif not feat_nodes.empty:
        feat_nodes = feat_nodes.head(40)

    vis_df  = pd.concat([class_nodes, feat_nodes], ignore_index=True)
    idx_map = {nid: i for i, nid in enumerate(vis_df["node_id"])}

    def _short_label(nid: str, raw_label: str) -> str:
        label = str(raw_label) if raw_label else ""
        if label and label != str(nid) and "//" not in label and "#" not in label:
            return label[:24]
        for sep in ("#", "/", ":"):
            if sep in str(nid):
                return str(nid).rsplit(sep, 1)[-1][:24]
        return str(nid)[:24]

    vis_nodes = []
    for _, row in vis_df.iterrows():
        nid   = row["node_id"]
        ntype = row.get("node_type", "feature") if type_col else "feature"
        color = (
            "#4a90d9" if ntype == "observable_class" else
            "#d95555" if ntype == "fault_class"      else
            "#5aad6e" if _is_class(ntype)             else
            "#b0b0b0"
        )
        vis_nodes.append({
            "label": _short_label(str(nid), str(row.get("label", ""))),
            "cls":   _is_class(ntype),
            "color": color,
        })

    vis_edges = []
    if edges_df is not None and not edges_df.empty:
        src_col = "source" if "source" in edges_df.columns else None
        tgt_col = "target" if "target" in edges_df.columns else None
        if src_col and tgt_col:
            for _, row in edges_df.iterrows():
                s = idx_map.get(row[src_col])
                t = idx_map.get(row[tgt_col])
                if s is not None and t is not None:
                    vis_edges.append({"s": s, "t": t})

    n_json = json.dumps(vis_nodes)
    e_json = json.dumps(vis_edges)
    n_cls  = int(class_mask.sum())
    n_feat = len(vis_df) - n_cls

    legend_items = []
    if any(row.get("node_type") == "observable_class" for _, row in class_nodes.iterrows()):
        legend_items.append(('<span style="background:#4a90d9"></span>Observable', ))
    if any(row.get("node_type") == "fault_class" for _, row in class_nodes.iterrows()):
        legend_items.append(('<span style="background:#d95555"></span>Fault', ))
    if not legend_items:
        legend_items.append(('<span style="background:#5aad6e"></span>Class', ))
    legend_html = "".join(
        f'<span class="kg-leg-item">{item[0]}</span>' for item in legend_items
    )
    if n_feat:
        legend_html += '<span class="kg-leg-item"><span style="background:#b0b0b0;width:7px;height:7px;border-radius:50%"></span>Feature</span>'

    return f"""
<p class="na" style="margin-bottom:0.4rem;">{n_cls} class nodes &middot; {n_feat} feature nodes &middot; {len(vis_edges)} edges. Hover to pause.</p>
<div class="kg-wrap">
  <canvas id="{canvas_id}"></canvas>
  <div class="kg-legend">{legend_html}</div>
</div>
<script>
(function(){{
  var cv = document.getElementById('{canvas_id}');
  var W = cv.parentElement.offsetWidth || 860, H = 420;
  cv.width = W; cv.height = H;
  var ctx = cv.getContext('2d');
  var ND = {n_json};
  var ED = {e_json};
  var paused = false;
  cv.addEventListener('mouseenter', function(){{ paused = true; }});
  cv.addEventListener('mouseleave', function(){{ paused = false; go(); }});

  var seed = 12345;
  function rnd(){{ seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 4294967296; }}

  var ns = ND.map(function(n) {{
    return {{ label:n.label, cls:n.cls, color:n.color,
             x: W/2 + (rnd()-.5)*W*.65, y: H/2 + (rnd()-.5)*H*.65,
             vx:0, vy:0 }};
  }});

  function step() {{
    var i, j, a, b, dx, dy, d2, d, f;
    for (i = 0; i < ns.length; i++) {{
      for (j = i + 1; j < ns.length; j++) {{
        a = ns[i]; b = ns[j];
        dx = b.x - a.x; dy = b.y - a.y;
        d2 = dx*dx + dy*dy + 1;
        f = (a.cls || b.cls) ? 9000/d2 : 1800/d2;
        a.vx -= f*dx; a.vy -= f*dy;
        b.vx += f*dx; b.vy += f*dy;
      }}
    }}
    for (i = 0; i < ED.length; i++) {{
      a = ns[ED[i].s]; b = ns[ED[i].t];
      if (!a || !b) continue;
      dx = b.x - a.x; dy = b.y - a.y;
      d = Math.sqrt(dx*dx + dy*dy) + .01;
      f = (d - 88) * .038;
      a.vx += f*dx/d; a.vy += f*dy/d;
      b.vx -= f*dx/d; b.vy -= f*dy/d;
    }}
    for (i = 0; i < ns.length; i++) {{
      ns[i].vx += (W/2 - ns[i].x) * .013;
      ns[i].vy += (H/2 - ns[i].y) * .013;
      ns[i].x  += ns[i].vx * .42;
      ns[i].y  += ns[i].vy * .42;
      ns[i].vx *= .76; ns[i].vy *= .76;
      ns[i].x = Math.max(32, Math.min(W-32, ns[i].x));
      ns[i].y = Math.max(22, Math.min(H-22, ns[i].y));
    }}
  }}

  function draw() {{
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = '#d5d5d5'; ctx.lineWidth = .9;
    for (var i = 0; i < ED.length; i++) {{
      var s = ns[ED[i].s], t = ns[ED[i].t];
      if (!s || !t) continue;
      ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(t.x, t.y); ctx.stroke();
    }}
    for (var i = 0; i < ns.length; i++) {{
      var n = ns[i], r = n.cls ? 14 : 5;
      ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, 2*Math.PI);
      ctx.fillStyle = n.color; ctx.fill();
      ctx.strokeStyle = 'rgba(0,0,0,.18)'; ctx.lineWidth = .6; ctx.stroke();
      if (n.cls) {{
        ctx.fillStyle = '#111'; ctx.font = 'bold 11px sans-serif';
        ctx.fillText(n.label, n.x + 17, n.y + 4);
      }}
    }}
  }}

  var iter = 0;
  function go() {{
    if (paused) return;
    for (var k = 0; k < 6; k++) step();
    draw(); iter++;
    if (iter < 160) requestAnimationFrame(go);
  }}
  go();
}})();
</script>
"""


# ── Section builders ──────────────────────────────────────────────────────────

def _quality_card(fam_dir: str) -> str:
    for fname in ("detection_report.json", "quality_report.json"):
        p = os.path.join(fam_dir, fname)
        rpt = _read_json(p)
        if rpt:
            break
    else:
        return ""

    rows: list[tuple[str, str]] = []

    if "cdh_timing" in rpt:
        t = rpt["cdh_timing"]
        rows += [
            ("Mean sample interval (s)", _fmt(t.get("mean_s"))),
            ("Interval std (s)",         _fmt(t.get("std_s"))),
            ("Max gap (s)",              _fmt(t.get("max_gap_s"))),
        ]
        stuck = rpt.get("adcs_stuck_channels", [])
        rows.append(("Stuck channels", str(len(stuck)) if stuck != [] else "0"))
        zscore = rpt.get("zscore_counts", {})
        rows.append(("Channels with z-score flags", str(len(zscore))))
        if "obdh_trend" in rpt:
            tr = rpt["obdh_trend"]
            rows.append(("OBDH thermal trend (deg/min)", _fmt(tr.get("slope_per_min"))))
            rows.append(("Trend R²", _fmt(tr.get("r_squared"))))

    elif "summary" in rpt:
        s = rpt["summary"]
        rows += [
            ("Rows",             str(s.get("n_rows", "—"))),
            ("Sensor channels",  f"{s.get('n_good_channels','—')} / {s.get('n_channels','—')}"),
            ("Stuck channels",   str(s.get("n_stuck", 0))),
            ("Z-score flagged",  str(s.get("n_zscore_flagged", 0))),
        ]
        t = rpt.get("timing", {})
        if t:
            rows.append(("Mean sample interval (s)", _fmt(t.get("mean_dt_s"))))
            rows.append(("Max gap (s)",              _fmt(t.get("max_gap_s"))))

    if not rows:
        return ""

    tr_html = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    return f"""
<h3>Data Quality (M2)</h3>
<table class="kv-table"><tbody>{tr_html}</tbody></table>
"""


def _ml_summary_card(fam_dir: str) -> str:
    ml_csv = os.path.join(fam_dir, "injected", "ml_metrics_by_tier.csv")
    df = _read_csv(ml_csv)
    if df is None or df.empty:
        return ""

    det_col = next((c for c in ["detector", "Detector"] if c in df.columns), None)
    if not det_col:
        return ""

    keep = [c for c in ["auc_roc", "auc_pr", "f1", "recall", "fpr"] if c in df.columns]
    if not keep:
        keep = df.select_dtypes("number").columns.tolist()[:4]

    summary = df.groupby(det_col)[keep].mean().reset_index().round(3)
    summary.columns = [c.replace("_", " ").upper() if c != det_col else "Detector"
                       for c in summary.columns]
    return f"""
<h3>Detector Performance — mean across tiers and variants (E2)</h3>
{_df_to_html(summary)}
"""


def _aggregate_quality_card(parent_dir: str) -> str:
    """Aggregate quality_report.json files from immediate subdirectories."""
    reports = []
    try:
        for sub in sorted(Path(parent_dir).iterdir()):
            if not sub.is_dir() or sub.name in {"semantic", "injected", "plots"}:
                continue
            rpt = _read_json(str(sub / "quality_report.json"))
            if rpt:
                reports.append(rpt)
    except OSError:
        pass

    if not reports:
        return ""

    n = len(reports)
    rows_list  = [r.get("summary", {}).get("n_rows", 0) for r in reports]
    stuck_list = [
        r.get("summary", {}).get("n_stuck", 0) or len(r.get("stuck_channels", []))
        for r in reports
    ]

    rows: list[tuple[str, str]] = [
        ("Records / periods processed", str(n)),
        ("Total rows (sum)",            f"{sum(rows_list):,}" if any(rows_list) else "—"),
        ("Mean stuck channels",         f"{sum(stuck_list)/n:.1f}"),
    ]
    tr_html = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    return f"""
<h3>Data Quality Summary (M2) — {n} records/periods</h3>
<table class="kv-table"><tbody>{tr_html}</tbody></table>
"""


def _aggregate_ml_card(parent_dir: str) -> str:
    """Aggregate ml_metrics_by_tier.csv from immediate subdirectories."""
    dfs = []
    try:
        for sub in sorted(Path(parent_dir).iterdir()):
            if not sub.is_dir() or sub.name in {"semantic", "injected", "plots"}:
                continue
            df = _read_csv(str(sub / "injected" / "ml_metrics_by_tier.csv"))
            if df is not None and not df.empty:
                dfs.append(df)
    except OSError:
        pass

    if not dfs:
        return ""

    combined = pd.concat(dfs, ignore_index=True)
    det_col  = next((c for c in ["detector", "Detector"] if c in combined.columns), None)
    if not det_col:
        return ""

    keep = [c for c in ["auc_roc", "auc_pr", "f1", "recall", "fpr"] if c in combined.columns]
    if not keep:
        return ""

    summary = combined.groupby(det_col)[keep].mean().reset_index().round(3)
    summary.columns = [c.replace("_", " ").upper() if c != det_col else "Detector"
                       for c in summary.columns]

    # Per-tier breakdown (collapsible)
    tier_col = next((c for c in ["tier", "Tier"] if c in combined.columns), None)
    tier_html = ""
    if tier_col:
        tier_summary = (
            combined.groupby([tier_col, det_col])[keep].mean()
            .reset_index().round(3)
        )
        tier_summary.columns = [
            c.replace("_", " ").upper() if c not in (tier_col, det_col) else c
            for c in tier_summary.columns
        ]
        tier_html = f"""
<details>
  <summary>Per-tier breakdown ({len(dfs)} records/periods aggregated)</summary>
  {_df_to_html(tier_summary, max_rows=60)}
</details>"""

    return f"""
<h3>Detector Performance — aggregated across {len(dfs)} records/periods (E2)</h3>
{_df_to_html(summary)}
{tier_html}
"""


def _family_section(fam_dir: str, family: str) -> str:
    parts: list[str] = []

    has_direct = (
        os.path.isfile(os.path.join(fam_dir, "quality_report.json")) or
        os.path.isfile(os.path.join(fam_dir, "detection_report.json")) or
        os.path.isfile(os.path.join(fam_dir, "injected", "ml_metrics_by_tier.csv"))
    )

    if has_direct:
        parts.append(_quality_card(fam_dir))
        parts.append(_ml_summary_card(fam_dir))
    else:
        parts.append(_aggregate_quality_card(fam_dir))
        parts.append(_aggregate_ml_card(fam_dir))

    # Plots: try direct dir, then subdirs (and their plots/ subdir)
    plots = _select_plots(fam_dir)
    if not plots:
        try:
            subdirs = sorted(
                d for d in Path(fam_dir).iterdir()
                if d.is_dir() and d.name not in {"semantic", "injected", "plots"}
            )
        except OSError:
            subdirs = []
        for sd in subdirs[:1]:
            plots = _select_plots(str(sd)) or _select_plots(str(sd / "plots"))
            if plots:
                break

    if plots:
        parts.append(f'<div class="fig-row">{"".join(_img_tag(str(p)) for p in plots)}</div>')

    body = "".join(p for p in parts if p)
    if not body:
        body = "<p class='na'>No results found for this family.</p>"

    return f"""
<section id="fam-{family.lower()}">
  <h2>{family}</h2>
  {body}
</section>
"""


def _semantic_section(results_dir: str, ontology_path: str = "") -> str:
    nodes_df, edges_df = _find_kg_csvs(results_dir)

    # Fall back to OWL-derived class graph if no CSV data
    if nodes_df is None and ontology_path:
        nodes_df, edges_df = _owl_kg_fallback(ontology_path)

    stats_html = ""
    if nodes_df is not None and not nodes_df.empty:
        type_col = "node_type" if "node_type" in nodes_df.columns else None
        n_total  = len(nodes_df)
        n_class  = int((nodes_df[type_col] == "ontologyclass").sum()) if type_col else 0
        n_feat   = n_total - n_class
        n_edges  = len(edges_df) if edges_df is not None else 0

        rows = [
            ("Total nodes",     f"{n_total:,}"),
            ("Ontology classes",f"{n_class:,}"),
            ("Feature nodes",   f"{n_feat:,}"),
            ("Edges",           f"{n_edges:,}"),
        ]
        if type_col and "node_type" in nodes_df.columns:
            for ntype, cnt in nodes_df[type_col].value_counts().head(6).items():
                rows.append((f"  {ntype}", f"{cnt:,}"))
        tr_html    = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
        stats_html = f'<table class="kv-table"><tbody>{tr_html}</tbody></table>'

    safe_edges = edges_df if (edges_df is not None and not edges_df.empty) else pd.DataFrame()
    graph_html = _kg_graph_html(nodes_df, safe_edges) if nodes_df is not None else \
                 "<p class='na'>No knowledge graph data found. Run the core pipeline (M4) to generate KG artifacts.</p>"

    return f"""
<section id="semantic-kg">
  <h2>Semantic Knowledge Graph (M4)</h2>
  {stats_html}
  {graph_html}
</section>
"""


def _ontology_section(owl_path: str) -> str:
    if not os.path.isfile(owl_path):
        return ""
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(owl_path)
        root = tree.getroot()
    except Exception:
        return ""

    OWL  = "http://www.w3.org/2002/07/owl#"
    RDF  = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    RDFS = "http://www.w3.org/2000/01/rdf-schema#"

    def _local(uri: str) -> str:
        if "#" in uri: return uri.split("#")[-1]
        return uri.rsplit(":", 1)[-1] if ":" in uri else uri

    observables: list[dict] = []
    faults: list[dict] = []

    for cls_el in root.findall(f".//{{{OWL}}}Class"):
        about  = cls_el.get(f"{{{RDF}}}about", "")
        local  = _local(about)
        if not local or local in ("SensorObservable", "SensorFault", ""):
            continue
        label_el   = cls_el.find(f"{{{RDFS}}}label")
        comment_el = cls_el.find(f"{{{RDFS}}}comment")
        sub_el     = cls_el.find(f"{{{RDFS}}}subClassOf")
        label   = label_el.text   if label_el   is not None and label_el.text   else local
        comment = comment_el.text if comment_el is not None and comment_el.text else ""
        parent  = _local(sub_el.get(f"{{{RDF}}}resource", "") if sub_el is not None else "")
        entry   = {"local": local, "label": label, "comment": comment}
        if "Fault" in parent:
            faults.append(entry)
        elif "Observable" in parent:
            observables.append(entry)

    ont_el  = root.find(f".//{{{OWL}}}Ontology")
    ont_iri = ont_el.get(f"{{{RDF}}}about", "") if ont_el is not None else ""

    def _class_table(items: list[dict]) -> str:
        if not items:
            return "<p class='na'>None.</p>"
        rows = "".join(
            f"<tr><td><code>{e['local']}</code></td><td>{e['label']}</td>"
            f"<td>{(e['comment'][:90] + '…') if len(e['comment']) > 90 else e['comment']}</td></tr>"
            for e in items
        )
        return (f"<table><thead><tr><th>Class</th><th>Label</th><th>Description</th></tr></thead>"
                f"<tbody>{rows}</tbody></table>")

    try:
        with open(owl_path, "r", encoding="utf-8") as fh:
            raw = fh.read()
        escaped   = raw.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        raw_block = (f"<details><summary>Show raw OWL/RDF source</summary>"
                     f"<pre>{escaped}</pre></details>")
    except Exception:
        raw_block = ""

    size_kb = os.path.getsize(owl_path) / 1024
    fname   = Path(owl_path).name
    iri_row = f"<tr><td>IRI</td><td><code>{ont_iri}</code></td></tr>" if ont_iri else ""

    return f"""
<section id="ontology">
  <h2>Domain Ontology (M4)</h2>
  <table class="kv-table"><tbody>
    <tr><td>File</td><td>{fname}</td></tr>
    <tr><td>Size</td><td>{size_kb:.1f} KB</td></tr>
    <tr><td>Observable classes</td><td>{len(observables)}</td></tr>
    <tr><td>Fault classes</td><td>{len(faults)}</td></tr>
    {iri_row}
  </tbody></table>
  <h3>Observable Classes</h3>
  {_class_table(observables)}
  <h3>Fault Classes</h3>
  {_class_table(faults)}
  {raw_block}
</section>
"""


def _provenance_section(prov_path: str) -> str:
    size_kb = os.path.getsize(prov_path) / 1024
    with open(prov_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    n_act   = sum(1 for l in lines if "prov:Activity" in l)
    preview = "".join(lines[:20]).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""
<section id="provenance">
  <h2>Provenance Trace (M5)</h2>
  <table class="kv-table"><tbody>
    <tr><td>File size</td><td>{size_kb:.1f} KB</td></tr>
    <tr><td>Activities recorded</td><td>~{n_act}</td></tr>
    <tr><td>Format</td><td>RDF Turtle (PROV-O + ProvONE)</td></tr>
    <tr><td>SHA-256 checksums</td><td>Enabled</td></tr>
  </tbody></table>
  <pre>{preview}...</pre>
</section>
"""


# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
/* ── Reset + base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { font-size: 16px; }
body {
  font-family: "Georgia", "Times New Roman", serif;
  background: #fafaf8;
  color: #1a1a1a;
  line-height: 1.7;
  max-width: 960px;
  margin: 0 auto;
  padding: 0 24px 60px;
}

/* ── Typography ── */
h1 { font-size: 1.7rem; font-weight: normal; letter-spacing: 0.01em; }
h2 { font-size: 1.2rem; font-weight: bold; margin: 2rem 0 0.75rem;
     padding-bottom: 0.35rem; border-bottom: 1px solid #ccc; }
h3 { font-size: 1rem; font-weight: bold; margin: 1.4rem 0 0.5rem; color: #333; }
p  { margin: 0.5rem 0; }
pre {
  font-family: "Courier New", Courier, monospace;
  font-size: 0.78rem;
  background: #f4f4f2;
  border: 1px solid #ddd;
  border-radius: 2px;
  padding: 10px 14px;
  overflow-x: auto;
  white-space: pre-wrap;
  max-height: 300px;
  overflow-y: auto;
  color: #333;
  margin: 0.75rem 0;
}

/* ── Header ── */
header {
  border-bottom: 2px solid #1a1a1a;
  padding: 28px 0 16px;
  margin-bottom: 8px;
}
header .meta {
  font-size: 0.85rem;
  color: #555;
  margin-top: 6px;
  font-family: "Helvetica Neue", Arial, sans-serif;
}

/* ── Navigation ── */
nav {
  font-family: "Helvetica Neue", Arial, sans-serif;
  font-size: 0.82rem;
  margin: 12px 0 28px;
  padding: 8px 0;
  border-top: 1px solid #ddd;
  border-bottom: 1px solid #ddd;
  display: flex;
  flex-wrap: wrap;
  gap: 6px 18px;
}
nav a { color: #1a56a0; text-decoration: none; }
nav a:hover { text-decoration: underline; }

/* ── Sections ── */
section { margin-bottom: 2.5rem; page-break-inside: avoid; }

/* ── Tables ── */
table {
  border-collapse: collapse;
  width: 100%;
  font-family: "Helvetica Neue", Arial, sans-serif;
  font-size: 0.85rem;
  margin: 0.5rem 0 1rem;
}
th {
  text-align: left;
  padding: 6px 12px;
  background: #f0eeec;
  border-top: 1px solid #bbb;
  border-bottom: 1px solid #bbb;
  font-weight: bold;
  color: #222;
}
td { padding: 5px 12px; border-bottom: 1px solid #e8e6e4; color: #333; }
tr:last-child td { border-bottom: none; }
table.kv-table td:first-child { color: #555; width: 55%; }
table.kv-table td:last-child  { font-variant-numeric: tabular-nums; text-align: right; width: 45%; }

/* ── Figures ── */
.fig-row {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 20px;
  margin: 1rem 0;
}
figure { margin: 0; text-align: center; }
figure img { width: 100%; border: 1px solid #ddd; display: block; }
figcaption {
  font-family: "Helvetica Neue", Arial, sans-serif;
  font-size: 0.78rem;
  color: #555;
  margin-top: 5px;
  font-style: italic;
}

/* ── Knowledge graph canvas ── */
.kg-wrap {
  position: relative;
  width: 100%;
  height: 420px;
  background: #f8f8f6;
  border: 1px solid #ddd;
  border-radius: 4px;
  overflow: hidden;
  margin: 0.75rem 0 1rem;
}
.kg-wrap canvas {
  display: block;
  width: 100%;
  height: 420px;
  cursor: crosshair;
}
.kg-legend {
  position: absolute;
  top: 8px;
  right: 10px;
  font: 11px "Helvetica Neue", Arial, sans-serif;
  background: rgba(255,255,255,0.92);
  padding: 5px 10px;
  border: 1px solid #ccc;
  border-radius: 3px;
  display: flex;
  gap: 10px;
  align-items: center;
}
.kg-leg-item {
  display: flex;
  align-items: center;
  gap: 5px;
  white-space: nowrap;
}
.kg-leg-item span {
  display: inline-block;
  width: 11px;
  height: 11px;
  border-radius: 50%;
  flex-shrink: 0;
}

/* ── Collapsible details ── */
details { margin: 0.75rem 0; }
details summary {
  cursor: pointer;
  font-family: "Helvetica Neue", Arial, sans-serif;
  font-size: 0.85rem;
  color: #1a56a0;
  user-select: none;
}
details[open] summary { margin-bottom: 0.5rem; }

/* ── Miscellaneous ── */
.na { color: #888; font-style: italic; font-size: 0.85rem; }
footer {
  margin-top: 3rem;
  padding-top: 1rem;
  border-top: 1px solid #ddd;
  font-family: "Helvetica Neue", Arial, sans-serif;
  font-size: 0.78rem;
  color: #888;
  text-align: center;
}
"""


# ── Main builder ──────────────────────────────────────────────────────────────

def _find_ontology(results_dir: str, domain: str) -> str:
    """Auto-discover the domain OWL file under results/ontologies/."""
    # Try two levels up from results_dir (handles results/ECG/mitdb → results/)
    search_roots = [
        os.path.dirname(os.path.abspath(results_dir)),
        os.path.dirname(os.path.dirname(os.path.abspath(results_dir))),
    ]
    for root in search_roots:
        ont_dir = os.path.join(root, "ontologies")
        for candidate in (f"{domain}.owl", f"{domain}system.owl", f"{domain}telemetry.owl"):
            p = os.path.join(ont_dir, candidate)
            if os.path.isfile(p):
                return p
        if os.path.isdir(ont_dir):
            for fname in sorted(os.listdir(ont_dir)):
                if fname.startswith(domain) and fname.endswith(".owl"):
                    return os.path.join(ont_dir, fname)
    return ""


def build_html_report(
    results_dir: str,
    output_path: str,
    domain: str = "satellite",
    ontology_path: str = "",
) -> str:
    families = sorted(
        e for e in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, e))
        and e not in {"semantic", "aggregated"}
    )

    if not ontology_path:
        ontology_path = _find_ontology(results_dir, domain)

    sections: list[str] = []
    nav_items: list[str] = []

    for fam in families:
        nav_items.append(f'<a href="#fam-{fam.lower()}">{fam}</a>')
        sections.append(_family_section(os.path.join(results_dir, fam), fam))

    # Semantic KG — always show (searches for CSVs, falls back to OWL)
    nav_items.append('<a href="#semantic-kg">Knowledge Graph</a>')
    sections.append(_semantic_section(results_dir, ontology_path))

    if os.path.isfile(ontology_path):
        nav_items.append('<a href="#ontology">Ontology</a>')
        sections.append(_ontology_section(ontology_path))

    prov_path = os.path.join(results_dir, "provenance.ttl")
    if os.path.isfile(prov_path):
        nav_items.append('<a href="#provenance">Provenance</a>')
        sections.append(_provenance_section(prov_path))

    ts = datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>SensorWF Run Report — {domain}</title>
<style>{_CSS}</style>
</head>
<body>
<header>
  <h1>SensorWF Pipeline Run Report</h1>
  <p class="meta">Domain: {domain} &nbsp;&middot;&nbsp; Generated: {ts}</p>
</header>
<nav>{"".join(nav_items)}</nav>
<main>
{"".join(sections)}
</main>
<footer>Generated by SensorWF &middot; <code>scripts/utils/html_report.py</code></footer>
</body>
</html>
"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SensorWF HTML run report")
    parser.add_argument("--results-dir", default=os.path.join("results", "satellite"),
                        help="Domain results directory (default: results/satellite)")
    parser.add_argument("--output",      default="",
                        help="Output HTML path (default: <results-dir>/run_report.html)")
    parser.add_argument("--domain",      default="satellite",
                        help="Domain name shown in the report header")
    parser.add_argument("--ontology",    default="",
                        help="Path to domain OWL file (auto-discovered if omitted)")
    parser.add_argument("--no-browser",  action="store_true",
                        help="Skip opening the report in the browser")
    args = parser.parse_args()

    out  = args.output or os.path.join(args.results_dir, "run_report.html")
    path = build_html_report(args.results_dir, out, domain=args.domain,
                             ontology_path=args.ontology)
    size_kb = os.path.getsize(path) / 1024
    print(f"[html_report] Report written ({size_kb:.0f} KB) -> {path}")
    if not args.no_browser:
        webbrowser.open(f"file://{os.path.abspath(path)}")


if __name__ == "__main__":
    main()
