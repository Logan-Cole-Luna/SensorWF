"""
html_report.py — Lightweight academic-style HTML run report for SensorWF.

Reads existing result artefacts from the results directory and assembles a
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
    """Convert a filename stem like '07_summary_dashboard' to 'Summary Dashboard'."""
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

# Priority order: prefer summary dashboard, then two domain-specific plots
_PREFERRED_PLOTS = [
    "07_summary_dashboard",
    "01_board_temperatures",
    "03_power_rails",
    "08_temperature_distributions",
    "16_imu_sensors",
    "17_imu_magnitudes",
]

def _select_plots(fam_dir: str, max_plots: int = 3) -> list[Path]:
    all_pngs = {p.stem: p for p in Path(fam_dir).glob("*.png")}
    chosen: list[Path] = []
    for pref in _PREFERRED_PLOTS:
        if pref in all_pngs:
            chosen.append(all_pngs[pref])
        if len(chosen) >= max_plots:
            break
    if not chosen:
        chosen = sorted(all_pngs.values())[:max_plots]
    return chosen


# ── Section builders ──────────────────────────────────────────────────────────

def _quality_card(fam_dir: str) -> str:
    """Read detection_report.json (satellite) or quality_report.json (generic)."""
    for fname in ("detection_report.json", "quality_report.json"):
        p = os.path.join(fam_dir, fname)
        rpt = _read_json(p)
        if rpt:
            break
    else:
        return ""

    rows: list[tuple[str, str]] = []

    # Satellite detection_report format
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

    # Generic quality_report format
    elif "summary" in rpt:
        s = rpt["summary"]
        rows += [
            ("Rows",               str(s.get("n_rows", "—"))),
            ("Sensor channels",    f"{s.get('n_good_channels','—')} / {s.get('n_channels','—')}"),
            ("Stuck channels",     str(s.get("n_stuck", 0))),
            ("Z-score flagged",    str(s.get("n_zscore_flagged", 0))),
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
<table class="kv-table">
  <tbody>{tr_html}</tbody>
</table>
"""


def _ml_summary_card(fam_dir: str) -> str:
    ml_csv = os.path.join(fam_dir, "injected", "ml_metrics_by_tier.csv")
    df = _read_csv(ml_csv)
    if df is None or df.empty:
        return ""

    num_cols = df.select_dtypes("number").columns.tolist()
    det_col  = next((c for c in ["detector", "Detector"] if c in df.columns), None)
    if not det_col or not num_cols:
        return ""

    keep = [c for c in ["auc_roc", "auc_pr", "f1", "recall", "fpr"] if c in df.columns]
    if not keep:
        keep = num_cols[:4]

    summary = df.groupby(det_col)[keep].mean().reset_index().round(3)
    summary.columns = [c.replace("_", " ").upper() if c != det_col else "Detector"
                       for c in summary.columns]

    return f"""
<h3>Detector Performance — mean across tiers and variants (E2)</h3>
{_df_to_html(summary)}
"""


def _family_section(fam_dir: str, family: str) -> str:
    parts: list[str] = []
    parts.append(_quality_card(fam_dir))
    parts.append(_ml_summary_card(fam_dir))

    plots = _select_plots(fam_dir)
    if plots:
        fig_html = "".join(_img_tag(str(p)) for p in plots)
        parts.append(f'<div class="fig-row">{fig_html}</div>')

    body = "".join(p for p in parts if p)
    if not body:
        body = "<p class='na'>No results found for this family.</p>"

    return f"""
<section id="fam-{family.lower()}">
  <h2>{family}</h2>
  {body}
</section>
"""


def _semantic_section(semantic_dir: str) -> str:
    nodes_csv = os.path.join(semantic_dir, "if_ontology_nodes.csv")
    edges_csv = os.path.join(semantic_dir, "if_ontology_edges.csv")
    df_nodes  = _read_csv(nodes_csv)
    df_edges  = _read_csv(edges_csv)

    rows: list[tuple[str, str]] = []
    if df_nodes is not None:
        rows.append(("Nodes", f"{len(df_nodes):,}"))
        if "node_type" in df_nodes.columns:
            top_types = df_nodes["node_type"].value_counts().head(5)
            for t, cnt in top_types.items():
                rows.append((f"  {t}", str(cnt)))
    if df_edges is not None:
        rows.append(("Edges", f"{len(df_edges):,}"))
        if "predicate" in df_edges.columns:
            top_preds = df_edges["predicate"].value_counts().head(5)
            for p, cnt in top_preds.items():
                rows.append((f"  {p}", str(cnt)))

    table_html = ""
    if rows:
        tr_html = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
        table_html = f'<table class="kv-table"><tbody>{tr_html}</tbody></table>'

    report_md = os.path.join(semantic_dir, "if_interpretability_report.md")
    preview_html = ""
    if os.path.isfile(report_md):
        with open(report_md, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        preview = "".join(lines[:40]).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        preview_html = f'<h3>Interpretability Report (excerpt)</h3><pre>{preview}</pre>'

    return f"""
<section id="semantic-kg">
  <h2>Semantic Knowledge Graph (M4)</h2>
  {table_html}
  {preview_html}
</section>
"""


def _provenance_section(prov_path: str) -> str:
    size_kb = os.path.getsize(prov_path) / 1024
    with open(prov_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    n_act = sum(1 for l in lines if "prov:Activity" in l)
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
section {
  margin-bottom: 2.5rem;
  page-break-inside: avoid;
}

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
figure {
  margin: 0;
  text-align: center;
}
figure img {
  width: 100%;
  border: 1px solid #ddd;
  display: block;
}
figcaption {
  font-family: "Helvetica Neue", Arial, sans-serif;
  font-size: 0.78rem;
  color: #555;
  margin-top: 5px;
  font-style: italic;
}

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

def build_html_report(
    results_dir: str,
    output_path: str,
    domain: str = "satellite",
) -> str:
    families = sorted(
        e for e in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, e))
        and e not in {"semantic", "aggregated", "satellite_telemetry"}
    )

    sections: list[str] = []
    nav_items: list[str] = []

    for fam in families:
        nav_items.append(f'<a href="#fam-{fam.lower()}">{fam}</a>')
        sections.append(_family_section(os.path.join(results_dir, fam), fam))

    semantic_dir = os.path.join(results_dir, "semantic")
    if os.path.isdir(semantic_dir):
        nav_items.append('<a href="#semantic-kg">Knowledge Graph</a>')
        sections.append(_semantic_section(semantic_dir))

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
<title>SensorWF Run Report</title>
<style>{_CSS}</style>
</head>
<body>
<header>
  <h1>SensorWF Pipeline Run Report</h1>
  <p class="meta">Domain: {domain} &nbsp;&middot;&nbsp; Generated: {ts}</p>
</header>
<nav>
  {"".join(nav_items)}
</nav>
<main>
{"".join(sections)}
</main>
<footer>
  Generated by SensorWF &middot; <code>scripts/utils/html_report.py</code>
</footer>
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
    parser.add_argument("--no-browser",  action="store_true",
                        help="Skip opening the report in the browser")
    args = parser.parse_args()

    out  = args.output or os.path.join(args.results_dir, "run_report.html")
    path = build_html_report(args.results_dir, out, domain=args.domain)
    size_kb = os.path.getsize(path) / 1024
    print(f"[html_report] Report written ({size_kb:.0f} KB) -> {path}")
    if not args.no_browser:
        webbrowser.open(f"file://{os.path.abspath(path)}")


if __name__ == "__main__":
    main()
