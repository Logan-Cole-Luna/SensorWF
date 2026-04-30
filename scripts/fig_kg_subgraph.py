"""
fig_kg_subgraph.py
Extract a readable 2-hop subgraph from the semantic knowledge graph and render it.

Reads results/semantic/if_ontology_graph.ttl (or the GraphML if TTL parse fails).
Produces:
  results/figures/fig_kg_subgraph.png
"""

from __future__ import annotations
import os
import re
import collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

GRAPHML_PATH = os.path.join("results", "semantic", "if_ontology_graph.graphml")
NODES_CSV    = os.path.join("results", "semantic", "if_ontology_nodes.csv")
EDGES_CSV    = os.path.join("results", "semantic", "if_ontology_edges.csv")
OUT_PATH     = os.path.join("results", "figures", "fig_kg_subgraph.png")

# ------------------------------------------------------------------
# Load from CSV (more reliable than TTL parse without rdflib)
# ------------------------------------------------------------------
import csv as csv_mod

def load_graph_from_csv(nodes_path: str, edges_path: str) -> nx.DiGraph:
    G = nx.DiGraph()
    with open(nodes_path, newline="", encoding="utf-8") as fh:
        for row in csv_mod.DictReader(fh):
            G.add_node(row["node_id"],
                       node_type=row.get("node_type", ""),
                       label=row.get("label", row["node_id"]))
    with open(edges_path, newline="", encoding="utf-8") as fh:
        for row in csv_mod.DictReader(fh):
            src = row.get("source") or row.get("src") or ""
            dst = row.get("target") or row.get("dst") or ""
            rel = row.get("relation") or row.get("type") or row.get("edge_type") or "rel"
            w   = float(row["weight"]) if "weight" in row and row["weight"] else 1.0
            if src and dst and src in G and dst in G:
                G.add_edge(src, dst, relation=rel, weight=w)
    return G


def short_label(node_id: str, label: str) -> str:
    name = label if label else node_id
    # Strip common prefixes
    for prefix in ("if:", "sat:", "ont:", "owl:", "xsd:"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    # CamelCase -> readable
    name = re.sub(r'(?<=[a-z])(?=[A-Z])', '\n', name)
    # Truncate
    parts = name.split("\n")
    if len(parts) > 3:
        name = "\n".join(parts[:3]) + ".."
    return name[:35]


G_full = load_graph_from_csv(NODES_CSV, EDGES_CSV)
print(f"Full graph: {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges")

# ------------------------------------------------------------------
# Extract subgraph: OntologyClass nodes and their top-weighted feature
# neighbors, for the AccelerometerTest family
# ------------------------------------------------------------------

# Node types present
type_counts = collections.Counter(
    G_full.nodes[n].get("node_type", "unknown") for n in G_full.nodes
)
print("Node types:", dict(type_counts))

# Priority: OntologyClass or Class nodes
ont_nodes = [n for n in G_full.nodes
             if G_full.nodes[n].get("node_type", "").lower() in
             ("ontologyclass", "class", "ont_class", "satellite_class", "subsystem")]

if not ont_nodes:
    # Fall back to nodes whose IDs or labels contain common ontology terms
    ont_nodes = [n for n in G_full.nodes
                 if any(kw in n.lower() for kw in
                        ("adcs", "eps", "obdh", "comms", "thermal", "power",
                         "attitude", "subsystem", "satellite"))]

print(f"Ontology anchor nodes: {len(ont_nodes)}")

# Take top-4 anchors by degree
ont_nodes = sorted(ont_nodes, key=lambda n: G_full.degree(n), reverse=True)[:5]

# Find top-weighted feature neighbors (2-hop: ontology -> feature -> anomaly)
sub_nodes: set[str] = set(ont_nodes)
for anchor in ont_nodes:
    # 1-hop neighbors, sorted by edge weight descending
    def edge_w(u, v):
        d = G_full.get_edge_data(u, v)
        return d.get("weight", 0) if d else 0

    neighbors = sorted(
        list(G_full.predecessors(anchor)) + list(G_full.successors(anchor)),
        key=lambda n: edge_w(n, anchor) + edge_w(anchor, n),
        reverse=True
    )
    # Add top-5 neighbors
    for nbr in neighbors[:5]:
        sub_nodes.add(nbr)
        # 2-hop: one more level from feature nodes
        feat_neighbors = sorted(
            list(G_full.predecessors(nbr)) + list(G_full.successors(nbr)),
            key=lambda n: edge_w(n, nbr),
            reverse=True
        )
        for fn in feat_neighbors[:2]:
            sub_nodes.add(fn)

# Build subgraph (remove self-loops which crash the edge renderer)
G = G_full.subgraph(sub_nodes).copy()
G.remove_edges_from(list(nx.selfloop_edges(G)))

# Keep only nodes with edges
isolated = [n for n in G.nodes if G.degree(n) == 0]
G.remove_nodes_from(isolated)

print(f"Subgraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Trim to at most 35 nodes for readability
if G.number_of_nodes() > 35:
    # Keep highest-degree nodes
    by_degree = sorted(G.nodes, key=lambda n: G.degree(n), reverse=True)[:35]
    G = G.subgraph(by_degree).copy()

# ------------------------------------------------------------------
# Assign colors by node type
# ------------------------------------------------------------------
TYPE_COLORS = {
    "ontologyclass":   "#E53935",
    "class":           "#E53935",
    "satellite_class": "#E53935",
    "subsystem":       "#E53935",
    "feature":         "#1E88E5",
    "feat":            "#1E88E5",
    "anomaly":         "#43A047",
    "anomalyinstance": "#43A047",
    "tag":             "#FB8C00",
    "family":          "#8E24AA",
    "anomalytag":      "#FB8C00",
}
DEFAULT_COLOR = "#90A4AE"

def node_color(n: str) -> str:
    t = G.nodes[n].get("node_type", "").lower()
    return TYPE_COLORS.get(t, DEFAULT_COLOR)

def node_size(n: str) -> int:
    t = G.nodes[n].get("node_type", "").lower()
    if t in ("ontologyclass", "class", "satellite_class", "subsystem"):
        return 900
    if t in ("feature", "feat"):
        return 500
    return 300

# ------------------------------------------------------------------
# Layout and draw
# ------------------------------------------------------------------
rng = np.random.default_rng(42)
try:
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
except Exception:
    pos = nx.spring_layout(G, seed=42, k=2.5)

# Jitter identical positions to avoid matplotlib arrow crash
seen: dict[tuple, int] = {}
for n in list(pos):
    key = (round(pos[n][0], 4), round(pos[n][1], 4))
    if key in seen:
        pos[n] = (pos[n][0] + rng.uniform(-0.05, 0.05),
                  pos[n][1] + rng.uniform(-0.05, 0.05))
    seen[key] = seen.get(key, 0) + 1

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_axis_off()

node_list   = list(G.nodes)
colors      = [node_color(n) for n in node_list]
sizes       = [node_size(n) for n in node_list]
lbl_dict    = {n: short_label(n, G.nodes[n].get("label", "")) for n in G.nodes}

nx.draw_networkx_nodes(G, pos, nodelist=node_list,
                       node_color=colors, node_size=sizes, ax=ax, alpha=0.88)
nx.draw_networkx_edges(G, pos, ax=ax, arrows=False,
                       edge_color="#9E9E9E", width=0.8, alpha=0.6)
nx.draw_networkx_labels(G, pos, labels=lbl_dict, font_size=6, ax=ax,
                        font_color="#212121")

# Legend
legend_items = [
    mpatches.Patch(color="#E53935", label="OWL Class (sat: ontology)"),
    mpatches.Patch(color="#1E88E5", label="Feature node"),
    mpatches.Patch(color="#43A047", label="Anomaly instance"),
    mpatches.Patch(color="#FB8C00", label="Anomaly tag"),
    mpatches.Patch(color="#8E24AA", label="Experiment family"),
]
ax.legend(handles=legend_items, loc="upper left", fontsize=7.5,
          frameon=True, framealpha=0.9)
ax.set_title("Ontology-Linked Knowledge Graph -- Selected Subgraph\n"
             "(Full graph: 765 nodes, 12,209 edges)",
             fontsize=10, fontweight="bold")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_PATH}")
