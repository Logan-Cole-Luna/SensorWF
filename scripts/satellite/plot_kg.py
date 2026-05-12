"""
Create interpretability figures from semantic IF-KG artifacts.

Inputs:
- results/semantic/if_ontology_nodes.csv
- results/semantic/if_ontology_edges.csv

Outputs:
- results/semantic/figures/*.png
- results/semantic/semantic_insights.md
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import networkx as nx


sns.set_theme(style="whitegrid")


def _safe_name(s: str) -> str:
    return (
        str(s)
        .replace("if:tag:", "")
        .replace("if:subsystem:", "")
        .replace("if:feature:", "")
        .replace("ont:class:", "")
    )


def _prep(edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    imp = edges[edges["predicate"] == "if:importantFeature"].copy()
    imp = imp[imp["src"].str.startswith("if:tag:")]
    imp["tag"] = imp["src"].map(_safe_name)
    imp["feature"] = imp["dst"].map(_safe_name)
    imp["family"] = imp["context"].fillna("")
    imp["weight"] = pd.to_numeric(imp["weight"], errors="coerce").fillna(0.0)

    f2s = edges[edges["predicate"] == "if:belongsToSubsystem"][["src", "dst"]].copy()
    f2s["feature"] = f2s["src"].map(_safe_name)
    f2s["subsystem"] = f2s["dst"].map(_safe_name)
    f2s = f2s[["feature", "subsystem"]].drop_duplicates()

    f2c = edges[edges["predicate"] == "if:mapsToSensorClass"][['src', 'dst']].copy()
    f2c["feature"] = f2c["src"].map(_safe_name)
    f2c["sensor_class"] = f2c["dst"].map(_safe_name)
    f2c = f2c[["feature", "sensor_class"]].drop_duplicates()

    split = edges[edges["predicate"] == "if:treeSplitFeature"].copy()
    split["tag"] = split["src"].map(_safe_name)
    split["feature"] = split["dst"].map(_safe_name)
    split["family"] = split["context"].fillna("")
    split["weight"] = pd.to_numeric(split["weight"], errors="coerce").fillna(0.0)

    local = edges[edges["predicate"] == "if:localContributionAbs"].copy()
    local = local[local["src"].str.startswith("if:instance:")]
    local["instance"] = local["src"].map(_safe_name)
    local["feature"] = local["dst"].map(_safe_name)
    local["family"] = local["context"].fillna("")
    local["weight"] = pd.to_numeric(local["weight"], errors="coerce").fillna(0.0)

    return imp, f2s, f2c, split, local


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--semantic-dir", default=os.path.join("results", "satellite", "semantic"))
    ap.add_argument("--pub-top-k", type=int, default=40)
    ap.add_argument("--pub-edge-quantile", type=float, default=0.75)
    args = ap.parse_args()

    sem_dir = args.semantic_dir
    nodes_p = os.path.join(sem_dir, "if_ontology_nodes.csv")
    edges_p = os.path.join(sem_dir, "if_ontology_edges.csv")
    out_dir = os.path.join(sem_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    nodes = pd.read_csv(nodes_p)
    edges = pd.read_csv(edges_p)
    imp, f2s, f2c, split, local = _prep(edges)

    # 1) Node type distribution
    plt.figure(figsize=(8, 4))
    nt = nodes["node_type"].value_counts().sort_values(ascending=False)
    sns.barplot(x=nt.index, y=nt.values, color="#4c78a8")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Count")
    plt.title("Semantic Graph Node Types")
    plt.tight_layout()
    p1 = os.path.join(out_dir, "01_node_type_distribution.png")
    plt.savefig(p1, dpi=160)
    plt.close()

    # 2) Predicate distribution
    plt.figure(figsize=(10, 4))
    pdist = edges["predicate"].value_counts().head(12)
    sns.barplot(x=pdist.index, y=pdist.values, color="#f58518")
    plt.xticks(rotation=40, ha="right")
    plt.ylabel("Count")
    plt.title("Top Edge Predicates")
    plt.tight_layout()
    p2 = os.path.join(out_dir, "02_predicate_distribution.png")
    plt.savefig(p2, dpi=160)
    plt.close()

    # 3) Family x subsystem evidence heatmap
    imp_sub = imp.merge(f2s, on="feature", how="left")
    fam_sub = (
        imp_sub.groupby(["family", "subsystem"], as_index=False)["weight"]
        .sum()
    )
    mat = fam_sub.pivot(index="family", columns="subsystem", values="weight").fillna(0.0)
    plt.figure(figsize=(8, 4))
    sns.heatmap(mat, cmap="YlOrRd", annot=True, fmt=".2f")
    plt.title("Family vs Subsystem (Summed IF Feature Importance)")
    plt.tight_layout()
    p3 = os.path.join(out_dir, "03_family_subsystem_heatmap.png")
    plt.savefig(p3, dpi=160)
    plt.close()

    # 4) Tag x subsystem (top 20 tags by evidence)
    tag_sub = (
        imp_sub.groupby(["tag", "subsystem"], as_index=False)["weight"]
        .sum()
    )
    top_tags = (
        tag_sub.groupby("tag")["weight"].sum().sort_values(ascending=False).head(20).index.tolist()
    )
    tag_sub = tag_sub[tag_sub["tag"].isin(top_tags)]
    tmat = tag_sub.pivot(index="tag", columns="subsystem", values="weight").fillna(0.0)
    plt.figure(figsize=(9, max(6, 0.35 * len(tmat))))
    sns.heatmap(tmat, cmap="Blues", annot=False)
    plt.title("Top Tags vs Subsystem Evidence")
    plt.tight_layout()
    p4 = os.path.join(out_dir, "04_tag_subsystem_heatmap.png")
    plt.savefig(p4, dpi=160)
    plt.close()

    # 5) Sensor-class usage from IF evidence
    imp_cls = imp.merge(f2c, on="feature", how="left")
    cls = imp_cls.groupby("sensor_class", as_index=False)["weight"].sum().sort_values("weight", ascending=False)
    plt.figure(figsize=(7, 4))
    sns.barplot(data=cls.head(10), x="sensor_class", y="weight", color="#54a24b")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Summed Importance")
    plt.title("Ontology Sensor-Class Evidence from IF")
    plt.tight_layout()
    p5 = os.path.join(out_dir, "05_sensor_class_evidence.png")
    plt.savefig(p5, dpi=160)
    plt.close()

    # 6) Top split features from interpretable tree mode
    sf = split.groupby("feature", as_index=False)["weight"].sum().sort_values("weight", ascending=False).head(20)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=sf, x="feature", y="weight", color="#e45756")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Split Count")
    plt.title("Top Tree Split Features (Interpretable IF Mode)")
    plt.tight_layout()
    p6 = os.path.join(out_dir, "06_top_tree_split_features.png")
    plt.savefig(p6, dpi=160)
    plt.close()

    # 7) Top local per-instance contribution features
    if not local.empty:
        lf = local.groupby("feature", as_index=False)["weight"].sum().sort_values("weight", ascending=False).head(20)
        plt.figure(figsize=(10, 5))
        sns.barplot(data=lf, x="feature", y="weight", color="#72b7b2")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Summed |Local Contribution|")
        plt.title("Top Local Per-Instance Contribution Features")
        plt.tight_layout()
    else:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No local contribution edges found", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
    p7 = os.path.join(out_dir, "07_top_local_contribution_features.png")
    plt.savefig(p7, dpi=160)
    plt.close()

    # 8) Traditional ontology class hierarchy graph
    onto_sub = edges[edges["predicate"] == "rdfs:subClassOf"].copy()
    if not onto_sub.empty:
        g_sub = nx.DiGraph()
        for _, r in onto_sub.iterrows():
            g_sub.add_edge(_safe_name(r["src"]), _safe_name(r["dst"]))
        plt.figure(figsize=(10, 7))
        pos = nx.spring_layout(g_sub, seed=42, k=1.0)
        nx.draw_networkx_nodes(g_sub, pos, node_size=900, node_color="#4c78a8", alpha=0.9)
        nx.draw_networkx_edges(g_sub, pos, arrows=True, arrowstyle="-|>", width=1.2, alpha=0.8)
        nx.draw_networkx_labels(g_sub, pos, font_size=8)
        plt.title("Ontology Class Hierarchy (rdfs:subClassOf)")
        plt.axis("off")
        plt.tight_layout()
    else:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No ontology subclass edges", ha="center", va="center")
        plt.axis("off")
    p8 = os.path.join(out_dir, "08_ontology_class_hierarchy_network.png")
    plt.savefig(p8, dpi=160)
    plt.close()

    # 9) Traditional ontology property domain/range network
    dom = edges[edges["predicate"] == "rdfs:domain"].copy()
    ran = edges[edges["predicate"] == "rdfs:range"].copy()
    g_pr = nx.DiGraph()
    for _, r in dom.iterrows():
        g_pr.add_edge(_safe_name(r["src"]), _safe_name(r["dst"]), t="domain")
    for _, r in ran.iterrows():
        g_pr.add_edge(_safe_name(r["src"]), _safe_name(r["dst"]), t="range")
    if len(g_pr) > 0:
        plt.figure(figsize=(10, 7))
        pos = nx.spring_layout(g_pr, seed=7, k=0.9)
        nx.draw_networkx_nodes(g_pr, pos, node_size=850, node_color="#f58518", alpha=0.9)
        nx.draw_networkx_edges(g_pr, pos, arrows=True, width=1.2, alpha=0.8)
        nx.draw_networkx_labels(g_pr, pos, font_size=8)
        plt.title("Ontology Property Domain/Range Network")
        plt.axis("off")
        plt.tight_layout()
    else:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No ontology domain/range edges", ha="center", va="center")
        plt.axis("off")
    p9 = os.path.join(out_dir, "09_ontology_property_domain_range_network.png")
    plt.savefig(p9, dpi=160)
    plt.close()

    # 10) Semantic KG subgraph (top weighted explainability edges)
    kg = nx.DiGraph()
    top_imp = edges[edges["predicate"] == "if:importantFeature"].copy()
    top_imp["weight"] = pd.to_numeric(top_imp["weight"], errors="coerce").fillna(0.0)
    top_imp = top_imp.sort_values("weight", ascending=False).head(120)
    top_loc = edges[edges["predicate"] == "if:localContributionAbs"].copy()
    top_loc["weight"] = pd.to_numeric(top_loc["weight"], errors="coerce").fillna(0.0)
    top_loc = top_loc.sort_values("weight", ascending=False).head(120)
    sub_edges = pd.concat([top_imp, top_loc], ignore_index=True)
    for _, r in sub_edges.iterrows():
        kg.add_edge(_safe_name(r["src"]), _safe_name(r["dst"]), predicate=r["predicate"], weight=float(r["weight"]))
    if len(kg) > 0:
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(kg, seed=10, k=1.1)
        nx.draw_networkx_nodes(kg, pos, node_size=220, node_color="#54a24b", alpha=0.85)
        nx.draw_networkx_edges(kg, pos, arrows=False, width=0.7, alpha=0.35)
        # label only the highest-degree nodes for readability
        deg = dict(kg.degree())
        top_nodes = set(sorted(deg, key=deg.get, reverse=True)[:18])
        nx.draw_networkx_labels(kg, pos, labels={n: n for n in top_nodes}, font_size=7)
        plt.title("Semantic IF Knowledge Graph (Top Explainability Subgraph)")
        plt.axis("off")
        plt.tight_layout()
    else:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No explainability edges for KG view", ha="center", va="center")
        plt.axis("off")
    p10 = os.path.join(out_dir, "10_semantic_kg_network_subgraph.png")
    plt.savefig(p10, dpi=160)
    plt.close()

    # 11) Publication-ready sparse explainability KG view
    pub_edges = pd.concat([top_imp, top_loc], ignore_index=True)
    pub_edges["src_n"] = pub_edges["src"].map(_safe_name)
    pub_edges["dst_n"] = pub_edges["dst"].map(_safe_name)
    # Use weighted degree to keep the most explainability-relevant nodes only.
    node_score: dict[str, float] = {}
    for _, r in pub_edges.iterrows():
        w = float(r["weight"])
        node_score[r["src_n"]] = node_score.get(r["src_n"], 0.0) + w
        node_score[r["dst_n"]] = node_score.get(r["dst_n"], 0.0) + w
    top_nodes = set(sorted(node_score, key=node_score.get, reverse=True)[: max(8, args.pub_top_k)])
    pub_edges = pub_edges[
        pub_edges["src_n"].isin(top_nodes) & pub_edges["dst_n"].isin(top_nodes)
    ].copy()
    if not pub_edges.empty:
        q = min(max(args.pub_edge_quantile, 0.0), 0.98)
        w_thr = float(pub_edges["weight"].quantile(q))
        pub_edges = pub_edges[pub_edges["weight"] >= w_thr].copy()

    kg_pub = nx.DiGraph()
    for _, r in pub_edges.iterrows():
        kg_pub.add_edge(
            r["src_n"],
            r["dst_n"],
            predicate=r["predicate"],
            weight=float(r["weight"]),
        )

    if len(kg_pub) > 0:
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(kg_pub, seed=23, k=0.9)
        weights = [kg_pub[u][v].get("weight", 0.0) for u, v in kg_pub.edges()]
        wmin = min(weights) if weights else 0.0
        wmax = max(weights) if weights else 1.0
        den = (wmax - wmin) if (wmax - wmin) > 0 else 1.0
        edge_w = [0.8 + 2.8 * ((w - wmin) / den) for w in weights]
        edge_c = ["#3b6fb6" if kg_pub[u][v].get("predicate") == "if:importantFeature" else "#2f9e44" for u, v in kg_pub.edges()]
        nsize = [180 + 14 * kg_pub.degree(n) for n in kg_pub.nodes()]
        nx.draw_networkx_nodes(kg_pub, pos, node_size=nsize, node_color="#f1a340", alpha=0.92)
        nx.draw_networkx_edges(kg_pub, pos, arrows=False, width=edge_w, edge_color=edge_c, alpha=0.55)
        nx.draw_networkx_labels(kg_pub, pos, font_size=8)
        plt.title("Publication View: Sparse Semantic IF Explainability Graph")
        plt.axis("off")
        plt.tight_layout()
    else:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No edges passed publication sparsification", ha="center", va="center")
        plt.axis("off")
    p11 = os.path.join(out_dir, "11_publication_sparse_kg.png")
    plt.savefig(p11, dpi=180)
    plt.close()

    # Text insights
    lines = []
    lines.append("# Semantic Interpretability Insights")
    lines.append("")
    lines.append(f"- Nodes: {len(nodes)}")
    lines.append(f"- Edges: {len(edges)}")
    lines.append("")

    lines.append("## Dominant Node/Edge Structures")
    lines.append("")
    for k, v in nodes["node_type"].value_counts().head(6).items():
        lines.append(f"- Node type {k}: {int(v)}")
    for k, v in edges["predicate"].value_counts().head(8).items():
        lines.append(f"- Predicate {k}: {int(v)}")

    lines.append("")
    lines.append("## Family-Level Semantic Evidence")
    lines.append("")
    for fam in mat.index:
        row = mat.loc[fam].sort_values(ascending=False)
        top = row.index[0]
        val = row.iloc[0]
        lines.append(f"- {fam}: strongest mapped subsystem is {top} (weight {val:.2f})")

    lines.append("")
    lines.append("## Interpretable Tree Behavior")
    lines.append("")
    for _, r in sf.head(8).iterrows():
        lines.append(f"- {r['feature']}: split count {r['weight']:.0f}")

    lines.append("")
    lines.append("## Local Per-Instance Explanations")
    lines.append("")
    if not local.empty:
        lf = local.groupby("feature", as_index=False)["weight"].sum().sort_values("weight", ascending=False)
        for _, r in lf.head(8).iterrows():
            lines.append(f"- {r['feature']}: summed |local contribution| {r['weight']:.4f}")
    else:
        lines.append("- No local per-instance contribution edges found.")

    lines.append("")
    lines.append("## Publication Graph Focus")
    lines.append("")
    lines.append(
        f"- Sparse KG uses top {max(8, args.pub_top_k)} weighted-degree nodes and keeps edges above the {args.pub_edge_quantile:.2f} weight quantile."
    )
    lines.append(f"- Sparse graph nodes: {kg_pub.number_of_nodes()}")
    lines.append(f"- Sparse graph edges: {kg_pub.number_of_edges()}")

    md_path = os.path.join(sem_dir, "semantic_insights.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("Saved figures:")
    for p in [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11]:
        print(" -", p)
    print("Saved summary:", md_path)


if __name__ == "__main__":
    main()
