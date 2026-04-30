# IsolationForest Interpretability + Ontology Integration

## Goal
Connect model-level anomaly evidence from IsolationForest to ontology semantics in `satellitesystem.owl` so that anomaly decisions can be interpreted as structured, queryable knowledge.

## What Was Implemented
A new script was added:

- `scripts/semantic_if_kg.py`

It builds a semantic knowledge graph by combining:

1. IsolationForest feature importances from `results/*/injected/ml_evaluation.json`.
2. Interpretable tree split and decision-path rules from a no-PCA/no-rotation IF refit mode.
3. Ontology classes and object-property structure from `satellitesystem.owl`.

Outputs:

- `results/semantic/if_ontology_nodes.csv`
- `results/semantic/if_ontology_edges.csv`
- `results/semantic/if_ontology_graph.ttl`
- `results/semantic/if_interpretability_report.md`

## Why This Is Useful
This enables:

- Feature-to-concept traceability: model signals mapped to ontology classes (`Accelerometer`, `Gyroscope`, `Magnetometer`, `ReactionWheel`, etc.).
- Rule-level interpretability: tree path rules linked to anomaly tags and subsystem concepts.
- Semantic comparison: expected anomaly subsystem vs dominant model-evidence subsystem.
- Reusable graph artifacts for SPARQL/graph analytics/visualization.

## Literature Context (Prior Work Patterns)
The implemented strategy follows known directions in XAI + semantic integration:

1. Isolation Forest explainability through split/path analysis and feature usage statistics
   - Liu et al., 2008 (Isolation Forest)
2. Model explanation via additive feature attribution and explanation tooling
   - Lundberg and Lee, 2017 (SHAP)
3. Sensor/observation semantics for interoperable knowledge graphs
   - W3C SSN/SOSA Recommendation, 2017
4. Provenance and semantic linkage for sensor data workflows
   - Compton et al., SSN/PROV alignment discussions

## How To Run
From repository root:

```bash
python -m scripts.semantic_if_kg \
  --results-dir results \
  --ontology satellitesystem.owl \
  --output-dir results/semantic
```

Optional:

```bash
python -m scripts.semantic_if_kg --max-tags-per-family 18
```

Increasing `max-tags-per-family` extracts tree-path rules for more anomaly tags.

## Notes
- Production IF in `evaluator.py` uses PCA + random rotations for robustness.
- For interpretability extraction, we intentionally fit an IF variant without PCA/rotation to keep rules in direct feature space.
- Both perspectives are retained: robust production scoring and human-readable semantic explanations.
