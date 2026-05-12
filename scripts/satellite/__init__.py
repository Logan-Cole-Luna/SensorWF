# scripts.satellite — Satellite-specific internals (SCOTTI/KISPE SATLL)
#
# These modules are used by scripts.adapters.satellite.SatelliteAdapter (M1)
# and by the satellite-specific anomaly detection use case (E1+E2).
# M2-M5 use the domain-agnostic implementations in pipeline_core and evaluator.
#
#   parser.py      — SCOTTI archive parser (used by SatelliteAdapter.load)
#   cleaner.py     — CDH/ADCS hex-decoding and cleaning (used by SatelliteAdapter.load)
#   detector.py    — satellite-specific supplementary quality checks (beyond generic M2)
#   plotter.py     — telemetry visualisation utility (not a core module)
#   ontology.py    — satellite OWL ontology generator (runtime, channel-grounded)
#   semantic_kg.py — M4 knowledge graph with multi-detector aggregation (post-E2)
#   plot_kg.py     — knowledge graph visualisation utility
