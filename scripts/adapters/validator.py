"""
validator.py — Adapter validation harness for SensorWF DomainAdapter subclasses.

Checks schema compliance before a pipeline run so integration problems are
caught early with clear diagnostics rather than mid-run stack traces.

Usage
-----
  from scripts.adapters.validator import validate_adapter

  adapter = MyDomainAdapter(...)
  ok, report = validate_adapter(adapter, sample_path="path/to/sample.csv")
  if not ok:
      for issue in report["errors"]:
          print("ERROR:", issue)
      raise SystemExit(1)

  # Or from the CLI:
  python -m scripts.adapters.validator --adapter satellite --path data/sat/Exp1_Analysis.txt
"""

from __future__ import annotations

import traceback
from typing import Any

import numpy as np
import pandas as pd

from scripts.adapters.base import DomainAdapter


# Required DataFrame columns with expected dtype families
_REQUIRED_COLUMNS: dict[str, str] = {
    "timestamp": "datetime",
    "elapsed_s": "numeric",
}

# Required abstract methods
_REQUIRED_METHODS = [
    "load",
    "get_quality_config",
    "get_feature_config",
    "get_ontology_path",
    "get_fault_types",
]

# Required quality config keys
_QUALITY_CONFIG_KEYS = {
    "zscore_threshold": (int, float),
    "stuck_unique_max": int,
}

# Required feature config keys
_FEATURE_CONFIG_KEYS = {
    "window": int,
}


def _check_methods(adapter: DomainAdapter) -> list[str]:
    errors: list[str] = []
    for method in _REQUIRED_METHODS:
        if not callable(getattr(adapter, method, None)):
            errors.append(f"Missing required method: {method}()")
    return errors


def _check_class_attrs(adapter: DomainAdapter) -> list[str]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(getattr(adapter, "name", None), str) or not adapter.name:
        errors.append("Class attribute 'name' must be a non-empty string")
    if not isinstance(getattr(adapter, "channels", None), list):
        errors.append("Class attribute 'channels' must be a list")
    elif not adapter.channels:
        warnings.append("'channels' list is empty — no default channel names declared")
    hz = getattr(adapter, "native_hz", None)
    if hz is None or not isinstance(hz, (int, float)) or hz <= 0:
        errors.append("Class attribute 'native_hz' must be a positive number")
    return errors, warnings


def _check_dataframe(df: pd.DataFrame, adapter_channels: list[str]) -> tuple[list[str], list[str]]:
    errors:   list[str] = []
    warnings: list[str] = []

    if not isinstance(df, pd.DataFrame):
        errors.append(f"load() must return pd.DataFrame, got {type(df).__name__}")
        return errors, warnings
    if df.empty:
        errors.append("load() returned an empty DataFrame (0 rows)")
        return errors, warnings

    for col, dtype_family in _REQUIRED_COLUMNS.items():
        if col not in df.columns:
            errors.append(f"Required column '{col}' missing from DataFrame")
            continue
        series = df[col]
        if dtype_family == "datetime":
            if not pd.api.types.is_datetime64_any_dtype(series):
                errors.append(
                    f"Column '{col}' must be datetime64, got {series.dtype}"
                )
        elif dtype_family == "numeric":
            if not pd.api.types.is_numeric_dtype(series):
                errors.append(
                    f"Column '{col}' must be numeric, got {series.dtype}"
                )

    # Check elapsed_s is non-negative and monotonically non-decreasing
    if "elapsed_s" in df.columns:
        es = pd.to_numeric(df["elapsed_s"], errors="coerce")
        if es.isna().any():
            errors.append("Column 'elapsed_s' contains non-numeric values")
        elif (es < 0).any():
            errors.append("Column 'elapsed_s' has negative values")
        elif not (es.diff().dropna() >= 0).all():
            warnings.append("Column 'elapsed_s' is not monotonically non-decreasing")

    # Check declared channels are present and numeric
    missing_channels = [c for c in adapter_channels if c not in df.columns]
    if missing_channels:
        warnings.append(f"Declared channels not in DataFrame: {missing_channels}")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    sensor_cols  = [c for c in df.columns if c not in {"timestamp", "elapsed_s"}]
    non_numeric  = [c for c in sensor_cols if c not in numeric_cols]
    if non_numeric:
        warnings.append(f"Non-numeric sensor columns: {non_numeric}")

    if len(numeric_cols) < 1:
        errors.append("DataFrame has no numeric sensor channels")

    # Check for extremely high NaN rates
    for col in numeric_cols:
        nan_rate = float(df[col].isna().mean())
        if nan_rate > 0.9:
            warnings.append(f"Channel '{col}' has {nan_rate*100:.0f}% NaN values")

    return errors, warnings


def _check_quality_config(cfg: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(cfg, dict):
        errors.append(f"get_quality_config() must return dict, got {type(cfg).__name__}")
        return errors
    for key, expected_type in _QUALITY_CONFIG_KEYS.items():
        if key not in cfg:
            errors.append(f"Quality config missing required key: '{key}'")
        elif not isinstance(cfg[key], expected_type):
            errors.append(
                f"Quality config '{key}' must be {expected_type.__name__}, "
                f"got {type(cfg[key]).__name__}"
            )
    return errors


def _check_feature_config(cfg: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(cfg, dict):
        errors.append(f"get_feature_config() must return dict, got {type(cfg).__name__}")
        return errors
    for key, expected_type in _FEATURE_CONFIG_KEYS.items():
        if key not in cfg:
            errors.append(f"Feature config missing required key: '{key}'")
        elif not isinstance(cfg[key], expected_type):
            errors.append(
                f"Feature config '{key}' must be {expected_type.__name__}, "
                f"got {type(cfg[key]).__name__}"
            )
    if "window" in cfg:
        w = cfg["window"]
        if isinstance(w, int) and w < 3:
            errors.append(f"Feature config 'window' must be >= 3, got {w}")
    return errors


def _check_fault_types(fault_types: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(fault_types, list):
        errors.append(f"get_fault_types() must return list, got {type(fault_types).__name__}")
        return errors
    if not fault_types:
        errors.append("get_fault_types() returned an empty list")
        return errors
    for i, ft in enumerate(fault_types):
        if not isinstance(ft, dict):
            errors.append(f"fault_types[{i}] must be a dict, got {type(ft).__name__}")
            continue
        if "tag" not in ft:
            errors.append(f"fault_types[{i}] missing required key 'tag'")
        if "description" not in ft and "name" not in ft:
            errors.append(f"fault_types[{i}] missing 'description' (or 'name') key")
    return errors


def validate_adapter(
    adapter: DomainAdapter,
    sample_path: str | None = None,
    load_kwargs: dict | None = None,
) -> tuple[bool, dict[str, Any]]:
    """
    Validate that *adapter* satisfies the SensorWF DomainAdapter schema.

    Parameters
    ----------
    adapter      : instantiated DomainAdapter subclass
    sample_path  : path passed to adapter.load() for DataFrame validation
                   (skipped when None)
    load_kwargs  : extra keyword arguments forwarded to adapter.load()

    Returns
    -------
    (ok, report) where ok is True iff zero errors; report contains:
      errors   : list of error strings (block pipeline)
      warnings : list of warning strings (non-fatal)
      checks   : dict of check-name → "pass" | "fail" | "skip"
    """
    errors:   list[str] = []
    warnings: list[str] = []
    checks:   dict[str, str] = {}

    # 1. Method presence
    meth_errors = _check_methods(adapter)
    errors.extend(meth_errors)
    checks["required_methods"] = "fail" if meth_errors else "pass"

    # 2. Class attributes
    attr_errors, attr_warnings = _check_class_attrs(adapter)
    errors.extend(attr_errors)
    warnings.extend(attr_warnings)
    checks["class_attributes"] = "fail" if attr_errors else "pass"

    # 3. Quality config
    try:
        qcfg = adapter.get_quality_config()
        qcfg_errors = _check_quality_config(qcfg)
        errors.extend(qcfg_errors)
        checks["quality_config"] = "fail" if qcfg_errors else "pass"
    except Exception as exc:
        errors.append(f"get_quality_config() raised: {exc}")
        checks["quality_config"] = "fail"

    # 4. Feature config
    try:
        fcfg = adapter.get_feature_config()
        fcfg_errors = _check_feature_config(fcfg)
        errors.extend(fcfg_errors)
        checks["feature_config"] = "fail" if fcfg_errors else "pass"
    except Exception as exc:
        errors.append(f"get_feature_config() raised: {exc}")
        checks["feature_config"] = "fail"

    # 5. Fault types
    try:
        ft = adapter.get_fault_types()
        ft_errors = _check_fault_types(ft)
        errors.extend(ft_errors)
        checks["fault_types"] = "fail" if ft_errors else "pass"
    except Exception as exc:
        errors.append(f"get_fault_types() raised: {exc}")
        checks["fault_types"] = "fail"

    # 6. DataFrame schema (requires sample_path)
    if sample_path is not None:
        try:
            df = adapter.load(sample_path, **(load_kwargs or {}))
            df_errors, df_warnings = _check_dataframe(df, getattr(adapter, "channels", []))
            errors.extend(df_errors)
            warnings.extend(df_warnings)
            checks["dataframe_schema"] = "fail" if df_errors else "pass"
        except Exception as exc:
            tb = traceback.format_exc()
            errors.append(f"adapter.load() raised:\n{tb}")
            checks["dataframe_schema"] = "fail"
    else:
        checks["dataframe_schema"] = "skip"

    report = {
        "adapter":  repr(adapter),
        "errors":   errors,
        "warnings": warnings,
        "checks":   checks,
        "ok":       len(errors) == 0,
    }
    return len(errors) == 0, report


def print_report(report: dict[str, Any]) -> None:
    """Print a validation report in human-readable form."""
    status = "PASS" if report["ok"] else "FAIL"
    print(f"\n{'='*60}")
    print(f"Adapter validation: {status}")
    print(f"Adapter: {report['adapter']}")
    print(f"{'='*60}")
    for check, result in report["checks"].items():
        icon = {"pass": "✓", "fail": "✗", "skip": "–"}.get(result, "?")
        print(f"  {icon} {check}: {result}")
    if report["errors"]:
        print(f"\nErrors ({len(report['errors'])}):")
        for e in report["errors"]:
            print(f"  ✗ {e}")
    if report["warnings"]:
        print(f"\nWarnings ({len(report['warnings'])}):")
        for w in report["warnings"]:
            print(f"  ! {w}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Validate a SensorWF DomainAdapter")
    parser.add_argument("--adapter", required=True,
                        choices=["ecg", "climate", "satellite"],
                        help="Built-in adapter to validate")
    parser.add_argument("--path", default=None,
                        help="Sample data file to pass to adapter.load()")
    parser.add_argument("--debug-path", default=None,
                        help="Satellite only: Debug .txt fallback file")
    args = parser.parse_args()

    load_kwargs: dict = {}
    if args.adapter == "ecg":
        from scripts.adapters.ecg import ECGAdapter
        adapter: DomainAdapter = ECGAdapter()
    elif args.adapter == "climate":
        from scripts.adapters.climate import ClimateAdapter
        adapter = ClimateAdapter()
    else:
        from scripts.adapters.satellite import SatelliteAdapter
        adapter = SatelliteAdapter()
        if args.debug_path:
            load_kwargs["debug_path"] = args.debug_path

    ok, report = validate_adapter(adapter, sample_path=args.path, load_kwargs=load_kwargs or None)
    print_report(report)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
