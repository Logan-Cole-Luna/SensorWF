"""
pipeline_core.py -- Domain-agnostic M1-M3 pipeline functions for SensorWF.

Works with any time-series DataFrame that has:
  - timestamp  : datetime column
  - elapsed_s  : float seconds from session start
  - N numeric channels (any names)

Used by run_ecg.py, run_climate.py, and run_sat.py (via domain runners).
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Shared spectral / statistical feature helpers  (used by M3 and evaluator)
# ---------------------------------------------------------------------------

def _estimate_sample_hz(df: pd.DataFrame) -> float:
    """Estimate sampling rate in Hz from elapsed_s column."""
    if "elapsed_s" not in df.columns:
        return 1.0
    t = pd.to_numeric(df["elapsed_s"], errors="coerce").dropna()
    dt = t.diff().dropna()
    dt_pos = dt[dt > 0]
    if dt_pos.empty:
        return 1.0
    return float(1.0 / max(float(dt_pos.median()), 1e-6))


def _rolling_spectral_features(
    x2d: np.ndarray,
    window: int,
    sample_hz: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised rolling spectral entropy and dominant frequency.

    Parameters
    ----------
    x2d      : (n_samples, n_channels) float64
    window   : FFT window length (samples)
    sample_hz: sampling rate in Hz (for frequency axis)

    Returns
    -------
    spec_entropy : (n_samples, n_channels) – Shannon entropy of power spectrum
    dom_freq     : (n_samples, n_channels) – peak frequency excluding DC (Hz)
    """
    n, nc = x2d.shape
    spec_ent = np.zeros((n, nc), dtype=float)
    dom_freq = np.zeros((n, nc), dtype=float)
    if n < window or window < 4:
        return spec_ent, dom_freq

    freqs = np.fft.rfftfreq(window, d=1.0 / max(sample_hz, 1e-6))

    for ci in range(nc):
        x = x2d[:, ci].astype(float)
        mu = x.mean()
        x_c = x - mu

        # Build sliding-window matrix (n, window) via stride tricks
        x_pad = np.pad(x_c, (window - 1, 0), mode="edge")
        stride = x_pad.strides[0]
        try:
            wins = np.lib.stride_tricks.as_strided(
                x_pad,
                shape=(n, window),
                strides=(stride, stride),
                writeable=False,
            ).copy()
        except Exception:
            continue

        fft_mag = np.abs(np.fft.rfft(wins, axis=1))          # (n, nf)

        # Dominant frequency (skip DC component at index 0)
        if fft_mag.shape[1] > 1:
            dom_freq[:, ci] = freqs[np.argmax(fft_mag[:, 1:], axis=1) + 1]

        # Spectral entropy: H = -sum(p * log2(p)) over normalised power spectrum
        psd = fft_mag ** 2
        psd_sum = psd.sum(axis=1, keepdims=True) + 1e-10
        p = np.clip(psd / psd_sum, 1e-10, None)
        spec_ent[:, ci] = -np.sum(p * np.log2(p), axis=1)

    return spec_ent, dom_freq


def _rolling_zcr(x2d: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling zero-crossing rate: fraction of adjacent sample pairs whose sign
    differs within the rolling window, computed relative to the channel mean.

    Returns (n_samples, n_channels) float64.
    """
    n, nc = x2d.shape
    result = np.zeros((n, nc), dtype=float)
    for ci in range(nc):
        x = x2d[:, ci]
        centered = x - x.mean()
        sign = np.sign(centered)
        crossings = (np.diff(sign, prepend=sign[0]) != 0).astype(float)
        result[:, ci] = (
            pd.Series(crossings).rolling(window, min_periods=1).mean().to_numpy()
        )
    return result


def compute_extended_features(
    raw_df: pd.DataFrame,
    cols: list[str],
    window: int,
    sample_hz: float = 1.0,
) -> tuple[list[np.ndarray], list[str]]:
    """
    Compute the 5 extended feature families for a set of channels.

    Returns arrays and name lists for:
      sk_{c}  – rolling skewness
      ku_{c}  – rolling excess kurtosis
      zcr_{c} – zero-crossing rate
      se_{c}  – spectral entropy
      df_{c}  – dominant frequency (Hz)

    Only called when len(df) >= window * 3.
    """
    raw = raw_df[cols].copy().astype(float).ffill().bfill().fillna(0.0)
    x2d = raw.to_numpy()
    mp3 = max(3, window // 3)
    mp4 = max(4, window // 4)

    skew = raw.rolling(window, min_periods=mp3).skew().fillna(0.0).to_numpy()
    kurt = raw.rolling(window, min_periods=mp4).kurt().fillna(0.0).to_numpy()
    zcr  = _rolling_zcr(x2d, window)
    spec_ent, dom_freq = _rolling_spectral_features(x2d, window, sample_hz)

    arrays = [skew, kurt, zcr, spec_ent, dom_freq]
    names  = (
        [f"sk_{c}"  for c in cols]
        + [f"ku_{c}"  for c in cols]
        + [f"zcr_{c}" for c in cols]
        + [f"se_{c}"  for c in cols]
        + [f"df_{c}"  for c in cols]
    )
    return arrays, names

# ---------------------------------------------------------------------------
# M2 -- Domain-agnostic quality assessment
# ---------------------------------------------------------------------------

_DEFAULT_QUALITY_CFG: dict[str, Any] = {
    "stuck_unique_max":   3,       # channels with <= this many unique values are stuck
    "zscore_threshold":   3.0,     # sigma threshold for per-channel z-score flagging
    "expected_dt_s":      None,    # expected sample interval (None = auto-estimate)
    "gap_multiplier":     5.0,     # gap > expected_dt * this is flagged
    "trend_channels":     [],      # channel names to run linear trend detection on
}


def run_quality_assessment(
    df: pd.DataFrame,
    channels: list[str],
    config: dict | None = None,
) -> dict:
    """
    Generic M2 quality assessment on any multi-channel time-series DataFrame.

    Parameters
    ----------
    df       : DataFrame with timestamp, elapsed_s, and sensor channels
    channels : list of numeric channel names to assess
    config   : optional dict overriding _DEFAULT_QUALITY_CFG keys

    Returns
    -------
    dict with keys: nan_rates, stuck_channels, timing, zscore_flags, trends
    """
    cfg = {**_DEFAULT_QUALITY_CFG, **(config or {})}
    report: dict[str, Any] = {}

    # 1. NaN rates per channel
    nan_rates: dict[str, float] = {}
    for col in channels:
        if col in df.columns:
            nan_rates[col] = round(float(df[col].isna().mean()), 4)
    report["nan_rates"] = nan_rates

    # 2. Stuck-channel detection
    stuck: list[str] = []
    for col in channels:
        if col not in df.columns:
            continue
        n_unique = df[col].nunique(dropna=True)
        if 0 < n_unique <= cfg["stuck_unique_max"]:
            stuck.append(col)
    report["stuck_channels"] = stuck

    # 3. Timing statistics
    if "elapsed_s" in df.columns:
        t = pd.to_numeric(df["elapsed_s"], errors="coerce").dropna()
        dt = t.diff().dropna()
        dt_pos = dt[dt > 0]
        if not dt_pos.empty:
            expected_dt = cfg["expected_dt_s"] or float(dt_pos.median())
            max_gap = float(dt_pos.max())
            gap_threshold = expected_dt * cfg["gap_multiplier"]
            n_gaps = int((dt_pos > gap_threshold).sum())
            report["timing"] = {
                "mean_dt_s":      round(float(dt_pos.mean()), 4),
                "std_dt_s":       round(float(dt_pos.std()),  4),
                "max_gap_s":      round(max_gap, 4),
                "expected_dt_s":  round(expected_dt, 4),
                "n_large_gaps":   n_gaps,
            }
        else:
            report["timing"] = {}
    else:
        report["timing"] = {}

    # 4. Per-channel Z-score flags
    zscore_flags: dict[str, int] = {}
    thr = float(cfg["zscore_threshold"])
    for col in channels:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce").ffill().bfill().fillna(0.0).to_numpy()
        std = x.std()
        if std < 1e-9:
            continue
        z = np.abs((x - x.mean()) / std)
        n_flagged = int((z > thr).sum())
        if n_flagged > 0:
            zscore_flags[col] = n_flagged
    report["zscore_flags"] = zscore_flags

    # 5. Linear trend detection on designated channels
    trends: dict[str, dict] = {}
    if "elapsed_s" in df.columns:
        t_all = pd.to_numeric(df["elapsed_s"], errors="coerce").ffill().bfill().values
        for col in cfg.get("trend_channels", []):
            if col not in df.columns:
                continue
            y = pd.to_numeric(df[col], errors="coerce").ffill().bfill().fillna(0.0).values
            n = min(len(t_all), len(y))
            if n < 4:
                continue
            try:
                slope, _, r, *_ = sp_stats.linregress(t_all[:n], y[:n])
                trends[col] = {
                    "slope_per_min": round(float(slope) * 60.0, 6),
                    "r_squared":     round(float(r ** 2), 6),
                    "delta":         round(float(y[n-1] - y[0]), 4),
                }
            except Exception:
                pass
    report["trends"] = trends

    # Summary counts
    n_total = len(df)
    n_good_channels = sum(1 for col in channels
                          if col in df.columns and df[col].notna().any())
    report["summary"] = {
        "n_rows":         n_total,
        "n_channels":     len(channels),
        "n_good_channels": n_good_channels,
        "n_stuck":        len(stuck),
        "n_zscore_flagged": len(zscore_flags),
    }

    return report


# ---------------------------------------------------------------------------
# M3 -- Domain-agnostic feature engineering
# ---------------------------------------------------------------------------

def build_generic_features(
    df: pd.DataFrame,
    channels: list[str],
    window: int = 15,
) -> tuple[np.ndarray | None, list[str]]:
    """
    Build a generic feature matrix from any set of numeric channels.

    Features per channel (when session >= 3× window)
    -------------------------------------------------
    1.  Raw value
    2.  First-order difference  (d_{c})
    3.  Rolling mean            (rm_{c})
    4.  Rolling std             (rs_{c})
    5.  Rolling skewness        (sk_{c})
    6.  Rolling excess kurtosis (ku_{c})
    7.  Zero-crossing rate      (zcr_{c})
    8.  Spectral entropy        (se_{c})
    9.  Dominant frequency (Hz) (df_{c})

    Timing feature
    --------------
    dt_sample : elapsed_s first difference (packet-interval / sample-gap)

    Returns
    -------
    (X, feature_names) : (n_samples, n_features) float64 array + name list
    Returns (None, []) if no usable channels found.
    """
    cols = [c for c in channels if c in df.columns and df[c].notna().any()]
    if not cols:
        return None, []

    raw = df[cols].copy().astype(float)
    raw = raw.ffill().bfill().fillna(0.0)

    arrays: list[np.ndarray] = [raw.to_numpy()]
    names:  list[str]        = list(cols)

    # First-order differences
    diff = raw.diff().fillna(0.0).to_numpy()
    arrays.append(diff)
    names += [f"d_{c}" for c in cols]

    # Timing feature
    if "elapsed_s" in df.columns:
        elapsed = pd.to_numeric(df["elapsed_s"], errors="coerce").ffill().bfill().fillna(0.0)
        dt = elapsed.diff().fillna(0.0)
        arrays.append(dt.to_numpy()[:, None])
        names.append("dt_sample")

    # Rolling statistics + spectral features (requires session >= 3× window)
    if len(df) >= window * 3:
        rm = raw.rolling(window, min_periods=1).mean().to_numpy()
        rs = raw.rolling(window, min_periods=1).std().fillna(0.0).to_numpy()
        arrays.append(rm)
        arrays.append(rs)
        names += [f"rm_{c}" for c in cols]
        names += [f"rs_{c}" for c in cols]

        hz = _estimate_sample_hz(df)
        ext_arrays, ext_names = compute_extended_features(raw, cols, window, hz)
        arrays.extend(ext_arrays)
        names.extend(ext_names)

    X = np.concatenate(arrays, axis=1).astype(np.float64)
    return X, names


# ---------------------------------------------------------------------------
# Helpers used by domain runners
# ---------------------------------------------------------------------------

def save_quality_report(report: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)


def write_owl_if_missing(path: str, content: str) -> str:
    """Write OWL/RDF content to path if the file does not yet exist.

    Called by DomainAdapter.ensure_ontology() so every domain generates
    its ontology at runtime on first run — no manually managed static files.
    Returns the (possibly newly created) file path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
    return path


def make_elapsed(df: pd.DataFrame, dt_s: float) -> pd.DataFrame:
    """Add elapsed_s column from row index when timestamps unavailable."""
    df = df.copy()
    df["elapsed_s"] = np.arange(len(df), dtype=float) * dt_s
    return df


# ---------------------------------------------------------------------------
# M4 -- Generic semantic annotation (KG builder)
# ---------------------------------------------------------------------------

_FEATURE_PREFIXES = ("d_", "rm_", "rs_", "sk_", "ku_", "zcr_", "se_", "df_")


def _base_channel(feat: str) -> str:
    """Strip standard feature prefix to recover the underlying channel name."""
    for pfx in _FEATURE_PREFIXES:
        if feat.startswith(pfx):
            return feat[len(pfx):]
    return feat


def build_semantic_kg(
    feature_names: list[str],
    class_map: dict[str, str],
    output_dir: str,
    ontology_path: str = "",
    ml_importances: dict[str, float] | None = None,
    file_prefix: str = "",
    domain_uri: str = "https://sensorwf.org/ontologies",
) -> dict[str, str]:
    """
    Build a lightweight ontology-linked knowledge graph from feature names.

    Parameters
    ----------
    feature_names   : list of feature column names from M3
    class_map       : {channel_name_or_prefix: ontology_class_uri}
                      Exact match is tried first, then prefix match.
    output_dir      : directory to write artefacts
    ontology_path   : path to domain OWL file (recorded in Turtle header)
    ml_importances  : optional {feature_name: importance} from IsolationForest
    file_prefix     : prepended to output filenames (e.g. "ecg_")
    domain_uri      : base URI for the Turtle document

    Returns
    -------
    dict of {artefact_key: file_path}
    """
    os.makedirs(output_dir, exist_ok=True)
    importances = ml_importances or {}

    nodes: list[dict] = []
    edges: list[dict] = []
    seen_classes: set[str] = set()

    for feat in feature_names:
        nodes.append({"node_id": feat, "node_type": "feature", "label": feat})
        base = _base_channel(feat)

        # Resolve class: exact match on feature, then base channel, then prefix scan
        cls = class_map.get(feat) or class_map.get(base)
        if cls is None:
            for key, val in class_map.items():
                if base.upper().startswith(key.upper()) or key.upper() in base.upper():
                    cls = val
                    break

        if cls is None:
            continue

        if cls not in seen_classes:
            local = cls.split(":")[-1] if ":" in cls else cls.rsplit("/", 1)[-1]
            nodes.append({"node_id": cls, "node_type": "ontologyclass", "label": local})
            seen_classes.add(cls)

        edges.append({
            "source":   feat,
            "target":   cls,
            "relation": "evidenceForClass",
            "weight":   importances.get(feat, 0.001),
        })

    fp = file_prefix
    nodes_path = os.path.join(output_dir, f"{fp}kg_nodes.csv")
    edges_path = os.path.join(output_dir, f"{fp}kg_edges.csv")
    ttl_path   = os.path.join(output_dir, f"{fp}kg.ttl")

    import pandas as _pd
    _pd.DataFrame(nodes).to_csv(nodes_path, index=False)
    _pd.DataFrame(edges).to_csv(edges_path, index=False)

    ttl_lines = [
        f"@base <{domain_uri}> .",
        "@prefix if:   <https://sensorwf.org/interpretability#> .",
        "@prefix prov: <http://www.w3.org/ns/prov#> .",
        "",
    ]
    if ontology_path:
        ttl_lines.append(f"# Ontology: {ontology_path}")
    for row in _pd.DataFrame(edges).itertuples(index=False):
        ttl_lines.append(
            f"<{row.source}> if:evidenceForClass <{row.target}> ;\n"
            f"    if:featureImportance {row.weight:.6f} ."
        )
    with open(ttl_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(ttl_lines) + "\n")

    return {"nodes_csv": nodes_path, "edges_csv": edges_path, "turtle": ttl_path}


# ---------------------------------------------------------------------------
# Generic visualisation -- time-series channel overview
# ---------------------------------------------------------------------------

def plot_channels(
    df: pd.DataFrame,
    channel_groups: dict[str, list[str]],
    output_dir: str,
    title_prefix: str = "",
    dpi: int = 150,
) -> list[str]:
    """
    Produce one PNG per channel group as a multi-panel time-series figure.

    Parameters
    ----------
    df             : DataFrame with elapsed_s and sensor columns
    channel_groups : {group_name: [channel_names]}  (missing channels skipped)
    output_dir     : directory for PNG output
    title_prefix   : prepended to each figure title
    dpi            : output resolution

    Returns
    -------
    List of saved file paths (one per non-empty group).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    saved: list[str] = []
    x_col = "elapsed_s" if "elapsed_s" in df.columns else None

    palette = [
        "#1565C0", "#C62828", "#2E7D32", "#6A1B9A", "#E65100",
        "#00695C", "#4527A0", "#AD1457", "#558B2F", "#0277BD",
    ]

    for idx, (group_name, channels) in enumerate(channel_groups.items(), start=1):
        cols = [c for c in channels if c in df.columns]
        if not cols:
            continue

        n = len(cols)
        fig, axes = plt.subplots(n, 1, figsize=(10, max(2.5, 1.8 * n)), sharex=True)
        if n == 1:
            axes = [axes]

        x = df[x_col].values if x_col else np.arange(len(df))
        x_label = "Elapsed time (s)"

        for ax, ch, color in zip(axes, cols, (palette * ((n // len(palette)) + 1))):
            y = pd.to_numeric(df[ch], errors="coerce").values
            ax.plot(x, y, lw=0.6, color=color)
            ax.set_ylabel(ch, fontsize=8, labelpad=2)
            ax.tick_params(labelsize=7)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

        axes[-1].set_xlabel(x_label, fontsize=9)
        title = f"{title_prefix} — {group_name}" if title_prefix else group_name
        fig.suptitle(title, fontsize=9, y=1.01)
        fig.tight_layout()

        slug = group_name.lower().replace(" ", "_").replace("/", "_")
        fname = f"{idx:02d}_{slug}.png"
        path  = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    return saved
