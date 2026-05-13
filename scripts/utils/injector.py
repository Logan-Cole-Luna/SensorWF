"""
injector.py — Literature-driven synthetic anomaly injection for SLL telemetry.

This module was rebuilt from scratch to produce anomalies that reflect patterns
reported in published telemetry/time-series anomaly literature:

1) Hundman et al., KDD 2018 (SMAP/MSL telemetry): sustained drifts, contextual
   anomalies, and thresholding pitfalls in spacecraft channels.
   DOI: 10.1145/3219819.3219845
2) Lavin & Ahmad, ICMLA 2015 (NAB): anomaly windows and early-detection
   evaluation for streaming data.
   DOI: 10.1109/ICMLA.2015.141
3) Zhang et al., MSCRED, 2018: anomalies that break inter-sensor correlation
   and temporal consistency in multivariate streams.
   arXiv:1811.08055
4) Wu & Keogh, TKDE/ICDE 2022: benchmark flaws and need for realistic, diverse,
   non-trivial anomaly morphology.
   DOI: 10.1109/TKDE.2021.3112126

Design principles applied here:
- Point + contextual + collective anomaly shapes (not only spikes).
- Tiered severity that changes both amplitude and morphology/duration.
- Cross-channel consistency breaks for multivariate ADCS faults.
- Timing anomalies represented in elapsed_s so dt-derived features can detect them.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Channel groups
# ---------------------------------------------------------------------------

_CDH_TEMP_COLS = [
    "TEMP_SOLAR", "TEMP_EPS", "TEMP_BATTERY", "TEMP_BACKPLANE",
    "TEMP_OBDH", "TEMP_ADCS", "TEMP_WHEEL", "TEMP_COMMS", "TEMP_PAYLOAD",
]
_GYRO_COLS = ["GYRO_X", "GYRO_Y", "GYRO_Z"]
_ACCEL_COLS = ["ACCEL_X", "ACCEL_Y", "ACCEL_Z"]
_MAG_COLS = ["MAG_X", "MAG_Y", "MAG_Z"]
_IMU_COLS = _ACCEL_COLS + _GYRO_COLS + _MAG_COLS

_TAG_TARGETS: dict[str, str] = {
    "T1_obdh_runaway": "cdh",
    "T2_board_thermal_shock": "cdh",
    "T3_thermal_bias_divergence": "cdh",
    "P1_bus_sag_recovery": "cdh",
    "P2_switching_noise_burst": "cdh",
    "P3_rail_latchup": "cdh",
    "A1_accel_packet_dropout": "adcs",
    "A2_gyro_clipping": "adcs",
    "A3_mag_field_inversion": "adcs",
    "A4_imu_correlation_break": "adcs",
    "A5_sun_sensor_blinding": "adcs",
    "W1_wheel_runaway": "adcs",
    "W2_wheel_stiction_stop": "adcs",
    "C1_packet_gap_jitter": "cdh",
    "C2_rssi_fade": "cdh",
    "C3_frame_error_avalanche": "cdh",
    "COMPOUND_thermal_power_fault": "cdh",
    "COMPOUND_adcs_power_fault": "cdh+adcs",
}


@dataclass(frozen=True)
class TierProfile:
    name: str
    amp: float
    dur: float
    spread: float


_TIERS = ["easy", "medium", "hard"]
_TIER_PROFILE: dict[str, TierProfile] = {
    # Easy uses short but high-signal events; this typically improves accuracy.
    "easy": TierProfile("easy", amp=1.45, dur=0.30, spread=1.25),
    "medium": TierProfile("medium", amp=1.05, dur=0.95, spread=1.00),
    # Hard uses longer, subtler events that are harder to separate.
    "hard": TierProfile("hard", amp=0.72, dur=1.55, spread=0.70),
}


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

def _ensure_label_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "anomaly" not in df.columns:
        df["anomaly"] = False
    if "anomaly_type" not in df.columns:
        df["anomaly_type"] = ""
    return df


def _mark_rows(df: pd.DataFrame, start: int, end: int, tag: str) -> pd.DataFrame:
    df = _ensure_label_cols(df)
    start = max(0, int(start))
    end = min(len(df), int(end))
    if end <= start:
        return df
    df.loc[start:end - 1, "anomaly"] = True
    df.loc[start:end - 1, "anomaly_type"] = tag
    return df


def _series(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").ffill().bfill().fillna(0.0).to_numpy(dtype=float, copy=True)


def _elapsed(df: pd.DataFrame) -> np.ndarray:
    if "elapsed_s" in df.columns:
        return pd.to_numeric(df["elapsed_s"], errors="coerce").ffill().bfill().fillna(0.0).to_numpy(dtype=float)
    return np.arange(len(df), dtype=float)


def _pick_window(n: int, rng: np.random.Generator, base_frac: float, dur_scale: float) -> tuple[int, int]:
    width = max(6, int(round(n * base_frac * dur_scale)))
    width = min(width, max(6, n - 2))
    if n <= width + 1:
        return 0, n
    max_start = n - width
    lo = max(1, n // 8)
    if max_start <= lo:
        start = max(0, max_start // 2)
    else:
        start = int(rng.integers(lo, max_start))
    return start, start + width


def _magnitude(x: np.ndarray, q: float = 0.9) -> float:
    ax = np.abs(x)
    s = float(np.nanquantile(ax, q))
    if not np.isfinite(s) or s <= 1e-9:
        s = float(np.nanstd(x))
    if not np.isfinite(s) or s <= 1e-9:
        s = 1.0
    return s


def _label(tag: str, start: int, end: int, severity: str, channels: list[str]) -> dict[str, Any]:
    return {
        "anomaly_type": tag,
        "start_idx": int(start),
        "end_idx": int(max(start, end - 1)),
        "severity": severity,
        "affected_channels": list(channels),
    }


def _sev_from_tier(tier: str) -> str:
    if tier == "hard":
        return "high"
    if tier == "easy":
        return "low"
    return "medium"


# ---------------------------------------------------------------------------
# From-scratch anomaly generators (single target)
# ---------------------------------------------------------------------------

def _inject_t1_obdh_runaway(df: pd.DataFrame, rng: np.random.Generator, tier: str) -> tuple[pd.DataFrame, dict]:
    tag = "T1_obdh_runaway"
    col = "TEMP_OBDH"
    out = df.copy()
    if col not in out.columns:
        return _ensure_label_cols(out), {}

    p = _TIER_PROFILE[tier]
    n = len(out)
    start, end = _pick_window(n, rng, base_frac=0.45, dur_scale=p.dur)
    x = _series(out, col)
    seg = np.arange(end - start, dtype=float)
    base = x[start - 1] if start > 0 else x[start]
    slope = 0.010 * p.amp * _magnitude(x, 0.8)
    curve = 0.0007 * p.amp * _magnitude(x, 0.8)
    x[start:end] = base + slope * seg + curve * (seg ** 2)
    if end < n:
        x[end:] = x[end:] + (x[end - 1] - x[end])

    out[col] = x
    out = _mark_rows(out, start, end, tag)
    return out, _label(tag, start, end, _sev_from_tier(tier), [col])


def _inject_t2_board_thermal_shock(df: pd.DataFrame, rng: np.random.Generator, tier: str) -> tuple[pd.DataFrame, dict]:
    tag = "T2_board_thermal_shock"
    out = df.copy()
    candidates = [c for c in _CDH_TEMP_COLS if c in out.columns and c != "TEMP_OBDH"]
    if not candidates:
        return _ensure_label_cols(out), {}

    col = str(rng.choice(candidates))
    p = _TIER_PROFILE[tier]
    n = len(out)
    start, end = _pick_window(n, rng, base_frac=0.22, dur_scale=p.dur)
    x = _series(out, col)
    k = end - start
    if k < 6:
        return _ensure_label_cols(out), {}

    amp = 0.45 * p.amp * _magnitude(x)
    step = int(round(k * 0.2))
    hold = int(round(k * 0.35))
    decay_len = max(1, k - step - hold)

    x[start:start + step] += np.linspace(0.0, amp, step, endpoint=False)
    x[start + step:start + step + hold] += amp
    decay = amp * np.exp(-np.arange(decay_len) / max(1.0, 0.18 * decay_len))
    x[start + step + hold:end] += decay

    out[col] = x
    out = _mark_rows(out, start, end, tag)
    return out, _label(tag, start, end, _sev_from_tier(tier), [col])


def _inject_t3_thermal_bias_divergence(df: pd.DataFrame, rng: np.random.Generator, tier: str) -> tuple[pd.DataFrame, dict]:
    tag = "T3_thermal_bias_divergence"
    out = df.copy()
    a = "TEMP_EPS"
    b = "TEMP_BATTERY"
    if a not in out.columns:
        return _ensure_label_cols(out), {}

    p = _TIER_PROFILE[tier]
    n = len(out)
    start, end = _pick_window(n, rng, base_frac=0.35, dur_scale=p.dur)
    xa = _series(out, a)
    xb = _series(out, b) if b in out.columns else None
    k = end - start
    drift = np.linspace(0.0, 0.35 * p.amp * _magnitude(xa), k)
    xa[start:end] += drift
    if xb is not None:
        xb[start:end] -= 0.65 * drift
        out[b] = xb

    out[a] = xa
    out = _mark_rows(out, start, end, tag)
    channels = [a] + ([b] if xb is not None else [])
    return out, _label(tag, start, end, _sev_from_tier(tier), channels)


def _inject_p1_bus_sag_recovery(df: pd.DataFrame, rng: np.random.Generator, tier: str) -> tuple[pd.DataFrame, dict]:
    tag = "P1_bus_sag_recovery"
    out = df.copy()
    vcol = "SYS_V"
    if vcol not in out.columns:
        return _ensure_label_cols(out), {}

    p = _TIER_PROFILE[tier]
    n = len(out)
    start, end = _pick_window(n, rng, base_frac=0.18, dur_scale=p.dur)
    x = _series(out, vcol)
    k = end - start
    pulse = np.exp(-((np.arange(k) - 0.35 * k) ** 2) / (2.0 * (0.22 * k) ** 2))
    drop = 0.28 * p.amp * _magnitude(x)
    x[start:end] -= drop * pulse
    out[vcol] = x

    channels = [vcol]
    if "BATT_CHR_V" in out.columns:
        y = _series(out, "BATT_CHR_V")
        y[start:end] -= 0.4 * drop * pulse
        out["BATT_CHR_V"] = y
        channels.append("BATT_CHR_V")
    if "SYS_I" in out.columns:
        i = _series(out, "SYS_I")
        i[start:end] += 0.5 * p.amp * _magnitude(i) * pulse
        out["SYS_I"] = i
        channels.append("SYS_I")

    out = _mark_rows(out, start, end, tag)
    return out, _label(tag, start, end, _sev_from_tier(tier), channels)


def _inject_p2_switching_noise_burst(df: pd.DataFrame, rng: np.random.Generator, tier: str) -> tuple[pd.DataFrame, dict]:
    tag = "P2_switching_noise_burst"
    out = df.copy()
    col = "OBC_PWR_V"
    if col not in out.columns:
        return _ensure_label_cols(out), {}

    p = _TIER_PROFILE[tier]
    n = len(out)
    start, end = _pick_window(n, rng, base_frac=0.10, dur_scale=p.dur)
    x = _series(out, col)
    k = end - start
    amp = 0.20 * p.amp * _magnitude(x)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    hf = np.sin(np.linspace(0.0 + phase, 18.0 * np.pi + phase, k))
    impulses = (rng.random(k) < (0.06 * p.spread)).astype(float)
    impulses *= rng.normal(0.0, 2.2 * amp, size=k)
    x[start:end] += amp * hf + impulses
    out[col] = x

    out = _mark_rows(out, start, end, tag)
    return out, _label(tag, start, end, _sev_from_tier(tier), [col])


def _inject_p3_rail_latchup(df: pd.DataFrame, rng: np.random.Generator, tier: str) -> tuple[pd.DataFrame, dict]:
    tag = "P3_rail_latchup"
    out = df.copy()
    col = "ADCS_5V_V" if "ADCS_5V_V" in out.columns else "PLAT_5V_V"
    if col not in out.columns:
        return _ensure_label_cols(out), {}

    p = _TIER_PROFILE[tier]
    n = len(out)
    start, end = _pick_window(n, rng, base_frac=0.50, dur_scale=p.dur)
    x = _series(out, col)
    base = float(np.nanmedian(x[max(0, start - 40):start + 1]))
    latch = base + (0.6 + 0.7 * p.amp)
    x[start:end] = latch + rng.normal(0.0, 0.02 * p.spread, size=end - start)
    out[col] = x

    out = _mark_rows(out, start, end, tag)
    return out, _label(tag, start, end, _sev_from_tier(tier), [col])


def _inject_a1_accel_packet_dropout(df: pd.DataFrame, rng: np.random.Generator, tier: str) -> tuple[pd.DataFrame, dict]:
    tag = "A1_accel_packet_dropout"
    out = df.copy()
    cols = [c for c in _ACCEL_COLS if c in out.columns]
    if not cols:
        return _ensure_label_cols(out), {}

    p = _TIER_PROFILE[tier]
    n = len(out)
    start, end = _pick_window(n, rng, base_frac=0.16, dur_scale=p.dur)
    k = end - start

    for c in cols:
        x = _series(out, c)
        mask = rng.random(k) < min(0.85, 0.22 * p.spread)
        seg = x[start:end].copy()
        seg[mask] = x[start - 1] if start > 0 else 0.0
        if tier == "easy":
            seg[~mask] *= 0.35
        x[start:end] = seg
        out[c] = x

    out = _mark_rows(out, start, end, tag)
    return out, _label(tag, start, end, _sev_from_tier(tier), cols)


def _inject_a2_gyro_clipping(df: pd.DataFrame, rng: np.random.Generator, tier: str) -> tuple[pd.DataFrame, dict]:
    tag = "A2_gyro_clipping"
    out = df.copy()
    cols = [c for c in _GYRO_COLS if c in out.columns]
    if not cols:
        return _ensure_label_cols(out), {}

    p = _TIER_PROFILE[tier]
    col = str(rng.choice(cols))
    n = len(out)
    start, end = _pick_window(n, rng, base_frac=0.22, dur_scale=p.dur)
    x = _series(out, col)
    q = 0.75 if tier == "easy" else (0.85 if tier == "medium" else 0.92)
    clip = float(np.nanquantile(np.abs(x), q))
    clip = max(clip, 0.1)
    seg = x[start:end]
    seg = np.clip(seg, -clip, clip)
    seg += 0.20 * p.amp * np.sign(seg + 1e-9)
    x[start:end] = seg
    out[col] = x

    out = _mark_rows(out, start, end, tag)
    return out, _label(tag, start, end, _sev_from_tier(tier), [col])


def _inject_a3_mag_field_inversion(df: pd.DataFrame, rng: np.random.Generator, tier: str) -> tuple[pd.DataFrame, dict]:
    tag = "A3_mag_field_inversion"
    out = df.copy()
    cols = [c for c in _MAG_COLS if c in out.columns]
    if not cols:
        return _ensure_label_cols(out), {}

    p = _TIER_PROFILE[tier]
    n = len(out)
    start, end = _pick_window(n, rng, base_frac=0.14, dur_scale=p.dur)

    for c in cols:
        x = _series(out, c)
        m = _magnitude(x)
        x[start:end] = -x[start:end] + (0.15 * p.amp * m)
        out[c] = x

    out = _mark_rows(out, start, end, tag)
    return out, _label(tag, start, end, _sev_from_tier(tier), cols)


def _inject_a4_imu_correlation_break(df: pd.DataFrame, rng: np.random.Generator, tier: str) -> tuple[pd.DataFrame, dict]:
    tag = "A4_imu_correlation_break"
    out = df.copy()
    cols = [c for c in _IMU_COLS if c in out.columns]
    if len(cols) < 3:
        return _ensure_label_cols(out), {}

    p = _TIER_PROFILE[tier]
    n = len(out)
    start, end = _pick_window(n, rng, base_frac=0.20, dur_scale=p.dur)
    k = end - start

    ref_col = str(rng.choice(cols))
    ref = _series(out, ref_col)
    lag = int(round(3 * p.spread))
    shifted = np.roll(ref[start:end], lag)

    picked = list(rng.choice(cols, size=max(3, int(round(len(cols) * min(0.8, 0.35 * p.spread)))), replace=False))
    for c in picked:
        x = _series(out, c)
        noise = rng.normal(0.0, 0.15 * p.amp * _magnitude(x), size=k)
        blend = 0.65 if tier == "easy" else (0.45 if tier == "medium" else 0.30)
        x[start:end] = blend * shifted + (1.0 - blend) * x[start:end] + noise
        out[c] = x

    out = _mark_rows(out, start, end, tag)
    return out, _label(tag, start, end, _sev_from_tier(tier), picked)


def _inject_a5_sun_sensor_blinding(df: pd.DataFrame, rng: np.random.Generator, tier: str) -> tuple[pd.DataFrame, dict]:
    tag = "A5_sun_sensor_blinding"
    out = df.copy()
    cols = [c for c in out.columns if c.startswith("SUN_SENSOR_")]
    if not cols:
        return _ensure_label_cols(out), {}

    p = _TIER_PROFILE[tier]
    n = len(out)
    start, end = _pick_window(n, rng, base_frac=0.12, dur_scale=p.dur)
    k = end - start
    frac = 1.0 if tier == "easy" else (0.7 if tier == "medium" else 0.4)
    m = max(1, int(round(len(cols) * frac)))
    picked = list(rng.choice(cols, size=m, replace=False))

    for c in picked:
        x = _series(out, c)
        sat = max(900.0, float(np.nanquantile(x, 0.99)) + 80.0 * p.amp)
        flicker = 6.0 * p.spread * np.sin(np.linspace(0.0, 6.0 * np.pi, k))
        x[start:end] = sat + flicker
        out[c] = x

    out = _mark_rows(out, start, end, tag)
    return out, _label(tag, start, end, _sev_from_tier(tier), picked)


def _inject_w1_wheel_runaway(df: pd.DataFrame, rng: np.random.Generator, tier: str) -> tuple[pd.DataFrame, dict]:
    tag = "W1_wheel_runaway"
    out = df.copy()
    col = "WHEEL_SPEED"
    if col not in out.columns:
        return _ensure_label_cols(out), {}

    p = _TIER_PROFILE[tier]
    n = len(out)
    start, end = _pick_window(n, rng, base_frac=0.34, dur_scale=p.dur)
    x = _series(out, col)
    k = end - start
    base = x[start - 1] if start > 0 else x[start]
    target = base + (2.8 + 2.2 * p.amp) * _magnitude(x)
    ramp = np.linspace(base, target, k)
    osc = 0.05 * p.spread * _magnitude(x) * np.sin(np.linspace(0.0, 5.5 * np.pi, k))
    x[start:end] = ramp + osc
    if end < n:
        x[end:] += (x[end - 1] - x[end])
    out[col] = x

    out = _mark_rows(out, start, end, tag)
    return out, _label(tag, start, end, _sev_from_tier(tier), [col])


def _inject_w2_wheel_stiction_stop(df: pd.DataFrame, rng: np.random.Generator, tier: str) -> tuple[pd.DataFrame, dict]:
    tag = "W2_wheel_stiction_stop"
    out = df.copy()
    col = "WHEEL_SPEED"
    if col not in out.columns:
        return _ensure_label_cols(out), {}

    p = _TIER_PROFILE[tier]
    n = len(out)
    start, end = _pick_window(n, rng, base_frac=0.25, dur_scale=p.dur)
    x = _series(out, col)
    k = end - start
    cut = max(2, int(round(0.18 * k)))
    hold = max(2, int(round(0.52 * k)))

    x[start:start + cut] = np.linspace(x[start], 0.05 * x[start], cut)
    x[start + cut:start + cut + hold] = 0.0 + rng.normal(0.0, 0.015 * _magnitude(x), size=hold)
    rem = k - cut - hold
    if rem > 0:
        rec_target = (0.12 if tier == "easy" else (0.35 if tier == "medium" else 0.55)) * x[start]
        x[start + cut + hold:end] = np.linspace(0.0, rec_target, rem)

    out[col] = x
    out = _mark_rows(out, start, end, tag)
    return out, _label(tag, start, end, _sev_from_tier(tier), [col])


def _inject_c1_packet_gap_jitter(df: pd.DataFrame, rng: np.random.Generator, tier: str) -> tuple[pd.DataFrame, dict]:
    tag = "C1_packet_gap_jitter"
    out = df.copy()
    if "elapsed_s" not in out.columns or len(out) < 20:
        return _ensure_label_cols(out), {}

    p = _TIER_PROFILE[tier]
    t = _elapsed(out)
    dt = np.diff(t)
    nominal = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 1.0
    nominal = max(0.1, nominal)
    n = len(out)

    gap_rows = max(2, int(round((4.0 + 6.0 * p.amp) / nominal)))
    split = int(rng.integers(max(5, n // 5), max(6, n - gap_rows - 5)))
    gap_end = min(n, split + gap_rows)

    before = out.iloc[:split].copy()
    after = out.iloc[gap_end:].copy()
    removed = float(t[gap_end - 1] - t[split]) if gap_end - split > 0 else (gap_rows * nominal)

    if not after.empty and not before.empty:
        after["elapsed_s"] = _elapsed(after) - removed + rng.normal(0.0, 0.15 * nominal * p.spread, size=len(after))

    out2 = pd.concat([before, after], ignore_index=True)
    out2 = _ensure_label_cols(out2)
    mark_start = max(0, split - 2)
    mark_end = min(len(out2), split + max(4, int(round(8 * p.spread))))
    out2 = _mark_rows(out2, mark_start, mark_end, tag)

    return out2, _label(tag, mark_start, mark_end, _sev_from_tier(tier), ["elapsed_s"])


def _inject_c2_rssi_fade(df: pd.DataFrame, rng: np.random.Generator, tier: str) -> tuple[pd.DataFrame, dict]:
    tag = "C2_rssi_fade"
    out = df.copy()
    col = "RSSI"
    if col not in out.columns:
        return _ensure_label_cols(out), {}

    p = _TIER_PROFILE[tier]
    n = len(out)
    start, end = _pick_window(n, rng, base_frac=0.28, dur_scale=p.dur)
    x = _series(out, col)
    k = end - start

    fade = np.linspace(0.0, 4.0 * p.amp * _magnitude(x, 0.6), k)
    rw = np.cumsum(rng.normal(0.0, 0.05 * p.spread * _magnitude(x, 0.6), size=k))
    x[start:end] = x[start:end] - fade + rw

    out[col] = x
    out = _mark_rows(out, start, end, tag)
    return out, _label(tag, start, end, _sev_from_tier(tier), [col])


def _inject_c3_frame_error_avalanche(df: pd.DataFrame, rng: np.random.Generator, tier: str) -> tuple[pd.DataFrame, dict]:
    tag = "C3_frame_error_avalanche"
    out = df.copy()
    col = "FRAME_ERR"
    if col not in out.columns:
        return _ensure_label_cols(out), {}

    p = _TIER_PROFILE[tier]
    n = len(out)
    start, end = _pick_window(n, rng, base_frac=0.16, dur_scale=p.dur)
    x = _series(out, col)
    k = end - start

    lam = np.linspace(0.8, 3.0 * p.amp, k)
    inc = rng.poisson(np.maximum(0.2, lam)).astype(float)
    x[start:end] += np.cumsum(inc)
    out[col] = x

    out = _mark_rows(out, start, end, tag)
    return out, _label(tag, start, end, _sev_from_tier(tier), [col])


_SINGLE_GENERATORS: dict[str, Any] = {
    "T1_obdh_runaway": _inject_t1_obdh_runaway,
    "T2_board_thermal_shock": _inject_t2_board_thermal_shock,
    "T3_thermal_bias_divergence": _inject_t3_thermal_bias_divergence,
    "P1_bus_sag_recovery": _inject_p1_bus_sag_recovery,
    "P2_switching_noise_burst": _inject_p2_switching_noise_burst,
    "P3_rail_latchup": _inject_p3_rail_latchup,
    "A1_accel_packet_dropout": _inject_a1_accel_packet_dropout,
    "A2_gyro_clipping": _inject_a2_gyro_clipping,
    "A3_mag_field_inversion": _inject_a3_mag_field_inversion,
    "A4_imu_correlation_break": _inject_a4_imu_correlation_break,
    "A5_sun_sensor_blinding": _inject_a5_sun_sensor_blinding,
    "W1_wheel_runaway": _inject_w1_wheel_runaway,
    "W2_wheel_stiction_stop": _inject_w2_wheel_stiction_stop,
    "C1_packet_gap_jitter": _inject_c1_packet_gap_jitter,
    "C2_rssi_fade": _inject_c2_rssi_fade,
    "C3_frame_error_avalanche": _inject_c3_frame_error_avalanche,
}


# ---------------------------------------------------------------------------
# Compound generators
# ---------------------------------------------------------------------------

def _compound_thermal_power_fault(cdh: pd.DataFrame, rng: np.random.Generator, tier: str) -> tuple[pd.DataFrame, dict]:
    a_df, a_lbl = _inject_t1_obdh_runaway(cdh, rng, tier)
    b_df, b_lbl = _inject_p1_bus_sag_recovery(a_df, rng, tier)
    s = min(a_lbl.get("start_idx", 0), b_lbl.get("start_idx", 0))
    e = max(a_lbl.get("end_idx", 0), b_lbl.get("end_idx", 0)) + 1
    channels = sorted(set(a_lbl.get("affected_channels", [])) | set(b_lbl.get("affected_channels", [])))
    lbl = _label("COMPOUND_thermal_power_fault", s, e, "high" if tier != "easy" else "medium", channels)
    return b_df, lbl


def _compound_adcs_power_fault(cdh: pd.DataFrame, adcs: pd.DataFrame, rng: np.random.Generator, tier: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    adcs_out, a_lbl = _inject_a4_imu_correlation_break(adcs, rng, tier)
    cdh_out, p_lbl = _inject_p3_rail_latchup(cdh, rng, tier)
    s = min(a_lbl.get("start_idx", 0), p_lbl.get("start_idx", 0))
    e = max(a_lbl.get("end_idx", 0), p_lbl.get("end_idx", 0)) + 1
    channels = sorted(set(a_lbl.get("affected_channels", [])) | set(p_lbl.get("affected_channels", [])))
    lbl = _label("COMPOUND_adcs_power_fault", s, e, "high" if tier != "easy" else "medium", channels)
    return cdh_out, adcs_out, lbl


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def run_injections(
    cdh: pd.DataFrame,
    adcs: pd.DataFrame,
    output_dir: str,
    seed: int = 42,
    n_variants: int = 2,
    tiers: list[str] | None = None,
) -> None:
    """
    Generate labelled injected variants and run ML evaluation.

    Export contract preserved:
    - injected/cdh_injected_<type>_<tier>_v<n>.csv
    - injected/adcs_injected_<type>_<tier>_v<n>.csv
    - injected/labels_<type>_<tier>_v<n>.csv
    - injected/injection_summary.csv
    - injected/ml_evaluation.json
    """
    injected_dir = os.path.join(output_dir, "injected")
    os.makedirs(injected_dir, exist_ok=True)

    if tiers is None:
        tiers = list(_TIERS)
    else:
        tiers = [t for t in tiers if t in _TIERS]
        if not tiers:
            tiers = list(_TIERS)

    base_rng = np.random.default_rng(seed)
    files_generated: list[tuple[str, str, int, str]] = []

    for tag, gen in _SINGLE_GENERATORS.items():
        target = _TAG_TARGETS[tag]
        source = cdh if target == "cdh" else adcs
        for tier in tiers:
            for v in range(1, n_variants + 1):
                rng = np.random.default_rng(int(base_rng.integers(0, 2**31)))
                try:
                    out, lbl = gen(source, rng, tier)
                except Exception as exc:
                    print(f"  [WARN] {tag} [{tier}] v{v} failed: {exc}")
                    continue
                if not lbl:
                    continue

                lbl["tier"] = tier
                out_path = os.path.join(injected_dir, f"{target}_injected_{tag}_{tier}_v{v}.csv")
                lbl_path = os.path.join(injected_dir, f"labels_{tag}_{tier}_v{v}.csv")
                out.to_csv(out_path, index=False)
                pd.DataFrame([lbl]).to_csv(lbl_path, index=False)
                files_generated.append((tag, target, v, tier))

    for tier in tiers:
        for v in range(1, n_variants + 1):
            rng = np.random.default_rng(int(base_rng.integers(0, 2**31)))
            tag = "COMPOUND_thermal_power_fault"
            try:
                out, lbl = _compound_thermal_power_fault(cdh, rng, tier)
            except Exception as exc:
                print(f"  [WARN] {tag} [{tier}] v{v} failed: {exc}")
                continue
            lbl["tier"] = tier
            out.to_csv(os.path.join(injected_dir, f"cdh_injected_{tag}_{tier}_v{v}.csv"), index=False)
            pd.DataFrame([lbl]).to_csv(os.path.join(injected_dir, f"labels_{tag}_{tier}_v{v}.csv"), index=False)
            files_generated.append((tag, "cdh", v, tier))

    for tier in tiers:
        for v in range(1, n_variants + 1):
            rng = np.random.default_rng(int(base_rng.integers(0, 2**31)))
            tag = "COMPOUND_adcs_power_fault"
            try:
                out_cdh, out_adcs, lbl = _compound_adcs_power_fault(cdh, adcs, rng, tier)
            except Exception as exc:
                print(f"  [WARN] {tag} [{tier}] v{v} failed: {exc}")
                continue
            lbl["tier"] = tier
            out_cdh.to_csv(os.path.join(injected_dir, f"cdh_injected_{tag}_{tier}_v{v}.csv"), index=False)
            out_adcs.to_csv(os.path.join(injected_dir, f"adcs_injected_{tag}_{tier}_v{v}.csv"), index=False)
            pd.DataFrame([lbl]).to_csv(os.path.join(injected_dir, f"labels_{tag}_{tier}_v{v}.csv"), index=False)
            files_generated.append((tag, "cdh+adcs", v, tier))

    counts: dict[str, dict[str, Any]] = defaultdict(lambda: {"target": "", "tier": "", "n": 0})
    for tag, target, _v, tier in files_generated:
        key = f"{tag}::{tier}"
        counts[key]["target"] = target
        counts[key]["tier"] = tier
        counts[key]["n"] += 1

    summary_rows = [
        {
            "type": key.split("::", 1)[0],
            "tier": data["tier"],
            "target": data["target"],
            "variants": data["n"],
        }
        for key, data in sorted(counts.items())
    ]
    pd.DataFrame(summary_rows).to_csv(os.path.join(injected_dir, "injection_summary.csv"), index=False)

    total_data = sum(2 if target == "cdh+adcs" else 1 for _, target, _, _ in files_generated)
    n_types = len({row["type"] for row in summary_rows})
    tiers = sorted({row["tier"] for row in summary_rows})
    print(f"  [M4] Injection: {n_types} fault types × {len(tiers)} tiers × 2 variants "
          f"= {len(files_generated)} labelled files → {injected_dir}/")

    try:
        from scripts.evaluator import (
            print_ml_summary,
            run_ml_evaluation,
            save_ml_results,
            save_ml_tier_metrics,
        )

        print("        Running ML evaluation (ZScore / RobustRollingZScore / IsoForest / Autoencoder) ...")
        ml_results = run_ml_evaluation(injected_dir, files_generated, cdh, adcs)
        if ml_results:
            ml_path  = save_ml_results(ml_results, injected_dir)
            tier_path = save_ml_tier_metrics(ml_results, injected_dir)
            # Compact per-detector AUC-ROC / AUC-PR summary (averaged across all variants)
            by_det: dict[str, list] = {}
            by_det_pr: dict[str, list] = {}
            for r in ml_results:
                if "error" in r:
                    continue
                d = r.get("detector", "?")
                by_det.setdefault(d, []).append(r.get("auc_roc", float("nan")))
                by_det_pr.setdefault(d, []).append(r.get("auc_pr", float("nan")))
            import math
            parts = " | ".join(
                f"{d}: {sum(v for v in vals if not math.isnan(v)) / max(1, sum(1 for v in vals if not math.isnan(v))):.3f}"
                for d, vals in by_det.items()
            )
            parts_pr = " | ".join(
                f"{d}: {sum(v for v in vals if not math.isnan(v)) / max(1, sum(1 for v in vals if not math.isnan(v))):.3f}"
                for d, vals in by_det_pr.items()
            )
            print(f"  [M5] AUC-ROC(total): {parts}")
            print(f"  [M5] AUC-PR (total): {parts_pr}")
    except Exception as exc:
        print(f"        [WARN] ML evaluation failed: {exc}")
