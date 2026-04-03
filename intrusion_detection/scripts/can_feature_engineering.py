"""
CAN Bus Feature Engineering for Satellite IDS
----------------------------------------------
Extracts 14 features from a sliding window of CAN frames suitable for
a TinyDecisionTree running on STM32.

Input: DataFrame with columns [timestamp, can_id, dlc, d0..d7, label]
Output: DataFrame of per-frame feature vectors

Features (14 total):
  1.  can_id_norm       — normalized arbitration ID
  2.  dlc               — data length code (0–8)
  3.  data_mean         — mean of data bytes
  4.  data_std          — std dev of data bytes
  5.  data_entropy      — Shannon entropy of data payload
  6.  data_range        — max − min of data bytes
  7.  hamming_dist      — bit-level distance from previous frame with same ID
  8.  inter_arrival_mean — mean Δt between frames with same ID (window)
  9.  id_freq           — how often this CAN ID appears in window (msgs/s)
  10. bus_load          — total messages/s across all IDs in window
  11. unique_ids        — number of distinct IDs seen in window
  12. dlc_anomaly       — 1 if DLC differs from this ID's baseline DLC
  13. id_is_known       — 1 if this CAN ID was seen in the training baseline
  14. payload_delta     — L1 distance between this frame's data and previous
                          frame with the same ID

NOTE: inter_arrival_std was removed. Computing a per-ID running standard
deviation on bare-metal firmware requires storing either all window
timestamps (excessive RAM per CAN ID) or a three-accumulator Welford
pass that complicates the fixed-size inference path. The mean alone
(a simple running sum / count) is sufficient to flag timing disruptions.

The window is 50 frames (configurable). Features 8–11 are computed over
the window; features 1–7, 12–14 are per-frame with rolling lookback.
"""

import numpy as np
import pandas as pd
from collections import deque, defaultdict

# CAN data byte columns
DATA_COLS = [f'd{i}' for i in range(8)]
WINDOW = 50          # sliding window size in frames
MAX_CAN_ID = 0x7FF   # 11-bit standard CAN ID max


def _entropy(byte_array: np.ndarray) -> float:
    """Shannon entropy of a byte array (bits)."""
    counts = np.bincount(byte_array.astype(np.uint8), minlength=256)
    probs = counts[counts > 0] / len(byte_array)
    return float(-np.sum(probs * np.log2(probs))) if len(probs) > 1 else 0.0


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    """Bit-level Hamming distance between two byte arrays."""
    return int(np.unpackbits(
        np.bitwise_xor(a.astype(np.uint8), b.astype(np.uint8))
    ).sum())


def build_baseline(train_df: pd.DataFrame) -> dict:
    """
    Compute per-ID baseline DLC and known ID set from training data.
    Returns dict: { can_id -> expected_dlc }
    """
    baseline = {}
    for cid, grp in train_df.groupby('can_id'):
        # Majority DLC for this ID
        baseline[int(cid)] = int(grp['dlc'].mode()[0])
    return baseline


def extract_features(df: pd.DataFrame, baseline: dict, window: int = WINDOW) -> pd.DataFrame:
    """
    Slide a window over df and extract CAN IDS features.

    df must have columns: timestamp (float seconds), can_id (int),
    dlc (int), d0..d7 (int 0-255), label (0=normal, 1=attack).

    Returns a DataFrame with 14 feature columns + 'label'.
    """
    known_ids = set(baseline.keys())
    baseline_dlc = baseline

    rows = []
    # Rolling window buffer: list of row dicts
    win_buf: deque = deque(maxlen=window)
    # Per-ID: last frame data bytes, list of arrival timestamps
    last_data: dict[int, np.ndarray] = {}
    last_times: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))

    for _, row in df.iterrows():
        cid   = int(row['can_id'])
        dlc   = int(row['dlc'])
        ts    = float(row['timestamp'])
        data  = row[DATA_COLS].values.astype(np.uint8)
        label = int(row['label'])

        # ── Per-frame features ──────────────────────────────────────────
        data_f = data.astype(np.float32)
        f_can_id_norm = cid / MAX_CAN_ID
        f_dlc         = dlc
        f_data_mean   = float(data_f.mean())
        f_data_std    = float(data_f.std())
        f_data_entropy = _entropy(data)
        f_data_range  = float(data_f.max() - data_f.min())

        # Hamming distance from previous frame of same ID
        if cid in last_data:
            f_hamming = _hamming(data, last_data[cid])
        else:
            f_hamming = 0

        # Payload delta (L1 norm vs previous frame of same ID)
        if cid in last_data:
            f_payload_delta = float(np.abs(data_f - last_data[cid].astype(np.float32)).sum())
        else:
            f_payload_delta = 0.0

        # DLC anomaly
        f_dlc_anomaly = 0 if (cid not in baseline_dlc or baseline_dlc[cid] == dlc) else 1

        # Known ID
        f_id_known = 1 if cid in known_ids else 0

        # ── Window features ─────────────────────────────────────────────
        win_buf.append({'can_id': cid, 'ts': ts})
        last_times[cid].append(ts)

        # Inter-arrival time for this CAN ID
        times = list(last_times[cid])
        if len(times) >= 2:
            deltas = np.diff(times)
            f_ia_mean = float(deltas.mean())
        else:
            f_ia_mean = 0.0

        # Window-level stats
        win_list = list(win_buf)
        win_ts   = [w['ts'] for w in win_list]
        win_ids  = [w['can_id'] for w in win_list]

        time_span = (win_ts[-1] - win_ts[0]) if len(win_ts) > 1 else 1.0
        time_span = max(time_span, 1e-6)

        f_id_freq    = float(sum(1 for i in win_ids if i == cid)) / time_span
        f_bus_load   = float(len(win_list)) / time_span
        f_unique_ids = float(len(set(win_ids)))

        # ── Update state ────────────────────────────────────────────────
        last_data[cid] = data.copy()

        rows.append({
            'can_id_norm':        f_can_id_norm,
            'dlc':                float(f_dlc),
            'data_mean':          f_data_mean,
            'data_std':           f_data_std,
            'data_entropy':       f_data_entropy,
            'data_range':         f_data_range,
            'hamming_dist':       float(f_hamming),
            'inter_arrival_mean': f_ia_mean,
            'id_freq':            f_id_freq,
            'bus_load':           f_bus_load,
            'unique_ids':         f_unique_ids,
            'dlc_anomaly':        float(f_dlc_anomaly),
            'id_is_known':        float(f_id_known),
            'payload_delta':      f_payload_delta,
            'label':              label,
        })

    return pd.DataFrame(rows)


FEATURE_NAMES = [
    'can_id_norm', 'dlc', 'data_mean', 'data_std', 'data_entropy',
    'data_range', 'hamming_dist', 'inter_arrival_mean',
    'id_freq', 'bus_load', 'unique_ids', 'dlc_anomaly', 'id_is_known',
    'payload_delta',
]
