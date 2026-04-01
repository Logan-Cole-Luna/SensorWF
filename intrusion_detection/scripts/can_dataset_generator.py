"""
Satellite CAN Bus Dataset Generator
-------------------------------------
Generates realistic synthetic CAN traffic modeled on the actual satellite
subsystems observed in data/:

  CDH    (0x005)  — housekeeping telemetry        ~1 Hz
  ADCS   (0x013)  — sensor data (gyro/accel/mag)  ~10 Hz
  WHEEL  (0x020)  — reaction wheel speed           ~5 Hz
  MAG    (0x030)  — magnetorquer commands           ~2 Hz
  COMMS  (0x040)  — comms status                   ~1 Hz
  PAYLOAD(0x050)  — payload diagnostics             ~0.5 Hz

Attack types injected:
  DoS      — flood bus with a single ID at 10× normal rate
  Fuzzy    — inject frames with random IDs and random data
  Spoofing — replace legitimate frames with crafted data
             (e.g. fake WHEEL_SPEED = 0 to mask a fault)
  Replay   — re-send a captured legitimate frame repeatedly

Output CSV columns:
  timestamp, can_id, dlc, d0..d7, label, attack_type
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Subsystem definitions ────────────────────────────────────────────────────

SUBSYSTEMS = {
    # name: (can_id, dlc, hz, data_generator_fn)
    'CDH':     {'id': 0x005, 'dlc': 8, 'hz': 1.0},
    'ADCS':    {'id': 0x013, 'dlc': 8, 'hz': 10.0},
    'WHEEL':   {'id': 0x020, 'dlc': 4, 'hz': 5.0},
    'MAG':     {'id': 0x030, 'dlc': 6, 'hz': 2.0},
    'COMMS':   {'id': 0x040, 'dlc': 4, 'hz': 1.0},
    'PAYLOAD': {'id': 0x050, 'dlc': 3, 'hz': 0.5},
}

# Known valid IDs (used by feature extractor)
KNOWN_IDS = {s['id'] for s in SUBSYSTEMS.values()}

RNG = np.random.default_rng(42)


def _cdh_frame(t: float) -> np.ndarray:
    """CDH housekeeping: sys_voltage, sys_current, temp_obdh, batt_v, status"""
    sys_v   = int(np.clip(RNG.normal(195, 2), 185, 210))   # 7.3–8.2V → x25
    sys_i   = int(np.clip(RNG.normal(180, 5), 160, 200))   # ~180 raw units
    temp    = int(np.clip(RNG.normal(80,  3),  60, 100))   # ~23°C scaled
    batt_v  = int(np.clip(RNG.normal(185, 1), 180, 195))
    status  = 0x01   # platform power on
    seq     = int(t * 10) & 0xFF
    return np.array([sys_v, sys_i, temp, batt_v, status, seq, 0x00, 0x00], dtype=np.uint8)


def _adcs_frame(t: float) -> np.ndarray:
    """ADCS sensor: gyro_x, gyro_y, gyro_z (signed 16-bit packed as 2 bytes each) + mode"""
    gx = int(np.clip(RNG.normal(0, 10), -127, 127)) & 0xFF
    gy = int(np.clip(RNG.normal(0, 10), -127, 127)) & 0xFF
    gz = int(np.clip(RNG.normal(0,  5), -127, 127)) & 0xFF
    ax = int(np.clip(RNG.normal(0, 20), -127, 127)) & 0xFF
    ay = int(np.clip(RNG.normal(0, 20), -127, 127)) & 0xFF
    az = int(np.clip(RNG.normal(127, 5), 100, 150))          # ~1g in Z
    mode = 0x02   # normal attitude control
    seq  = int(t * 100) & 0xFF
    return np.array([gx, gy, gz, ax, ay, az, mode, seq], dtype=np.uint8)


def _wheel_frame(t: float) -> np.ndarray:
    """Reaction wheel speed (RPM encoded as uint16 BE)"""
    rpm = int(np.clip(RNG.normal(1500, 50), 1200, 1800))
    hi  = (rpm >> 8) & 0xFF
    lo  = rpm & 0xFF
    status = 0x01
    seq    = int(t * 50) & 0xFF
    return np.array([hi, lo, status, seq], dtype=np.uint8)


def _mag_frame(t: float) -> np.ndarray:
    """Magnetorquer X/Y/Z commands"""
    mx = int(np.clip(RNG.normal(128, 10), 100, 155))
    my = int(np.clip(RNG.normal(128, 10), 100, 155))
    mz = int(np.clip(RNG.normal(128,  5), 115, 140))
    status = 0x01
    seq    = int(t * 20) & 0xFF
    return np.array([mx, my, mz, status, seq, 0x00], dtype=np.uint8)


def _comms_frame(t: float) -> np.ndarray:
    """Comms status: TX power, RSSI, frame error count, status"""
    tx_pwr = 2
    rssi   = int(np.clip(RNG.normal(200, 10), 170, 240))
    err    = 0
    status = 0x01
    return np.array([tx_pwr, rssi, err, status], dtype=np.uint8)


def _payload_frame(t: float) -> np.ndarray:
    """Payload diagnostic: status byte + 2 sensor bytes"""
    return np.array([0x44, int(RNG.integers(60, 80)), int(RNG.integers(20, 40))], dtype=np.uint8)


DATA_GEN = {
    0x005: _cdh_frame,
    0x013: _adcs_frame,
    0x020: _wheel_frame,
    0x030: _mag_frame,
    0x040: _comms_frame,
    0x050: _payload_frame,
}


# ── Normal traffic generator ─────────────────────────────────────────────────

def generate_normal(duration_s: float = 300.0) -> pd.DataFrame:
    """Generate normal CAN traffic for `duration_s` seconds."""
    events = []   # (timestamp, can_id, dlc, data_fn)

    for name, cfg in SUBSYSTEMS.items():
        period = 1.0 / cfg['hz']
        t = RNG.uniform(0, period)   # staggered start
        while t < duration_s:
            # Add small jitter (~2% of period)
            jitter = RNG.normal(0, period * 0.02)
            events.append((t + jitter, cfg['id'], cfg['dlc']))
            t += period

    events.sort(key=lambda e: e[0])

    rows = []
    for ts, cid, dlc in events:
        data = DATA_GEN[cid](ts)[:dlc]
        row = {'timestamp': round(ts, 6), 'can_id': cid, 'dlc': dlc,
               'label': 0, 'attack_type': 'normal'}
        for i, b in enumerate(data):
            row[f'd{i}'] = int(b)
        for i in range(len(data), 8):
            row[f'd{i}'] = 0
        rows.append(row)

    return pd.DataFrame(rows)


# ── Attack generators ────────────────────────────────────────────────────────

def inject_dos(normal_df: pd.DataFrame, target_id: int = 0x013,
               multiplier: int = 10, fraction: float = 0.15) -> pd.DataFrame:
    """
    DoS attack: flood bus with target_id at multiplier× its normal rate.
    Injected into a random `fraction` of the timeline.
    """
    duration = normal_df['timestamp'].max()
    start = RNG.uniform(0, duration * (1 - fraction))
    end   = start + duration * fraction

    attack_rows = []
    # Estimate target period from normal traffic
    target_times = normal_df[normal_df['can_id'] == target_id]['timestamp'].values
    if len(target_times) < 2:
        period = 0.1
    else:
        period = float(np.median(np.diff(target_times))) / multiplier

    t = start
    while t < end:
        data = DATA_GEN[target_id](t)
        row = {'timestamp': round(t, 6), 'can_id': target_id,
               'dlc': SUBSYSTEMS[[k for k, v in SUBSYSTEMS.items() if v['id'] == target_id][0]]['dlc'],
               'label': 1, 'attack_type': 'dos'}
        for i, b in enumerate(data):
            row[f'd{i}'] = int(b)
        for i in range(len(data), 8):
            row[f'd{i}'] = 0
        attack_rows.append(row)
        t += period + RNG.normal(0, period * 0.01)

    attack_df = pd.DataFrame(attack_rows)
    combined  = pd.concat([normal_df, attack_df], ignore_index=True)
    return combined.sort_values('timestamp').reset_index(drop=True)


def inject_fuzzy(normal_df: pd.DataFrame, fraction: float = 0.10) -> pd.DataFrame:
    """
    Fuzzy attack: inject frames with random IDs (some known, some unknown)
    and random data at irregular intervals.
    """
    duration = normal_df['timestamp'].max()
    start = RNG.uniform(0, duration * (1 - fraction))
    end   = start + duration * fraction
    n_inject = int((end - start) * 200)   # ~200 Hz injection rate

    timestamps = np.sort(RNG.uniform(start, end, n_inject))
    attack_rows = []
    for ts in timestamps:
        # Mix of known and unknown IDs
        if RNG.random() < 0.4:
            cid = int(RNG.choice(list(KNOWN_IDS)))
        else:
            cid = int(RNG.integers(0x060, 0x7FF))
        dlc  = int(RNG.integers(1, 9))
        data = RNG.integers(0, 256, 8, dtype=np.uint8)
        row  = {'timestamp': round(float(ts), 6), 'can_id': cid, 'dlc': dlc,
                'label': 1, 'attack_type': 'fuzzy'}
        for i in range(8):
            row[f'd{i}'] = int(data[i])
        attack_rows.append(row)

    attack_df = pd.DataFrame(attack_rows)
    combined  = pd.concat([normal_df, attack_df], ignore_index=True)
    return combined.sort_values('timestamp').reset_index(drop=True)


def inject_spoofing(normal_df: pd.DataFrame,
                    target_id: int = 0x020, fraction: float = 0.15) -> pd.DataFrame:
    """
    Spoofing: replace WHEEL_SPEED frames with crafted values (RPM=0)
    to mask a reaction wheel fault.
    """
    df = normal_df.copy()
    duration = df['timestamp'].max()
    start = RNG.uniform(0, duration * (1 - fraction))
    end   = start + duration * fraction

    mask = (df['can_id'] == target_id) & (df['timestamp'] >= start) & (df['timestamp'] <= end)
    # Craft: RPM = 0, status = 0x01 (looks normal)
    df.loc[mask, 'd0'] = 0
    df.loc[mask, 'd1'] = 0
    df.loc[mask, 'd2'] = 1
    df.loc[mask, 'label'] = 1
    df.loc[mask, 'attack_type'] = 'spoofing'
    return df


def inject_replay(normal_df: pd.DataFrame,
                  target_id: int = 0x013, fraction: float = 0.12) -> pd.DataFrame:
    """
    Replay: capture a valid ADCS frame and re-inject it repeatedly
    (e.g., to hide attitude changes during a manoeuvre).
    """
    duration = normal_df['timestamp'].max()
    # Capture a frame from the first 10% of the session
    capture_window = normal_df[
        (normal_df['can_id'] == target_id) &
        (normal_df['timestamp'] < duration * 0.1)
    ]
    if capture_window.empty:
        return normal_df
    captured = capture_window.iloc[0]

    lo = duration * 0.3
    hi = duration * (1 - fraction)
    start = RNG.uniform(lo, hi) if hi > lo else lo
    end   = start + duration * fraction
    period = 0.1   # inject at 10 Hz

    attack_rows = []
    t = start
    while t < end:
        row = {'timestamp': round(t, 6), 'can_id': int(captured['can_id']),
               'dlc': int(captured['dlc']), 'label': 1, 'attack_type': 'replay'}
        for i in range(8):
            row[f'd{i}'] = int(captured[f'd{i}'])
        attack_rows.append(row)
        t += period

    attack_df = pd.DataFrame(attack_rows)
    combined  = pd.concat([normal_df, attack_df], ignore_index=True)
    return combined.sort_values('timestamp').reset_index(drop=True)


# ── Top-level builder ────────────────────────────────────────────────────────

def build_dataset(duration_s: float = 600.0,
                  attack_fraction: float = 0.30,
                  output_path: Path = None) -> pd.DataFrame:
    """
    Build a labelled CAN bus dataset.
    ~70% normal traffic, ~30% attack traffic across all four attack types.
    """
    print(f"  Generating {duration_s}s of normal CAN traffic...")
    normal = generate_normal(duration_s)
    print(f"  Normal frames: {len(normal):,}")

    per_attack = attack_fraction / 4
    print("  Injecting attacks...")

    # Apply attacks on separate copies of normal traffic, then combine
    dos_df      = inject_dos(     generate_normal(duration_s * per_attack), fraction=1.0)
    fuzzy_df    = inject_fuzzy(   generate_normal(duration_s * per_attack), fraction=1.0)
    spoof_df    = inject_spoofing(generate_normal(duration_s * per_attack), fraction=1.0)
    replay_df   = inject_replay(  generate_normal(duration_s * per_attack), fraction=1.0)

    # Filter to only attack frames from attack datasets, combine with normal
    attacks = pd.concat([
        dos_df[dos_df['label'] == 1],
        fuzzy_df[fuzzy_df['label'] == 1],
        spoof_df[spoof_df['label'] == 1],
        replay_df[replay_df['label'] == 1],
    ], ignore_index=True)

    combined = pd.concat([normal, attacks], ignore_index=True)
    combined = combined.sort_values('timestamp').reset_index(drop=True)

    # Re-index timestamps to be monotonically increasing globally
    combined['timestamp'] = combined['timestamp'].rank(method='first') * 0.001

    attack_ratio = combined['label'].mean()
    print(f"  Total frames: {len(combined):,}  Attack ratio: {attack_ratio:.1%}")
    print(f"  Attack breakdown:")
    for atype, cnt in combined[combined['label'] == 1]['attack_type'].value_counts().items():
        print(f"    {atype:12} {cnt:6,}")

    if output_path:
        combined.to_csv(output_path, index=False)
        print(f"  Saved → {output_path}")

    return combined


if __name__ == '__main__':
    out = Path('/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection/datasets/CAN_SATELLITE')
    out.mkdir(parents=True, exist_ok=True)
    df = build_dataset(duration_s=600.0, output_path=out / 'satellite_can_train.csv')
    test_df = build_dataset(duration_s=120.0, output_path=out / 'satellite_can_test.csv')
    print("Done.")
