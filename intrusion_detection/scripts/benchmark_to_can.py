#!/usr/bin/env python3
"""
Convert UNSW-NB15 or NSL-KDD tabular IDS datasets into CAN-frame streams.

Output schema (compatible with CAN pipeline):
  timestamp, can_id, dlc, d0..d7, label, attack_type

Design:
  - One tabular sample becomes multiple CAN frames:
      1 meta frame + N feature payload frames (8 bytes per frame)
  - Features are encoded to uint8 with min-max scaling learned on TRAIN split.
  - Attack label is attached to every generated frame for supervised training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


ROOT = Path('/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection')
DATASETS_DIR = ROOT / 'datasets'

# Dataset-specific CAN ID ranges
CAN_LAYOUT = {
    'unsw': {
        'meta_id': 0x100,
        'feature_base_id': 0x200,
        'dataset_code': 1,
    },
    'nsl': {
        'meta_id': 0x180,
        'feature_base_id': 0x300,
        'dataset_code': 2,
    },
}


def _encode_categoricals(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Label-encode object columns consistently across train/test."""
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    n_train = len(train_df)

    for col in combined.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))

    combined = combined.apply(pd.to_numeric, errors='coerce').fillna(0)
    return combined.iloc[:n_train].copy(), combined.iloc[n_train:].copy()


def load_unsw() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    train_df = pd.read_parquet(DATASETS_DIR / 'UNSW_NB15_training-set.parquet')
    test_df = pd.read_parquet(DATASETS_DIR / 'UNSW_NB15_testing-set.parquet')

    label_col = 'label' if 'label' in train_df.columns else 'Attack'
    attack_col = 'attack_cat' if 'attack_cat' in train_df.columns else None

    y_train = train_df[label_col].astype(int).values
    y_test = test_df[label_col].astype(int).values

    if attack_col is not None:
        attack_train = np.where(y_train == 0, 'normal', train_df[attack_col].astype(str).values)
        attack_test = np.where(y_test == 0, 'normal', test_df[attack_col].astype(str).values)
    else:
        attack_train = np.where(y_train == 0, 'normal', 'attack')
        attack_test = np.where(y_test == 0, 'normal', 'attack')

    drop_cols = [label_col]
    if attack_col is not None:
        drop_cols.append(attack_col)

    X_train_df = train_df.drop(columns=drop_cols)
    X_test_df = test_df.drop(columns=drop_cols)
    X_train_df, X_test_df = _encode_categoricals(X_train_df, X_test_df)

    feature_names = X_train_df.columns.tolist()
    return (
        X_train_df.values.astype(np.float32),
        X_test_df.values.astype(np.float32),
        y_train,
        y_test,
        feature_names,
        attack_train,
        attack_test,
    )


def load_nsl() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
        'label', 'difficulty_score'
    ]

    train_df = pd.read_csv(DATASETS_DIR / 'NSL-KDD' / 'KDDTrain+.txt', header=None, names=columns)
    test_df = pd.read_csv(DATASETS_DIR / 'NSL-KDD' / 'KDDTest+.txt', header=None, names=columns)

    y_train = (train_df['label'] != 'normal').astype(int).values
    y_test = (test_df['label'] != 'normal').astype(int).values

    attack_train = np.where(y_train == 0, 'normal', train_df['label'].astype(str).values)
    attack_test = np.where(y_test == 0, 'normal', test_df['label'].astype(str).values)

    X_train_df = train_df.drop(columns=['label', 'difficulty_score'])
    X_test_df = test_df.drop(columns=['label', 'difficulty_score'])
    X_train_df, X_test_df = _encode_categoricals(X_train_df, X_test_df)

    feature_names = X_train_df.columns.tolist()
    return (
        X_train_df.values.astype(np.float32),
        X_test_df.values.astype(np.float32),
        y_train,
        y_test,
        feature_names,
        attack_train,
        attack_test,
    )


def _fit_uint8_scaler(X_train: np.ndarray) -> dict:
    mins = X_train.min(axis=0)
    maxs = X_train.max(axis=0)
    spans = np.where((maxs - mins) < 1e-12, 1.0, (maxs - mins))
    return {'min': mins, 'max': maxs, 'span': spans}


def _to_uint8(X: np.ndarray, scaler: dict) -> np.ndarray:
    z = (X - scaler['min']) / scaler['span']
    z = np.clip(z, 0.0, 1.0)
    return np.rint(z * 255.0).astype(np.uint8)


def _sample_rows(X: np.ndarray, y: np.ndarray, attack_names: np.ndarray, max_rows: int, rng: np.random.Generator):
    if max_rows <= 0 or len(X) <= max_rows:
        return X, y, attack_names

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    ratio_attack = float(len(idx1)) / max(1, len(y))
    n1 = min(len(idx1), int(max_rows * ratio_attack))
    n0 = max_rows - n1

    if n0 > len(idx0):
        n0 = len(idx0)
        n1 = max_rows - n0
        n1 = min(n1, len(idx1))

    chosen = np.concatenate([
        rng.choice(idx0, size=n0, replace=False),
        rng.choice(idx1, size=n1, replace=False),
    ])
    rng.shuffle(chosen)
    return X[chosen], y[chosen], attack_names[chosen]


def tabular_to_can_frames(
    X_u8: np.ndarray,
    y: np.ndarray,
    attack_names: np.ndarray,
    dataset: str,
    split_code: int,
    start_ts: float = 0.0,
    frame_dt: float = 0.001,
) -> pd.DataFrame:
    """Encode each sample into CAN frames using fixed IDs and byte chunks."""
    layout = CAN_LAYOUT[dataset]
    n_features = X_u8.shape[1]
    n_chunks = (n_features + 7) // 8

    rows = []
    ts = start_ts

    for sample_idx in range(len(X_u8)):
        label = int(y[sample_idx])
        attack_type = str(attack_names[sample_idx])

        # Meta frame
        meta = np.zeros(8, dtype=np.uint8)
        meta[0] = sample_idx & 0xFF
        meta[1] = (sample_idx >> 8) & 0xFF
        meta[2] = (sample_idx >> 16) & 0xFF
        meta[3] = label
        meta[4] = n_chunks
        meta[5] = layout['dataset_code']
        meta[6] = split_code
        meta[7] = 0
        row = {
            'timestamp': round(ts, 6),
            'can_id': layout['meta_id'],
            'dlc': 8,
            'label': label,
            'attack_type': attack_type,
        }
        for i in range(8):
            row[f'd{i}'] = int(meta[i])
        rows.append(row)
        ts += frame_dt

        # Feature payload frames
        feat = X_u8[sample_idx]
        for chunk_idx in range(n_chunks):
            start = chunk_idx * 8
            end = min(start + 8, n_features)
            payload = np.zeros(8, dtype=np.uint8)
            payload[:(end - start)] = feat[start:end]
            dlc = end - start

            row = {
                'timestamp': round(ts, 6),
                'can_id': int(layout['feature_base_id'] + chunk_idx),
                'dlc': int(dlc),
                'label': label,
                'attack_type': attack_type,
            }
            for i in range(8):
                row[f'd{i}'] = int(payload[i])
            rows.append(row)
            ts += frame_dt

    return pd.DataFrame(rows)


def build_benchmark_can_dataset(dataset: str, out_dir: Path, max_train_rows: int = 120000, max_test_rows: int = 50000) -> dict:
    rng = np.random.default_rng(42)

    if dataset == 'unsw':
        X_train, X_test, y_train, y_test, feature_names, a_train, a_test = load_unsw()
    elif dataset == 'nsl':
        X_train, X_test, y_train, y_test, feature_names, a_train, a_test = load_nsl()
    else:
        raise ValueError(f'Unsupported dataset: {dataset}')

    X_train, y_train, a_train = _sample_rows(X_train, y_train, a_train, max_train_rows, rng)
    X_test, y_test, a_test = _sample_rows(X_test, y_test, a_test, max_test_rows, rng)

    scaler = _fit_uint8_scaler(X_train)
    X_train_u8 = _to_uint8(X_train, scaler)
    X_test_u8 = _to_uint8(X_test, scaler)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_can = tabular_to_can_frames(X_train_u8, y_train, a_train, dataset=dataset, split_code=1)
    test_can = tabular_to_can_frames(X_test_u8, y_test, a_test, dataset=dataset, split_code=2, start_ts=float(train_can['timestamp'].max() + 0.01))

    train_path = out_dir / f'{dataset}_can_train.csv'
    test_path = out_dir / f'{dataset}_can_test.csv'
    train_can.to_csv(train_path, index=False)
    test_can.to_csv(test_path, index=False)

    scaler_meta = {
        'dataset': dataset,
        'feature_names': feature_names,
        'feature_min': scaler['min'].astype(float).tolist(),
        'feature_max': scaler['max'].astype(float).tolist(),
        'n_features': len(feature_names),
        'train_tabular_rows': int(len(X_train)),
        'test_tabular_rows': int(len(X_test)),
        'train_can_frames': int(len(train_can)),
        'test_can_frames': int(len(test_can)),
    }
    with open(out_dir / f'{dataset}_can_encoding_meta.json', 'w') as f:
        json.dump(scaler_meta, f, indent=2)

    return {
        'dataset': dataset,
        'train_path': str(train_path),
        'test_path': str(test_path),
        'meta_path': str(out_dir / f'{dataset}_can_encoding_meta.json'),
        'n_features': len(feature_names),
        'train_rows': int(len(X_train)),
        'test_rows': int(len(X_test)),
        'train_frames': int(len(train_can)),
        'test_frames': int(len(test_can)),
    }


def main():
    ap = argparse.ArgumentParser(description='Convert UNSW or NSL tabular IDS data into CAN frame datasets')
    ap.add_argument('--dataset', choices=['unsw', 'nsl'], required=True)
    ap.add_argument('--out-dir', default=str(ROOT / 'datasets' / 'CAN_FROM_BENCHMARK'))
    ap.add_argument('--max-train-rows', type=int, default=120000)
    ap.add_argument('--max-test-rows', type=int, default=50000)
    args = ap.parse_args()

    info = build_benchmark_can_dataset(
        dataset=args.dataset,
        out_dir=Path(args.out_dir),
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
    )

    print('=' * 72)
    print(f"Converted {args.dataset.upper()} to CAN stream format")
    print('=' * 72)
    print(f"Tabular rows  : train={info['train_rows']:,} test={info['test_rows']:,}")
    print(f"CAN frames    : train={info['train_frames']:,} test={info['test_frames']:,}")
    print(f"Feature count : {info['n_features']}")
    print(f"Train CSV     : {info['train_path']}")
    print(f"Test CSV      : {info['test_path']}")
    print(f"Meta JSON     : {info['meta_path']}")


if __name__ == '__main__':
    main()
