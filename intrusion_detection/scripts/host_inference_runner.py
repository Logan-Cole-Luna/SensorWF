#!/usr/bin/env python3
"""
host_inference_runner.py
========================
Drives the STM32F373C8T IDS benchmark firmware over UART and collects results.

Sends the pre-extracted CAN feature test set to the MCU, receives per-sample
measurements and the final summary, then writes a full report.

Usage:
    python host_inference_runner.py --port /dev/tty.usbmodem14101 [--samples 1000]
    python host_inference_runner.py --port /dev/ttyUSB0            [--samples all]
    python host_inference_runner.py --dry-run                       # simulates MCU

Outputs:
    results/stm32f373_benchmark_raw.csv    — per-sample: cycles, stack, heap, pred, gt
    results/stm32f373_benchmark_report.json — full measurements + ML scores
    results/stm32f373_benchmark_report.txt  — human-readable summary
"""

import sys
sys.path.insert(0, '/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection')

import argparse
import struct
import time
import json
import csv
import math
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)

# ── Paths ────────────────────────────────────────────────────────────────────

MODELS_DIR  = Path('models/trained_models')
RESULTS_DIR = Path('results')
DATASET_DIR = Path('datasets/CAN_SATELLITE')
RESULTS_DIR.mkdir(exist_ok=True)

FEATURE_NAMES = [
    'can_id_norm', 'dlc', 'data_mean', 'data_std', 'data_entropy',
    'data_range', 'hamming_dist', 'inter_arrival_mean', 'inter_arrival_std',
    'id_freq', 'bus_load', 'unique_ids', 'dlc_anomaly', 'id_is_known',
    'payload_delta',
]

# ── Wire format (must match firmware structs exactly) ─────────────────────────

# ids_sample_result_t: uint8 pred, uint8 gt, uint32 cycles, uint32 hwm,
#                      uint32 heap_free, uint32 reserved  → 16 bytes
SAMPLE_FMT  = '<BBIIIi'
SAMPLE_SIZE = struct.calcsize(SAMPLE_FMT)   # 16

# ids_bench_summary_t: matches C struct (n_samples, n_correct, tp, tn, fp, fn,
#   cyc_min, cyc_max, cyc_sum_hi, cyc_sum_lo, stack_hwm_min, energy_nj,
#   inf_us_mean, accuracy, precision, recall, f1, fpr)  → 18 fields
SUMMARY_FMT  = '<IIIIIIIIIIIfffffff'
SUMMARY_SIZE = struct.calcsize(SUMMARY_FMT)

# ── Hardware constants (STM32F373C8T @ 72 MHz, Vdd=3.3V) ────────────────────
# Datasheet Table 28: IDD typ = 29.2 mA @ 72 MHz, 3.6V, peripherals off.
# Derated to 3.3V ≈ 27 mA.
HCLK_HZ         = 72_000_000
VDD_V           = 3.3
RUN_CURRENT_A   = 0.027   # whole-chip active; IDS duty fraction applied below

# ── UART helpers ─────────────────────────────────────────────────────────────

class MCULink:
    """Wraps pyserial to talk to the benchmark firmware."""
    def __init__(self, port: str, baud: int = 115200, timeout: float = 10.0):
        import serial
        self._ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(0.2)
        self._ser.reset_input_buffer()

    def send(self, data: bytes):
        self._ser.write(data)

    def recv(self, n: int) -> bytes:
        buf = b''
        deadline = time.time() + 10.0
        while len(buf) < n:
            chunk = self._ser.read(n - len(buf))
            if not chunk:
                if time.time() > deadline:
                    raise TimeoutError(f'UART timeout waiting for {n} bytes (got {len(buf)})')
            buf += chunk
        return buf

    def flush_log(self):
        """Drain any [IDS] log lines before binary exchange."""
        self._ser.timeout = 0.5
        while True:
            line = self._ser.readline()
            if not line:
                break
            print(f'  MCU log: {line.decode(errors="replace").rstrip()}')
        self._ser.timeout = 10.0

    def close(self):
        self._ser.close()


# ── Dry-run simulator (no hardware needed) ───────────────────────────────────

class DryRunLink:
    """
    Simulates firmware responses using the Python sklearn model.
    Useful for validating the host script without hardware.
    """
    def __init__(self):
        self._clf    = joblib.load(MODELS_DIR / 'can_decision_tree.joblib')
        self._scaler = joblib.load(MODELS_DIR / 'can_scaler.joblib')
        self._pending_features = None
        self._pending_gt       = None
        self._n_samples        = 0
        self._results          = []
        print('  [dry-run] Using sklearn model as MCU simulator')

    def send(self, data: bytes):
        # Parse outgoing bytes to know what the MCU would receive
        if len(data) == 3 and data[0] == ord('S'):
            self._n_samples = struct.unpack('<H', data[1:])[0]
            self._results.clear()
        elif len(data) == 60:   # 15 × float32
            self._pending_features = np.frombuffer(data, dtype=np.float32).copy()
        elif len(data) == 1:    # ground-truth label
            gt = data[0]
            if self._pending_features is not None:
                feat_scaled = self._scaler.transform(self._pending_features.reshape(1, -1))[0]
                pred = int(self._clf.predict(feat_scaled.reshape(1, -1))[0])
                cycles      = int(np.random.normal(115, 8))   # realistic for 480 MHz tree
                stack_bytes = 64    # canary measurement: float[15] + locals
                self._results.append((pred, gt, cycles, stack_bytes))
                self._pending_features = None

    def recv(self, n: int) -> bytes:
        if n == SAMPLE_SIZE and self._results:
            pred, gt, cycles, stack_bytes = self._results[-1]
            return struct.pack(SAMPLE_FMT, pred, gt, cycles, stack_bytes, 0, 0)

        if n == SUMMARY_SIZE:
            res = self._results
            tp = sum(1 for p, g, c, s in res if p == 1 and g == 1)
            tn = sum(1 for p, g, c, s in res if p == 0 and g == 0)
            fp = sum(1 for p, g, c, s in res if p == 1 and g == 0)
            fn = sum(1 for p, g, c, s in res if p == 0 and g == 1)
            ns = len(res)
            cycles_list  = [c for _, _, c, _ in res]
            stack_max    = max(s for _, _, _, s in res)
            acc  = (tp + tn) / ns if ns else 0
            prec = tp / (tp + fp) if (tp + fp) else 0
            rec  = tp / (tp + fn) if (tp + fn) else 0
            f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0
            fpr  = fp / (fp + tn) if (fp + tn) else 0
            cmean = sum(cycles_list) / len(cycles_list) if cycles_list else 0
            inf_us = cmean / (HCLK_HZ / 1e6)
            energy_nj = VDD_V * RUN_CURRENT_A * (inf_us / 1e6) * 1e9
            csum = sum(cycles_list)
            return struct.pack(SUMMARY_FMT,
                ns, tp+tn, tp, tn, fp, fn,
                min(cycles_list), max(cycles_list),
                csum >> 32, csum & 0xFFFFFFFF,
                stack_max,              # stack_used_bytes_max (canary)
                energy_nj, inf_us,
                acc, prec, rec, f1, fpr
            )
        return b'\x00' * n

    def flush_log(self):
        pass

    def close(self):
        pass


# ── Main runner ───────────────────────────────────────────────────────────────

def run(port: str | None, n_samples: int | None, dry_run: bool):
    print("=" * 70)
    print("STM32F373 CAN IDS — On-Board Inference Benchmark")
    print("=" * 70)

    # ── Load test dataset ─────────────────────────────────────────────────
    print("\n[1/4] Loading CAN test features...")
    feat_df = pd.read_csv(DATASET_DIR / 'can_test_features.csv')
    X_raw   = feat_df[FEATURE_NAMES].values.astype(np.float32)
    y_true  = feat_df['label'].values.astype(np.uint8)

    if n_samples is not None:
        X_raw  = X_raw[:n_samples]
        y_true = y_true[:n_samples]

    n = len(X_raw)
    print(f"  Samples: {n:,}  Attack ratio: {y_true.mean():.1%}")

    # ── Connect ───────────────────────────────────────────────────────────
    print("\n[2/4] Connecting to MCU...")
    if dry_run:
        link = DryRunLink()
    else:
        link = MCULink(port)
        link.flush_log()
        print(f"  Connected to {port}")

    # ── Stream samples ────────────────────────────────────────────────────
    print(f"\n[3/4] Streaming {n:,} samples to MCU and collecting measurements...")

    # Send start command: 'S' + uint16 sample count
    link.send(b'S' + struct.pack('<H', n))
    time.sleep(0.05)
    link.flush_log()

    raw_rows = []
    t_stream_start = time.perf_counter()

    for i, (features, label) in enumerate(zip(X_raw, y_true)):
        # Send 15 × float32 (60 bytes) then 1-byte label
        link.send(features.tobytes())
        link.send(bytes([label]))

        # Receive per-sample result (16 bytes)
        data = link.recv(SAMPLE_SIZE)
        pred, gt, cycles, stack_used_bytes, _r0, _r1 = struct.unpack(SAMPLE_FMT, data)

        inf_us  = cycles / (HCLK_HZ / 1e6)
        raw_rows.append({
            'sample_idx':       i,
            'prediction':       pred,
            'ground_truth':     gt,
            'cycles':           cycles,
            'inf_us':           round(inf_us, 4),
            'stack_used_bytes': stack_used_bytes,
        })

        if (i + 1) % 500 == 0 or (i + 1) == n:
            elapsed = time.perf_counter() - t_stream_start
            print(f"  [{i+1:>5}/{n}] elapsed={elapsed:.1f}s  last_inf={inf_us:.2f}µs  "
                  f"stack_used={stack_used_bytes}B")

    stream_total_s = time.perf_counter() - t_stream_start

    # Receive summary (64 bytes)
    summary_data = link.recv(SUMMARY_SIZE)
    link.flush_log()
    link.close()

    (ns, n_correct, tp, tn, fp, fn,
     cyc_min, cyc_max, cyc_sum_hi, cyc_sum_lo,
     hwm_min_words, energy_nj, inf_us_mean,
     acc, prec, rec, f1, fpr) = struct.unpack(SUMMARY_FMT, summary_data)
    _ = n_correct  # firmware convenience field (tp+tn), use h_* below

    cyc_sum  = (cyc_sum_hi << 32) | cyc_sum_lo
    cyc_mean = cyc_sum / ns

    # ── Build report ──────────────────────────────────────────────────────
    print("\n[4/4] Computing report...")

    # Host-side cross-check using raw sample data
    h_pred  = np.array([r['prediction'] for r in raw_rows], dtype=int)
    h_true  = np.array([r['ground_truth'] for r in raw_rows], dtype=int)
    h_acc   = accuracy_score(h_true, h_pred)
    h_prec  = precision_score(h_true, h_pred, zero_division=0)
    h_rec   = recall_score(h_true, h_pred, zero_division=0)
    h_f1    = f1_score(h_true, h_pred, zero_division=0)
    h_tn, h_fp, h_fn, h_tp = confusion_matrix(h_true, h_pred).ravel()
    h_fpr   = h_fp / (h_fp + h_tn) if (h_fp + h_tn) > 0 else 0

    # Per-attack-type breakdown from raw rows joined with original dataset
    attack_types = pd.read_csv(DATASET_DIR / 'satellite_can_test.csv')['attack_type'].values[:n]
    per_attack = {}
    for atype in sorted(set(attack_types)):
        if atype == 'normal':
            continue
        mask = np.array([(at == atype or at == 'normal') for at in attack_types])
        yt_a = (attack_types[mask] != 'normal').astype(int)
        yp_a = h_pred[mask]
        if yt_a.sum() == 0:
            continue
        atn, afp, afn, atp = confusion_matrix(yt_a, yp_a, labels=[0,1]).ravel()
        per_attack[atype] = {
            'recall': round(float(atp/(atp+afn)) if (atp+afn) > 0 else 0, 4),
            'fpr':    round(float(afp/(afp+atn)) if (afp+atn) > 0 else 0, 4),
            'n_attacks': int(yt_a.sum()),
        }

    # Resource measurements
    cycles_arr   = np.array([r['cycles'] for r in raw_rows])
    stack_arr    = np.array([r['stack_used_bytes'] for r in raw_rows])
    inf_us_arr   = cycles_arr / (HCLK_HZ / 1e6)

    # Energy — two estimates:
    #   1. Whole-chip active-run (conservative, from firmware)
    #   2. IDS-only: scale chip current by fraction of time spent in ids_predict
    ids_duty     = float(inf_us_arr.mean()) / (1e6 / 100)   # at 100 Hz poll
    energy_ids_nj = VDD_V * RUN_CURRENT_A * ids_duty * (float(inf_us_arr.mean())/1e6) * 1e9

    # Flash usage: tree + scaler = 429 + ~600 bytes
    tree_flash_bytes  = 429
    scaler_flash_bytes = 15 * 4 * 2   # 15 features × 4 bytes × (mean + scale)
    total_flash_bytes  = tree_flash_bytes + scaler_flash_bytes

    report = {
        'hardware': {
            'mcu': 'STM32F373C8T',
            'hclk_hz': HCLK_HZ,
            'vdd_v': VDD_V,
            'active_run_current_a': RUN_CURRENT_A,
        },
        'model': {
            'name': 'TinyDecisionTree (CAN)',
            'n_features': len(FEATURE_NAMES),
            'feature_names': FEATURE_NAMES,
            'tree_nodes': 39,
            'tree_depth': 5,
            'flash_bytes': total_flash_bytes,
            'flash_kb':    round(total_flash_bytes / 1024, 3),
        },
        'performance': {
            'n_samples': int(ns),
            'accuracy':  round(float(h_acc),  4),
            'precision': round(float(h_prec), 4),
            'recall':    round(float(h_rec),  4),
            'f1':        round(float(h_f1),   4),
            'fpr':       round(float(h_fpr),  4),
            'tp': int(h_tp), 'tn': int(h_tn),
            'fp': int(h_fp), 'fn': int(h_fn),
            'per_attack_type': per_attack,
        },
        'timing': {
            'cycles_min':  int(cyc_min),
            'cycles_max':  int(cyc_max),
            'cycles_mean': round(cyc_mean, 2),
            'inf_us_min':  round(float(cyc_min)  / (HCLK_HZ/1e6), 4),
            'inf_us_max':  round(float(cyc_max)  / (HCLK_HZ/1e6), 4),
            'inf_us_mean': round(float(inf_us_arr.mean()), 4),
            'inf_us_std':  round(float(inf_us_arr.std()),  4),
            'inf_us_p95':  round(float(np.percentile(inf_us_arr, 95)), 4),
            'inf_us_p99':  round(float(np.percentile(inf_us_arr, 99)), 4),
        },
        'resource_usage': {
            'stack_used_bytes_min':  int(stack_arr.min()),
            'stack_used_bytes_max':  int(stack_arr.max()),
            'stack_canary_budget':   1024,
            'stack_headroom_bytes':  1024 - int(stack_arr.max()),
            'flash_bytes_tree':      tree_flash_bytes,
            'flash_bytes_scaler':    scaler_flash_bytes,
            'flash_bytes_total':     total_flash_bytes,
            'ram_bytes_features':    len(FEATURE_NAMES) * 4,
            'note':                  'bare-metal, no heap (no dynamic allocation)',
        },
        'energy': {
            'method': 'whole_chip_active_run_current × inference_duration',
            'note': ('Conservative upper bound. Actual IDS contribution is '
                     '~(inf_us/poll_period) fraction of this.'),
            'whole_chip_energy_per_inf_nj': round(float(energy_nj), 4),
            'ids_duty_fraction_at_100hz':   round(ids_duty, 6),
            'energy_per_inf_nj_estimate':   round(energy_ids_nj, 6),
            'energy_per_inf_pj_estimate':   round(energy_ids_nj * 1000, 4),
            'power_overhead_uw_at_100hz':   round(energy_ids_nj * 1e-9 * 100 * 1e6, 4),
        },
        'host_stream_time_s': round(stream_total_s, 3),
        'firmware_summary_match': bool(abs(acc - h_acc) < 0.001),
    }

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / 'stm32f373_benchmark_raw.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=raw_rows[0].keys())
        writer.writeheader()
        writer.writerows(raw_rows)
    print(f"\n  Raw data → {csv_path}")

    # ── Save JSON ─────────────────────────────────────────────────────────
    json_path = RESULTS_DIR / 'stm32f373_benchmark_report.json'
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  JSON     → {json_path}")

    # ── Print human-readable report ───────────────────────────────────────
    txt_path = RESULTS_DIR / 'stm32f373_benchmark_report.txt'
    lines = _format_report(report)
    txt_path.write_text('\n'.join(lines))
    print(f"  Text     → {txt_path}")
    print()
    print('\n'.join(lines))

    return report


def _format_report(r: dict) -> list[str]:
    hw  = r['hardware']
    m   = r['model']
    p   = r['performance']
    t   = r['timing']
    res = r['resource_usage']
    e   = r['energy']

    lines = [
        "=" * 70,
        "STM32F373 CAN IDS — On-Board Inference Benchmark Report",
        "=" * 70,
        "",
        f"Hardware : {hw['mcu']} @ {hw['hclk_hz']//1_000_000} MHz  "
        f"Vdd={hw['vdd_v']}V  Iactive≈{hw['active_run_current_a']*1000:.0f}mA",
        f"Model    : {m['name']}  {m['tree_nodes']} nodes  depth {m['tree_depth']}",
        f"Features : {m['n_features']}  ({', '.join(m['feature_names'][:5])}, ...)",
        f"Dataset  : {p['n_samples']:,} CAN frames (satellite subsystems)",
        "",
        "── ML Performance ─────────────────────────────────────────────────",
        f"  Accuracy  : {p['accuracy']:.4f}",
        f"  Precision : {p['precision']:.4f}",
        f"  Recall    : {p['recall']:.4f}",
        f"  F1-Score  : {p['f1']:.4f}",
        f"  FPR       : {p['fpr']:.4f}",
        f"  Confusion : TP={p['tp']}  TN={p['tn']}  FP={p['fp']}  FN={p['fn']}",
        "",
        "  Per-attack-type:",
        f"  {'Attack':12}  {'Recall':>8}  {'FPR':>8}  {'N':>6}",
        f"  {'-'*40}",
    ]
    for atype, am in p['per_attack_type'].items():
        lines.append(f"  {atype:12}  {am['recall']:>8.4f}  {am['fpr']:>8.4f}  {am['n_attacks']:>6,}")

    lines += [
        "",
        "── Timing (on STM32F373C8T) ────────────────────────────────────────",
        f"  Cycles   min/mean/max : {t['cycles_min']} / {t['cycles_mean']:.1f} / {t['cycles_max']}",
        f"  Time(µs) min/mean/max : {t['inf_us_min']} / {t['inf_us_mean']} / {t['inf_us_max']}",
        f"  Time(µs) p95 / p99    : {t['inf_us_p95']} / {t['inf_us_p99']}",
        f"  Std dev               : {t['inf_us_std']} µs",
        f"  At 100 Hz poll rate   : {t['inf_us_mean'] * 100 / 1e6 * 100:.4f}% CPU duty",
        "",
        "── Resource Usage ──────────────────────────────────────────────────",
        f"  Flash  (tree + scaler): {res['flash_bytes_total']} bytes  "
        f"({m['flash_kb']:.3f} KB)",
        f"    Tree arrays          : {res['flash_bytes_tree']} bytes",
        f"    Scaler params        : {res['flash_bytes_scaler']} bytes",
        f"  RAM    (feature buf)  : {res['ram_bytes_features']} bytes  "
        f"(float[{m['n_features']}] on stack)",
        f"  Stack  used (canary)  : {res['stack_used_bytes_max']} bytes  "
        f"(headroom: {res['stack_headroom_bytes']} bytes)",
        f"  Heap                  : {res['note']}",
        "",
        "── Energy (IDS model only) ─────────────────────────────────────────",
        f"  Method: {e['method']}",
        f"  Whole-chip per-inf    : {e['whole_chip_energy_per_inf_nj']:.4f} nJ",
        f"  IDS duty @ 100 Hz     : {e['ids_duty_fraction_at_100hz']*100:.5f}%",
        f"  IDS power overhead    : {e['power_overhead_uw_at_100hz']:.4f} µW  "
        f"(at 100 Hz continuous)",
        f"  Note: {e['note']}",
        "",
        "=" * 70,
    ]
    return lines


# ── CLI ────────────────────────────────────────────────────────────────────────

IDS_TASK_STACK_BYTES = 1024 * 4

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='STM32F373 CAN IDS host runner')
    parser.add_argument('--port',    default=None,
                        help='Serial port (e.g. /dev/tty.usbmodem14101)')
    parser.add_argument('--baud',    type=int, default=115200)
    parser.add_argument('--samples', default='all',
                        help='Number of samples to run (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Simulate MCU using the sklearn model (no hardware needed)')
    args = parser.parse_args()

    if not args.dry_run and args.port is None:
        print("Error: --port required unless --dry-run")
        sys.exit(1)

    n = None if args.samples == 'all' else int(args.samples)
    run(port=args.port, n_samples=n, dry_run=args.dry_run)
