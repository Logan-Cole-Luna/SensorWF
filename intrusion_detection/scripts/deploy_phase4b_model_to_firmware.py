#!/usr/bin/env python3
"""
Deploy a phase4b benchmark-to-CAN model into firmware include headers used by EmbeddedBabel.

Copies selected float32 model header + generated scaler header into:
  intrusion_detection/firmware/stm32h7_can_ids_float32.h
  intrusion_detection/firmware/stm32h7_can_ids_scaler.h

This keeps existing EmbeddedBabel includes unchanged while swapping model payload.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path('/Users/loganluna/VSCode/SatelliteAnomalyDetection/intrusion_detection')
MODELS = ROOT / 'models' / 'trained_models'
FIRMWARE = ROOT / 'firmware'


def main():
    ap = argparse.ArgumentParser(description='Deploy phase4b model headers to firmware include names')
    ap.add_argument('--dataset', choices=['unsw', 'nsl'], required=True)
    args = ap.parse_args()

    model_src = MODELS / f'stm32f373_{args.dataset}_can_ids_float32.h'
    scaler_src = MODELS / f'stm32f373_{args.dataset}_can_ids_scaler.h'

    if not model_src.exists():
        raise FileNotFoundError(f'Model header not found: {model_src}')
    if not scaler_src.exists():
        raise FileNotFoundError(f'Scaler header not found: {scaler_src}')

    model_dst = FIRMWARE / 'stm32h7_can_ids_float32.h'
    scaler_dst = FIRMWARE / 'stm32h7_can_ids_scaler.h'

    shutil.copyfile(model_src, model_dst)
    shutil.copyfile(scaler_src, scaler_dst)

    print('=' * 72)
    print(f'Deployed {args.dataset.upper()} benchmark-to-CAN model to firmware headers')
    print('=' * 72)
    print(f'Model  : {model_src.name} -> {model_dst}')
    print(f'Scaler : {scaler_src.name} -> {scaler_dst}')


if __name__ == '__main__':
    main()
