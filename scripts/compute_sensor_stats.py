"""Compute global min/max of full-resolution sensor measurements.

Iterates the training split of HFD_data (sensor_down_sample_rate=1),
computes `gt_norm @ R_matrix` per sample, and accumulates scalar
min/max across all samples, pixels, and sensor channels. Writes a
JSON file used by HFD_SensorSR_data for [-1, 1] normalization.

Usage:
    python -m scripts.compute_sensor_stats --R-n 1
"""
import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm

# allow running as module or script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader.HFD_dataset import HFD_data


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str,
                   default='dataset/HFD100 Mat dataset')
    p.add_argument('--R-n', type=int, default=1, dest='R_n')
    p.add_argument('--type', type=str, default='Flower',
                   choices=['Flower', 'Leaves', 'Scenses'])
    p.add_argument('--eval_ratio', type=float, default=0.1)
    p.add_argument('--out_path', type=str, default=None,
                   help='Override output JSON path. Default: '
                        '<data_path>/sensor_stats_R{n}.json')
    return p.parse_args()


def main():
    args = parse_args()

    ds = HFD_data(
        data_path=args.data_path,
        train_mode='image',
        data_format='image',
        split='train',
        eval_ratio=args.eval_ratio,
        type=args.type,
        R_n=args.R_n,
        sensor_down_sample_rate=1,
    )

    # HFD_data.__getitem__ already returns the full-res sensor_data
    # (no downsampling because sensor_down_sample_rate=1 on the image path).
    smin, smax = np.inf, -np.inf
    n = 0
    for i in tqdm(range(len(ds)), desc='stats'):
        _, sensor_data = ds[i]  # [H, W, 30] float32
        smin = min(smin, float(sensor_data.min()))
        smax = max(smax, float(sensor_data.max()))
        n += 1

    out_path = args.out_path or os.path.join(
        args.data_path, f'sensor_stats_R{args.R_n}.json'
    )
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(
            {
                'sensor_min': smin,
                'sensor_max': smax,
                'n_samples': n,
                'dataset': args.data_path,
                'R_n': args.R_n,
                'type': args.type,
            },
            f, indent=2,
        )
    print(f'Wrote {out_path}: min={smin:.6f} max={smax:.6f} n={n}')


if __name__ == '__main__':
    main()
