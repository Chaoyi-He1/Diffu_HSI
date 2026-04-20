"""Dataset for sensor super-resolution diffusion.

Yields (full_sensor, lowres_sensor) pairs where:
  - full_sensor  : [H, W, 30] float32, normalized to [-1, 1]
  - lowres_sensor: [H/ds, W/ds, 30] float32, same normalization

Full-res sensor is x_0 for diffusion; low-res sensor is the conditioning.
Normalization uses a precomputed global min/max (see
scripts/compute_sensor_stats.py).
"""
import json
import os

import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as Dataset
from scipy.interpolate import interp1d


class HFD_SensorSR_data(Dataset.Dataset):
    def __init__(
        self,
        data_path,
        stats_path,
        split='train',
        eval_ratio=0.1,
        type='Flower',
        R_n=1,
        sr_downsample_rate=8,
        sr_downsample_method='strided',
    ):
        super().__init__()
        assert split in ('train', 'test')
        assert type in ('Flower', 'Leaves', 'Scenses')
        assert sr_downsample_method in ('strided', 'avg_pool')

        self.data_path = data_path
        self.split = split
        self.eval_ratio = eval_ratio
        self.type = type
        self.R_n = R_n
        self.sr_downsample_rate = sr_downsample_rate
        self.sr_downsample_method = sr_downsample_method

        # --- file list (mirrors HFD_data) ---
        self.data_folder = os.path.join(
            data_path, f'Mat{type}60', 'Train'
        )
        self.img_list = []
        for folder in os.listdir(self.data_folder):
            folder_path = os.path.join(self.data_folder, folder)
            if os.path.isdir(folder_path):
                for f in os.listdir(folder_path):
                    if f.endswith('.mat'):
                        self.img_list.append(os.path.join(folder_path, f))
        self.img_list.sort()
        self.img_list = self.img_list[:10000]  # match HFD_data truncation
        self._split_data()

        # --- sensor response (mirrors HFD_data.load_sensor_response_new) ---
        self.wavelens = np.linspace(451, 855, 31)
        self._load_sensor_response(R_n)

        # --- global normalization stats ---
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f'sensor stats JSON not found at {stats_path}. '
                f'Run: python -m scripts.compute_sensor_stats --R-n {R_n}'
            )
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        self.sensor_min = float(stats['sensor_min'])
        self.sensor_max = float(stats['sensor_max'])
        print(f'[HFD_SensorSR_data] global stats: '
              f'min={self.sensor_min:.4f} max={self.sensor_max:.4f}')

    def _split_data(self):
        n_total = len(self.img_list)
        n_eval = int(n_total * self.eval_ratio)
        n_train = n_total - n_eval
        if self.split == 'train':
            self.img_list = self.img_list[:n_train]
        else:
            self.img_list = self.img_list[n_train:]

    def _load_sensor_response(self, n):
        path = os.path.join(self.data_path, f'R_Device{n}.mat')
        R = np.array(sio.loadmat(path)['R'])  # [C, N]
        N_total = R.shape[1]
        indices = np.linspace(0, N_total - 1, 30, dtype=int).tolist()
        R = R[:, indices]

        sensor_wavelens = np.linspace(400, 1000, R.shape[0])
        f = interp1d(sensor_wavelens, R, axis=0, kind='linear',
                     bounds_error=False, fill_value='extrapolate')
        R = f(self.wavelens)  # [31, 30]

        R_min = R.min(axis=0)
        R_max = R.max(axis=0)
        R = (R - R_min) / (R_max - R_min + 1e-20)
        self.sensor_R_matrix = R  # [31, 30]

    def _normalize(self, x):
        """Scale x to [-1, 1] using global min/max, then clip."""
        rng = self.sensor_max - self.sensor_min + 1e-20
        x = 2.0 * (x - self.sensor_min) / rng - 1.0
        return np.clip(x, -1.0, 1.0)

    def _downsample(self, full_sensor):
        ds = self.sr_downsample_rate
        if ds <= 1:
            return full_sensor
        if self.sr_downsample_method == 'strided':
            return full_sensor[::ds, ::ds, :]
        # avg_pool: reshape-based block mean
        H, W, C = full_sensor.shape
        Hn, Wn = H // ds, W // ds
        return (
            full_sensor[:Hn * ds, :Wn * ds, :]
            .reshape(Hn, ds, Wn, ds, C)
            .mean(axis=(1, 3))
        )

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        gt = sio.loadmat(self.img_list[idx])['truth']  # [H, W, 31]
        gt = np.array(gt, dtype=np.float32)

        # Per-sample gt → [-1, 1] (matches HFD_data for R-matrix input scale)
        g_min, g_max = gt.min(), gt.max()
        gt = (gt - g_min) / (g_max - g_min + 1e-20)
        gt = gt * 2.0 - 1.0

        H, W, C = gt.shape
        assert H % self.sr_downsample_rate == 0 and \
               W % self.sr_downsample_rate == 0, (
            f'image {H}x{W} not divisible by ds={self.sr_downsample_rate}'
        )

        full_sensor = (gt.reshape(-1, C) @ self.sensor_R_matrix) \
            .reshape(H, W, -1).astype(np.float32)  # [H, W, 30]

        full_sensor = self._normalize(full_sensor)
        lowres_sensor = self._downsample(full_sensor)

        return full_sensor, lowres_sensor


def sensor_sr_image_collate_fn(batch):
    """Stack [H,W,30] → [B,30,H,W] for both full-res and low-res."""
    full, low = list(zip(*batch))
    full = torch.tensor(np.stack(full, axis=0), dtype=torch.float32)
    low = torch.tensor(np.stack(low, axis=0), dtype=torch.float32)
    full = full.permute(0, 3, 1, 2).contiguous()  # [B, 30, H, W]
    low = low.permute(0, 3, 1, 2).contiguous()   # [B, 30, H/ds, W/ds]
    return full, low


if __name__ == '__main__':
    ds = HFD_SensorSR_data(
        data_path='dataset/HFD100 Mat dataset',
        stats_path='dataset/HFD100 Mat dataset/sensor_stats_R1.json',
        split='train',
        R_n=1,
        sr_downsample_rate=8,
        sr_downsample_method='strided',
    )
    print(f'num samples: {len(ds)}')
    full, low = ds[0]
    print(f'full_sensor: shape={full.shape} dtype={full.dtype} '
          f'min={full.min():.4f} max={full.max():.4f}')
    print(f'lowres_sensor: shape={low.shape} dtype={low.dtype} '
          f'min={low.min():.4f} max={low.max():.4f}')
    assert full.shape[-1] == 30
    assert low.shape[-1] == 30
    assert low.shape[0] == full.shape[0] // 8
    assert full.min() >= -1.0 - 1e-6 and full.max() <= 1.0 + 1e-6
    # Also exercise avg_pool path.
    ds2 = HFD_SensorSR_data(
        data_path='dataset/HFD100 Mat dataset',
        stats_path='dataset/HFD100 Mat dataset/sensor_stats_R1.json',
        sr_downsample_rate=8, sr_downsample_method='avg_pool',
    )
    f2, l2 = ds2[0]
    print(f'avg_pool lowres: shape={l2.shape} '
          f'min={l2.min():.4f} max={l2.max():.4f}')
    # Collate
    batch = [ds[0], ds[1]]
    fb, lb = sensor_sr_image_collate_fn(batch)
    print(f'collated: full={fb.shape} low={lb.shape}')
    assert fb.shape[1] == 30 and lb.shape[1] == 30
    print('OK')
