# Sensor Super-Resolution Diffusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second diffusion model that spatially upsamples sensor measurements (e.g. `[B,30,H/8,W/8] → [B,30,H,W]`), leaving the existing HSI-diffusion path untouched.

**Architecture:** New `U2NetSensorSR` model (copy of `U2NetHyperspectral` with a low-res-aware context encoder and matching in/out channels), new `HFD_SensorSR_data` dataset (reuses `HFD_data`'s sensor-response pipeline but returns full-res/low-res sensor pairs), one-off stats script for global `[-1,1]` normalization, new FSDP entrypoint.

**Tech Stack:** PyTorch 2.10, FSDP (ZeRO-3), bf16, torchrun, `scipy.io`, `numpy`. Same as current training stack.

**Reference spec:** `docs/superpowers/specs/2026-04-19-sensor-sr-diffusion-design.md`

**Testing convention:** This repo has no pytest suite. Smoke tests go in `if __name__ == "__main__":` blocks (see `model/u2net_hyperspectral.py:491-528` for the established pattern) and are executed via `python -m path.to.file`.

---

## File Structure

New files:
- `scripts/compute_sensor_stats.py` — one-shot CLI that writes `sensor_stats_R{n}.json`.
- `data_loader/HFD_sensor_sr_dataset.py` — `HFD_SensorSR_data` + `sensor_sr_image_collate_fn`.
- `model/u2net_sensor_sr.py` — `U2NetSensorSR` (U²-Net backbone + low-res context encoder).
- `train_eval/train_sensor_sr.py` — training / validation one-epoch helpers.
- `main_2d_sensor_sr_fsdp.py` — FSDP entrypoint.
- `job_sensor_sr.slurm` — SLURM script.

Generated artifact (not in git):
- `dataset/HFD100 Mat dataset/sensor_stats_R1.json`

Modified files: none.

---

## Task 1: Stats script

**Files:**
- Create: `scripts/compute_sensor_stats.py`

- [ ] **Step 1.1: Create the `scripts` directory**

```bash
mkdir -p /mnt/shared-scratch/Katehi_L/chaoyi_he/Diffu/scripts
```

- [ ] **Step 1.2: Write `scripts/compute_sensor_stats.py`**

```python
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
```

- [ ] **Step 1.3: Run the stats script**

```bash
cd /mnt/shared-scratch/Katehi_L/chaoyi_he/Diffu
python -m scripts.compute_sensor_stats --R-n 1
```

Expected: produces `dataset/HFD100 Mat dataset/sensor_stats_R1.json` containing numeric `sensor_min` and `sensor_max`. No error. `n_samples == 9000` (first 9000 entries of Flower train split per `HFD_dataset.py:50`).

- [ ] **Step 1.4: Verify output file**

```bash
cat "dataset/HFD100 Mat dataset/sensor_stats_R1.json"
```

Expected: valid JSON with all five keys, `sensor_min < 0 < sensor_max` (because gt_norm is in [-1,1] and R is in [0,1]).

- [ ] **Step 1.5: Commit**

```bash
git add scripts/compute_sensor_stats.py
git commit -m "Add one-shot script to compute global sensor [-1,1] normalization stats"
```

The generated JSON lives under `dataset/…` which is already git-ignored — no need to commit it.

---

## Task 2: Dataset class

**Files:**
- Create: `data_loader/HFD_sensor_sr_dataset.py`

- [ ] **Step 2.1: Write `data_loader/HFD_sensor_sr_dataset.py`**

```python
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
import random

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
```

- [ ] **Step 2.2: Run the dataset smoke test**

```bash
cd /mnt/shared-scratch/Katehi_L/chaoyi_he/Diffu
python -m data_loader.HFD_sensor_sr_dataset
```

Expected: prints `num samples:`, shape/min/max lines for both strided and avg_pool modes, collated batch shapes `[2, 30, H, W]` and `[2, 30, H/8, W/8]`, and final `OK`. No assertion errors.

- [ ] **Step 2.3: Commit**

```bash
git add data_loader/HFD_sensor_sr_dataset.py
git commit -m "Add HFD_SensorSR_data for sensor super-resolution diffusion"
```

---

## Task 3: Model

**Files:**
- Create: `model/u2net_sensor_sr.py`

- [ ] **Step 3.1: Write `model/u2net_sensor_sr.py`**

```python
"""Sensor super-resolution U²-Net for diffusion.

Mirrors U2NetHyperspectral but:
  - in_channels == out_channels == sensor_channels (default 30)
  - context_encoder consumes a low-resolution sensor tensor
    [B, sensor_channels, H/ds, W/ds] and produces features that are
    resampled inside forward() to each stage's spatial resolution.

The U²-Net backbone (U2NetBlock2D), time embedding, attention, and
use_channel_3d_conv toggle are reused verbatim from the existing
implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    BasicTransformerBlock,
    ConvBlock2D,
    ConvBlock2D_ChannelAware,
    TimeEmbedding,
)
from .u2net_hyperspectral import U2NetBlock2D


class U2NetSensorSR(nn.Module):
    """Diffusion denoiser for sensor super-resolution.

    Args:
        sensor_channels (int): channel count for both x_t and cond (default 30)
        base_channels (int): network width
        use_channel_3d_conv (bool): switch backbone conv blocks to 3D
        channel_kernel, spatial_kernel, channel_num_filters: forwarded
            to ConvBlock2D_ChannelAware when use_channel_3d_conv=True.
    """

    def __init__(
        self,
        sensor_channels=30,
        base_channels=64,
        use_channel_3d_conv=False,
        channel_kernel=3,
        spatial_kernel=3,
        channel_num_filters=4,
    ):
        super().__init__()
        self.sensor_channels = sensor_channels
        self.base_channels = base_channels
        time_dim = base_channels * 8

        self.time_embedding = nn.Sequential(
            TimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Context encoder: consumes low-res sensor, produces base_C*8 features
        # at low-res. Spatial alignment to each stage is done in forward().
        self.context_encoder = nn.Sequential(
            nn.Conv2d(sensor_channels, base_channels,
                      kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels * 2,
                      kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.GELU(),
            nn.Conv2d(base_channels * 2, base_channels * 8,
                      kernel_size=1),
        )

        self.input_proj = nn.Conv2d(
            sensor_channels, base_channels, kernel_size=3, padding=1
        )

        block_kwargs = dict(
            time_dim=time_dim,
            use_channel_3d_conv=use_channel_3d_conv,
            channel_kernel=channel_kernel,
            spatial_kernel=spatial_kernel,
            channel_num_filters=channel_num_filters,
        )

        # Bridge conv selection (mirrors u2net_hyperspectral.py:356-364)
        if use_channel_3d_conv:
            BridgeConvCls = ConvBlock2D_ChannelAware
            bridge_conv_kwargs = dict(
                channel_kernel=channel_kernel,
                spatial_kernel=spatial_kernel,
                num_filters=channel_num_filters,
                time_dim=time_dim,
            )
        else:
            BridgeConvCls = ConvBlock2D
            bridge_conv_kwargs = dict(time_dim=time_dim)

        # --- encoder ---
        self.stage1 = U2NetBlock2D(
            base_channels, base_channels, base_channels,
            context_dim=base_channels * 8, **block_kwargs,
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.stage2 = U2NetBlock2D(
            base_channels, base_channels * 2, base_channels * 2,
            context_dim=base_channels * 8, **block_kwargs,
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.stage3 = U2NetBlock2D(
            base_channels * 2, base_channels * 4, base_channels * 4,
            context_dim=base_channels * 8, **block_kwargs,
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        # --- bridge ---
        bridge_channels = base_channels * 8
        self.bridge = BridgeConvCls(
            base_channels * 4, bridge_channels, **bridge_conv_kwargs
        )
        self.bridge_norm = nn.LayerNorm(bridge_channels)
        self.bridge_attn = BasicTransformerBlock(
            dim=bridge_channels,
            n_heads=8,
            d_head=bridge_channels // 8,
            gated_ff=True,
            dropout=0.0,
        )

        # --- decoder ---
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear',
                               align_corners=True)
        self.stage4 = U2NetBlock2D(
            base_channels * 12, base_channels * 4, base_channels * 4,
            context_dim=base_channels * 8, **block_kwargs,
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear',
                               align_corners=True)
        self.stage5 = U2NetBlock2D(
            base_channels * 6, base_channels * 2, base_channels * 2,
            context_dim=base_channels * 8, **block_kwargs,
        )

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear',
                               align_corners=True)
        self.stage6 = U2NetBlock2D(
            base_channels * 3, base_channels, base_channels,
            context_dim=base_channels * 8, **block_kwargs,
        )

        self.final = nn.Conv2d(
            base_channels, sensor_channels, kernel_size=1
        )

    def forward(self, x_t, cond, t):
        """
        Args:
            x_t  : [B, sensor_channels, H, W]        — noisy full-res sensor
            cond : [B, sensor_channels, H/ds, W/ds]  — low-res conditioning
            t    : [B]                                — timestep
        Returns:
            [B, sensor_channels, H, W]
        """
        B, _, H, W = x_t.shape

        t_emb = self.time_embedding(t)  # [B, base_C*8]

        # Encode low-res context → [B, base_C*8, Hc, Wc]
        context_feat = self.context_encoder(cond)

        def context_for(h, w):
            # Bilinear resample to (h, w), flatten to [B, h*w, base_C*8],
            # add time embedding broadcast across the spatial axis.
            ctx = F.interpolate(
                context_feat, size=(h, w),
                mode='bilinear', align_corners=True,
            )
            ctx = ctx.view(B, self.base_channels * 8, h * w).permute(0, 2, 1)
            return ctx + t_emb[:, None, :].expand(-1, h * w, -1)

        x = self.input_proj(x_t)  # [B, base_C, H, W]

        ctx_full = context_for(H, W)
        x1 = self.stage1(x, ctx_full, t_emb)
        x = self.pool1(x1)

        ctx_half = context_for(H // 2, W // 2)
        x2 = self.stage2(x, ctx_half, t_emb)
        x = self.pool2(x2)

        ctx_quarter = context_for(H // 4, W // 4)
        x3 = self.stage3(x, ctx_quarter, t_emb)
        x = self.pool3(x3)

        # Bridge
        x = self.bridge(x, t_emb)
        ctx_eighth = context_for(H // 8, W // 8)
        xb = x.view(B, self.base_channels * 8, (H // 8) * (W // 8)) \
              .permute(0, 2, 1)
        xb = self.bridge_attn(xb, ctx_eighth)
        x = xb.permute(0, 2, 1).view(
            B, self.base_channels * 8, H // 8, W // 8
        )

        # Decoder
        x = self.up1(x)
        x = torch.cat([x, x3], dim=1)
        x = self.stage4(x, ctx_quarter, t_emb)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.stage5(x, ctx_half, t_emb)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.stage6(x, ctx_full, t_emb)

        return self.final(x)


if __name__ == '__main__':
    B, H, W = 2, 64, 64
    S = 30
    ds = 8

    for tag, flag in [('2dconv', False), ('3dconv', True)]:
        model = U2NetSensorSR(
            sensor_channels=S,
            base_channels=32,
            use_channel_3d_conv=flag,
            channel_kernel=7,
            spatial_kernel=3,
            channel_num_filters=4,
        )
        x_t = torch.randn(B, S, H, W)
        cond = torch.randn(B, S, H // ds, W // ds)
        t = torch.randint(0, 1000, (B,))
        with torch.no_grad():
            y = model(x_t, cond, t)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'[{tag}] params={n:,} out={y.shape}')
        assert y.shape == x_t.shape
    print('OK')
```

- [ ] **Step 3.2: Run the model smoke test**

```bash
cd /mnt/shared-scratch/Katehi_L/chaoyi_he/Diffu
python -m model.u2net_sensor_sr
```

Expected: two lines `[2dconv] params=... out=torch.Size([2, 30, 64, 64])` and `[3dconv] ...`, then `OK`. No error. Param counts are printed for sanity only.

- [ ] **Step 3.3: Commit**

```bash
git add model/u2net_sensor_sr.py
git commit -m "Add U2NetSensorSR diffusion denoiser for sensor super-resolution"
```

---

## Task 4: Training helpers

**Files:**
- Create: `train_eval/train_sensor_sr.py`

- [ ] **Step 4.1: Write `train_eval/train_sensor_sr.py`**

```python
"""One-epoch train / validate helpers for sensor-SR diffusion.

The FSDP entrypoint imports these. They are thin copies of the helpers
inside main_2d_fsdp.py with no semantic changes; isolating them makes
the entrypoint easier to read and the helpers reusable.
"""
import torch
import torch.distributed as dist

from misc.util import MetricLogger, SmoothedValue


def train_one_epoch(
    model, diffusion_trainer, loader, optimizer, epoch,
    local_rank, world_size, grad_clip, log_interval,
    use_bf16=True, verbose_log=True,
):
    model.train()
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch + 1}]'

    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    device = torch.device(f'cuda:{local_rank}')

    for step, (data, cond) in enumerate(
        metric_logger.log_every(loader, log_interval, header, verbose=verbose_log)
    ):
        data = data.to(device, non_blocking=True)
        cond = cond.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_bf16):
            loss, _ = diffusion_trainer.get_loss(
                model, data, cond, loss_in_fp32=use_bf16
            )

        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            model.clip_grad_norm_(grad_clip)

        optimizer.step()

        metric_logger.update(loss=loss.item(),
                              lr=optimizer.param_groups[0]['lr'])
        metric_logger.meters['loss'].update(loss.item(), n=data.shape[0])

    avg = torch.tensor(metric_logger.meters['loss'].global_avg,
                       device=device, dtype=torch.float32)
    dist.all_reduce(avg, op=dist.ReduceOp.AVG)
    return {
        'loss': avg.item(),
        'learning_rate': optimizer.param_groups[0]['lr'],
    }


@torch.no_grad()
def validate_one_epoch(model, diffusion_trainer, loader, local_rank,
                       use_bf16=True):
    model.eval()
    device = torch.device(f'cuda:{local_rank}')
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    total, n = 0.0, 0
    for data, cond in loader:
        data = data.to(device, non_blocking=True)
        cond = cond.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_bf16):
            loss, _ = diffusion_trainer.get_loss(
                model, data, cond, loss_in_fp32=use_bf16
            )
        total += loss.item()
        n += 1
    avg = torch.tensor(total / max(n, 1), device=device, dtype=torch.float32)
    dist.all_reduce(avg, op=dist.ReduceOp.AVG)
    return {'val_loss': avg.item()}
```

- [ ] **Step 4.2: Commit**

```bash
git add train_eval/train_sensor_sr.py
git commit -m "Add sensor-SR one-epoch train/validate helpers"
```

No standalone smoke test — these helpers require a live FSDP process group and are exercised by the end-to-end run in Task 6.

---

## Task 5: FSDP entrypoint

**Files:**
- Create: `main_2d_sensor_sr_fsdp.py`

- [ ] **Step 5.1: Write `main_2d_sensor_sr_fsdp.py`**

```python
"""Sensor super-resolution diffusion — FSDP multi-GPU training.

Runs a separate diffusion model (U2NetSensorSR) whose target is the
full-resolution sensor measurement and whose conditioning is a
spatially-downsampled version of the same measurement.

Launch (single node, 2 GPUs):
    torchrun --standalone --nproc_per_node=2 main_2d_sensor_sr_fsdp.py [args]
"""
import argparse
import functools
import json
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullStateDictConfig,
    FullOptimStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model.diffusion_trainer import DiffusionTrainer
from model.u2net_sensor_sr import U2NetSensorSR
from data_loader.HFD_sensor_sr_dataset import (
    HFD_SensorSR_data, sensor_sr_image_collate_fn,
)
from train_eval.train_sensor_sr import train_one_epoch, validate_one_epoch


def get_parser():
    p = argparse.ArgumentParser(
        description='Sensor Super-Resolution Diffusion — FSDP Training'
    )
    # dataset
    p.add_argument('--data_path', type=str,
                   default='dataset/HFD100 Mat dataset')
    p.add_argument('--type', type=str, default='Flower',
                   choices=['Flower', 'Leaves', 'Scenses'])
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--eval_ratio', type=float, default=0.1)
    p.add_argument('--R-n', type=int, default=1, dest='R_n')
    p.add_argument('--ds_sr', type=int, default=8,
                   help='Spatial downsample rate for sensor conditioning.')
    p.add_argument('--sr_downsample_method', type=str, default='strided',
                   choices=['strided', 'avg_pool'])
    p.add_argument('--sensor_stats_path', type=str, required=True,
                   help='JSON file produced by scripts/compute_sensor_stats.py')

    # model
    p.add_argument('--sensor_channels', type=int, default=30)
    p.add_argument('--base_channels', type=int, default=256)
    p.add_argument('--use_channel_3d_conv', action='store_true')
    p.add_argument('--channel_kernel', type=int, default=7)
    p.add_argument('--spatial_kernel', type=int, default=3)
    p.add_argument('--channel_num_filters', type=int, default=4)

    # optimisation
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--num_epochs', type=int, default=100)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--lrf', type=float, default=0.033)
    p.add_argument('--grad_clip', type=float, default=1.0)

    # training schedule
    p.add_argument('--val_every', type=int, default=100)
    p.add_argument('--save_every', type=int, default=10)
    p.add_argument('--log_interval', type=int, default=200)

    # diffusion
    p.add_argument('--loss_type', type=str, default='l1',
                   choices=['l1', 'l2'])
    p.add_argument('--noise_schedule', type=str, default='linear',
                   choices=['linear'])
    p.add_argument('--timesteps', type=int, default=1000)
    p.add_argument('--prediction_type', type=str, default='v',
                   choices=['eps', 'x0', 'v'])

    # FSDP
    p.add_argument('--sharding_strategy', type=str, default='FULL_SHARD',
                   choices=['FULL_SHARD', 'SHARD_GRAD_OP', 'NO_SHARD'])
    p.add_argument('--no_mixed_precision', action='store_true')
    p.add_argument('--cpu_offload', action='store_true')
    p.add_argument('--wrap_min_params', type=int, default=1_000_000)

    # output / resume
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--save_path', type=str, required=True)
    p.add_argument('--seed', type=int, default=42)
    return p


def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def is_main_process():
    return dist.get_rank() == 0


def print_rank0(*a, **k):
    if is_main_process():
        print(*a, **k)


def save_fsdp_checkpoint(model, optimizer, scheduler, epoch, loss, save_path):
    rank = dist.get_rank()
    model_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, model_cfg):
        model_state = model.state_dict()
    optim_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                               optim_state_dict_config=optim_cfg):
        optim_state = FSDP.optim_state_dict(model, optimizer)
    if rank == 0:
        ckpt = {
            'epoch': epoch,
            'loss': loss,
            'model_state_dict': model_state,
            'optimizer_state_dict': optim_state,
            'scheduler_state_dict':
                scheduler.state_dict() if scheduler is not None else None,
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(ckpt, save_path)
        print(f'[rank 0] Checkpoint saved → {save_path}')
    dist.barrier()


def load_checkpoint_to_cpu(resume_path):
    if resume_path and resume_path.endswith('.pth') \
            and os.path.exists(resume_path):
        print_rank0(f'Loading checkpoint from {resume_path} …')
        return torch.load(resume_path, map_location='cpu',
                          weights_only=False)
    return None


def wrap_model_with_fsdp(model, args, local_rank):
    strategy_map = {
        'FULL_SHARD': ShardingStrategy.FULL_SHARD,
        'SHARD_GRAD_OP': ShardingStrategy.SHARD_GRAD_OP,
        'NO_SHARD': ShardingStrategy.NO_SHARD,
    }
    if not args.no_mixed_precision:
        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        )
    else:
        mp = None
    policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=args.wrap_min_params
    )
    cpu_offload = CPUOffload(offload_params=True) if args.cpu_offload else None
    return FSDP(
        model,
        sharding_strategy=strategy_map[args.sharding_strategy],
        mixed_precision=mp,
        auto_wrap_policy=policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        cpu_offload=cpu_offload,
        device_id=local_rank,
        sync_module_states=True,
        use_orig_params=True,
    )


def main(args):
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    use_bf16 = not args.no_mixed_precision
    batch_per_gpu = args.batch_size // world_size
    assert batch_per_gpu > 0, \
        f'batch_size ({args.batch_size}) >= world_size ({world_size})'

    if is_main_process():
        os.makedirs(args.save_path, exist_ok=True)
        with open(os.path.join(args.save_path, 'args_sensor_sr.json'),
                  'w') as f:
            json.dump(vars(args), f, indent=2, default=str)

    # Dataset / loaders
    ds_kwargs = dict(
        data_path=args.data_path,
        stats_path=args.sensor_stats_path,
        eval_ratio=args.eval_ratio,
        type=args.type,
        R_n=args.R_n,
        sr_downsample_rate=args.ds_sr,
        sr_downsample_method=args.sr_downsample_method,
    )
    train_ds = HFD_SensorSR_data(split='train', **ds_kwargs)
    eval_ds = HFD_SensorSR_data(split='test', **ds_kwargs)

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank,
        shuffle=True, seed=args.seed,
    )
    eval_sampler = DistributedSampler(
        eval_ds, num_replicas=world_size, rank=rank,
        shuffle=False, seed=args.seed,
    )

    nw = min(args.num_workers, batch_per_gpu, 4)
    train_loader = DataLoader(
        train_ds, batch_size=batch_per_gpu, sampler=train_sampler,
        num_workers=nw, collate_fn=sensor_sr_image_collate_fn,
        pin_memory=True, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=batch_per_gpu, sampler=eval_sampler,
        num_workers=nw, collate_fn=sensor_sr_image_collate_fn,
        pin_memory=True, drop_last=False,
    )

    print_rank0(
        f'Train: {len(train_ds)}  Eval: {len(eval_ds)}  '
        f'Batch/GPU: {batch_per_gpu}  World: {world_size}  '
        f'ds_sr: {args.ds_sr} ({args.sr_downsample_method})'
    )

    # Model
    model = U2NetSensorSR(
        sensor_channels=args.sensor_channels,
        base_channels=args.base_channels,
        use_channel_3d_conv=args.use_channel_3d_conv,
        channel_kernel=args.channel_kernel,
        spatial_kernel=args.spatial_kernel,
        channel_num_filters=args.channel_num_filters,
    )

    ckpt = load_checkpoint_to_cpu(args.resume)
    start_epoch = 0
    ckpt_optim_state = None
    ckpt_sched_state = None
    if ckpt is not None:
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        start_epoch = ckpt.get('epoch', 0) + 1
        ckpt_optim_state = ckpt.get('optimizer_state_dict')
        ckpt_sched_state = ckpt.get('scheduler_state_dict')
        print_rank0(f'Resuming from epoch {start_epoch} '
                    f'(ckpt loss: {ckpt.get("loss", "?")})')
        del ckpt

    model = wrap_model_with_fsdp(model, args, local_rank)
    n_params = sum(p.numel() for p in model.parameters())
    print_rank0(
        f'\nFSDP model ready — {n_params:,} params total\n'
        f'  sharding: {args.sharding_strategy}  bf16: {use_bf16}  '
        f'cpu_offload: {args.cpu_offload}\n'
        f'  use_channel_3d_conv: {args.use_channel_3d_conv}  '
        f'base_channels: {args.base_channels}'
    )

    diffusion_trainer = DiffusionTrainer(
        n_timesteps=args.timesteps,
        loss_type=args.loss_type,
        prediction_type=args.prediction_type,
        beta_schedule=args.noise_schedule,
        device=device,
    )

    adam_eps = 1e-6 if use_bf16 else 1e-8
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, eps=adam_eps,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr * args.lrf
    )

    if ckpt_optim_state is not None:
        sharded = FSDP.optim_state_dict_to_load(
            model, optimizer, ckpt_optim_state
        )
        optimizer.load_state_dict(sharded)
        print_rank0('Optimizer state restored.')
        del ckpt_optim_state

    for g in optimizer.param_groups:
        g['eps'] = adam_eps

    if ckpt_sched_state is not None:
        scheduler.load_state_dict(ckpt_sched_state)
        print_rank0(f'Scheduler restored (last_epoch={scheduler.last_epoch}).')

    print_rank0('=' * 80)
    print_rank0(f'Starting training: epochs {start_epoch + 1} → {args.num_epochs}')
    print_rank0('=' * 80)

    train_history = []
    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.num_epochs):
        train_sampler.set_epoch(epoch)
        t0 = time.time()
        tm = train_one_epoch(
            model, diffusion_trainer, train_loader, optimizer,
            epoch, local_rank, world_size,
            grad_clip=args.grad_clip, log_interval=args.log_interval,
            use_bf16=use_bf16, verbose_log=is_main_process(),
        )
        dt = time.time() - t0
        scheduler.step()
        train_history.append(tm)
        print_rank0(
            f'Epoch {epoch+1}/{args.num_epochs}  '
            f'loss: {tm["loss"]:.6f}  lr: {tm["learning_rate"]:.2e}  '
            f'time: {dt:.1f}s'
        )

        if args.val_every > 0 and (epoch + 1) % args.val_every == 0:
            eval_sampler.set_epoch(epoch)
            vm = validate_one_epoch(
                model, diffusion_trainer, eval_loader, local_rank, use_bf16
            )
            print_rank0(f'  → val_loss: {vm["val_loss"]:.6f}')
            if is_main_process() and vm['val_loss'] < best_val_loss:
                best_val_loss = vm['val_loss']
                save_fsdp_checkpoint(
                    model, optimizer, scheduler, epoch, best_val_loss,
                    os.path.join(args.save_path, 'best_model.pth'),
                )

        if (epoch + 1) % args.save_every == 0:
            save_fsdp_checkpoint(
                model, optimizer, scheduler, epoch, tm['loss'],
                os.path.join(args.save_path,
                             f'checkpoint_epoch_{epoch+1}.pth'),
            )

    if is_main_process():
        save_fsdp_checkpoint(
            model, optimizer, scheduler, args.num_epochs - 1,
            train_history[-1]['loss'] if train_history else 0.0,
            os.path.join(args.save_path, 'final_model.pth'),
        )
        if train_history:
            header = ','.join(train_history[0].keys())
            rows = [[m[k] for k in train_history[0]] for m in train_history]
            with open(os.path.join(args.save_path, 'train_history.txt'),
                      'w') as f:
                f.write(header + '\n')
                np.savetxt(f, np.array(rows), fmt='%.6f', delimiter=',')
            with open(os.path.join(args.save_path, 'train_history.json'),
                      'w') as f:
                json.dump(train_history, f, indent=2)
        losses = [h['loss'] for h in train_history]
        print('\nTraining complete!')
        if losses:
            print(f'  Initial loss: {losses[0]:.6f}')
            print(f'  Final   loss: {losses[-1]:.6f}')
            print(f'  Best    loss: {min(losses):.6f}')

    dist.destroy_process_group()


if __name__ == '__main__':
    args = get_parser().parse_args()
    if int(os.environ.get('RANK', 0)) == 0:
        print('=' * 80)
        print('Sensor Super-Resolution Diffusion — FSDP Multi-GPU Training')
        print('=' * 80)
        for k, v in vars(args).items():
            print(f'  {k}: {v}')
        print('=' * 80)
    main(args)
```

- [ ] **Step 5.2: Verify the entrypoint imports cleanly (no launch yet)**

```bash
cd /mnt/shared-scratch/Katehi_L/chaoyi_he/Diffu
python -c "import main_2d_sensor_sr_fsdp; print('import OK')"
```

Expected: `import OK`. Any ImportError here means a path or symbol is wrong and must be fixed before launching torchrun.

- [ ] **Step 5.3: Commit**

```bash
git add main_2d_sensor_sr_fsdp.py
git commit -m "Add FSDP entrypoint for sensor super-resolution diffusion"
```

---

## Task 6: SLURM script

**Files:**
- Create: `job_sensor_sr.slurm`

- [ ] **Step 6.1: Write `job_sensor_sr.slurm`**

```bash
#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=sensor_sr_diff
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:2
#SBATCH -p gpu-research
#SBATCH --qos=olympus-research-gpu
#SBATCH -o sensor_sr_diff_%j.log
#SBATCH -e sensor_sr_diff_%j.err

cd /mnt/shared-scratch/Katehi_L/chaoyi_he/Diffu/
source /mnt/shared-scratch/Katehi_L/chaoyi_he/conda/miniconda/bin/activate
conda activate /mnt/shared-scratch/Katehi_L/chaoyi_he/conda_envs/hsi

export PYTORCH_ALLOC_CONF=expandable_segments:True

which python
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA devices:', torch.cuda.device_count())"

GPUS_PER_NODE=2
MASTER_PORT=29500

# Sensor stats file must exist before launch:
#   python -m scripts.compute_sensor_stats --R-n 1
STATS_PATH="dataset/HFD100 Mat dataset/sensor_stats_R1.json"

torchrun \
    --standalone \
    --nproc_per_node=$GPUS_PER_NODE \
    main_2d_sensor_sr_fsdp.py \
    --sensor_stats_path "$STATS_PATH" \
    --batch_size 16 \
    --lr 1e-4 \
    --base_channels 256 \
    --use_channel_3d_conv \
    --channel_num_filters 4 \
    --prediction_type v \
    --lrf 0.033 \
    --ds_sr 8 \
    --sr_downsample_method strided \
    --save_path results/2d_sensor_sr/ds8_strided/HFD/R_1/3dconv
```

- [ ] **Step 6.2: Commit**

```bash
git add job_sensor_sr.slurm
git commit -m "Add SLURM launcher for sensor-SR diffusion training"
```

---

## Task 7: End-to-end smoke run

**Files:** none — runs the code written above.

- [ ] **Step 7.1: Run a 1-epoch smoke launch on 1 GPU**

This exercises all the new code paths (dataset load, DataLoader, FSDP wrap, forward, backward, optimizer step, checkpoint save) on a tiny schedule so any wiring bug surfaces immediately.

```bash
cd /mnt/shared-scratch/Katehi_L/chaoyi_he/Diffu
torchrun --standalone --nproc_per_node=1 main_2d_sensor_sr_fsdp.py \
    --sensor_stats_path "dataset/HFD100 Mat dataset/sensor_stats_R1.json" \
    --batch_size 2 \
    --num_epochs 1 \
    --save_every 1 \
    --log_interval 5 \
    --base_channels 32 \
    --ds_sr 8 \
    --sr_downsample_method strided \
    --save_path results/2d_sensor_sr_smoketest
```

Expected: reports `Train:` / `Eval:` counts, prints `FSDP model ready — ... params total`, runs one epoch, prints `Epoch 1/1 loss: X.X ...`, saves `checkpoint_epoch_1.pth` and `final_model.pth` under `results/2d_sensor_sr_smoketest/`. No exceptions.

- [ ] **Step 7.2: Verify checkpoints exist**

```bash
ls -la results/2d_sensor_sr_smoketest/
```

Expected: `args_sensor_sr.json`, `checkpoint_epoch_1.pth`, `final_model.pth`, `train_history.json`, `train_history.txt`.

- [ ] **Step 7.3: Clean up the smoke-test artifacts**

```bash
rm -rf results/2d_sensor_sr_smoketest
```

- [ ] **Step 7.4: (Optional) Verify 3D-conv backbone loads end-to-end**

```bash
torchrun --standalone --nproc_per_node=1 main_2d_sensor_sr_fsdp.py \
    --sensor_stats_path "dataset/HFD100 Mat dataset/sensor_stats_R1.json" \
    --batch_size 2 --num_epochs 1 --save_every 1 --log_interval 5 \
    --base_channels 32 --ds_sr 8 --use_channel_3d_conv \
    --save_path results/2d_sensor_sr_smoketest_3d
```

Expected: same as 7.1 but with `use_channel_3d_conv: True` in the startup banner and a larger param count.

```bash
rm -rf results/2d_sensor_sr_smoketest_3d
```

No commit — these are verification runs, not source changes.

---

## Self-Review — spec coverage

| Spec section | Covered by |
|---|---|
| §2 Task & tensor contract | Task 2 (dataset `__getitem__` + collate produce `[B,30,H,W]` / `[B,30,H/ds,W/ds]`) |
| §3.1 New model architecture | Task 3 |
| §3.2 3D-conv switchability | Task 3 (`use_channel_3d_conv` constructor arg + smoke test with both modes); Task 7.4 end-to-end |
| §4.1 Dataset | Task 2 |
| §4.2 Stats script | Task 1 |
| §5.1 New files | Tasks 3/4/5/6 |
| §5.2 CLI changes | Task 5 `get_parser` |
| §5.3 Output layout | Task 6 SLURM `--save_path` |
| §6 Reused components | Imports in Tasks 3–5 (diffusion_trainer, layers, misc.util, FSDP helpers) |
| §7 Validation & error handling | Task 2 stats-missing FileNotFoundError + assert on divisibility; Task 5 `--sensor_stats_path` required |
| §8 Out of scope | (nothing to implement) |
