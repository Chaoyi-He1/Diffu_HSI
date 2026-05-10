"""Inference + MSE/MAE evaluation for the sensor-SR diffusion model.

Runs the trained `U2NetSensorSR` on a handful of validation samples,
computes per-sample (and aggregate) MSE / MAE / PSNR between the
reconstructed full-resolution sensor measurement and ground truth, and
writes comparison PNGs (GT / low-res cond / generated / |diff|) plus a
`metrics_sensor_sr.txt` table.

Launch (single GPU — no torchrun needed):
    python visual/visual_sensor_sr.py \
        --model_path results/2d_sensor_sr/ds8_strided/HFD/R_1/l1_loss/3dconv/checkpoint_epoch_40.pth \
        --sensor_stats_path "dataset/HFD100 Mat dataset/sensor_stats_R1.json" \
        --save_path results/2d_sensor_sr_visual/ds8_strided/HFD/R_1/l1_loss/3dconv \
        --num_examples 4 --num_steps 250 --method ddim
"""
import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader.HFD_sensor_sr_dataset import (
    HFD_SensorSR_data,
    sensor_sr_image_collate_fn,
)
from model.diffusion_trainer import DiffusionTrainer
from model.u2net_sensor_sr import U2NetSensorSR


def get_parser():
    p = argparse.ArgumentParser(
        description='Sensor-SR Diffusion — inference + MSE/MAE eval'
    )
    # dataset (must match training)
    p.add_argument('--data_path', type=str,
                   default='dataset/HFD100 Mat dataset')
    p.add_argument('--type', type=str, default='Flower',
                   choices=['Flower', 'Leaves', 'Scenses'])
    p.add_argument('--R-n', type=int, default=1, dest='R_n')
    p.add_argument('--ds_sr', type=int, default=8)
    p.add_argument('--sr_downsample_method', type=str, default='strided',
                   choices=['strided', 'avg_pool'])
    p.add_argument('--sensor_stats_path', type=str, required=True)
    p.add_argument('--eval_ratio', type=float, default=0.1)
    p.add_argument('--split', type=str, default='test',
                   choices=['train', 'test'])
    p.add_argument('--num_workers', type=int, default=2)

    # model (must match training)
    p.add_argument('--sensor_channels', type=int, default=30)
    p.add_argument('--base_channels', type=int, default=256)
    p.add_argument('--use_channel_3d_conv', action='store_true')
    p.add_argument('--channel_kernel', type=int, default=7)
    p.add_argument('--spatial_kernel', type=int, default=3)
    p.add_argument('--channel_num_filters', type=int, default=4)

    # diffusion (must match training)
    p.add_argument('--loss_type', type=str, default='l1',
                   choices=['l1', 'l2'])
    p.add_argument('--noise_schedule', type=str, default='linear',
                   choices=['linear'])
    p.add_argument('--timesteps', type=int, default=1000)
    p.add_argument('--prediction_type', type=str, default='v',
                   choices=['eps', 'x0', 'v'])

    # sampling
    p.add_argument('--num_steps', type=int, default=250,
                   help='Number of denoising steps (<= --timesteps).')
    p.add_argument('--method', type=str, default='ddim',
                   choices=['ddpm', 'ddim'])
    p.add_argument('--eta', type=float, default=0.0,
                   help='DDIM stochasticity (0 = deterministic).')

    # I/O
    p.add_argument('--model_path', type=str, required=True,
                   help='Path to *.pth checkpoint saved by main_2d_sensor_sr_fsdp.py')
    p.add_argument('--save_path', type=str, required=True)
    p.add_argument('--num_examples', type=int, default=4)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--num_probe_pixels', type=int, default=2,
                   help='Random pixels per sample whose 30-ch sensor response '
                        '(GT vs Gen vs Cond) is plotted alongside the images.')

    # runtime
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=42)
    return p


def load_model(args, device):
    model = U2NetSensorSR(
        sensor_channels=args.sensor_channels,
        base_channels=args.base_channels,
        use_channel_3d_conv=args.use_channel_3d_conv,
        channel_kernel=args.channel_kernel,
        spatial_kernel=args.spatial_kernel,
        channel_num_filters=args.channel_num_filters,
    )
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f'Checkpoint not found: {args.model_path}')
    print(f'Loading checkpoint {args.model_path}')
    ckpt = torch.load(args.model_path, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'  [warn] missing keys (first 5): {missing[:5]}')
    if unexpected:
        print(f'  [warn] unexpected keys (first 5): {unexpected[:5]}')
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model loaded — {n_params:,} params '
          f'(epoch {ckpt.get("epoch", "?")}, loss {ckpt.get("loss", "?")})')
    return model


def rgb_from_sensor(img_chw, channels=(0, 15, 29)):
    """Pick 3 sensor bands and normalize to [0,1] for display."""
    rgb = np.stack([img_chw[c] for c in channels], axis=-1)
    lo, hi = float(rgb.min()), float(rgb.max())
    if hi - lo < 1e-8:
        return np.zeros_like(rgb)
    return np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)


def save_comparison(gt, low, gen, pixel_coords, ds_sr, idx, save_path):
    """One figure per sample — GT RGB / low-res cond RGB / gen RGB / |diff| /
    per-pixel 30-ch sensor response curves (GT vs Gen vs Cond).

    pixel_coords: list of (y, x) full-res pixel coordinates to probe.
    ds_sr: downsample rate used to map full-res pixel -> cond pixel.
    """
    gt_rgb = rgb_from_sensor(gt)
    low_rgb = rgb_from_sensor(low)
    gen_rgb = rgb_from_sensor(gen)

    # absolute diff averaged over all 30 bands, then normalized for display
    diff = np.mean(np.abs(gt - gen), axis=0)
    d_lo, d_hi = float(diff.min()), float(diff.max())
    diff_vis = (diff - d_lo) / (d_hi - d_lo + 1e-8)

    n_probe = len(pixel_coords)
    n_cols = 4 + n_probe
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    fig.suptitle(f'Sensor-SR example {idx}', fontsize=14)

    axes[0].imshow(gt_rgb)
    axes[0].set_title(f'GT full-res\n[{gt.shape[1]}x{gt.shape[2]}]')
    axes[1].imshow(low_rgb)
    axes[1].set_title(f'Low-res cond\n[{low.shape[1]}x{low.shape[2]}]')
    axes[2].imshow(gen_rgb)
    axes[2].set_title('Generated full-res')
    im = axes[3].imshow(diff_vis, cmap='hot')
    axes[3].set_title(f'|GT - Gen| (mean over 30 ch)\nmax={d_hi:.4f}')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    for a in axes[:4]:
        a.axis('off')

    # Overlay pixel markers on GT / Gen / Diff panels (colors match curves)
    colors = plt.cm.tab10(np.arange(max(n_probe, 1)))
    for (y, x), c in zip(pixel_coords, colors):
        for ax in (axes[0], axes[2], axes[3]):
            ax.plot(x, y, marker='o', markersize=8,
                    markeredgecolor='white', markerfacecolor=c,
                    markeredgewidth=1.5)

    # Per-pixel 30-ch response curves
    n_ch = gt.shape[0]
    ch_idx = np.arange(n_ch)
    for i, ((y, x), c) in enumerate(zip(pixel_coords, colors)):
        ax = axes[4 + i]
        gt_curve = gt[:, y, x]
        gen_curve = gen[:, y, x]
        ly, lx = min(y // ds_sr, low.shape[1] - 1), min(x // ds_sr, low.shape[2] - 1)
        low_curve = low[:, ly, lx]
        px_mse = float(np.mean((gt_curve - gen_curve) ** 2))
        ax.plot(ch_idx, gt_curve, label='GT', color='black', linewidth=2)
        ax.plot(ch_idx, gen_curve, label='Gen', color=c,
                linewidth=1.8, linestyle='--')
        ax.plot(ch_idx, low_curve, label=f'Cond ({lx},{ly})',
                color='tab:gray', linewidth=1, linestyle=':', marker='.',
                markersize=4)
        ax.set_xlabel('channel')
        ax.set_ylabel('value (normalized)')
        ax.set_title(f'Pixel ({x},{y})  MSE={px_mse:.4f}')
        ax.set_ylim(-1.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='best')

    plt.tight_layout()
    out = os.path.join(save_path, f'sensor_sr_example_{idx}.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    return out


def compute_metrics(gt, gen):
    """gt, gen: np.ndarray [30, H, W] in [-1, 1]. Peak range = 2 → MAX^2 = 4."""
    err = gt - gen
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    psnr = 10.0 * np.log10(4.0 / mse) if mse > 0 else float('inf')
    per_band_mse = np.mean(err ** 2, axis=(1, 2))     # [30]
    per_band_mae = np.mean(np.abs(err), axis=(1, 2))  # [30]
    return {
        'mse': mse,
        'mae': mae,
        'psnr': psnr,
        'mean_band_mse': float(per_band_mse.mean()),
        'mean_band_mae': float(per_band_mae.mean()),
        'max_band_mse': float(per_band_mse.max()),
        'max_band_mae': float(per_band_mae.max()),
    }


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.save_path, exist_ok=True)

    ds = HFD_SensorSR_data(
        data_path=args.data_path,
        stats_path=args.sensor_stats_path,
        split=args.split,
        eval_ratio=args.eval_ratio,
        type=args.type,
        R_n=args.R_n,
        sr_downsample_rate=args.ds_sr,
        sr_downsample_method=args.sr_downsample_method,
    )
    print(f'Dataset ({args.split}): {len(ds)} samples')

    batch_size = min(args.batch_size, args.num_examples)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=args.num_workers,
        collate_fn=sensor_sr_image_collate_fn,
    )

    model = load_model(args, device)

    diffusion_trainer = DiffusionTrainer(
        n_timesteps=args.timesteps,
        loss_type=args.loss_type,
        prediction_type=args.prediction_type,
        beta_schedule=args.noise_schedule,
        device=device,
    )

    all_metrics = []
    processed = 0
    for full, low in tqdm(loader, desc='Batches'):
        if processed >= args.num_examples:
            break
        take = min(args.num_examples - processed, full.shape[0])
        full = full[:take].to(device)
        low = low[:take].to(device)

        print(f'\nBatch {processed // batch_size + 1}: '
              f'full={tuple(full.shape)} low={tuple(low.shape)} '
              f'full∈[{full.min():.3f},{full.max():.3f}] '
              f'low∈[{low.min():.3f},{low.max():.3f}]')

        with torch.no_grad():
            gen = diffusion_trainer.sample(
                model=model,
                cond=low,
                shape=tuple(full.shape),
                n_steps=args.num_steps,
                method=args.method,
                eta=args.eta,
                progress=True,
            )

        gen = gen.clamp(-1.0, 1.0)
        full_np = full.detach().cpu().numpy()
        low_np = low.detach().cpu().numpy()
        gen_np = gen.detach().cpu().numpy()

        for i in range(take):
            m = compute_metrics(full_np[i], gen_np[i])
            all_metrics.append(m)
            ex_idx = processed + i + 1
            H, W = full_np[i].shape[1], full_np[i].shape[2]
            pixel_coords = [
                (int(np.random.randint(0, H)), int(np.random.randint(0, W)))
                for _ in range(args.num_probe_pixels)
            ]
            save_comparison(full_np[i], low_np[i], gen_np[i],
                            pixel_coords, args.ds_sr,
                            ex_idx, args.save_path)
            print(f'  example {ex_idx}: '
                  f'MSE={m["mse"]:.6f}  MAE={m["mae"]:.6f}  '
                  f'PSNR={m["psnr"]:.2f} dB  '
                  f'band_MSE_mean={m["mean_band_mse"]:.6f}')

        processed += take

    # Write metrics table
    metrics_path = os.path.join(args.save_path, 'metrics_sensor_sr.txt')
    with open(metrics_path, 'w') as f:
        cols = ['example', 'mse', 'mae', 'psnr',
                'mean_band_mse', 'mean_band_mae',
                'max_band_mse', 'max_band_mae']
        f.write('\t'.join(cols) + '\n')
        for i, m in enumerate(all_metrics):
            f.write(f'{i + 1}\t{m["mse"]:.6f}\t{m["mae"]:.6f}\t'
                    f'{m["psnr"]:.2f}\t{m["mean_band_mse"]:.6f}\t'
                    f'{m["mean_band_mae"]:.6f}\t{m["max_band_mse"]:.6f}\t'
                    f'{m["max_band_mae"]:.6f}\n')
        if all_metrics:
            avg = lambda k: np.mean([m[k] for m in all_metrics])
            f.write(
                f'AVG\t{avg("mse"):.6f}\t{avg("mae"):.6f}\t'
                f'{avg("psnr"):.2f}\t{avg("mean_band_mse"):.6f}\t'
                f'{avg("mean_band_mae"):.6f}\t{avg("max_band_mse"):.6f}\t'
                f'{avg("max_band_mae"):.6f}\n'
            )

    if all_metrics:
        mses = [m['mse'] for m in all_metrics]
        maes = [m['mae'] for m in all_metrics]
        psnrs = [m['psnr'] for m in all_metrics]
        print('\n' + '=' * 60)
        print(f'Sensor-SR inference — {len(all_metrics)} samples '
              f'({args.method.upper()}, {args.num_steps} steps)')
        print(f'  MSE  : mean={np.mean(mses):.6f}  std={np.std(mses):.6f}  '
              f'min={np.min(mses):.6f}  max={np.max(mses):.6f}')
        print(f'  MAE  : mean={np.mean(maes):.6f}  std={np.std(maes):.6f}  '
              f'min={np.min(maes):.6f}  max={np.max(maes):.6f}')
        print(f'  PSNR : mean={np.mean(psnrs):.2f} dB  '
              f'min={np.min(psnrs):.2f}  max={np.max(psnrs):.2f}')
        print('=' * 60)
        print(f'Results: {args.save_path}')
        print(f'  - per-sample PNGs: sensor_sr_example_*.png')
        print(f'  - metrics table : {metrics_path}')


if __name__ == '__main__':
    args = get_parser().parse_args()
    print('Sensor-SR visualization args:')
    for k, v in vars(args).items():
        print(f'  {k}: {v}')
    main(args)
