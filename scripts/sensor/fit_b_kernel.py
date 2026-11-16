"""Fit the B+ kernel: the error covariance of the guided prior's output, measured on held-out calibration cubes.

Calibration cubes are the gate cubes that are NOT in B's test subset (`ddpm1000_subset`) and whose source image is not
one of the test subset's source images, one cube per source image (the first by sorted path). On each cube the guided
prior draws K cubes at the reference setting (device1, sequential, --snr_db; ΠGDM, --pigdm_steps); the kernel is the
[64, 64] covariance of the per-pixel spectral residual truth - mean(draws), pooled over pixels and cubes
(model.posterior_draw.pooled_error_covariance). The file is written once to results/residual_warmstart/sensor/ and is
what sd_eval.py --b_kernel reads; no test cube is ever touched.

    python scripts/sensor/fit_b_kernel.py --uncond <ckpt> --device cuda:0
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from data_loader.hfd_residual import limit_worker_threads  # noqa: E402
from data_loader.hfd_sensor_sim import HFDSensorSim  # noqa: E402
from data_loader.sensor_config import CONFIG_PATH, SENSOR_DIR, load_config, training_sha  # noqa: E402
from data_loader.sensor_forward import guidance_operator  # noqa: E402
from data_loader.warmstart_paths import DATA_ROOT, PRIOR, SUBSET, git_head  # noqa: E402
from model.diffusion_trainer import DiffusionTrainer  # noqa: E402
from model.guidance import LinearMeasurement, draw_guided  # noqa: E402
from model.posterior_draw import kernel_sha256, pooled_error_covariance  # noqa: E402
from scripts.paired_stats import source_image  # noqa: E402
from scripts.sensor.sd_eval import N_TIMESTEPS, check_models  # noqa: E402

REF_DEVICE, REF_MODE = 'device1', 'sequential'


def calibration_files(subset_path=SUBSET, n=0):
    """Relative paths of the calibration cubes: gate minus the test subset and minus every cube sharing a source image
    with it, one per source image (sorted); n > 0 keeps the first n."""
    with open(subset_path) as f:
        sub = json.load(f)
    test = set(sub['ddpm1000_subset'])
    test_images = {source_image(f) for f in test}
    seen, out = set(), []
    for f in sorted(sub['files']):
        img = source_image(f)
        if f in test or img in test_images or img in seen:
            continue
        seen.add(img); out.append(f)
    if not out:
        raise ValueError('no calibration cube: every gate cube shares a source image with the test subset')
    return out[:n] if n else out


def get_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--uncond', required=True, help="B's prior checkpoint")
    ap.add_argument('--out', default=os.path.join(SENSOR_DIR, 'b_kernel.npz'))
    ap.add_argument('--snr_db', type=float, default=20.0)
    ap.add_argument('--K', type=int, default=8); ap.add_argument('--pigdm_steps', type=int, default=50)
    ap.add_argument('--n_cubes', type=int, default=0, help='0 = every calibration cube')
    ap.add_argument('--sensor_config', default=CONFIG_PATH)
    ap.add_argument('--batch_size', type=int, default=4); ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--data_root', default=DATA_ROOT); ap.add_argument('--prior', default=PRIOR)
    ap.add_argument('--subset', default=SUBSET)
    return ap


@torch.no_grad()
def fit(model, trainer, ds, loader, args, device):
    """-> (Sigma_e [64, 64] float64, per-cube RMSE % of the ensemble mean)."""
    Hn, bn = guidance_operator(ds.R, ds.A)
    H_op, b_op = torch.from_numpy(np.asarray(Hn)), torch.from_numpy(np.asarray(bn))
    resid, rmse = [], []
    for bi, batch in enumerate(loader):
        x = batch['x'].float().to(device)
        meas = LinearMeasurement(H_op, b_op, batch['y'].float().to(device), None, float(batch['sigma'][0]))
        mean = draw_guided(model, trainer, meas, tuple(x.shape), int(args.K), int(args.pigdm_steps), 'pigdm').mean(0)
        r = (x - mean).double().cpu().numpy()
        resid.append(r); rmse.extend((50.0 * np.sqrt((r ** 2).mean((1, 2, 3)))).tolist())
        print(f'batch {bi + 1}/{len(loader)} mean RMSE so far {np.mean(rmse):.3f} %', flush=True)
    return pooled_error_covariance(np.concatenate(resid)), rmse


def main(argv=None):
    args = get_parser().parse_args(argv)
    if os.path.exists(args.out):
        raise FileExistsError(f'{args.out} exists; the kernel is fitted once (pick another --out)')
    cfg = load_config(args.sensor_config)
    device = torch.device(args.device)
    ns = argparse.Namespace(methods=['B'], mode=REF_MODE, uncond=args.uncond, regress=None, varhead=None, cond=None)
    models, prov = check_models(ns, cfg, device)
    trainer = DiffusionTrainer(device=device, n_timesteps=N_TIMESTEPS, loss_type='l1',
                               prediction_type=prov['uncond']['prediction_type'], snr_gamma=None)
    files = calibration_files(args.subset, args.n_cubes)
    ds = HFDSensorSim(args.data_root, 'test', cfg, device=REF_DEVICE, setting=dict(mode=REF_MODE, snr_db=args.snr_db),
                      prior_path=args.prior)
    have = {os.path.abspath(f) for f in ds.img_list}
    paths = [os.path.join(args.data_root, f) for f in files]
    missing = [f for f in paths if os.path.abspath(f) not in have]
    if missing:
        raise ValueError(f'{len(missing)} calibration files are not in the test split, e.g. {missing[0]}')
    ds.img_list = paths
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                         collate_fn=ds.collate_fn, worker_init_fn=limit_worker_threads)
    t0 = time.time()
    Sigma_e, rmse = fit(models['B'], trainer, ds, loader, args, device)
    meta = dict(files=files, n_cubes=len(files), setting=dict(sensor_device=REF_DEVICE, mode=REF_MODE, snr_db=float(args.snr_db)),
                K=int(args.K), pigdm_steps=int(args.pigdm_steps), residual='truth - mean of K guided draws',
                uncond_path=os.path.abspath(args.uncond), uncond_sha256=prov['uncond']['sha256'],
                sensor_config_sha=training_sha(cfg), R_sha256=ds.sensor['sha256'], git_head=git_head(),
                created=time.strftime('%Y-%m-%dT%H:%M:%S'), seconds=time.time() - t0,
                mean_rmse_pct=float(np.mean(rmse)), kernel_rms_pct=float(50.0 * np.sqrt(np.trace(Sigma_e) / Sigma_e.shape[0])),
                sha256=kernel_sha256(Sigma_e))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + '.tmp.npz'
    np.savez(tmp, Sigma_e=Sigma_e, meta=np.array(meta, dtype=object))
    os.replace(tmp, args.out)
    print(f"wrote {args.out}: {len(files)} cubes, kernel rms {meta['kernel_rms_pct']:.3f} % of the range, sha {meta['sha256'][:12]}")
    return Sigma_e, meta


if __name__ == '__main__':
    main()
