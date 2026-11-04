"""Phase 3 Stage 0: the closed-form MMSE baseline L (and L_demosaic in snapshot mode) on the 200 gate cubes for every
device x mode x SNR, with the normalised response matrix (all three devices) and the raw one (device1 only), the
normalised-vs-raw shape check ("normalisation does not change the conclusions") and the principal angles between the
devices' response subspaces. Records go to results/residual_warmstart/sensor/stage0/ and are never overwritten.

    python scripts/sensor/stage0_baselines.py --device cuda:0

L is closed form: --device only moves tensors. The raw variant measures with R_raw and sets sigma from the gate
cubes' own mean y^2 under R_raw, so both variants sit at the same nominal SNR.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.linalg import subspace_angles

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from data_loader.hfd_residual import limit_worker_threads  # noqa: E402
from data_loader.hfd_sensor_sim import HFDSensorSim, load_cube01  # noqa: E402
from data_loader.mmse import demosaic_then_mmse  # noqa: E402
from data_loader.sensor_config import CONFIG_PATH, SENSOR_DIR, load_config, training_sha  # noqa: E402
from data_loader.sensor_forward import sigma_from_snr  # noqa: E402
from data_loader.warmstart_paths import DATA_ROOT, PRIOR, git_head  # noqa: E402
from scripts.eval_warmstart import class_of, load_subset  # noqa: E402
from scripts.uq_eval import gaussian_metrics  # noqa: E402

STAGE0_DIR = os.path.join(SENSOR_DIR, 'stage0')
SNRS = (60.0, 50.0, 40.0, 30.0, 20.0)
DEVICES = ('device1', 'device2', 'ph5')
MODES = ('sequential', 'snapshot')
MAX_DEV = 0.10


def _extend(acc, method, d):
    a = acc.setdefault(method, {})
    for k, v in d.items():
        a.setdefault(k, []).extend(v)


@torch.no_grad()
def run_L(ds, loader, device):
    """Per-cube metrics of L (and L_demosaic when the dataset's setting is snapshot) -> dict(per_cube, files, sigma)."""
    snapshot = ds.setting['mode'] == 'snapshot'
    acc, sigma = {}, None
    for batch in loader:
        s_b = batch['sigma']
        if not bool(torch.all(s_b == s_b[0])) or (sigma is not None and float(s_b[0]) != sigma):
            raise ValueError('the cubes of one record must share one sigma (a fixed setting)')
        sigma = float(s_b[0])
        x, x0_hat, s_raw = (batch[k].float().to(device) for k in ('x', 'x0_hat', 's_raw'))
        _extend(acc, 'L', gaussian_metrics(x0_hat, s_raw, x))
        if snapshot:
            xs, ss = [], []
            for i in range(batch['y'].shape[0]):
                y = batch['y'][i].permute(1, 2, 0).numpy()
                mask = batch['mask'][i].permute(1, 2, 0).numpy()
                xh, sd = demosaic_then_mmse(y, mask, ds.R, ds.prior, sigma, ds.A, ds.config['e2_demosaic'])
                xs.append(torch.from_numpy(np.ascontiguousarray(xh.transpose(2, 0, 1))))
                ss.append(torch.from_numpy(np.ascontiguousarray(sd.transpose(2, 0, 1))))
            _extend(acc, 'L_demosaic', gaussian_metrics(torch.stack(xs).float().to(device),
                                                        torch.stack(ss).float().to(device), x))
    return dict(per_cube=acc, files=[os.path.relpath(f, ds.data_root) for f in ds.img_list], sigma=sigma)


def raw_P_y(R_raw, files, cubes=None):
    """Mean y^2 of the gate cubes measured sequentially with R_raw (the quantity P_y is for the normalised R)."""
    cubes = [load_cube01(f) for f in files] if cubes is None else cubes
    return float(np.mean([np.mean((c @ R_raw) ** 2) for c in cubes]))


def raw_sigma(R_raw, files, snr_db, P_y=None, cubes=None):
    """sigma of the raw variant at snr_db: sqrt(P_y_raw / 10^(snr/10)); pass P_y to reuse it across SNRs."""
    return sigma_from_snr(snr_db, raw_P_y(R_raw, files, cubes) if P_y is None else P_y)


def shape_check(curve_norm, curve_raw):
    """Both curves {snr: rmse} must fall (weakly) as the SNR rises and raw/norm must be flat across the SNRs."""
    snrs = sorted(curve_norm)
    if sorted(curve_raw) != snrs:
        raise ValueError('the two curves must cover the same SNRs')

    def mono(c):                       # ascending SNR -> non-increasing RMSE
        v = [c[s] for s in snrs]
        return bool(all(b <= a + 1e-12 for a, b in zip(v, v[1:])))
    ratios = np.array([curve_raw[s] / curve_norm[s] for s in snrs])
    max_dev = float(np.max(np.abs(ratios - np.median(ratios))))
    mn, mr = mono(curve_norm), mono(curve_raw)
    return dict(monotone_norm=mn, monotone_raw=mr, max_dev=max_dev, ratios=ratios.tolist(),
                passed=bool(mn and mr and max_dev <= MAX_DEV))


def principal_angles(R_a, R_b):
    """Principal angles (degrees) between the column spaces of two response matrices."""
    return np.degrees(subspace_angles(np.asarray(R_a, np.float64), np.asarray(R_b, np.float64))).tolist()


def write_json(path, obj):
    if os.path.exists(path):
        raise FileExistsError(f'{path} exists; Stage 0 outputs are never overwritten')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def out_path(prefix, device, mode, snr):
    return os.path.join(STAGE0_DIR, f'{prefix}__{device}__{mode}__{int(snr)}.json')


def means_of(per_cube):
    return {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in per_cube.items()}


def make_loader(ds, args):
    return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                       collate_fn=ds.collate_fn, worker_init_fn=limit_worker_threads)


def gate_dataset(args, cfg, files, device, R_override=None):
    ds = HFDSensorSim(args.data_root, 'test', cfg, device=device, setting=dict(mode='sequential', snr_db=60.0),
                      prior_path=args.prior, R_override=R_override)
    have = {os.path.abspath(f) for f in ds.img_list}
    missing = [f for f in files if os.path.abspath(f) not in have]
    if missing:
        raise ValueError(f'{len(missing)} of {len(files)} gate files are not in the test split, e.g. {missing[0]}')
    ds.img_list = list(files)
    return ds


def record(prefix, ds, res, device, mode, snr, cfg, args, t0, extra=None):
    st = dict(sensor_device=device, mode=mode, snr_db=float(snr), sigma=res['sigma'], R_sha256=ds.R_sha256,
              seed_scheme=ds.checkpoint_extra()['seed_scheme'], R_variant='raw' if prefix == 'Lraw' else 'normalised')
    st.update(extra or {})
    return dict(name=f'{prefix}__{device}__{mode}__{int(snr)}', setting=st, n=len(ds), files=res['files'],
                classes=[class_of(f) for f in ds.img_list], per_cube=res['per_cube'], means=means_of(res['per_cube']),
                provenance=dict(sensor_config_sha=training_sha(cfg), config_path=os.path.abspath(args.sensor_config),
                                git_head=git_head(), created=time.strftime('%Y-%m-%dT%H:%M:%S'),
                                seconds=time.time() - t0, device=str(args.device), limit_files=int(args.limit_files)))


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default=DATA_ROOT); ap.add_argument('--prior', default=PRIOR)
    ap.add_argument('--sensor_config', default=CONFIG_PATH)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--limit_files', type=int, default=0)
    ap.add_argument('--batch_size', type=int, default=8); ap.add_argument('--num_workers', type=int, default=4)
    return ap


def main(argv=None):
    args = get_parser().parse_args(argv)
    cfg = load_config(args.sensor_config)
    device = torch.device(args.device)
    files = load_subset(args.data_root, 'files')
    if args.limit_files:
        files = files[:args.limit_files]
    summary_path = os.path.join(STAGE0_DIR, 'summary.json')
    if os.path.exists(summary_path):
        raise FileExistsError(f'{summary_path} exists; Stage 0 outputs are never overwritten')
    curves = {d: {m: {} for m in MODES} for d in DEVICES}
    curves_raw = {m: {} for m in MODES}
    R_norm, P_y_raw, raw_cfg = {}, None, None
    for dev in DEVICES:
        ds = gate_dataset(args, cfg, files, dev)
        R_norm[dev] = ds.R
        if dev == 'device1':
            P_y_raw = raw_P_y(ds.sensor['R_raw'], files)
            raw_cfg = copy.copy(cfg); raw_cfg['P_y'] = P_y_raw
            ds_raw = gate_dataset(args, raw_cfg, files, dev, R_override=ds.sensor['R_raw'])
        for mode in MODES:
            for snr in SNRS:
                t0 = time.time()
                ds.setting = dict(mode=mode, snr_db=snr)
                res = run_L(ds, make_loader(ds, args), device)
                rec = record('L', ds, res, dev, mode, snr, cfg, args, t0)
                write_json(out_path('L', dev, mode, snr), rec)
                curves[dev][mode][snr] = rec['means']['L']['mean_rmse_pct']
                print(f'L {dev} {mode} {snr:.0f} dB: rmse {curves[dev][mode][snr]:.4f} % ({time.time() - t0:.0f} s)', flush=True)
                if dev == 'device1':
                    t0 = time.time()
                    ds_raw.setting = dict(mode=mode, snr_db=snr)
                    res = run_L(ds_raw, make_loader(ds_raw, args), device)
                    rec = record('Lraw', ds_raw, res, dev, mode, snr, cfg, args, t0, extra=dict(P_y_raw=P_y_raw))
                    write_json(out_path('Lraw', dev, mode, snr), rec)
                    curves_raw[mode][snr] = rec['means']['L']['mean_rmse_pct']
                    print(f'Lraw {dev} {mode} {snr:.0f} dB: rmse {curves_raw[mode][snr]:.4f} %', flush=True)
    checks = {m: shape_check(curves['device1'][m], curves_raw[m]) for m in MODES}
    angles = {f'device1_{o}': principal_angles(R_norm['device1'], R_norm[o]) for o in ('device2', 'ph5')}
    overall = all(c['passed'] for c in checks.values())
    summary = dict(snrs=list(SNRS), files_n=len(files), curves_normalised_device1={m: {str(k): v for k, v in curves['device1'][m].items()} for m in MODES},
                   curves_raw_device1={m: {str(k): v for k, v in curves_raw[m].items()} for m in MODES},
                   shape_check=checks, normalisation_check_passed=bool(overall), P_y_raw=P_y_raw, P_y=cfg['P_y'],
                   principal_angles_deg=angles,
                   L_curves={d: {m: {str(k): v for k, v in curves[d][m].items()} for m in MODES} for d in DEVICES},
                   sanity_reference_rmse_pct=curves['device1']['sequential'][60.0],
                   provenance=dict(sensor_config_sha=training_sha(cfg), git_head=git_head(),
                                   created=time.strftime('%Y-%m-%dT%H:%M:%S'), limit_files=int(args.limit_files)))
    write_json(summary_path, summary)
    for m in MODES:
        print(f"NORMALISATION CHECK [{m}]: {'PASS' if checks[m]['passed'] else 'FAIL'} (max_dev={checks[m]['max_dev']:.4f})")
    print(f"NORMALISATION CHECK: {'PASS' if overall else 'FAIL'} (max_dev={max(c['max_dev'] for c in checks.values()):.4f})")
    print('wrote', summary_path)


if __name__ == '__main__':
    main()
