"""Phase 3 evaluation: one record per (sensor device, measurement mode, SNR, subset) scoring, per cube, the classical
MMSE baseline L (and L_demosaic in snapshot mode), the family regressor D, the variance head V, the family-conditioned
diffusion C (K draws) and its one-call point C1, and the guided unconditional prior B (K ΠGDM / DPS draws). Everything
runs in float32 without autocast (the guidance operator is cast to y's dtype). Records go to
results/residual_warmstart/sensor/eval/<name>.json and are never overwritten.

    python scripts/sensor/sd_eval.py --name dev1_seq_30 --sensor_device device1 --mode sequential --snr_db 30 \
        --regress <ckpt> --varhead <ckpt> --cond <ckpt> --uncond <ckpt> --K 32 --pigdm_steps 50 --device cuda:0

--methods defaults to every applicable method: L always, L_demosaic in snapshot mode, D / V / C and C1 / B when
--regress / --varhead / --cond / --uncond is given. A requested method without its checkpoint is refused; a checkpoint
whose method is not requested is ignored. Sample coverage is exact only for K >= min_k_for(0.9) (coverage_exact).
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from data_loader.hfd_sensor_sim import HFDSensorSim  # noqa: E402
from data_loader.mmse import demosaic_then_mmse  # noqa: E402
from data_loader.sensor_config import CONFIG_PATH, SENSOR_DIR, load_config, training_sha  # noqa: E402
from data_loader.sensor_forward import guidance_operator  # noqa: E402
from data_loader.warmstart_paths import DATA_ROOT, PRIOR, file_sha256, git_head  # noqa: E402
from model.diffusion_trainer import DiffusionTrainer  # noqa: E402
from model.guidance import LinearMeasurement, draw_guided  # noqa: E402
from model.residual_wrapper import LOGVAR_RANGE, build_residual_model, compose, regress_forward, varhead_forward  # noqa: E402
from data_loader.hfd_residual import limit_worker_threads  # noqa: E402
from scripts.eval_warmstart import class_of, load_subset  # noqa: E402
from scripts.uq.metrics import min_k_for  # noqa: E402
from scripts.uq_eval import gaussian_metrics, point_metrics, sample_metrics  # noqa: E402

METHODS = ('L', 'L_demosaic', 'D', 'V', 'C', 'C1', 'B')
EVAL_DIR = os.path.join(SENSOR_DIR, 'eval')
CKPT_FLAG = {'D': 'regress', 'V': 'varhead', 'C': 'cond', 'C1': 'cond', 'B': 'uncond'}
FLAG_MODE = {'regress': 'regress_fam', 'varhead': 'varhead_fam', 'cond': 'cond_fam', 'uncond': 'uncond'}
FLAG_ALLOWED_MODES = {**{f: (m,) for f, m in FLAG_MODE.items()}, 'uncond': ('uncond', 'uncond_tt')}   # B: either prior
SAMPLED_FLAGS = ('cond', 'uncond')              # sampled on N_TIMESTEPS steps
N_TIMESTEPS = 1000
C_CLAMP = (-8.0, 8.0)                           # the standardised residual's x0 estimate in C's sampler


# ----------------------------------------------------------------------------- methods and checkpoints
def resolve_methods(args):
    """args.methods if given (validated), else every applicable method."""
    snapshot = args.mode == 'snapshot'
    if args.methods:
        bad = [m for m in args.methods if m not in METHODS]
        if bad:
            raise ValueError(f'unknown methods {bad}; expected a subset of {METHODS}')
        if 'L_demosaic' in args.methods and not snapshot:
            raise ValueError('L_demosaic needs --mode snapshot (there is no mosaic to demosaic in sequential mode)')
        return [m for m in METHODS if m in args.methods]
    return [m for m in METHODS if m == 'L' or (m == 'L_demosaic' and snapshot)
            or (m in CKPT_FLAG and getattr(args, CKPT_FLAG[m], None))]


def ckpt_epoch(path):
    m = re.search(r'checkpoint_epoch_(\d+)\.pth$', path)
    return int(m.group(1)) if m else None


def check_models(args, cfg, device):
    """Load the checkpoint of every requested method (C and C1 share --cond) and refuse (ValueError, naming the file)
    one whose meta mode is not the method's or whose sensor_config_sha is not training_sha(cfg); the sampled models
    (cond must be v-prediction, uncond v or x0). Returns (models keyed D / V / C / B, provenance keyed by the flag)."""
    sha = training_sha(cfg)
    models, prov = {}, {}
    for flag in dict.fromkeys(CKPT_FLAG[m] for m in resolve_methods(args) if m in CKPT_FLAG):
        path = getattr(args, flag, None)
        if not path:
            need = [m for m, f in CKPT_FLAG.items() if f == flag]
            raise ValueError(f'methods {need} need --{flag} (a {FLAG_MODE[flag]} checkpoint)')
        ck = torch.load(path, map_location='cpu', weights_only=False)
        meta = (ck.get('meta') if isinstance(ck, dict) else None) or {}
        bad = []
        if meta.get('mode') not in FLAG_ALLOWED_MODES[flag]:
            want = ' or '.join(repr(m) for m in FLAG_ALLOWED_MODES[flag])
            bad.append(f"mode: checkpoint {meta.get('mode', '<missing>')!r} vs required {want}")
        if meta.get('sensor_config_sha') != sha:
            bad.append(f"sensor_config_sha: checkpoint {meta.get('sensor_config_sha', '<missing>')!r} vs config {sha!r}")
        allowed_pt = {'cond': ('v',), 'uncond': ('v', 'x0')}.get(flag)   # B's prior may be v- or x0-prediction
        if allowed_pt and meta.get('prediction_type', 'v') not in allowed_pt:
            want = ' or '.join(repr(a) for a in allowed_pt)
            bad.append(f"prediction_type: checkpoint {meta.get('prediction_type')!r} vs required {want}")
        if bad:
            raise ValueError(f'--{flag} checkpoint {path} does not fit this evaluation: ' + '; '.join(bad))
        model = build_residual_model(meta['mode'], int(meta['base_channels']))
        model.load_state_dict(ck['model_state_dict'], strict=True)
        model = model.float().to(device).eval()
        for m, f in CKPT_FLAG.items():
            if f == flag and m != 'C1':
                models[m] = model
        prov[flag] = dict(path=os.path.abspath(path), epoch=ckpt_epoch(path), sha256=file_sha256(path),
                          mode=meta.get('mode'), git_head=meta.get('git_head'))
        if flag == 'uncond':
            prov[flag]['prediction_type'] = meta.get('prediction_type', 'v')
    return models, prov


# ----------------------------------------------------------------------------- the run
def _extend(acc, method, d):
    a = acc.setdefault(method, {})
    for k, v in d.items():
        a.setdefault(k, []).extend(v)


def _demosaic_baseline(batch, ds, sigma, e2):
    """demosaic_then_mmse per cube on the CPU -> (x_hat, std), [B, 64, H, W] float32 on the trainer scale."""
    xs, ss = [], []
    for i in range(batch['y'].shape[0]):
        y = batch['y'][i].permute(1, 2, 0).numpy()
        mask = batch['mask'][i].permute(1, 2, 0).numpy()
        xh, sd = demosaic_then_mmse(y, mask, ds.R, ds.prior, sigma, ds.A, e2)
        xs.append(torch.from_numpy(np.ascontiguousarray(xh.transpose(2, 0, 1))))
        ss.append(torch.from_numpy(np.ascontiguousarray(sd.transpose(2, 0, 1))))
    return torch.stack(xs).float(), torch.stack(ss).float()


@torch.no_grad()
def run_eval(models, ds, loader, args, cfg, provenance, device):
    methods = resolve_methods(args)
    K, snapshot = int(args.K), args.mode == 'snapshot'
    trainer = DiffusionTrainer(device=device, n_timesteps=N_TIMESTEPS, loss_type='l1', prediction_type='v', snr_gamma=None)
    b_pt = (provenance.get('uncond') or {}).get('prediction_type', 'v')
    if 'B' in methods:
        trainer_B = DiffusionTrainer(device=device, n_timesteps=N_TIMESTEPS, loss_type='l1', prediction_type=b_pt,
                                     snr_gamma=None)               # B's prior may be x0-prediction; C keeps the 'v' trainer
        Hn, bn = guidance_operator(ds.R, ds.A)
        H_op, b_op = torch.from_numpy(np.asarray(Hn)), torch.from_numpy(np.asarray(bn))
    acc, sigma, t0 = {}, None, time.time()
    for bi, batch in enumerate(loader):
        s_b = batch['sigma']
        if not bool(torch.all(s_b == s_b[0])) or (sigma is not None and float(s_b[0]) != sigma):
            raise ValueError('the cubes of one record must share one sigma (a fixed setting)')
        sigma = float(s_b[0])
        x, x0_hat = batch['x'].float().to(device), batch['x0_hat'].float().to(device)
        s_post, s_raw = batch['s_post'].float().to(device), batch['s_raw'].float().to(device)
        cond = {'sensor': None, 'x0_hat': x0_hat, 's_post': s_post}
        shape = tuple(x.shape)
        if 'L' in methods:
            _extend(acc, 'L', gaussian_metrics(x0_hat, s_raw, x))
        if 'L_demosaic' in methods:
            xd, sd = _demosaic_baseline(batch, ds, sigma, cfg['e2_demosaic'])
            _extend(acc, 'L_demosaic', gaussian_metrics(xd.to(device), sd.to(device), x))
        if 'D' in methods:
            _extend(acc, 'D', point_metrics(compose(x0_hat, s_post, regress_forward(models['D'], cond)), x))
        if 'V' in methods:
            logvar = varhead_forward(models['V'], cond).clamp(*LOGVAR_RANGE)
            _extend(acc, 'V', gaussian_metrics(x0_hat, s_post * torch.exp(0.5 * logvar), x))
        if 'C' in methods:
            draws = []
            for k in range(K):
                torch.manual_seed(k)
                r = trainer.sample(models['C'], cond, shape, n_steps=10, method='ddim', progress=False, clamp_x0=C_CLAMP)
                draws.append(compose(x0_hat, s_post, r))
            _extend(acc, 'C', sample_metrics(torch.stack(draws, 0), x))
        if 'C1' in methods:
            torch.manual_seed(0)
            r1 = trainer.sample(models['C'], cond, shape, n_steps=1, method='ddim', progress=False, clamp_x0=C_CLAMP)
            _extend(acc, 'C1', point_metrics(compose(x0_hat, s_post, r1), x))
        if 'B' in methods:
            y = batch['y'].float().to(device)
            mask = batch['mask'].float().to(device) if snapshot else None
            meas = LinearMeasurement(H_op, b_op, y, mask, sigma)
            _extend(acc, 'B', sample_metrics(draw_guided(models['B'], trainer_B, meas, shape, K, int(args.pigdm_steps),
                                                         args.guidance), x))
        print(f'batch {bi + 1}/{len(loader)} ({time.time() - t0:.0f} s)', flush=True)
    means = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in acc.items()}
    return dict(name=args.name,
                setting=dict(sensor_device=args.sensor_device, mode=args.mode, snr_db=float(args.snr_db), sigma=sigma,
                             R_sha256=ds.sensor['sha256'], seed_scheme=ds.checkpoint_extra()['seed_scheme']),
                subset=args.subset, K=K, pigdm_steps=int(args.pigdm_steps), guidance=args.guidance, n=len(ds),
                files=[os.path.relpath(f, args.data_root) for f in ds.img_list], classes=[class_of(f) for f in ds.img_list],
                per_cube=acc, means=means, coverage_exact=bool(K >= min_k_for(0.9)),
                provenance=dict(sensor_config_sha=training_sha(cfg), config_path=os.path.abspath(args.sensor_config),
                                checkpoints=provenance, b_prediction_type=b_pt if 'B' in methods else None, git_head=git_head(), created=time.strftime('%Y-%m-%dT%H:%M:%S'),
                                seconds=time.time() - t0, seeds=list(range(K)), device=str(device),
                                batch_size=int(args.batch_size), limit_files=int(args.limit_files)))


def output_path(name):
    return os.path.join(EVAL_DIR, name + '.json')


def write_record(path, record):
    if os.path.exists(path):
        raise FileExistsError(f'{path} exists; records are never overwritten (pick another --name)')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(record, f)
    os.replace(tmp, path)


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--sensor_device', choices=['device1', 'device2', 'ph5'], default='device1')
    ap.add_argument('--mode', choices=['sequential', 'snapshot'], default='sequential')
    ap.add_argument('--snr_db', type=float, required=True)
    ap.add_argument('--subset', choices=['gate', 'ddpm'], default='gate')
    ap.add_argument('--methods', nargs='+', choices=METHODS, default=None)
    ap.add_argument('--regress', default=None); ap.add_argument('--varhead', default=None)
    ap.add_argument('--cond', default=None); ap.add_argument('--uncond', default=None)
    ap.add_argument('--K', type=int, default=32); ap.add_argument('--pigdm_steps', type=int, default=50)
    ap.add_argument('--guidance', choices=['pigdm', 'dps'], default='pigdm')
    ap.add_argument('--sensor_config', default=CONFIG_PATH)
    ap.add_argument('--batch_size', type=int, default=4); ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--limit_files', type=int, default=0)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--data_root', default=DATA_ROOT); ap.add_argument('--prior', default=PRIOR)
    return ap


def main():
    args = get_parser().parse_args()
    out = output_path(args.name)
    if os.path.exists(out):
        raise FileExistsError(f'{out} exists; pick another --name')
    cfg = load_config(args.sensor_config)
    args.methods = resolve_methods(args)
    device = torch.device(args.device)
    models, prov = check_models(args, cfg, device)
    ds = HFDSensorSim(args.data_root, 'test', cfg, device=args.sensor_device,
                      setting=dict(mode=args.mode, snr_db=args.snr_db), prior_path=args.prior)
    files = load_subset(args.data_root, 'files' if args.subset == 'gate' else 'ddpm1000_subset')
    have = {os.path.abspath(f) for f in ds.img_list}
    missing = [f for f in files if os.path.abspath(f) not in have]
    if missing:
        raise ValueError(f'{len(missing)} of {len(files)} subset files are not in the test split, e.g. {missing[0]}')
    if args.limit_files:
        files = files[:args.limit_files]
    ds.img_list = list(files)
    # make_eval_loader hard-codes residual_collate_fn: the same loader with the dataset's dict collate_fn
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                         collate_fn=ds.collate_fn, worker_init_fn=limit_worker_threads)
    rec = run_eval(models, ds, loader, args, cfg, prov, device)
    write_record(out, rec)
    print('wrote', out)
    for m, d in rec['means'].items():
        print(f'  {m:10s} ' + ' '.join(f'{k}={v:.4f}' for k, v in d.items()
                                       if k in ('rmse_pct', 'mean_rmse_pct', 'crps', 'cov90', 'ause')))


if __name__ == '__main__':
    main()
