"""Evaluation harness for the residual warm-start project (spec §7).

  python scripts/eval_warmstart.py --d 4 --baseline
  python scripts/eval_warmstart.py --d 4 --ckpt results/residual_warmstart/residual/d4/checkpoint_epoch_200.pth \
         --mode residual --base_channels 256 [--seeds 0 1] [--device cuda:0]

Metrics are on the [0, 1] display scale of (x + 1) / 2: RMSE in % of range, PSNR with peak 1, SAM in degrees.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data_loader.hfd_residual import HFDResidualData, residual_collate_fn  # noqa: E402
from model.diffusion_trainer import DiffusionTrainer  # noqa: E402
from model.residual_wrapper import ResidualWrapper  # noqa: E402
from model.u2net_hyperspectral import U2NetHyperspectral  # noqa: E402

DATA_ROOT = os.environ.get('HFD_DATA_ROOT', '/data/chaoyi_he/HSI/Diffu/dataset/HFD100 Mat dataset')
PRIOR = os.path.join(REPO, 'results', 'residual_warmstart', 'linear_prior_R1.npz')
SUBSET = os.path.join(REPO, 'tests', 'data', 'val_subset_200.json')
EVAL_DIR = os.path.join(REPO, 'results', 'residual_warmstart', 'eval')


# ----------------------------------------------------------------------------- metrics
def metrics(x_hat, x):
    """x_hat, x: [B, C, H, W] on [-1, 1]. Returns per-item tensors on the [0, 1] scale."""
    a, b = (x_hat.float() + 1) / 2, (x.float() + 1) / 2
    err2 = ((a - b) ** 2).flatten(1).mean(1)
    rmse_pct = err2.sqrt() * 100
    psnr = -10 * torch.log10(err2.clamp(min=1e-12))
    a64, b64 = a.double(), b.double()                    # float32 cos rounds to 1 - k*6e-8: ~0.005 deg floor on identical cubes
    cos = (a64 * b64).sum(1) / (a64.norm(dim=1) * b64.norm(dim=1) + 1e-12)   # per pixel over bands
    sam_deg = torch.rad2deg(torch.arccos(cos.clamp(-1, 1))).flatten(1).mean(1).float()
    return dict(rmse_pct=rmse_pct, psnr=psnr, sam_deg=sam_deg)


def summarize(per_item, classes):
    """per_item: dict metric -> np.array [N]; classes: list of class ids per item."""
    out = {}
    for k, v in per_item.items():
        v = np.asarray(v, dtype=np.float64)
        out[k] = float(v.mean()); out[k + '_se'] = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0
        for c in sorted(set(classes)):
            sel = np.array([ci == c for ci in classes])
            out[f'{k}_{c}'] = float(v[sel].mean())
    return out


# ----------------------------------------------------------------------------- data
def load_subset(data_root, which='files'):
    sub = json.load(open(SUBSET))
    return [os.path.join(data_root, f) for f in sub[which]]


def dataset_for(data_root, prior_path, d, files=None):
    ds = HFDResidualData(data_path=data_root, split='test', sensor_down_sample_rate=d, prior_path=prior_path)
    if files is not None:
        wanted = {os.path.abspath(f) for f in files}
        ds.img_list = [f for f in ds.img_list if os.path.abspath(f) in wanted]
    return ds


def class_of(path):
    return os.path.basename(os.path.dirname(path))


# ----------------------------------------------------------------------------- baseline
def baseline_eval(data_root, prior_path, d, files=None, batch_size=16):
    ds = dataset_for(data_root, prior_path, d, files)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=residual_collate_fn)
    per = {k: [] for k in ('rmse_pct', 'psnr', 'sam_deg')}
    for x, _, xh in loader:
        m = metrics(xh, x)
        for k in per: per[k] += m[k].tolist()
    classes = [class_of(f) for f in ds.img_list]
    return dict(d=d, n=len(ds), per_item=per, files=[os.path.relpath(f, data_root) for f in ds.img_list],
                classes=classes, mean=summarize(per, classes), nfe=0)


# ----------------------------------------------------------------------------- samplers
def sampler_grid():
    g = [dict(name='ddpm1000', method='ddpm', n_steps=1000)]
    g += [dict(name=f'ddim{k}', method='ddim', n_steps=k) for k in (1, 2, 5, 10, 20)]
    g += [dict(name=f'warm{t}_ddim', method='ddim', n_steps=10, t_start=t) for t in (50, 100, 200, 400)]
    g += [dict(name=f'warm{t}_ddpm', method='ddpm', n_steps=1000, t_start=t) for t in (50, 100, 200, 400)]
    return g


def reconstruct(model, trainer, mode, batch, cfg, sigma_d, seed):
    """batch = (x, y_d, x0_hat) tensors. Returns x_hat [B, 64, H, W] clamped to [-1, 1]."""
    x, y_d, x0_hat = [b.to(trainer.device) for b in batch]
    cond = {'sensor': y_d, 'x0_hat': x0_hat}
    torch.manual_seed(seed)
    kw = dict(n_steps=cfg['n_steps'], method=cfg['method'], progress=False)
    if 't_start' in cfg:
        kw['t_start'] = cfg['t_start']
        kw['x_init'] = torch.zeros_like(x0_hat) if mode == 'residual' else x0_hat
    if mode == 'residual':
        # The residual target (x - x0_hat) / sigma_d is unit-RMS and unbounded in [-1, 1]; both x and x0_hat lie in
        # [-1, 1], so |residual| <= 2 / sigma_d is the exact scalar envelope of feasible residuals.
        kw['clamp_x0'] = (-2.0 / sigma_d, 2.0 / sigma_d)
    out = trainer.sample(model, cond, tuple(x0_hat.shape), **kw)
    if mode == 'residual':
        out = x0_hat + sigma_d * out
    return out.clamp(-1, 1)


# ----------------------------------------------------------------------------- y-swap
def _one_step_x0(model, trainer, mode, x_target, cond, t_val):
    B = x_target.shape[0]
    t = torch.full((B,), t_val, device=trainer.device, dtype=torch.long)
    torch.manual_seed(1234)
    x_t, _ = trainer.q_sample(x_target, t)
    with torch.no_grad():
        x0_pred, _ = trainer._model_to_x0_eps(model(x_t, cond, t), x_t, t)
    return x0_pred


def yswap_check(model, trainer, mode, batch, t_values=(100, 300), sigma_d=1.0):
    x, y_d, x0_hat = [b.to(trainer.device) for b in batch]
    if x.shape[0] < 2:
        raise ValueError('y-swap check needs a batch of at least 2 items')
    target = (x - x0_hat) / sigma_d if mode == 'residual' else x
    conds = {
        'true': {'sensor': y_d, 'x0_hat': x0_hat},
        'swap_full': {'sensor': y_d.roll(1, 0), 'x0_hat': x0_hat.roll(1, 0)},
        'swap_sensor_only': {'sensor': y_d.roll(1, 0), 'x0_hat': x0_hat},
        'zero': {'sensor': torch.zeros_like(y_d), 'x0_hat': torch.zeros_like(x0_hat)},
    }
    res, ok = {}, True
    for tv in t_values:
        row = {}
        for name, c in conds.items():
            pred = _one_step_x0(model, trainer, mode, target, c, tv)
            x_hat = (c['x0_hat'] + sigma_d * pred) if mode == 'residual' else pred
            row[name] = float(metrics(x_hat.clamp(-1, 1), x)['rmse_pct'].mean())
        res[f't{tv}'] = row
        ok = ok and (row['true'] <= 0.9 * row['swap_full'])
    res['passes'] = bool(ok)
    return res


# ----------------------------------------------------------------------------- checkpoint evaluation
def build_model(mode, base_channels, device):
    extra = 64 if mode == 'residual' else 0
    net = U2NetHyperspectral(spectral_channels=64, sensor_channels=30, base_channels=base_channels, extra_in_channels=extra)
    return ResidualWrapper(net, use_x0_hat=(mode == 'residual')).to(device).eval()


def checkpoint_eval(args):
    device = torch.device(args.device)
    trainer = DiffusionTrainer(device=device, prediction_type=args.prediction_type, loss_type='l1', snr_gamma=None)
    model = build_model(args.mode, args.base_channels, device)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck)
    model.load_state_dict(sd, strict=True)
    ds_all = dataset_for(args.data_root, args.prior, args.d, load_subset(args.data_root, 'files'))
    ds_ddpm = dataset_for(args.data_root, args.prior, args.d, load_subset(args.data_root, 'ddpm1000_subset'))
    sigma_d = ds_all.sigma_d
    base = baseline_eval(args.data_root, args.prior, args.d, ds_all.img_list)
    rows = []
    for cfg in sampler_grid():
        ds = ds_ddpm if cfg['name'] == 'ddpm1000' else ds_all
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=residual_collate_fn)
        per = {k: [] for k in ('rmse_pct', 'psnr', 'sam_deg')}; base_r = []; secs = 0.0; nfe = None
        for seed in args.seeds:
            for bi, batch in enumerate(loader):
                t0 = time.time()
                x_hat = reconstruct(model, trainer, args.mode, batch, cfg, sigma_d, seed * 1000 + bi)
                secs += time.time() - t0
                nfe = trainer.last_nfe
                m = metrics(x_hat, batch[0].to(device)); mb = metrics(batch[2].to(device), batch[0].to(device))
                for k in per: per[k] += m[k].tolist()
                base_r += mb['rmse_pct'].tolist()
        classes = [class_of(f) for f in ds.img_list] * len(args.seeds)
        diff = np.asarray(per['rmse_pct']) - np.asarray(base_r)
        rows.append(dict(cfg, nfe=nfe, sec_per_cube=secs / len(per['rmse_pct']), **summarize(per, classes),
                         paired_rmse_diff=float(diff.mean()), paired_rmse_se=float(diff.std(ddof=1) / math.sqrt(len(diff)))))
        print(f"{cfg['name']:>14} nfe={nfe:5d} rmse={rows[-1]['rmse_pct']:.3f}±{rows[-1]['rmse_pct_se']:.3f} "
              f"(baseline {np.mean(base_r):.3f}, paired {diff.mean():+.3f}±{rows[-1]['paired_rmse_se']:.3f}) sam={rows[-1]['sam_deg']:.2f}")
    batch = next(iter(torch.utils.data.DataLoader(ds_all, batch_size=8, shuffle=False, collate_fn=residual_collate_fn)))
    ys = yswap_check(model, trainer, args.mode, batch, sigma_d=sigma_d)
    print('y-swap:', json.dumps(ys))
    out = dict(ckpt=args.ckpt, mode=args.mode, d=args.d, sigma_d=sigma_d, seeds=args.seeds, baseline=base['mean'], rows=rows, yswap=ys)
    os.makedirs(EVAL_DIR, exist_ok=True)
    path = os.path.join(EVAL_DIR, f"{os.path.splitext(os.path.basename(args.ckpt))[0]}_{args.mode}_d{args.d}.json")
    json.dump(out, open(path, 'w'), indent=1)
    print('wrote', path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, required=True)
    ap.add_argument('--baseline', action='store_true')
    ap.add_argument('--ckpt'); ap.add_argument('--mode', choices=['residual', 'standard'])
    ap.add_argument('--base_channels', type=int, default=64)
    ap.add_argument('--prediction_type', default='v')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--data_root', default=DATA_ROOT); ap.add_argument('--prior', default=PRIOR)
    args = ap.parse_args()
    if args.baseline:
        out = baseline_eval(args.data_root, args.prior, args.d)
        os.makedirs(EVAL_DIR, exist_ok=True)
        path = os.path.join(EVAL_DIR, f'baseline_d{args.d}.json')
        json.dump(out, open(path, 'w'), indent=1)
        m = out['mean']
        print(f"baseline d={args.d} on {out['n']} files: RMSE {m['rmse_pct']:.3f}±{m['rmse_pct_se']:.3f} % "
              f"(P022 {m['rmse_pct_P022']:.3f}, P023 {m['rmse_pct_P023']:.3f}, P024 {m['rmse_pct_P024']:.3f}) "
              f"PSNR {m['psnr']:.2f} SAM {m['sam_deg']:.3f}; wrote {path}")
    else:
        assert args.ckpt and args.mode, '--ckpt and --mode are required without --baseline'
        checkpoint_eval(args)


if __name__ == '__main__':
    main()
