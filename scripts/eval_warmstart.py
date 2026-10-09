"""Evaluation harness for the residual warm-start project (spec §7).

  python scripts/eval_warmstart.py --d 4 --baseline
  python scripts/eval_warmstart.py --d 4 --ckpt results/residual_warmstart/residual/d4/checkpoint_epoch_200.pth \
         --mode residual --base_channels 256 [--seeds 0 1] [--device cuda:0]

Metrics are on the [0, 1] display scale of (x + 1) / 2: RMSE in % of range, PSNR with peak 1, SAM in degrees.
"""
import argparse
import hashlib
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


def _by_seed(v, n_seeds):
    """Flat seed-major list [n_seeds * N] -> float64 array [n_seeds, N]."""
    v = np.asarray(v, dtype=np.float64)
    if v.size % n_seeds:
        raise ValueError(f'{v.size} values cannot be split into {n_seeds} seeds of equal length')
    return v.reshape(n_seeds, -1)


def summarize_seeds(per_item, classes, n_seeds):
    """Like `summarize`, for values collected over several seeds (flat, seed-major: seed 0's cubes, then seed 1's, ...).

    The cube is the independent unit, so each cube's metric is first averaged over seeds; the mean, the standard error
    and the per-class means are then taken over the N cubes. The spread between seeds is reported separately as
    `<metric>_seed_std` (std over seeds of the per-seed means)."""
    by_seed = {k: _by_seed(v, n_seeds) for k, v in per_item.items()}
    n = next(iter(by_seed.values())).shape[1]
    if len(classes) != n:
        raise ValueError(f'{len(classes)} classes for {n} cubes')
    out = summarize({k: v.mean(0) for k, v in by_seed.items()}, classes)
    for k, v in by_seed.items():
        out[k + '_seed_std'] = float(v.mean(1).std(ddof=1)) if n_seeds > 1 else 0.0
    return out


def aggregate_row(per_item, base_rmse, classes, n_seeds):
    """summarize_seeds plus the paired RMSE difference to the baseline. base_rmse: per-cube baseline RMSE [N] (it does
    not depend on the seed). The difference is formed per cube (averaged over seeds) before its mean and SE."""
    out = summarize_seeds(per_item, classes, n_seeds)
    diff = (_by_seed(per_item['rmse_pct'], n_seeds) - np.asarray(base_rmse, dtype=np.float64)[None]).mean(0)
    out['paired_rmse_diff'] = float(diff.mean())
    out['paired_rmse_se'] = float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else 0.0
    return out


# ----------------------------------------------------------------------------- data
def file_sha256(path):
    """sha256 of a file's bytes (provenance of the prior npz)."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_subset(data_root, which='files'):
    sub = json.load(open(SUBSET))
    return [os.path.join(data_root, f) for f in sub[which]]


def dataset_for(data_root, prior_path, d, files=None):
    ds = HFDResidualData(data_path=data_root, split='test', sensor_down_sample_rate=d, prior_path=prior_path)
    if files is not None:
        wanted = {os.path.abspath(f) for f in files}
        ds.img_list = [f for f in ds.img_list if os.path.abspath(f) in wanted]
        assert len(ds.img_list) == len(wanted), \
            f'only {len(ds.img_list)} of {len(wanted)} requested files are in the validation split'
    return ds


def subset_class_counts():
    """Files per class of the validation subset, in the JSON's key order (the file list is grouped in that order)."""
    return dict(json.load(open(SUBSET))['classes'])


def yswap_files(files, class_counts, n=8):
    """Stratified, deterministic y-swap batch: `files` is grouped in blocks of `class_counts` (class -> number of
    files, in order); from each block take max(1, round(n * block / total)) evenly spaced files (200-file subset ->
    1 x P022, 6 x P023, 1 x P024). Evenly spaced files within a class are unlikely to be patches of one source image,
    and the hard class P024 is always represented."""
    total = sum(class_counts.values())
    if total != len(files):
        raise ValueError(f'class counts add up to {total} files but {len(files)} were given')
    picked, start = [], 0
    for c, n_c in class_counts.items():
        end = start + n_c
        if any(class_of(f) != c for f in files[start:end]):
            raise ValueError(f'files[{start}:{end}] are not all of class {c}')
        k = min(n_c, max(1, round(n * n_c / total)))
        picked += [files[i] for i in np.linspace(start, end - 1, k).round().astype(int)]
        start = end
    return picked


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


def yswap_check(model, trainer, mode, batch, t_values=(100, 300), sigma_d=None):
    """One-step x0 error of `model` under the true condition and under swapped / zeroed conditions.

    Only the network's inputs (`cond`) are swapped; every variant is composed with the TRUE x0_hat
    (x_hat = x0_hat_true + sigma_d * pred in residual mode, x_hat = pred in standard mode), so the score measures
    what the network does with its condition and not the gap between two cubes' priors. Pairs are formed by rolling
    the batch by B // 2. `sigma_d` is required in residual mode."""
    x, y_d, x0_hat = [b.to(trainer.device) for b in batch]
    if x.shape[0] < 2:
        raise ValueError('y-swap check needs a batch of at least 2 items')
    if mode == 'residual' and sigma_d is None:
        raise ValueError('yswap_check in residual mode needs sigma_d (the residual scale of this d)')
    k = x.shape[0] // 2
    target = (x - x0_hat) / sigma_d if mode == 'residual' else x
    conds = {
        'true': {'sensor': y_d, 'x0_hat': x0_hat},
        'swap_full': {'sensor': y_d.roll(k, 0), 'x0_hat': x0_hat.roll(k, 0)},
        'swap_sensor_only': {'sensor': y_d.roll(k, 0), 'x0_hat': x0_hat},
        'zero': {'sensor': torch.zeros_like(y_d), 'x0_hat': torch.zeros_like(x0_hat)},
    }
    res, ok = {}, True
    for tv in t_values:
        row = {}
        for name, c in conds.items():
            pred = _one_step_x0(model, trainer, mode, target, c, tv)
            x_hat = (x0_hat + sigma_d * pred) if mode == 'residual' else pred
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
    n_seeds = len(args.seeds)
    rows = []
    for cfg in sampler_grid():
        ds = ds_ddpm if cfg['name'] == 'ddpm1000' else ds_all
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=residual_collate_fn)
        per = {k: [] for k in ('rmse_pct', 'psnr', 'sam_deg')}; base_r = []; secs = 0.0; nfe = None
        for si, seed in enumerate(args.seeds):
            for bi, batch in enumerate(loader):
                t0 = time.time()
                x_hat = reconstruct(model, trainer, args.mode, batch, cfg, sigma_d, seed * 1000 + bi)
                if trainer.device.type == 'cuda':
                    torch.cuda.synchronize(trainer.device)          # queued GPU work belongs to this cube's time
                secs += time.time() - t0
                nfe = trainer.last_nfe
                m = metrics(x_hat, batch[0].to(device))
                for k in per: per[k] += m[k].tolist()
                if si == 0:                                          # the baseline does not depend on the seed
                    base_r += metrics(batch[2].to(device), batch[0].to(device))['rmse_pct'].tolist()
        assert len(per['rmse_pct']) == n_seeds * len(ds)
        agg = aggregate_row(per, base_r, [class_of(f) for f in ds.img_list], n_seeds)
        rows.append(dict(cfg, nfe=nfe, sec_per_cube=secs / len(per['rmse_pct']), seeds=list(args.seeds),
                         batch_size=args.batch_size, files=[os.path.relpath(f, args.data_root) for f in ds.img_list],
                         per_item={k: _by_seed(v, n_seeds).tolist() for k, v in per.items()},   # [seed][cube]; cube order = this row's `files` (dataset sorted order, NOT the JSON subset order)
                         **agg))
        print(f"{cfg['name']:>14} nfe={nfe:5d} rmse={agg['rmse_pct']:.3f}±{agg['rmse_pct_se']:.3f} "
              f"(baseline {np.mean(base_r):.3f}, paired {agg['paired_rmse_diff']:+.3f}±{agg['paired_rmse_se']:.3f}) "
              f"seed-std {agg['rmse_pct_seed_std']:.3f} sam={agg['sam_deg']:.2f}")
    ds_ys = dataset_for(args.data_root, args.prior, args.d, yswap_files(load_subset(args.data_root, 'files'), subset_class_counts()))
    batch = residual_collate_fn([ds_ys[i] for i in range(len(ds_ys))])
    ys = yswap_check(model, trainer, args.mode, batch, sigma_d=sigma_d)
    print('y-swap:', json.dumps(ys))
    out = dict(ckpt=args.ckpt, mode=args.mode, d=args.d, sigma_d=sigma_d, seeds=args.seeds, baseline=base['mean'],
               baseline_per_item=base['per_item'], baseline_files=base['files'], rows=rows, yswap=ys,
               yswap_files=[os.path.relpath(f, args.data_root) for f in ds_ys.img_list])
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
