"""Phase 2a evaluation (spec §5): K samples per gate cube from the regressor-centred residual diffusion, the variance
head's and the deep ensemble's Gaussian predictions, the regressor and the baseline as point estimates, and (d = 4)
B's samples around x0_hat; CRPS, coverage, AUSE, spread ratio, HF-energy ratio, gradient W1, RMSE/PSNR/SAM per cube;
the y-swap of the mean-of-K; one record per run under uq/d<d>/eval/<name>.json (never overwritten).

    python scripts/uq_eval.py --d 8 --name d8_main --centre runs/regress_l1_sel_d8_b64/checkpoint_epoch_11.pth \
        --diffusion runs/residual_reg_d8_b64/checkpoint_epoch_100.pth --varhead runs/varhead_d8_b64/checkpoint_epoch_E.pth \
        --ensemble <regressor ckpts> --xr_cache $R/uq/d8/xr_cache --K 32 --sampler ddim10 --device cuda:0

--diffusion and --varhead are optional (the Stage 0 dry run scores the regressor, the ensemble and B only). Sample
coverage is exact only for K >= scripts.uq.metrics.min_k_for(level) (19 for the 90 % interval); the record states
this as coverage_min_k / coverage_exact. The first batch's samples (--store_cubes of them) go to
uq/d<d>/eval/<name>__cubes.npz, also never overwritten.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from data_loader.hfd_residual import residual_collate_fn  # noqa: E402
from data_loader.warmstart_paths import DATA_ROOT, PRIOR, UQ_DIR, file_sha256, git_head  # noqa: E402
from data_loader.xr_cache import meta_sha256  # noqa: E402
from model.diffusion_trainer import DiffusionTrainer  # noqa: E402
from model.residual_wrapper import LOGVAR_RANGE, build_residual_model, compose, regress_forward, varhead_forward  # noqa: E402
from scripts.eval_warmstart import class_of, dataset_for, load_subset, make_eval_loader, metrics, yswap_batch  # noqa: E402
from scripts.uq import metrics as M  # noqa: E402

SAMPLERS = {'ddim10': dict(method='ddim', n_steps=10), 'ddpm1000': dict(method='ddpm', n_steps=1000)}
PREREG_CENTRE = {8: ('regress_l1_sel_d8_b64', 11), 4: ('regress_l1_sel_d4_b64', 17)}      # spec §4.3; checked by uq_verdict
N_TIMESTEPS = 1000


# ----------------------------------------------------------------------------- checkpoints
def load_ckpt(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    return ck, ((ck.get('meta') if isinstance(ck, dict) else None) or {})


def ckpt_epoch(path):
    m = re.search(r'checkpoint_epoch_(\d+)\.pth$', path)
    return int(m.group(1)) if m else None


def ckpt_record(path, meta):
    return dict(path=os.path.abspath(path), run=os.path.basename(os.path.dirname(os.path.abspath(path))), epoch=ckpt_epoch(path),
                sha256=file_sha256(path), mode=meta.get('mode'), d=meta.get('d'), train_half=int((meta.get('args') or {}).get('train_half', 0) or 0),
                sigma_d=meta.get('sigma_d'), sigma_r=meta.get('sigma_r'), centre=meta.get('centre'), xr_cache_sha256=meta.get('xr_cache_sha256'))


def expect(meta, path, **want):
    bad = [f'{k}: checkpoint {meta.get(k, "<missing>")!r} vs required {v!r}' for k, v in want.items() if meta.get(k) != v]
    if bad:
        raise ValueError(f'checkpoint {path} does not fit this evaluation: ' + '; '.join(bad))


def _load(ck, meta, device):
    model = build_residual_model(meta['mode'], int(meta['base_channels']))
    model.load_state_dict(ck['model_state_dict'], strict=True)
    return model.to(device).eval()


def check_models(args, device):
    """Load every checkpoint and refuse (ValueError) any that does not fit: each one's mode and d; the diffusion and the
    variance head must have been trained around --centre (their meta centre.sha256) on the cache now on disk (--xr_cache,
    when given) with one sigma_r; --centre_halves must be the half-1 and half-2 regressors; an ensemble needs >= 2
    members; with --prior on disk every checkpoint's prior_sha256 must be that file's (the prior fixes x0_hat and
    sigma_d). Returns (models, sigma_r or None, provenance)."""
    if len(getattr(args, 'ensemble', None) or []) == 1:
        raise ValueError('--ensemble needs at least 2 members (the spread of one member is undefined)')
    models = dict(centre=None, diffusion=None, varhead=None, ensemble=[], b=None, halves=None)
    prov = dict(centre=None, diffusion=None, varhead=None, ensemble=[], b=None, halves=None, xr_cache=None, prior=None)
    prior = getattr(args, 'prior', '')
    prior_sha = file_sha256(prior) if prior and os.path.isfile(prior) else None
    if prior_sha is not None:
        prov['prior'] = dict(path=os.path.abspath(prior), sha256=prior_sha)

    def checked(path, mode):
        ck, meta = load_ckpt(path)
        expect(meta, path, mode=mode, d=args.d, **({} if prior_sha is None else dict(prior_sha256=prior_sha)))
        return ck, meta

    ck, mc = checked(args.centre, 'regress')
    models['centre'], prov['centre'] = _load(ck, mc, device), ckpt_record(args.centre, mc)
    centre_sha, sigma_r = prov['centre']['sha256'], None
    cache_sha = meta_sha256(args.xr_cache) if getattr(args, 'xr_cache', '') else None
    if cache_sha is not None:
        prov['xr_cache'] = dict(dir=args.xr_cache, sha256=cache_sha)
    for key, mode in (('diffusion', 'residual_reg'), ('varhead', 'varhead')):
        path = getattr(args, key, None)
        if not path:
            continue
        ck, meta = checked(path, mode)
        if (meta.get('centre') or {}).get('sha256') != centre_sha:
            raise ValueError(f'{key} checkpoint {path} was trained around centre {meta.get("centre")!r} but --centre is '
                             f'{prov["centre"]["run"]}@{prov["centre"]["epoch"]} ({centre_sha[:12]}...)')
        if cache_sha is not None and meta.get('xr_cache_sha256') != cache_sha:
            raise ValueError(f'{key} checkpoint {path}: xr_cache_sha256 {meta.get("xr_cache_sha256")!r} differs from the cache '
                             f'on disk {cache_sha!r} (the cache was rebuilt)')
        s = float(meta['sigma_r'])
        if sigma_r is not None and abs(s - sigma_r) >= 1e-9:
            raise ValueError(f'{key} sigma_r {s} differs from the diffusion sigma_r {sigma_r}')
        sigma_r = s
        models[key], prov[key] = _load(ck, meta, device), ckpt_record(path, meta)
    for path in (getattr(args, 'ensemble', None) or []):
        ck, meta = checked(path, 'regress')
        models['ensemble'].append(_load(ck, meta, device)); prov['ensemble'].append(ckpt_record(path, meta))
    if getattr(args, 'b', None):
        ck, meta = checked(args.b, 'residual')
        models['b'], prov['b'] = _load(ck, meta, device), ckpt_record(args.b, meta)
    if getattr(args, 'centre_halves', None):
        models['halves'], prov['halves'] = [], []
        for path, th in zip(args.centre_halves, (1, 2)):
            ck, meta = checked(path, 'regress')
            if int((meta.get('args') or {}).get('train_half', 0) or 0) != th:
                raise ValueError(f'--centre_halves: {path} is not a half-{th} regressor')
            models['halves'].append(_load(ck, meta, device)); prov['halves'].append(ckpt_record(path, meta))
    return models, sigma_r, prov


# ----------------------------------------------------------------------------- metrics per batch
def point_metrics(x_hat, x):
    """Per cube of [B, C, H, W] on [-1, 1]: RMSE (%), PSNR, SAM (deg), log HF-energy ratio, gradient W1 (lists of B)."""
    m = {k: v.tolist() for k, v in metrics(x_hat, x).items()}
    a, b = M.to01(x_hat), M.to01(x)
    m['hf'] = [M.hf_energy_ratio(a[i], b[i]) for i in range(a.shape[0])]
    m['w1'] = [M.grad_w1(a[i], b[i]) for i in range(a.shape[0])]
    return m


def sample_metrics(samples, x):
    """samples [K, B, C, H, W], x [B, C, H, W] on [-1, 1]: point metrics of the mean-of-K (mean_*) and of the seed-0 draw
    (single_*), then per cube on [0, 1] the sample CRPS (%), the empirical and Gaussian-fit coverage, AUSE and the
    spread ratio. Each cube is moved to the CPU (float64) on its own."""
    out = {f'mean_{k}': v for k, v in point_metrics(samples.mean(0), x).items()}
    out.update({f'single_{k}': v for k, v in point_metrics(samples[0], x).items()})
    keys = ('crps', 'cov50', 'cov90', 'cov50_gauss', 'cov90_gauss', 'ause', 'spread_ratio')
    acc = {k: [] for k in keys}
    for i in range(x.shape[0]):
        s, t = M.to01(samples[:, i]), M.to01(x[i])
        mu, sd = s.mean(0), s.std(0, ddof=1)
        c, g = M.coverage_samples(s, t), M.coverage_gaussian(mu, sd, t)
        acc['crps'].append(100.0 * float(M.crps_samples(s, t).mean()))
        acc['cov50'].append(c[0.5]); acc['cov90'].append(c[0.9]); acc['cov50_gauss'].append(g[0.5]); acc['cov90_gauss'].append(g[0.9])
        acc['ause'].append(M.ause(sd, mu - t)); acc['spread_ratio'].append(M.spread_ratio(sd ** 2, mu - t))
    out.update(acc)
    return out


def gaussian_metrics(mu, sigma, x):
    """A Gaussian prediction N(mu, sigma^2), mu and sigma on the [-1, 1] scale (sigma halved for [0, 1]): point metrics of
    mu (mean_*), then per cube the closed-form CRPS (%), coverage, AUSE and spread ratio."""
    out = {f'mean_{k}': v for k, v in point_metrics(mu, x).items()}
    mu01, x01 = M.to01(mu), M.to01(x)
    sd01 = sigma.detach().double().cpu().numpy() / 2.0
    acc = {k: [] for k in ('crps', 'cov50', 'cov90', 'ause', 'spread_ratio')}
    for i in range(x01.shape[0]):
        m_, s_, t = mu01[i], sd01[i], x01[i]
        c = M.coverage_gaussian(m_, s_, t)
        acc['crps'].append(100.0 * float(M.crps_gaussian(m_, s_, t).mean()))
        acc['cov50'].append(c[0.5]); acc['cov90'].append(c[0.9])
        acc['ause'].append(M.ause(s_, m_ - t)); acc['spread_ratio'].append(M.spread_ratio(s_ ** 2, m_ - t))
    out.update(acc)
    return out


@torch.no_grad()
def draw_samples(model, trainer, cond, shape, sigma, cfg, K, centre):
    """K draws compose(centre, sigma, r_k), r_k sampled with torch.manual_seed(k); the standardised residual's x0 estimate
    is clamped to +-2 / sigma (the reach of [-1, 1] from any centre)."""
    out = []
    for k in range(K):
        torch.manual_seed(k)
        r = trainer.sample(model, cond, shape, n_steps=cfg['n_steps'], method=cfg['method'], progress=False,
                           clamp_x0=(-2.0 / sigma, 2.0 / sigma))
        out.append(compose(centre, sigma, r))
    return torch.stack(out, 0)


@torch.no_grad()
def regress_centre(model, y, xh, sigma_d):
    return compose(xh, sigma_d, regress_forward(model, {'sensor': y, 'x0_hat': xh}))


# ----------------------------------------------------------------------------- the run
def _extend(acc, method, d):
    a = acc.setdefault(method, {})
    for k, v in d.items():
        a.setdefault(k, []).extend(v)


@torch.no_grad()
def run_eval(models, sigma_r, ds, loader, ys, args, provenance, device, store_dir=None):
    cfg = SAMPLERS[args.sampler]
    K, sigma_d = int(args.K), float(ds.sigma_d)
    trainer = DiffusionTrainer(device=device, n_timesteps=N_TIMESTEPS, prediction_type='v', loss_type='l1', snr_gamma=None)
    stored = None
    if models['diffusion'] is not None and store_dir is not None and args.store_cubes:
        stored = os.path.join(store_dir, f'{args.name}__cubes.npz')
        if os.path.exists(stored):
            raise FileExistsError(f'{stored} exists; stored cubes are never overwritten (pick another --name)')
        os.makedirs(store_dir, exist_ok=True)
    acc, t0 = {}, time.time()
    for bi, (x, y, xh) in enumerate(loader):
        x, y, xh = x.to(device), y.to(device), xh.to(device)
        xr = regress_centre(models['centre'], y, xh, sigma_d)
        cond = {'sensor': y, 'x0_hat': xr}
        _extend(acc, 'baseline', point_metrics(xh, x))
        _extend(acc, 'regressor', point_metrics(xr, x))
        if models['diffusion'] is not None:
            s = draw_samples(models['diffusion'], trainer, cond, tuple(xr.shape), sigma_r, cfg, K, xr)
            _extend(acc, 'diffusion', sample_metrics(s, x))
            if bi == 0 and stored is not None:
                n = min(int(args.store_cubes), x.shape[0])
                np.savez(stored, samples=s[:, :n].half().cpu().numpy(), x=x[:n].cpu().numpy(), x0_hat=xh[:n].cpu().numpy(),
                         x_r=xr[:n].cpu().numpy(), files=np.array([os.path.relpath(f, args.data_root) for f in ds.img_list[:n]]))
        if models['varhead'] is not None:
            logvar = varhead_forward(models['varhead'], cond).clamp(*LOGVAR_RANGE)
            _extend(acc, 'varhead', gaussian_metrics(xr, sigma_r * torch.exp(0.5 * logvar), x))
        if models['ensemble']:
            members = torch.stack([regress_centre(m, y, xh, sigma_d) for m in models['ensemble']], 0)
            _extend(acc, 'ensemble', gaussian_metrics(members.mean(0), members.std(0, unbiased=True), x))
        if models['b'] is not None:
            sb = draw_samples(models['b'], trainer, {'sensor': y, 'x0_hat': xh}, tuple(xh.shape), sigma_d, cfg, K, xh)
            _extend(acc, 'b', sample_metrics(sb, x))
        if models['halves'] and models['diffusion'] is not None:
            xr_h = torch.stack([regress_centre(m, y, xh, sigma_d) for m in models['halves']], 0).mean(0)
            _extend(acc, 'regressor_halves', point_metrics(xr_h, x))
            sh = draw_samples(models['diffusion'], trainer, {'sensor': y, 'x0_hat': xr_h}, tuple(xr.shape), sigma_r, cfg, K, xr_h)
            _extend(acc, 'diffusion_halfcentre', sample_metrics(sh, x))
        print(f'batch {bi + 1}/{len(loader)} ({time.time() - t0:.0f} s)', flush=True)
    yswap = None
    if models['diffusion'] is not None and ys is not None:
        files, x, y, xh = ys
        x, y, xh = x.to(device), y.to(device), xh.to(device)
        xr = regress_centre(models['centre'], y, xh, sigma_d)
        k = x.shape[0] // 2
        s_true = draw_samples(models['diffusion'], trainer, {'sensor': y, 'x0_hat': xr}, tuple(xr.shape), sigma_r, cfg, K, xr).mean(0)
        s_swap = draw_samples(models['diffusion'], trainer, {'sensor': y.roll(k, 0), 'x0_hat': xr.roll(k, 0)}, tuple(xr.shape), sigma_r, cfg, K, xr).mean(0)
        yswap = dict(K=K, files=[os.path.relpath(f, args.data_root) for f in files], true=metrics(s_true, x)['rmse_pct'].tolist(),
                     swap_full=metrics(s_swap, x)['rmse_pct'].tolist(), baseline=metrics(xh, x)['rmse_pct'].tolist(),
                     regressor=metrics(xr, x)['rmse_pct'].tolist())
    means = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in acc.items()}
    return dict(name=args.name, d=int(args.d), K=K, sampler=args.sampler, n=len(ds), files=[os.path.relpath(f, args.data_root) for f in ds.img_list],
                classes=[class_of(f) for f in ds.img_list], sigma_d=sigma_d, sigma_r=sigma_r,
                coverage_min_k={'0.5': M.min_k_for(0.5), '0.9': M.min_k_for(0.9)}, coverage_exact=bool(K >= M.min_k_for(0.9)),
                per_cube=acc, means=means, yswap=yswap,
                stored_cubes=stored, provenance=dict(provenance, git_head=git_head(), created=time.strftime('%Y-%m-%dT%H:%M:%S'),
                                                      seeds=list(range(K)), device=str(device), batch_size=int(args.batch_size),
                                                      subset=args.subset, limit_files=int(args.limit_files), seconds=time.time() - t0))


def output_path(d, name):
    return os.path.join(UQ_DIR, f'd{d}', 'eval', f'{name}.json')


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
    ap.add_argument('--d', type=int, required=True); ap.add_argument('--name', required=True)
    ap.add_argument('--centre', required=True); ap.add_argument('--diffusion', default=None); ap.add_argument('--varhead', default=None)
    ap.add_argument('--ensemble', nargs='*', default=[]); ap.add_argument('--b', default=None); ap.add_argument('--centre_halves', nargs=2, default=None)
    ap.add_argument('--xr_cache', default='')
    ap.add_argument('--K', type=int, default=32); ap.add_argument('--sampler', choices=sorted(SAMPLERS), default='ddim10')
    ap.add_argument('--subset', choices=['gate', 'ddpm'], default='gate'); ap.add_argument('--yswap_n', type=int, default=32)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch_size', type=int, default=4); ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--store_cubes', type=int, default=8); ap.add_argument('--limit_files', type=int, default=0)
    ap.add_argument('--data_root', default=DATA_ROOT); ap.add_argument('--prior', default=PRIOR)
    return ap


def main():
    args = get_parser().parse_args()
    out = output_path(args.d, args.name)
    if os.path.exists(out):
        raise FileExistsError(f'{out} exists; pick another --name')
    device = torch.device(args.device)
    models, sigma_r, prov = check_models(args, device)
    files = load_subset(args.data_root, 'files' if args.subset == 'gate' else 'ddpm1000_subset')
    if args.limit_files:
        files = files[:args.limit_files]
    ds = dataset_for(args.data_root, args.prior, args.d, files)
    loader = make_eval_loader(ds, args.batch_size, args.num_workers)
    ys = None
    if models['diffusion'] is not None:
        ds_ys, batch = yswap_batch(args.data_root, args.prior, args.d, n=args.yswap_n)
        ys = (list(ds_ys.img_list), *batch)
    rec = run_eval(models, sigma_r, ds, loader, ys, args, prov, device, store_dir=os.path.dirname(out))
    write_record(out, rec)
    print('wrote', out)
    for m, d in rec['means'].items():
        print(f"  {m:22s} " + ' '.join(f'{k}={v:.4f}' for k, v in d.items() if k in ('rmse_pct', 'mean_rmse_pct', 'single_rmse_pct', 'crps', 'cov90', 'ause', 'mean_hf', 'single_hf', 'hf')))


if __name__ == '__main__':
    main()
