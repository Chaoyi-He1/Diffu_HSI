"""Evaluation harness for the residual warm-start project (spec §7).

  python scripts/eval_warmstart.py --d 4 --baseline
  python scripts/eval_warmstart.py --d 4 --ckpt results/residual_warmstart/residual/d4/checkpoint_epoch_200.pth \
         --mode residual --base_channels 256 [--seeds 0 1] [--device cuda:0] [--grid quick]
  python scripts/eval_warmstart.py --d 4 --ckpt <run>/checkpoint_epoch_100.pth --mode regress --prediction_type direct --grid direct

Mode regress (the one-step regressor) is scored on the one-row direct grid; its y-swap is one deterministic call at
t = REGRESS_T on 32 cubes (--yswap_n).

Baselines go to results/residual_warmstart/eval/baseline_d<d>.json; checkpoint evaluations to
results/residual_warmstart/eval/<mode>/d<d>/<run_name>__<ckpt>[__quick|__direct].json (no suffix for --grid full)
with a `provenance` record. A checkpoint carrying a `meta` dict must agree with the CLI args, the prior's sigma_d and
the prior file's sha256 (else ValueError).

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
from data_loader.hfd_residual import HFDResidualData, limit_worker_threads, residual_collate_fn  # noqa: E402
from model.diffusion_trainer import DiffusionTrainer  # noqa: E402
from model.residual_wrapper import MODES, REGRESS_T, build_residual_model, regress_forward  # noqa: E402
from scripts.paired_stats import cluster_se  # noqa: E402
# Defined once (fix item 15) and re-exported from here: scripts/make_baseline_record.py and the tests import them from
# this module, and check_meta / checkpoint_eval use file_sha256 and git_head.
from data_loader.warmstart_paths import (DATA_ROOT, EVAL_DIR, PRIOR, RESULTS_ROOT, SUBSET,  # noqa: E402,F401
                                         file_sha256, git_head)


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


UNSEEN_CLASSES = ('P023', 'P024')     # the loader trains on P001-P022 only; P022 is the one seen validation class


def _mean_se(v):
    v = np.asarray(v, dtype=np.float64)
    return float(v.mean()), (float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0)


def _splits(classes, unseen=UNSEEN_CLASSES):
    """Split name -> boolean mask over items: one per class, plus 'unseen' (P023 + P024) when any is present."""
    classes = np.asarray(classes)
    out = {c: classes == c for c in sorted(set(classes.tolist()))}
    un = np.isin(classes, unseen)
    if un.any():
        out['unseen'] = un
    return out


def summarize(per_item, classes):
    """per_item: dict metric -> np.array [N]; classes: list of class ids per item. For each metric: mean and SE over
    items (`<k>`, `<k>_se`), and per split (`<k>_<class>`, `<k>_<class>_se`, `<k>_unseen`, `<k>_unseen_se`)."""
    out = {}
    splits = _splits(classes)
    for k, v in per_item.items():
        v = np.asarray(v, dtype=np.float64)
        out[k], out[k + '_se'] = _mean_se(v)
        for name, sel in splits.items():
            out[f'{k}_{name}'], out[f'{k}_{name}_se'] = _mean_se(v[sel])
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
    and the per-split means and SEs are then taken over the N cubes. The spread between seeds is reported separately as
    `<metric>_seed_std` (std over seeds of the per-seed means)."""
    by_seed = {k: _by_seed(v, n_seeds) for k, v in per_item.items()}
    n = next(iter(by_seed.values())).shape[1]
    if len(classes) != n:
        raise ValueError(f'{len(classes)} classes for {n} cubes')
    out = summarize({k: v.mean(0) for k, v in by_seed.items()}, classes)
    for k, v in by_seed.items():
        out[k + '_seed_std'] = float(v.mean(1).std(ddof=1)) if n_seeds > 1 else 0.0
    return out


def aggregate_row(per_item, base_rmse, classes, n_seeds, files=None):
    """summarize_seeds plus the paired RMSE difference to the baseline. base_rmse: per-cube baseline RMSE [N] (it does
    not depend on the seed). The difference is formed per cube (averaged over seeds) before its mean and SE, overall
    (`paired_rmse_diff`, `paired_rmse_se`) and per split (`paired_rmse_diff_<split>`, `paired_rmse_se_<split>`).

    With the cubes' `files`, it also reports the cluster-robust (CR1) SE of the paired difference over source images
    and the number of images, overall (`paired_rmse_se_cluster`, `n_clusters`) and per split (`..._<split>`; the SE is
    None for a split from one image). Patches of one image are not independent (fix item 10)."""
    out = summarize_seeds(per_item, classes, n_seeds)
    diff = (_by_seed(per_item['rmse_pct'], n_seeds) - np.asarray(base_rmse, dtype=np.float64)[None]).mean(0)
    out['paired_rmse_diff'], out['paired_rmse_se'] = _mean_se(diff)
    splits = _splits(classes)
    for name, sel in splits.items():
        out[f'paired_rmse_diff_{name}'], out[f'paired_rmse_se_{name}'] = _mean_se(diff[sel])
    if files is not None:
        files = list(files)
        out['paired_rmse_se_cluster'], out['n_clusters'] = cluster_se(diff, files)
        for name, sel in splits.items():
            out[f'paired_rmse_se_cluster_{name}'], out[f'n_clusters_{name}'] = \
                cluster_se(diff[sel], [f for f, s in zip(files, sel) if s])
    return out


# ----------------------------------------------------------------------------- data
def make_eval_loader(ds, batch_size, num_workers=4):
    """The harness's only DataLoader factory: in order, no shuffling, residual_collate_fn, and one BLAS thread per
    worker (limit_worker_threads; fix item 5) whatever the launching shell exported."""
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                       collate_fn=residual_collate_fn, worker_init_fn=limit_worker_threads)


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


def yswap_batch(data_root, prior_path, d, n=8):
    """The y-swap batch: yswap_files(n) over the 200-cube subset, collated in the dataset's (sorted) order. Returns
    (dataset, batch); dataset.img_list is the batch order. Shared by checkpoint_eval and scripts/yswap_one_step.py."""
    ds = dataset_for(data_root, prior_path, d, yswap_files(load_subset(data_root, 'files'), subset_class_counts(), n))
    return ds, residual_collate_fn([ds[i] for i in range(len(ds))])


# ----------------------------------------------------------------------------- baseline
def baseline_eval(data_root, prior_path, d, files=None, batch_size=16):
    ds = dataset_for(data_root, prior_path, d, files)
    loader = make_eval_loader(ds, batch_size)
    per = {k: [] for k in ('rmse_pct', 'psnr', 'sam_deg')}
    for x, _, xh in loader:
        m = metrics(xh, x)
        for k in per: per[k] += m[k].tolist()
    classes = [class_of(f) for f in ds.img_list]
    return dict(d=d, n=len(ds), per_item=per, files=[os.path.relpath(f, data_root) for f in ds.img_list],
                classes=classes, mean=summarize(per, classes), nfe=0)


# ----------------------------------------------------------------------------- samplers
QUICK_GRID = ('ddim1', 'ddim2', 'ddim5', 'ddim10', 'ddim20', 'warm50_ddim', 'warm100_ddim', 'warm200_ddim', 'warm400_ddim')


def sampler_grid(which='full'):
    if which == 'direct':                        # the one-step regressor: one deterministic call, no sampler
        return [dict(name='direct', method='direct', n_steps=1)]
    g = [dict(name='ddpm1000', method='ddpm', n_steps=1000)]
    g += [dict(name=f'ddim{k}', method='ddim', n_steps=k) for k in (1, 2, 5, 10, 20)]
    g += [dict(name=f'warm{t}_ddim', method='ddim', n_steps=10, t_start=t) for t in (50, 100, 200, 400)]
    g += [dict(name=f'warm{t}_ddpm', method='ddpm', n_steps=1000, t_start=t) for t in (50, 100, 200, 400)]
    if which == 'full':
        return g
    if which == 'quick':                         # milestone checks: every row costs <= 20 NFE, no 50-cube DDPM row
        return [c for c in g if c['name'] in QUICK_GRID]
    raise ValueError(f"grid must be 'full', 'quick' or 'direct', got {which!r}")


def reconstruct(model, trainer, mode, batch, cfg, sigma_d, seed):
    """batch = (x, y_d, x0_hat) tensors. Returns x_hat [B, 64, H, W] clamped to [-1, 1]. Mode regress takes the
    'direct' cfg only (one deterministic call; `seed` unused) and every other mode takes a sampler cfg."""
    x, y_d, x0_hat = [b.to(trainer.device) for b in batch]
    cond = {'sensor': y_d, 'x0_hat': x0_hat}
    if (cfg['method'] == 'direct') != (mode == 'regress'):
        raise ValueError(f"the direct cfg is for mode 'regress' only, and mode 'regress' takes only the direct cfg "
                         f"(mode={mode!r}, method={cfg['method']!r})")
    if mode == 'regress':
        with torch.no_grad():
            r = regress_forward(model, cond)
        trainer.last_nfe = 1
        return (x0_hat + sigma_d * r).clamp(-1, 1)        # clamping the cube = clamping r to the feasible envelope
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
YSWAP_NOISE_SEED = 1234
YSWAP_X_T = ('q_sample', 'noise')      # the network's x_t in the one-step y-swap (see yswap_check)


def _one_step_x0(model, trainer, mode, x_target, cond, t_val, x_t='q_sample'):
    B = x_target.shape[0]
    t = torch.full((B,), t_val, device=trainer.device, dtype=torch.long)
    torch.manual_seed(YSWAP_NOISE_SEED)
    if x_t == 'noise':                   # the ddim1 sampler's x_T, drawn as trainer.sample draws it (same seed, shape)
        xt = torch.randn(tuple(x_target.shape), device=trainer.device)
    else:
        xt, _ = trainer.q_sample(x_target, t)
    with torch.no_grad():
        x0_pred, _ = trainer._model_to_x0_eps(model(xt, cond, t), xt, t)
    return x0_pred


def check_yswap_x_t(x_t, t_values, n_timesteps):
    """x_t must be one of YSWAP_X_T; 'noise' is the ddim1 sampler's start x_T and is defined only at its one call,
    t = n_timesteps - 1 (999). Raises ValueError otherwise."""
    if x_t not in YSWAP_X_T:
        raise ValueError(f'x_t must be one of {YSWAP_X_T}, got {x_t!r}')
    if x_t == 'noise' and any(int(t) != n_timesteps - 1 for t in t_values):
        raise ValueError(f"x_t='noise' is the ddim1 sampler's pure-noise start x_T, defined only at its one call "
                         f"t = {n_timesteps - 1}; got t = {list(t_values)}")


YSWAP_RATIO = 0.9          # spec §7: the true condition must score <= 0.9 x swap_full at every t of the check


def yswap_check(model, trainer, mode, batch, t_values=(100, 300), sigma_d=None, files=None, x_t='q_sample'):
    """One-step x0 error of `model` under the true condition and under swapped / zeroed conditions.

    Only the network's inputs (`cond`) are swapped; every variant is composed with the TRUE x0_hat
    (x_hat = x0_hat_true + sigma_d * pred in residual mode, x_hat = pred in standard mode), so the score measures
    what the network does with its condition and not the gap between two cubes' priors. Pairs are formed by rolling
    the batch by B // 2. `sigma_d` is required in residual mode.

    Returns `t<t>` -> {variant: mean RMSE %} and `passes` (true <= YSWAP_RATIO x swap_full at every t; spec §7 uses
    t = 100 and 300, the default), plus, additively (fix item 1): `per_cube` -> {str(t): {variant: [RMSE % per cube]}},
    `baseline_per_cube` (the prior x0_hat's RMSE %, as baseline_eval scores it) and `files` (the cubes in batch
    order, None when not given).

    x_t (final-review F1) is the network's x_t, the same for every variant (one eps, seed YSWAP_NOISE_SEED):
      'q_sample' (default; spec §7 at t = 100 / 300 and every Phase 1a record): x_t = q_sample(TRUE target, t). At
                 t = 999 that is 0.00635 r + 0.99998 eps: it carries the true residual r, which B reads, and B's real
                 one-step read-out (ddim1) never sees it -- a leak that flatters 'true' (and every variant).
      'noise':   x_t = eps, exactly the ddim1 sampler's x_T (trainer.sample draws randn of the batch shape after the
                 same seed), so at t = 999 'true' is reconstruct(mode, ddim1) seeded with YSWAP_NOISE_SEED. Defined
                 only at t = n_timesteps - 1 = 999 (ValueError at any other t): elsewhere a sampler never feeds pure
                 noise.
    Mode regress: one deterministic call per variant at t = REGRESS_T on its zero x_t (t_values and x_t do not
    apply)."""
    x, y_d, x0_hat = [b.to(trainer.device) for b in batch]
    if x.shape[0] < 2:
        raise ValueError('y-swap check needs a batch of at least 2 items')
    if mode in ('residual', 'regress') and sigma_d is None:
        raise ValueError('yswap_check in residual and regress modes needs sigma_d (the residual scale of this d)')
    if mode == 'regress':
        t_values = (REGRESS_T,)          # one deterministic call: the noise level does not apply
    check_yswap_x_t(x_t, t_values, trainer.n_timesteps)
    k = x.shape[0] // 2
    target = (x - x0_hat) / sigma_d if mode in ('residual', 'regress') else x
    conds = {
        'true': {'sensor': y_d, 'x0_hat': x0_hat},
        'swap_full': {'sensor': y_d.roll(k, 0), 'x0_hat': x0_hat.roll(k, 0)},
        'swap_sensor_only': {'sensor': y_d.roll(k, 0), 'x0_hat': x0_hat},
        'zero': {'sensor': torch.zeros_like(y_d), 'x0_hat': torch.zeros_like(x0_hat)},
    }
    if files is not None and len(files) != x.shape[0]:
        raise ValueError(f'{len(files)} files for a batch of {x.shape[0]}')
    res, per_cube, ok = {}, {}, True
    for tv in t_values:
        row, cubes = {}, {}
        for name, c in conds.items():
            if mode == 'regress':
                with torch.no_grad():
                    pred = regress_forward(model, c)
            else:
                pred = _one_step_x0(model, trainer, mode, target, c, tv, x_t)
            x_hat = (x0_hat + sigma_d * pred) if mode in ('residual', 'regress') else pred
            rmse = metrics(x_hat.clamp(-1, 1), x)['rmse_pct']
            row[name], cubes[name] = float(rmse.mean()), rmse.tolist()
        res[f't{tv}'], per_cube[str(tv)] = row, cubes
        ok = ok and (row['true'] <= YSWAP_RATIO * row['swap_full'])
    res['passes'] = bool(ok)
    res['per_cube'] = per_cube
    res['baseline_per_cube'] = metrics(x0_hat, x)['rmse_pct'].tolist()
    res['files'] = None if files is None else list(files)
    return res


# ----------------------------------------------------------------------------- checkpoint evaluation
def build_model(mode, base_channels, device):
    return build_residual_model(mode, base_channels).to(device).eval()


def eval_output_path(mode, d, run_name, ckpt, grid='full'):
    """eval/<mode>/d<d>/<run_name>__<ckpt stem>[__<grid>].json. run_name defaults to the checkpoint's parent directory
    name, so the pilot (…_b64) and the full run (…_b256) of one mode and d never overwrite each other; a non-full grid
    is suffixed so a quick milestone evaluation never overwrites the final full one."""
    run_name = run_name or os.path.basename(os.path.dirname(os.path.abspath(ckpt)))
    stem = os.path.splitext(os.path.basename(ckpt))[0]
    suffix = '' if grid == 'full' else f'__{grid}'
    return os.path.join(EVAL_DIR, mode, f'd{d}', f'{run_name}__{stem}{suffix}.json')


META_ARGS = ('mode', 'd', 'base_channels', 'prediction_type')


def check_meta(ck, args, sigma_d, prior_sha256):
    """Compare a checkpoint's `meta` dict (written by the training script) with this evaluation: META_ARGS against
    the CLI args, sigma_d against the prior's (|diff| < 1e-9), prior_sha256 against the prior file's hash. Raises
    ValueError naming every mismatch (a missing key counts as one). Without `meta`, prints a one-line warning and
    returns False (the CLI args are trusted); returns True when the meta was checked."""
    meta = ck.get('meta') if isinstance(ck, dict) else None
    if meta is None:
        print(f'warning: checkpoint {args.ckpt} carries no meta; trusting the CLI args '
              f'({", ".join(f"{k}={getattr(args, k)!r}" for k in META_ARGS)})')
        return False
    bad = [f'{k}: checkpoint {meta.get(k, "<missing>")!r} vs args {getattr(args, k)!r}'
           for k in META_ARGS if k not in meta or meta[k] != getattr(args, k)]
    if 'sigma_d' not in meta or abs(float(meta['sigma_d']) - sigma_d) >= 1e-9:
        bad.append(f'sigma_d: checkpoint {meta.get("sigma_d", "<missing>")!r} vs prior {sigma_d!r}')
    if meta.get('prior_sha256') != prior_sha256:
        bad.append(f'prior_sha256: checkpoint {meta.get("prior_sha256", "<missing>")!r} vs prior file {prior_sha256!r}')
    if bad:
        raise ValueError(f'checkpoint {args.ckpt} does not match this evaluation: ' + '; '.join(bad))
    return True


def checkpoint_eval(args):
    grid = getattr(args, 'grid', 'full')
    if (args.mode == 'regress') != (grid == 'direct'):
        raise ValueError(f"--mode regress needs --grid direct, and --grid direct needs --mode regress "
                         f"(got mode={args.mode!r}, grid={grid!r})")
    if (args.mode == 'regress') != (args.prediction_type == 'direct'):
        raise ValueError(f"--mode regress needs --prediction_type direct, and 'direct' needs --mode regress "
                         f"(got mode={args.mode!r}, prediction_type={args.prediction_type!r})")
    yswap_n = getattr(args, 'yswap_n', None) or (32 if args.mode == 'regress' else 8)
    device = torch.device(args.device)
    run_name = getattr(args, 'run_name', None) or os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))
    trainer = DiffusionTrainer(device=device, prediction_type=args.prediction_type, loss_type='l1', snr_gamma=None)
    model = build_model(args.mode, args.base_channels, device)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    targs = ((ck.get('meta') or {}).get('args') or {}) if isinstance(ck, dict) else {}
    ds_all = dataset_for(args.data_root, args.prior, args.d, load_subset(args.data_root, 'files'))
    sigma_d = ds_all.sigma_d
    prior_sha256 = file_sha256(args.prior)
    meta_checked = check_meta(ck, args, sigma_d, prior_sha256)                 # before any expensive work
    sd = ck.get('model_state_dict', ck)
    model.load_state_dict(sd, strict=True)
    ds_ddpm = dataset_for(args.data_root, args.prior, args.d, load_subset(args.data_root, 'ddpm1000_subset'))
    provenance = dict(mode=args.mode, d=args.d, base_channels=args.base_channels, prediction_type=args.prediction_type,
                      n_timesteps=trainer.n_timesteps, sigma_d=sigma_d, prior_s=float(ds_all.prior['s']),
                      prior_sha256=prior_sha256, prior_path=os.path.abspath(args.prior),
                      ckpt=os.path.abspath(args.ckpt), git_head=git_head(), meta_checked=meta_checked,
                      loss_type=targs.get('loss_type'), train_seed=targs.get('seed'), yswap_n=yswap_n)
    base = baseline_eval(args.data_root, args.prior, args.d, ds_all.img_list)
    n_seeds = len(args.seeds)
    rows = []
    for cfg in sampler_grid(grid):
        ds = ds_ddpm if cfg['name'] == 'ddpm1000' else ds_all
        loader = make_eval_loader(ds, args.batch_size)
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
        agg = aggregate_row(per, base_r, [class_of(f) for f in ds.img_list], n_seeds, files=ds.img_list)
        rows.append(dict(cfg, nfe=nfe, sec_per_cube=secs / len(per['rmse_pct']), seeds=list(args.seeds),
                         batch_size=args.batch_size, files=[os.path.relpath(f, args.data_root) for f in ds.img_list],
                         per_item={k: _by_seed(v, n_seeds).tolist() for k, v in per.items()},   # [seed][cube]; cube order = this row's `files` (dataset sorted order, NOT the JSON subset order)
                         **agg))
        print(f"{cfg['name']:>14} nfe={nfe:5d} rmse={agg['rmse_pct']:.3f}±{agg['rmse_pct_se']:.3f} "
              f"(baseline {np.mean(base_r):.3f}, paired {agg['paired_rmse_diff']:+.3f}±{agg['paired_rmse_se']:.3f}) "
              f"seed-std {agg['rmse_pct_seed_std']:.3f} sam={agg['sam_deg']:.2f}")
    ds_ys, batch = yswap_batch(args.data_root, args.prior, args.d, yswap_n)
    ys = yswap_check(model, trainer, args.mode, batch, sigma_d=sigma_d,
                     files=[os.path.relpath(f, args.data_root) for f in ds_ys.img_list])
    print('y-swap:', json.dumps(ys))
    out = dict(provenance=provenance, ckpt=args.ckpt, mode=args.mode, d=args.d, sigma_d=sigma_d, seeds=args.seeds,
               grid=grid, run_name=run_name,
               baseline=base['mean'], baseline_per_item=base['per_item'], baseline_files=base['files'], rows=rows,
               yswap=ys, yswap_files=[os.path.relpath(f, args.data_root) for f in ds_ys.img_list])
    path = eval_output_path(args.mode, args.d, run_name, args.ckpt, grid)      # spec S12; baselines stay in EVAL_DIR
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, required=True)
    ap.add_argument('--baseline', action='store_true')
    ap.add_argument('--ckpt'); ap.add_argument('--mode', choices=list(MODES))
    ap.add_argument('--grid', choices=['full', 'quick', 'direct'], default='full')
    ap.add_argument('--run_name', default=None)
    ap.add_argument('--base_channels', type=int, default=64)
    ap.add_argument('--prediction_type', default='v', help="'direct' for --mode regress")
    ap.add_argument('--yswap_n', type=int, default=None, help='y-swap cubes (default 32 for --mode regress, 8 otherwise)')
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
