"""Step-0 diagnostics of the trained residual diffusion model B (Phase 1a-prime, Task 3).

Is B's sampling chain centred on r = 0 (H0: it samples an unconditional residual) or on its one-step estimate (H1: a
conditional posterior sampler)? Is its spread calibrated to its one-step error? Does averaging B's t = 999 one-step
output over K noise draws beat a single draw? (Phase 1a RESULTS note, Decision step 0.)

  sample  reconstructs every cube of the requested rows for each requested draw k with the harness's own seeding
          (torch.manual_seed(k * 1000 + batch index), batch 8, dataset order) -- so draws 0 and 1 reproduce B's
          stored evaluation when sampled on CUDA (the sampler draws its noise from the device's generator; a CPU
          shard draws independent noise) -- and saves per-cube running sums to a new .npz shard (never overwritten).
  merge   adds shards with distinct, disjoint draws of one checkpoint and prior, sampled at batch 8 (the t999 row is
          required), reloads the cubes and B's evaluation, and writes the statistics JSON (never over an input).

Rows: ddim10 and t999 (the ddim1 sampler: one call at t = 999) on the 200-cube subset; ddpm1000 on the 50-cube DDPM
subset, as the harness scores it.

  python scripts/diagnose_b.py sample --rows ddim10 t999 --draws 0 1 2 3 4 5 6 7 --device cuda:0 --out <diag>/b_fast.npz
  python scripts/diagnose_b.py sample --rows ddpm1000 --draws 0 1 --device cuda:0 --out <diag>/b_ddpm_a.npz
  python scripts/diagnose_b.py sample --rows ddpm1000 --draws 2 3 --device cuda:1 --out <diag>/b_ddpm_b.npz
  python scripts/diagnose_b.py merge --inputs <diag>/b_fast.npz <diag>/b_ddpm_a.npz <diag>/b_ddpm_b.npz \
      --out <diag>/b_diagnostics.json

Energies are in pct² of the [0, 1] display scale, E = 2500 * mean((x_hat - x)^2), so sqrt(E) is the harness's RMSE %.
Per cube, with K draws x_k, their mean x_bar and the truth x:
  o      = mean_k E(x_k)                                   single-draw error energy
  ebar   = E(x_bar)                                        error energy of the K-draw mean
  spread = 2500 * sum_k mean((x_k - x_bar)^2) / (K - 1)    unbiased spread energy (per-element variance across draws)
  c2     = ebar - spread / K                               error energy of the chain's centre (unbiased)
The one-step centre energy c1 is the t999 row's c2: the error energy of the noise-free centre of B's t = 999 output.
(A single ddim1 draw's RMSE^2 also holds that output's own noise energy, so it is a biased H1 endpoint.)
lambda = (mean b^2 - mean c2) / (mean b^2 - mean c1): 0 = centred on r = 0 (b: the baseline's RMSE), 1 = centred on
the one-step centre; SE by a jackknife over source images.
calibration: ratio_of_means = mean spread / mean c1 and self_ratio = mean spread / mean c2 are both 1 for a calibrated
posterior sampler; per_cube_slope_on_b2 (spread regressed on the noise-free b^2) is then close to the reference slope
of c1 on b^2, while a slope near 0 means a cube-independent spread whatever the ratios.
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data_loader.warmstart_paths import DATA_ROOT, EVAL_DIR, PRIOR, RESULTS_ROOT, file_sha256, git_head  # noqa: E402
from model.diffusion_trainer import DiffusionTrainer  # noqa: E402
from scripts.eval_warmstart import (_mean_se, build_model, check_meta, dataset_for, load_subset,  # noqa: E402
                                    make_eval_loader, metrics, reconstruct, sampler_grid)
from scripts.paired_stats import cluster_se, source_image  # noqa: E402

B_CKPT = os.path.join(RESULTS_ROOT, 'runs', 'residual_d4_b64', 'checkpoint_epoch_100.pth')
B_EVAL = os.path.join(EVAL_DIR, 'residual', 'd4', 'residual_d4_b64__checkpoint_epoch_100.json')
DIAG_DIR = os.path.join(RESULTS_ROOT, 'diagnostics')
ROWS = {'ddim10': ('ddim10', 'files'), 't999': ('ddim1', 'files'), 'ddpm1000': ('ddpm1000', 'ddpm1000_subset')}
SCALE = 2500.0      # E = 2500 * mean squared error on [-1, 1] = (RMSE % of the [0, 1] scale)^2
HARNESS_BATCH = 8   # B's stored evaluation seeds per batch index of batch-8 loaders; other batch sizes draw other noise


def under(path, root):
    path, root = os.path.realpath(path), os.path.realpath(root)
    return os.path.commonpath([path, root]) == root


def check_out(path):
    if under(path, RESULTS_ROOT) and not under(path, DIAG_DIR):
        raise SystemExit(f'{path} is inside the results tree {RESULTS_ROOT}; diagnostics write there only under {DIAG_DIR}')


def row_cfg(row):
    return next(c for c in sampler_grid('full') if c['name'] == ROWS[row][0])


def accumulate(batches, recon, draws, n_cubes):
    """batches: the cubes in order as (x, y_d, x0_hat) tensors; recon(batch, seed) -> x_hat on [-1, 1]. For each draw k
    and batch index bi calls recon(batch, k * 1000 + bi) (the harness's seeding) and returns per-cube sums: sum_x
    [N, C, H, W] float64, sum_x2 [N] (sum over draws of the per-element mean of x_hat^2), err [len(draws), N] RMSE %."""
    sum_x, sum_x2, err = None, np.zeros(n_cubes), np.zeros((len(draws), n_cubes))
    for di, k in enumerate(draws):
        i0 = 0
        for bi, batch in enumerate(batches):
            x_hat = recon(batch, k * 1000 + bi)
            xh = x_hat.detach().double().cpu().numpy()
            n = xh.shape[0]
            if i0 + n > n_cubes:
                raise ValueError(f'batches hold more than the expected {n_cubes} cubes')
            if sum_x is None:
                sum_x = np.zeros((n_cubes,) + xh.shape[1:])
            sum_x[i0:i0 + n] += xh
            sum_x2[i0:i0 + n] += (xh ** 2).reshape(n, -1).mean(1)
            err[di, i0:i0 + n] = metrics(x_hat.detach().cpu().double(), batch[0].cpu().double())['rmse_pct'].numpy()
            i0 += n
        if i0 != n_cubes:
            raise ValueError(f'batches hold {i0} cubes, expected {n_cubes}')
    return sum_x, sum_x2, err


def row_stats(sum_x, sum_x2, err, x):
    """Per-cube energies from K draws (see the module docstring). x: [N, C, H, W] truth on [-1, 1]."""
    K = err.shape[0]
    if K < 2:
        raise ValueError('need at least 2 draws for a spread estimate')
    xbar = sum_x / K
    ebar = SCALE * ((xbar - x) ** 2).reshape(len(x), -1).mean(1)
    spread = SCALE * (sum_x2 - K * (xbar ** 2).reshape(len(x), -1).mean(1)) / (K - 1)
    return dict(K=int(K), o=(err ** 2).mean(0), ebar=ebar, spread=spread, c2=ebar - spread / K,
                kmean_rmse=np.sqrt(ebar), single_rmse=err.mean(0))


def centre_lambda(b, c2, c2_one_step):
    """(mean b^2 - mean c2) / (mean b^2 - mean c2_one_step), all in pct^2: 0 = the chain is centred on r = 0, 1 = on
    the one-step centre (c2_one_step: the t999 row's per-cube c2, the noise-free error energy of B's one-step output)."""
    b2 = float(np.mean(np.asarray(b) ** 2))
    return (b2 - float(np.mean(c2))) / (b2 - float(np.mean(c2_one_step)))


def jackknife_se(stat, files):
    """Leave-one-source-image-out jackknife SE of stat(mask) over the cubes `files`; None with fewer than 2 images."""
    groups = sorted({source_image(f) for f in files})
    if len(groups) < 2:
        return None
    gid = np.array([groups.index(source_image(f)) for f in files])
    vals = np.array([stat(gid != g) for g in range(len(groups))])
    G = len(groups)
    return float(np.sqrt((G - 1) / G * ((vals - vals.mean()) ** 2).sum()))


def merge_shards(shards):
    """Add shards row by row; their draws must be distinct within and disjoint across shards, their cube lists
    identical."""
    out = {}
    for sh in shards:
        for row, d in sh.items():
            draws = list(d['draws'])
            repeated = sorted({k for k in draws if draws.count(k) > 1})
            if repeated:
                raise ValueError(f'{row}: draws {repeated} repeat within a shard')
            if row not in out:
                out[row] = dict(files=list(d['files']), draws=list(d['draws']), sum_x=d['sum_x'].copy(),
                                sum_x2=d['sum_x2'].copy(), err=d['err'].copy())
                continue
            o = out[row]
            if list(d['files']) != o['files']:
                raise ValueError(f'{row}: shards score different cubes')
            both = sorted(set(d['draws']) & set(o['draws']))
            if both:
                raise ValueError(f'{row}: draws {both} appear in two shards')
            o['draws'] += list(d['draws'])
            o['sum_x'] += d['sum_x']
            o['sum_x2'] += d['sum_x2']
            o['err'] = np.concatenate([o['err'], d['err']])
    return out


def save_shard(path, rows, meta):
    arrays = {'meta': np.array(json.dumps(meta))}
    for row, d in rows.items():
        arrays[f'{row}.files'] = np.array(d['files'])
        arrays[f'{row}.draws'] = np.array(d['draws'])
        arrays[f'{row}.sum_x'] = d['sum_x']
        arrays[f'{row}.sum_x2'] = d['sum_x2']
        arrays[f'{row}.err'] = d['err']
    tmp = path + '.tmp.npz'
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def load_shard(path):
    z = np.load(path)
    rows = {}
    for key in z.files:
        if key != 'meta':
            row, field = key.split('.', 1)
            rows.setdefault(row, {})[field] = z[key]
    for d in rows.values():
        d['files'] = [str(f) for f in d['files']]
        d['draws'] = [int(k) for k in d['draws']]
    return rows, json.loads(str(z['meta']))


def _ms(v):
    m, se = _mean_se(np.asarray(v, dtype=float))
    return dict(mean=float(m), se=float(se))


def _pair(d, files):
    se_cluster, n_clusters = cluster_se(d, files)
    return dict(_ms(d), se_cluster=se_cluster, n_clusters=n_clusters)


STORED = {'ddim10': 'ddim10', 't999': 'ddim1', 'ddpm1000': 'ddpm1000'}


def consistency(d, ev_b, row):
    """Max |RMSE %| difference between this run's draws 0 / 1 and B's stored evaluation (same seeds, same batching)."""
    r = next(x for x in ev_b['rows'] if x['name'] == STORED[row])
    col = {f: i for i, f in enumerate(r['files'])}
    out = {}
    for di, k in enumerate(d['draws']):
        if k in r['seeds']:
            s = r['per_item']['rmse_pct'][r['seeds'].index(k)]
            out[str(k)] = float(max(abs(d['err'][di, i] - s[col[f]]) for i, f in enumerate(d['files'])))
    return out


def cmd_sample(args):
    check_out(args.out)
    if os.path.exists(args.out):
        raise SystemExit(f'{args.out} exists; sample never overwrites a shard (choose another --out)')
    if len(set(args.draws)) != len(args.draws):
        raise ValueError(f'--draws {args.draws} repeat a draw; each draw k is one independent noise seed')
    device = torch.device(args.device)
    trainer = DiffusionTrainer(device=device, prediction_type='v', loss_type='l1', snr_gamma=None)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model, rows, t0 = None, {}, time.time()
    for row in args.rows:
        ds = dataset_for(args.data_root, args.prior, args.d, load_subset(args.data_root, ROWS[row][1]))
        if model is None:
            check_meta(ck, SimpleNamespace(ckpt=args.checkpoint, mode='residual', d=args.d, base_channels=args.base_channels,
                                           prediction_type='v'), ds.sigma_d, file_sha256(args.prior))
            model = build_model('residual', args.base_channels, device)
            model.load_state_dict(ck.get('model_state_dict', ck), strict=True)
        batches = list(make_eval_loader(ds, args.batch_size))
        cfg, sigma_d = row_cfg(row), ds.sigma_d
        sum_x, sum_x2, err = accumulate(
            batches, lambda batch, seed: reconstruct(model, trainer, 'residual', batch, cfg, sigma_d, seed),
            args.draws, len(ds))
        rows[row] = dict(files=[os.path.relpath(f, args.data_root) for f in ds.img_list], draws=list(args.draws),
                         sum_x=sum_x, sum_x2=sum_x2, err=err)
        print(f'{row}: {len(ds)} cubes x {len(args.draws)} draws, mean single-draw RMSE {err.mean():.4f} % '
              f'({time.time() - t0:.0f} s)', flush=True)
    meta = dict(checkpoint=os.path.abspath(args.checkpoint), checkpoint_sha256=file_sha256(args.checkpoint),
                prior_sha256=file_sha256(args.prior), device=str(device), batch_size=args.batch_size,
                git_head=git_head(), torch=torch.__version__, seconds=time.time() - t0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_shard(args.out, rows, meta)
    print('wrote', args.out)


def cmd_merge(args):
    check_out(args.out)
    if os.path.realpath(args.out) in {os.path.realpath(p) for p in args.inputs}:
        raise SystemExit(f'--out {args.out} is one of the --inputs; merge never overwrites a shard')
    loaded = [load_shard(p) for p in args.inputs]
    for field in ('checkpoint_sha256', 'prior_sha256'):                   # checked only where a shard records it
        seen = sorted({str(m[field]) for _, m in loaded if field in m})
        if len(seen) > 1:
            raise ValueError(f'shards disagree on {field}: {seen}; merge only draws of one checkpoint and prior')
    bad_batch = [(p, m['batch_size']) for p, (_, m) in zip(args.inputs, loaded)
                 if 'batch_size' in m and m['batch_size'] != HARNESS_BATCH]
    if bad_batch:
        raise ValueError(f'batch_size must be {HARNESS_BATCH} (the harness seeds per batch index): {bad_batch}')
    non_cuda = [m.get('device') for _, m in loaded if not str(m.get('device', '')).startswith('cuda')]
    if non_cuda:
        print(f'WARNING: shards sampled on {non_cuda}: the sampler noise comes from the device generator, so draws 0/1 '
              f'cannot reproduce the GPU-made evaluation; consistency_vs_stored is not a seeding check for them', flush=True)
    rows = merge_shards([sh for sh, _ in loaded])
    if 't999' not in rows:
        raise ValueError('merge needs the t999 row: its per-cube c2 is the one-step centre energy (the H1 endpoint)')
    ev_b = json.load(open(args.b_eval))
    base = dict(zip(ev_b['baseline_files'], ev_b['baseline_per_item']['rmse_pct']))
    d1 = next(r for r in ev_b['rows'] if r['name'] == 'ddim1')
    m1 = dict(zip(d1['files'], np.asarray(d1['per_item']['rmse_pct'], dtype=float).mean(0)))
    out, one_step_c2 = {}, None
    for row in sorted(rows, key=lambda r: r != 't999'):            # t999 first: its c2 is the one-step centre energy
        d = rows[row]
        files = d['files']
        ds = dataset_for(args.data_root, args.prior, args.d, [os.path.join(args.data_root, f) for f in files])
        if [os.path.relpath(f, args.data_root) for f in ds.img_list] != files:
            raise ValueError(f'{row}: the dataset order differs from the shard order')
        x = np.stack([ds[i][0] for i in range(len(ds))]).transpose(0, 3, 1, 2).astype(np.float64)
        s = row_stats(d['sum_x'], d['sum_x2'], d['err'], x)
        b = np.array([base[f] for f in files])
        if row == 't999':
            one_step_c2 = dict(zip(files, s['c2']))
        c1 = np.array([one_step_c2[f] for f in files])           # H1 centre energy per cube (pct^2)
        ms = np.array([m1[f] for f in files])                    # B's stored ddim1 RMSE, mean of its 2 seeds

        def lam(mask):
            return centre_lambda(b[mask], s['c2'][mask], c1[mask])

        def lam_single(mask):
            return centre_lambda(b[mask], s['c2'][mask], ms[mask] ** 2)

        def slope(mask):
            return float(np.polyfit(b[mask] ** 2, s['spread'][mask], 1)[0])
        every = np.ones(len(files), bool)
        rec = dict(K=s['K'], draws=sorted(d['draws']), n=len(files), baseline_rmse=float(b.mean()),
                   ddim1_rmse=float(ms.mean()), single_rmse=_ms(s['single_rmse']), kmean_rmse=_ms(s['kmean_rmse']),
                   kmean_minus_single=_pair(s['kmean_rmse'] - s['single_rmse'], files),
                   mean_o=float(s['o'].mean()), mean_ebar=float(s['ebar'].mean()),
                   mean_spread=float(s['spread'].mean()), mean_c2=float(s['c2'].mean()),
                   mean_c2_one_step=float(c1.mean()),
                   pred_h0=float(np.mean(np.sqrt(np.clip(b ** 2 + (s['o'] - b ** 2) / s['K'], 0, None)))),
                   pred_h1=float(np.mean(np.sqrt(np.clip(c1 + (s['o'] - c1) / s['K'], 0, None)))),
                   calibration=dict(ratio_of_means=float(s['spread'].mean() / c1.mean()),
                                    self_ratio=float(s['spread'].mean() / s['c2'].mean()),
                                    per_cube_slope_on_b2=slope(every), per_cube_slope_se=jackknife_se(slope, files),
                                    reference_slope_c2_one_step_on_b2=float(np.polyfit(b ** 2, c1, 1)[0]),
                                    per_cube_corr=float(np.corrcoef(b ** 2, s['spread'])[0, 1])),
                   consistency_vs_stored=consistency(d, ev_b, row))
        if row != 't999':                                        # for t999 itself lambda is 1 by construction
            rec['lambda_vs_one_step_centre'] = dict(value=lam(every), se_cluster=jackknife_se(lam, files))
        rec['lambda_vs_ddim1_single_draw'] = dict(
            value=lam_single(every), se_cluster=jackknife_se(lam_single, files),
            note="reference only: a single ddim1 draw's RMSE^2 includes the one-step output's own noise energy, so a "
                 "chain centred on the one-step centre reads above 1; for the t999 row this value measures that bias")
        rec['per_cube'] = dict(files=files, single_rmse=s['single_rmse'].tolist(), kmean_rmse=s['kmean_rmse'].tolist(),
                               spread=s['spread'].tolist(), c2=s['c2'].tolist())
        out[row] = rec
        cal = rec['calibration']
        lam_txt = (f"lambda {rec['lambda_vs_one_step_centre']['value']:.3f} "
                   if 'lambda_vs_one_step_centre' in rec else '')
        print(f"{row}: K={s['K']} single {rec['single_rmse']['mean']:.4f} K-mean {rec['kmean_rmse']['mean']:.4f} "
              f"(pred H0 {rec['pred_h0']:.4f}, H1 {rec['pred_h1']:.4f}) {lam_txt}calibration {cal['ratio_of_means']:.3f} "
              f"slope {cal['per_cube_slope_on_b2']:.3f} (reference {cal['reference_slope_c2_one_step_on_b2']:.3f}) "
              f"consistency {rec['consistency_vs_stored']}", flush=True)
    result = dict(provenance=dict(inputs=[os.path.abspath(p) for p in args.inputs], shards=[m for _, m in loaded],
                                  b_eval=os.path.abspath(args.b_eval), git_head=git_head()), rows=out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(result, f, indent=1)
    os.replace(tmp, args.out)
    print('wrote', args.out)


def get_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('sample')
    s.add_argument('--checkpoint', default=B_CKPT)
    s.add_argument('--rows', nargs='+', choices=list(ROWS), required=True)
    s.add_argument('--draws', type=int, nargs='+', required=True)
    s.add_argument('--device', default='cuda:0')
    s.add_argument('--batch_size', type=int, default=HARNESS_BATCH,
                   help=f'{HARNESS_BATCH} reproduces the stored evaluation (seed per batch index); merge refuses others')
    s.add_argument('--base_channels', type=int, default=64)
    s.add_argument('--d', type=int, default=4)
    s.add_argument('--data_root', default=DATA_ROOT)
    s.add_argument('--prior', default=PRIOR)
    s.add_argument('--out', required=True)
    m = sub.add_parser('merge')
    m.add_argument('--inputs', nargs='+', required=True)
    m.add_argument('--b_eval', default=B_EVAL)
    m.add_argument('--d', type=int, default=4)
    m.add_argument('--data_root', default=DATA_ROOT)
    m.add_argument('--prior', default=PRIOR)
    m.add_argument('--out', required=True)
    return ap


def main(argv=None):
    args = get_parser().parse_args(argv)
    (cmd_sample if args.cmd == 'sample' else cmd_merge)(args)


if __name__ == '__main__':
    main()
