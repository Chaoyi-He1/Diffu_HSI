"""Build the regressor-centre cache of one stride (spec §4.3; layout in data_loader/xr_cache.py).

    python scripts/uq/build_xr_cache.py --d 8 --half1 runs/regress_l1_half1_d8_b64/checkpoint_epoch_E.pth \
        --half2 runs/regress_l1_half2_d8_b64/checkpoint_epoch_E.pth --full runs/regress_l1_sel_d8_b64/checkpoint_epoch_11.pth

Training cubes of half 1 are centred by the half-2 regressor and vice versa (cross-fitting); every test-split cube
(monitoring + gate) is centred by the full regressor (decision D3). sigma_r = RMS of x - x_r over all cross-fitted
training cubes (float32, before the float16 cast; the definition of scripts/compute_residual_scale.py); sigma_full =
the same over the monitoring files only. A sigma_full / sigma_r ratio outside [0.9, 1.1] sets sigma_flag (reported,
not fatal). Each half regressor's meta must name the split read here (split_sha256), every regressor must have the
sigma_d of the dataset it centres and, when --prior exists, that prior's sha256 (prior_sha256). The split must put
every training file in exactly one half. All of this is checked before the first cube is written. Refuses an
existing cache.
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
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from data_loader.hfd_residual import HFDResidualData, limit_worker_threads, residual_collate_fn  # noqa: E402
from data_loader.warmstart_paths import DATA_ROOT, PRIOR, SPLIT_JSON, SUBSET, UQ_DIR, file_sha256, git_head  # noqa: E402
from data_loader.xr_cache import META_NAME, save_xr, write_meta  # noqa: E402
from model.residual_wrapper import build_residual_model, compose, regress_forward  # noqa: E402
from scripts.uq.split_halves import read_split  # noqa: E402


def ckpt_info(path, role, want_half, d, split_sha=None):
    """Load a regressor checkpoint and check it fits its role; split_sha (the sha256 of the split being read) is
    required of a half regressor's meta, so its half is the half this cache treats as unseen."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    meta = (ck.get('meta') if isinstance(ck, dict) else None) or {}
    bad = []
    if meta.get('mode') != 'regress':
        bad.append(f"mode {meta.get('mode')!r} (need 'regress')")
    if int(meta.get('d', -1)) != int(d):
        bad.append(f"d={meta.get('d')!r} (need d={d})")
    th = int((meta.get('args') or {}).get('train_half', meta.get('train_half', 0)) or 0)
    if th != want_half:
        bad.append(f'train_half {th} (a {role} regressor must have train_half {want_half})')
    if want_half and split_sha is not None and meta.get('split_sha256') != split_sha:
        bad.append(f"split_sha256 {meta.get('split_sha256')!r} (a {role} regressor must be trained on the split read "
                   f"here, sha256 {split_sha})")
    if bad:
        raise ValueError(f'{role} checkpoint {path}: ' + '; '.join(bad))
    m = re.search(r'checkpoint_epoch_(\d+)\.pth$', path)
    info = dict(path=os.path.abspath(path), run=os.path.basename(os.path.dirname(os.path.abspath(path))),
                epoch=int(m.group(1)) if m else None, sha256=file_sha256(path), train_half=int(want_half),
                base_channels=int(meta['base_channels']), sigma_d=float(meta['sigma_d']),
                prior_sha256=meta.get('prior_sha256'))
    return ck, info


def load_regressor(ck, base_channels, device):
    model = build_residual_model('regress', base_channels)
    model.load_state_dict(ck['model_state_dict'], strict=True)
    return model.to(device).eval()


def half_of(relpath, half1, half2):
    if relpath in half1:
        return 1
    if relpath in half2:
        return 2
    raise KeyError(f'{relpath} is in neither half of the split')


@torch.no_grad()
def centre_batch(model, y, xh, sigma_d):
    return compose(xh, sigma_d, regress_forward(model, {'sensor': y, 'x0_hat': xh}))


def gate_relpaths(data_root):
    with open(SUBSET) as f:
        return set(json.load(f)['files'])


def _loader(ds, args):
    return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                       collate_fn=residual_collate_fn, worker_init_fn=limit_worker_threads)


def run(args, make_ds=None):
    if os.path.exists(os.path.join(args.out, META_NAME)):
        raise FileExistsError(f'{args.out} already holds a cache; delete the directory to rebuild')
    device = torch.device(args.device)
    split = read_split(args.split_json)
    split_sha = file_sha256(args.split_json)
    half1, half2 = set(split['half1']), set(split['half2'])
    both = half1 & half2
    if both:
        raise ValueError(f'split {args.split_json}: {len(both)} files are in both halves, e.g. {sorted(both)[0]}')
    (ck1, i1), (ck2, i2), (ckf, i_f) = (ckpt_info(args.half1, 'half1', 1, args.d, split_sha),
                                        ckpt_info(args.half2, 'half2', 2, args.d, split_sha),
                                        ckpt_info(args.full, 'full', 0, args.d))
    infos = dict(half1=i1, half2=i2, full=i_f)
    prior_sha = file_sha256(args.prior) if os.path.exists(args.prior) else None     # the stub tests pass no prior file
    if prior_sha is not None:
        bad = [f"{role} checkpoint {info['path']}: prior_sha256 {info['prior_sha256']!r}" for role, info in infos.items()
               if info['prior_sha256'] != prior_sha]
        if bad:
            raise ValueError(f'--prior {args.prior} has sha256 {prior_sha}, but ' + '; '.join(bad))
    if len({i1['base_channels'], i2['base_channels'], i_f['base_channels']}) != 1:
        raise ValueError('the three regressors differ in base_channels')
    gate = gate_relpaths(args.data_root)
    if make_ds is None:
        def make_ds(split_name):
            """Production (limit_files 0): all 9 000 training cubes and all 1 000 test-split cubes. With --limit_files N
            (smoke tests): train[:N], and for val the first max(8, N // 8) NON-gate files -- the trainer's own order
            (make_datasets drops the gate cubes before applying its limit), so the two agree on which files exist."""
            ds = HFDResidualData(data_path=args.data_root, split='train' if split_name == 'train' else 'test',
                                 sensor_down_sample_rate=args.d, prior_path=args.prior)
            if args.limit_files:
                if split_name == 'train':
                    ds.img_list = ds.img_list[:args.limit_files]
                else:
                    gate_abs = {os.path.normpath(os.path.join(args.data_root, f)) for f in gate}
                    ds.img_list = [f for f in ds.img_list if os.path.normpath(f) not in gate_abs][:max(8, args.limit_files // 8)]
            return ds
    dss = {split_name: make_ds(split_name) for split_name in ('train', 'val')}   # both checked before the first write
    for split_name, roles in (('train', ('half1', 'half2')), ('val', ('full',))):
        for role in roles:
            if abs(infos[role]['sigma_d'] - float(dss[split_name].sigma_d)) >= 1e-9:
                raise ValueError(f"{role} checkpoint {infos[role]['path']}: sigma_d {infos[role]['sigma_d']!r}, but the "
                                 f"{split_name} dataset (--prior {args.prior}, d={args.d}) has sigma_d {float(dss[split_name].sigma_d)!r}")
    rels = {split_name: [os.path.relpath(f, args.data_root) for f in ds.img_list] for split_name, ds in dss.items()}
    missing = [f for f in rels['train'] if f not in half1 and f not in half2]
    if missing:
        raise ValueError(f"split {args.split_json}: {len(missing)} of {len(rels['train'])} training files are in neither "
                         f'half, e.g. {missing[0]}')
    nets = {1: load_regressor(ck1, i1['base_channels'], device), 2: load_regressor(ck2, i2['base_channels'], device),
            0: load_regressor(ckf, i_f['base_channels'], device)}
    stats, counts, t0 = {}, {}, time.time()
    for split_name, ds in dss.items():
        sigma_d = float(ds.sigma_d)
        rel = rels[split_name]
        ss, n, n_sigma = 0.0, 0, 0
        pos = 0
        for x, y, xh in _loader(ds, args):
            B = x.shape[0]
            files = rel[pos:pos + B]
            pos += B
            x, y, xh = x.to(device), y.to(device), xh.to(device)
            if split_name == 'train':
                which = torch.tensor([3 - half_of(f, half1, half2) for f in files], device=device)   # H1 -> net 2, H2 -> net 1
                x_r = torch.empty_like(xh)
                for k in (1, 2):
                    sel = which == k
                    if sel.any():
                        x_r[sel] = centre_batch(nets[k], y[sel], xh[sel], sigma_d)
                use = torch.ones(B, dtype=torch.bool, device=device)
            else:
                x_r = centre_batch(nets[0], y, xh, sigma_d)
                use = torch.tensor([f not in gate for f in files], device=device)      # sigma_full: monitoring files only
            if use.any():
                ss += float(((x[use] - x_r[use]).double() ** 2).sum()); n += int(x[use].numel())
                n_sigma += int(use.sum())
            for b, f in enumerate(files):
                save_xr(args.out, split_name, f, x_r[b].permute(1, 2, 0).cpu().numpy())
        stats[split_name] = float(np.sqrt(ss / max(n, 1)))
        counts[split_name] = (len(rel), n_sigma)
        print(f'{split_name}: {len(rel)} cubes cached ({time.time() - t0:.0f} s), RMS residual {stats[split_name]:.6f} over {n_sigma} cubes', flush=True)
    ratio = stats['val'] / stats['train']
    meta = dict(d=int(args.d), split_path=os.path.abspath(args.split_json), split_sha256=split_sha,
                prior_path=os.path.abspath(args.prior), prior_sha256=prior_sha, half1=i1, half2=i2, full=i_f, sigma_r=stats['train'], sigma_full=stats['val'], sigma_ratio=ratio,
                sigma_flag=bool(not 0.9 <= ratio <= 1.1), n_train=counts['train'][0], n_val=counts['val'][0],
                n_val_monitoring=counts['val'][1], dtype='float16', git_head=git_head(), created=time.strftime('%Y-%m-%dT%H:%M:%S'),
                limit_files=int(args.limit_files))
    write_meta(args.out, meta)
    print(f"wrote {os.path.join(args.out, META_NAME)}: sigma_r {meta['sigma_r']:.6f}, sigma_full {meta['sigma_full']:.6f}, "
          f"ratio {ratio:.4f}{' FLAG (outside [0.9, 1.1])' if meta['sigma_flag'] else ''}")
    return meta


def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, required=True)
    ap.add_argument('--half1', required=True); ap.add_argument('--half2', required=True); ap.add_argument('--full', required=True)
    ap.add_argument('--out', default=None, help='default: UQ_DIR/d<d>/xr_cache')
    ap.add_argument('--split_json', default=SPLIT_JSON)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch_size', type=int, default=16); ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--limit_files', type=int, default=0, help='smoke tests: the first N training files and max(8, N // 8) test-split files')
    ap.add_argument('--data_root', default=DATA_ROOT); ap.add_argument('--prior', default=PRIOR)
    return ap


if __name__ == '__main__':
    a = get_parser().parse_args()
    a.out = a.out or os.path.join(UQ_DIR, f'd{a.d}', 'xr_cache')
    run(a)
