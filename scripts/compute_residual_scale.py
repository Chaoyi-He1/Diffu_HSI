"""Compute sigma_d = RMS of (x - x0_hat) over training files for each requested d and store it in the prior.
Run from the repo root after scripts/fit_linear_prior.py:
    python scripts/compute_residual_scale.py [--d_values 1 2 4 8] [--n_files 500] [--seed 0]

Each d draws its own files from a generator seeded by (seed, d), so sigma_d does not depend on which other d are
requested or in which order. The new (d, sigma_d) pairs are merged into the prior's table (an existing d is
replaced, the others are kept, sorted by d) and the npz is rewritten atomically.
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data_loader.hfd_residual import HFDResidualData  # noqa: E402
from scripts.fit_linear_prior import atomic_savez  # noqa: E402

DATA_ROOT = os.environ.get('HFD_DATA_ROOT', '/data/chaoyi_he/HSI/Diffu/dataset/HFD100 Mat dataset')
PRIOR = os.path.join(REPO, 'results', 'residual_warmstart', 'linear_prior_R1.npz')


def sample_indices(n_total, n_files, d, seed=0):
    """The training-file indices used for sigma_d at this d: a function of (seed, d) only."""
    rng = np.random.default_rng(seed * 1000 + d)
    return rng.choice(n_total, min(n_files, n_total), replace=False)


def merge_sigma(d_values, sigma_d, new_pairs):
    """Merge (d, sigma) pairs into the table: replace an existing d, keep the others, sort by d."""
    d_values, sigma_d = np.asarray(d_values), np.asarray(sigma_d)
    if d_values.shape != sigma_d.shape:
        raise ValueError(f'd_values {d_values.shape} and sigma_d {sigma_d.shape} do not match')
    table = {int(d): float(s) for d, s in zip(d_values, sigma_d)}
    table.update({int(d): float(s) for d, s in new_pairs})
    ds = sorted(table)
    return np.array(ds, dtype=int), np.array([table[d] for d in ds], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_files', type=int, default=500)
    ap.add_argument('--d_values', type=int, nargs='+', default=[1, 2, 4, 8])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--prior', default=PRIOR)
    args = ap.parse_args()
    new = []
    for d in args.d_values:
        ds = HFDResidualData(data_path=DATA_ROOT, split='train', sensor_down_sample_rate=d, prior_path=args.prior,
                             require_sigma_d=False)
        ss, n = 0.0, 0
        for i in sample_indices(len(ds), args.n_files, d, args.seed):
            x, _, xh = ds[int(i)]
            ss += float(((x - xh) ** 2).sum()); n += x.size
        new.append((d, float(np.sqrt(ss / n))))
        print(f'd={d}: sigma_d = {new[-1][1]:.5f} ([-1,1] scale) = {new[-1][1] / 2 * 100:.3f} % of range RMSE of the estimate')
    with np.load(args.prior) as f:
        z = {k: f[k] for k in f.files}
    z['d_values'], z['sigma_d'] = merge_sigma(z['d_values'], z['sigma_d'], new)
    atomic_savez(args.prior, **z)
    print('updated', args.prior, '; sigma_d table:', dict(zip(z['d_values'].tolist(), z['sigma_d'].round(6).tolist())))


if __name__ == '__main__':
    main()
