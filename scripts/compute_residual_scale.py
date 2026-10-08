"""Compute sigma_d = RMS of (x - x0_hat) over training files for d in {1, 2, 4, 8} and store it in the prior.
Run from the repo root after scripts/fit_linear_prior.py:
    python scripts/compute_residual_scale.py [--n_files 500]
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data_loader.hfd_residual import HFDResidualData  # noqa: E402

DATA_ROOT = os.environ.get('HFD_DATA_ROOT', '/data/chaoyi_he/HSI/Diffu/dataset/HFD100 Mat dataset')
PRIOR = os.path.join(REPO, 'results', 'residual_warmstart', 'linear_prior_R1.npz')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_files', type=int, default=500)
    ap.add_argument('--d_values', type=int, nargs='+', default=[1, 2, 4, 8])
    ap.add_argument('--prior', default=PRIOR)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    sig = []
    for d in args.d_values:
        ds = HFDResidualData(data_path=DATA_ROOT, split='train', sensor_down_sample_rate=d, prior_path=args.prior)
        idx = rng.choice(len(ds), min(args.n_files, len(ds)), replace=False)
        ss, n = 0.0, 0
        for i in idx:
            x, _, xh = ds[int(i)]
            ss += float(((x - xh) ** 2).sum()); n += x.size
        sig.append(np.sqrt(ss / n))
        print(f'd={d}: sigma_d = {sig[-1]:.5f} ([-1,1] scale) = {sig[-1] / 2 * 100:.3f} % of range RMSE of the estimate')
    z = dict(np.load(args.prior))
    z['sigma_d'] = np.array(sig); z['d_values'] = np.array(args.d_values, dtype=int)
    np.savez(args.prior, **z)
    print('updated', args.prior)


if __name__ == '__main__':
    main()
