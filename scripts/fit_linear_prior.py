"""Fit the per-pixel linear Gaussian prior used for the warm-start estimate.

Writes results/residual_warmstart/linear_prior_R1.npz with mu, C, K, s, R (and sigma_d, filled by
scripts/compute_residual_scale.py in Task 3). Fitted on pixels of the loader's TRAINING split only.
Run from the repo root:
    python scripts/fit_linear_prior.py [--n_files 3000] [--px_per_file 256]
"""
import argparse
import os
import random
import sys

import numpy as np
import scipy.io as sio

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data_loader.HFD_dataset import HFD_data  # noqa: E402

DATA_ROOT = os.environ.get('HFD_DATA_ROOT', '/data/chaoyi_he/HSI/Diffu/dataset/HFD100 Mat dataset')
OUT_DIR = os.path.join(REPO, 'results', 'residual_warmstart')
S_CANDIDATES = [1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 1e-3]


def load_cube(path):
    """The loader's per-cube normalisation, on the 31 native bands: [H, W, 31] in [-1, 1]."""
    g = np.array(sio.loadmat(path)['truth'], dtype=np.float64)
    return (g - g.min()) / (g.max() - g.min() + 1e-20) * 2.0 - 1.0


def fit_prior(train_files, R, n_files=3000, px_per_file=256, seed=0):
    """Mean and covariance of 31-band spectra from random pixels of random training files."""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    files = rng.sample(list(train_files), min(n_files, len(train_files)))
    P = []
    for f in files:
        x = load_cube(f).reshape(-1, 31)
        P.append(x[nrng.choice(len(x), px_per_file, replace=False)])
    P = np.concatenate(P)
    return dict(mu=P.mean(0), C=np.cov(P.T), n_fit_files=len(files), n_fit_pixels=len(P), seed=seed)


def make_gain(C, R, s):
    """K = C R (R^T C R + s^2 I)^-1 via a whitened SVD, stable for the ill-conditioned R."""
    L = np.linalg.cholesky(C + 1e-12 * np.eye(C.shape[0]))
    U, S, Vt = np.linalg.svd(R.T @ L, full_matrices=False)       # R^T L = U S V^T
    filt = S / (S ** 2 + s ** 2) if s > 0 else np.where(S > S.max() * 1e-13, 1.0 / np.maximum(S, 1e-300), 0.0)
    return L @ Vt.T @ np.diag(filt) @ U.T                           # [31, 30]


def linear_estimate(Y, prior, K, R):
    """Y: [..., 30] -> [..., 31] on the [-1, 1] scale (not clipped)."""
    return prior['mu'] + (Y - prior['mu'] @ R) @ K.T


def choose_s(prior, R, heldout_pixels, candidates=S_CANDIDATES):
    """Pick the regulariser with the lowest per-element RMSE on held-out spectra [N, 31]."""
    Y = (heldout_pixels @ R).astype(np.float32).astype(np.float64)  # float32 like the loader's collate
    best, best_err = None, np.inf
    for s in candidates:
        Xh = linear_estimate(Y, prior, make_gain(prior['C'], R, s), R)
        err = np.sqrt(((Xh - heldout_pixels) ** 2).mean())
        if err < best_err:
            best, best_err = s, err
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_files', type=int, default=3000)
    ap.add_argument('--px_per_file', type=int, default=256)
    ap.add_argument('--n_heldout_files', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'linear_prior_R1.npz'))
    args = ap.parse_args()

    ds = HFD_data(data_path=DATA_ROOT, train_mode='image', eval_ratio=0.1, split='train',
                  data_format='image', type='Flower', R_n=1, sensor_down_sample_rate=1)
    R = np.asarray(ds.sensor_R_matrix, dtype=np.float64)              # [31, 30]
    files = list(ds.img_list)
    rng = random.Random(args.seed + 1)
    rng.shuffle(files)
    heldout, fit_files = files[:args.n_heldout_files], files[args.n_heldout_files:]

    prior = fit_prior(fit_files, R, args.n_files, args.px_per_file, args.seed)
    held_px = np.concatenate([load_cube(f).reshape(-1, 31)[::64] for f in heldout])
    s = choose_s(prior, R, held_px)
    K = make_gain(prior['C'], R, s)
    Xh = linear_estimate((held_px @ R).astype(np.float32).astype(np.float64), prior, K, R)
    err = np.sqrt(((Xh - held_px) ** 2).mean())
    print(f'fit on {prior["n_fit_pixels"]} pixels of {prior["n_fit_files"]} files; s = {s:g}; '
          f'held-out per-element RMSE {err:.5f} ([-1,1] scale) = {err / 2 * 100:.3f} % of range')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, mu=prior['mu'], C=prior['C'], K=K, s=np.float64(s), R=R,
             sigma_d=np.zeros(0), d_values=np.zeros(0, dtype=int),
             n_fit_files=prior['n_fit_files'], n_fit_pixels=prior['n_fit_pixels'], seed=args.seed)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
