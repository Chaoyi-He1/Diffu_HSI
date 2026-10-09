"""Fit the per-pixel linear Gaussian prior used for the warm-start estimate.

Writes results/residual_warmstart/linear_prior_R1.npz with mu, C, K, s, R (and sigma_d / d_values, filled by
scripts/compute_residual_scale.py; an existing file's sigma_d table is kept unless --overwrite_sigma). The write
is atomic (temp file + os.replace). Fitted on pixels of the loader's TRAINING split only.
Run from the repo root:
    python scripts/fit_linear_prior.py [--n_files 3000] [--px_per_file 256] [--overwrite_sigma]
"""
import argparse
import os
import random
import sys
import tempfile

import numpy as np
import scipy.io as sio

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data_loader.HFD_dataset import HFD_data  # noqa: E402
from data_loader.linear_estimate import linear_inverse  # noqa: E402

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


def heldout_pixels(files, px_per_file, seed):
    """`px_per_file` random pixels (seeded) of each file: [len(files) * px_per_file, 31] on the [-1, 1] scale."""
    rng = np.random.default_rng(seed)
    out = []
    for f in files:
        x = load_cube(f).reshape(-1, 31)
        out.append(x[rng.choice(len(x), px_per_file, replace=False)])
    return np.concatenate(out)


def sensor_float32(X, R):
    """The sensor reading of spectra X [..., 31] as the network sees it: float32-rounded, returned as float64."""
    return (X @ R).astype(np.float32).astype(np.float64)


def score_s(prior, R, held, s):
    """Per-element RMSE ([-1, 1] scale) of the production estimator (clipped linear_inverse) on held-out spectra
    [N, 31], from float32 sensor values."""
    Xh = linear_inverse(sensor_float32(held, R), prior['mu'], make_gain(prior['C'], R, s), R)
    return float(np.sqrt(((Xh - held) ** 2).mean()))


def choose_s(prior, R, held, candidates=S_CANDIDATES):
    """Pick the regulariser with the lowest score_s on held-out spectra [N, 31]."""
    best, best_err = None, np.inf
    for s in candidates:
        err = score_s(prior, R, held, s)
        if err < best_err:
            best, best_err = s, err
    return best


def atomic_savez(path, **arrays):
    """np.savez to a temp file in the target's directory, then os.replace: a crash never leaves a half-written npz."""
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(path):
        mode = os.stat(path).st_mode & 0o777
    else:
        umask = os.umask(0); os.umask(umask)
        mode = 0o666 & ~umask
    fd, tmp = tempfile.mkstemp(dir=out_dir, prefix='.' + os.path.basename(path) + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as f:
            np.savez(f, **arrays)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def write_prior(path, arrays, overwrite_sigma=False):
    """Write the prior npz atomically. If `path` exists, its sigma_d / d_values table is carried over (unless
    overwrite_sigma) instead of the empty one in `arrays`. Returns the list of d values kept."""
    arrays = dict(arrays)
    kept = []
    if os.path.exists(path) and not overwrite_sigma:
        with np.load(path) as old:
            if 'd_values' in old.files and 'sigma_d' in old.files:
                arrays['d_values'], arrays['sigma_d'] = old['d_values'], old['sigma_d']
                kept = [int(v) for v in old['d_values']]
    atomic_savez(path, **arrays)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_files', type=int, default=3000)
    ap.add_argument('--px_per_file', type=int, default=256)
    ap.add_argument('--n_heldout_files', type=int, default=300)
    ap.add_argument('--heldout_px_per_file', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'linear_prior_R1.npz'))
    ap.add_argument('--overwrite_sigma', action='store_true',
                    help='drop the sigma_d table of an existing output file (default: keep it)')
    args = ap.parse_args()

    ds = HFD_data(data_path=DATA_ROOT, train_mode='image', eval_ratio=0.1, split='train',
                  data_format='image', type='Flower', R_n=1, sensor_down_sample_rate=1)
    R = np.asarray(ds.sensor_R_matrix, dtype=np.float64)              # [31, 30]
    files = list(ds.img_list)
    rng = random.Random(args.seed + 1)
    rng.shuffle(files)
    heldout, fit_files = files[:args.n_heldout_files], files[args.n_heldout_files:]

    prior = fit_prior(fit_files, R, args.n_files, args.px_per_file, args.seed)
    held_px = heldout_pixels(heldout, args.heldout_px_per_file, args.seed + 2)
    print('held-out RMSE ([-1,1] scale) per candidate s:',
          ', '.join(f'{c:g}: {score_s(prior, R, held_px, c):.6f}' for c in S_CANDIDATES))
    s = choose_s(prior, R, held_px)
    K = make_gain(prior['C'], R, s)
    err = score_s(prior, R, held_px, s)
    print(f'fit on {prior["n_fit_pixels"]} pixels of {prior["n_fit_files"]} files; s = {s:g}; '
          f'held-out per-element RMSE {err:.5f} ([-1,1] scale, clipped, float32 y) = {err / 2 * 100:.3f} % of range '
          f'({len(held_px)} random pixels of {len(heldout)} files)')

    kept = write_prior(args.out, dict(mu=prior['mu'], C=prior['C'], K=K, s=np.float64(s), R=R,
                                      sigma_d=np.zeros(0), d_values=np.zeros(0, dtype=int),
                                      n_fit_files=prior['n_fit_files'], n_fit_pixels=prior['n_fit_pixels'],
                                      seed=args.seed),
                       overwrite_sigma=args.overwrite_sigma)
    print('wrote', args.out)
    if kept:
        print(f'kept the existing sigma_d for d = {kept}; they belong to the previous gain, so re-run '
              f'scripts/compute_residual_scale.py --d_values {" ".join(map(str, kept))} if the fit changed')


if __name__ == '__main__':
    main()
