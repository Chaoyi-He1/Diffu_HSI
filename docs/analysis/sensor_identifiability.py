"""
Numerical facts behind docs/why_diffusion_recovers_hsi.md.

Reproduces, with the exact preprocessing of the data loaders:
  * singular values / effective rank / condition number of the sensor matrix R^T
  * intrinsic (PCA) dimension of the spectra and the restricted conditioning of R^T
    on the top-k principal subspace (identifiability on the spectral manifold)
  * noise amplification of the pseudo-inverse vs. a Gaussian-prior MMSE estimate
  * distance of R^T and of the spectral covariance to the nearest Toeplitz matrix
    (how far the problem is from being shift-invariant along wavelength)
  * what a stride-2 sensor grid (--ds 2) costs if filled in by bilinear interpolation

Run from the repo root with the `hsi` environment:
    python docs/analysis/sensor_identifiability.py
"""
import glob
import os
import random
import re
import sys

import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
np.set_printoptions(precision=4, suppress=True, linewidth=140)

HFD_ROOT = 'dataset/HFD100 Mat dataset'
HASCID_ROOT = 'dataset/HASCID-Dataset'
HFD_WL = np.linspace(451, 855, 31)
N_SENSOR = 30


def load_R(root, n, wl):
    """Same processing as HFD_data / HASCID_data: pick 30 columns uniformly, resample to the
    data wavelengths, min-max normalise each column."""
    if n is None:
        R = np.array(sio.loadmat(os.path.join(root, 'PH5_interp_results.mat'))['PH5_interp'])
        swl = np.arange(400, 1560, 10)
    else:
        R = np.array(sio.loadmat(os.path.join(root, f'R_Device{n}.mat'))['R'])
        swl = np.linspace(400, 1000, R.shape[0])
    idx = np.linspace(0, R.shape[1] - 1, N_SENSOR, dtype=int)
    R = R[:, idx]
    R = interp1d(swl, R, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')(wl)
    return (R - R.min(0)) / (R.max(0) - R.min(0) + 1e-20)  # [L, N]


def toeplitz_distance(M):
    """||M - P(M)||_F / ||M||_F where P averages along diagonals = projection onto Toeplitz
    matrices = Reynolds average over simultaneous row/column shifts."""
    T = np.zeros_like(M)
    m, n = M.shape
    for d in range(-m + 1, n):
        idx = np.where(np.eye(m, n, k=d) == 1)
        T[idx] = M[idx].mean()
    return np.linalg.norm(M - T) / np.linalg.norm(M)


def effective_rank(s, tau):
    return int((s / s[0] > tau).sum())


def sample_hfd(n_files=300, px_per_file=400, seed=0):
    files = sorted(glob.glob(os.path.join(HFD_ROOT, 'MatFlower60', 'Train', '*', '*.mat')))
    random.seed(seed)
    np.random.seed(seed)
    files = random.sample(files[:10000], n_files)  # same 10k subset as the loader
    X, cubes = [], []
    for f in files:
        g = np.array(sio.loadmat(f)['truth'], dtype=np.float64)
        g = (g - g.min()) / (g.max() - g.min() + 1e-20) * 2 - 1  # per-image [-1, 1] as in loader
        cubes.append(g)
        P = g.reshape(-1, g.shape[2])
        X.append(P[np.random.choice(len(P), px_per_file, replace=False)])
    return np.concatenate(X), cubes


def report_sensor(name, R, C, V):
    Rt = R.T
    s = np.linalg.svd(Rt, compute_uv=False)
    print(f'\n--- {name}: R^T is {Rt.shape}, rank {np.linalg.matrix_rank(Rt)}, '
          f'cond {s[0] / s[-1]:.2e}, null dim {Rt.shape[1] - np.linalg.matrix_rank(Rt)}')
    print('  singular values:', s)
    print(f'  effective rank (sigma_i/sigma_1 > tau): tau=1e-2 -> {effective_rank(s, 1e-2)}, '
          f'1e-3 -> {effective_rank(s, 1e-3)}, 5e-4 (fp16 eps) -> {effective_rank(s, 5e-4)}')
    print('  restricted conditioning sigma_min/sigma_max of R^T V_k on top-k PCA subspace:')
    print('   ' + ' | '.join(f'k={k}: {(lambda sv: sv[-1] / sv[0])(np.linalg.svd(Rt @ V[:, :k], compute_uv=False)):.1e}'
                             for k in [3, 5, 8, 12, 19, 29] if k <= V.shape[1]))
    for tau in [1e-2, 1e-3]:
        m = effective_rank(s, tau)
        Vt = np.linalg.svd(Rt)[2]
        Pm = Vt[:m].T @ Vt[:m]
        print(f'  fraction of spectral variance inside the {m} well-measured directions (tau={tau}): '
              f'{np.trace(Pm @ C) / np.trace(C):.4f}')
    print(f'  relative distance of R^T to nearest Toeplitz matrix: {toeplitz_distance(Rt):.3f}')


def main():
    print('=' * 80)
    print('HFD100 (MatFlower60/Train), 31 bands 451-855 nm, 30 sensor channels')
    print('=' * 80)
    X, cubes = sample_hfd()
    mu = X.mean(0)
    C = np.cov((X - mu).T)
    ev, V = np.linalg.eigh(C)
    ev, V = ev[::-1], V[:, ::-1]
    cum = np.cumsum(ev) / ev.sum()
    print(f'sampled pixels: {X.shape}')
    print('PCA cumulative variance (first 12):', cum[:12])
    for th in [0.99, 0.999, 0.9999]:
        print(f'  k for {th * 100:.2f}% variance: {int(np.searchsorted(cum, th)) + 1}')
    print(f'spectral covariance: relative distance to nearest Toeplitz matrix {toeplitz_distance(C):.3f}')
    print(f'reference: i.i.d. Gaussian 30x31 matrix is {toeplitz_distance(np.random.randn(30, 31)):.3f} from Toeplitz')

    Rs = {'R_Device1': load_R(HFD_ROOT, 1, HFD_WL),
          'R_Device2': load_R(HFD_ROOT, 2, HFD_WL),
          'PH5': load_R(HFD_ROOT, None, HFD_WL)}
    for name, R in Rs.items():
        report_sensor(name, R, C, V)

    # --- linear inversion vs. Gaussian-prior MMSE under small measurement noise (R_Device1)
    R = Rs['R_Device1']
    Rt = R.T
    Y = X @ R
    pinv = np.linalg.pinv(Rt)
    rel = lambda A: (np.linalg.norm(A - X, axis=1) / np.linalg.norm(X, axis=1)).mean()
    print('\n--- Linear inversion with R_Device1 (relative L2 error per pixel)')
    print(f'  noiseless pseudo-inverse: {rel((pinv @ Y.T).T):.4f}  (= energy in the 1-dim null space)')
    for snr_db in [60, 40, 30]:
        sig = np.sqrt((Y ** 2).mean()) * 10 ** (-snr_db / 20)
        Yn = Y + sig * np.random.randn(*Y.shape)
        K = C @ R @ np.linalg.inv(Rt @ C @ R + sig ** 2 * np.eye(N_SENSOR))
        Xm = mu + (K @ (Yn - mu @ R).T).T
        print(f'  SNR {snr_db} dB: pseudo-inverse {rel((pinv @ Yn.T).T):.3g} | Gaussian-prior MMSE {rel(Xm):.4f}')

    # --- what --ds 2 removes
    import torch
    import torch.nn.functional as F
    errs, c1, c2 = [], [], []
    for g in cubes[:200]:
        t = torch.tensor(g).permute(2, 0, 1)[None]
        up = F.interpolate(t[:, :, ::2, ::2], size=t.shape[-2:], mode='bilinear', align_corners=False)
        errs.append(((up - t).norm() / t.norm()).item())
        gc = g - g.mean((0, 1), keepdims=True)
        c1.append((gc[:, 1:] * gc[:, :-1]).sum() / np.sqrt((gc[:, 1:] ** 2).sum() * (gc[:, :-1] ** 2).sum()))
        c2.append((gc[:, 2:] * gc[:, :-2]).sum() / np.sqrt((gc[:, 2:] ** 2).sum() * (gc[:, :-2] ** 2).sum()))
    print('\n--- Spatial structure')
    print(f'  spatial autocorrelation: lag 1 = {np.mean(c1):.3f}, lag 2 = {np.mean(c2):.3f}')
    print(f'  rel. error of bilinear up-sampling of a stride-2 subsampled cube: {np.mean(errs):.4f}')
    print(f'\nE|eps| for eps ~ N(0,1) (trivial eps-L1 baseline): {np.sqrt(2 / np.pi):.4f}')

    # --- HASCID
    print('\n' + '=' * 80)
    print('HASCID, 204 bands (first 160 reconstructed), 30 sensor channels')
    print('=' * 80)
    from data_loader.my_dataset import HASCID_data
    import inspect
    src = inspect.getsource(HASCID_data.__init__)
    wl_h = np.array([float(v) for v in re.search(r'np\.array\(\[(.*?)\]\)', src, re.S).group(1).split(',')])
    gts = sorted(glob.glob(os.path.join(HASCID_ROOT, 'gt_files', 'gtRef_*.npy')))
    Xh = []
    for f in random.sample(gts, 40):
        g = np.load(f).astype(np.float64)
        P = g.reshape(-1, g.shape[2])
        Xh.append(P[np.random.choice(len(P), 1500, replace=False)])
    Xh = np.concatenate(Xh)
    Rh = load_R(HASCID_ROOT, 1, wl_h)
    X160 = (Xh[:, :160] - 0.5) * 2
    Ch = np.cov((X160 - X160.mean(0)).T)
    eh, Vh = np.linalg.eigh(Ch)
    eh, Vh = eh[::-1], Vh[:, ::-1]
    cum = np.cumsum(eh) / eh.sum()
    print(f'sampled pixels: {Xh.shape}; PCA k99={np.searchsorted(cum, .99) + 1}, '
          f'k99.9={np.searchsorted(cum, .999) + 1}, k99.99={np.searchsorted(cum, .9999) + 1}')
    print(f'spectral covariance (160 bands): distance to Toeplitz {toeplitz_distance(Ch):.3f}')
    Y = Xh @ Rh
    Ynuis = Xh[:, 160:] @ Rh[160:]
    print(f'fraction of measurement energy contributed by bands 161-204 (not reconstructed): '
          f'{(Ynuis ** 2).sum() / (Y ** 2).sum():.3f}')
    report_sensor('R_Device1 restricted to the 160 target bands', Rh[:160], Ch, Vh)


if __name__ == '__main__':
    main()
