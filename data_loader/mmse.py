"""Noise-aware Gaussian MMSE estimates of the 31-band spectrum from the (noisy) sensor reading.

Posterior maths is on the physical x01 = (x + 1) / 2 scale; mmse_maps / demosaic_then_mmse return the
64-band mean and std on the trainer [-1, 1] scale. numpy / scipy only.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

TILE = (5, 6)


def prior01(prior):
    """Trainer-scale prior (mu, C) -> physical x01 scale."""
    return (np.asarray(prior['mu'], np.float64) + 1.0) / 2.0, np.asarray(prior['C'], np.float64) / 4.0


def posterior_sequential(y, R, mu01, C01, noise_var):
    """y [N, S]; noise_var float or [S] per-channel variance. Returns xhat [N, L], Sigma [L, L]."""
    R = np.asarray(R, np.float64); C = np.asarray(C01, np.float64)
    nv = np.broadcast_to(np.asarray(noise_var, np.float64), (R.shape[1],))
    Lc = np.linalg.cholesky(C + 1e-12 * np.eye(C.shape[0]))
    D = 1.0 / np.sqrt(nv)
    M = D[:, None] * (R.T @ Lc)                                   # [S, L]
    # the thin SVD with the "+1" in S / (S^2 + 1) is the whitened gain only when S <= L: for S > L the complement of U
    # (directions with singular value 0, gain 0 but not dropped from U^T) would be missing and the gain wrong
    assert M.shape[0] <= M.shape[1], \
        f'posterior_sequential needs channels S <= bands L: R is {R.shape} (L, S), whitened M is {M.shape} (S, L)'
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    K = Lc @ Vt.T @ np.diag(S / (S ** 2 + 1.0)) @ U.T @ np.diag(D)   # [L, S]
    y = np.asarray(y, np.float64)
    xhat = mu01 + (y - mu01 @ R) @ K.T
    Sigma = C - K @ (R.T @ C)
    return xhat, 0.5 * (Sigma + Sigma.T)


def posterior_snapshot(y_k, k, R, mu01, C01, noise_var):
    """One measured channel per pixel. y_k [N], k [N] int. Returns xhat [N, L], Sigma_n [N, L, L]."""
    R = np.asarray(R, np.float64); C = np.asarray(C01, np.float64)
    k = np.asarray(k).astype(int)
    r = R[:, k].T                                                  # [N, L]
    Cr = r @ C                                                     # [N, L] (C symmetric)
    g = Cr / ((Cr * r).sum(1) + noise_var)[:, None]                # [N, L]
    resid = np.asarray(y_k, np.float64) - r @ mu01
    xhat = mu01 + g * resid[:, None]
    Sigma = C[None] - np.einsum('ni,nj->nij', g, Cr)
    return xhat, Sigma


def posterior_snapshot_lowrank(y_k, k, R, mu01, C01, noise_var):
    """Low-rank form of posterior_snapshot. Returns xhat [N, L], Cr [N, L], den [N] with
    Sigma_n = C - Cr Cr^T / den (never materialised)."""
    R = np.asarray(R, np.float64); C = np.asarray(C01, np.float64)
    r = R[:, np.asarray(k).astype(int)].T
    Cr = r @ C
    den = (Cr * r).sum(1) + noise_var
    resid = np.asarray(y_k, np.float64) - r @ mu01
    return mu01 + (Cr / den[:, None]) * resid[:, None], Cr, den


def lift_mean_std_lowrank(xhat31, C, Cr, den, A):
    """Mean / std on the 64-band grid for Sigma_n = C - Cr Cr^T / den: var = diag(A C A^T) - (A Cr)^2 / den."""
    ACr = Cr @ A.T                                                 # [N, M]
    base = np.einsum('ij,jk,ik->i', A, np.asarray(C, np.float64), A)
    var = base[None, :] - ACr ** 2 / den[:, None]
    return xhat31 @ A.T, np.sqrt(np.maximum(var, 0.0))


def lift_mean_std(xhat31, Sigma, A):
    """xhat31 [N, L]; Sigma [L, L] or [N, L, L]; A [M, L]. Returns xhat64, std64 [N, M]."""
    Sigma = np.asarray(Sigma)
    if Sigma.ndim == 2:
        var = np.einsum('ij,jk,ik->i', A, Sigma, A)[None, :]
        var = np.broadcast_to(var, (xhat31.shape[0], A.shape[0]))
    else:
        var = np.einsum('ij,njk,ik->ni', A, Sigma, A)
    return xhat31 @ A.T, np.sqrt(np.maximum(var, 0.0))


def _to_trainer(xhat64, std64, shape):
    return ((2.0 * xhat64 - 1.0).reshape(shape).astype(np.float32),
            (2.0 * std64).reshape(shape).astype(np.float32))


def mmse_maps(y, mask, R, prior, sigma, A):
    """y [H, W, 30]. mask None / all-ones -> sequential; one-hot -> snapshot. Returns [H, W, M] float32 x2 (trainer scale)."""
    H, W, S = y.shape
    mu01, C01 = prior01(prior)
    nv = float(sigma) ** 2
    shape = (H, W, A.shape[0])
    if mask is None or np.all(mask == 1):
        xh, Sig = posterior_sequential(y.reshape(-1, S), R, mu01, C01, nv)
    else:
        m = np.asarray(mask).reshape(-1, S)
        k = m.argmax(1)
        xh, Cr, den = posterior_snapshot_lowrank(y.reshape(-1, S)[np.arange(len(k)), k], k, R, mu01, C01, nv)
        return _to_trainer(*lift_mean_std_lowrank(xh, C01, Cr, den, A), shape)
    x64, s64 = lift_mean_std(xh, Sig, A)
    return _to_trainer(x64, s64, shape)


def demosaic(y_m, mask):
    """Per-channel linear interpolation of the mosaic-measured lattice onto the full grid. -> [H, W, S]."""
    H, W, S = y_m.shape
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    pts = np.stack([ii.ravel(), jj.ravel()], -1)
    out = np.empty((H, W, S), np.float64)
    for k in range(S):
        rows = np.arange(k // TILE[1], H, TILE[0])
        cols = np.arange(k % TILE[1], W, TILE[1])
        vals = np.asarray(y_m)[rows[:, None], cols[None, :], k]
        f = RegularGridInterpolator((rows, cols), vals, method='linear', bounds_error=False, fill_value=None)
        out[..., k] = f(pts).reshape(H, W)
    return out


def demosaic_then_mmse(y_m, mask, R, prior, sigma, A, e2):
    """Sequential posterior of the demosaiced y with per-channel noise_var = sigma^2 + e2."""
    H, W, S = y_m.shape
    mu01, C01 = prior01(prior)
    d = demosaic(y_m, mask)
    xh, Sig = posterior_sequential(d.reshape(-1, S), R, mu01, C01, float(sigma) ** 2 + np.asarray(e2, np.float64))
    x64, s64 = lift_mean_std(xh, Sig, A)
    return _to_trainer(x64, s64, (H, W, A.shape[0]))
