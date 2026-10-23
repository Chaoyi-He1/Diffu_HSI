"""Uncertainty and texture metrics of Phase 2a (spec §5.3-5.4), numpy, per element or per cube.

Inputs are on the [0, 1] display scale (to01 converts the trainer's [-1, 1]); `samples` are [K, ...] with the sample
axis first; everything else broadcasts over the trailing element axes.
"""
import math

import numpy as np
import torch

FRACTIONS = np.arange(0, 1.0, 0.02)                                  # removal fractions of the sparsification curve
Z = {0.5: 0.6744897501960817, 0.9: 1.6448536269514722}             # central-interval half-widths of N(0, 1)


def to01(t):
    a = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    return (a.astype(np.float64) + 1.0) / 2.0


def crps_samples(samples, x):
    """Fair (unbiased) sample CRPS: mean_k |s_k - x| - sum_{j<k} |s_j - s_k| / (K (K - 1)); the pairwise sum is
    sum_i (2 i - K + 1) s_(i) over the ascending order statistics."""
    s = np.sort(np.asarray(samples, dtype=np.float64), axis=0)
    K = s.shape[0]
    if K < 2:
        raise ValueError(f'crps_samples needs K >= 2 samples, got K={K}')
    x = np.asarray(x, dtype=np.float64)
    term1 = np.abs(s - x[None]).mean(0)
    w = (2.0 * np.arange(K) - K + 1.0)
    term2 = np.tensordot(w, s, axes=(0, 0)) / (K * (K - 1))
    return term1 - term2


def crps_gaussian(mu, sigma, x):
    mu, x = np.asarray(mu, dtype=np.float64), np.asarray(x, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
    z = (x - mu) / sigma
    Phi = 0.5 * (1.0 + torch.erf(torch.from_numpy(z / math.sqrt(2.0))).numpy())
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def coverage_samples(samples, x, levels=(0.5, 0.9)):
    s, x = np.asarray(samples, dtype=np.float64), np.asarray(x, dtype=np.float64)
    out = {}
    for lv in levels:
        lo, hi = np.quantile(s, [(1.0 - lv) / 2.0, 1.0 - (1.0 - lv) / 2.0], axis=0, method='linear')
        out[lv] = float(np.mean((x >= lo) & (x <= hi)))
    return out


def coverage_gaussian(mu, sigma, x, levels=(0.5, 0.9)):
    mu, sigma, x = (np.asarray(a, dtype=np.float64) for a in (mu, sigma, x))
    return {lv: float(np.mean(np.abs(x - mu) <= Z[lv] * sigma)) for lv in levels}


def sparsification_curve(score, err2, fractions=FRACTIONS):
    """RMSE of the elements left after removing the top-f fraction by `score` (descending), divided by the full RMSE."""
    order = np.argsort(-np.asarray(score, dtype=np.float64).ravel(), kind='stable')
    e = np.asarray(err2, dtype=np.float64).ravel()[order]
    n = e.size
    tail = np.cumsum(e[::-1])[::-1]                                   # tail[i] = sum e[i:]
    full = e.mean()
    out = np.empty(len(fractions))
    for j, f in enumerate(fractions):
        i = int(round(f * n))
        out[j] = math.sqrt(tail[i] / (n - i) / full) if full > 0 else 1.0
    return out


def ause(pred_std, err):
    err2 = np.asarray(err, dtype=np.float64) ** 2
    return float(np.mean(sparsification_curve(pred_std, err2) - sparsification_curve(np.abs(err), err2)))


def spread_ratio(pred_var, err):
    return float(np.mean(pred_var) / np.mean(np.asarray(err, dtype=np.float64) ** 2))


def _pad(u):
    return np.pad(np.asarray(u, dtype=np.float64), [(0, 0)] * (np.ndim(u) - 2) + [(1, 1), (1, 1)], mode='reflect')


def laplacian(u):
    p = _pad(u)
    return p[..., :-2, 1:-1] + p[..., 2:, 1:-1] + p[..., 1:-1, :-2] + p[..., 1:-1, 2:] - 4.0 * np.asarray(u, dtype=np.float64)


def hf_energy_ratio(x_hat, x):
    e_hat, e = float((laplacian(x_hat) ** 2).mean()), float((laplacian(x) ** 2).mean())
    return float(math.log(max(e_hat, 1e-20) / max(e, 1e-20)))


def sobel_mag(u):
    p = _pad(u)
    gx = (p[..., :-2, 2:] + 2.0 * p[..., 1:-1, 2:] + p[..., 2:, 2:]) - (p[..., :-2, :-2] + 2.0 * p[..., 1:-1, :-2] + p[..., 2:, :-2])
    gy = (p[..., 2:, :-2] + 2.0 * p[..., 2:, 1:-1] + p[..., 2:, 2:]) - (p[..., :-2, :-2] + 2.0 * p[..., :-2, 1:-1] + p[..., :-2, 2:])
    return np.sqrt(gx * gx + gy * gy) / 8.0


def grad_w1(x_hat, x, bins=100, vmax=0.5):
    edges = np.linspace(0.0, vmax, bins + 1)
    h1, _ = np.histogram(np.clip(sobel_mag(x_hat), 0, vmax), edges)
    h2, _ = np.histogram(np.clip(sobel_mag(x), 0, vmax), edges)
    c1, c2 = np.cumsum(h1) / max(h1.sum(), 1), np.cumsum(h2) / max(h2.sum(), 1)
    return float(np.abs(c1 - c2).sum() * (vmax / bins))
