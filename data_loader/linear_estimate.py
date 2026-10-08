"""Closed-form estimate of the hyperspectral cube from the (low-resolution) sensor image.

x0_hat = lift_to_grid( clip(mu + K (y - R^T mu)) @ A^T )   with A the loader's 31->64 band interpolation.
All functions are numpy and deterministic; the dataset calls them per sample.
"""
import numpy as np


def expand_matrix(wavelens=np.linspace(451, 855, 31), n_out=64):
    """Matrix form of HFD_data.expand_wavelens (piecewise-linear interpolation, extrapolating at the ends)."""
    wl = np.asarray(wavelens, dtype=np.float64)
    new = np.linspace(wl[0], wl[-1], n_out)
    A = np.zeros((n_out, len(wl)))
    for k, w in enumerate(new):
        j = np.clip(np.searchsorted(wl, w, side='right') - 1, 0, len(wl) - 2)
        t = (w - wl[j]) / (wl[j + 1] - wl[j])
        A[k, j], A[k, j + 1] = 1.0 - t, t
    return A


def linear_inverse(y, mu, K, R):
    """Per-pixel linear MMSE estimate on the loader's [-1, 1] scale, clipped. y: [..., N] -> [..., L]."""
    xh = mu + (y - mu @ R) @ K.T
    return np.clip(xh, -1.0, 1.0)


def lift_to_grid(x_lo, d, H, W):
    """Grid-aligned bilinear upsampling: x_lo[i, j] sits at high-res pixel (i*d, j*d); beyond the last
    sample the value is held constant (edge extension). Identity when d == 1."""
    if d == 1:
        return x_lo
    h, w = x_lo.shape[:2]

    def weights(n_hi, n_lo):
        pos = np.arange(n_hi, dtype=np.float64) / d
        i0 = np.minimum(np.floor(pos).astype(int), n_lo - 1)
        i1 = np.minimum(i0 + 1, n_lo - 1)
        t = np.clip(pos - i0, 0.0, 1.0)
        t[i1 == i0] = 0.0
        return i0, i1, t

    r0, r1, tr = weights(H, h)
    c0, c1, tc = weights(W, w)
    rows = x_lo[r0] * (1 - tr)[:, None, None] + x_lo[r1] * tr[:, None, None]         # [H, w, C]
    out = rows[:, c0] * (1 - tc)[None, :, None] + rows[:, c1] * tc[None, :, None]    # [H, W, C]
    return out


def estimate_from_sensor(y_d, prior, d, H, W, A):
    """y_d: [h, w, 30] low-resolution sensor -> [H, W, 64] estimate on the [-1, 1] scale."""
    x31 = linear_inverse(y_d, prior['mu'], prior['K'], prior['R'])   # [h, w, 31]
    x64 = x31 @ A.T                                                   # [h, w, 64]
    return lift_to_grid(x64, d, H, W)


def load_prior(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}
