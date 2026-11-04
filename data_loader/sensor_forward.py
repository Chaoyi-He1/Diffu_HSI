"""Forward models for the Phase 3 poor-sensor simulation (sequential and 5x6 snapshot)."""
import numpy as np
import torch

TILE = (5, 6)
N_CHANNELS = 30


def mosaic_mask(H, W, n_channels=N_CHANNELS):
    """One-hot [H, W, n_channels] float32; pixel (i, j) measures k = (i % 5) * 6 + (j % 6)."""
    i = np.arange(H)[:, None] % TILE[0]
    j = np.arange(W)[None, :] % TILE[1]
    k = i * TILE[1] + j
    mask = np.zeros((H, W, n_channels), np.float32)
    np.put_along_axis(mask, np.broadcast_to(k, (H, W))[..., None], 1.0, axis=-1)
    return mask


def sigma_from_snr(snr_db, P_y):
    """sigma^2 = P_y / 10^(SNR/10)."""
    return float(np.sqrt(P_y / 10.0 ** (snr_db / 10.0)))


def measure(x01_31, R, mode, sigma, rng):
    """y = x01_31 @ R + sigma * N(0, 1); snapshot additionally masks to the mosaic (off-mask exactly 0)."""
    if mode not in ('sequential', 'snapshot'):
        raise ValueError(f"unknown mode {mode!r}; expected 'sequential' or 'snapshot'")
    y = x01_31 @ R
    y = y + sigma * rng.standard_normal(y.shape)
    H, W = y.shape[:2]
    mask = np.ones((H, W, y.shape[-1]), np.float32) if mode == 'sequential' else mosaic_mask(H, W, y.shape[-1])
    return (y * mask).astype(np.float32), mask


def guidance_operator(R, A):
    """H = R^T A^+ / 2 [30, 64], b = H @ 1_64 [30], so that y = H x64 + b for x64 = A (2 x01 - 1)."""
    H = 0.5 * R.T @ np.linalg.pinv(A)
    return H, H.sum(axis=1)


def apply_H(x64, H):
    return torch.einsum('oc,bchw->bohw', H, x64)


def apply_HT(v, H):
    return torch.einsum('oc,bohw->bchw', H, v)
