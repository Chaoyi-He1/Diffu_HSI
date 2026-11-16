"""Last-step Gaussian posterior draw for the guided prior (line B+).

ΠGDM returns the denoiser's x0 estimate at its last time step: a mean, not a sample. The prior network has a
reconstruction floor (its x0 error does not fall below ~1.7 % however small t is), so the K returned cubes nearly
coincide and their spread says nothing about the error. B+ replaces that last step by a draw: each returned cube s is the
centre of a Gaussian component N(s, Sigma_e) per pixel, Sigma_e [C, C] being the error covariance of the sampler's
output measured on held-out calibration cubes (scripts/sensor/fit_b_kernel.py), and the component is conditioned exactly
on the pixel's measurement y = H x + b + n. The draw uses Matheron's rule, so one code path serves every measurement
pattern (all rows, one row per pixel, any per-element mask, any per-element noise variance):

    e ~ N(0, Sigma_e),  n' ~ N(0, N),   x = s + e + G (y - b - H (s + e) - n'),   G = Sigma_e H^T (H Sigma_e H^T + N)^-1

which is an exact draw of N(mu, Sigma_post) with mu = s + G (y - b - H s), Sigma_post = Sigma_e - G H Sigma_e.
Everything is float64 on the samples' device; cubes are [B, C, H, W] on the trainer scale.
"""
import hashlib

import numpy as np
import torch

SEED_OFFSET = 100_003          # draw k uses the generator seed SEED_OFFSET + k (x_T of draw k uses seed k)


def kernel_sha256(Sigma_e):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(Sigma_e, dtype=np.float64)).tobytes()).hexdigest()


def load_kernel(path):
    """-> (Sigma_e [C, C] float64 numpy, meta dict). The file is the one fit_b_kernel.py writes."""
    d = np.load(path, allow_pickle=True)
    S = np.asarray(d['Sigma_e'], dtype=np.float64)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or not np.allclose(S, S.T, atol=1e-10):
        raise ValueError(f'{path}: Sigma_e must be a symmetric square matrix, got shape {S.shape}')
    if np.linalg.eigvalsh(S).min() < -1e-8 * max(1.0, float(np.abs(S).max())):
        raise ValueError(f'{path}: Sigma_e is not positive semi-definite')
    meta = d['meta'].item() if 'meta' in d.files else {}
    return S, meta


def _sqrt_psd(S):
    w, V = torch.linalg.eigh(0.5 * (S + S.T))
    return V * torch.sqrt(w.clamp(min=0.0))[None, :]


def _noise(shape, seed, device):
    g = torch.Generator(device='cpu').manual_seed(int(seed))
    return torch.randn(shape, generator=g, dtype=torch.float64).to(device)


def posterior_draw(s, H, b, y, sigma2, Sigma_e, seed, mask=None, clamp=(-1.0, 1.0)):
    """One draw per cube of N(mu, Sigma_post) around the centre s.

    s [B, C, H, W]; H [O, C]; b [O]; y [B, O, H, W]; sigma2 a float or a [B, O, H, W] tensor of per-element noise
    variances; Sigma_e [C, C]; mask None (every row measured) or a [B, O, H, W] 0/1 tensor (a pixel with no measured
    row gets s + e). The noise is keyed by `seed` alone (as model.guidance._init: the same e for every batch of one
    shape). Returns [B, C, H, W] in s's dtype."""
    dev, out_dtype = s.device, s.dtype
    f = dict(device=dev, dtype=torch.float64)
    s64, Hm, bv, yv = s.to(**f), torch.as_tensor(H).to(**f), torch.as_tensor(b).to(**f), y.to(**f)
    S = torch.as_tensor(Sigma_e).to(**f)
    B, C, Hh, Ww = s64.shape
    O = Hm.shape[0]
    if S.shape != (C, C) or Hm.shape[1] != C or yv.shape != (B, O, Hh, Ww):
        raise ValueError(f'shapes do not fit: s {tuple(s.shape)}, H {tuple(Hm.shape)}, y {tuple(y.shape)}, Sigma_e {tuple(S.shape)}')
    e = torch.einsum('bhwc,dc->bdhw', _noise((B, Hh, Ww, C), seed, dev), _sqrt_psd(S))
    z = _noise((B, O, Hh, Ww), int(seed) + 7_919_000, dev)
    scalar = not torch.is_tensor(sigma2)
    if scalar and float(sigma2) <= 0.0:
        raise ValueError('sigma2 must be positive')
    nv = float(sigma2) if scalar else sigma2.to(**f)
    if not scalar and (nv.shape != yv.shape or bool((nv <= 0).any())):
        raise ValueError('a per-element sigma2 must have y\'s shape and be positive')
    xe = s64 + e
    resid = yv - bv[None, :, None, None] - torch.einsum('oc,bchw->bohw', Hm, xe) - (nv ** 0.5) * z
    M = Hm @ S @ Hm.T                                                # [O, O]
    SHt = S @ Hm.T                                                   # [C, O]
    if mask is None and scalar:
        A = M + nv * torch.eye(O, **f)
        v = torch.linalg.solve(A, resid.permute(1, 0, 2, 3).reshape(O, -1)).reshape(O, B, Hh, Ww).permute(1, 0, 2, 3)
    else:
        m = torch.ones_like(yv) if mask is None else mask.to(**f)
        if m.shape != yv.shape or not bool(((m == 0) | (m == 1)).all()):
            raise ValueError('mask must be a 0/1 tensor of y\'s shape')
        mp = m.permute(0, 2, 3, 1).reshape(-1, O)                    # [P, O]
        nvp = torch.full_like(mp, nv) if scalar else nv.permute(0, 2, 3, 1).reshape(-1, O)
        A = mp[:, :, None] * M[None] * mp[:, None, :] + torch.diag_embed(torch.where(mp > 0, nvp, torch.ones_like(nvp)))
        r = (resid.permute(0, 2, 3, 1).reshape(-1, O) * mp)[..., None]
        v = (torch.linalg.solve(A, r)[..., 0] * mp).reshape(B, Hh, Ww, O).permute(0, 3, 1, 2)
    x = xe + torch.einsum('co,bohw->bchw', SHt, v)
    if clamp is not None:
        x = x.clamp(clamp[0], clamp[1])
    return x.to(out_dtype)


def posterior_draws(draws, H, b, y, sigma2, Sigma_e, mask=None, clamp=(-1.0, 1.0)):
    """draws [K, B, C, H, W] -> the same shape; draw k uses the seed SEED_OFFSET + k."""
    return torch.stack([posterior_draw(draws[k], H, b, y, sigma2, Sigma_e, SEED_OFFSET + k, mask=mask, clamp=clamp)
                        for k in range(draws.shape[0])])


def pooled_error_covariance(resid):
    """resid [N, C, H, W] (truth - ensemble mean, trainer scale) -> the [C, C] covariance of the per-pixel spectral
    residual pooled over pixels and cubes (the per-band mean is removed)."""
    r = np.moveaxis(np.asarray(resid, dtype=np.float64), 1, -1).reshape(-1, resid.shape[1])
    r = r - r.mean(0, keepdims=True)
    return r.T @ r / r.shape[0]
