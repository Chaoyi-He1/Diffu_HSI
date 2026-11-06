"""Posterior sampling with an unconditional diffusion prior and the known linear measurement y = H x64 + b + n.

ΠGDM (Song et al., ICLR 2023) and DPS (Chung et al., 2022) on the trainer's VP schedule, both on the repo's DDIM
grid ts = range(T - 1, -1, -(T // n_steps)) with the eta = 0 update in the form of DiffusionTrainer.sample.
Sequential mode: every pixel measures all 30 rows (mask None or all-ones); snapshot mode: a one-hot mask selects one
row per pixel. An unconditional model is called as model(x_t, {'sensor': None}, t).
"""
import torch
from tqdm import tqdm

from data_loader.sensor_forward import apply_H, apply_HT

UNCOND = {'sensor': None}


def _clamp(x, clamp):
    return x if clamp is None else x.clamp(clamp[0], clamp[1])


def _grid(trainer, n_steps):
    return list(range(trainer.n_timesteps - 1, -1, -max(1, trainer.n_timesteps // n_steps)))


def _view(a, x):
    return a.view(-1, *([1] * (x.dim() - 1)))


class LinearMeasurement:
    """y_eff = (y - b) * mask; apply(x) = (H x) * mask; solve applies (r2 H H^T + sigma^2 I)^-1 on the measured rows."""

    def __init__(self, H, b, y, mask, sigma_y):
        self.H = H.to(device=y.device, dtype=y.dtype)
        self.b = b.to(device=y.device, dtype=y.dtype)
        self.sigma2 = float(sigma_y) ** 2
        if mask is None or bool(torch.all(mask == 1)):
            self.mask, self.snapshot = None, False
        else:
            mask = mask.to(device=y.device, dtype=y.dtype)
            if mask.shape[1] != self.H.shape[0] or not bool(torch.all((mask == 0) | (mask == 1))) \
                    or not bool(torch.all(mask.sum(1) == 1)):
                raise ValueError('mask must be None, all-ones (sequential) or one-hot over the 30 rows (snapshot)')
            self.mask, self.snapshot = mask, True
        self.y_eff = self._m(y - self.b[None, :, None, None])

    def _m(self, v):
        return v if self.mask is None else v * self.mask

    def apply(self, x64):
        return self._m(apply_H(x64, self.H))

    def residual(self, x0):
        return self.y_eff - self.apply(x0)

    def solve(self, resid, r2):
        r2 = r2.to(device=resid.device, dtype=resid.dtype)
        if self.snapshot:
            hk2 = (self.H ** 2).sum(1)                                              # ||h_k||^2, [30]
            denom = r2[:, None, None, None] * hk2[None, :, None, None] + self.sigma2
            return resid / denom * self.mask
        B, O = resid.shape[:2]
        A = (r2[:, None, None] * self.H[None]) @ self.H.T + self.sigma2 * torch.eye(O, device=resid.device,
                                                                                     dtype=resid.dtype)[None]
        return torch.linalg.solve(A, resid.reshape(B, O, -1)).reshape(resid.shape)


def _x0_with_grad(model, trainer, x, t, clamp):
    x = x.detach().requires_grad_(True)
    x0 = _clamp(trainer._model_to_x0_eps(model(x, UNCOND, t), x, t)[0], clamp)
    return x, x0


def _ddim_step(x, x0, ab_t, ab_prev):
    """eta = 0 DDIM update with eps recomputed to be consistent with x0 (as DiffusionTrainer.sample)."""
    eps = (x - torch.sqrt(ab_t) * x0) / torch.sqrt(1.0 - ab_t)
    return torch.sqrt(ab_prev) * x0 + torch.sqrt((1.0 - ab_prev).clamp(min=0.0)) * eps


def _init(trainer, shape, seed):
    """x_T for draw `seed`. The generator is keyed by the seed alone, so every batch of the same shape gets the SAME x_T for
    seed k: cube i of batch 1 and cube i of batch 2 start draw k from identical noise (a correlation across batches, not
    across draws). Kept deliberately: mixing the batch index into the seed would make the existing B records
    non-reproducible (final-review Minor 10, ruled DOCUMENT only)."""
    g = torch.Generator(device='cpu').manual_seed(int(seed))
    return torch.randn(shape, generator=g).to(trainer.device)


def pigdm_sample(model, trainer, meas, shape, n_steps, seed, clamp=(-1.0, 1.0), progress=False):
    ts = _grid(trainer, n_steps)
    trainer.last_nfe = len(ts)
    x = _init(trainer, shape, seed)
    B = shape[0]
    for i, step in enumerate(tqdm(ts, desc='PiGDM') if progress else ts):
        t = torch.full((B,), step, device=trainer.device, dtype=torch.long)
        with torch.enable_grad():
            x_in, x0 = _x0_with_grad(model, trainer, x, t, clamp)
            with torch.no_grad():
                ab = trainer.alpha_bars[t]
                v = meas.solve(meas.residual(x0), 1.0 - ab)
                w = apply_HT(meas._m(v), meas.H)
            g = torch.autograd.grad((x0 * w).sum(), x_in)[0]
        with torch.no_grad():
            ab_t = _view(ab, x)
            # g = grad_{x_t} log p(y | x_t) under the PiGDM Gaussian; Tweedie on p(x_t | y) gives
            # E[x0 | x_t, y] = x0 + (1 - ab_t) / sqrt(ab_t) * g = x0 + r_t^2 / sqrt(ab_t) * g (exact for a Gaussian prior
            # with r_t^2 the true posterior variance; a sqrt(ab_t) * g step on x0 is off by ab_t / r_t^2 and diverges).
            x0g = _clamp(x0.detach() + (1.0 - ab_t) / torch.sqrt(ab_t) * g, clamp)
            if i + 1 == len(ts):
                return x0g
            ab_prev = _view(trainer.alpha_bars[torch.full_like(t, ts[i + 1])], x)
            x = _ddim_step(x_in.detach(), x0g, ab_t, ab_prev)


def dps_sample(model, trainer, meas, shape, n_steps, seed, zeta=1.0, clamp=(-1.0, 1.0)):
    ts = _grid(trainer, n_steps)
    trainer.last_nfe = len(ts)
    x = _init(trainer, shape, seed)
    B = shape[0]
    for i, step in enumerate(ts):
        t = torch.full((B,), step, device=trainer.device, dtype=torch.long)
        with torch.enable_grad():
            x_in, x0 = _x0_with_grad(model, trainer, x, t, clamp)
            nrm = meas.residual(x0).flatten(1).norm(dim=1)
            g = torch.autograd.grad(nrm.sum(), x_in)[0]
        with torch.no_grad():
            x0 = x0.detach()
            if i + 1 == len(ts):
                return x0
            ab_t = _view(trainer.alpha_bars[t], x)
            ab_prev = _view(trainer.alpha_bars[torch.full_like(t, ts[i + 1])], x)
            x = _ddim_step(x_in.detach(), x0, ab_t, ab_prev) - zeta * g


def draw_guided(model, trainer, meas, shape, K, n_steps, method='pigdm', clamp=(-1.0, 1.0)):
    if method == 'pigdm':
        fn = lambda k: pigdm_sample(model, trainer, meas, shape, n_steps, seed=k, clamp=clamp)  # noqa: E731
    elif method == 'dps':
        fn = lambda k: dps_sample(model, trainer, meas, shape, n_steps, seed=k, clamp=clamp)  # noqa: E731
    else:
        raise ValueError(f"unknown method {method!r}; expected 'pigdm' or 'dps'")
    return torch.stack([fn(k) for k in range(K)])
