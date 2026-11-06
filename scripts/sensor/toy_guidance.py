"""Analytic Gaussian toy for the guided samplers: with a Gaussian prior the exact denoiser and the exact posterior
p(x0 | y) are known, so ΠGDM / DPS samples can be scored against the truth (RMSE excess, variance ratio, coverage,
CRPS ratio). The isotropic prior gates the ΠGDM step count for line B; the HFD-fitted prior is report-only.
    python scripts/sensor/toy_guidance.py [--out results/residual_warmstart/sensor/toy_guidance.json]
        [--steps 50 100] [--K 250] [--B 4] [--hw 8] [--device cuda:0] [--prior <npz>] [--data_root <dir>]
Refuses to overwrite --out.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from data_loader.linear_estimate import expand_matrix  # noqa: E402
from data_loader.sensor_forward import apply_H, guidance_operator, mosaic_mask, sigma_from_snr  # noqa: E402
from data_loader.warmstart_paths import DATA_ROOT, PRIOR  # noqa: E402
from model import guidance as G  # noqa: E402
from scripts.uq.metrics import coverage_samples, crps_gaussian, crps_samples  # noqa: E402

GATE = dict(rmse_excess=0.02, var_tol=0.10, cov90=(0.85, 0.95), cov50=(0.45, 0.55))


class GaussianDenoiser(torch.nn.Module):
    """The exact E[x0 | x_t] under x0 ~ N(mu, C), per pixel, emitted in the trainer's own parameterisation
    (trainer.prediction_type 'v', 'x0' or 'eps'), so the sampler's _model_to_x0_eps recovers it whatever the trainer."""
    PREDICTION_TYPES = ('v', 'x0', 'eps')

    def __init__(self, mu, C, trainer):
        super().__init__()
        pt = getattr(trainer, 'prediction_type', None)
        if pt not in self.PREDICTION_TYPES:
            raise ValueError(f'GaussianDenoiser: trainer.prediction_type {pt!r} is not one of {self.PREDICTION_TYPES}')
        self.register_buffer('mu', mu.to(trainer.device, torch.float32))
        self.register_buffer('C', C.to(trainer.device, torch.float32))
        self.trainer = trainer

    def forward(self, x_t, cond, t):
        ab = self.trainer.alpha_bars[t].to(x_t.dtype)                                # [B]
        n = self.C.shape[0]
        S = ab[:, None, None] * self.C[None] + (1.0 - ab)[:, None, None] * torch.eye(n, device=x_t.device)[None]
        M = torch.sqrt(ab)[:, None, None] * torch.linalg.solve(S, self.C[None].expand_as(S), left=False)  # C S^-1
        d = x_t - torch.sqrt(ab)[:, None, None, None] * self.mu[None, :, None, None]
        x0 = self.mu[None, :, None, None] + torch.einsum('bij,bjhw->bihw', M, d)
        sab = torch.sqrt(ab)[:, None, None, None]
        pt = self.trainer.prediction_type
        if pt == 'x0':
            return x0
        if pt == 'eps':
            return (x_t - sab * x0) / torch.sqrt(1.0 - ab)[:, None, None, None]
        return (sab * x_t - x0) / torch.sqrt(1.0 - ab)[:, None, None, None]                     # 'v'


def exact_posterior(mu, C, H, b, y, sigma, mask):
    """Per-pixel Gaussian conditioning of N(mu, C) on y = H x + b + n, n ~ N(0, sigma^2): (mean, var) [B, 64, H, W].
    Sequential (mask None / all-ones): all 30 rows; snapshot (one-hot mask): the pixel's measured row only."""
    mu, C, H, b, y = (a.to(torch.float64) for a in (mu, C, H, b, y))
    s2 = float(sigma) ** 2
    d = y - b[None, :, None, None] - torch.einsum('oc,c->o', H, mu)[None, :, None, None]
    if mask is None or bool(torch.all(mask == 1)):
        S = H @ C @ H.T + s2 * torch.eye(H.shape[0], dtype=torch.float64)
        Kg = torch.linalg.solve(S, H @ C).T                                          # C H^T S^-1, [64, 30]
        mean = mu[None, :, None, None] + torch.einsum('co,bohw->bchw', Kg, d)
        var = torch.diagonal(C - Kg @ H @ C).clamp(min=0.0)
        return mean, var[None, :, None, None].expand_as(mean).clone()
    k = mask.argmax(1)                                                               # [B, H, W]
    Ch = C @ H.T                                                                     # [64, 30]: C h_k
    Sk = (H * Ch.T).sum(1) + s2                                                      # h_k^T C h_k + s2, [30]
    gain = Ch / Sk[None]                                                             # [64, 30]
    vark = (torch.diagonal(C)[:, None] - Ch ** 2 / Sk[None]).clamp(min=0.0)          # [64, 30]
    dk = torch.gather(d, 1, k[:, None]).squeeze(1)                                   # [B, H, W]
    mean = mu[None, :, None, None] + gain[:, k].permute(1, 0, 2, 3) * dk[:, None]
    var = vark[:, k].permute(1, 0, 2, 3)
    return mean, var


def _prior(C_kind, prior_path, device):
    if C_kind == 'isotropic':
        return torch.zeros(64, dtype=torch.float64), torch.eye(64, dtype=torch.float64)
    if C_kind != 'hfd':
        raise ValueError(f"unknown C_kind {C_kind!r}; expected 'isotropic' or 'hfd'")
    with np.load(prior_path) as f:
        mu31, C31 = f['mu'].astype(np.float64), f['C'].astype(np.float64)            # trainer scale [-1, 1], 31 bands
    A = expand_matrix(np.linspace(451, 855, 31), 64)
    return torch.from_numpy(A @ mu31), torch.from_numpy(A @ C31 @ A.T + 1e-6 * np.eye(64))


def run_toy(trainer, H, b, mode, snr_db, n_steps, K, B, hw, seed, method, C_kind='isotropic', prior_path=PRIOR):
    if mode not in ('sequential', 'snapshot'):
        raise ValueError(f"unknown mode {mode!r}; expected 'sequential' or 'snapshot'")
    dev = trainer.device
    H, b = H.to(dev, torch.float32), b.to(dev, torch.float32)
    mu, C = _prior(C_kind, prior_path, dev)
    gen = torch.Generator(device='cpu').manual_seed(10_000 + int(seed))             # disjoint from the draw seeds 0..K-1
    L = torch.linalg.cholesky(C)
    z = torch.randn((B, 64, hw, hw), generator=gen, dtype=torch.float64)
    x0 = (mu[None, :, None, None] + torch.einsum('ij,bjhw->bihw', L, z)).to(torch.float32).to(dev)
    y_clean = apply_H(x0, H) + b[None, :, None, None]
    sigma = sigma_from_snr(snr_db, float((y_clean.double() ** 2).mean()))
    n = sigma * torch.randn(y_clean.shape, generator=gen).to(dev)
    mask = None
    if mode == 'snapshot':
        mask = torch.from_numpy(mosaic_mask(hw, hw)).permute(2, 0, 1)[None].expand(B, -1, -1, -1).to(dev)
    y = y_clean + n if mask is None else (y_clean + n) * mask
    meas = G.LinearMeasurement(H, b, y, mask, sigma)
    den = GaussianDenoiser(mu.float(), C.float(), trainer)
    samples = G.draw_guided(den, trainer, meas, (B, 64, hw, hw), K=K, n_steps=n_steps, method=method, clamp=None)
    nfe = trainer.last_nfe
    mean_e, var_e = exact_posterior(mu, C, H.cpu(), b.cpu(), y.cpu(), sigma, None if mask is None else mask.cpu())
    s, x, me, ve = (a.detach().cpu().double().numpy() for a in (samples, x0, mean_e, var_e))
    rmse = lambda a: float(np.sqrt(np.mean((a - x) ** 2)))  # noqa: E731
    prior_mean = np.broadcast_to(mu.numpy()[None, :, None, None], x.shape)
    m_cpu = None if mask is None else mask.cpu().double()

    def meas_rmse(a):
        """RMSE of H (a - x0) on the measured entries: the direction the measurement informs (in snapshot mode one scalar
        of 64 per pixel, so the full-space RMSE barely moves from the prior's)."""
        r = apply_H(torch.from_numpy(np.ascontiguousarray(a - x)), H.cpu().double())
        if m_cpu is None:
            return float(torch.sqrt((r ** 2).mean()))
        return float(torch.sqrt((r ** 2 * m_cpu).sum() / m_cpu.sum()))

    cov = coverage_samples(s, x)
    return dict(rmse_excess=rmse(s.mean(0)) / rmse(me) - 1.0,
                var_ratio=float(s.var(0, ddof=1).mean() / ve.mean()),
                cov50=cov[0.5], cov90=cov[0.9],
                crps_ratio=float(crps_samples(s, x).mean() / crps_gaussian(me, np.sqrt(ve), x).mean()),
                nfe=int(nfe),
                rmse=rmse(s.mean(0)), rmse_prior=rmse(prior_mean),                       # the guided mean's / the prior mean's
                rmse_meas=meas_rmse(s.mean(0)), rmse_meas_prior=meas_rmse(prior_mean))   # the same on the measured entries of H x


def passes_gate(r):
    return (r['rmse_excess'] <= GATE['rmse_excess'] and abs(r['var_ratio'] - 1.0) <= GATE['var_tol']
            and GATE['cov90'][0] <= r['cov90'] <= GATE['cov90'][1] and GATE['cov50'][0] <= r['cov50'] <= GATE['cov50'][1])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/residual_warmstart/sensor/toy_guidance.json')
    ap.add_argument('--steps', type=int, nargs='+', default=[50, 100])
    ap.add_argument('--K', type=int, default=250)
    ap.add_argument('--B', type=int, default=4)
    ap.add_argument('--hw', type=int, default=8)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--prior', default=PRIOR)
    ap.add_argument('--data_root', default=DATA_ROOT)
    ap.add_argument('--snr_db', type=float, default=30.0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise FileExistsError(f'{a.out} exists; refusing to overwrite')

    from data_loader.sensor_model import load_sensor
    from model.diffusion_trainer import DiffusionTrainer
    sensor = load_sensor('device1', a.data_root)
    Hn, bn = guidance_operator(sensor['R'], expand_matrix(np.linspace(451, 855, 31), 64))
    H, b = torch.tensor(Hn, dtype=torch.float32), torch.tensor(bn, dtype=torch.float32)
    tr = DiffusionTrainer(device=torch.device(a.device), prediction_type='v', snr_gamma=None)

    results = []
    for C_kind in ('isotropic', 'hfd'):
        for method in ('pigdm', 'dps'):
            for mode in ('sequential', 'snapshot'):
                for n_steps in a.steps:
                    t0 = time.time()
                    r = run_toy(tr, H, b, mode, a.snr_db, n_steps, a.K, a.B, a.hw, a.seed, method, C_kind,
                                prior_path=a.prior)
                    r.update(C_kind=C_kind, method=method, mode=mode, n_steps=n_steps, seconds=time.time() - t0)
                    results.append(r)
                    print(json.dumps(r), flush=True)

    def ok(n):
        rs = [r for r in results if r['C_kind'] == 'isotropic' and r['method'] == 'pigdm' and r['n_steps'] == n]
        return len(rs) == 2 and all(passes_gate(r) for r in rs)
    passing = [n for n in sorted(a.steps) if ok(n)]
    verdict = f'steps_for_B = {passing[0]}' if passing else 'FAIL: use DPS'
    out = dict(results=results, verdict=verdict, steps_for_B=passing[0] if passing else None, gate=GATE,
               args=vars(a), sensor_sha256=sensor['sha256'])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'x') as f:
        json.dump(out, f, indent=2)
    print(verdict)


if __name__ == '__main__':
    main()
