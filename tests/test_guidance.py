# tests/test_guidance.py
import numpy as np, torch
from model import guidance as G
from model.diffusion_trainer import DiffusionTrainer
from data_loader.sensor_forward import guidance_operator, mosaic_mask
from data_loader.linear_estimate import expand_matrix
from scripts.sensor import toy_guidance as T

def _ops(small_R):
    H, b = guidance_operator(small_R, expand_matrix(np.linspace(451, 855, 31), 64)); return torch.tensor(H, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)

def test_sequential_solve_matches_dense(small_R):
    torch.manual_seed(0)
    H, b = _ops(small_R); y = torch.randn(2, 30, 3, 3); m = G.LinearMeasurement(H, b, y, None, 0.1)
    r = torch.randn(2, 30, 3, 3); r2 = torch.tensor([0.3, 0.7]); v = m.solve(r, r2)
    for i in range(2):
        Hd = H.double(); A = r2[i].double() * Hd @ Hd.T + 0.01 * torch.eye(30, dtype=torch.double)
        ref = torch.linalg.solve(A, r[i].double().reshape(30, -1)).reshape(30, 3, 3)
        assert torch.allclose(v[i].double(), ref, rtol=1e-3, atol=1e-3)

def test_snapshot_solve_is_scalar_on_the_measured_row_and_zero_elsewhere(small_R):
    H, b = _ops(small_R); mask = torch.tensor(mosaic_mask(5, 6)).permute(2, 0, 1)[None]; y = torch.randn(1, 30, 5, 6) * mask
    m = G.LinearMeasurement(H, b, y, mask, 0.1); r = torch.randn(1, 30, 5, 6) * mask; v = m.solve(r, torch.tensor([0.5]))
    assert torch.all(v[mask == 0] == 0)
    k = 7; i, j = 1, 1                                                     # tile position of channel 7 is (1, 1)
    assert torch.isclose(v[0, k, i, j], r[0, k, i, j] / (0.5 * (H[k] ** 2).sum() + 0.01), atol=1e-5)

def test_pigdm_recovers_the_isotropic_gaussian_posterior(small_R):
    tr = DiffusionTrainer(device=torch.device('cpu'), prediction_type='v', snr_gamma=None); H, b = _ops(small_R)
    out = T.run_toy(tr, H, b, 'sequential', 30.0, n_steps=50, K=200, B=1, hw=8, seed=0, method='pigdm')
    assert out['rmse_excess'] <= 0.03 and 0.85 <= out['var_ratio'] <= 1.15 and 0.83 <= out['cov90'] <= 0.97 and 0.42 <= out['cov50'] <= 0.58

def test_pigdm_snapshot_toy_runs_and_beats_the_prior(small_R):
    tr = DiffusionTrainer(device=torch.device('cpu'), prediction_type='v', snr_gamma=None); H, b = _ops(small_R)
    out = T.run_toy(tr, H, b, 'snapshot', 30.0, n_steps=20, K=32, B=1, hw=10, seed=0, method='pigdm')
    assert out['nfe'] == 20
    # the guided mean must beat the prior's guess (mu) where y informs it (H x on the measured entries; one scalar of 64 per
    # pixel, so the full-space RMSE can only stay near the prior's), not merely stay within 50 % of the exact posterior's
    assert out['rmse_meas'] < 0.3 * out['rmse_meas_prior'] and out['rmse'] <= 1.05 * out['rmse_prior'] and out['rmse_excess'] < 0.5

def test_gaussian_denoiser_follows_the_trainer_prediction_type():
    x, t = torch.randn(2, 64, 3, 3), torch.tensor([100, 700])
    mu, C = 0.1 * torch.ones(64), 0.5 * torch.eye(64)
    x0s = []
    for pt in ('v', 'x0', 'eps'):
        tr = DiffusionTrainer(device=torch.device('cpu'), prediction_type=pt, snr_gamma=None)
        x0s.append(tr._model_to_x0_eps(T.GaussianDenoiser(mu, C, tr)(x, None, t), x, t)[0])
    assert torch.allclose(x0s[0], x0s[1], atol=1e-4) and torch.allclose(x0s[0], x0s[2], atol=1e-3)

def test_dps_runs_and_reduces_the_residual(small_R):
    tr = DiffusionTrainer(device=torch.device('cpu'), prediction_type='v', snr_gamma=None); H, b = _ops(small_R)
    den = T.GaussianDenoiser(torch.zeros(64), torch.eye(64), tr); y = torch.randn(1, 30, 4, 4)
    m = G.LinearMeasurement(H, b, y, None, 0.1)
    x_g = G.dps_sample(den, tr, m, (1, 64, 4, 4), 20, seed=0, clamp=None); x_u = tr.sample(den, {'sensor': None}, (1, 64, 4, 4), n_steps=20, method='ddim', progress=False, clamp_x0=(-1e9, 1e9))
    assert m.residual(x_g).norm() < m.residual(x_u).norm()

def test_seed_reproducible_and_draw_guided_shape(small_R):
    tr = DiffusionTrainer(device=torch.device('cpu'), prediction_type='v', snr_gamma=None); H, b = _ops(small_R)
    den = T.GaussianDenoiser(torch.zeros(64), torch.eye(64), tr); m = G.LinearMeasurement(H, b, torch.randn(1, 30, 4, 4), None, 0.1)
    a = G.pigdm_sample(den, tr, m, (1, 64, 4, 4), 10, seed=3, clamp=None); bb = G.pigdm_sample(den, tr, m, (1, 64, 4, 4), 10, seed=3, clamp=None)
    assert torch.equal(a, bb) and G.draw_guided(den, tr, m, (1, 64, 4, 4), K=3, n_steps=5, clamp=None).shape == (3, 1, 64, 4, 4)


# ----------------------------------------------------------------------------- the general measurement (Phase 3b)
import pytest  # noqa: E402


def _dense_v(H, r, mask, nv, r2):
    """Per-pixel reference: (r2 M H H^T M + diag(nv measured, 1 unmeasured)) v = M r, returned as M v (float64)."""
    B, O, Hh, Ww = r.shape
    Hd = H.double(); out = torch.zeros(B, O, Hh, Ww, dtype=torch.float64)
    for b_ in range(B):
        for i in range(Hh):
            for j in range(Ww):
                m = mask[b_, :, i, j].double(); M = torch.diag(m)
                A = float(r2[b_]) * M @ Hd @ Hd.T @ M + torch.diag(torch.where(m > 0, nv[b_, :, i, j].double(), torch.ones(O, dtype=torch.float64)))
                out[b_, :, i, j] = m * torch.linalg.solve(A, m * r[b_, :, i, j].double())
    return out


def test_general_path_equals_the_sequential_and_snapshot_paths(small_R):
    torch.manual_seed(0)
    H, b = _ops(small_R); y = torch.randn(2, 30, 5, 6); r = torch.randn(2, 30, 5, 6); r2 = torch.tensor([0.3, 0.7])
    seq = G.LinearMeasurement(H, b, y, None, 0.1)
    gen = G.LinearMeasurement(H, b, y, None, 0.1, noise_var=torch.full_like(y, 0.01))
    gen1 = G.LinearMeasurement(H, b, y, torch.ones_like(y), 0.1, noise_var=torch.full_like(y, 0.01))
    assert not seq.general and gen.general and gen1.general
    # H H^T is ill-conditioned (v reaches ~300): the float32 sequential solve carries ~2e-3 of rounding, so the general
    # path (float64 inside) is compared with the sequential path run in float64 at 1e-4, and with the float32 run at
    # 1e-4 of the solution's scale
    seq64 = G.LinearMeasurement(H.double(), b.double(), y.double(), None, 0.1).solve(r.double(), r2.double())
    vs = seq.solve(r, r2)
    for g_ in (gen, gen1):
        v = g_.solve(r, r2)
        assert v.dtype == torch.float32 and torch.allclose(v.double(), seq64, atol=1e-4)
        assert float((v - vs).abs().max()) <= 1e-4 * float(vs.abs().max())
    x64 = torch.randn(2, 64, 5, 6)
    assert torch.allclose(gen.y_eff, seq.y_eff) and torch.allclose(gen.residual(x64), seq.residual(x64))
    mask = torch.tensor(mosaic_mask(5, 6)).permute(2, 0, 1)[None].expand(2, -1, -1, -1).contiguous()
    snap = G.LinearMeasurement(H, b, y * mask, mask, 0.1)
    gs = G.LinearMeasurement(H, b, y * mask, mask, 0.1, noise_var=torch.full_like(y, 0.01))
    assert snap.snapshot and not snap.general and gs.general
    assert torch.allclose(gs.solve(r * mask, r2), snap.solve(r * mask, r2), atol=1e-4)            # scalar per pixel: well conditioned
    x = torch.randn(2, 64, 5, 6)
    assert torch.allclose(gs.y_eff, snap.y_eff) and torch.allclose(gs.apply(x), snap.apply(x))


def test_general_solve_matches_a_dense_reference_and_unmeasured_pixels_get_zero(small_R):
    torch.manual_seed(1)
    H, b = _ops(small_R)
    mask = (torch.rand(2, 30, 3, 4) < 0.5).float(); mask[1, :, 2, 3] = 0                 # one pixel with no measured row
    nv = 0.001 + 0.05 * torch.rand(2, 30, 3, 4)
    y = torch.randn(2, 30, 3, 4) * mask; r = torch.randn(2, 30, 3, 4); r2 = torch.tensor([0.2, 0.9])
    m = G.LinearMeasurement(H, b, y, mask, 0.1, noise_var=nv)
    v = m.solve(r, r2)
    assert v.dtype == r.dtype and torch.allclose(v.double(), _dense_v(H, r, mask, nv, r2), atol=1e-4, rtol=1e-4)
    assert torch.all(v[1, :, 2, 3] == 0) and torch.all(v[mask == 0] == 0)
    m0 = G.LinearMeasurement(H, b, y, mask, 0.1)                                     # a 0/1 mask that is not one-hot, no noise_var
    assert m0.general and torch.allclose(m0.solve(r, r2).double(), _dense_v(H, r, mask, torch.full_like(nv, 0.01), r2), atol=1e-4, rtol=1e-4)


def test_general_path_with_a_state_subset(small_R):
    torch.manual_seed(2)
    H, b = _ops(small_R); H, b = H[:10], b[:10]
    mask = (torch.rand(1, 10, 3, 3) < 0.6).float(); nv = 0.01 + 0.02 * torch.rand(1, 10, 3, 3)
    r = torch.randn(1, 10, 3, 3); r2 = torch.tensor([0.4])
    m = G.LinearMeasurement(H, b, torch.randn(1, 10, 3, 3) * mask, mask, 0.1, noise_var=nv)
    assert torch.allclose(m.solve(r, r2).double(), _dense_v(H, r, mask, nv, r2), atol=1e-4, rtol=1e-4)
    assert m.apply(torch.randn(1, 64, 3, 3)).shape == (1, 10, 3, 3)
    seq = G.LinearMeasurement(H, b, torch.randn(1, 10, 3, 3), None, 0.1)            # the sequential path takes any row count too
    assert seq.solve(r, r2).shape == (1, 10, 3, 3)


def test_refuses_a_mask_that_is_not_0_1_or_has_the_wrong_rows(small_R):
    H, b = _ops(small_R); y = torch.randn(1, 30, 2, 2)
    with pytest.raises(ValueError, match='mask'):
        G.LinearMeasurement(H, b, y, 0.5 * torch.ones_like(y), 0.1)
    with pytest.raises(ValueError, match='mask'):
        G.LinearMeasurement(H, b, y, torch.zeros(1, 29, 2, 2), 0.1)
    with pytest.raises(ValueError, match='noise_var'):
        G.LinearMeasurement(H, b, y, None, 0.1, noise_var=torch.ones(1, 30, 2, 3))


def test_dps_refuses_a_general_measurement(small_R):
    """Fix round 1, M4: DPS uses the unweighted residual norm, so it would ignore per-element variances."""
    tr = DiffusionTrainer(device=torch.device('cpu'), prediction_type='v', snr_gamma=None); H, b = _ops(small_R)
    den = T.GaussianDenoiser(torch.zeros(64), torch.eye(64), tr); y = torch.randn(1, 30, 4, 4)
    m = G.LinearMeasurement(H, b, y, None, 0.1, noise_var=torch.full_like(y, 0.01))
    with pytest.raises(ValueError, match='general'):
        G.dps_sample(den, tr, m, (1, 64, 4, 4), 5, seed=0, clamp=None)
    with pytest.raises(ValueError, match='general'):
        G.draw_guided(den, tr, m, (1, 64, 4, 4), K=1, n_steps=5, method='dps', clamp=None)


def test_pigdm_with_a_stride_mask_reduces_the_measured_residual(small_R):
    tr = DiffusionTrainer(device=torch.device('cpu'), prediction_type='v', snr_gamma=None); H, b = _ops(small_R)
    den = T.GaussianDenoiser(torch.zeros(64), torch.eye(64), tr)
    pix = torch.zeros(1, 1, 8, 8); pix[..., ::4, ::4] = 1.0
    mask = pix.expand(1, 30, 8, 8).contiguous()
    y = torch.randn(1, 30, 8, 8) * mask
    m = G.LinearMeasurement(H, b, y, mask, 0.1, noise_var=torch.full_like(y, 0.01))
    assert m.general
    x_g = G.pigdm_sample(den, tr, m, (1, 64, 8, 8), 20, seed=0, clamp=None)
    x_u = tr.sample(den, {'sensor': None}, (1, 64, 8, 8), n_steps=20, method='ddim', progress=False, clamp_x0=(-1e9, 1e9))
    assert torch.isfinite(x_g).all() and m.residual(x_g).norm() < m.residual(x_u).norm()
