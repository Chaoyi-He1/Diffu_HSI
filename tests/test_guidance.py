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
    assert out['rmse_excess'] < 0.5 and out['nfe'] == 20

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
