### Task 9: Guidance samplers (ΠGDM, DPS) and the toy that licenses them

**Files:**
- Create: `model/guidance.py`, `scripts/sensor/toy_guidance.py`
- Test: `tests/test_guidance.py`

**Interfaces:**
- Consumes: `DiffusionTrainer` (`alpha_bars`, `_model_to_x0_eps`, the DDIM grid rule `ts = range(999, -1, -(1000 // n_steps))`), `sensor_forward.{apply_H, apply_HT}`.
- Produces in `model/guidance.py`: `class LinearMeasurement` with `__init__(self, H: torch.Tensor [30, 64], b: torch.Tensor [30], y: torch.Tensor [B, 30, H, W], mask: torch.Tensor | None [B, 30, H, W], sigma_y: float)` (stores `y_eff = (y − b[None, :, None, None]) ⊙ mask`), `apply(self, x64) -> [B, 30, H, W]` (= apply_H(x64) ⊙ mask), `residual(self, x0) -> y_eff − apply(x0)`, `solve(self, resid, r2: torch.Tensor [B]) -> v [B, 30, H, W]` — mask None / all-ones: v = (r2 H Hᵀ + σ² I)⁻¹ resid, one 30 × 30 solve per cube broadcast over pixels (`torch.linalg.solve`); one-hot mask: v = resid / (r2 ‖h_k‖² + σ²) on the measured row k of each pixel, 0 elsewhere (‖h_k‖² = row norms of H gathered by the mask) — the two cases are distinguished by `mask.sum(1).max() == 1`; `def pigdm_sample(model, trainer, meas: LinearMeasurement, shape, n_steps: int, seed: int, clamp=(-1.0, 1.0), progress=False) -> torch.Tensor` (x = randn(shape, generator seeded with `seed`); for each t in the DDIM grid: x.requires_grad_(True); x0 = clamp(trainer._model_to_x0_eps(model(x, {'sensor': None}, t), x, t)[0]); v = meas.solve(meas.residual(x0), r2 = 1 − ᾱ_t); g = torch.autograd.grad((x0 * apply_HT(v ⊙ mask, H)).sum(), x)[0]; x0g = clamp(x0 + √ᾱ_t · g); DDIM η = 0 step to t_prev with eps consistent with x0g (as `DiffusionTrainer.sample`'s DDIM branch); returns x0g of the last step; sets `trainer.last_nfe = len(ts)`; `clamp=None` disables clamping); `def dps_sample(model, trainer, meas, shape, n_steps, seed, zeta: float = 1.0, clamp=(-1.0, 1.0)) -> torch.Tensor` (Chung et al. 2022: x0 = D(x_t); nrm = ‖meas.residual(x0)‖₂ per cube; g = grad(nrm.sum(), x_t); unguided DDIM step to x'_{t−1}; x_{t−1} = x'_{t−1} − ζ · g); `def draw_guided(model, trainer, meas, shape, K: int, n_steps: int, method: str = 'pigdm', clamp=(-1.0, 1.0)) -> torch.Tensor [K, B, 64, H, W]` (seeds 0 … K − 1).
- Produces in `scripts/sensor/toy_guidance.py`: `class GaussianDenoiser(torch.nn.Module)` with `__init__(self, mu: torch.Tensor [64], C: torch.Tensor [64, 64], trainer)` and `forward(x_t, cond, t)` returning the v-prediction of the exact posterior mean x̂₀ = μ + √ᾱ C (ᾱ C + (1 − ᾱ) I)⁻¹ (x_t − √ᾱ μ) per pixel (v = (√ᾱ x_t − x̂₀) / √(1 − ᾱ)); `def exact_posterior(mu, C, H, b, y, sigma, mask) -> tuple[mean [B, 64, H, W], var [B, 64, H, W]]` (per pixel: sequential → one 64-dim Gaussian conditioning; snapshot → per-row conditioning); `def run_toy(trainer, H, b, mode: str, snr_db: float, n_steps: int, K: int, B: int, hw: int, seed: int, method: str, C_kind: str = 'isotropic') -> dict(rmse_excess, var_ratio, cov50, cov90, crps_ratio, nfe)` where x₀ ~ N(0, I₆₄) (`isotropic`) or N(μ₆₄, C₆₄) from a prior npz lifted by A (`hfd`), y = H x₀ + b + n with σ from the SNR against P_y of the toy's own y, samples via `draw_guided(..., clamp=None)`, `rmse_excess = RMSE(mean-of-K vs x₀) / RMSE(exact mean vs x₀) − 1`, `var_ratio = mean(sample var) / mean(exact var)`, cov50 / cov90 by `metrics.coverage_samples`, `crps_ratio = CRPS(samples) / CRPS(exact Gaussian)`; CLI: `--out results/residual_warmstart/sensor/toy_guidance.json --steps 50 100 --K 250 --B 4 --hw 8 --device cuda:0` runs ΠGDM and DPS, both modes, isotropic (gating) and hfd (report) priors at 30 dB, writes the JSON (refusing to overwrite) and prints `steps_for_B = <smallest step count whose ΠGDM isotropic results meet: rmse_excess <= 0.02, |var_ratio - 1| <= 0.10, 0.85 <= cov90 <= 0.95, 0.45 <= cov50 <= 0.55 in BOTH modes>` or `FAIL: use DPS` when neither 50 nor 100 does.

- [ ] **Step 1: Write the failing tests**

```python
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
    H, b = _ops(small_R); y = torch.randn(2, 30, 3, 3); m = G.LinearMeasurement(H, b, y, None, 0.1)
    r = torch.randn(2, 30, 3, 3); r2 = torch.tensor([0.3, 0.7]); v = m.solve(r, r2)
    for i in range(2):
        A = r2[i] * H @ H.T + 0.01 * torch.eye(30); ref = torch.linalg.solve(A, r[i].reshape(30, -1)).reshape(30, 3, 3)
        assert torch.allclose(v[i], ref, atol=1e-4)

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
```

- [ ] **Step 2: Run to verify failure.** Expected: FAIL, modules missing.

- [ ] **Step 3: Implement `model/guidance.py` and `scripts/sensor/toy_guidance.py`** per Interfaces (the DDIM update copied in form from `DiffusionTrainer.sample`'s DDIM branch, with `x0g` in place of `x0_pred`; gradients only through the one `autograd.grad` per step, the rest under `torch.no_grad`).

- [ ] **Step 4: Run to verify pass.** Expected: 6 passed (the isotropic test is the one that matters; if it fails on tolerance, the implementation is wrong, not the tolerance).

- [ ] **Step 5: Commit** — `git add model/guidance.py scripts/sensor/toy_guidance.py tests/test_guidance.py` ; `git commit -m "feat(sensor): ΠGDM and DPS guidance over the linear measurement (sequential and one-hot snapshot solves) and the Gaussian toy that licenses the calibration clauses (Phase 3 Task 9)"`.

