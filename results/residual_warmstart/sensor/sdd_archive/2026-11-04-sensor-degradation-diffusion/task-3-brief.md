### Task 3: Noise-aware MMSE — sequential and snapshot posteriors, lift to 64 bands, demosaic-then-MMSE

**Files:**
- Create: `data_loader/mmse.py`
- Test: `tests/test_mmse.py`

**Interfaces:**
- Consumes: `fit_linear_prior.make_gain`'s whitened-SVD approach (re-implemented locally for an arbitrary noise variance; `scripts/` is not imported from `data_loader/`), `linear_estimate.expand_matrix`, `sensor_forward.mosaic_mask`.
- Produces (numpy float64, x₀₁ scale unless stated): `def prior01(prior: dict) -> tuple[np.ndarray, np.ndarray]` (mu01 = (mu + 1) / 2, C01 = C / 4); `def posterior_sequential(y: np.ndarray, R: np.ndarray, mu01, C01, noise_var) -> tuple[np.ndarray, np.ndarray]` (y [N, 30]; noise_var a float or a [30] vector (per-channel variance); returns xhat [N, 31], Sigma [31, 31]; K = C R (Rᵀ C R + diag(noise_var))⁻¹ via Cholesky of C and an SVD of the whitened system, Sigma = C − K Rᵀ C); `def posterior_snapshot(y_k: np.ndarray, k: np.ndarray, R, mu01, C01, noise_var: float) -> tuple[np.ndarray, np.ndarray]` (y_k [N], k [N] int channel index; per pixel with r = R[:, k]: g = C r / (rᵀ C r + σ²), xhat = mu + g (y_k − rᵀ mu); returns xhat [N, 31] and Sigma_n [N, 31, 31] = C − g rᵀ C); `def lift_mean_std(xhat31: np.ndarray, Sigma, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]` (Sigma [31, 31] shared or [N, 31, 31] per pixel; returns xhat64 [N, 64] = xhat31 @ A.T and std64 [N, 64] = sqrt(diag(A Sigma Aᵀ)) per pixel); `def mmse_maps(y: np.ndarray, mask: np.ndarray | None, R, prior: dict, sigma: float, A) -> tuple[np.ndarray, np.ndarray]` (y [H, W, 30]; mask None or all-ones → sequential, one-hot → snapshot; returns xhat64 and std64 as [H, W, 64] float32 **on the trainer scale**: 2 xhat01 − 1 and 2 std01); `def demosaic(y_m: np.ndarray, mask: np.ndarray) -> np.ndarray` ([H, W, 30]: per channel k, `scipy.interpolate.RegularGridInterpolator` over the channel's measured rows and columns (`method='linear', bounds_error=False, fill_value=None`) evaluated on the full grid); `def demosaic_then_mmse(y_m, mask, R, prior, sigma: float, A, e2: np.ndarray) -> tuple[np.ndarray, np.ndarray]` (sequential posterior of the demosaiced y with noise_var = σ² + e2 per channel; same trainer-scale outputs as `mmse_maps`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mmse.py
import numpy as np, pytest
from data_loader import mmse as MM
from data_loader.linear_estimate import expand_matrix, linear_inverse
from data_loader.sensor_forward import mosaic_mask, sigma_from_snr

def _toy(seed=0, L=5, S=3):
    rng = np.random.default_rng(seed); B = rng.standard_normal((L, L)); C = B @ B.T + 0.1 * np.eye(L)
    return rng.standard_normal(L), C, rng.random((L, S))

def test_sequential_posterior_matches_bruteforce_gaussian_conditioning():
    mu, C, R = _toy(); s2 = 0.05
    Syy = R.T @ C @ R + s2 * np.eye(3); Sxy = C @ R
    y = np.random.default_rng(1).standard_normal((4, 3))
    xhat, Sig = MM.posterior_sequential(y, R, mu, C, s2)
    np.testing.assert_allclose(xhat, mu + (y - mu @ R) @ np.linalg.solve(Syy, Sxy.T), atol=1e-9)
    np.testing.assert_allclose(Sig, C - Sxy @ np.linalg.solve(Syy, Sxy.T), atol=1e-9)

def test_per_channel_noise_vector_is_accepted():
    mu, C, R = _toy(); y = np.zeros((2, 3))
    a, _ = MM.posterior_sequential(y, R, mu, C, np.array([0.1, 0.2, 0.3])); b, _ = MM.posterior_sequential(y, R, mu, C, 0.1)
    assert not np.allclose(a, b)

def test_snapshot_posterior_equals_sequential_with_one_row():
    mu, C, R = _toy(); y = np.random.default_rng(2).standard_normal(6); k = np.array([0, 1, 2, 0, 1, 2])
    xs, Sn = MM.posterior_snapshot(y, k, R, mu, C, 0.05)
    for n in range(6):
        xr, Sr = MM.posterior_sequential(y[n:n + 1, None], R[:, k[n]:k[n] + 1], mu, C, 0.05)
        np.testing.assert_allclose(xs[n], xr[0], atol=1e-9); np.testing.assert_allclose(Sn[n], Sr, atol=1e-9)

def test_sigma_to_prior_regulariser_reproduces_linear_inverse(small_R, synthetic_prior):
    mu, C = synthetic_prior['mu'], synthetic_prior['C']                     # trainer-scale prior, as the npz
    s = 3e-7
    from scripts.fit_linear_prior import make_gain
    K = make_gain(C, small_R, s)
    x_tr = np.clip(synthetic_prior['X'][:50], -0.9, 0.9); y_tr = x_tr @ small_R
    ref = linear_inverse(y_tr, mu, K, small_R)                                 # trainer scale, clipped (inactive here)
    mu01, C01 = MM.prior01(dict(mu=mu, C=C)); y01 = (y_tr + small_R.sum(0)) / 2
    xhat01, _ = MM.posterior_sequential(y01, small_R, mu01, C01, (s / 2) ** 2)      # C01 = C / 4, so nu = s^2 / 4 reproduces K(s)
    np.testing.assert_allclose(2 * xhat01 - 1, ref, atol=1e-4)

def test_posterior_std_grows_as_snr_falls_and_is_finite_at_the_floor(small_R, synthetic_prior):
    mu01, C01 = MM.prior01(dict(mu=synthetic_prior['mu'], C=synthetic_prior['C'])); P_y = 50.0
    stds = [np.sqrt(np.diag(MM.posterior_sequential(np.zeros((1, 30)), small_R, mu01, C01, sigma_from_snr(snr, P_y) ** 2)[1])).mean() for snr in (60, 40, 20)]
    assert np.all(np.isfinite(stds)) and stds[0] < stds[1] < stds[2]

def test_mmse_maps_shapes_scales_and_mask_dispatch(small_R, synthetic_prior):
    prior = dict(mu=synthetic_prior['mu'], C=synthetic_prior['C']); A = expand_matrix(np.linspace(451, 855, 31), 64)
    y = np.random.default_rng(0).random((10, 12, 30)); m = mosaic_mask(10, 12)
    xs, ss = MM.mmse_maps(y, None, small_R, prior, 0.1, A); xm, sm = MM.mmse_maps(y * m, m, small_R, prior, 0.1, A)
    assert xs.shape == ss.shape == xm.shape == sm.shape == (10, 12, 64) and xs.dtype == np.float32
    assert ss.min() > 0 and sm.mean() > ss.mean()                               # one measurement per pixel is less informative

def test_demosaic_recovers_a_linear_ramp_in_the_interior():
    H, W = 20, 24; m = mosaic_mask(H, W); ramp = np.add.outer(np.arange(H), 2 * np.arange(W)).astype(float)
    y_full = np.repeat(ramp[..., None], 30, -1); y_m = y_full * m
    d = MM.demosaic(y_m, m)
    np.testing.assert_allclose(d[5:15, 6:18], y_full[5:15, 6:18], atol=1e-6)

def test_demosaic_then_mmse_runs(small_R, synthetic_prior):
    prior = dict(mu=synthetic_prior['mu'], C=synthetic_prior['C']); A = expand_matrix(np.linspace(451, 855, 31), 64)
    m = mosaic_mask(10, 12); y = np.random.default_rng(0).random((10, 12, 30)) * m
    xh, s = MM.demosaic_then_mmse(y, m, small_R, prior, 0.1, A, np.full(30, 0.01))
    assert xh.shape == (10, 12, 64) and np.isfinite(xh).all() and s.min() > 0
```

- [ ] **Step 2: Run to verify failure.** Expected: FAIL, module missing.

- [ ] **Step 3: Implement `data_loader/mmse.py`** per Interfaces. `posterior_sequential`: L = chol(C + 1e-12 I); whiten the noise with D = diag(noise_var)^(-1/2): M = D Rᵀ L (30 × 31), SVD M = U S Vᵀ, K = L Vᵀᵀ diag(S / (S² + 1)) Uᵀ D (the s = 1 whitened form of `make_gain`); Sigma = C − K Rᵀ C. `posterior_snapshot` vectorised over pixels with `np.einsum`. `lift_mean_std` for the per-pixel case: `np.einsum('ij,njk,ik->ni', A, Sigma, A)`.

- [ ] **Step 4: Run to verify pass.** Expected: 8 passed.

- [ ] **Step 5: Commit** — `git add data_loader/mmse.py tests/test_mmse.py` ; `git commit -m "feat(sensor): noise-aware MMSE — sequential and snapshot posteriors, 64-band mean / std lift, demosaic-then-MMSE (Phase 3 Task 3)"`.

