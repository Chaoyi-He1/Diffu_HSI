# Residual Warm-Start Diffusion — Phase 0 (Foundation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and test everything Phase 1 needs — the linear prior, the residual dataset, the additive model/sampler changes, the evaluation harness with the baseline — and record the official baselines and training-step timings.

**Architecture:** A per-pixel linear inverse of the low-resolution sensor (plus grid-aligned bilinear upsampling) gives a cheap estimate x̂₀ of the cube. A new dataset subclass returns (x, Y_d, x̂₀); a thin wrapper module feeds [r_t, x̂₀] and the sensor context to the existing U-Net so `DiffusionTrainer` is unchanged except for an additive warm-start option in `sample()`. An evaluation harness scores the baseline and any checkpoint on the loader's validation split with the full sampler grid and a y-swap check.

**Tech Stack:** Python 3.10 (`/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python`), PyTorch 2.8 (CUDA), NumPy, SciPy, pytest 9. Dataset at `dataset/HFD100 Mat dataset` (symlink or copy; see Global Constraints).

**Spec:** `docs/superpowers/specs/2026-10-08-residual-warm-start-design.md`

## Global Constraints

- The current `HFD_data` loader is used unchanged: `MatFlower60/Train`, sorted file list, first 10 000 files, in-order 90/10 split, `R_n=1` (R_Device1), 64 interpolated bands, strided sensor downsampling `[::d, ::d]`.
- No noise is ever added to the sensor values; `y = x @ R` exactly as the loader computes it.
- Target rates d ∈ {1, 2, 4, 8}; d = 4 is the first training target.
- All numbers on the loader's scale: cubes in [−1, 1] per-cube min–max; RMSE reported in % of the [0, 1] display scale `(x + 1) / 2`.
- Existing files `model/u2net_hyperspectral.py` and `model/diffusion_trainer.py` may only be changed additively: every existing call site keeps its behaviour and every existing checkpoint still loads with `strict=True`.
- Training is on this server (2 × RTX A4500); FSDP across both GPUs for large models. Before any GPU use, check `nvidia-smi` and do not start on a GPU with less than 3 GB free.
- Outputs go under `results/residual_warmstart/` (git-ignored). The stored validation subset list and the baseline record go under `tests/data/` (committed).
- The repo root in this worktree has no `dataset/` entry; the dataset lives at `/data/chaoyi_he/HSI/Diffu/dataset`. Every script and test resolves the dataset root as `os.environ.get('HFD_DATA_ROOT', '/data/chaoyi_he/HSI/Diffu/dataset/HFD100 Mat dataset')`. Do not add a `dataset` symlink to the repo (`.gitignore` ignores `dataset/*`, not the link itself).
- Run tests with `cd <repo> && /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests -q`. Tests that read the dataset are marked `@pytest.mark.dataset` and skip if the root is missing.
- Commit after every task with the attribution lines from the session reminder.

## Review Focus

Inputs the spec implies but no task's tests would otherwise exercise; each line's test is added to the owning task:

1. **Non-divisible sizes:** 64 is not a multiple of d = 3 or 6, and `[::d]` then yields ⌈64/d⌉ samples. `lift_to_grid` must handle any `n_lo = ceil(H/d)` without an off-by-one at the right/bottom edge (Task 2 test `test_lift_non_divisible`).
2. **d = 1 is the identity:** the estimate path at d = 1 must return the per-pixel inverse unchanged, with no interpolation blur (Task 2 test `test_lift_identity_d1`).
3. **Constant cubes:** the loader's per-cube normalisation divides by `max − min + 1e-20`; a flat patch becomes all −1 and `y` is constant. The linear inverse must not produce NaN/inf for such input (Task 1 test `test_inverse_constant_pixel`).
4. **Zero-residual network:** if the network outputs r̂₀ = 0, the harness must return exactly the baseline, and the sampler's warm start with `t_start=0` must return `x_init` unchanged (Task 5 test `test_warm_start_t0_zero_returns_init`, Task 7 test `test_zero_residual_equals_baseline`).
5. **Batch size 1 in the y-swap check:** `roll(1, 0)` on a batch of one is a no-op and would pass a broken model; the check must refuse batches smaller than 2 (Task 7 test `test_yswap_requires_batch_ge_2`).

---

## File Structure

| file | responsibility |
|---|---|
| `scripts/fit_linear_prior.py` | Fit (μ, C) on training pixels of the loader's training split; choose s on held-out training files; compute K (whitened SVD); compute σ_d; write `results/residual_warmstart/linear_prior_R1.npz` |
| `data_loader/linear_estimate.py` | Pure functions: `linear_inverse(y, prior)`, `lift_to_grid(x_lo, d, H, W)`, `estimate_from_sensor(y_d, prior, d, H, W, band_matrix)` and `expand_matrix()` (the loader's 31→64 interpolation as a matrix) |
| `data_loader/hfd_residual.py` | `HFDResidualData(HFD_data)`: returns `(x64, y_d, x0_hat)`; `residual_collate_fn` |
| `model/u2net_hyperspectral.py` | `extra_in_channels=0` argument widening `input_proj` (additive) |
| `model/residual_wrapper.py` | `ResidualWrapper(net, x0_hat_channels)`: `forward(x_t, cond, t)` where `cond = {'x0_hat': ..., 'sensor': ...}`; concatenates x̂₀ into the input and passes the sensor as context |
| `model/diffusion_trainer.py` | `sample(..., x_init=None, t_start=None)` warm start (additive) |
| `scripts/eval_warmstart.py` | Baseline and checkpoint evaluation over the stored validation subset; sampler grid; metrics; y-swap; JSON output |
| `scripts/time_train_step.py` | Times one training step for `base_channels` 64 (1 GPU) and 256 (FSDP, 2 GPUs) |
| `tests/` | `conftest.py`, `test_linear_prior.py`, `test_linear_estimate.py`, `test_residual_dataset.py`, `test_model_extra_in.py`, `test_wrapper.py`, `test_warm_start_sampler.py`, `test_eval_harness.py`, `data/val_subset_200.json`, `data/baseline_phase0.json` |

Task order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9. Tasks 4 and 5 are independent of each other.

---

### Task 1: Linear prior fit script

**Files:**
- Create: `scripts/fit_linear_prior.py`
- Create: `tests/conftest.py`
- Test: `tests/test_linear_prior.py`

**Interfaces:**
- Consumes: `data_loader.HFD_dataset.HFD_data` (file list, split, `sensor_R_matrix` [31, 30], `wavelens`).
- Produces: `fit_prior(train_files, R, n_files, px_per_file, seed) -> dict(mu[31], C[31,31])`; `make_gain(C, R, s) -> K[31,30]` (whitened SVD); `choose_s(prior, R, heldout_files, candidates) -> float`; the saved npz with keys `mu, C, K, s, R, sigma_d, d_values, n_fit_files, n_fit_pixels, seed`. `sigma_d` is filled in Task 3 (this task writes it as an empty array).

- [ ] **Step 1: Write the test fixtures**

`tests/conftest.py`:

```python
import os
import sys
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

DATA_ROOT = os.environ.get('HFD_DATA_ROOT', '/data/chaoyi_he/HSI/Diffu/dataset/HFD100 Mat dataset')


def pytest_configure(config):
    config.addinivalue_line('markers', 'dataset: needs the HFD100 dataset on disk')


def pytest_collection_modifyitems(config, items):
    if os.path.isdir(DATA_ROOT):
        return
    skip = pytest.mark.skip(reason=f'dataset not found at {DATA_ROOT}')
    for item in items:
        if 'dataset' in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope='session')
def data_root():
    return DATA_ROOT


@pytest.fixture(scope='session')
def small_R():
    """A deterministic 31x30 sensor matrix with the real matrix's shape and [0, 1] column scaling."""
    rng = np.random.default_rng(0)
    wl = np.linspace(0, 1, 31)
    centers = np.linspace(0, 1, 30)
    R = np.exp(-((wl[:, None] - centers[None, :]) ** 2) / (2 * 0.08 ** 2))  # smooth, overlapping filters
    R = (R - R.min(0)) / (R.max(0) - R.min(0))
    return R


@pytest.fixture(scope='session')
def synthetic_prior(small_R):
    """Gaussian spectra living on a 6-dim subspace, so the 30-channel sensor identifies them."""
    rng = np.random.default_rng(1)
    basis = rng.standard_normal((31, 6))
    coef = rng.standard_normal((20000, 6))
    X = 0.2 * coef @ basis.T
    mu = X.mean(0)
    C = np.cov(X.T) + 1e-6 * np.eye(31)
    return dict(mu=mu, C=C, X=X)
```

- [ ] **Step 2: Write the failing tests**

`tests/test_linear_prior.py`:

```python
import numpy as np
import pytest

from scripts.fit_linear_prior import make_gain, choose_s, fit_prior


def test_make_gain_shape_and_noiseless_recovery(small_R, synthetic_prior):
    mu, C, X = synthetic_prior['mu'], synthetic_prior['C'], synthetic_prior['X']
    K = make_gain(C, small_R, s=1e-6)
    assert K.shape == (31, 30)
    Y = X @ small_R
    Xh = mu + (Y - mu @ small_R) @ K.T
    rel = np.linalg.norm(Xh - X, axis=1) / np.linalg.norm(X, axis=1)
    assert rel.mean() < 0.05          # 6-dim spectra seen through 30 channels are recoverable


def test_make_gain_is_finite_for_tiny_s(small_R, synthetic_prior):
    K = make_gain(synthetic_prior['C'], small_R, s=0.0)
    assert np.isfinite(K).all()


def test_inverse_constant_pixel(small_R, synthetic_prior):
    mu, C = synthetic_prior['mu'], synthetic_prior['C']
    K = make_gain(C, small_R, s=1e-6)
    y = (-np.ones(31)) @ small_R        # a flat patch: every band is -1 after the loader's normalisation
    xh = mu + (y - mu @ small_R) @ K.T
    assert np.isfinite(xh).all()


def test_choose_s_picks_a_candidate(small_R, synthetic_prior):
    rng = np.random.default_rng(2)
    held = synthetic_prior['X'][:2000]
    s = choose_s(synthetic_prior, small_R, held, candidates=[1e-7, 1e-5, 1e-3, 1e-1])
    assert s in [1e-7, 1e-5, 1e-3, 1e-1]


@pytest.mark.dataset
def test_fit_prior_on_real_files(data_root):
    from data_loader.HFD_dataset import HFD_data
    ds = HFD_data(data_path=data_root, train_mode='image', eval_ratio=0.1, split='train',
                  data_format='image', type='Flower', R_n=1, sensor_down_sample_rate=1)
    prior = fit_prior(ds.img_list, ds.sensor_R_matrix, n_files=20, px_per_file=64, seed=0)
    assert prior['mu'].shape == (31,) and prior['C'].shape == (31, 31)
    assert np.all(np.linalg.eigvalsh(prior['C']) > -1e-8)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_linear_prior.py -q`
Expected: errors with `ModuleNotFoundError: No module named 'scripts.fit_linear_prior'` (add an empty `scripts/__init__.py` in the next step so the package imports).

- [ ] **Step 4: Write the script**

Create `scripts/__init__.py` (empty) and `scripts/fit_linear_prior.py`:

```python
"""Fit the per-pixel linear Gaussian prior used for the warm-start estimate.

Writes results/residual_warmstart/linear_prior_R1.npz with mu, C, K, s, R (and sigma_d, filled by
scripts/compute_residual_scale.py in Task 3). Fitted on pixels of the loader's TRAINING split only.
Run from the repo root:
    python scripts/fit_linear_prior.py [--n_files 3000] [--px_per_file 256]
"""
import argparse
import os
import random
import sys

import numpy as np
import scipy.io as sio

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data_loader.HFD_dataset import HFD_data  # noqa: E402

DATA_ROOT = os.environ.get('HFD_DATA_ROOT', '/data/chaoyi_he/HSI/Diffu/dataset/HFD100 Mat dataset')
OUT_DIR = os.path.join(REPO, 'results', 'residual_warmstart')
S_CANDIDATES = [1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 1e-3]


def load_cube(path):
    """The loader's per-cube normalisation, on the 31 native bands: [H, W, 31] in [-1, 1]."""
    g = np.array(sio.loadmat(path)['truth'], dtype=np.float64)
    return (g - g.min()) / (g.max() - g.min() + 1e-20) * 2.0 - 1.0


def fit_prior(train_files, R, n_files=3000, px_per_file=256, seed=0):
    """Mean and covariance of 31-band spectra from random pixels of random training files."""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    files = rng.sample(list(train_files), min(n_files, len(train_files)))
    P = []
    for f in files:
        x = load_cube(f).reshape(-1, 31)
        P.append(x[nrng.choice(len(x), px_per_file, replace=False)])
    P = np.concatenate(P)
    return dict(mu=P.mean(0), C=np.cov(P.T), n_fit_files=len(files), n_fit_pixels=len(P), seed=seed)


def make_gain(C, R, s):
    """K = C R (R^T C R + s^2 I)^-1 via a whitened SVD, stable for the ill-conditioned R."""
    L = np.linalg.cholesky(C + 1e-12 * np.eye(C.shape[0]))
    U, S, Vt = np.linalg.svd(R.T @ L, full_matrices=False)       # R^T L = U S V^T
    filt = S / (S ** 2 + s ** 2) if s > 0 else np.where(S > S.max() * 1e-13, 1.0 / np.maximum(S, 1e-300), 0.0)
    return L @ Vt.T @ np.diag(filt) @ U.T                           # [31, 30]


def linear_estimate(Y, prior, K, R):
    """Y: [..., 30] -> [..., 31] on the [-1, 1] scale (not clipped)."""
    return prior['mu'] + (Y - prior['mu'] @ R) @ K.T


def choose_s(prior, R, heldout_pixels, candidates=S_CANDIDATES):
    """Pick the regulariser with the lowest per-element RMSE on held-out spectra [N, 31]."""
    Y = (heldout_pixels @ R).astype(np.float32).astype(np.float64)  # float32 like the loader's collate
    best, best_err = None, np.inf
    for s in candidates:
        Xh = linear_estimate(Y, prior, make_gain(prior['C'], R, s), R)
        err = np.sqrt(((Xh - heldout_pixels) ** 2).mean())
        if err < best_err:
            best, best_err = s, err
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_files', type=int, default=3000)
    ap.add_argument('--px_per_file', type=int, default=256)
    ap.add_argument('--n_heldout_files', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'linear_prior_R1.npz'))
    args = ap.parse_args()

    ds = HFD_data(data_path=DATA_ROOT, train_mode='image', eval_ratio=0.1, split='train',
                  data_format='image', type='Flower', R_n=1, sensor_down_sample_rate=1)
    R = np.asarray(ds.sensor_R_matrix, dtype=np.float64)              # [31, 30]
    files = list(ds.img_list)
    rng = random.Random(args.seed + 1)
    rng.shuffle(files)
    heldout, fit_files = files[:args.n_heldout_files], files[args.n_heldout_files:]

    prior = fit_prior(fit_files, R, args.n_files, args.px_per_file, args.seed)
    held_px = np.concatenate([load_cube(f).reshape(-1, 31)[::64] for f in heldout])
    s = choose_s(prior, R, held_px)
    K = make_gain(prior['C'], R, s)
    Xh = linear_estimate((held_px @ R).astype(np.float32).astype(np.float64), prior, K, R)
    err = np.sqrt(((Xh - held_px) ** 2).mean())
    print(f'fit on {prior["n_fit_pixels"]} pixels of {prior["n_fit_files"]} files; s = {s:g}; '
          f'held-out per-element RMSE {err:.5f} ([-1,1] scale) = {err / 2 * 100:.3f} % of range')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, mu=prior['mu'], C=prior['C'], K=K, s=np.float64(s), R=R,
             sigma_d=np.zeros(0), d_values=np.zeros(0, dtype=int),
             n_fit_files=prior['n_fit_files'], n_fit_pixels=prior['n_fit_pixels'], seed=args.seed)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_linear_prior.py -q`
Expected: 5 passed (the dataset test runs because the dataset exists on this server).

- [ ] **Step 6: Run the script for real**

Run: `cd <repo> && /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python scripts/fit_linear_prior.py`
Expected: prints a chosen `s` in the 1e-8 … 1e-6 range and a held-out RMSE near 0.3 % of range (the Phase-0 note's d = 1 figure was 0.31 %); writes `results/residual_warmstart/linear_prior_R1.npz`. Record the printed line in the commit message.

- [ ] **Step 7: Commit**

```bash
git add scripts/__init__.py scripts/fit_linear_prior.py tests/conftest.py tests/test_linear_prior.py
git commit -m "feat(warmstart): linear Gaussian prior fit with whitened-SVD gain

<paste the printed fit line>

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JwXhqBayn3pVe5Y9kfV6YP"
```

---

### Task 2: Estimate functions (inverse, band expansion, grid-aligned lift)

**Files:**
- Create: `data_loader/linear_estimate.py`
- Test: `tests/test_linear_estimate.py`

**Interfaces:**
- Consumes: the npz from Task 1 (`mu, K, R`).
- Produces:
  - `expand_matrix(wavelens=np.linspace(451, 855, 31), n_out=64) -> np.ndarray [64, 31]` reproducing `HFD_data.expand_wavelens` exactly;
  - `linear_inverse(y, mu, K, R) -> np.ndarray [..., 31]`, clipped to [−1, 1];
  - `lift_to_grid(x_lo, d, H, W) -> np.ndarray [H, W, C]` grid-aligned bilinear upsampling of `x_lo [h, w, C]` where `x_lo[i, j]` sits at high-res `(i*d, j*d)`; edge extension beyond the last sample; identity for d = 1;
  - `estimate_from_sensor(y_d, prior, d, H, W, A) -> np.ndarray [H, W, 64]` = `lift_to_grid(linear_inverse(y_d) @ A.T, d, H, W)`.
  - `load_prior(path) -> dict` with numpy arrays `mu, C, K, s, R, sigma_d, d_values`.

- [ ] **Step 1: Write the failing tests**

`tests/test_linear_estimate.py`:

```python
import numpy as np
import pytest

from data_loader.linear_estimate import expand_matrix, linear_inverse, lift_to_grid, estimate_from_sensor


def test_expand_matrix_matches_loader_interpolation():
    from scipy.interpolate import interp1d
    wl = np.linspace(451, 855, 31)
    A = expand_matrix(wl, 64)
    assert A.shape == (64, 31)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(31)
    ref = interp1d(wl, x, kind='linear', bounds_error=False, fill_value='extrapolate')(np.linspace(451, 855, 64))
    assert np.allclose(A @ x, ref, atol=1e-12)


def test_linear_inverse_is_clipped(small_R, synthetic_prior):
    from scripts.fit_linear_prior import make_gain
    mu, C = synthetic_prior['mu'], synthetic_prior['C']
    K = make_gain(C, small_R, 1e-6)
    y = (3.0 * np.ones(31)) @ small_R           # far outside [-1, 1]
    xh = linear_inverse(y[None, :], mu, K, small_R)
    assert xh.shape == (1, 31) and xh.max() <= 1.0 and xh.min() >= -1.0


def test_lift_identity_d1():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 64, 3))
    assert np.array_equal(lift_to_grid(x, 1, 64, 64), x)


def test_lift_reproduces_ramp_exactly():
    H = W = 64
    ramp = np.arange(H, dtype=np.float64)[:, None].repeat(W, 1)[..., None]   # linear in rows
    for d in (2, 4, 8):
        lo = ramp[::d, ::d]
        up = lift_to_grid(lo, d, H, W)
        last = (lo.shape[0] - 1) * d
        assert np.allclose(up[:last + 1], ramp[:last + 1])         # exact between samples
        assert np.allclose(up[last:], ramp[last])                   # edge-extended after the last sample


def test_lift_non_divisible():
    H = W = 64
    rng = np.random.default_rng(1)
    x = rng.standard_normal((H, W, 2))
    for d in (3, 6):
        lo = x[::d, ::d]                                            # ceil(64/d) samples
        up = lift_to_grid(lo, d, H, W)
        assert up.shape == (H, W, 2)
        assert np.allclose(up[::d, ::d], lo)                        # samples land back on the grid


def test_estimate_from_sensor_shape(small_R, synthetic_prior):
    from scripts.fit_linear_prior import make_gain
    prior = dict(mu=synthetic_prior['mu'], K=make_gain(synthetic_prior['C'], small_R, 1e-6), R=small_R)
    A = expand_matrix()
    y_d = np.zeros((16, 16, 30))
    out = estimate_from_sensor(y_d, prior, 4, 64, 64, A)
    assert out.shape == (64, 64, 64) and np.isfinite(out).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_linear_estimate.py -q`
Expected: `ModuleNotFoundError: No module named 'data_loader.linear_estimate'`.

- [ ] **Step 3: Write the module**

`data_loader/linear_estimate.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_linear_estimate.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add data_loader/linear_estimate.py tests/test_linear_estimate.py
git commit -m "feat(warmstart): linear inverse, band expansion and grid-aligned lift

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JwXhqBayn3pVe5Y9kfV6YP"
```

---

### Task 3: Residual dataset and residual scale σ_d

**Files:**
- Create: `data_loader/hfd_residual.py`
- Create: `scripts/compute_residual_scale.py`
- Test: `tests/test_residual_dataset.py`

**Interfaces:**
- Consumes: `HFD_data`, `linear_estimate.{estimate_from_sensor, expand_matrix, load_prior}`.
- Produces:
  - `HFDResidualData(data_path, split, sensor_down_sample_rate, prior_path, eval_ratio=0.1)`: `__getitem__` returns `(x64 [H,W,64] float32, y_d [h,w,30] float32, x0_hat [H,W,64] float32)`; attributes `d`, `prior`, `A`, `sigma_d` (float, from the prior file for this d; 1.0 if not yet computed).
  - `residual_collate_fn(batch) -> (x [B,64,H,W], y_d [B,30,h,w], x0_hat [B,64,H,W])` float32 tensors.
  - `scripts/compute_residual_scale.py` fills `sigma_d` and `d_values` in the prior npz for d ∈ {1, 2, 4, 8} from the training split.

- [ ] **Step 1: Write the failing tests**

`tests/test_residual_dataset.py`:

```python
import os
import numpy as np
import pytest
import torch

PRIOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'results', 'residual_warmstart', 'linear_prior_R1.npz')
needs_prior = pytest.mark.skipif(not os.path.exists(PRIOR), reason='run scripts/fit_linear_prior.py first')


@pytest.mark.dataset
@needs_prior
def test_item_shapes_and_consistency(data_root):
    from data_loader.hfd_residual import HFDResidualData
    from data_loader.HFD_dataset import HFD_data
    ds = HFDResidualData(data_path=data_root, split='test', sensor_down_sample_rate=4, prior_path=PRIOR)
    base = HFD_data(data_path=data_root, train_mode='image', eval_ratio=0.1, split='test',
                    data_format='image', type='Flower', R_n=1, sensor_down_sample_rate=4)
    assert len(ds) == len(base) == 1000
    x, y, xh = ds[0]
    bx, by = base[0]
    assert x.shape == (64, 64, 64) and y.shape == (16, 16, 30) and xh.shape == (64, 64, 64)
    assert np.allclose(x, bx, atol=1e-6) and np.allclose(y, by, atol=1e-6)   # same x and y as HFD_data
    assert xh.min() >= -1.0 and xh.max() <= 1.0
    assert np.sqrt(((xh - x) ** 2).mean()) < 0.2                              # a sensible estimate, not garbage


@pytest.mark.dataset
@needs_prior
def test_collate(data_root):
    from data_loader.hfd_residual import HFDResidualData, residual_collate_fn
    ds = HFDResidualData(data_path=data_root, split='test', sensor_down_sample_rate=2, prior_path=PRIOR)
    x, y, xh = residual_collate_fn([ds[0], ds[1]])
    assert x.shape == (2, 64, 64, 64) and y.shape == (2, 30, 32, 32) and xh.shape == (2, 64, 64, 64)
    assert x.dtype == torch.float32 and xh.dtype == torch.float32


@pytest.mark.dataset
@needs_prior
def test_sigma_d_is_unit_rms_after_scale_script(data_root):
    from data_loader.hfd_residual import HFDResidualData
    from data_loader.linear_estimate import load_prior
    prior = load_prior(PRIOR)
    if prior['sigma_d'].size == 0:
        pytest.skip('run scripts/compute_residual_scale.py first')
    ds = HFDResidualData(data_path=data_root, split='train', sensor_down_sample_rate=4, prior_path=PRIOR)
    rng = np.random.default_rng(0)
    r2 = []
    for i in rng.choice(len(ds), 40, replace=False):
        x, _, xh = ds[int(i)]
        r2.append(((x - xh) / ds.sigma_d) ** 2)
    rms = np.sqrt(np.mean(r2))
    assert 0.8 < rms < 1.2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_residual_dataset.py -q`
Expected: `ModuleNotFoundError: No module named 'data_loader.hfd_residual'`.

- [ ] **Step 3: Write the dataset**

`data_loader/hfd_residual.py`:

```python
"""HFD dataset that also returns the closed-form estimate x0_hat computed from the low-resolution sensor.

Same files, split, normalisation and strided sensor as HFD_data (which it subclasses); only the return
value changes to (x64, y_d, x0_hat).
"""
import numpy as np
import torch

from data_loader.HFD_dataset import HFD_data
from data_loader.linear_estimate import estimate_from_sensor, expand_matrix, load_prior


class HFDResidualData(HFD_data):
    def __init__(self, data_path, split, sensor_down_sample_rate, prior_path, eval_ratio=0.1, type='Flower'):
        super().__init__(data_path=data_path, train_mode='image', eval_ratio=eval_ratio, split=split,
                         data_format='image', type=type, R_n=1, sensor_down_sample_rate=sensor_down_sample_rate)
        self.d = int(sensor_down_sample_rate)
        self.prior = load_prior(prior_path)
        self.A = expand_matrix(self.wavelens, 64)
        dv = list(self.prior['d_values'].tolist()) if self.prior['d_values'].size else []
        self.sigma_d = float(self.prior['sigma_d'][dv.index(self.d)]) if self.d in dv else 1.0

    def __getitem__(self, idx):
        x64, y_d = super().__getitem__(idx)                       # [H, W, 64], [h, w, 30]
        H, W = x64.shape[:2]
        x0_hat = estimate_from_sensor(np.asarray(y_d, dtype=np.float64), self.prior, self.d, H, W, self.A)
        return (np.asarray(x64, dtype=np.float32), np.asarray(y_d, dtype=np.float32),
                np.asarray(x0_hat, dtype=np.float32))


def residual_collate_fn(batch):
    x, y, xh = zip(*batch)
    to = lambda a: torch.tensor(np.stack(a, 0), dtype=torch.float32).permute(0, 3, 1, 2).contiguous()
    return to(x), to(y), to(xh)
```

- [ ] **Step 4: Write the scale script**

`scripts/compute_residual_scale.py`:

```python
"""Compute sigma_d = RMS of (x - x0_hat) over training files for d in {1, 2, 4, 8} and store it in the prior.
Run from the repo root after scripts/fit_linear_prior.py:
    python scripts/compute_residual_scale.py [--n_files 500]
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data_loader.hfd_residual import HFDResidualData  # noqa: E402

DATA_ROOT = os.environ.get('HFD_DATA_ROOT', '/data/chaoyi_he/HSI/Diffu/dataset/HFD100 Mat dataset')
PRIOR = os.path.join(REPO, 'results', 'residual_warmstart', 'linear_prior_R1.npz')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_files', type=int, default=500)
    ap.add_argument('--d_values', type=int, nargs='+', default=[1, 2, 4, 8])
    ap.add_argument('--prior', default=PRIOR)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    sig = []
    for d in args.d_values:
        ds = HFDResidualData(data_path=DATA_ROOT, split='train', sensor_down_sample_rate=d, prior_path=args.prior)
        idx = rng.choice(len(ds), min(args.n_files, len(ds)), replace=False)
        ss, n = 0.0, 0
        for i in idx:
            x, _, xh = ds[int(i)]
            ss += float(((x - xh) ** 2).sum()); n += x.size
        sig.append(np.sqrt(ss / n))
        print(f'd={d}: sigma_d = {sig[-1]:.5f} ([-1,1] scale) = {sig[-1] / 2 * 100:.3f} % of range RMSE of the estimate')
    z = dict(np.load(args.prior))
    z['sigma_d'] = np.array(sig); z['d_values'] = np.array(args.d_values, dtype=int)
    np.savez(args.prior, **z)
    print('updated', args.prior)


if __name__ == '__main__':
    main()
```

- [ ] **Step 5: Run the scale script, then the tests**

Run: `cd <repo> && /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python scripts/compute_residual_scale.py`
Expected: four lines; σ_d roughly 0.006 / 0.03 / 0.07 / 0.12 for d = 1 / 2 / 4 / 8 (training files; the validation-split baselines will be close). Record them.

Run: `/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_residual_dataset.py -q`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add data_loader/hfd_residual.py scripts/compute_residual_scale.py tests/test_residual_dataset.py
git commit -m "feat(warmstart): residual dataset returning (x, y_d, x0_hat); sigma_d script

<paste the four sigma_d lines>

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JwXhqBayn3pVe5Y9kfV6YP"
```

---

### Task 4: `extra_in_channels` on U2NetHyperspectral and the residual wrapper

**Files:**
- Modify: `model/u2net_hyperspectral.py` (constructor signature at line ~299; `input_proj` at line ~345; the `forward` docstring)
- Create: `model/residual_wrapper.py`
- Test: `tests/test_model_extra_in.py`, `tests/test_wrapper.py`

**Interfaces:**
- Produces: `U2NetHyperspectral(spectral_channels, sensor_channels, base_channels=64, use_channel_3d_conv=False, channel_kernel=3, spatial_kernel=3, channel_num_filters=4, extra_in_channels=0)`; `input_proj` takes `spectral_channels + extra_in_channels` channels; output stays `[B, spectral_channels, H, W]`.
- `ResidualWrapper(net: nn.Module, use_x0_hat: bool)`: `forward(x_t, cond, t)` with `cond = {'sensor': [B,30,h,w], 'x0_hat': [B,64,H,W]}`; if `use_x0_hat`, feeds `cat([x_t, cond['x0_hat']], 1)` to `net` with `cond['sensor']` as context; else feeds `x_t` alone (ablation A). `DiffusionTrainer` keeps calling `model(x_t, cond, t)`; `cond` is a dict here, which the trainer passes through untouched.

- [ ] **Step 1: Write the failing tests**

`tests/test_model_extra_in.py`:

```python
import torch

from model.u2net_hyperspectral import U2NetHyperspectral


def test_default_signature_unchanged():
    net = U2NetHyperspectral(spectral_channels=8, sensor_channels=5, base_channels=16)
    assert net.input_proj.in_channels == 8
    out = net(torch.randn(2, 8, 16, 16), torch.randn(2, 5, 16, 16), torch.randint(0, 1000, (2,)))
    assert out.shape == (2, 8, 16, 16)


def test_extra_in_channels_widens_input_only():
    net = U2NetHyperspectral(spectral_channels=8, sensor_channels=5, base_channels=16, extra_in_channels=8)
    assert net.input_proj.in_channels == 16
    out = net(torch.randn(2, 16, 16, 16), torch.randn(2, 5, 16, 16), torch.randint(0, 1000, (2,)))
    assert out.shape == (2, 8, 16, 16)


def test_old_state_dict_still_loads_strict():
    a = U2NetHyperspectral(spectral_channels=8, sensor_channels=5, base_channels=16)
    b = U2NetHyperspectral(spectral_channels=8, sensor_channels=5, base_channels=16)
    b.load_state_dict(a.state_dict(), strict=True)
```

`tests/test_wrapper.py`:

```python
import torch

from model.residual_wrapper import ResidualWrapper
from model.u2net_hyperspectral import U2NetHyperspectral


def _cond(B=2, H=16, d=4):
    return {'sensor': torch.randn(B, 5, H // d, H // d), 'x0_hat': torch.randn(B, 8, H, H)}


def test_residual_mode_concatenates_x0_hat():
    net = U2NetHyperspectral(spectral_channels=8, sensor_channels=5, base_channels=16, extra_in_channels=8)
    m = ResidualWrapper(net, use_x0_hat=True)
    out = m(torch.randn(2, 8, 16, 16), _cond(), torch.randint(0, 1000, (2,)))
    assert out.shape == (2, 8, 16, 16)


def test_standard_mode_ignores_x0_hat():
    net = U2NetHyperspectral(spectral_channels=8, sensor_channels=5, base_channels=16)
    m = ResidualWrapper(net, use_x0_hat=False)
    x_t, t = torch.randn(2, 8, 16, 16), torch.randint(0, 1000, (2,))
    c1, c2 = _cond(), _cond()
    c2['sensor'] = c1['sensor']                      # same sensor, different x0_hat
    torch.manual_seed(0); o1 = m(x_t, c1, t)
    torch.manual_seed(0); o2 = m(x_t, c2, t)
    assert torch.allclose(o1, o2)


def test_residual_mode_uses_x0_hat():
    net = U2NetHyperspectral(spectral_channels=8, sensor_channels=5, base_channels=16, extra_in_channels=8)
    m = ResidualWrapper(net, use_x0_hat=True).eval()
    x_t, t = torch.randn(2, 8, 16, 16), torch.randint(0, 1000, (2,))
    c1, c2 = _cond(), _cond()
    c2['sensor'] = c1['sensor']
    with torch.no_grad():
        assert not torch.allclose(m(x_t, c1, t), m(x_t, c2, t))


def test_wrapper_rejects_wrong_channel_count():
    import pytest
    net = U2NetHyperspectral(spectral_channels=8, sensor_channels=5, base_channels=16)   # no extra channels
    m = ResidualWrapper(net, use_x0_hat=True)
    with pytest.raises(ValueError):
        m(torch.randn(2, 8, 16, 16), _cond(), torch.randint(0, 1000, (2,)))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_model_extra_in.py tests/test_wrapper.py -q`
Expected: `TypeError: ... unexpected keyword argument 'extra_in_channels'` and `ModuleNotFoundError: No module named 'model.residual_wrapper'`.

- [ ] **Step 3: Make the model change**

In `model/u2net_hyperspectral.py`, add the argument to the constructor and use it for `input_proj`:

```python
    def __init__(
        self,
        spectral_channels,
        sensor_channels,
        base_channels=64,
        use_channel_3d_conv=False,
        channel_kernel=3,
        spatial_kernel=3,
        channel_num_filters=4,
        extra_in_channels=0,
    ):
        """
        Args:
            ...
            extra_in_channels (int): Extra channels concatenated to x_t at the input (e.g. a
                warm-start estimate x0_hat). The output keeps spectral_channels. Default 0 keeps
                the original architecture and checkpoint compatibility.
        """
```

and replace the `input_proj` line with:

```python
        self.extra_in_channels = extra_in_channels
        # Input projection from spectral (+ extra) channels to base channels
        self.input_proj = nn.Conv2d(spectral_channels + extra_in_channels, base_channels, kernel_size=3, padding=1)
```

In `forward`, change `B, L, H, W = x_t.shape` to `B, _, H, W = x_t.shape` (the output width comes from `self.final`), and add to the docstring: `x_t: [B, L + extra_in_channels, H, W]`.

- [ ] **Step 4: Write the wrapper**

`model/residual_wrapper.py`:

```python
"""Adapter so DiffusionTrainer can keep calling model(x_t, cond, t) with a dict condition.

cond = {'sensor': [B, S, h, w] low-resolution sensor image, 'x0_hat': [B, L, H, W] warm-start estimate}
  use_x0_hat=True  (method B): net(cat([x_t, x0_hat], 1), sensor, t)   -- net needs extra_in_channels == L
  use_x0_hat=False (ablation A): net(x_t, sensor, t)
"""
import torch
import torch.nn as nn


class ResidualWrapper(nn.Module):
    def __init__(self, net: nn.Module, use_x0_hat: bool):
        super().__init__()
        self.net = net
        self.use_x0_hat = bool(use_x0_hat)

    def forward(self, x_t, cond, t):
        if not isinstance(cond, dict) or 'sensor' not in cond:
            raise ValueError("cond must be a dict with keys 'sensor' and (for residual mode) 'x0_hat'")
        if self.use_x0_hat:
            x0_hat = cond['x0_hat']
            expected = getattr(self.net, 'input_proj').in_channels
            if x_t.shape[1] + x0_hat.shape[1] != expected:
                raise ValueError(f'net.input_proj expects {expected} channels, got {x_t.shape[1]} + {x0_hat.shape[1]}')
            inp = torch.cat([x_t, x0_hat.to(x_t.dtype)], dim=1)
        else:
            inp = x_t
        return self.net(inp, cond['sensor'], t)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_model_extra_in.py tests/test_wrapper.py -q`
Expected: 7 passed. Also run `tests/test_linear_prior.py` again to be sure nothing else broke.

- [ ] **Step 6: Commit**

```bash
git add model/u2net_hyperspectral.py model/residual_wrapper.py tests/test_model_extra_in.py tests/test_wrapper.py
git commit -m "feat(warmstart): extra_in_channels on U2NetHyperspectral; ResidualWrapper for dict conditions

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JwXhqBayn3pVe5Y9kfV6YP"
```

---

### Task 5: Warm-start option in `DiffusionTrainer.sample`

**Files:**
- Modify: `model/diffusion_trainer.py` (`sample`, lines ~246–360)
- Test: `tests/test_warm_start_sampler.py`

**Interfaces:**
- Produces: `sample(model, cond, shape, n_steps=None, method='ddpm', ..., x_init=None, t_start=None)`.
  - Defaults (`x_init=None, t_start=None`) behave exactly as today.
  - With `x_init` (a tensor of `shape`, the clean-space estimate) and `t_start` (int in [0, T−1]): the chain starts at `x = sqrt(ab[t_start]) * x_init + sqrt(1 - ab[t_start]) * randn` and the timestep list is the descending subsequence of the schedule starting at `t_start`: for `n_steps` requested, `ts = unique(round(linspace(t_start, 0, min(n_steps, t_start + 1) + 1)))[:-1]` in descending order followed by the final x₀ step; `t_start = 0` returns `x_init` clamped, with one model call that is skipped (0 NFE).
  - The returned tensor is the clamped x₀ prediction, as today. The method also records `self.last_nfe` (number of model calls) for the harness.

- [ ] **Step 1: Write the failing tests**

`tests/test_warm_start_sampler.py`:

```python
import torch
import pytest

from model.diffusion_trainer import DiffusionTrainer


class CountingIdentityX0(torch.nn.Module):
    """x0-prediction model that always predicts the constant x0 it is given, and counts calls."""
    def __init__(self, x0):
        super().__init__()
        self.x0 = x0
        self.calls = 0
        self.seen_t = []

    def forward(self, x_t, cond, t):
        self.calls += 1
        self.seen_t.append(int(t[0]))
        return self.x0.expand_as(x_t).clone()


def _trainer():
    return DiffusionTrainer(device=torch.device('cpu'), prediction_type='x0', clamp_x0=(-1.0, 1.0), snr_gamma=None)


def test_defaults_unchanged_against_reference_run():
    tr = _trainer()
    x0 = torch.full((1, 2, 4, 4), 0.3)
    m = CountingIdentityX0(x0)
    torch.manual_seed(0)
    out_a = tr.sample(m, None, (1, 2, 4, 4), n_steps=10, method='ddim', progress=False)
    m2 = CountingIdentityX0(x0)
    torch.manual_seed(0)
    out_b = tr.sample(m2, None, (1, 2, 4, 4), n_steps=10, method='ddim', progress=False, x_init=None, t_start=None)
    assert torch.allclose(out_a, out_b) and m.calls == m2.calls == 10
    assert tr.last_nfe == 10


def test_warm_start_first_call_is_t_start_and_ends_at_zero():
    tr = _trainer()
    x0 = torch.full((1, 2, 4, 4), 0.3)
    m = CountingIdentityX0(x0)
    tr.sample(m, None, (1, 2, 4, 4), n_steps=10, method='ddim', progress=False, x_init=torch.zeros(1, 2, 4, 4), t_start=200)
    assert m.seen_t[0] == 200 and m.seen_t[-1] > 0 and m.calls == 10 and tr.last_nfe == 10
    assert all(a > b for a, b in zip(m.seen_t, m.seen_t[1:]))      # strictly descending


def test_warm_start_small_t_start_has_no_repeated_steps():
    tr = _trainer()
    m = CountingIdentityX0(torch.zeros(1, 2, 4, 4))
    tr.sample(m, None, (1, 2, 4, 4), n_steps=10, method='ddpm', progress=False, x_init=torch.zeros(1, 2, 4, 4), t_start=5)
    assert len(set(m.seen_t)) == len(m.seen_t) and m.calls <= 6


def test_warm_start_t0_zero_returns_init():
    tr = _trainer()
    m = CountingIdentityX0(torch.zeros(1, 2, 4, 4))
    x_init = torch.full((1, 2, 4, 4), 1.7)
    out = tr.sample(m, None, (1, 2, 4, 4), n_steps=10, method='ddim', progress=False, x_init=x_init, t_start=0)
    assert m.calls == 0 and tr.last_nfe == 0
    assert torch.allclose(out, x_init.clamp(-1, 1))


def test_warm_start_noise_level_matches_schedule():
    tr = _trainer()
    seen = {}

    class Probe(torch.nn.Module):
        def forward(self, x_t, cond, t):
            seen['x'] = x_t.clone(); return torch.zeros_like(x_t)

    x_init = torch.zeros(1, 1, 64, 64)
    torch.manual_seed(1)
    tr.sample(Probe(), None, (1, 1, 64, 64), n_steps=1, method='ddim', progress=False, x_init=x_init, t_start=400)
    expected_std = float(tr.sqrt_one_minus_alpha_bars[400])
    assert abs(seen['x'].std().item() - expected_std) < 0.05 * expected_std


def test_warm_start_requires_both_args():
    tr = _trainer()
    m = CountingIdentityX0(torch.zeros(1, 2, 4, 4))
    with pytest.raises(ValueError):
        tr.sample(m, None, (1, 2, 4, 4), n_steps=10, progress=False, x_init=torch.zeros(1, 2, 4, 4))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_warm_start_sampler.py -q`
Expected: `TypeError: sample() got an unexpected keyword argument 'x_init'` (and the defaults test fails on `last_nfe`).

- [ ] **Step 3: Make the sampler change**

In `model/diffusion_trainer.py`, change the signature of `sample` to add, after `clamp_x0`:

```python
        x_init: Optional[torch.Tensor] = None,
        t_start: Optional[int] = None,
```

extend the docstring:

```
            x_init: optional clean-space estimate of shape `shape`. With `t_start`, the chain
                    starts from sqrt(ab[t_start]) * x_init + sqrt(1 - ab[t_start]) * noise and
                    runs only the steps from t_start down to 0 (warm start). Both or neither.
            t_start: integer in [0, n_timesteps - 1]; 0 returns clamp(x_init) with no model call.
        Side effect: sets self.last_nfe to the number of model evaluations made.
```

and replace the block from `x = torch.randn(shape, device=self.device)` through the `iterator = ...` line with:

```python
        if (x_init is None) != (t_start is None):
            raise ValueError('x_init and t_start must be given together')
        warm = x_init is not None
        if warm:
            if not (0 <= int(t_start) < self.n_timesteps):
                raise ValueError(f't_start must be in [0, {self.n_timesteps - 1}]')
            t_start = int(t_start)
            x_init = x_init.to(self.device)
            if t_start == 0:
                self.last_nfe = 0
                return _maybe_clamp(x_init)
            ab = self.alpha_bars[t_start]
            x = torch.sqrt(ab) * x_init + torch.sqrt(1.0 - ab) * torch.randn(shape, device=self.device)
            n_eff = min(n_steps, t_start + 1)
            grid = torch.linspace(t_start, 0, n_eff + 1).round().long().tolist()[:-1]   # descending, excludes 0
            ts = sorted(set(grid), reverse=True)
        else:
            x = torch.randn(shape, device=self.device)
            # Build descending timestep schedule (respaced if n_steps < n_timesteps)
            step_interval = max(1, self.n_timesteps // n_steps)
            ts = list(range(self.n_timesteps - 1, -1, -step_interval))
        self.last_nfe = len(ts)
        iterator = tqdm(ts, desc=f"{method.upper()} Sampling") if progress else ts
```

(Everything after this line — the loop, the clamp, the DDPM/DDIM updates, the final x₀ step when `prev_timestep < 0` — is unchanged.) Also initialise `self.last_nfe = 0` in `__init__`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_warm_start_sampler.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add model/diffusion_trainer.py tests/test_warm_start_sampler.py
git commit -m "feat(warmstart): x_init/t_start warm start in DiffusionTrainer.sample (additive)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JwXhqBayn3pVe5Y9kfV6YP"
```

---

### Task 6: Validation subset and baseline record

**Files:**
- Create: `scripts/make_val_subset.py`
- Create: `tests/data/val_subset_200.json` (generated by the script, committed)
- Create: `tests/data/baseline_phase0.json` (generated in Task 7 step 6, committed)

**Interfaces:**
- Produces: `tests/data/val_subset_200.json` = `{"seed": 0, "files": [<relative path under MatFlower60/Train>, ...200], "classes": {"P022": 24, "P023": 152, "P024": 24}, "ddpm1000_subset": [<50 of those files>]}`; the harness reads it. Relative paths are relative to the dataset root.

- [ ] **Step 1: Write the script**

`scripts/make_val_subset.py`:

```python
"""Fixed, stratified subset of the loader's validation split for diffusion evaluation (spec §7).
24 cubes from P022, 152 from P023, 24 from P024; the first 50 (stratified 6/38/6) are the DDPM-1000 subset.
Run from the repo root:  python scripts/make_val_subset.py
"""
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data_loader.HFD_dataset import HFD_data  # noqa: E402

DATA_ROOT = os.environ.get('HFD_DATA_ROOT', '/data/chaoyi_he/HSI/Diffu/dataset/HFD100 Mat dataset')
QUOTA = {'P022': 24, 'P023': 152, 'P024': 24}
DDPM_QUOTA = {'P022': 6, 'P023': 38, 'P024': 6}


def main(seed=0):
    ds = HFD_data(data_path=DATA_ROOT, train_mode='image', eval_ratio=0.1, split='test',
                  data_format='image', type='Flower', R_n=1, sensor_down_sample_rate=1)
    by_cls = {}
    for f in ds.img_list:
        by_cls.setdefault(os.path.basename(os.path.dirname(f)), []).append(f)
    rng = random.Random(seed)
    files, ddpm = [], []
    for c, n in QUOTA.items():
        pick = rng.sample(sorted(by_cls[c]), n)
        files += pick
        ddpm += pick[:DDPM_QUOTA[c]]
    rel = lambda f: os.path.relpath(f, DATA_ROOT)
    out = dict(seed=seed, files=[rel(f) for f in files], classes=QUOTA, ddpm1000_subset=[rel(f) for f in ddpm])
    path = os.path.join(REPO, 'tests', 'data', 'val_subset_200.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, 'w'), indent=1)
    print('wrote', path, len(files), 'files,', len(ddpm), 'for DDPM-1000')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run it and check the output**

Run: `cd <repo> && /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python scripts/make_val_subset.py && /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -c "import json; d=json.load(open('tests/data/val_subset_200.json')); print(len(d['files']), len(d['ddpm1000_subset']), d['files'][0])"`
Expected: `wrote ... 200 files, 50 for DDPM-1000` and `200 50 MatFlower60/Train/P022/...`.

- [ ] **Step 3: Commit**

```bash
git add scripts/make_val_subset.py tests/data/val_subset_200.json
git commit -m "feat(warmstart): fixed stratified validation subset for diffusion evaluation

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JwXhqBayn3pVe5Y9kfV6YP"
```

---

### Task 7: Evaluation harness (metrics, baseline, sampler grid, y-swap)

**Files:**
- Create: `scripts/eval_warmstart.py`
- Test: `tests/test_eval_harness.py`

**Interfaces:**
- Consumes: `HFDResidualData`, `ResidualWrapper`, `DiffusionTrainer.sample(..., x_init, t_start)`, `tests/data/val_subset_200.json`, the prior npz.
- Produces (importable functions, all torch, batch-first):
  - `metrics(x_hat, x) -> dict(rmse_pct [B], psnr [B], sam_deg [B])` on the [0, 1] scale of `(x + 1) / 2`;
  - `baseline_eval(data_root, prior_path, d, files=None) -> dict` with per-file metrics and means for the baseline x̂₀ over the 1 000 validation files (or `files`);
  - `sampler_grid() -> list[dict]` = `[{'name': 'ddpm1000', 'method': 'ddpm', 'n_steps': 1000}, {'name': 'ddim1', 'method': 'ddim', 'n_steps': 1}, ..., ddim2, ddim5, ddim10, ddim20, {'name': 'warm50_ddim', 'method': 'ddim', 'n_steps': 10, 't_start': 50}, warm100_ddim, warm200_ddim, warm400_ddim, {'name': 'warm50_ddpm', 'method': 'ddpm', 'n_steps': 1000, 't_start': 50}, warm100_ddpm, warm200_ddpm, warm400_ddpm]`;
  - `reconstruct(model, trainer, mode, batch, cfg, sigma_d, seed) -> x_hat [B,64,H,W]`: for `mode='residual'` samples r̂₀ (warm starts from `x_init = 0`) and returns `clamp(x0_hat + sigma_d * r_hat)`; for `mode='standard'` samples x directly (warm starts from `x_init = x0_hat`);
  - `yswap_check(model, trainer, mode, batch, t_values=(100, 300)) -> dict` returning the one-step x₀ RMSE (on the [0,1] scale, %) for `true`, `swap_full`, `swap_sensor_only`, `zero` at each t, and `passes` (true ≤ 0.9 × swap_full at both t); raises `ValueError` if the batch has fewer than 2 items;
  - CLI: `python scripts/eval_warmstart.py --d 4 --baseline` writes `results/residual_warmstart/eval/baseline_d4.json`; `--ckpt PATH --mode residual|standard` writes `results/residual_warmstart/eval/<ckpt-name>_d4.json` with every grid row (mean ± SE, paired difference to the baseline, per-class breakdown, NFE, seconds per cube) and the y-swap result.

- [ ] **Step 1: Write the failing tests**

`tests/test_eval_harness.py`:

```python
import json
import os
import numpy as np
import pytest
import torch

from scripts.eval_warmstart import metrics, sampler_grid, reconstruct, yswap_check
from model.diffusion_trainer import DiffusionTrainer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_metrics_perfect_and_known_error():
    x = torch.rand(2, 4, 8, 8) * 2 - 1
    m = metrics(x, x)
    assert torch.allclose(m['rmse_pct'], torch.zeros(2)) and torch.all(m['sam_deg'] < 1e-3)
    x_hat = x.clone(); x_hat[0] += 0.2                    # +0.1 on the [0,1] scale for item 0
    m = metrics(x_hat.clamp(-1, 1), x)
    assert abs(m['rmse_pct'][0].item() - 10.0) < 1.5 and m['rmse_pct'][1].item() == 0.0
    assert abs(m['psnr'][0].item() - 20.0) < 1.5


def test_sampler_grid_names_and_nfe_budget():
    g = sampler_grid()
    names = [c['name'] for c in g]
    assert names[:6] == ['ddpm1000', 'ddim1', 'ddim2', 'ddim5', 'ddim10', 'ddim20']
    assert 'warm200_ddim' in names and 'warm400_ddpm' in names
    assert all('t_start' in c for c in g if c['name'].startswith('warm'))


class ZeroModel(torch.nn.Module):
    def forward(self, x_t, cond, t):
        return torch.zeros_like(x_t)


def test_zero_residual_equals_baseline():
    tr = DiffusionTrainer(device=torch.device('cpu'), prediction_type='x0', snr_gamma=None)
    x0_hat = torch.rand(2, 4, 8, 8) * 2 - 1
    batch = (torch.zeros(2, 4, 8, 8), torch.zeros(2, 3, 2, 2), x0_hat)
    for cfg in [dict(name='ddim5', method='ddim', n_steps=5), dict(name='warm50_ddim', method='ddim', n_steps=10, t_start=50)]:
        out = reconstruct(ZeroModel(), tr, 'residual', batch, cfg, sigma_d=0.1, seed=0)
        assert torch.allclose(out, x0_hat, atol=1e-6)


def test_yswap_requires_batch_ge_2():
    tr = DiffusionTrainer(device=torch.device('cpu'), prediction_type='x0', snr_gamma=None)
    batch = (torch.zeros(1, 4, 8, 8), torch.zeros(1, 3, 2, 2), torch.zeros(1, 4, 8, 8))
    with pytest.raises(ValueError):
        yswap_check(ZeroModel(), tr, 'residual', batch)


def test_yswap_fails_for_condition_blind_model():
    tr = DiffusionTrainer(device=torch.device('cpu'), prediction_type='x0', snr_gamma=None)
    x = torch.rand(4, 4, 8, 8) * 2 - 1
    batch = (x, torch.rand(4, 3, 2, 2), torch.rand(4, 4, 8, 8) * 2 - 1)
    res = yswap_check(ZeroModel(), tr, 'residual', batch)
    assert res['passes'] is False
    assert set(res['t100'].keys()) == {'true', 'swap_full', 'swap_sensor_only', 'zero'}


def test_yswap_passes_for_oracle_model():
    """A model that reads x0_hat (= the truth here) passes: swapping the condition hurts it."""
    class Oracle(torch.nn.Module):
        def forward(self, x_t, cond, t):
            return (x_t * 0 + 0)  # residual-mode x0 prediction of r = 0 → x_hat = cond['x0_hat']
    tr = DiffusionTrainer(device=torch.device('cpu'), prediction_type='x0', snr_gamma=None)
    x = torch.rand(4, 4, 8, 8) * 2 - 1
    batch = (x, torch.rand(4, 3, 2, 2), x.clone())           # x0_hat equals the truth
    res = yswap_check(Oracle(), tr, 'residual', batch)
    assert res['passes'] is True


@pytest.mark.dataset
def test_baseline_eval_reproduces_recorded_numbers(data_root):
    from scripts.eval_warmstart import baseline_eval
    rec_path = os.path.join(REPO, 'tests', 'data', 'baseline_phase0.json')
    prior = os.path.join(REPO, 'results', 'residual_warmstart', 'linear_prior_R1.npz')
    if not (os.path.exists(rec_path) and os.path.exists(prior)):
        pytest.skip('baseline record or prior missing')
    rec = json.load(open(rec_path))
    sub = json.load(open(os.path.join(REPO, 'tests', 'data', 'val_subset_200.json')))
    files = [os.path.join(data_root, f) for f in sub['files'][:40]]
    out = baseline_eval(data_root, prior, d=4, files=files)
    assert abs(out['mean']['rmse_pct'] - rec['d4']['subset40_rmse_pct']) < 0.01
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_eval_harness.py -q`
Expected: `ModuleNotFoundError: No module named 'scripts.eval_warmstart'`.

- [ ] **Step 3: Write the harness**

`scripts/eval_warmstart.py`:

```python
"""Evaluation harness for the residual warm-start project (spec §7).

  python scripts/eval_warmstart.py --d 4 --baseline
  python scripts/eval_warmstart.py --d 4 --ckpt results/residual_warmstart/residual/d4/checkpoint_epoch_200.pth \
         --mode residual --base_channels 256 [--seeds 0 1] [--device cuda:0]

Metrics are on the [0, 1] display scale of (x + 1) / 2: RMSE in % of range, PSNR with peak 1, SAM in degrees.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data_loader.hfd_residual import HFDResidualData, residual_collate_fn  # noqa: E402
from model.diffusion_trainer import DiffusionTrainer  # noqa: E402
from model.residual_wrapper import ResidualWrapper  # noqa: E402
from model.u2net_hyperspectral import U2NetHyperspectral  # noqa: E402

DATA_ROOT = os.environ.get('HFD_DATA_ROOT', '/data/chaoyi_he/HSI/Diffu/dataset/HFD100 Mat dataset')
PRIOR = os.path.join(REPO, 'results', 'residual_warmstart', 'linear_prior_R1.npz')
SUBSET = os.path.join(REPO, 'tests', 'data', 'val_subset_200.json')
EVAL_DIR = os.path.join(REPO, 'results', 'residual_warmstart', 'eval')


# ----------------------------------------------------------------------------- metrics
def metrics(x_hat, x):
    """x_hat, x: [B, C, H, W] on [-1, 1]. Returns per-item tensors on the [0, 1] scale."""
    a, b = (x_hat.float() + 1) / 2, (x.float() + 1) / 2
    err2 = ((a - b) ** 2).flatten(1).mean(1)
    rmse_pct = err2.sqrt() * 100
    psnr = -10 * torch.log10(err2.clamp(min=1e-12))
    cos = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12)           # per pixel over bands
    sam_deg = torch.rad2deg(torch.arccos(cos.clamp(-1, 1))).flatten(1).mean(1)
    return dict(rmse_pct=rmse_pct, psnr=psnr, sam_deg=sam_deg)


def summarize(per_item, classes):
    """per_item: dict metric -> np.array [N]; classes: list of class ids per item."""
    out = {}
    for k, v in per_item.items():
        v = np.asarray(v, dtype=np.float64)
        out[k] = float(v.mean()); out[k + '_se'] = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0
        for c in sorted(set(classes)):
            sel = np.array([ci == c for ci in classes])
            out[f'{k}_{c}'] = float(v[sel].mean())
    return out


# ----------------------------------------------------------------------------- data
def load_subset(data_root, which='files'):
    sub = json.load(open(SUBSET))
    return [os.path.join(data_root, f) for f in sub[which]]


def dataset_for(data_root, prior_path, d, files=None):
    ds = HFDResidualData(data_path=data_root, split='test', sensor_down_sample_rate=d, prior_path=prior_path)
    if files is not None:
        wanted = {os.path.abspath(f) for f in files}
        ds.img_list = [f for f in ds.img_list if os.path.abspath(f) in wanted]
    return ds


def class_of(path):
    return os.path.basename(os.path.dirname(path))


# ----------------------------------------------------------------------------- baseline
def baseline_eval(data_root, prior_path, d, files=None, batch_size=16):
    ds = dataset_for(data_root, prior_path, d, files)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=residual_collate_fn)
    per = {k: [] for k in ('rmse_pct', 'psnr', 'sam_deg')}
    for x, _, xh in loader:
        m = metrics(xh, x)
        for k in per: per[k] += m[k].tolist()
    classes = [class_of(f) for f in ds.img_list]
    return dict(d=d, n=len(ds), per_item=per, files=[os.path.relpath(f, data_root) for f in ds.img_list],
                classes=classes, mean=summarize(per, classes), nfe=0)


# ----------------------------------------------------------------------------- samplers
def sampler_grid():
    g = [dict(name='ddpm1000', method='ddpm', n_steps=1000)]
    g += [dict(name=f'ddim{k}', method='ddim', n_steps=k) for k in (1, 2, 5, 10, 20)]
    g += [dict(name=f'warm{t}_ddim', method='ddim', n_steps=10, t_start=t) for t in (50, 100, 200, 400)]
    g += [dict(name=f'warm{t}_ddpm', method='ddpm', n_steps=1000, t_start=t) for t in (50, 100, 200, 400)]
    return g


def reconstruct(model, trainer, mode, batch, cfg, sigma_d, seed):
    """batch = (x, y_d, x0_hat) tensors. Returns x_hat [B, 64, H, W] clamped to [-1, 1]."""
    x, y_d, x0_hat = [b.to(trainer.device) for b in batch]
    cond = {'sensor': y_d, 'x0_hat': x0_hat}
    torch.manual_seed(seed)
    kw = dict(n_steps=cfg['n_steps'], method=cfg['method'], progress=False)
    if 't_start' in cfg:
        kw['t_start'] = cfg['t_start']
        kw['x_init'] = torch.zeros_like(x0_hat) if mode == 'residual' else x0_hat
    out = trainer.sample(model, cond, tuple(x0_hat.shape), **kw)
    if mode == 'residual':
        out = x0_hat + sigma_d * out
    return out.clamp(-1, 1)


# ----------------------------------------------------------------------------- y-swap
def _one_step_x0(model, trainer, mode, x_target, cond, t_val):
    B = x_target.shape[0]
    t = torch.full((B,), t_val, device=trainer.device, dtype=torch.long)
    torch.manual_seed(1234)
    x_t, _ = trainer.q_sample(x_target, t)
    with torch.no_grad():
        x0_pred, _ = trainer._model_to_x0_eps(model(x_t, cond, t), x_t, t)
    return x0_pred


def yswap_check(model, trainer, mode, batch, t_values=(100, 300), sigma_d=1.0):
    x, y_d, x0_hat = [b.to(trainer.device) for b in batch]
    if x.shape[0] < 2:
        raise ValueError('y-swap check needs a batch of at least 2 items')
    target = (x - x0_hat) / sigma_d if mode == 'residual' else x
    conds = {
        'true': {'sensor': y_d, 'x0_hat': x0_hat},
        'swap_full': {'sensor': y_d.roll(1, 0), 'x0_hat': x0_hat.roll(1, 0)},
        'swap_sensor_only': {'sensor': y_d.roll(1, 0), 'x0_hat': x0_hat},
        'zero': {'sensor': torch.zeros_like(y_d), 'x0_hat': torch.zeros_like(x0_hat)},
    }
    res, ok = {}, True
    for tv in t_values:
        row = {}
        for name, c in conds.items():
            pred = _one_step_x0(model, trainer, mode, target, c, tv)
            x_hat = (c['x0_hat'] + sigma_d * pred) if mode == 'residual' else pred
            row[name] = float(metrics(x_hat.clamp(-1, 1), x)['rmse_pct'].mean())
        res[f't{tv}'] = row
        ok = ok and (row['true'] <= 0.9 * row['swap_full'])
    res['passes'] = bool(ok)
    return res


# ----------------------------------------------------------------------------- checkpoint evaluation
def build_model(mode, base_channels, device):
    extra = 64 if mode == 'residual' else 0
    net = U2NetHyperspectral(spectral_channels=64, sensor_channels=30, base_channels=base_channels, extra_in_channels=extra)
    return ResidualWrapper(net, use_x0_hat=(mode == 'residual')).to(device).eval()


def checkpoint_eval(args):
    device = torch.device(args.device)
    trainer = DiffusionTrainer(device=device, prediction_type=args.prediction_type, loss_type='l1', snr_gamma=None)
    model = build_model(args.mode, args.base_channels, device)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck)
    model.net.load_state_dict({k.replace('net.', '', 1) if k.startswith('net.') else k: v for k, v in sd.items()}, strict=True)
    ds_all = dataset_for(args.data_root, args.prior, args.d, load_subset(args.data_root, 'files'))
    ds_ddpm = dataset_for(args.data_root, args.prior, args.d, load_subset(args.data_root, 'ddpm1000_subset'))
    sigma_d = ds_all.sigma_d
    base = baseline_eval(args.data_root, args.prior, args.d, ds_all.img_list)
    rows = []
    for cfg in sampler_grid():
        ds = ds_ddpm if cfg['name'] == 'ddpm1000' else ds_all
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=residual_collate_fn)
        per = {k: [] for k in ('rmse_pct', 'psnr', 'sam_deg')}; base_r = []; secs = 0.0; nfe = None
        for seed in args.seeds:
            for bi, batch in enumerate(loader):
                t0 = time.time()
                x_hat = reconstruct(model, trainer, args.mode, batch, cfg, sigma_d, seed * 1000 + bi)
                secs += time.time() - t0
                nfe = trainer.last_nfe
                m = metrics(x_hat, batch[0].to(device)); mb = metrics(batch[2].to(device), batch[0].to(device))
                for k in per: per[k] += m[k].tolist()
                base_r += mb['rmse_pct'].tolist()
        classes = [class_of(f) for f in ds.img_list] * len(args.seeds)
        diff = np.asarray(per['rmse_pct']) - np.asarray(base_r)
        rows.append(dict(cfg, nfe=nfe, sec_per_cube=secs / len(per['rmse_pct']), **summarize(per, classes),
                         paired_rmse_diff=float(diff.mean()), paired_rmse_se=float(diff.std(ddof=1) / math.sqrt(len(diff)))))
        print(f"{cfg['name']:>14} nfe={nfe:5d} rmse={rows[-1]['rmse_pct']:.3f}±{rows[-1]['rmse_pct_se']:.3f} "
              f"(baseline {np.mean(base_r):.3f}, paired {diff.mean():+.3f}±{rows[-1]['paired_rmse_se']:.3f}) sam={rows[-1]['sam_deg']:.2f}")
    batch = next(iter(torch.utils.data.DataLoader(ds_all, batch_size=8, shuffle=False, collate_fn=residual_collate_fn)))
    ys = yswap_check(model, trainer, args.mode, batch, sigma_d=sigma_d)
    print('y-swap:', json.dumps(ys))
    out = dict(ckpt=args.ckpt, mode=args.mode, d=args.d, sigma_d=sigma_d, seeds=args.seeds, baseline=base['mean'], rows=rows, yswap=ys)
    os.makedirs(EVAL_DIR, exist_ok=True)
    path = os.path.join(EVAL_DIR, f"{os.path.splitext(os.path.basename(args.ckpt))[0]}_{args.mode}_d{args.d}.json")
    json.dump(out, open(path, 'w'), indent=1)
    print('wrote', path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, required=True)
    ap.add_argument('--baseline', action='store_true')
    ap.add_argument('--ckpt'); ap.add_argument('--mode', choices=['residual', 'standard'])
    ap.add_argument('--base_channels', type=int, default=64)
    ap.add_argument('--prediction_type', default='v')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--data_root', default=DATA_ROOT); ap.add_argument('--prior', default=PRIOR)
    args = ap.parse_args()
    if args.baseline:
        out = baseline_eval(args.data_root, args.prior, args.d)
        os.makedirs(EVAL_DIR, exist_ok=True)
        path = os.path.join(EVAL_DIR, f'baseline_d{args.d}.json')
        json.dump(out, open(path, 'w'), indent=1)
        m = out['mean']
        print(f"baseline d={args.d} on {out['n']} files: RMSE {m['rmse_pct']:.3f}±{m['rmse_pct_se']:.3f} % "
              f"(P022 {m['rmse_pct_P022']:.3f}, P023 {m['rmse_pct_P023']:.3f}, P024 {m['rmse_pct_P024']:.3f}) "
              f"PSNR {m['psnr']:.2f} SAM {m['sam_deg']:.3f}; wrote {path}")
    else:
        assert args.ckpt and args.mode, '--ckpt and --mode are required without --baseline'
        checkpoint_eval(args)


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the unit tests to verify they pass**

Run: `/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_eval_harness.py -q -k "not reproduces"`
Expected: 6 passed (the record test is skipped until step 6).

- [ ] **Step 5: Run the official baselines**

Run, from the repo root, for d in 1 2 4 8:
`/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python scripts/eval_warmstart.py --d <d> --baseline`
Expected: four lines like `baseline d=4 on 1000 files: RMSE 3.5xx±0.0xx % (...)`. These are the Phase 0 baselines (grid-aligned upsampling; expect them at or slightly below the earlier 1.64 / 3.57 / 6.14 %).

- [ ] **Step 6: Write the baseline record and make the record test pass**

Create `tests/data/baseline_phase0.json` from the four JSON files plus the 40-file subset value the test checks:

```bash
/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python - <<'EOF'
import json, os, sys
sys.path.insert(0, '.')
from scripts.eval_warmstart import baseline_eval, DATA_ROOT, PRIOR
rec = {}
for d in (1, 2, 4, 8):
    full = json.load(open(f'results/residual_warmstart/eval/baseline_d{d}.json'))['mean']
    sub = json.load(open('tests/data/val_subset_200.json'))
    files = [os.path.join(DATA_ROOT, f) for f in sub['files'][:40]]
    s40 = baseline_eval(DATA_ROOT, PRIOR, d, files)['mean']['rmse_pct']
    rec[f'd{d}'] = dict(rmse_pct=round(full['rmse_pct'], 4), psnr=round(full['psnr'], 3), sam_deg=round(full['sam_deg'], 4),
                        rmse_pct_P022=round(full['rmse_pct_P022'], 4), rmse_pct_P023=round(full['rmse_pct_P023'], 4),
                        rmse_pct_P024=round(full['rmse_pct_P024'], 4), subset40_rmse_pct=round(s40, 4))
json.dump(rec, open('tests/data/baseline_phase0.json', 'w'), indent=1)
print(json.dumps(rec, indent=1))
EOF
```

Run: `/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_eval_harness.py -q`
Expected: 7 passed.

- [ ] **Step 7: Commit**

```bash
git add scripts/eval_warmstart.py tests/test_eval_harness.py tests/data/baseline_phase0.json
git commit -m "feat(warmstart): evaluation harness with baseline, sampler grid and y-swap check; Phase 0 baselines

<paste the four baseline lines>

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JwXhqBayn3pVe5Y9kfV6YP"
```

---

### Task 8: Training-step timing

**Files:**
- Create: `scripts/time_train_step.py`

**Interfaces:**
- Consumes: `HFDResidualData`, `residual_collate_fn`, `ResidualWrapper`, `U2NetHyperspectral(extra_in_channels=64)`, `DiffusionTrainer.get_loss`.
- Produces: printed and JSON-recorded seconds per step and peak memory for (`base_channels=64`, batch 8, 1 GPU, bf16 autocast) and (`base_channels=256`, batch 8 per GPU, FSDP on 2 GPUs, bf16), written to `results/residual_warmstart/timing.json`. No training code is reused from `main_2d_fsdp.py` beyond the FSDP wrapping pattern; the real training entry point is Phase 1.

- [ ] **Step 1: Write the script**

`scripts/time_train_step.py`:

```python
"""Time one training step of the residual model for the pilot and full-size configurations.

Single GPU:   python scripts/time_train_step.py --base_channels 64 --batch_size 8 --device cuda:0
FSDP, 2 GPUs: torchrun --standalone --nproc_per_node=2 scripts/time_train_step.py --base_channels 256 --batch_size 8 --fsdp
Checks nvidia-smi free memory first and refuses to run on a GPU with < 3 GB free.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data_loader.hfd_residual import HFDResidualData, residual_collate_fn  # noqa: E402
from model.diffusion_trainer import DiffusionTrainer  # noqa: E402
from model.residual_wrapper import ResidualWrapper  # noqa: E402
from model.u2net_hyperspectral import U2NetHyperspectral  # noqa: E402

DATA_ROOT = os.environ.get('HFD_DATA_ROOT', '/data/chaoyi_he/HSI/Diffu/dataset/HFD100 Mat dataset')
PRIOR = os.path.join(REPO, 'results', 'residual_warmstart', 'linear_prior_R1.npz')


def free_mb(index):
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits', f'--id={index}'],
                         capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_channels', type=int, default=64)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--d', type=int, default=4)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--fsdp', action='store_true')
    args = ap.parse_args()

    if args.fsdp:
        torch.distributed.init_process_group('nccl')
        rank = torch.distributed.get_rank(); local = int(os.environ['LOCAL_RANK'])
        device = torch.device(f'cuda:{local}')
    else:
        rank, local = 0, int(args.device.split(':')[-1]); device = torch.device(args.device)
    if free_mb(local) < 3000:
        raise SystemExit(f'GPU {local} has only {free_mb(local)} MiB free; refusing to run')
    torch.cuda.set_device(device)

    ds = HFDResidualData(data_path=DATA_ROOT, split='train', sensor_down_sample_rate=args.d, prior_path=PRIOR)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=residual_collate_fn)
    net = U2NetHyperspectral(spectral_channels=64, sensor_channels=30, base_channels=args.base_channels, extra_in_channels=64)
    model = ResidualWrapper(net, use_x0_hat=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if args.fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        import functools
        model = FSDP(model, auto_wrap_policy=functools.partial(size_based_auto_wrap_policy, min_num_params=1_000_000),
                     mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16),
                     device_id=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = DiffusionTrainer(device=device, prediction_type='v', loss_type='l1', snr_gamma=5.0)

    times = []
    it = iter(loader)
    for step in range(args.steps + 3):
        x, y_d, xh = next(it)
        x, y_d, xh = x.to(device), y_d.to(device), xh.to(device)
        r = (x - xh) / ds.sigma_d
        torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = trainer.get_loss(model, r, {'sensor': y_d, 'x0_hat': xh}, loss_in_fp32=True)
        opt.zero_grad(set_to_none=True); loss.backward()
        (model.clip_grad_norm_(1.0) if args.fsdp else torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step(); torch.cuda.synchronize()
        if step >= 3: times.append(time.time() - t0)
    sec = sum(times) / len(times); peak = torch.cuda.max_memory_allocated(device) / 2 ** 30
    if rank == 0:
        n_train = len(ds); world = torch.distributed.get_world_size() if args.fsdp else 1
        epoch_sec = sec * n_train / (args.batch_size * world)
        rec = dict(base_channels=args.base_channels, batch_size=args.batch_size, fsdp=args.fsdp, world=world, params=n_params,
                   sec_per_step=sec, peak_gb=peak, sec_per_epoch=epoch_sec, loss=float(loss))
        print(json.dumps(rec, indent=1))
        path = os.path.join(REPO, 'results', 'residual_warmstart', 'timing.json')
        allrec = json.load(open(path)) if os.path.exists(path) else []
        allrec.append(rec); json.dump(allrec, open(path, 'w'), indent=1)
    if args.fsdp:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run both configurations**

Check first: `nvidia-smi`. Then:

```bash
cd <repo>
/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python scripts/time_train_step.py --base_channels 64 --batch_size 8 --device cuda:0
/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/torchrun --standalone --nproc_per_node=2 scripts/time_train_step.py --base_channels 256 --batch_size 8 --fsdp
```

Expected: two JSON records with finite loss, peak memory under 18 GB, and `sec_per_epoch` for each. If the FSDP run OOMs, rerun with `--batch_size 4` and note it. Write the two `sec_per_epoch` values into the commit message; they set the Phase 1 budgets (pilot: 100 epochs or 24 h; full: 200 epochs).

- [ ] **Step 3: Commit**

```bash
git add scripts/time_train_step.py
git commit -m "feat(warmstart): training-step timing for pilot and FSDP configurations

<paste the two timing records>

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JwXhqBayn3pVe5Y9kfV6YP"
```

---

### Task 9: Phase 0 report and gate check

**Files:**
- Create: `docs/superpowers/plans/2026-10-08-residual-warm-start-phase0-RESULTS.md`

- [ ] **Step 1: Run the whole test suite once more**

Run: `/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests -q`
Expected: all tests pass (dataset tests included on this server).

- [ ] **Step 2: Write the results note**

```markdown
# Phase 0 results — <date>

## Linear prior
<the fit line from Task 1: files, pixels, s, held-out RMSE>

## Residual scale sigma_d (training files)
| d | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| sigma_d ([-1,1] scale) | | | | |

## Official baselines (loader validation split, 1000 files, grid-aligned upsampling)
| d | RMSE % | ± SE | P022 | P023 | P024 | PSNR | SAM ° |
|---|---|---|---|---|---|---|---|
| 1 | | | | | | | |
| 2 | | | | | | | |
| 4 | | | | | | | |
| 8 | | | | | | | |
Phase 1 gate at d = 4: RMSE ≤ 0.85 × <d=4 value> = <value> % at ≤ 10 NFE.

## Training-step timing
| config | params | sec/step | peak GB | sec/epoch | 100 epochs | 200 epochs |
|---|---|---|---|---|---|---|
| base 64, batch 8, 1 GPU | | | | | | |
| base 256, batch 8/GPU, FSDP 2 GPUs | | | | | | |

## Gate
- [ ] tests pass
- [ ] baselines recorded in tests/data/baseline_phase0.json
- [ ] timings recorded in results/residual_warmstart/timing.json
```

Fill every cell from the committed JSON files (no blanks left).

- [ ] **Step 3: Commit and push**

```bash
git add docs/superpowers/plans/2026-10-08-residual-warm-start-phase0-RESULTS.md
git commit -m "docs(warmstart): Phase 0 results and gate

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JwXhqBayn3pVe5Y9kfV6YP"
git push git@github.com:Chaoyi-He1/Diffu_HSI.git worktree-docs-diffusion-proof
```

Then present the results note to the user: this is the first decision point in the spec (baselines and timings), before the Phase 1 plan is written.
