# Diffusion Reconstruction From a Poor EC Sensor (Phase 3) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the simulated EC-sensor forward models (sequential and snapshot mode, additive noise at a chosen SNR, three devices), the noise-aware MMSE baseline, two new diffusion lines (an unconditional prior with ΠGDM guidance, and a family-conditioned residual diffusion) plus family-trained regressor and variance-head baselines, and the evaluation / verdict harness, then run Stage 0, the four trainings and the pre-registered E1–E3 evaluations of the spec.

**Architecture:** A new dataset class simulates the measurement y = Rᵀx₀₁ + n from the native 31-band cube with a chosen device, mode and SNR and returns, next to the cube, the MMSE estimate x̂ and posterior std s (the sufficient-statistic conditioning); the networks never see raw y. Four new modes ride on the existing `ResidualWrapper` / `main_2d_residual_fsdp.py` machinery: `uncond` (no context, no extra channels) and the family modes `cond_fam` / `regress_fam` / `varhead_fam` (input [x_t, x̂, s], residual whitened by s). `model/guidance.py` adds ΠGDM and DPS samplers over the existing linear schedule for the prior line. `scripts/sensor/sd_eval.py` writes one record per (device, mode, SNR, subset) and `sd_verdict.py` applies the pre-registered T1–T4 from a manifest of records.

**Tech Stack:** Python 3.10, PyTorch, numpy, scipy (conda env `hsi`, `/home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python`), pytest; the existing `main_2d_residual_fsdp.py`, `model/residual_wrapper.py`, `model/diffusion_trainer.py`, `model/regression_trainer.py`, `model/varhead_trainer.py`, `scripts/uq/metrics.py`, `scripts/uq/select_epoch.py`, `scripts/paired_stats.py`, `scripts/eval_warmstart.py::metrics`, `scripts/pilot/launch_run.sh`.

**Spec:** `docs/superpowers/specs/2026-11-03-sensor-degradation-diffusion-design.md` (approved by the user 2026-11-04, commit ac4e4fd). Earlier notes it builds on: `docs/superpowers/plans/2026-10-19-b-track-prereg.md` (B.1–B.8), `docs/superpowers/plans/2026-10-22-regressor-centred-uq-diffusion-RESULTS.md` (Phase 2a).

## Global Constraints

- Scales. Trainer scale is [−1, 1]; x₀₁ = (x + 1) / 2 is the physical reflectance scale on which y, σ, P_y, s_floor are defined. s_floor = 0.01 on x₀₁ = **0.02 on the trainer scale**; posterior stds are converted by × 2 when they enter the trainer scale. Metrics are on the [0, 1] display scale in % (`scripts/eval_warmstart.py::metrics`, `scripts/uq/metrics.py::to01`).
- Sensor model (spec §4.1, user Q1–Q3): legacy wavelength axes (`device1`, `device2`: `np.linspace(400, 1000, n_wavelengths)`; `ph5`: `np.arange(400, 1560, 10)`), 30 states by `np.linspace(0, N − 1, 30, dtype=int)`, per-column min–max normalisation over the 31 HFD bands; `R_Device2.mat` is transposed before use; the files' own `wavelength` fields are never used.
- Noise (spec §4.3): white Gaussian, equal σ on every channel, σ² = P_y / 10^(SNR/10) with ONE global P_y (device1, sequential, 9 000 training cubes); no σ = 0 anywhere: the floor is 60 dB. Family: SNR ~ U[20, 50] dB, floor with p = 0.1, snapshot with p = 0.5, device1 only. Evaluation grid {60, 50, 40, 30, 20} dB.
- Snapshot tile (5, 6): pixel (i, j) measures channel k = (i mod 5) · 6 + (j mod 6); unmeasured entries of y are 0 and the one-hot mask travels with y.
- Guidance operator on the trainer scale: y = H x₆₄ + b + n with H = ½ Rᵀ A⁺ (30 × 64), b = ½ Rᵀ A⁺ 𝟙₆₄, A = `linear_estimate.expand_matrix(wavelens, 64)` (64 × 31), A⁺ = `np.linalg.pinv(A)`.
- Samplers: K = 32 everywhere; C: DDIM-10 from pure noise, the whitened residual r clamped to [−8, 8] per step; B: ΠGDM with r_t² = 1 − ᾱ_t, guidance scale √ᾱ_t, x̂₀ clamped to [−1, 1], 50 steps unless the toy test (Task 9) requires 100; seeds 0 … K − 1 per cube batch.
- B is evaluated on the 50-cube subset (`ddpm1000_subset`) in every setting; every other method on the 200 gate cubes and, through the record's files, pairable on the 50. B's GPU cap: 12 h; drop order PH5 40 dB, E3 60 dB, PH5 20 dB, Device 2 40 dB.
- Thresholds (spec §6.3) are coded once, in `scripts/sensor/sd_verdict.py` (Task 11), before any Stage 1 number exists: T1 ratio ≤ 1.05 to D by 2 clustered SE; T2 CRPS below L's posterior and V by 2 clustered SE and cov90 ∈ [0.85, 0.95] at 30 and 20 dB; T3 (RMSE_line / RMSE_L)_device2 ≤ 1.10 · (RMSE_line / RMSE_L)_device1 at 30 dB; T4 ratio ≤ 0.85 to L_demosaic by 2 clustered SE at 30 dB snapshot; main line = more passes, tie → T2 at 20 dB.
- Results under `results/residual_warmstart/sensor/` (`config.json`, `stage0/`, `toy_guidance.json`, `eval/`, `manifest__*.json`, `verdict__*.json`); run directories `results/residual_warmstart/runs/{uncond_b64,cond_fam_b64,regress_fam_b64,varhead_fam_b64}`. Never write under `eval/`, `uq/`, `diagnostics/` of earlier phases; every record-writing script refuses to overwrite.
- `data_loader/HFD_dataset.py`, `data_loader/hfd_residual.py`, `data_loader/linear_estimate.py`, the Phase 1/2a evaluators and the shared prior `linear_prior_R1.npz` (sha 1ad9108f…) are not modified.
- GPU runs go through `scripts/pilot/launch_run.sh`; never share a GPU with the user's `dm_snn` process (the launcher refuses a busy GPU). GPU 1 is free; GPU 0 is free until the user's other project claims it.
- Tests run on CPU: `CUDA_VISIBLE_DEVICES= /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/<file> -q`; tests that need the dataset carry `@pytest.mark.dataset` (conftest skips them without it). Synthetic .mat fixtures live in `tests/conftest.py` (Task 1 adds `fake_sensor_root`).
- Commit after every task with the trailer lines `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01F64Hjw8qDzR4Dw6y9Y9C2C`; never stage `model/__pycache__/*.pyc`; push with `git push git@github.com:Chaoyi-He1/Diffu_HSI.git worktree-phase1-plan:worktree-phase1-plan`. If the hook refuses a compound command, write the helper to a file first and run it plainly.

## Design decisions (refinements of the spec, recorded in its as-built block by Task 19)

- **D1 — one dict batch for the sensor modes.** `HFDSensorSim` items are dicts and `sensor_collate_fn` returns a dict of channel-first tensors (`x`, `y`, `x0_hat`, `s_post`, `s_raw`, `mask`, `sigma`, `snr_db`, `snapshot`); the trainer's loops call `batch_to_cond(mode, batch, sigma_d, device)` (Task 6), which also accepts the legacy `(x, y_d, xh)` tuple, so the Phase 1/2a modes run unchanged. The centre stays in the `x0_hat` cond slot (as Phase 2a's D1), `s_post` is a new slot, and `cond['sensor']` is None for every sensor mode (no raw-y context; spec §5.2).
- **D2 — the per-element scale replaces σ_d for the family modes.** `target_for(mode, x, x0_hat, scale)` receives the tensor s̃ = max(2 s₀₁, 0.02) for `cond_fam` / `regress_fam` / `varhead_fam` and the scalar σ_d otherwise. The checkpoint meta's `sigma_d` field holds 0.02 (the trainer-scale floor) for the sensor datasets, and `checkpoint_extra()` carries the training-config sha, s_floor, P_y, the floor SNR, the family and the training device.
- **D3 — no persistent workers for datasets with `set_epoch`.** The family's noise is redrawn per epoch through `HFDSensorSim.set_epoch(e)` (seed = (config seed, epoch, file index)); worker processes copy the dataset at fork, so `make_loader` passes `persistent_workers=False` when the dataset has `set_epoch`. The monitoring dataset never calls `set_epoch` (fixed seeds, epoch 0).
- **D4 — `uncond` takes `--ds 1` and `--sensor_config` like the family modes** (it reads cubes through `HFDSensorSim` too, ignoring y) so one dataset class, one meta and one launcher path serve all four new modes; the launcher appends `--ds 1` for them.
- **D5 — the toy that licenses the calibration clauses uses an isotropic unit prior** (N(0, I₆₄), no clamp), for which ΠGDM's r_t² = 1 − ᾱ_t is the exact p(x₀ | x_t) covariance; a second toy with the HFD prior covariance is run and recorded report-only (it measures the cost of the isotropic approximation on this data).
- **D6 — L's Gaussian posterior uses the unfloored std** (`s_raw`), the networks the floored one (`s_post`).

## Review Focus

1. A checkpoint trained under one training config (prior, axis rule, P_y, family) evaluated with another: `sd_eval.check_models` must refuse on `sensor_config_sha` — pinned in Task 10.
2. Garbage in the unmeasured entries of a snapshot y must not leak: `measure` zeroes them (Task 2), `LinearMeasurement.solve` ignores unmeasured rows (Task 9), and `sd_eval`'s B output is unchanged when off-mask entries are corrupted — pinned in Task 10.
3. The 60 dB floor keeps every MMSE solve finite on the real, numerically rank-deficient R: `posterior_sequential` at σ(60 dB) on `small_R` returns finite values — pinned in Task 3.
4. Persistent DataLoader workers would freeze the family's noise across epochs: the trainer test checks that epoch 2's first batch differs from epoch 1's while the monitoring batch is identical — pinned in Task 7.
5. A 50-cube B record paired with 200-cube records: `sd_verdict` pairs on the common files and refuses an intersection below 45 cubes — pinned in Task 11.

## File structure

| path | responsibility |
|---|---|
| `data_loader/sensor_model.py` | `load_sensor(device, data_root)`: legacy-axis R (normalised and raw), state selection, sha |
| `data_loader/sensor_forward.py` | `mosaic_mask`, `sigma_from_snr`, `measure`, `guidance_operator`, `apply_H`, `apply_HT` |
| `data_loader/mmse.py` | noise-aware MMSE: sequential and snapshot posteriors, lift to 64 bands, demosaic-then-MMSE |
| `data_loader/sensor_config.py` | the forward-model config file: load, validate, `training_sha` |
| `scripts/sensor/make_config.py` | computes P_y and the demosaic error table, writes `config.json` |
| `data_loader/hfd_sensor_sim.py` | `HFDSensorSim` dataset (family or fixed setting), `sensor_collate_fn` |
| `model/residual_wrapper.py` | modes `uncond`, `cond_fam`, `regress_fam`, `varhead_fam`; `FAM_MODES`, `SENSOR_MODES`; wrapper `use_s_post` / `use_context`; `batch_to_cond` |
| `model/u2net_hyperspectral.py` | `forward` skips the context encoder when `context is None` |
| `main_2d_residual_fsdp.py` | `--sensor_config`, `--sensor_device`, mode checks, dataset branch, collate choice, loops via `batch_to_cond`, `set_epoch`, meta + resume check |
| `scripts/pilot/launch_run.sh` | the four new modes, `--sensor_config` guard, appended `--ds 1` |
| `model/guidance.py` | `LinearMeasurement`, `pigdm_sample`, `dps_sample`, `draw_guided` |
| `scripts/sensor/toy_guidance.py` | `GaussianDenoiser`, `exact_posterior`, `run_toy`, CLI that fixes the step count |
| `scripts/sensor/sd_eval.py` | one record per setting: L, L_demosaic, D, V, C, B |
| `scripts/sensor/sd_verdict.py` | T1–T4 from a manifest, pairing on common files, main-line rule |
| `scripts/sensor/stage0_baselines.py` | L on every setting with normalised and raw R; the 10 % check; principal angles |
| `scripts/pilot/smoke_sensor.sh` | CPU end-to-end smoke on 16 files |
| `tests/conftest.py` | `fake_sensor_root` fixture (tiny cubes + synthetic .mat response files) |

---

### Task 1: Sensor response loader on the legacy axes

**Files:**
- Create: `data_loader/sensor_model.py`
- Modify: `tests/conftest.py` (add the `fake_sensor_root` fixture)
- Test: `tests/test_sensor_model.py`

**Interfaces:**
- Produces: `DEVICES = ('device1', 'device2', 'ph5')`; `FILES = {'device1': ('R_Device1.mat', 'R'), 'device2': ('R_Device2.mat', 'R'), 'ph5': ('PH5_interp_results.mat', 'PH5_interp')}`; `def legacy_axis(device: str, n_wavelengths: int) -> np.ndarray` (device1 / device2: `np.linspace(400, 1000, n)`; ph5: `np.arange(400, 1560, 10)`, which must have length n or raise ValueError); `def raw_matrix(device: str, data_root: str) -> np.ndarray` ([n_wavelengths, n_states] float64; device2 transposed); `def load_sensor(device: str, data_root: str, n_states: int = 30, wavelens: np.ndarray = np.linspace(451, 855, 31)) -> dict` with keys `device, R (31 × 30 normalised), R_raw (31 × 30, same axis and states, un-normalised), selected_states (list[int]), n_states_total, axis_rule ('legacy'), file, sha256` where `sha256 = hashlib.sha256(np.ascontiguousarray(R, dtype=np.float64).tobytes()).hexdigest()`.
- Fixture `fake_sensor_root(tmp_path_factory)`: a directory with `MatFlower60/Train/P022/{a_1,a_2,b_1,b_2}.mat`, `P023/{c_1,c_2,c_3,d_1}.mat`, `P024/{e_1,e_2,e_3,e_4}.mat` (12 cubes, `truth` [16, 16, 31] float32 in [0, 1], smooth spectra on a 6-dim subspace like `synthetic_prior`), `R_Device1.mat` (`R` [401 × 71] Gaussian filters over 400–800 nm, `wavelength`, `voltage`), `R_Device2.mat` (`R` stored [42 × 400], i.e. transposed), `PH5_interp_results.mat` (`PH5_interp` [116 × 101], values of order 1e-7).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sensor_model.py
import numpy as np, pytest, scipy.io as sio
from data_loader import sensor_model as SM

def test_device2_file_is_transposed_before_use(fake_sensor_root):
    assert SM.raw_matrix('device2', fake_sensor_root).shape == (400, 42)
    assert SM.raw_matrix('device1', fake_sensor_root).shape == (401, 71)

def test_legacy_axes():
    np.testing.assert_allclose(SM.legacy_axis('device1', 401)[[0, -1]], [400, 1000])
    assert len(SM.legacy_axis('ph5', 116)) == 116 and SM.legacy_axis('ph5', 116)[0] == 400
    with pytest.raises(ValueError):
        SM.legacy_axis('ph5', 100)

def test_load_sensor_shapes_selection_and_normalisation(fake_sensor_root):
    s = SM.load_sensor('device1', fake_sensor_root)
    assert s['R'].shape == (31, 30) and s['R_raw'].shape == (31, 30)
    assert s['selected_states'] == np.linspace(0, 70, 30, dtype=int).tolist() and s['n_states_total'] == 71
    np.testing.assert_allclose(s['R'].min(0), 0, atol=1e-12); np.testing.assert_allclose(s['R'].max(0), 1, atol=1e-12)
    assert s['axis_rule'] == 'legacy' and len(s['sha256']) == 64
    assert SM.load_sensor('ph5', fake_sensor_root)['R'].shape == (31, 30)

def test_sha_changes_with_device(fake_sensor_root):
    assert SM.load_sensor('device1', fake_sensor_root)['sha256'] != SM.load_sensor('device2', fake_sensor_root)['sha256']

@pytest.mark.dataset
def test_device1_equals_legacy_loader(data_root):
    from data_loader.HFD_dataset import HFD_data
    ds = HFD_data(data_path=data_root, train_mode='image', eval_ratio=0.1, split='test', data_format='image', type='Flower', R_n=1, sensor_down_sample_rate=1)
    np.testing.assert_allclose(SM.load_sensor('device1', data_root)['R'], ds.sensor_R_matrix, atol=1e-12)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests/test_sensor_model.py -q`
Expected: FAIL with `ModuleNotFoundError: data_loader.sensor_model` (and the fixture missing).

- [ ] **Step 3: Add the `fake_sensor_root` fixture to `tests/conftest.py`** (session scope; writes the 12 cubes and the three .mat files described in Interfaces; Device 1 filters: 71 Gaussians with centres spread over 400–800 nm and width 25 nm; Device 2: 42 filters, stored transposed; PH5: 101 filters × 1e-7).

- [ ] **Step 4: Implement `data_loader/sensor_model.py`** with the functions of Interfaces. `load_sensor`: `raw_matrix` → select states → `scipy.interpolate.interp1d(legacy_axis(...), R_sel, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')(wavelens)` → `R_raw`; `R` = per-column min–max of `R_raw` (`+ 1e-20` in the denominator as the legacy loader).

- [ ] **Step 5: Run the tests to verify they pass** (same command; the dataset test runs on this machine). Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add data_loader/sensor_model.py tests/conftest.py tests/test_sensor_model.py
git commit -m "feat(sensor): response loader on the legacy wavelength axes for device1 / device2 (transposed file) / ph5, normalised and raw R with sha (Phase 3 Task 1)"
```

### Task 2: Forward models — sequential and snapshot measurement, noise, guidance operator

**Files:**
- Create: `data_loader/sensor_forward.py`
- Test: `tests/test_sensor_forward.py`

**Interfaces:**
- Consumes: `linear_estimate.expand_matrix`.
- Produces: `TILE = (5, 6)`; `N_CHANNELS = 30`; `def mosaic_mask(H: int, W: int, n_channels: int = 30) -> np.ndarray` ([H, W, 30] float32 one-hot, k = (i % 5) * 6 + (j % 6)); `def sigma_from_snr(snr_db: float, P_y: float) -> float`; `def measure(x01_31: np.ndarray, R: np.ndarray, mode: str, sigma: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]` (y [H, W, 30] float32 = x01_31 @ R + sigma · rng.standard_normal, mask [H, W, 30] float32; mode 'sequential' → mask of ones; 'snapshot' → y ⊙ mosaic_mask, so unmeasured entries are exactly 0; any other mode raises ValueError); `def guidance_operator(R: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]` (H [30, 64], b [30] as in Global Constraints); `def apply_H(x64: torch.Tensor, H: torch.Tensor) -> torch.Tensor` ([B, 64, H, W] → [B, 30, H, W], `torch.einsum('oc,bchw->bohw')`); `def apply_HT(v: torch.Tensor, H: torch.Tensor) -> torch.Tensor` ([B, 30, H, W] → [B, 64, H, W]).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sensor_forward.py
import numpy as np, torch
from data_loader import sensor_forward as F
from data_loader.linear_estimate import expand_matrix

def test_mosaic_mask_one_hot_and_tile_layout():
    m = F.mosaic_mask(16, 16)
    assert m.shape == (16, 16, 30) and m.dtype == np.float32
    np.testing.assert_array_equal(m.sum(-1), 1)
    assert m[0, 0, 0] == 1 and m[0, 1, 1] == 1 and m[1, 0, 6] == 1 and m[4, 5, 29] == 1 and m[5, 6, 0] == 1
    assert set(np.argmax(m[:5, :6], -1).ravel()) == set(range(30))      # one full tile holds every channel once

def test_sigma_from_snr():
    assert np.isclose(F.sigma_from_snr(30.0, 50.0), np.sqrt(50.0 / 1000.0))
    assert F.sigma_from_snr(60.0, 1.0) > 0

def test_measure_sequential_is_RT_x_plus_seeded_noise(small_R):
    x = np.random.default_rng(0).random((8, 8, 31))
    y, m = F.measure(x, small_R, 'sequential', 0.0, np.random.default_rng(1))
    np.testing.assert_allclose(y, x @ small_R, atol=1e-6); assert m.min() == 1 and y.dtype == np.float32
    y1, _ = F.measure(x, small_R, 'sequential', 0.1, np.random.default_rng(7)); y2, _ = F.measure(x, small_R, 'sequential', 0.1, np.random.default_rng(7))
    np.testing.assert_array_equal(y1, y2); assert not np.allclose(y1, y)

def test_measure_snapshot_zeroes_unmeasured_entries(small_R):
    x = np.random.default_rng(0).random((16, 16, 31))
    y, m = F.measure(x, small_R, 'snapshot', 0.05, np.random.default_rng(1))
    assert np.all(y[m == 0] == 0) and np.all(y[m == 1] != 0)
    with __import__('pytest').raises(ValueError):
        F.measure(x, small_R, 'bayer', 0.0, np.random.default_rng(0))

def test_measured_snr_matches_nominal(small_R):
    rng = np.random.default_rng(0); X = rng.random((50, 8, 8, 31))
    Y = np.stack([x @ small_R for x in X]); P_y = float((Y ** 2).mean()); sigma = F.sigma_from_snr(30.0, P_y)
    N = np.stack([F.measure(x, small_R, 'sequential', sigma, np.random.default_rng(i))[0] for i, x in enumerate(X)]) - Y
    assert abs(10 * np.log10((Y ** 2).mean() / (N ** 2).mean()) - 30.0) < 0.2

def test_guidance_operator_reproduces_y_from_the_64_band_cube(small_R):
    wl = np.linspace(451, 855, 31); A = expand_matrix(wl, 64)
    x01 = np.random.default_rng(0).random((5, 31)); x64 = (2 * x01 - 1) @ A.T          # trainer-scale 64-band cube
    H, b = F.guidance_operator(small_R, A)
    assert H.shape == (30, 64) and b.shape == (30,)
    np.testing.assert_allclose(x64 @ H.T + b, x01 @ small_R, atol=1e-9)

def test_apply_H_and_adjoint(small_R):
    H = torch.tensor(F.guidance_operator(small_R, expand_matrix(np.linspace(451, 855, 31), 64))[0])
    x = torch.randn(2, 64, 4, 4, dtype=torch.float64); v = torch.randn(2, 30, 4, 4, dtype=torch.float64)
    assert F.apply_H(x, H).shape == (2, 30, 4, 4)
    assert torch.allclose((F.apply_H(x, H) * v).sum(), (x * F.apply_HT(v, H)).sum())
```

- [ ] **Step 2: Run to verify failure** — Run: `... -m pytest tests/test_sensor_forward.py -q`. Expected: FAIL, module missing.

- [ ] **Step 3: Implement `data_loader/sensor_forward.py`** per Interfaces (numpy for the dataset side, torch for `apply_H` / `apply_HT`; `guidance_operator` uses `np.linalg.pinv(A)`).

- [ ] **Step 4: Run to verify pass.** Expected: 7 passed.

- [ ] **Step 5: Commit** — `git add data_loader/sensor_forward.py tests/test_sensor_forward.py` ; `git commit -m "feat(sensor): forward models — sequential and 5x6 snapshot measurement, SNR-defined noise, guidance operator H = R^T A^+ / 2 (Phase 3 Task 2)"`.

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
    xhat01, _ = MM.posterior_sequential(y01, small_R, mu01, C01, (2 * s) ** 2)
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

### Task 4: The forward-model config — P_y, demosaic error table, training sha

**Files:**
- Create: `data_loader/sensor_config.py`, `scripts/sensor/__init__.py`, `scripts/sensor/make_config.py`
- Test: `tests/test_sensor_config.py`

**Interfaces:**
- Consumes: `sensor_model.load_sensor`, `sensor_forward.measure` / `mosaic_mask`, `mmse.demosaic`, `warmstart_paths.{DATA_ROOT, RESULTS_ROOT, PRIOR, file_sha256}`, `HFD_data` (file list of the training split), `fit_linear_prior.load_cube`'s normalisation (re-implemented in Task 5's `load_cube01`; here `make_config` uses `HFDSensorSim.load_cube01` — so **Task 4's script imports from Task 5's module**; implement the script's cube loop against `data_loader.hfd_sensor_sim.load_cube01` and finish Task 5 before running `make_config.py` for real; the unit tests of Task 4 do not touch cubes).
- Produces: `SENSOR_DIR = os.path.join(RESULTS_ROOT, 'sensor')`; `CONFIG_PATH = os.path.join(SENSOR_DIR, 'config.json')`; `TRAIN_KEYS = ('prior_sha256', 'axis_rule', 'normalisation', 'state_selection', 'P_y', 's_floor01', 'snr_floor_db', 'family', 'device_train', 'tile', 'seed')`; `ALL_KEYS = TRAIN_KEYS + ('devices', 'e2_demosaic', 'n_files_P_y', 'created', 'prior_path')`; `def load_config(path: str = CONFIG_PATH) -> dict` (raises FileNotFoundError / ValueError on missing keys via `check_config`); `def check_config(cfg: dict) -> None`; `def training_sha(cfg: dict) -> str` (sha256 of `json.dumps({k: cfg[k] for k in TRAIN_KEYS}, sort_keys=True, separators=(',', ':'))`); `def default_training_fields(prior_sha: str) -> dict` = `dict(prior_sha256=prior_sha, axis_rule='legacy', normalisation='column-minmax', state_selection='linspace30', s_floor01=0.01, snr_floor_db=60.0, family=dict(snr_low_db=20.0, snr_high_db=50.0, p_floor=0.1, p_snapshot=0.5), device_train='device1', tile=[5, 6], seed=0)` (P_y added by the script).
- `scripts/sensor/make_config.py` CLI: `--out` (default CONFIG_PATH), `--data_root`, `--prior`, `--n_e2_files 200`, `--limit_files 0` (debug: P_y over the first N training cubes), `--seed 0`; computes `P_y` = mean of y² (physical, sequential, device1) over the training split's cubes in float64, `e2_demosaic` [30] = per-channel mean squared error of `demosaic(measure_snapshot(x))` against the clean sequential y over the first `n_e2_files` training cubes (noise-free), `devices` = {name: {sha256, file, n_states_total, selected_states}} for the three devices; refuses to overwrite `--out`; prints P_y and training_sha.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sensor_config.py
import json, pytest
from data_loader import sensor_config as SC

def _cfg(**kw):
    c = SC.default_training_fields('p' * 64); c.update(P_y=50.0, devices={}, e2_demosaic=[0.0] * 30, n_files_P_y=10, created='t', prior_path='x'); c.update(kw); return c

def test_training_sha_ignores_non_training_keys_and_key_order():
    a, b = _cfg(created='t1', n_files_P_y=10), _cfg(created='t2', n_files_P_y=9000)
    assert SC.training_sha(a) == SC.training_sha(b)
    assert SC.training_sha(dict(reversed(list(a.items())))) == SC.training_sha(a)

def test_training_sha_changes_with_P_y_and_family():
    assert SC.training_sha(_cfg(P_y=50.0)) != SC.training_sha(_cfg(P_y=51.0))
    f = dict(_cfg()['family'], p_snapshot=0.4); assert SC.training_sha(_cfg(family=f)) != SC.training_sha(_cfg())

def test_check_config_refuses_missing_keys(tmp_path):
    c = _cfg(); del c['P_y']
    with pytest.raises(ValueError, match='P_y'):
        SC.check_config(c)
    p = tmp_path / 'c.json'; p.write_text(json.dumps(_cfg())); assert SC.load_config(str(p))['s_floor01'] == 0.01
    with pytest.raises(FileNotFoundError):
        SC.load_config(str(tmp_path / 'none.json'))

def test_make_config_refuses_overwrite(tmp_path, monkeypatch):
    from scripts.sensor import make_config as MC
    p = tmp_path / 'c.json'; p.write_text('{}')
    with pytest.raises(FileExistsError):
        MC.main(['--out', str(p), '--data_root', str(tmp_path), '--prior', 'none.npz'])
```

- [ ] **Step 2: Run to verify failure.** Expected: FAIL, module missing.

- [ ] **Step 3: Implement `data_loader/sensor_config.py` and `scripts/sensor/make_config.py`** per Interfaces (`make_config.main(argv=None)` checks `--out` first, before loading anything).

- [ ] **Step 4: Run to verify pass.** Expected: 4 passed.

- [ ] **Step 5: Commit** — `git add data_loader/sensor_config.py scripts/sensor/__init__.py scripts/sensor/make_config.py tests/test_sensor_config.py` ; `git commit -m "feat(sensor): forward-model config with a training sha over the fields a checkpoint must match; make_config computes P_y and the demosaic error table (Phase 3 Task 4)"`.

### Task 5: `HFDSensorSim` — the family / fixed-setting dataset and its collate

**Files:**
- Create: `data_loader/hfd_sensor_sim.py`
- Test: `tests/test_hfd_sensor_sim.py`

**Interfaces:**
- Consumes: `HFD_data` (file list and split only), `sensor_model.load_sensor`, `sensor_forward.{measure, sigma_from_snr, mosaic_mask}`, `mmse.mmse_maps`, `linear_estimate.{expand_matrix, load_prior}`, `sensor_config.{training_sha, check_config}`.
- Produces: `def load_cube01(path: str) -> np.ndarray` ([H, W, 31] float64 = (g − g.min()) / (g.max() − g.min() + 1e-20), the loader's per-cube normalisation on [0, 1]); `MODES_FWD = ('sequential', 'snapshot')`; `class HFDSensorSim(torch.utils.data.Dataset)` with `__init__(self, data_root: str, split: str, config: dict, device: str = 'device1', setting: dict | None = None, prior_path: str = PRIOR, eval_ratio: float = 0.1, type: str = 'Flower')` — `setting=None` → training family; `setting=dict(mode=..., snr_db=...)` → fixed; attributes `img_list` (from an internal `HFD_data(..., R_n=1, sensor_down_sample_rate=1)`, settable), `sensor` (the `load_sensor` dict), `R`, `prior`, `A`, `wavelens`, `d = 1`, `sigma_d = 2 * config['s_floor01']`, `s_floor_tr = 2 * config['s_floor01']`, `P_y`, `epoch = 0`, `collate_fn = sensor_collate_fn`, `config_sha = training_sha(config)`; methods `set_epoch(self, e: int)`, `draw_setting(self, idx: int) -> tuple[str, float]` (rng = `np.random.default_rng([config['seed'], self.epoch, idx])`: snapshot with p_snapshot, SNR = floor with p_floor else U[low, high]; fixed setting returns it), `__getitem__(self, idx) -> dict` with keys `x` [H, W, 64] float32 trainer scale (= A applied to 2 x01 − 1), `y` [H, W, 30] float32 physical, `x0_hat` [H, W, 64] float32 trainer scale, `s_post` [H, W, 64] float32 = max(s_tr, s_floor_tr), `s_raw` [H, W, 64] float32 (unfloored, trainer scale), `mask` [H, W, 30] float32, `sigma` float32, `snr_db` float32, `snapshot` float32 (0/1); noise rng = `np.random.default_rng([config['seed'] + 1, self.epoch, idx])`; `checkpoint_extra(self) -> dict(sensor_config_sha, s_floor01, P_y, snr_floor_db, family, device_train, sensor_device, R_sha256)`; `def sensor_collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]` (HWC → BCHW for the map keys, 1-D tensors for the scalars).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hfd_sensor_sim.py
import numpy as np, pytest, torch
from data_loader import hfd_sensor_sim as DS
from data_loader.sensor_config import default_training_fields
from data_loader.sensor_forward import mosaic_mask

def _cfg(): return dict(default_training_fields('p' * 64), P_y=50.0)

@pytest.fixture
def ds(fake_sensor_root, synthetic_prior, tmp_path):
    np.savez(tmp_path / 'prior.npz', mu=synthetic_prior['mu'], C=synthetic_prior['C'], s=3e-7)
    return DS.HFDSensorSim(fake_sensor_root, 'train', _cfg(), device='device1', prior_path=str(tmp_path / 'prior.npz'))

def test_item_keys_shapes_and_scales(ds):
    it = ds[0]
    assert set(it) == {'x', 'y', 'x0_hat', 's_post', 's_raw', 'mask', 'sigma', 'snr_db', 'snapshot'}
    assert it['x'].shape == it['x0_hat'].shape == it['s_post'].shape == (16, 16, 64) and it['y'].shape == it['mask'].shape == (16, 16, 30)
    assert it['x'].dtype == np.float32 and -1 <= it['x'].min() and it['x'].max() <= 1
    assert it['s_post'].min() >= 2 * 0.01 - 1e-6 and (it['s_raw'] <= it['s_post'] + 1e-6).all()

def test_y_is_R_transpose_x01_plus_seeded_noise(ds):
    ds_fixed = DS.HFDSensorSim(ds.data_root, 'train', _cfg(), setting=dict(mode='sequential', snr_db=60.0), prior_path=ds.prior_path)
    it = ds_fixed[1]; x01 = DS.load_cube01(ds_fixed.img_list[1]); clean = x01 @ ds_fixed.R
    assert np.abs(it['y'] - clean).max() < 5 * it['sigma'] + 1e-6 and not np.allclose(it['y'], clean)
    np.testing.assert_array_equal(ds_fixed[1]['y'], it['y'])                      # same epoch, same idx -> same noise

def test_set_epoch_changes_training_noise_but_not_the_cube(ds):
    a = ds[2]; ds.set_epoch(1); b = ds[2]
    np.testing.assert_array_equal(a['x'], b['x']); assert not np.array_equal(a['y'], b['y'])

def test_family_frequencies(ds):
    draws = [ds.draw_setting(i) for i in range(2000)]
    assert abs(np.mean([m == 'snapshot' for m, _ in draws]) - 0.5) < 0.04
    assert abs(np.mean([s == 60.0 for _, s in draws]) - 0.1) < 0.03
    others = [s for _, s in draws if s != 60.0]; assert min(others) >= 20.0 and max(others) <= 50.0

def test_snapshot_item_mask_and_zero_off_mask(fake_sensor_root, ds):
    d2 = DS.HFDSensorSim(fake_sensor_root, 'train', _cfg(), setting=dict(mode='snapshot', snr_db=30.0), prior_path=ds.prior_path)
    it = d2[0]; np.testing.assert_array_equal(it['mask'], mosaic_mask(16, 16)); assert np.all(it['y'][it['mask'] == 0] == 0) and it['snapshot'] == 1

def test_collate_and_extra(ds):
    b = DS.sensor_collate_fn([ds[0], ds[1]])
    assert b['x'].shape == (2, 64, 16, 16) and b['y'].shape == (2, 30, 16, 16) and b['sigma'].shape == (2,)
    e = ds.checkpoint_extra(); assert e['sensor_config_sha'] == ds.config_sha and e['sensor_device'] == 'device1' and len(e['R_sha256']) == 64

def test_device2_dataset_builds(fake_sensor_root, ds):
    assert DS.HFDSensorSim(fake_sensor_root, 'test', _cfg(), device='device2', setting=dict(mode='sequential', snr_db=30.0), prior_path=ds.prior_path)[0]['y'].shape == (16, 16, 30)

@pytest.mark.dataset
def test_x64_matches_the_legacy_loader(data_root):
    from data_loader.HFD_dataset import HFD_data
    from data_loader.warmstart_paths import PRIOR
    leg = HFD_data(data_path=data_root, train_mode='image', eval_ratio=0.1, split='test', data_format='image', type='Flower', R_n=1, sensor_down_sample_rate=1)
    new = DS.HFDSensorSim(data_root, 'test', dict(default_training_fields('p' * 64), P_y=50.0), setting=dict(mode='sequential', snr_db=60.0), prior_path=PRIOR)
    assert new.img_list == leg.img_list
    x_leg, y_leg = leg[3]; it = new[3]
    np.testing.assert_allclose(it['x'], x_leg, atol=1e-5)
    np.testing.assert_allclose(it['y'] - (y_leg + leg.sensor_R_matrix.sum(0)) / 2, 0, atol=5 * it['sigma'])   # physical y = (loader y + R^T 1) / 2 up to noise
```

- [ ] **Step 2: Run to verify failure.** Expected: FAIL, module missing.

- [ ] **Step 3: Implement `data_loader/hfd_sensor_sim.py`** per Interfaces. `__getitem__`: x01 = load_cube01 → (mode, snr) = draw_setting → σ = sigma_from_snr(snr, P_y) → (y, mask) = measure(x01, R, mode, σ, rng) → (x̂₆₄, s₆₄) = mmse_maps(y, mask if snapshot else None, R, prior, σ, A) → x = ((2 x01 − 1) @ A.T).astype(float32). The prior check of `HFDResidualData` (prior R equals the loader's R) is **not** applied: the prior's μ, C are device-independent; only `mu`, `C`, `s` are read.

- [ ] **Step 4: Run to verify pass** (the dataset test runs here too). Expected: 8 passed.

- [ ] **Step 5: Commit** — `git add data_loader/hfd_sensor_sim.py tests/test_hfd_sensor_sim.py` ; `git commit -m "feat(sensor): HFDSensorSim — simulated EC measurement (family or fixed setting) with MMSE centre and posterior-std maps as the conditioning, per-epoch noise, dict collate (Phase 3 Task 5)"`.

### Task 6: The four new modes in the wrapper and the U-Net's context-free path

**Files:**
- Modify: `model/residual_wrapper.py`, `model/u2net_hyperspectral.py:424-452` (the `context_encoded` / `create_context_for_resolution` lines of `forward`)
- Test: `tests/test_sensor_modes.py`

**Interfaces:**
- Produces: `MODES = ('residual', 'standard', 'standard_xhat', 'regress', 'residual_reg', 'varhead', 'uncond', 'cond_fam', 'regress_fam', 'varhead_fam')`; `FAM_MODES = ('cond_fam', 'regress_fam', 'varhead_fam')`; `SENSOR_MODES = ('uncond',) + FAM_MODES`; `CENTRED_MODES += FAM_MODES`; `ONE_CALL_MODES = ('regress', 'varhead', 'regress_fam', 'varhead_fam')`; `ResidualWrapper.__init__(self, net, use_x0_hat: bool, use_s_post: bool = False, use_context: bool = True)`; `forward(x_t, cond, t)`: inp = cat([x_t] + [cond['x0_hat']] if use_x0_hat + [cond['s_post']] if use_s_post), context = cond['sensor'] if use_context else None (the `'sensor'` key is required only when use_context); `build_residual_model(mode, base_channels, spectral_channels=64, sensor_channels=30)`: `uncond` → extra 0, use_x0_hat False, use_context False; FAM → extra 2 · spectral, use_x0_hat True, use_s_post True, use_context False; legacy modes unchanged; `def target_for(mode, x, x0_hat, scale)` (scale scalar or tensor; `uncond` and `standard*` return x); `def batch_to_cond(mode: str, batch, sigma_d: float, device) -> tuple[torch.Tensor, dict, object]` — tuple batch `(x, y_d, xh)` → `(x, {'sensor': y_d, 'x0_hat': xh}, sigma_d)`; dict batch → `(b['x'], {'sensor': None, 'x0_hat': b['x0_hat'], 's_post': b['s_post']}, b['s_post'])` for FAM and `(b['x'], {'sensor': None}, None)` for `uncond`; all tensors moved to `device` (non_blocking); `regress_forward` / `varhead_forward` unchanged (the zero placeholder has `cond['x0_hat']`'s 64 channels; the wrapper appends s_post); `U2NetHyperspectral.forward`: `context_encoded = self.context_encoder(context) if context is not None else None` and `create_context_for_resolution` returns None when it is None (every block and `bridge_attn` already accept None).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sensor_modes.py
import torch
from model import residual_wrapper as RW
from model.u2net_hyperspectral import U2NetHyperspectral

def test_mode_tables():
    assert set(RW.FAM_MODES) <= set(RW.MODES) and 'uncond' in RW.MODES and set(RW.FAM_MODES) <= set(RW.CENTRED_MODES)

def test_u2net_runs_without_context_and_depends_on_t():
    net = U2NetHyperspectral(spectral_channels=8, sensor_channels=5, base_channels=16)
    x = torch.randn(2, 8, 16, 16)
    a = net(x, None, torch.tensor([10, 10])); b = net(x, None, torch.tensor([900, 900]))
    assert a.shape == (2, 8, 16, 16) and not torch.allclose(a, b)

def test_uncond_and_fam_models_shapes():
    u = RW.build_residual_model('uncond', 16, spectral_channels=8, sensor_channels=5)
    assert u.net.input_proj.in_channels == 8 and u(torch.randn(2, 8, 16, 16), {'sensor': None}, torch.zeros(2, dtype=torch.long)).shape == (2, 8, 16, 16)
    c = RW.build_residual_model('cond_fam', 16, spectral_channels=8, sensor_channels=5)
    cond = {'sensor': None, 'x0_hat': torch.randn(2, 8, 16, 16), 's_post': torch.rand(2, 8, 16, 16) + 0.02}
    assert c.net.input_proj.in_channels == 24 and c(torch.randn(2, 8, 16, 16), cond, torch.zeros(2, dtype=torch.long)).shape == (2, 8, 16, 16)
    r = RW.build_residual_model('regress_fam', 16, spectral_channels=8, sensor_channels=5)
    assert RW.regress_forward(r, cond).shape == (2, 8, 16, 16) and RW.varhead_forward(RW.build_residual_model('varhead_fam', 16, 8, 5), cond).shape == (2, 8, 16, 16)

def test_legacy_modes_unchanged():
    b = RW.build_residual_model('residual', 16, spectral_channels=8, sensor_channels=5)
    assert b.net.input_proj.in_channels == 16 and b.use_context and not b.use_s_post
    b2 = RW.build_residual_model('residual', 16, spectral_channels=8, sensor_channels=5); b2.load_state_dict(b.state_dict(), strict=True)

def test_target_for_accepts_tensor_scale_and_uncond_returns_x():
    x, xh, s = torch.ones(1, 2, 2, 2), torch.zeros(1, 2, 2, 2), torch.full((1, 2, 2, 2), 0.5)
    assert torch.equal(RW.target_for('cond_fam', x, xh, s), 2 * torch.ones_like(x)) and torch.equal(RW.target_for('uncond', x, None, None), x)

def test_batch_to_cond_both_batch_kinds():
    x, cond, scale = RW.batch_to_cond('residual', (torch.zeros(1, 4, 2, 2), torch.zeros(1, 3, 1, 1), torch.zeros(1, 4, 2, 2)), 0.05, torch.device('cpu'))
    assert set(cond) == {'sensor', 'x0_hat'} and scale == 0.05
    b = dict(x=torch.zeros(1, 4, 2, 2), y=torch.zeros(1, 3, 2, 2), x0_hat=torch.zeros(1, 4, 2, 2), s_post=torch.full((1, 4, 2, 2), 0.1), s_raw=torch.zeros(1, 4, 2, 2), mask=torch.ones(1, 3, 2, 2), sigma=torch.zeros(1), snr_db=torch.zeros(1), snapshot=torch.zeros(1))
    x, cond, scale = RW.batch_to_cond('cond_fam', b, 0.02, torch.device('cpu')); assert cond['sensor'] is None and torch.equal(scale, b['s_post'])
    x, cond, scale = RW.batch_to_cond('uncond', b, 0.02, torch.device('cpu')); assert cond == {'sensor': None} and scale is None
```

- [ ] **Step 2: Run to verify failure.** Expected: FAIL (`FAM_MODES` missing; U-Net raises on None context).

- [ ] **Step 3: Implement** the wrapper changes and the two-line U-Net change per Interfaces. Keep `MODES`' first six entries in their current order (the launcher's and the Phase 1/2a tests' lists depend on the names, not the order, but do not rename anything).

- [ ] **Step 4: Run the new tests and the old mode tests.** Run: `... -m pytest tests/test_sensor_modes.py tests/test_wrapper.py tests/test_model_extra_in.py tests/test_uq_modes.py tests/test_regress_mode.py -q`. Expected: all pass.

- [ ] **Step 5: Commit** — `git add model/residual_wrapper.py model/u2net_hyperspectral.py tests/test_sensor_modes.py` ; `git commit -m "feat(sensor): modes uncond / cond_fam / regress_fam / varhead_fam — context-free U-Net path, s_post input slot, per-element residual scale, batch_to_cond for tuple and dict batches (Phase 3 Task 6)"`.

### Task 7: The trainer learns the sensor modes

**Files:**
- Modify: `main_2d_residual_fsdp.py` (`get_parser` 66–101, `make_datasets` 135–160, `make_loader` 163–173, `check_mode_args` 176–189, `make_trainer` 192–199, `checkpoint_meta` 225–243, `check_resume_meta` 279–312, `train_one_epoch` 397–442, `validate` 445–466, `build_everything` 469–486, `run_epochs` 489–553)
- Test: `tests/test_sensor_training.py`

**Interfaces:**
- Consumes: `HFDSensorSim`, `sensor_collate_fn`, `sensor_config.load_config`, `residual_wrapper.{SENSOR_MODES, FAM_MODES, ONE_CALL_MODES, batch_to_cond, target_for}`.
- Produces: flags `--sensor_config` (default '') and `--sensor_device` (choices `DEVICES`, default 'device1'); `check_mode_args`: `(args.mode in SENSOR_MODES) == bool(args.sensor_config)` else ValueError; SENSOR_MODES require `args.d == 1`; `(args.mode in ONE_CALL_MODES) == (args.prediction_type == 'direct')`; `(args.mode in ('varhead', 'varhead_fam')) == (args.loss_type == 'nll')`; `make_datasets`: SENSOR_MODES → `train = HFDSensorSim(args.data_root, 'train', cfg, device=args.sensor_device)`, `val = HFDSensorSim(args.data_root, 'test', cfg, device=args.sensor_device)` (family draws with fixed epoch 0), gate cubes removed from `val.img_list` by the existing block (factored into `def exclude_gate(val, data_root)`); `make_loader`: `collate_fn=getattr(ds, 'collate_fn', residual_collate_fn)`, `persistent_workers = args.num_workers > 0 and not hasattr(ds, 'set_epoch')`; `make_trainer`: `regress_fam` → `RegressionTrainer`, `varhead_fam` → `VarheadTrainer`; `train_one_epoch` / `validate`: `x, cond, scale = batch_to_cond(mode, batch, sigma_d, device)`, `target = target_for(mode, x, cond.get('x0_hat'), scale)`; `run_epochs`: `if hasattr(ds_train, 'set_epoch'): ds_train.set_epoch(epoch)` before each epoch (the tuple-`parts` path uses `parts[0]`); `check_resume_meta(..., sensor_sha=None)`: when given, `meta['sensor_config_sha']` must equal it (listed among the mismatches); `build_everything` passes `train_ds.checkpoint_extra()['sensor_config_sha']` for SENSOR_MODES; `checkpoint_meta` unchanged (`extra` already carries the dataset's `checkpoint_extra()`).

- [ ] **Step 1: Write the failing tests** (pattern of `tests/test_uq_training.py`: a `StubSensorSet` whose items are the Task 5 dict with H = 16, 8 cubes, `collate_fn = sensor_collate_fn`, `set_epoch` that re-seeds the noise, `checkpoint_extra()` returning `dict(sensor_config_sha='cfgsha', s_floor01=0.01, P_y=50.0, snr_floor_db=60.0, family={}, device_train='device1', sensor_device='device1', R_sha256='r')`, `sigma_d = 0.02`, `prior = {'s': 3e-7}`, `img_list`)

```python
# tests/test_sensor_training.py  (names and assertions; bodies follow test_uq_training.py's helpers)
def test_check_mode_args_refusals():
    # uncond without --sensor_config; cond_fam with --ds 4; regress_fam without direct; varhead_fam without nll;
    # residual WITH --sensor_config  -> each raises ValueError
def test_make_loader_uses_the_dataset_collate_and_no_persistent_workers_with_set_epoch():
    # loader.collate_fn is sensor_collate_fn; loader.persistent_workers is False for StubSensorSet (num_workers=1), True for a tuple dataset
@pytest.mark.parametrize('mode,pt,loss', [('uncond', 'v', 'l1'), ('cond_fam', 'v', 'l1'), ('regress_fam', 'direct', 'l1'), ('varhead_fam', 'direct', 'nll')])
def test_two_epochs_train_on_cpu_and_log_finite_losses(tmp_path, mode, pt, loss):
    # tr.run_epochs(args, (ds, loader, model, trainer, opt, sched), tr.NO_DIST, CPU) with num_epochs=2, base 16, spectral 64, batch 4
    # -> train_log.jsonl has 2 records, both losses finite, DONE == '2', checkpoint meta has sensor_config_sha == 'cfgsha' and sigma_d == 0.02
def test_set_epoch_changes_training_noise_but_monitoring_is_fixed():
    # first batch 'y' of epoch 0 != epoch 1 on the train stub after set_epoch; the val stub (no set_epoch call) gives identical 'y'
def test_resume_refused_when_sensor_config_sha_differs(tmp_path):
    # check_resume_meta(ck with meta sensor_config_sha='other', ..., sensor_sha='cfgsha') raises ValueError mentioning sensor_config_sha
def test_legacy_residual_mode_still_trains(tmp_path):
    # the Phase 1 tuple dataset stub of tests/test_train_residual.py runs one epoch unchanged
```

- [ ] **Step 2: Run to verify failure.** Run: `... -m pytest tests/test_sensor_training.py -q`. Expected: FAIL (unknown flags / modes).

- [ ] **Step 3: Implement the trainer changes** per Interfaces.

- [ ] **Step 4: Run the new tests and every existing trainer test.** Run: `... -m pytest tests/test_sensor_training.py tests/test_train_residual.py tests/test_uq_training.py tests/test_regress_mode.py -q`. Expected: all pass.

- [ ] **Step 5: Commit** — `git add main_2d_residual_fsdp.py tests/test_sensor_training.py` ; `git commit -m "feat(sensor): trainer runs the four sensor modes — --sensor_config / --sensor_device, dict batches via batch_to_cond, per-epoch noise with non-persistent workers, config sha in the meta and the resume check (Phase 3 Task 7)"`.

### Task 8: Launcher guards for the sensor modes

**Files:**
- Modify: `scripts/pilot/launch_run.sh` (mode list line 22, the regress block 37–63, the varhead line 64–66, the `nohup` line 91)
- Test: `tests/test_launch_run_sensor.py`

**Interfaces:**
- Produces: the mode list becomes `residual standard standard_xhat regress residual_reg varhead uncond cond_fam regress_fam varhead_fam`; for MODE ∈ {uncond, cond_fam, regress_fam, varhead_fam}: `--sensor_config <path>` must appear in the extra args (else exit 1 "needs --sensor_config"), a `--ds N` with N ≠ 1 in the extra args is refused, and the launcher appends `--ds 1` after `"$@"`; the regress guards (loss l1|l2, `_l1_` / `_l2_` consistency, seed) apply to `regress_fam` as to `regress`; `varhead_fam` needs `--loss_type nll` like `varhead`; the longer-mode RUN_NAME guard covers `regress_fam_*` under `regress` and `varhead_fam_*` under `varhead` automatically through the list.

- [ ] **Step 1: Write the failing tests** (the stub pattern of `tests/test_launch_run_name.py`)

```python
def test_sensor_mode_without_config_is_refused(env, tmp_path):        # uncond with no --sensor_config -> returncode 1, 'needs --sensor_config', no run dir
def test_sensor_mode_appends_ds_1_and_passes_config(env, tmp_path):   # RUN_NAME=cond_fam_b64 cond_fam 0 --sensor_config /c.json -> stub args end with '--sensor_config /c.json --ds 1'
def test_sensor_mode_refuses_other_ds(env, tmp_path):                 # uncond ... --sensor_config /c.json --ds 4 -> refused
def test_regress_fam_needs_loss_and_direct_like_regress(env, tmp_path):  # regress_fam without --loss_type -> refused; with --loss_type l1 and RUN_NAME regress_fam_l2_x -> refused
def test_varhead_fam_needs_nll(env, tmp_path)
def test_longer_mode_name_guard_covers_fam(env, tmp_path):            # RUN_NAME=regress_fam_b64 under mode regress -> refused 'belongs to mode regress_fam'
```

- [ ] **Step 2: Run to verify failure.** Expected: FAIL (modes unknown to the list; no `--ds 1` appended).

- [ ] **Step 3: Implement the launcher changes.**

- [ ] **Step 4: Run the new and old launcher tests.** Run: `... -m pytest tests/test_launch_run_sensor.py tests/test_launch_run_name.py -q`. Expected: all pass.

- [ ] **Step 5: Commit** — `git add scripts/pilot/launch_run.sh tests/test_launch_run_sensor.py` ; `git commit -m "feat(sensor): launch_run.sh knows the sensor modes — --sensor_config required, --ds 1 appended, regress / varhead guards extended to the family modes (Phase 3 Task 8)"`.

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

### Task 10: The evaluation harness `sd_eval.py`

**Files:**
- Create: `scripts/sensor/sd_eval.py`
- Test: `tests/test_sd_eval.py`

**Interfaces:**
- Consumes: `uq_eval.{point_metrics, sample_metrics, gaussian_metrics}` (imported from `scripts.uq_eval`), `eval_warmstart.{metrics, load_subset, class_of, make_eval_loader}`, `HFDSensorSim`, `mmse.demosaic_then_mmse`, `residual_wrapper.{build_residual_model, compose, regress_forward, varhead_forward, LOGVAR_RANGE, FAM_MODES}`, `guidance.{LinearMeasurement, draw_guided}`, `sensor_forward.guidance_operator`, `sensor_config.{load_config, training_sha}`, `warmstart_paths.{git_head, file_sha256}`.
- Produces: `METHODS = ('L', 'L_demosaic', 'D', 'V', 'C', 'B')`; `EVAL_DIR = os.path.join(SENSOR_DIR, 'eval')`; CLI `--name` (required), `--sensor_device {device1,device2,ph5}`, `--mode {sequential,snapshot}`, `--snr_db` (float), `--subset {gate,ddpm}`, `--methods` (nargs +, default all applicable), `--regress`, `--varhead`, `--cond`, `--uncond` (checkpoint paths; each required iff its method is requested: D ← regress, V ← varhead, C ← cond, B ← uncond), `--K 32`, `--pigdm_steps 50`, `--guidance {pigdm,dps}`, `--sensor_config CONFIG_PATH`, `--batch_size 4`, `--num_workers 4`, `--limit_files 0`, `--device`, `--data_root`, `--prior`; `def check_models(args, cfg, device) -> tuple[dict, dict]` (loads each requested checkpoint, requires `meta['mode']` to be the method's mode and `meta['sensor_config_sha'] == training_sha(cfg)`, else ValueError naming the file; provenance per checkpoint: path, epoch, sha256, mode, git_head); `def run_eval(models, ds, loader, args, cfg, provenance, device) -> dict` with `per_cube[method]` lists: `L` → `gaussian_metrics(x0_hat, s_raw, x)` (mean_* keys hold the point metrics of x̂; `s_raw` on the trainer scale); `L_demosaic` (snapshot only) → point metrics of `demosaic_then_mmse(...)`'s x̂ plus its Gaussian metrics; `D` → `point_metrics(compose(x0_hat, s_post, regress_forward(models['D'], cond)), x)`; `V` → `gaussian_metrics(x0_hat, s_post · exp(0.5 · logvar.clamp(*LOGVAR_RANGE)), x)`; `C` → `sample_metrics(stack_k compose(x0_hat, s_post, r_k), x)` with r_k from `trainer.sample(models['C'], cond, shape, n_steps=10, method='ddim', clamp_x0=(-8.0, 8.0))` under `torch.manual_seed(k)`; `B` → `sample_metrics(draw_guided(models['B'], trainer, LinearMeasurement(H, b, y, mask or None, sigma), shape, K, pigdm_steps, args.guidance), x)`; record keys `name, setting=dict(sensor_device, mode, snr_db, sigma, R_sha256), subset, K, pigdm_steps, guidance, n, files, classes, per_cube, means, coverage_exact=bool(K >= min_k_for(0.9)), provenance=dict(sensor_config_sha, config_path, checkpoints, git_head, created, seconds, seeds, device, batch_size, limit_files)`; `def output_path(name) -> os.path.join(EVAL_DIR, name + '.json')`; refuses to overwrite; prints one line per method with rmse / crps / cov90 / ause.

- [ ] **Step 1: Write the failing tests** (stub dataset yielding the Task 5 dict with H = 8, `sigma_d = 0.02`, `R`, `A`, `img_list`; stub nets `Const` / `Noisy` from `tests/test_uq_eval.py`; checkpoints written with `torch.save(dict(model_state_dict=..., meta=dict(mode=..., sensor_config_sha=...)))`)

```python
def test_record_holds_every_requested_method_with_paired_files(tmp_path)   # methods L D V C B on a sequential stub; len(per_cube[m]['...']) == n for each; 'L_demosaic' absent
def test_snapshot_adds_L_demosaic(tmp_path)                                  # snapshot stub -> 'L_demosaic' present with mean_rmse_pct
def test_refuses_checkpoint_with_other_config_sha(tmp_path)                 # meta sensor_config_sha 'other' -> ValueError naming the checkpoint
def test_refuses_wrong_mode_checkpoint(tmp_path)                            # --cond pointing at a regress_fam checkpoint -> ValueError
def test_refuses_overwrite(tmp_path)
def test_B_is_unchanged_when_off_mask_entries_are_corrupted(tmp_path)        # snapshot stub, Const uncond model, K=2, 2 steps: per_cube['B'] identical after adding 1e3 to y where mask == 0
def test_coverage_exact_flag(tmp_path)                                       # K=32 -> True, K=8 -> False
```

- [ ] **Step 2: Run to verify failure.** Expected: FAIL, module missing.

- [ ] **Step 3: Implement `scripts/sensor/sd_eval.py`** per Interfaces; `main()` builds `HFDSensorSim(args.data_root, 'test', cfg, device=args.sensor_device, setting=dict(mode=args.mode, snr_db=args.snr_db), prior_path=args.prior)` restricted to the subset's files (`load_subset(args.data_root, 'files' | 'ddpm1000_subset')`, in that order), `make_eval_loader` with the dataset's `collate_fn`.

- [ ] **Step 4: Run to verify pass.** Expected: 7 passed.

- [ ] **Step 5: Commit** — `git add scripts/sensor/sd_eval.py tests/test_sd_eval.py` ; `git commit -m "feat(sensor): sd_eval — one record per (device, mode, SNR, subset) with L / L_demosaic / D / V / C / B, config-sha and mode checks, never overwrites (Phase 3 Task 10)"`.

### Task 11: The verdict `sd_verdict.py` — T1–T4 from a manifest

**Files:**
- Create: `scripts/sensor/sd_verdict.py`
- Test: `tests/test_sd_verdict.py`

**Interfaces:**
- Consumes: `paired_stats.cluster_se`, `metrics.min_k_for`, the `sd_eval` record format.
- Produces: thresholds as module constants `T1_RATIO = 1.05`, `T2_COV90 = (0.85, 0.95)`, `T2_SNRS = (30.0, 20.0)`, `T3_RATIO = 1.10`, `T3_SNR = 30.0`, `T4_RATIO = 0.85`, `T4_SNR = 30.0`, `MIN_COMMON = 45`; `MANIFEST_KEYS`: a manifest JSON `{"records": {"<device>/<mode>/<snr>": ["<record name>", ...]}, "name": "<verdict name>"}` (several records per setting are merged by method; a method present twice is an error); `def load_manifest(path) -> dict[str, dict[str, tuple[dict, list[str]]]]` (setting → method → (per_cube dict, files)); `def paired_on_common(a: dict, af: list, b: dict, bf: list, key_a: str, key_b: str) -> dict(mean, se, se_cluster, n, clusters, ratio_of_means)` (restricts both to the common files in `af`'s order; raises ValueError when fewer than `MIN_COMMON` are common); `def ratio_clause(num, den, files, key_num, key_den, limit, direction) -> dict(passed, ratio, ratio_2se, mean, se_cluster, n)` — the paired test m − limit · b with its clustered SE: `direction='le'` passes when mean + 2 se_cluster ≤ 0; `def t1(settings, line) -> dict[snr -> clause]` (line ∈ {'B', 'C'}; `mean_rmse_pct` of the line vs `rmse_pct` of D over E1's five SNRs); `def t2(settings, line) -> dict[snr -> dict(crps_vs_L, crps_vs_V, cov90, passed)]` at T2_SNRS (CRPS 2-SE clauses against `L` and `V`; cov90 of the line's record within T2_COV90; needs `K >= min_k_for(0.9)` of that record else ValueError unless `--report_only`); `def t3(settings, line) -> dict(ratio_dev1, ratio_dev2, passed)` ((RMSE_line / RMSE_L) on `device2/sequential/30` ≤ T3_RATIO × the same on `device1/sequential/30`, each ratio of means on common files; PH5's ratio reported under `ph5`); `def t4(settings, line) -> clause` (line vs `L_demosaic` on `device1/snapshot/30`, limit T4_RATIO, plus D's for the report); `def main_line(res: dict) -> str` (count passes of T1 (all five SNRs count as one pass when all pass), T2 (both SNRs), T3, T4 per line; more passes wins; tie → lower CRPS vs V at 20 dB); `def verdict(settings, report_only=False) -> dict(T1, T2, T3, T4, counts, main_line, report_only, missing)`; CLI `python scripts/sensor/sd_verdict.py --manifest <path> [--report_only]` writes `os.path.join(SENSOR_DIR, 'verdict__' + name + ('__report' if report_only else '') + '.json')`, refusing to overwrite; with `--report_only` missing settings are listed instead of raising and `main_line` is None.

- [ ] **Step 1: Write the failing tests** (synthetic records built by a helper `_rec(files, **method_arrays)` in the sd_eval format)

```python
def test_paired_on_common_restricts_to_common_files_and_refuses_small_overlap()   # 200 vs 50 files -> n == 50 in the 50's order; 200 vs 30 -> ValueError
def test_ratio_clause_pass_and_fail()                                             # line = 1.02 D everywhere -> T1 passes; 1.08 D -> fails
def test_t2_requires_K_19_unless_report_only()
def test_t3_uses_the_device1_reference_ratio()                                    # ratios 0.8 / 0.86 -> pass (1.075 <= 1.10); 0.8 / 0.9 -> fail
def test_t4_against_demosaic()
def test_main_line_counts_and_tiebreak()                                          # B passes T1-T4, C passes T1,T2 -> 'B'; equal counts -> lower CRPS vs V at 20 dB
def test_duplicate_method_in_a_setting_is_an_error()
def test_report_only_lists_missing_settings_and_writes_report_name(tmp_path)
```

- [ ] **Step 2: Run to verify failure.** Expected: FAIL, module missing.

- [ ] **Step 3: Implement `scripts/sensor/sd_verdict.py`** per Interfaces.

- [ ] **Step 4: Run to verify pass.** Expected: 8 passed.

- [ ] **Step 5: Commit** — `git add scripts/sensor/sd_verdict.py tests/test_sd_verdict.py` ; `git commit -m "feat(sensor): sd_verdict — pre-registered T1–T4 from a manifest of records, pairing on common files, main-line rule (Phase 3 Task 11)"`.

### Task 12: Stage 0 script, the normalised-vs-raw check, and the CPU smoke

**Files:**
- Create: `scripts/sensor/stage0_baselines.py`, `scripts/pilot/smoke_sensor.sh`
- Test: `tests/test_stage0.py`

**Interfaces:**
- Consumes: `HFDSensorSim` (with a `sensor_override` hook: `HFDSensorSim.__init__(..., R_override: np.ndarray | None = None)` added here — when given, `self.R = R_override` and `R_sha256` is of it; add this one-line hook to Task 5's class), `mmse.demosaic_then_mmse`, `sensor_model.load_sensor`, `uq_eval.gaussian_metrics / point_metrics`, `scipy.linalg.subspace_angles`.
- Produces: `STAGE0_DIR = os.path.join(SENSOR_DIR, 'stage0')`; `SNRS = (60.0, 50.0, 40.0, 30.0, 20.0)`; `def run_L(ds, loader, device) -> dict(per_cube=dict(L=..., L_demosaic=... if snapshot), files)`; `def raw_sigma(R_raw, files, snr_db) -> float` (σ for the raw variant from the gate cubes' own mean y² under R_raw, so both variants sit at the same SNR); `def shape_check(curve_norm: dict[snr -> rmse], curve_raw) -> dict(monotone_norm, monotone_raw, max_dev, passed)` (both curves non-increasing in SNR; `ratios = raw / norm`; `max_dev = max |ratio − median(ratios)|`; passed iff both monotone and max_dev ≤ 0.10); `def principal_angles(R_a, R_b) -> list[float]` (degrees, `subspace_angles`); CLI `--data_root --prior --sensor_config --device cuda:0 --limit_files 0`: for device ∈ (device1, device2, ph5) × mode ∈ (sequential, snapshot) × SNRS: L (and L_demosaic) with the normalised R on the 200 gate cubes → `stage0/L__<device>__<mode>__<snr>.json`; the same with `R_raw` for device1 only → `stage0/Lraw__device1__<mode>__<snr>.json`; `stage0/summary.json` with the two curves per mode, `shape_check` results, the principal angles device1–device2 and device1–ph5, and the 60 dB sequential device1 L RMSE as `sanity_reference_rmse_pct`; prints `NORMALISATION CHECK: PASS|FAIL (max_dev=...)`.
- `scripts/pilot/smoke_sensor.sh`: CPU, `set -euo pipefail`, results root `${RESIDUAL_WARMSTART_RESULTS:-...}/smoke_sensor_<timestamp>` (removed at the end only if the script created it): `make_config.py --out $T/config.json --limit_files 16 --n_e2_files 4` → four trainings via `main_2d_residual_fsdp.py` directly (`--mode <m> --ds 1 --sensor_config $T/config.json --base_channels 16 --batch_size 4 --num_workers 2 --device cpu --num_epochs 1 --save_every 1 --val_every 1 --limit_files 16 --save_dir $T/runs/<m>` with the mode's prediction / loss flags) → `sd_eval.py` on sequential 30 dB and snapshot 30 dB with `--methods L L_demosaic D V C B --K 3 --pigdm_steps 2 --limit_files 8 --subset ddpm --sensor_config $T/config.json --prior ...` → `sd_verdict.py --manifest $T/manifest.json --report_only` → `toy_guidance.py --out $T/toy.json --steps 5 --K 8 --B 1 --hw 4 --device cpu` → prints `SMOKE OK`.

- [ ] **Step 1: Write the failing tests**

```python
def test_shape_check_passes_same_shape_and_fails_on_a_kink()      # raw = 1.3 * norm -> passed; raw with one SNR scaled by 1.2 -> max_dev > 0.10 -> failed; a non-monotone curve -> failed
def test_principal_angles_identity_and_orthogonal()                # angles(R, R) all ~0; angles of disjoint-support filters > 0
def test_raw_sigma_matches_nominal_snr(small_R)
```

- [ ] **Step 2: Run to verify failure.** Expected: FAIL, module missing.

- [ ] **Step 3: Implement `scripts/sensor/stage0_baselines.py`, the `R_override` hook, and `scripts/pilot/smoke_sensor.sh`.**

- [ ] **Step 4: Run the tests, then the smoke.** Run: `... -m pytest tests/test_stage0.py tests/test_hfd_sensor_sim.py -q` → pass; `CUDA_VISIBLE_DEVICES= bash scripts/pilot/smoke_sensor.sh` → ends with `SMOKE OK` (≈ 10 min on CPU). Then the whole suite: `... -m pytest tests -q -x` → all pass (the Phase 1/2a suites included).

- [ ] **Step 5: Commit** — `git add scripts/sensor/stage0_baselines.py scripts/pilot/smoke_sensor.sh data_loader/hfd_sensor_sim.py tests/test_stage0.py` ; `git commit -m "feat(sensor): Stage 0 baselines with the normalised-vs-raw shape check and principal angles; CPU end-to-end smoke of the Phase 3 pipeline (Phase 3 Task 12)"`.

### Task 13: Stage 0 — config, toy, baselines (GPU / CPU, no training)

**Files:** results only (`results/residual_warmstart/sensor/{config.json, toy_guidance.json, stage0/}`); `docs/superpowers/plans/2026-11-03-sensor-degradation-diffusion-RESULTS.md` created with the Stage 0 section.

- [ ] **Step 1: Build the config.** Run from the worktree root: `OMP_NUM_THREADS=8 $PY scripts/sensor/make_config.py` (≈ 15 min CPU: 9 000 cubes for P_y, 200 for e2). Record P_y, training_sha, e2 range in the results note.
- [ ] **Step 2: Toy guidance test on GPU 0 (if free; else GPU 1).** `$PY scripts/sensor/toy_guidance.py --device cuda:0` → `steps_for_B`. If it prints `FAIL: use DPS`, amend the spec's §5.5 as the spec allows (DPS replaces ΠGDM; `--guidance dps` everywhere) and record it. Write the chosen step count into the results note; it is used by every B evaluation.
- [ ] **Step 3: Stage 0 baselines.** `$PY scripts/sensor/stage0_baselines.py --device cuda:0` (≈ 1 h). Read `stage0/summary.json`: if `NORMALISATION CHECK: FAIL`, stop and report to the user (spec §4.1) before any training; otherwise record the five-SNR L curves for the three devices and both modes, `sanity_reference_rmse_pct`, the principal angles, and the raw-vs-normalised max deviation.
- [ ] **Step 4: Write the results note's header and Stage 0 section; commit** (`docs(sensor): Phase 3 Stage 0 — config (P_y, sha), toy guidance step count, L curves for three devices and two modes, normalisation check`). Push.

### Task 14: Phase B — the four trainings

**Files:** run directories `results/residual_warmstart/runs/{uncond_b64,cond_fam_b64,regress_fam_b64,varhead_fam_b64}`; results note Phase B section.

- [ ] **Step 1: Launch the two diffusion lines** (both GPUs free; the user's other project may claim GPU 0 later — the launcher refuses a busy GPU): `RUN_NAME=uncond_b64 bash scripts/pilot/launch_run.sh uncond 0 --sensor_config $CFG --keep_epochs 25 50 90 100` and `RUN_NAME=cond_fam_b64 bash scripts/pilot/launch_run.sh cond_fam 1 --sensor_config $CFG --keep_epochs 25 50 90 100` (v-prediction, L1, 100 epochs, ≈ 8–9 h each; per-epoch noise draws; monitoring every 5 epochs). Arm one background waiter per run on `DONE` / process exit (as for B.8).
- [ ] **Step 2: When a GPU frees, launch the baselines**: `RUN_NAME=regress_fam_b64 bash scripts/pilot/launch_run.sh regress_fam <gpu> --sensor_config $CFG --prediction_type direct --loss_type l1 --val_every 1 --save_every 1 --keep_last 100` (100 epochs, ≈ 7.5 h) and `RUN_NAME=varhead_fam_b64 bash scripts/pilot/launch_run.sh varhead_fam <gpu> --sensor_config $CFG --prediction_type direct --loss_type nll --val_every 1 --save_every 1 --keep_last 100 --num_epochs 40` (≈ 4 h).
- [ ] **Step 3: Selection.** D and V: `$PY scripts/uq/select_epoch.py <run_dir>` → the selected epoch (ties → earlier); C and B: epoch 100 (monitoring loss must be non-increasing over the last 20 epochs as in Phase 2a, else the lowest-loss kept checkpoint and a note). Prune D and V to the selected + 25 / 50 / 100 (V: selected + 40) **after** the user's say-so, as before.
- [ ] **Step 4: R2 check (spec §10).** Only if D's monitoring loss looks poor: train the 25-epoch 30 dB-only regressor (`--sensor_config` with a one-off config whose family is {30 dB, sequential}; ≈ 2 GPU-h) and apply the 10 % rule; otherwise record "oracle run not needed".
- [ ] **Step 5: Results note Phase B section (runs, epochs, times, selected checkpoints, monitoring curves); commit; push.**

### Task 15: E1 and the verdict on T1 / T2

- [ ] **Step 1: Evaluate D, V, C, L on the 200 gate cubes** for device1 sequential at each SNR in (60, 50, 40, 30, 20): `$PY scripts/sensor/sd_eval.py --name e1_dev1_seq_<snr>_main --sensor_device device1 --mode sequential --snr_db <snr> --subset gate --methods L D V C --regress <D ckpt> --varhead <V ckpt> --cond <C ckpt> --K 32 --device cuda:<g>` (≈ 0.5 GPU-h each).
- [ ] **Step 2: Evaluate B on the 50-cube subset** at the same SNRs: `--name e1_dev1_seq_<snr>_B --subset ddpm --methods L B --uncond <B ckpt> --K 32 --pigdm_steps <steps_for_B> --guidance pigdm` (≈ 0.63 GPU-h each at 50 steps; log the cumulative B GPU time against the 12 h cap).
- [ ] **Step 3: Manifest and verdict.** Write `results/residual_warmstart/sensor/manifest__e1.json` listing both records per setting; `$PY scripts/sensor/sd_verdict.py --manifest ... --report_only` (E2/E3 missing → report) and read T1 and T2 for both lines. Record the five-SNR table (L, D, V, C mean-of-32, B mean-of-32: RMSE, CRPS, cov90, AUSE) in the results note; commit; push.

### Task 16: E2 — zero-shot transfer to Device 2 and PH5, verdict on T3

- [ ] **Step 1:** `sd_eval.py` for device2 and ph5, sequential, SNR ∈ (40, 30, 20): the main record (L D V C, 200 cubes) and the B record (L B, 50 cubes) each — twelve records (≈ 7 GPU-h total, B ≈ 3.8 h).
- [ ] **Step 2:** Add the settings to a `manifest__e12.json`, run the verdict report-only, read T3 for both lines (and PH5's reported ratios); results note E2 section; commit; push.

### Task 17: E3 — snapshot mode, verdict on T4 and the full verdict

- [ ] **Step 1:** `sd_eval.py` for device1 snapshot, SNR ∈ (60, 40, 30, 20): main record with `--methods L L_demosaic D V C` and the B record (L L_demosaic B, 50 cubes) — eight records (B ≈ 2.5 h; apply the cap's drop order if the cumulative B time would exceed 12 h, and record every drop).
- [ ] **Step 2:** `manifest__e123.json` with E1 + E2 + E3; `$PY scripts/sensor/sd_verdict.py --manifest results/residual_warmstart/sensor/manifest__e123.json` (the one full verdict: T1–T4 per line, counts, `main_line`); results note E3 and verdict sections with the printed verdict verbatim; commit; push.

### Task 18 (only if GPU time remains): E4 — Device 2 snapshot at 30 dB

- [ ] **Step 1:** two records (main, B) for device2 snapshot 30 dB; add to a `manifest__e1234.json`; report-only verdict; results note; commit; push.

### Task 19: Write-up

- [ ] **Step 1: Results note complete** — predictions of spec §6.3 against the outcomes (T1–T4 per line, main line), the four paper figures' tables (error vs SNR; calibration at 30 / 20 dB; transfer; snapshot), the GPU hours spent against the budget, deviations (step count, drops, DPS if used), deferred items.
- [ ] **Step 2: Spec as-built block** appended to `docs/superpowers/specs/2026-11-03-sensor-degradation-diffusion-design.md`: decisions D1–D6 of this plan, the toy outcome, any amendment (DPS, step count, drops).
- [ ] **Step 3: Results page** in the GDN-report style, built from the records like `docs/analysis/phase2a_page/` (data extractor → compact → build; source `docs/phase3_sensor_results.src.html`, built `docs/phase3_sensor_results.html`), published as an artifact; the page is checked by at least two adversarial page checkers before publication, as in Phase 2a.
- [ ] **Step 4: Memory update** (`residual-warmstart-plan.md` / a new `phase3-sensor-degradation.md`, `MEMORY.md` index line); final commit; push; tell the user the branch head and the page URL.
