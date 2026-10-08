# Residual Warm-Start Diffusion for Low-Resolution Sensors — Design Spec

Status: design approved in conversation; written spec awaiting review
Date: 2026-10-08

## 1. Motivation

The warm-start investigation (`docs/warm_start_sampling.html`, published at
https://claude.ai/artifact/ShLD86QyzLt41o6Lk4tGW2) found:

- Starting the reverse process at an intermediate step t0 from a sensor-based estimate is sound, but on the
  1-D HASCID model plain 5-step DDIM from noise was equally good, so no speed gain was shown.
- The loader's sensor values are noise-free. At full sensor resolution the closed-form per-pixel linear
  inverse x̂ = μ + K(y − Rᵀμ) reconstructs held-out HFD cubes at about 0.3 % RMSE and beats every diffusion
  output measured. There is nothing left for diffusion to win at d = 1.
- The 2-D `l2_loss/checkpoint_epoch_101.pth` model ignores which sensor values it is given.

With a spatially downsampled sensor (`--ds d`, d > 1) most pixels are unmeasured, the per-pixel inverse cannot
fill them, and there is real headroom. This project builds a diffusion model that starts from the cheap
physics estimate and learns only what it misses, so that it (a) beats that estimate and (b) needs only a few
sampling steps.

## 2. Agreed Decisions

| topic | decision |
|---|---|
| Goal | Quality and speed: beat the linear-inverse + upsampling baseline at d ≥ 2 with ≤ 10 network evaluations |
| Sensor noise | None. Sensor values stay exactly `y = x @ R` as in the loader |
| Target rates | d = 2, 4, 8; d = 1 only as a sanity control. d = 4 first |
| Data | The current `HFD_data` loader unchanged: MatFlower60/Train, sorted, first 10 000 files, in-order 90/10 split (train P001–P022, validation P022 122 / P023 760 / P024 118 files), R_Device1, 64 interpolated bands, strided sensor downsampling |
| Approach | Method B (residual diffusion around the estimate) as the main method; ablation A (standard model + training-free warm start); check C (sensor-SR + linear inverse) only if sensor-SR checkpoints are available |
| Compute | This server (2 × RTX A4500, 20 GB each). Large models use FSDP across both GPUs (`torchrun --standalone --nproc_per_node=2`). A cluster may be added later, named by the user |

## 3. Baseline to Beat

`baseline(Y_d)` = per-pixel linear inverse of the measured low-resolution pixels, clipped to [−1, 1], mapped
31 → 64 bands with the loader's interpolation, then bilinearly upsampled to 64 × 64.

Measured on the loader's 1 000 validation files with (μ, C) fitted on its training files, using default
`F.interpolate(..., mode='bilinear', align_corners=False)` upsampling (not grid-aligned):

| d | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| RMSE (% of cube range) | 0.31 | 1.64 | 3.57 | 6.14 |
| SAM (degrees) | 0.45 | 1.54 | 2.67 | 4.22 |

Phase 0 re-measures these with grid-aligned upsampling (§4.1). The re-measured values are the official
baseline for every gate.

## 4. Method B: Residual Diffusion Around the Estimate

### 4.1 Estimate x̂₀(Y_d)

Computed inside the dataset for every sample, from the sensor only:

1. Linear inverse at each measured low-resolution pixel: x̂ = μ + K(y − Rᵀμ). (μ, C) are fitted once on
   pixels of the loader's training files. K = C R (Rᵀ C R + s² I)⁻¹ is computed with a whitened SVD solve
   (C = L Lᵀ, SVD of Rᵀ L). The regulariser s is chosen on held-out training files only and stored with the
   prior.
2. Clip to [−1, 1]; map 31 → 64 bands with the loader's linear interpolation matrix.
3. Upsample to 64 × 64 with grid-aligned bilinear interpolation: low-resolution sample k sits at
   high-resolution pixel k·d (matching the loader's `[::d]` sampling); pixels beyond the last sample use edge
   extension. At d = 1 this step is the identity.

### 4.2 Residual target

r = (x − x̂₀) / σ_d, where σ_d is the root-mean-square of (x − x̂₀) over all elements of the training files at
rate d (expected roughly 0.03 / 0.07 / 0.12 on the [−1, 1] scale for d = 2 / 4 / 8; measured in Phase 0 and
stored in the prior file). Scaling makes r unit-size, which the diffusion noise schedule assumes.

### 4.3 Network and conditioning

- `U2NetHyperspectral` with a new `extra_in_channels` argument: the input is the concatenation
  [r_t, x̂₀] (64 + 64 channels).
- The condition is the low-resolution sensor Y_d through the existing cross-attention context path.
- Unchanged from `main_2d_fsdp.py`: FiLM time conditioning, v-prediction, min-SNR γ = 5 weighting, L1 loss,
  linear β schedule (1e-4 to 0.02, T = 1000). Plain dense conv blocks (`use_channel_3d_conv` off).

### 4.4 Sampling and output

Output x̂ = x̂₀ + σ_d · r̂₀, clipped to [−1, 1]. Samplers evaluated:

- full DDPM, 1000 steps;
- DDIM (η = 0) with 1, 2, 5, 10, 20 steps from t = 999;
- warm starts from r̂₀ = 0 (the estimate itself) at t0 ∈ {50, 100, 200, 400}, followed by DDIM with
  min(10, t0) steps and by DDPM.

If the network predicts r̂₀ = 0 everywhere, the output equals the baseline exactly.

## 5. Ablation A and Check C

- **A (standard model):** same network size and training budget; target x, input x_t only (no x̂₀ channels),
  condition Y_d. Sampled from noise with the same sampler grid, and with the training-free warm start
  x_t0 = √ᾱ_t0 · x̂₀ + √(1 − ᾱ_t0) · ε at the same t0 values.
- **C (two-stage):** only if `U2NetSensorSR` checkpoints for the same d are available: super-resolve the
  sensor image, then apply the linear inverse at full resolution. Evaluated with the same harness.

## 6. Components

| unit | file | purpose | change to existing code |
|---|---|---|---|
| Linear prior | `scripts/fit_linear_prior.py` (new) | Fit (μ, C) on training pixels; choose s; compute K and σ_d for d = 1, 2, 4, 8; write `results/residual_warmstart/linear_prior_R1.npz` | none |
| Residual dataset | `data_loader/hfd_residual.py` (new) | Subclass of `HFD_data` (same files, split, normalisation, strided sensor); returns (x, Y_d, x̂₀) | none |
| Network | `model/u2net_hyperspectral.py` | `extra_in_channels=0` argument widening `input_proj` | additive; existing checkpoints still load |
| Wrapper | `model/residual_wrapper.py` (new) | Concatenates x̂₀ to the input and passes Y_d as context, so `DiffusionTrainer` keeps calling `model(x_t, cond, t)` | none |
| Sampler | `model/diffusion_trainer.py` | `sample()` gains `x_init` and `t_start` for warm starts | additive; defaults reproduce current behaviour |
| Training entry | `main_2d_residual_fsdp.py` (new) | Based on `main_2d_fsdp.py`; `--mode residual` (B) or `--mode standard` (A); `--ds`; FSDP on 2 GPUs or a single GPU; atomic checkpoint saves (temporary file + `os.replace`), keeps the last two; resumable | none |
| Evaluation harness | `scripts/eval_warmstart.py` (new) | Baseline and any checkpoint over the fixed validation subset with the full sampler grid; metrics, NFE, wall-clock, y-swap check; JSON output | none |
| Tests | `tests/` (new, pytest) | See §9 | none |

Data flow — training: loader (x, Y_d) → x̂₀ = lift(K, Y_d) → r = (x − x̂₀)/σ_d → trainer noises r →
wrapper feeds [r_t, x̂₀] and context Y_d to the U-Net. Evaluation: sampler → r̂₀ → x̂ = x̂₀ + σ_d r̂₀ → metrics
against x.

## 7. Evaluation Protocol

- **Test data:** the loader's 1 000 validation files. Baselines on all 1 000. Diffusion samplers on a fixed
  stratified subset of 200 cubes (24 P022 / 152 P023 / 24 P024, fixed seed, list stored in the repo) with 2
  noise seeds; full DDPM-1000 on 50 of those 200.
- **Metrics:** RMSE in % of each cube's range on the [0, 1] scale (primary); PSNR (peak 1); SAM in degrees;
  NFE; wall-clock per cube. Every result is paired with the baseline on the same cubes and reported as mean ±
  standard error, overall and split into P022 (seen class) and P023–P024 (unseen classes).
- **y-swap check:** for every checkpoint, the one-step x̂₀ error at t = 100 and t = 300 with the true
  condition, a condition taken from another cube, and a zero condition. For B it is run twice: swapping
  (x̂₀, Y_d) together and swapping Y_d alone. A run passes if, at both t, the one-step error with the true
  condition is at least 10 % lower than with the swapped full condition.
- **No peeking:** each run is evaluated at its final checkpoint after a fixed budget, never at a best
  validation epoch. The full sampler grid is reported; the gate uses the best setting with ≤ 10 NFE, marked as
  selected from the grid.

## 8. Phases, Budgets and Gates

| phase | content | hardware | gate |
|---|---|---|---|
| 0. Foundation | Prior fit, residual dataset, network/wrapper/sampler changes, harness, tests; re-measure baselines with grid-aligned upsampling; time one training step of each model size | this server | Tests pass; baselines reproduced and recorded; timings recorded |
| 1a. Pilot | B and A at d = 4, `base_channels` 64, one GPU each in parallel; 100 epochs or 24 h, whichever first; evaluate at epochs 25, 50, 100 | GPU 0 + GPU 1 | At the best ≤ 10-NFE setting, B's RMSE is lower than both the baseline and A's by more than 2 paired standard errors; otherwise stop and diagnose |
| 1b. Full B | B at d = 4, `base_channels` 256, FSDP, bf16, 200 epochs | both GPUs | RMSE ≥ 15 % below the Phase 0 baseline at d = 4 with ≤ 10 NFE, and the y-swap check passes |
| 2. Full A (+ C) | A at d = 4, same size and budget as 1b; C if available | both GPUs | none; explains the 1b result |
| 3. Sweep | B at d = 2 and d = 8, plus a d = 1 control | this server or a cluster named by the user | none |
| 4. Write-up | Results page in the style of the earlier reports | this server | none |

Shared training settings unless a phase says otherwise: AdamW, cosine learning rate, gradient clipping 1.0,
checkpoints every 10 epochs. Each phase gets its own implementation plan and the user's approval before it
starts. The user decides at three points: after Phase 0 (baselines and timings), after 1a (go to full size),
after 1b (go to the sweep, and where).

## 9. Validation & Error Handling

Tests (pytest, CPU, small synthetic or few-file inputs):

- the linear inverse at d = 1 reproduces the recorded baseline on a fixed set of validation files within
  0.01 RMSE points;
- grid-aligned upsampling places low-resolution sample k at pixel k·d exactly;
- r has unit RMS (within 5 %) on a sample of training files for each d;
- wrapper output shapes for `--mode residual` and `--mode standard`;
- warm-start sampler: first model call at t = t_start, no noise added at t = 0, NFE counted correctly, and
  defaults identical to the current `sample()`;
- the y-swap utility reports "fails" for a stub model that ignores its condition.

Runtime safeguards: check GPU memory with `nvidia-smi` before launch and never start on a busy GPU; atomic
checkpoint writes; resume from the latest complete checkpoint.

## 10. Out of Scope

- Adding sensor noise; changing the data split or the 10 000-file cap.
- A ResShift-style shifting Markov chain (the follow-up if few-step sampling lags many-step sampling).
- Distillation or consistency-model training.
- Cluster runs until the user names a cluster.
- Changes to the existing `main_2d.py`, `main_2d_fsdp.py` and sensor-SR pipelines beyond the additive changes
  in §6.

## 11. Risks

| risk | signal | response |
|---|---|---|
| B copies x̂₀ (output ≈ baseline) | paired gain ≈ 0 at every sampler setting | check σ_d scaling and that the loss falls below its starting value; then try direct x₀-prediction of the residual (one-step, InDI-style) |
| Network ignores the sensor | y-swap: full-condition swap changes nothing | the run is invalid; for B, a Y_d-only swap having no effect is acceptable because x̂₀ carries y |
| Gain only on the seen class | P022 improves, P023–P024 do not | report as a limit of the 22-class loader |
| Few steps lose to many steps | DDIM-10 clearly worse than DDPM-1000 | report the speed–quality curve; consider the shifting-chain follow-up |
| Estimate depends on the solver | baseline changes with s or solver | whitened SVD solve, s fixed on training files and stored, pinned by a test |
| GPUs busy | pre-launch check | wait; runs are resumable |
| Corrupt checkpoint | — | atomic writes, keep the last two |

## 12. File Manifest

New: `scripts/fit_linear_prior.py`, `data_loader/hfd_residual.py`, `model/residual_wrapper.py`,
`main_2d_residual_fsdp.py`, `scripts/eval_warmstart.py`, `tests/` (test files and the stored validation
subset list).
Modified (additive only): `model/u2net_hyperspectral.py`, `model/diffusion_trainer.py`.
Outputs (git-ignored, under `results/`): `results/residual_warmstart/linear_prior_R1.npz`,
`results/residual_warmstart/<mode>/d<d>/...` (checkpoints, logs, evaluation JSON).
