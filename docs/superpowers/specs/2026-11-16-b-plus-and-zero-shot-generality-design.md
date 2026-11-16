# Phase 3b — B⁺ (a calibrated guided prior) and four zero-shot generality tests: design and pre-registration

Date 2026-11-16. Follows `2026-11-03-sensor-degradation-diffusion-design.md` (Phase 3) and its results note. User's
instruction (2026-11-16): fix B's calibration, then run items 1–4 below. Everything here is fixed before any record on
the test cubes exists; the Phase 3 verdict (`verdict__e123.json`, main line B) is not reopened — every number below is
**report-only** and is presented as an amendment.

## Part A — B⁺

### A.1 Diagnosis (calibration cubes only)

B's 32 draws nearly coincide (spread ratio 0.09–0.12 at 30 / 20 dB, cov90 0.29–0.37). The cause is the prior network,
not the guidance. On 16 calibration cubes the prior's x̂₀ error is **1.70 % at t = 0, 1.71 % at t = 5, 1.72 % at t = 19**
while the input noise there is 0.50 / 1.50 / 3.80 %: the network has a reconstruction floor and is worse than its own
input for t ≤ 5. ΠGDM returns the network's x̂₀ at its last step (t = 19), i.e. a mean with that floor, so every draw
converges to the same smoothed estimate and the fine-scale posterior spread is removed. A finer final grid cannot help
(the floor is the same at every small t).

### A.2 Method

The last step returns a draw instead of a mean. Each guided draw s_k is the centre of a Gaussian component
N(s_k, Σ_e) per pixel, Σ_e [64 × 64] being the measured error covariance of the sampler's output, and the component is
conditioned exactly on the pixel's measurement y = H x + b + n (`model/posterior_draw.py`, Matheron's rule):

    e ~ N(0, Σ_e), n' ~ N(0, N):  x_k = s_k + e + G (y − b − H (s_k + e) − n'),  G = Σ_e Hᵀ (H Σ_e Hᵀ + N)⁻¹

So p(x | y) ≈ (1/K) Σ_k N(μ_k, Σ_post), μ_k = s_k + G (y − b − H s_k). The same code serves sequential mode, snapshot
mode, any per-element mask and any per-element noise variance. B⁺ uses the same K guided draws as B.

### A.3 The kernel (fixed here)

- **Calibration cubes:** the gate cubes that are not in B's 50-cube test subset and share no source image with it, one
  cube per source image (68 cubes; `scripts/sensor/fit_b_kernel.py::calibration_files`).
- **Σ_e** = the covariance of the per-pixel spectral residual (truth − mean of K = 8 guided draws), pooled over pixels
  and cubes, at **one reference setting: Device 1, sequential, 20 dB** (the least informative E1 setting, so the kernel
  holds what prior and sampler cannot resolve and the conditioning supplies what the measurement can).
- **One kernel for every setting**, device and mode; **no scale factor** (τ = 1), no per-setting refit. The file is
  written once (`sensor/b_kernel.npz`).

Evidence on the 16 calibration cubes (2-fold, kernel from the other fold; K = 20): 30 dB cov50 / cov90 0.53 / 0.90
(B: 0.16 / 0.35), CRPS −23 %, mean RMSE −9 %; 20 dB with the 30 dB kernel 0.50 / 0.87; 30 dB with the 20 dB kernel
0.56 / 0.92; 60 dB with the 20 dB kernel 0.51 / 0.90 and mean RMSE 2.10 → 0.83 % (MMSE 1.17 %); snapshot 30 dB with
the sequential 20 dB kernel 0.45 / 0.83 (B: 0.29 / 0.67), mean unchanged. The conditioning uses y a second time (the
guided draw already saw it); the calibration cubes show no over-confidence from that in sequential mode.

### A.4 Evaluation

All 13 B settings are re-run on the 50 test cubes with K = 32 and the same seeds; each record holds B and Bp scored on
the same draws (`sd_eval.py --methods B Bp --b_kernel …`), and B's raw draws are saved. **Integrity check:** B's per-cube
numbers must reproduce the existing B records. The clauses T1–T4 are recomputed for B⁺ with the unchanged
`sd_verdict.py` (manifest `rename: {Bp: B}`, report-only).

### A.5 Predictions (before the test records)

- T2 at 30 and 20 dB: **pass** (cov90 in [0.85, 0.95]; CRPS below L and V by 2 SE).
- T1: B⁺ ≤ 1.05 × D at 30 / 20 dB (as B) and now also at 60 dB; 50 and 40 dB are the open cells. If all five hold, B⁺
  passes T1.
- T3: pass (B passes; the conditioning uses the device's own R).
- T4: **fail** (in snapshot mode one row per pixel leaves the mean at the demosaic level).
- Snapshot coverage stays slightly low (cov90 ≈ 0.83): the kernel was measured in sequential mode and does not hold
  the demosaicing error. Reported as is.

## Part B — four zero-shot generality tests

**Common protocol.** Nothing is retrained. The new forward model (response matrix, measured rows, noise model) is known
at test time. L is the noise-aware MMSE with that forward model; D, V, C are the Phase 3 family networks fed the MMSE
statistics (x̂, s) of that forward model; B and B⁺ use the Phase 3 prior, guidance and kernel with that forward model.
L / D / V / C on the 200 gate cubes; B / B⁺ on the 50-cube subset with **K = 20** (≥ 19 for exact 90 % coverage),
ΠGDM 50 steps; every comparison involving B is paired on the common cubes with the clustered SE. Detector noise is
referred to the same global P_y as Phase 3. Metrics: RMSE of the point estimate, CRPS, cov90. No clause; each item
states its prediction.

### B.1 Sensor variation (sequential, 30 dB) — 8 settings

- **Blend** R(α) = (1 − α) R₁ + α R₂, α ∈ {0.25, 0.5, 0.75} (α = 0, 1 are E1 / E2).
- **Wavelength shift**: every state's response moved by Δ ∈ {1, 2, 4} bands toward longer wavelengths (linear
  interpolation along the band axis, edge value held).
- **Per-state gain**: R[:, k] ← R[:, k] · exp(δ g_k), g_k ~ N(0, 1) (seed 0), δ ∈ {0.2, 0.5}.

Prediction: B's and B⁺'s RMSE stays within 10 % of its Device 1 value at every setting; D's advantage over L (0.59 × on
Device 1, 0.80 × on Device 2) shrinks with the size of the change; B⁺ is the best point estimate at every setting.

### B.2 Fewer voltage states (sequential) — 6 settings

Evenly spaced subsets of Device 1's 30 states, S ∈ {15, 10, 5} (indices round(linspace(0, 29, S))), at 40 and 20 dB.
Prediction: every method degrades as S falls; B⁺ / L falls (the prior matters more with fewer states); D, never trained
on a subset, loses its advantage over L by S = 5.

### B.3 One prior, other tasks — 5 settings

- **Dead pixels** (sequential, 30 dB): a random fraction p ∈ {0.5, 0.9} of the pixels has no measurement (seeded per
  cube).
- **Spatial ×d** (sequential): only pixels with i mod d = j mod d = 0 are measured (Phase 1's stride), d ∈ {4, 8} at the
  60 dB floor and d = 4 at 30 dB.

Baselines: L exact (an unmeasured pixel gets the prior mean), L_interp (the measured readings interpolated over the
image, then the sequential MMSE with the interpolation-error variance, estimated on the calibration cubes, added to
σ²), D / V / C with the exact statistics. The Phase 1 regressors trained for d = 4 and 8 (noise-free) are quoted from
their own records as the trained-for-the-task reference. Together with E1 (denoising), E2 (another sensor) and E3 / E4
(demosaicing) this fills a task × method table. Prediction: B⁺ beats L_interp on dead pixels at both fractions and at
d = 4; the task-specific Phase 1 regressor stays ahead at d = 4 noise-free.

### B.4 Non-Gaussian measurement noise (sequential, Device 1) — 5 settings

- **Shot noise**: y ~ Poisson(γ Rᵀx) / γ, γ chosen so the mean variance equals σ²(SNR) at 30 and 20 dB.
- **Saturation**: 30 dB Gaussian noise, then readings clipped at the q-quantile of the clean readings, q ∈ {0.9, 0.7}.
- **Quantisation**: 40 dB Gaussian noise, then a 4-bit ADC over [0, max clean reading].

"Aware" methods use the matching likelihood: per-element variance max(y, 0)/γ for shot noise (with a floor), saturated
readings dropped, σ² + Δ²/12 for the ADC. L, D (aware statistics) and B / B⁺ (aware guidance and conditioning) are
aware; L_naive and D_naive assume white Gaussian noise at the nominal σ. Prediction: the aware methods beat their naive
versions; B⁺ beats D with aware statistics under saturation (the statistics leave the family's training distribution).

### Budget

A.4: 13 × 1.7 = 22 GPU-h. B: 24 settings × ≈ 1.05 GPU-h = 25 GPU-h, plus minutes for L / D / V / C. Two RTX A4500;
nothing is launched while another project uses a GPU.

### What would count against the claim

B⁺ failing T2's coverage band on the test cubes; B⁺'s mean getting worse than B's anywhere in sequential mode; in
Part B, D with the new forward model's statistics matching or beating B⁺ across B.1–B.2 (the zero-shot advantage of the
prior would then be small). All outcomes are reported.
