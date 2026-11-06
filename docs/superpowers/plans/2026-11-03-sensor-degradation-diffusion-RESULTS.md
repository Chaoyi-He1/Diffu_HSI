# Phase 3 — Diffusion Reconstruction From the Poor EC Sensor Under Noise, Across Devices and in Snapshot Mode: Results

Spec: `docs/superpowers/specs/2026-11-03-sensor-degradation-diffusion-design.md` (approved 2026-11-04, ac4e4fd). Plan: `docs/superpowers/plans/2026-11-04-sensor-degradation-diffusion.md` (2c49684, amended during execution: Task 3 test clip bc63d8c, Task 6 legacy assertions / FiLM 7a4c359, ΠGDM coefficient 6d2d431, crc32 file seeds 15ce085). Build phase (Tasks 1–12 + the carried-fix batch 11b) complete 2026-11-04 at 3bc4a1d: 546 tests green, CPU smoke `SMOKE OK`. Execution ledger: `.superpowers/sdd/2026-11-04-sensor-degradation-diffusion/progress.md` (archived under `results/residual_warmstart/sensor/sdd_archive/` at the end).

Conventions: every number is on the 200 gate cubes of `tests/data/val_subset_200.json` unless stated (B on its 50-cube `ddpm1000_subset`, paired on the common files); errors in % of the [0, 1] range; CRPS in %; "±" is the iid SE over cubes with the source-image-clustered SE in parentheses where a paired difference is reported; K = 32 samples per cube; SNR in dB against the global P_y of device1 in sequential mode; the 60 dB floor is the noise-free end (no σ = 0 anywhere).

Design decisions taken during the build (plan D1–D6 and the rulings in the ledger): dict batches for the sensor modes; the per-element posterior std replaces σ_d as the residual scale (floor 0.02 trainer scale); no persistent DataLoader workers for datasets with `set_epoch`; `uncond` shares the dataset / meta / launcher path; the toy that licenses the calibration clauses is isotropic N(0, I); L's Gaussian posterior uses the unfloored std. Rulings that changed the plan text: ΠGDM's correction is x̂₀ + (1 − ᾱ_t)/√ᾱ_t · g (the score correction in x₀ coordinates; the draft's √ᾱ_t coefficient diverged on the isotropic toy); the per-cube noise seed is crc32 of the cube's path relative to the data root, not its list position, so 50-cube and 200-cube records pair by file; `uncond`'s time path is FiLM-only and t-blind at initialisation (zero-initialised FiLM layers).

## Stage 0 (Task 13)

### Config (2026-11-04 17:29, `scripts/sensor/make_config.py`, CPU)
`results/residual_warmstart/sensor/config.json`: **P_y = 32.948137** (mean of y² over all pixels and channels of the 9 000 training cubes, device1 sequential, physical scale); training sha **d62dea69278c6d8f3f58b6e1d7dbf04721bec360ad56581caf7b1839bad05126**; s_floor01 0.01, floor 60 dB, family U[20, 50] dB with p_floor 0.1 / p_snapshot 0.5, tile 5 × 6, seed 0; device shas 405b355b… (device1), a60c3c39… (device2), 96def0ad… (ph5); demosaic error table e2 (30 channels, 200 cubes): min 0.0062, median 0.134, max 0.568 (y² units; at 30 dB σ² = P_y / 1000 = 0.033, so the demosaic-then-MMSE baseline is interpolation-limited below ≈ 25 dB).

### Toy guidance test (2026-11-04, GPU 0, `scripts/sensor/toy_guidance.py`, K = 250 × B = 4 = 1 000 draws per cell, 8 × 8, 30 dB, real device-1 R; record `sensor/toy_guidance.json`)
Gating cells (isotropic N(0, I) prior, ΠGDM):

| mode | steps | rmse_excess | var_ratio | cov50 | cov90 | crps_ratio |
|---|---|---|---|---|---|---|
| sequential | 50 | 0.0026 | 0.926 | 0.490 | 0.888 | 1.0009 |
| sequential | 100 | 0.0027 | 0.962 | 0.498 | 0.895 | 1.0007 |
| snapshot | 50 | 0.0025 | 0.927 | 0.486 | 0.888 | 1.0008 |
| snapshot | 100 | 0.0026 | 0.963 | 0.495 | 0.895 | 1.0006 |

Every gating threshold (rmse_excess ≤ 0.02, |var_ratio − 1| ≤ 0.10, cov90 ∈ [0.85, 0.95], cov50 ∈ [0.45, 0.55]) is met at 50 steps in both modes → **`steps_for_B = 50`**, as pre-registered (spec §5.5). The 50-step DDIM discretisation under-disperses by ≈ 7 % in variance (0.93), 4 % at 100 steps; the posterior mean is exact to 0.3 %.

Report-only cells: with the HFD prior covariance (A C Aᵀ lifted; ΠGDM's isotropic r_t² is then only an approximation) sequential ΠGDM has rmse_excess 0.043 / 0.045 and var_ratio 0.82 / 0.92 (cov90 0.843 / 0.866) at 50 / 100 steps — the isotropic approximation costs ≈ 4 % in RMSE and visibly under-disperses on a correlated prior, less so at 100 steps; snapshot is unaffected (0.007, var 0.92 / 0.96). DPS on the HFD prior in sequential mode is far worse (rmse_excess 0.33 / 0.13, var_ratio 1.65 / 1.19); DPS on the isotropic prior is slightly worse than ΠGDM (0.029 / 0.017). ΠGDM stays the method. Ruling: B runs at 50 steps everywhere as pre-registered; if B's calibration clauses fail in E1, the HFD toy is the first explanation to test, and a report-only 100-step run of the two T2 settings is the natural addendum if GPU time remains after E1–E3.

### Baselines (2026-11-04 17:35–18:45, `scripts/sensor/stage0_baselines.py --device cuda:1 --num_workers 8`; 41 records under `sensor/stage0/`, `summary.json`)

**Normalisation check (spec §4.1, user Q2): PASS** — sequential max_dev 0.083 (raw/normalised RMSE ratios 0.88 / 0.89 / 0.96 / 1.03 / 1.03 at 60 / 50 / 40 / 30 / 20 dB, both curves monotone), snapshot max_dev 0.003 (ratios ≈ 1.036 at every SNR). The raw-response curve has the same shape as the normalised one within the 10 % rule, so the normalised R stays the paper's sensor and the raw curve goes to the appendix (P_y_raw = 9.624 for the raw variant's σ against P_y = 32.948).

L = the noise-aware MMSE (closed form, no training), RMSE % on the 200 gate cubes; snapshot mode shows the exact per-pixel posterior (one measured channel per pixel → essentially the prior mean) and the demosaic-then-MMSE baseline that T4 is scored against:

| sensor | mode | 60 dB (floor) | 50 dB | 40 dB | 30 dB | 20 dB |
|---|---|---|---|---|---|---|
| device1 | sequential | **1.105** | 1.556 | 2.622 | **4.143** | **6.425** |
| device1 | snapshot, exact | 12.107 | 12.108 | 12.118 | 12.218 | 12.877 |
| device1 | snapshot, demosaic-then-MMSE | 5.574 | 5.578 | 5.613 | **5.835** | 6.867 |
| device2 | sequential | 1.430 | 2.095 | 3.185 | **5.122** | 7.682 |
| device2 | snapshot, demosaic-then-MMSE | 7.890 | 7.882 | 7.819 | 7.726 | 8.248 |
| ph5 | sequential | 6.070 | 7.304 | 8.963 | 10.413 | 11.666 |
| ph5 | snapshot, demosaic-then-MMSE | 11.517 | 11.512 | 11.486 | 11.474 | 11.924 |

Readings. (i) **The sensor's poorness is now visible:** at the 60 dB floor the MMSE error is 1.105 %, four times the noise-free legacy figure (0.266 % with the s = 3e-7 regulariser), because 19 of R's 30 singular values are below 1 % of the largest and amplify even 60 dB noise; the error then grows by ≈ 1.5 × per 10 dB. `sanity_reference_rmse_pct` = 1.105 (device1, sequential, 60 dB) is the reference every learned method must not exceed by more than 2 clustered SE. (ii) L's Gaussian posterior is close to calibrated and slightly over-dispersed at every SNR (device1 sequential: cov50 0.559 / 0.549 / 0.541 and cov90 0.926 / 0.922 / 0.922 at 60 / 30 / 20 dB; CRPS 0.590 / 2.260 / 3.560; AUSE 0.330 / 0.363 / 0.409) — a strong classical probabilistic baseline for T2. (iii) Snapshot mode: a single measured channel per pixel carries almost no spectral information on its own (exact posterior ≈ 12 % ≈ the prior's spread); demosaicing the 30 channels across the 5 × 6 tile and running the sequential MMSE on the result gives 5.6–6.9 % (device1), interpolation-limited (flat from 60 to 40 dB); **T4's reference at 30 dB is 5.835 %, so the 15 % target is 4.960 %.** (iv) Device 2 is a harder sensor than Device 1 at every SNR (1.43 → 7.68 %); PH5 under the legacy axis is far worse (6.07 % even at the floor) and its demosaic baseline barely beats the prior — PH5 is report-only, as the user chose. (v) Principal angles (R5): the device1 and device2 column spaces (both 30-dimensional in the 31-band space) share a 29-dimensional subspace by construction — only ONE principal angle is non-zero, 68.1° — so the transfer test is about how the two sensors weight directions (their singular-value structure), not about different subspaces; device1–ph5: ph5's legacy-axis response has rank 16 (its 30 states collapse to 16 independent directions), all 16 angles ≤ 2.8°, i.e. ph5 measures a 16-dimensional subspace of what device1 measures, with far worse conditioning.

Phase B may start: Stage 0 took ≈ 1 h; no number of a learned method has been looked at.

## Phase B — trainings (Task 14)

- `uncond_b64` launched 2026-11-04 18:50 on GPU 0 (pid 2370859; `RUN_NAME=uncond_b64 launch_run.sh uncond 0 --sensor_config $CFG --keep_epochs 25 50 90 100`; v-prediction, L1, 100 epochs, per-epoch noise, monitoring every 5 epochs); `cond_fam_b64` launched 18:50 on GPU 1 (pid 2370874, same flags, mode cond_fam). Both ≈ 1 125 steps per epoch at ≈ 0.27 s per step (≈ 300 s per epoch, ≈ 8.5 h to DONE); first-epoch losses 0.05–0.07 (v-prediction L1) at step 400.

- `varhead_fam_b64`: launched 2026-11-05 00:50 on GPU 1; DONE = 40 at 04:18 (3.49 h, 299 s/epoch, 0 skipped). **Pre-registered selection = epoch 38** (monitoring NLL 0.20257; epoch 40 = 0.20282 — a plateau, no overfitting in the family setting, unlike the noise-free Phase 1a′ regressors).
- `regress_fam_b64`: launched 00:50 on GPU 0; DONE = 100 at 10:08 (9.21 h, 319 s/epoch — data-bound by the per-item MMSE in the workers, 0 skipped). **Pre-registered selection = epoch 85** (monitoring 0.33482; epoch 100 = 0.33583 — plateau).
- The four pre-registered checkpoints are therefore `uncond_b64@100`, `cond_fam_b64@100`, `regress_fam_b64@85`, `varhead_fam_b64@38`. The regress run holds 100 checkpoints (≈ 70 GB) and the varhead run 40 (≈ 28 GB); pruning to the selected + 25 / 50 / 100 awaits the user's say-so, as before.

## E1 — device1, sequential, five SNRs (Task 15)

Records `sensor/eval/e1_dev1_seq_<snr>_{main,C,B}.json` (main = L D V on the 200 gate cubes; C = C C1 on the 200; B on the 50-cube subset — the three record kinds per setting because `sd_verdict` refuses a method present twice per setting and pairs B against the 200-cube L / D on the common 50). Report-only verdict `sensor/verdict__e1__report.json`. All 200-cube numbers below; RMSE in % of range, CRPS in %.

| SNR | L: RMSE / CRPS / cov50 / cov90 / AUSE | D: RMSE | V: CRPS / cov50 / cov90 / AUSE | C: mean-of-32 RMSE / one-call C1 / single sample | C: CRPS / cov50 / cov90 / AUSE |
|---|---|---|---|---|---|
| 60 (floor) | 1.105 / 0.590 / 0.559 / 0.926 / 0.330 | **1.052** | 0.584 / 0.503 / 0.897 / **0.302** | 1.107 / 1.091 / 1.558 | **0.583** / 0.555 / 0.921 / 0.401 |
| 50 | 1.556 / 0.840 / 0.557 / 0.926 / 0.355 | **1.352** | 0.832 / 0.498 / 0.893 / **0.323** | 1.423 / 1.414 / 2.052 | **0.759** / 0.548 / 0.925 / 0.398 |
| 40 | 2.622 / 1.405 / 0.546 / 0.920 / 0.340 | **1.820** | 1.383 / 0.490 / 0.896 / **0.309** | 1.933 / 2.044 / 3.055 | **1.053** / 0.597 / 0.944 / 0.411 |
| 30 | 4.143 / 2.260 / 0.549 / 0.922 / 0.363 | **2.518** | 2.194 / 0.479 / 0.904 / **0.319** | 3.245 / 3.495 / 5.023 | **1.757** / 0.598 / 0.942 / 0.391 |
| 20 | 6.425 / 3.560 / 0.541 / 0.922 / 0.409 | **3.238** | 3.493 / 0.466 / 0.890 / **0.355** | 4.845 / 5.159 / 7.540 | **2.649** / 0.631 / 0.956 / 0.384 |

Clauses (pre-registered, spec §6.3), line C:
- **T1 (mean-of-32 RMSE within 5 % of D by 2 clustered SE): FAIL at every SNR** — ratio of means 1.052 / 1.052 / 1.062 / 1.289 / 1.496 at 60 / 50 / 40 / 30 / 20 dB (2-SE upper bounds 1.08 / 1.08 / 1.10 / 1.38 / 1.60). At the high-SNR end C's posterior mean is 5 % behind the regressor (the 1/K Monte-Carlo term alone adds ≈ 1.5 % of the spread variance at K = 32; C1, the one-call point, is within 4 % at 60 / 50 dB); from 30 dB down the gap opens to 29–50 %: the regressor's conditional mean is far better than the diffusion's at low SNR, where C's single samples (5.0 / 7.5 %) carry the √2 spread cost and its mean is pulled by the over-dispersion below.
- **T2 (CRPS below L's posterior AND below V by 2 clustered SE, cov90 ∈ [0.85, 0.95]): PASS at 30 dB, FAIL at 20 dB on cov90 only.** At 30 dB: CRPS − L = −0.503 ± 0.015 (cluster), − V = −0.437 ± 0.013, cov90 0.942. At 20 dB: CRPS − L = −0.911 ± 0.027, − V = −0.845 (both clear passes; C's CRPS is 26 % below L's and 24 % below V's), but cov90 = 0.956 lies 0.006 above the band: C is slightly over-dispersed (cov50 0.63 against 0.50 nominal), increasingly so as the SNR falls.
- Sanity: every method's RMSE is non-increasing in SNR; D, V, C and C1 are at or below L + 2 SE at the 60 dB floor (C1 1.091 and D 1.052 beat the MMSE's 1.105); B fails the floor (6.2 %).

Reading. (i) For the point estimate the family regressor is the best method at every SNR and halves the MMSE error at 20 dB (3.24 vs 6.43 %); C's posterior mean never beats it, consistent with Phase 1a′'s lesson (under RMSE the regressor is the right tool) and the spec's own prediction for T1 was wrong at low SNR. (ii) For the predictive distribution the picture inverts where the sensor is poor: from 40 dB down the diffusion's CRPS is 20–26 % below both the variance head's and the MMSE posterior's, with 90 % coverage inside or within 0.006 of the nominal band — this is the paper's figure (ii), and it is the opposite of Phase 2a's outcome (there the variance head won CRPS by 10 %). The variance head still ranks errors better (AUSE 0.30–0.36 against C's 0.38–0.41), so C's advantage is in the shape of the marginals, not in localising the uncertainty. (iii) C's single samples are rough (log HF-energy ratio +1.1 to +4.0 against the truth), the mean of 32 smooth (+0.3 to +2.0): texture is still not a strength. (iv) B: see the next section — not a valid line in this experiment.

## B — the guided unconditional prior fails on its prior (Task 15, diagnosis 2026-11-05)

The two B records that exist (60 and 50 dB, 50 cubes, K = 32, ΠGDM 50 steps) are degenerate: mean-of-32 RMSE 6.21 / 6.25 %, single sample 24.8 / 25.4 %, CRPS 6.6 / 6.8, cov50 0.83, log HF +3.9, identical at the two SNRs. Diagnosis (controller probe + a dedicated diagnostic subagent; scripts and report under the job's `diagB/`):
- **The sampler is sound.** H x + b reproduces the clean y to 1e-7; `y_eff` = y − b; dtypes and devices are right; `draw_guided` returns the guided x̂₀. With the learned prior replaced by the per-pixel spectral Wiener filter as the denoiser, the SAME ΠGDM code reconstructs the 8 probe cubes to 1.47 % at 60 dB and 6.22 % at 30 dB (MMSE 1.075 / 5.81 on those cubes). No sampler-side change helps the learned prior (σ floors 1e-4…1e-2, guidance scale 0.1 / 0.3, 100 steps, DPS ζ 1 / 0.1: 13–21 % mean RMSE against 13.7 as-is on the 8 cubes).
- **The learned prior is not a usable generative model.** Unguided DDIM-50 samples have 39–41 % RMSE to the truth with 30–37 % of their elements saturated at ±1. Its x̂₀-estimate error against t (16 gate cubes, 60 dB) is hump-shaped — 4.4 / 8.0 / 14.2 / 19.1 / 22.1 / 22.8 / 18.6 / 13.1 / 8.2 / 14.6 % at t = 25 / 50 / 100 / 150 / 200 / 299 / 400 / 500 / 700 / 900 — worse than predicting the mean cube (17.2 %) over t = 100–400, and the SAME at epochs 25 and 100. The network does respond to t (its output changes by 60–200 % when t is swapped), so it is not t-blind.
- **What the curve is:** an INPUT-BLIND predictor, v̂ = E[v | t] = −√(1 − ᾱ_t) · (leave-one-out mean cube), has the x̂₀ error 4.7 / 8.6 / 15.4 / 20.9 / 24.6 / 26.8 / 24.7 / 21.5 / 18.6 / 18.3 at the same t. The learned priors match it to the first decimal for t ≤ 200 and beat it only from t ≈ 300 up (clearly at 700–900, where predicting the image from near-pure noise is a denoising-autoencoder task the network did learn). In words: at low and medium noise the network ignores x_t and emits the per-t mean of the target; the guided chain's trajectory through t ≈ 150–500 is therefore driven by garbage estimates and its samples saturate.
- **Three recipe variants reproduce the curve exactly after 10 epochs** (pilots on GPU 1, 0.9 GPU-h each; probe at epochs 5 and 10): uniform v-L1 weighting (`--snr_gamma 0`): 4.7 / 8.5 / 15.3 / 20.6 / 23.9 / 24.9 / 20.7 / 15.3 / 10.6 / 14.9; a time-token context path (`uncond_tt`, the legacy models' way of injecting t through cross-attention, commit 65d028a): 4.7 / 8.5 / 15.3 / 20.5 / 23.8 / 24.6 / 20.1 / 14.2 / 8.9 / 15.0; both with 34–37 % saturated unguided samples. Neither the loss weighting nor the t path is the cause; the curve does not move between epochs 5, 10, 25 and 100. The x₀-prediction pilot (`uncond_x0pilot_b64`) is the last variant tried — see below.
- **Context.** This repo's recipe (base 64, 112 k steps at batch 8) has never produced a working full-cube x-space diffusion model: Phase 1a's conditional A / A′ were unconverged at ≈ 7 % after 100 epochs; Phase 2a's residual diffusion was under-fit at t ≤ 299 (worse than a linear filter) but survived because its output is composed with a good centre; the Phase 0 warm-start investigation found the legacy 2-D l2 checkpoint sampling at 57 % RMSE. Residual / whitened targets train well; the image itself does not.

- **The x₀-prediction pilot breaks the pattern** (`uncond_x0pilot_b64`, `--prediction_type x0`, 10 epochs, 0.9 GPU-h): x̂₀ error 3.3 / 3.4 / 3.4 / 3.5 / 3.6 / 4.0 / 4.6 / 5.7 / 10.6 / 23.5 % at t = 25 … 900 (epoch 5: 5.0 … 23.7 — still improving), monotone, well below the input-blind curve everywhere up to t = 700 (at t = 299: 4.0 against 24.6 for the v-prediction priors), and its unguided DDIM-50 samples have 27.5 % RMSE to the truth with 0.1 % of their elements saturated (a random plausible cube would score ≈ 24 % on these per-cube-normalised data). Its only weak spot is the far high-t end (23.5 % at t = 900, where x₀-prediction is known to be noisy), which the chain's first steps and the clamp tolerate. The reading is a parameterisation effect: with v-prediction the low-t target is ≈ ε and the network settled on the input-blind per-t mean; with x₀-prediction the low-t target is ≈ the identity, which forces it to read x_t.

**Ruling (controller, 2026-11-05, superseding "failed-by-prior"):** the pilot meets the pre-registered decision rule (monotone and ≥ 30 % below the current prior at t ≤ 300) by a wide margin → **B's prior is retrained with x₀-prediction** (`uncond_x0_b64`, 100 epochs, L1, min-SNR γ = 5, launched 2026-11-05 14:35 on GPU 1, ≈ 7 h) and the harness builds B's sampler from the checkpoint's own prediction type (C stays v-prediction). This amends spec §5.5's "v-prediction" for the prior only, on the evidence above. The v-prediction prior's 60 / 50 dB records stay in the archive as the record of the failure. Budget: 3 pilots (2.7 GPU-h) + 7 GPU-h retrain + ≈ 24 GPU-h of B records (2 GPU-h each, not the plan's 0.63) — the user's GPUs are otherwise idle; PH5's B rows are dropped first if time runs short (spec drop order).

## E2 — zero-shot transfer to Device 2 and PH5, sequential (Task 16; D, V, C trained on Device 1 only)

Records `e2_{device2,ph5}_sequential_<snr>_{main,C}.json` (200 gate cubes); report-only verdict `verdict__e123_noB__report.json`.

| sensor | SNR | L: RMSE / CRPS | D: RMSE | V: CRPS / cov50 / cov90 / AUSE | C: mean-of-32 / C1 / single | C: CRPS / cov50 / cov90 / AUSE |
|---|---|---|---|---|---|---|
| device2 | 40 | 3.185 / 1.751 | **3.067** | 1.735 / 0.488 / 0.893 / 0.347 | 3.122 / 2.972 / 4.241 | **1.700** / 0.473 / 0.889 / 0.406 |
| device2 | 30 | 5.122 / 2.779 | **4.133** | 2.739 / 0.495 / 0.897 / 0.319 | 4.834 / 4.523 / 6.686 | **2.619** / 0.490 / 0.904 / 0.380 |
| device2 | 20 | 7.682 / 4.288 | **6.849** | **4.234** / 0.488 / 0.885 / 0.339 | 8.355 / 7.421 / 10.894 | 4.728 / 0.443 / 0.843 / 0.432 |
| ph5 | 40 | **8.963** / **4.722** | 10.084 | 4.737 / 0.502 / 0.908 / 0.251 | 16.267 / 12.533 / 18.417 | 9.135 / 0.247 / 0.629 / 0.231 |
| ph5 | 30 | **10.413** / **5.589** | 12.147 | 5.638 / 0.509 / 0.890 / 0.296 | 19.516 / 14.165 / 21.872 | 11.128 / 0.243 / 0.627 / 0.249 |
| ph5 | 20 | **11.666** / **6.426** | 15.878 | 6.634 / 0.464 / 0.812 / 0.387 | 22.244 / 15.823 / 24.538 | 13.403 / 0.220 / 0.598 / 0.365 |

- **T3 (line C, 30 dB): FAIL** — (RMSE_C / RMSE_L) is 0.783 on device1 and 0.944 on device2, against the limit 1.10 × 0.783 = 0.861. The sufficient-statistic conditioning transfers only partly: on device2 C still beats the MMSE at 40 and 30 dB (by 2 % and 6 %) and beats V on CRPS at 40 / 30 dB (1.700 vs 1.735; 2.619 vs 2.739) with cov90 0.89–0.90, but at 20 dB it falls behind both L and V (8.36 % vs 7.68; CRPS 4.73 vs 4.23) with cov90 0.843. The regressor D transfers better: 4 / 19 / 11 % below L at 40 / 30 / 20 dB (31 / 39 / 50 % on device1).
- **PH5 (report-only):** nothing learned on device1 transfers. D is 12–36 % WORSE than the MMSE, C's mean 1.8–1.9 × worse, C badly under-dispersed (cov50 0.22–0.25, cov90 0.60–0.63); only V's CRPS stays level with L's. PH5's legacy-axis response has rank 16 (Stage 0) and a very different posterior-std structure, so its (x̂, s) maps are out of the training distribution of every family model; this is the sensor on which the paper's "no retraining" claim is false for the conditional networks.

## E3 — snapshot mode, device1 (Task 17; D, V, C saw snapshot cubes in training; L_demosaic is the classical baseline)

Records `e3_device1_snapshot_<snr>_{main,C}.json` (200 gate cubes).

| SNR | L exact: RMSE / CRPS | L_demosaic: RMSE / CRPS | D: RMSE | V: CRPS / cov50 / cov90 / AUSE | C: mean-of-32 / C1 / single | C: CRPS / cov50 / cov90 / AUSE |
|---|---|---|---|---|---|---|
| 60 | 12.107 / 6.847 | 5.574 / 2.868 | **2.696** | 6.084 / 0.236 / 0.956 / 0.086 | 6.531 / 7.585 / 13.069 | 3.861 / 0.776 / 0.993 / 0.360 |
| 40 | 12.118 / 6.854 | 5.613 / 2.899 | **2.727** | 6.093 / 0.238 / 0.955 / 0.087 | 6.566 / 7.614 / 13.094 | 3.878 / 0.774 / 0.993 / 0.360 |
| 30 | 12.218 / 6.912 | 5.835 / 3.063 | **2.950** | 6.170 / 0.256 / 0.950 / 0.099 | 6.841 / 7.826 / 13.287 | 4.011 / 0.758 / 0.992 / 0.363 |
| 20 | 12.877 / 7.278 | 6.867 / 3.745 | **3.971** | 6.635 / 0.320 / 0.940 / 0.150 | 7.719 / 8.781 / 14.142 | 4.467 / 0.732 / 0.985 / 0.384 |

- **T4 (line C vs L_demosaic at 30 dB, limit 0.85): FAIL** — ratio 1.172 (2-SE bound 1.211): C's posterior mean (6.84 %) is WORSE than demosaic-then-MMSE (5.84 %). D passes the same clause overwhelmingly: **2.95 % = half the demosaic baseline** (ratio 0.506), at every SNR.
- Reading. In snapshot mode the centre x̂ is the exact per-pixel posterior from ONE measured channel (≈ 12 %, essentially the prior mean) and s is correspondingly wide, so the whitened residual r = (x − x̂)/s is large and highly structured: it IS the demosaicing problem. The one-call regressor learns it (2.7–4.0 %); the diffusion trained on the same target does not — its one-call point C1 is 7.6–8.8 %, its mean of 32 samples 6.5–7.7 %, its samples hugely over-dispersed (cov50 0.73–0.78, cov90 0.99) and still rough (single sample 13 %). The variance head inherits the poor centre (its mean is x̂) and cannot shrink enough to be calibrated (cov50 0.24). So the regressor-centred design that served C well in sequential mode handicaps the probabilistic lines wherever the centre is poor; a diffusion centred on D's prediction (Phase 2a's construction) would have been the fairer snapshot test — out of scope here, recorded for the write-up.

## B's retrained prior (x₀-prediction) — `uncond_x0_b64`
Launched 2026-11-05 14:35 on GPU 1; DONE = 100 at 22:40 (8.07 h, 288 s/epoch, 0 skipped); monitoring loss (x₀-L1, min-SNR γ = 5) 0.0827 → 0.0397, plateau over epochs 85–100 → the pre-registered epoch-100 checkpoint. Probe on the 16 cubes at 60 dB: **x̂₀ error 2.3 / 2.3 / 2.5 / 2.6 / 2.8 / 3.2 / 3.8 / 4.6 / 7.9 / 15.3 % at t = 25 / 50 / 100 / 150 / 200 / 299 / 400 / 500 / 700 / 900** (epoch 50: 2.6 … 15.9) — monotone and below the mean-cube predictor (17.2 %) at every t; unguided DDIM-50 samples 25.0 % RMSE to the truth (≈ a random plausible cube) with 0.0 % of elements saturated. B's records are taken with this prior (ΠGDM, 50 steps, K = 32, 50-cube subset); the v-prediction prior's two records are archived under `sensor/eval/vpred_prior/`.

## Verdict so far (report-only, B records in progress)
Line C passes 1 of the pre-registered clauses (T2 at 30 dB), fails T1 at every SNR, T2 at 20 dB on coverage only, T3 and T4. Line B: pending. Sanity: every method monotone in SNR; D, V, C, C1 at or below the MMSE floor reference on device1 at 60 dB.
