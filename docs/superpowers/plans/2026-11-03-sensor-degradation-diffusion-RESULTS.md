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
