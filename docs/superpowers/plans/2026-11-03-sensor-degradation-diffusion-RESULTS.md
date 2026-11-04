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

### Baselines (`scripts/sensor/stage0_baselines.py`)
_pending_
