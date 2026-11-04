# Phase 3 — Diffusion Reconstruction From the Poor EC Sensor Under Noise, Across Devices and in Snapshot Mode: Results

Spec: `docs/superpowers/specs/2026-11-03-sensor-degradation-diffusion-design.md` (approved 2026-11-04, ac4e4fd). Plan: `docs/superpowers/plans/2026-11-04-sensor-degradation-diffusion.md` (2c49684, amended during execution: Task 3 test clip bc63d8c, Task 6 legacy assertions / FiLM 7a4c359, ΠGDM coefficient 6d2d431, crc32 file seeds 15ce085). Build phase (Tasks 1–12 + the carried-fix batch 11b) complete 2026-11-04 at 3bc4a1d: 546 tests green, CPU smoke `SMOKE OK`. Execution ledger: `.superpowers/sdd/2026-11-04-sensor-degradation-diffusion/progress.md` (archived under `results/residual_warmstart/sensor/sdd_archive/` at the end).

Conventions: every number is on the 200 gate cubes of `tests/data/val_subset_200.json` unless stated (B on its 50-cube `ddpm1000_subset`, paired on the common files); errors in % of the [0, 1] range; CRPS in %; "±" is the iid SE over cubes with the source-image-clustered SE in parentheses where a paired difference is reported; K = 32 samples per cube; SNR in dB against the global P_y of device1 in sequential mode; the 60 dB floor is the noise-free end (no σ = 0 anywhere).

Design decisions taken during the build (plan D1–D6 and the rulings in the ledger): dict batches for the sensor modes; the per-element posterior std replaces σ_d as the residual scale (floor 0.02 trainer scale); no persistent DataLoader workers for datasets with `set_epoch`; `uncond` shares the dataset / meta / launcher path; the toy that licenses the calibration clauses is isotropic N(0, I); L's Gaussian posterior uses the unfloored std. Rulings that changed the plan text: ΠGDM's correction is x̂₀ + (1 − ᾱ_t)/√ᾱ_t · g (the score correction in x₀ coordinates; the draft's √ᾱ_t coefficient diverged on the isotropic toy); the per-cube noise seed is crc32 of the cube's path relative to the data root, not its list position, so 50-cube and 200-cube records pair by file; `uncond`'s time path is FiLM-only and t-blind at initialisation (zero-initialised FiLM layers).

## Stage 0 (Task 13)

_pending_
