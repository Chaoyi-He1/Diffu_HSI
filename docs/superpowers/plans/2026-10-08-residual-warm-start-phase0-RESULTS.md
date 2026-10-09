# Phase 0 results — 2026-10-08

## Linear prior
fit on 768000 pixels of 3000 files; s = 3e-07; held-out per-element RMSE 0.00788 ([-1,1]) = 0.394 % of range

This is the 31-band unclipped metric on held-out training files. The 64-band clipped metric on validation cubes is the d = 1 baseline below.

## Residual scale sigma_d (training files)
| d | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| sigma_d ([-1,1] scale) | 0.0053638 | 0.021141 | 0.049963 | 0.099054 |
| sigma_d (% of the [0,1] range) | 0.27 | 1.06 | 2.50 | 4.95 |

## Official baselines (loader validation split, 1000 files, grid-aligned upsampling)
| d | RMSE % | ± SE | P022 | P023 | P024 | PSNR | SAM ° |
|---|---|---|---|---|---|---|---|
| 1 | 0.2576 | 0.001 | 0.2478 | 0.2632 | 0.2318 | 51.913 | 0.3712 |
| 2 | 1.0335 | 0.008 | 0.9654 | 1.0106 | 1.2512 | 39.957 | 1.0233 |
| 4 | 2.3126 | 0.023 | 2.1254 | 2.2164 | 3.1254 | 33.140 | 1.7785 |
| 8 | 4.4904 | 0.049 | 4.1841 | 4.2083 | 6.6237 | 27.464 | 3.0196 |

Phase 1 gate at d = 4: RMSE ≤ 0.85 × 2.3126 = 1.966 % at ≤ 10 NFE.

## Training-step timing
| config | params | sec/step | peak GB | sec/epoch | 100 epochs | 200 epochs |
|---|---|---|---|---|---|---|
| base 64, batch 8, 1 GPU | 58,942,272 | 0.228 | 3.36 | 256.5 | 7.1 h | 14.2 h |
| base 256, batch 8/GPU, FSDP 2 GPUs | 940,117,056 | 0.882 | 16.09 | 496.3 | 13.8 h | 27.6 h |

## Notes for the Phase 1 decision
- The baselines use the grid-aligned bilinear lift (low-res sample k sits at high-res pixel k·d, edge-extended), which reproduces the loader's `[::d]` grid exactly. The earlier figures 1.64 / 3.57 / 6.14 % (d = 2/4/8) came from torch's `align_corners=False` bilinear, a half-pixel-shifted grid, and are obsolete; a half-pixel-shifted lift reproduces them (1.53 / 3.41 / 5.89 % on 40 files). The Phase 1 gate therefore uses the aligned values.
- Timing is compute-only (the timer excludes data loading). The residual dataset's `__getitem__` costs ~0.17 s per cube, so with `num_workers=4` the pilot would be loader-bound at ~0.34 s/step (~385 s/epoch, ~10.7 h per 100 epochs); with `num_workers ≥ 8` it is compute-bound at the recorded figure. The FSDP configuration (2 × 4 workers) is compute-bound as recorded. Phase 1 must use ≥ 8 workers per process and a `DistributedSampler`.
- The full configuration peaks at 16.1 GB of 20 GB with no EMA, validation or checkpoint gather; a sharded fp32 EMA brings it to ≈ 18 GB. Phase 1 must choose: sharded/offloaded EMA at batch 8/GPU, or batch 4/GPU.
- Rulings that bind Phase 1 code: checkpoints save/load the `ResidualWrapper`'s state dict (keys `net.*`, `strict=True`); the residual chain is sampled with `clamp_x0 = (−2/σ_d, 2/σ_d)`, never the default ±1 (which would cap every correction at ±σ_d); the cond dict is moved to device per key; the dataset's `x0_hat` is always used, never recomputed from float32 `y_d`; per-cube values in eval JSON rows are ordered by each row's `files` (dataset order), not the subset JSON order.
- The y-swap check now swaps only the network's inputs and composes with the true `x0_hat` (a condition-blind network fails it); SEs are per-cube after averaging over seeds; the y-swap batch is 1 P022 / 6 P023 / 1 P024 cubes from distinct source images.

## Gate
- [x] tests pass (47 passed, dataset tests included)
- [x] baselines recorded in tests/data/baseline_phase0.json
- [x] timings recorded in results/residual_warmstart/timing.json (two records; `results/` is git-ignored, so this file, the per-d SEs and `linear_prior_R1.npz` live on this server only)
