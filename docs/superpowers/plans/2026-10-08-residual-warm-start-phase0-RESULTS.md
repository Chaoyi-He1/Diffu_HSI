# Phase 0 results — 2026-10-08 (regenerated after the final fix wave)

## Linear prior
fit on 768000 pixels of 3000 files; s = 3e-07; held-out per-element RMSE 0.00817 ([-1,1] scale, clipped, float32 y) = 0.408 % of range (19200 random pixels of 300 files)

`s` is chosen with the production estimator (`linear_inverse`, clipped) on float32 sensor values, scored on 64 seeded random pixels per held-out training file. Held-out RMSE per candidate ([-1,1] scale): 1e-7: 0.008987, **3e-7: 0.008169**, 1e-6: 0.008715, with every other candidate worse. μ, C, K and s are byte-identical to the first fit. The earlier line (0.394 %) used the unclipped estimate on a `[::64]` stride, which samples column 0 only. This is a 31-band metric on training files; the 64-band metric on validation cubes is the d = 1 baseline below. Prior file sha256 `1ad9108f17bee02eceadf63e5cd59671fe3475a9b015aa8c7c9acfd5adaba2b0`.

## Residual scale sigma_d (training files)
| d | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| sigma_d ([-1,1] scale) | 0.0053784 | 0.022324 | 0.051661 | 0.095731 |
| sigma_d (% of the [0,1] range) | 0.269 | 1.116 | 2.583 | 4.787 |

Each d now uses its own 500 training files, drawn by a generator seeded with (seed, d), so σ_d does not depend on the order of `--d_values`. Compared with the first table (0.0053638 / 0.021141 / 0.049963 / 0.099054), d ≥ 2 moved by +5.6 / +3.4 / −3.4 %. The float32 change moves the d ≥ 2 baselines by < 0.1 %, so these shifts are file-sampling spread: a 500-file σ_d is good to a few percent. The unit-RMS test allows ±20 %, and passes for all four d.

## Official baselines (loader validation split, grid-aligned upsampling, x̂₀ from float32 y_d)
| d | RMSE % (1000 files) | ± SE | P022 | P023 | P024 | PSNR | SAM ° | 200-cube subset RMSE % ± SE | 50-cube DDPM subset RMSE % |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.2632 | 0.0014 | 0.2516 | 0.2687 | 0.2400 | 51.709 | 0.3924 | 0.2664 ± 0.0032 | 0.2699 |
| 2 | 1.0343 | 0.0079 | 0.9659 | 1.0115 | 1.2521 | 39.949 | 1.0310 | 1.0564 ± 0.0186 | 1.0819 |
| 4 | 2.3129 | 0.0230 | 2.1257 | 2.2168 | 3.1257 | 33.139 | 1.7824 | 2.3512 ± 0.0529 | 2.4280 |
| 8 | 4.4905 | 0.0487 | 4.1842 | 4.2085 | 6.6238 | 27.464 | 3.0218 | 4.5799 ± 0.1130 | 4.6930 |

All numbers are from `tests/data/baseline_phase0.json`, written by `scripts/make_baseline_record.py` from `results/residual_warmstart/eval/baseline_d{1,2,4,8}.json`. The subset columns are recomputed with `baseline_eval` and cross-checked per cube against the 1000-file files. Against the previous record (0.2576 / 1.0335 / 2.3126 / 4.4904), d = 1 rose by 2.2 %, because the estimate no longer sees float64 sensor precision; d = 2 / 4 / 8 moved by +0.08 / +0.01 / +0.00 %.

**Phase 1b gate at d = 4:** B's RMSE on the 200-cube subset ≤ 0.85 × the subset's baseline = 0.85 × 2.3512 = 1.9985 %, at ≤ 10 NFE, plus the y-swap check; DDIM-10 vs DDPM-1000 on the 50-cube intersection.

## Training-step timing
These figures are unchanged from Task 8; they do not depend on the prior.

| config | params | sec/step | peak GB | sec/epoch | 100 epochs | 200 epochs |
|---|---|---|---|---|---|---|
| base 64, batch 8, 1 GPU | 58,942,272 | 0.228 | 3.36 | 256.5 | 7.1 h | 14.2 h |
| base 256, batch 8/GPU, FSDP 2 GPUs | 940,117,056 | 0.882 | 16.09 | 496.3 | 13.8 h | 27.6 h |

## Notes for the Phase 1 decision
- **Sensor precision ruling.** The sensor data is the float32 `y_d` the network consumes. `HFDResidualData` computes x̂₀ from that array; the estimate itself runs in float64. x̂₀ is therefore a deterministic function of the returned `y_d`, and recomputing it with `estimate_from_sensor` reproduces the item's x̂₀ exactly, as tested at d = 1 and d = 4. The prior's `s` is chosen on the same float32 values with the same clipped estimator. Before this ruling, the baseline read float64 `y`, a precision no sensor and not the network had.
- **Fresh clones.** `results/` is git-ignored. A clone without `results/residual_warmstart/linear_prior_R1.npz` skips the 13 prior-dependent tests:
  - in `test_residual_dataset.py`: `test_item_shapes_and_consistency`, `test_x0_hat_is_a_function_of_the_returned_float32_y_d[1, 4]`, `test_collate`, `test_sigma_d_is_unit_rms_after_scale_script[1, 2, 4, 8]`, `test_missing_sigma_d_raises` and `test_prior_with_a_different_sensor_matrix_is_rejected`;
  - in `test_eval_harness.py`: `test_baseline_eval_reproduces_recorded_numbers`, `test_dataset_for_rejects_files_outside_the_validation_split` and `test_checkpoint_eval_records_per_cube_values`.

  Measured on an archive of HEAD: 51 passed, 13 skipped. To restore them, run `scripts/fit_linear_prior.py`, then `scripts/compute_residual_scale.py --d_values 1 2 4 8`, and check the prior's sha256 against `prior_sha256` in `tests/data/baseline_phase0.json`.
- **Ablation A is confounded (review finding).** The cross-attention context path has no positional encoding: `create_context_for_resolution` adds only the time embedding to the flattened sensor tokens (`model/u2net_hyperspectral.py:443-451`). The sensor tokens are therefore position-blind. B gets spatial alignment through the x̂₀ concatenation; A gets none. Phase 1 needs an A′ variant (standard target + x̂₀ concatenated) next to A.
- The baselines use the grid-aligned bilinear lift: low-res sample k sits at high-res pixel k·d, with edge extension, which reproduces the loader's `[::d]` grid exactly. The earlier figures 1.64 / 3.57 / 6.14 % (d = 2/4/8) came from torch's `align_corners=False` bilinear, a half-pixel-shifted grid, and are obsolete. A half-pixel-shifted lift approximately reproduces them (40-file subset): 1.53 / 3.41 / 5.89 %. The Phase 1 gate therefore uses the aligned values.
- Timing is compute-only: the timer excludes data loading. The residual dataset's `__getitem__` costs ~0.17 s per cube. With `num_workers=4` the pilot would be loader-bound at ~0.34 s/step (~385 s/epoch, ~10.7 h per 100 epochs); with `num_workers ≥ 8` it is compute-bound at the recorded figure. The FSDP configuration (2 × 4 workers) is compute-bound as recorded. Phase 1 must use ≥ 8 workers per process and a `DistributedSampler`.
- The full configuration peaks at 16.1 GB of 20 GB with no EMA, validation or checkpoint gather; a sharded fp32 EMA brings it to ≈ 18 GB. Phase 1 must choose: sharded/offloaded EMA at batch 8/GPU, or batch 4/GPU.
- Rulings that bind Phase 1 code:
  - Checkpoints save and load the `ResidualWrapper`'s state dict (keys `net.*`, `strict=True`). They also carry a `meta` dict (`mode, d, base_channels, prediction_type, sigma_d, prior_sha256`), which `checkpoint_eval` checks against its arguments and the prior.
  - The residual chain is sampled with `clamp_x0 = (−2/σ_d, 2/σ_d)`, never the default ±1, which would cap every correction at ±σ_d.
  - The cond dict is moved to the device per key.
  - Per-cube values in eval JSON rows are ordered by each row's `files` (dataset order), not the subset JSON order.
  - Checkpoint evaluations are written to `results/residual_warmstart/eval/<mode>/d<d>/<ckpt>.json`, with a `provenance` record.
- The y-swap check swaps only the network's inputs and composes with the true `x0_hat`, so a condition-blind network fails it. SEs are per cube after averaging over seeds, overall, per class (P022 / P023 / P024) and for the unseen aggregate P023 + P024; paired differences to the baseline are reported the same way. The y-swap batch is 1 P022 / 6 P023 / 1 P024 cubes from distinct source images.
- σ_d must exist in the prior for the requested d; a missing d is an error, with no fallback to 1. The prior's R must equal the loader's. Re-fitting keeps an existing σ_d table unless `--overwrite_sigma` is passed. Both scripts write the npz atomically.

## Gate
- [x] tests pass: `64 passed in 60.55s (0:01:00)` (`python -m pytest tests -q`, dataset and prior tests included)
- [x] baselines recorded in `tests/data/baseline_phase0.json`: 1000-file, 200-cube and 50-cube values with SE, plus provenance (prior s, σ_d, prior sha256). Pinned on 40 files within 0.01 at d = 1 and d = 4.
- [x] timings recorded in `results/residual_warmstart/timing.json` (two records). `results/` is git-ignored, so this file, the per-cube baseline files and `linear_prior_R1.npz` live on this server only.
