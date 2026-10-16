# Phase 1a results — 2026-10-12

**Gate 1a verdict: INVALID.** B fails the y-swap rule of spec §7 (true ≤ 0.9 × swap_full at t = 100 and 300), so by spec §11 the run is invalid and the comparison numbers are not acted on. The failure is a lack of power at B's gain size, not an unused condition: at t = 999 a full-condition swap removes B's gain (Observation 2). B's best setting is one DDIM step: 2.2426 % RMSE against the baseline's 2.3512 %, a gain of 0.1086 ± 0.0060 (± 0.0068 clustered by source image; 4.6 %). Every further sampling step makes B worse, as RMSE predicts for any stochastic sampler (Observation 1); at the pre-registered DDIM-10 it is 0.7349 ± 0.0168 above the baseline. Neither ablation comes within 4.5 points of the baseline.

Sources (all under `results/residual_warmstart/`, git-ignored): `runs/<run>/{train_log.jsonl,args.json,launches,DONE}`, `runs/pilot_events.log`, `runs/pilot_driver.log`, `eval/<mode>/d4/<run>__checkpoint_epoch_{25,50}__quick.json` and `…_epoch_100.json`, `eval/residual/d4/gate_1a__residual_d4_b64.json` (the official gate record) and `…/gate_1a__residual_d4_b64__postreview.json` (a post-review re-run with the fix-wave gate; see Gate 1a), `timing_phase1.json`, `timing.json`. The Phase 0 record is `tests/data/baseline_phase0.json`. The t = 999 y-swap numbers in Observation 2 come from a post-review CPU run of `scripts/yswap_one_step.py`; its output is not in the results tree, and the command is given there. *Correction (2026-10-16):* those t = 999 runs fed B a leaky x_t (Observation 2); the leak-free 32-cube record is `diagnostics/yswap_residual_d4_b64_t999_n32__xT.json` (see `2026-10-12-residual-warm-start-phase1a-prime-RESULTS.md`). This note was revised after the final review; the controller's rulings and the deferred backlog are in `2026-10-09-residual-warm-start-phase1a-LEDGER.md`. Every evaluation carries `meta_checked = true` and `git_head` 2fd2bee. RMSE is in % of each cube's range. "±" is one standard error over cubes. Cubes are patches of 113 source images (1–5 patches each), so on the 200-cube comparisons cube-level SEs understate the uncertainty by ≈ 1.1–1.2× (up to 1.33× for some B − ablation comparisons). Cluster-robust SEs (CR1 over source images) are given where they matter. "Paired" means the per-cube difference against the baseline (or another arm) on the same cubes, averaged over the 2 noise seeds.

## Runs (d = 4, base 64, batch 8, 1 GPU each, 100 epochs or 24 h; no EMA)
| run | GPU | launched → DONE | epochs done | wall h | s/epoch median (data %) | loss epoch 1 → 100 | val loss (800 non-gate files) epoch 5 → 100 | peak GB | skipped steps | launches (relaunches) |
|---|---|---|---|---|---|---|---|---|---|---|
| B (residual) | 0 | 10-11 14:21:22 → 10-11 23:24:23 | 100 | 9.05 | 312.0 (0.77 %) | 0.1232 → 0.1060 | 0.1191 → 0.1032 | 3.37 | 0 | 1 (0) |
| A (standard) | 1 | 10-11 14:21:22 → 10-11 23:29:44 | 100 | 9.14 | 314.8 (0.74 %) | 0.1105 → 0.0747 | 0.0847 → 0.0732 | 3.35 | 0 | 1 (0) |
| A′ (standard_xhat) | 1 | 10-12 00:54:33 → 10-12 08:37:26 | 100 | 7.71 | 260.6 (0.84 %) | 0.1110 → 0.0774 | 0.0860 → 0.0756 | 3.37 | 0 | 1 (0) |

- `wall h` is `elapsed_hours` of epoch 100. `data %` is the median of `data_sec / sec` per epoch. The DONE times are the DONE files' mtimes. Parameters: B and A′ 58,942,272; A 58,905,408 (no x̂₀ input channels). All three runs use 8 loader workers, lr 1e-4 → 3.32e-6 (cosine to 0.033·lr), v-prediction, L1, min-SNR 5, seed 42.
- The 24 h cap never fired. Each run finished its full cosine schedule in 100 epochs.
- Losses are not comparable across arms: B's target is the unit-RMS residual (x − x̂₀)/σ_d (σ₄ = 0.051661), while A's and A′'s target is x. B's validation loss falls monotonically and is flat over the last 20 epochs (0.1035 at epoch 80, 0.1032 at 100). A's and A′'s validation losses are not monotone: A has 0.0728 at epoch 20, 0.0826 at 35 and 0.0732 at 100; A′ has 0.0751 at 15, 0.0882 at 55 and 0.0756 at 100.
- **Restart:** the first launch of B and A (2026-10-11 14:03:19) ran 2 epochs at ≈ 465 s/epoch because of CPU oversubscription. It was stopped, the two run directories were deleted and the runs relaunched at 14:21:22 with single-threaded BLAS. The logs above hold only that second launch (Observation 7).
- **Manual launch:** the controller launched A′ at 2026-10-12 00:54:33 (pid 646601; the MANUAL line is logged at 00:54:44), because the driver could not (Observation 7).

## Milestone trajectory (quick grid, 200 cubes × 2 seeds, RMSE %)
Epochs 25 and 50 use the quick grid. Epoch 100 is the final full-grid evaluation: the same 200 cubes, seeds and samplers, plus the DDPM rows. The y-swap passes if true ≤ 0.9 × swap_full at both t.

| run | epoch | grid | ddim1 | ddim5 | ddim10 | ddim20 | warm50_ddim | warm100_ddim | y-swap | true/swap_full t = 100 | true/swap_full t = 300 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B | 25 | quick | 2.2873 | 2.9312 | 3.2166 | 3.4030 | 2.3949 | 2.4955 | FAIL | 0.998 | 0.986 |
| B | 50 | quick | 2.2615 | 2.8291 | 3.0664 | 3.2158 | 2.3919 | 2.4854 | FAIL | 0.997 | 0.985 |
| B | 100 | full | 2.2426 | 2.8379 | 3.0861 | 3.2485 | 2.3896 | 2.4778 | FAIL | 0.995 | 0.985 |
| A | 25 | quick | 13.6210 | 25.3026 | 27.7101 | 28.8310 | 7.0828 | 11.9078 | FAIL | 0.984 | 0.886 |
| A | 50 | quick | 14.2770 | 25.9249 | 28.3918 | 29.5418 | 7.0894 | 11.9600 | FAIL | 0.984 | 0.890 |
| A | 100 | full | 13.8290 | 25.3543 | 27.8522 | 28.9891 | 6.9338 | 11.8005 | FAIL | 0.985 | 0.885 |
| A′ | 25 | quick | 9.4774 | 23.4957 | 26.2876 | 27.6267 | 7.3750 | 12.4089 | FAIL | 0.952 | 0.815 |
| A′ | 50 | quick | 9.2443 | 24.1834 | 27.1421 | 28.5565 | 7.8326 | 12.9987 | FAIL | 0.960 | 0.843 |
| A′ | 100 | full | 8.1405 | 22.4523 | 25.3730 | 26.6951 | 7.0798 | 12.0899 | FAIL | 0.919 | 0.755 |

## Final full grid (RMSE % ± SE on the 200-cube subset; baseline 2.3512 ± 0.0529; DDPM-1000 on 50 cubes, baseline 2.4280)
All rows are at epoch 100 with seeds [0, 1]. The baseline (x̂₀ itself, 0 NFE) has SAM 1.767°. The three JSONs carry the same baseline on the same files.

| row | NFE | cubes | B | A | A′ | B − baseline (paired) | SAM ° B | sec/cube B |
|---|---|---|---|---|---|---|---|---|
| ddim1 | 1 | 200 | 2.2426 ± 0.0507 | 13.8290 ± 0.2167 | 8.1405 ± 0.0728 | −0.1086 ± 0.0060 | 1.766 | 0.0271 |
| ddim2 | 2 | 200 | 2.3471 ± 0.0489 | 16.1564 ± 0.2180 | 11.0790 ± 0.0426 | −0.0040 ± 0.0076 | 2.138 | 0.0528 |
| ddim5 | 5 | 200 | 2.8379 ± 0.0432 | 25.3543 ± 0.1660 | 22.4523 ± 0.0355 | +0.4867 ± 0.0137 | 3.375 | 0.1305 |
| ddim10 | 10 | 200 | 3.0861 ± 0.0413 | 27.8522 ± 0.1530 | 25.3730 ± 0.0298 | +0.7349 ± 0.0168 | 3.888 | 0.2612 |
| ddim20 | 20 | 200 | 3.2485 ± 0.0409 | 28.9891 ± 0.1449 | 26.6951 ± 0.0267 | +0.8973 ± 0.0178 | 4.182 | 0.5228 |
| warm50_ddim | 10 | 200 | 2.3896 ± 0.0521 | 6.9338 ± 0.0254 | 7.0798 ± 0.0239 | +0.0384 ± 0.0009 | 1.956 | 0.2616 |
| warm100_ddim | 10 | 200 | 2.4778 ± 0.0503 | 11.8005 ± 0.0283 | 12.0899 ± 0.0255 | +0.1266 ± 0.0029 | 2.294 | 0.2613 |
| warm200_ddim | 10 | 200 | 2.7221 ± 0.0460 | 19.0225 ± 0.0346 | 19.4819 ± 0.0308 | +0.3709 ± 0.0074 | 3.021 | 0.2614 |
| warm400_ddim | 10 | 200 | 3.0744 ± 0.0409 | 24.8741 ± 0.0299 | 25.3295 ± 0.0238 | +0.7232 ± 0.0127 | 3.885 | 0.2613 |
| warm50_ddpm | 50 | 200 | 2.4230 ± 0.0514 | 9.1210 ± 0.0265 | 9.3117 ± 0.0240 | +0.0718 ± 0.0017 | 2.093 | 1.3042 |
| warm100_ddpm | 100 | 200 | 2.5813 ± 0.0483 | 15.2932 ± 0.0333 | 15.6602 ± 0.0296 | +0.2302 ± 0.0050 | 2.624 | 2.6067 |
| warm200_ddpm | 200 | 200 | 2.9482 ± 0.0427 | 22.4649 ± 0.0266 | 22.9088 ± 0.0222 | +0.5970 ± 0.0108 | 3.584 | 5.2116 |
| warm400_ddpm | 400 | 200 | 3.2813 ± 0.0384 | 26.3068 ± 0.0301 | 26.5496 ± 0.0198 | +0.9301 ± 0.0153 | 4.341 | 10.4343 |
| ddpm1000 | 1000 | 50 | 3.4016 ± 0.0628 | 28.7238 ± 0.2056 | 26.8702 ± 0.0506 | +0.9736 ± 0.0369 | 4.551 | 26.2444 |

`sec/cube` is the GPU sampling time per cube per seed at eval batch 8. A's and A′'s sec/cube are within 2 % of B's on every row.

Clustered SEs (CR1 over the 113 source images) for the paired B − baseline column: ddim1 0.0068, ddim10 0.0187.

### Per split (B at its selected setting ddim1 and at the pre-registered ddim10; 200-cube subset)
| split | cubes | baseline | B ddim1 | B ddim1 − baseline (paired) | B ddim10 | B ddim10 − baseline (paired) |
|---|---|---|---|---|---|---|
| all | 200 | 2.3512 ± 0.0529 | 2.2426 ± 0.0507 | −0.1086 ± 0.0060 | 3.0861 ± 0.0413 | +0.7349 ± 0.0168 |
| P022 (seen) | 24 | 2.0699 ± 0.0825 | 1.9541 ± 0.0836 | −0.1158 ± 0.0125 | 2.8475 ± 0.0615 | +0.7776 ± 0.0337 |
| P023 | 152 | 2.2545 ± 0.0546 | 2.1633 ± 0.0527 | −0.0911 ± 0.0064 | 3.0319 ± 0.0436 | +0.7774 ± 0.0174 |
| P024 | 24 | 3.2449 ± 0.1746 | 3.0333 ± 0.1736 | −0.2116 ± 0.0143 | 3.6679 ± 0.1488 | +0.4230 ± 0.0432 |
| unseen (P023 + P024) | 176 | 2.3895 ± 0.0585 | 2.2820 ± 0.0559 | −0.1076 ± 0.0066 | 3.1186 ± 0.0457 | +0.7291 ± 0.0185 |

P023 ddim1 clustered SE 0.0073 (paired column, 80 source images).

For reference, the Phase 0 baseline on all 1000 validation files is 2.3129 ± 0.0230 (P022 2.1257 / P023 2.2168 / P024 3.1257). At ddim1 B's SAM matches the baseline's (1.766° vs 1.767°): the RMSE gain is not accompanied by a better spectral angle.

### y-swap (epoch 100; one-step x̂₀ error, RMSE %, on 8 subset cubes: 1 P022, 6 P023, 1 P024)
| run | t | true | swap_full | swap_sensor_only | zero | true / swap_full | true / swap_sensor_only | true ≤ 0.9 × swap_full? |
|---|---|---|---|---|---|---|---|---|
| B | 100 | 0.7274 | 0.7311 | 0.7288 | 0.8280 | 0.995 | 0.998 | no |
| B | 300 | 1.2419 | 1.2611 | 1.2531 | 2.6221 | 0.985 | 0.991 | no |
| A | 100 | 10.7913 | 10.9608 | 10.9608 | 13.2363 | 0.985 | 0.985 | no |
| A | 300 | 16.5354 | 18.6766 | 18.6766 | 32.7836 | 0.885 | 0.885 | yes |
| A′ | 100 | 11.1615 | 12.1416 | 11.3260 | 13.8208 | 0.919 | 0.985 | no |
| A′ | 300 | 16.9891 | 22.5144 | 19.0976 | 32.3993 | 0.755 | 0.890 | yes |

`passes` is false for all three arms, and was false at epochs 25 and 50 as well. A takes no x̂₀ input, so its swap_full and swap_sensor_only coincide by construction. Every variant is composed with the true x̂₀ (`yswap_check`), so the score isolates what the network does with its inputs. Under the linear β schedule, ᾱ_t is 0.8951 at t = 100 and 0.3940 at t = 300: the noisy input still carries the target with amplitude 0.9461 and 0.6277 against noise 0.3238 and 0.7785.

### Speed–quality (`ddim10_minus_ddpm1000` from gate_1a__residual_d4_b64.json, paired on the 50-cube intersection)
| run | DDIM-10 − DDPM-1000 | n | in SE |
|---|---|---|---|
| B | −0.2621 ± 0.0189 | 50 | −13.9 |
| A | −1.0695 ± 0.1374 | 50 | −7.8 |
| A′ | −1.5727 ± 0.0506 | 50 | −31.1 |

B's sec/cube per row is the last column of the full-grid table. One NFE costs 0.0271 s/cube and DDPM-1000 costs 26.2444 s/cube. For B the cheapest setting is also the most accurate, as RMSE predicts: ddim1 is a near-deterministic point estimate, and every sampled row pays for its spread (Observation 1).

## Gate 1a
Printed by `scripts/gate_1a.py` in the driver's gate step (`runs/pilot_driver.log`). Re-running the pilot's gate code (unchanged from 1ce9edf to a9bb5c9) from the worktree against the same three JSONs (written to a scratch file, not over the official one) printed the same lines, and its JSON equalled `gate_1a__residual_d4_b64.json` except for the `inputs` paths.

```
gate 1a (<= 10 NFE, k = 2.0, DONE required): INVALID
  epochs  B=100  A=100  A'=100
  B          ddim1 nfe=  1 RMSE 2.2426 %   checkpoint_epoch_100.pth
  A    warm50_ddim nfe= 10 RMSE 6.9338 %   checkpoint_epoch_100.pth
  A'   warm50_ddim nfe= 10 RMSE 7.0798 %   checkpoint_epoch_100.pth
  baseline (200 cubes) 2.3512 %   y-swap B FAIL
  B - baseline  -0.1086 ± 0.0060 (n=200)  pass
  B - A         -4.6911 ± 0.0308 (n=200)  pass  [ablation y-swap failed]
  B - A'        -4.8372 ± 0.0324 (n=200)  pass  [ablation y-swap failed]
pre-registered ddim10:  RMSE B 3.0861 %  A 27.8522 %  A' 25.3730 %
  B - baseline  +0.7349 ± 0.0168 (n=200)  fail
  B - A         -24.7661 ± 0.1366 (n=200)  pass  [ablation y-swap failed]
  B - A'        -22.2869 ± 0.0374 (n=200)  pass  [ablation y-swap failed]
```

Post-review re-run with the fix-wave gate, written to the new file `gate_1a__residual_d4_b64__postreview.json` (the official file is unchanged, sha256 `75a4e286…`). It adds the INVALID banner, cluster-robust SEs and the `ablation worse than baseline` flag, and it stores `verdict_if_valid` = PASS and `invalid_reason` (t100 0.99493, t300 0.98477, threshold 0.9). Every difference and cube SE is unchanged:

```
gate 1a (<= 10 NFE, k = 2.0, DONE required): INVALID
INVALID: B y-swap failed (t100 0.995, t300 0.985 > 0.9); comparisons below are NOT a pass (the comparisons alone would read PASS)
  epochs  B=100  A=100  A'=100
  B          ddim1 nfe=  1 RMSE 2.2426 %   checkpoint_epoch_100.pth
  A    warm50_ddim nfe= 10 RMSE 6.9338 %   checkpoint_epoch_100.pth
  A'   warm50_ddim nfe= 10 RMSE 7.0798 %   checkpoint_epoch_100.pth
  baseline (200 cubes) 2.3512 %   y-swap B FAIL
  B - baseline  -0.1086 ± 0.0060 (n=200, clustered ± 0.0068 over 113 images)  below threshold
  B - A         -4.6911 ± 0.0308 (n=200, clustered ± 0.0366 over 113 images)  below threshold  [ablation y-swap failed]  [ablation worse than baseline]
  B - A'        -4.8372 ± 0.0324 (n=200, clustered ± 0.0385 over 113 images)  below threshold  [ablation y-swap failed]  [ablation worse than baseline]
pre-registered ddim10:  RMSE B 3.0861 %  A 27.8522 %  A' 25.3730 %
  B - baseline  +0.7349 ± 0.0168 (n=200, clustered ± 0.0187 over 113 images)  fail
  B - A         -24.7661 ± 0.1366 (n=200, clustered ± 0.1813 over 113 images)  below threshold  [ablation y-swap failed]  [ablation worse than baseline]
  B - A'        -22.2869 ± 0.0374 (n=200, clustered ± 0.0414 over 113 images)  below threshold  [ablation y-swap failed]  [ablation worse than baseline]
```

**Verdict: INVALID.** The gate returns INVALID whenever B's y-swap fails, and here that follows mechanically from the spec §7 threshold (true ≤ 0.9 × swap_full at t = 100 and 300). Spec §11's underlying signal, a full-condition swap that changes nothing, is not met: B's swaps change its one-step error significantly, and in the right direction, at every t (Observation 2). The comparisons are reported, not acted on. Without the y-swap condition, the selected-setting comparisons would have read PASS (the post-review record's `verdict_if_valid`): all three differences are negative by more than 2 SE. But A and A′ are not working comparators at this budget (best rows ≈ 3× the baseline; DDPM-1000/DDIM-1 ratios 2.08 and 3.29 (both rows on the 50 DDPM cubes), against √2 for calibrated sampling), so the B − A and B − A′ conjuncts carry no information about the residual parameterisation. At the pre-registered ddim10, B is worse than the baseline. Applying B's setting (ddim1) to the ablations gives B − A = −11.5864 ± 0.1928 and B − A′ = −5.8979 ± 0.0642 (`b_setting_applied_to`).

The selected settings (B ddim1, A and A′ warm50_ddim) are selected from the grid (spec §7); ddim10 is the pre-registered setting. Each arm is one training seed. Cube-level SEs, clustered or not, say nothing about training-seed variance, so any decisive margin should be confirmed with a second training seed before 1b. B's selected margin over the baseline is 18.1 cube SE (15.9 SE clustered by source image; cluster-bootstrap 95 % CI [−0.1223, −0.0956]; 186/200 cubes and 109/113 images improve), but it is a single seed and the run is INVALID.

## Observations
1. **B gets worse with every sampling step.**
   - The DDIM ladder rises monotonically: ddim1 2.2426 → ddim2 2.3471 → ddim5 2.8379 → ddim10 3.0861 → ddim20 3.2485. DDPM-1000 reaches 3.4016 on the 50 cubes, whose baseline is 2.4280.
   - The warm starts order the same way: the earlier the start, the worse (warm50_ddim 2.3896 → warm400_ddim 3.0744; warm50_ddpm 2.4230 → warm400_ddpm 3.2813). No setting with more than one NFE beats the baseline: ddim2 ties it (−0.0040 ± 0.0076) and every other setting is worse.
   - DDPM-1000 over the 50-cube baseline is 1.401 (ratio of means; mean per-cube ratio 1.446; pooled-RMS ratio, i.e. √ of the pooled-MSE ratio, 1.378 against the baseline and 1.438 against ddim1).
   - *Interpretation:* the √2 level alone does not show unconditional sampling. Under RMSE every sampled output pays for its spread (E‖x̂ − x‖² = ‖E x̂ − x‖² + tr Cov x̂), so even a calibrated posterior sampler scores √2 × its posterior mean's error. The inputs explain only ≈ 9 % of the residual energy, so on these 50 cubes the two predictions differ by just 4.6 %: √2 × baseline predicts 3.434 (obs − pred −0.032 ± 0.063) and √2 × ddim1 predicts 3.282 (+0.120 ± 0.059). A unit-RMS independent draw predicts 3.569 (−0.167 ± 0.029) and is rejected. These three ± are clustered over the 50 cubes' 45 source images.
   - The per-cube structure is informative. The chain adds an almost cube-independent error energy rather than doubling each cube's: obs² ≈ 5.39 + 1.03·base² (DDPM-1000, 50 cubes), 4.98 + 0.97·base² (DDIM-20, 200 cubes) and 5.09 + 0.98·base² (warm400_ddpm). The added energy over the one-step estimate, 4.3–6.1 pct² (mean obs² − ddim1², DDIM-10, DDIM-20, warm400_ddpm and DDPM-1000; 3.8–5.6 pct² over base²), does not grow with cube difficulty, although condition-only features predict ≈ 70 % of the per-cube baseline energy (leave-one-image-out). The spread is a plausible residual, not this cube's. Per-seed RMSE does not settle whether the full-noise chain's centre tracks the one-step estimate: obs² loads ≈ 2.2–2.6 on the part of ddim1² that base² does not explain (≈ 1.7–2.2 when the chain and ddim1 come from different noise seeds), while warm starts show almost no such loading (≤ 0.2). The K-sample-mean diagnostic (Decision, step 0) settles it.
   - Consequence: with ρ² ≈ 9 %, even a perfectly calibrated conditional chain would score ≈ √(2(1 − ρ²)) ≈ 1.35 × the baseline RMSE. A sample beats x̂₀ in RMSE only if the inputs explain more than 50 % of the residual energy, so better or more sampling steps cannot by themselves beat the one-step read-out under this metric.
   - *Interpretation, continued:* one-step DDIM is one network call at t = 999 (diffusion_trainer.py:313-317) and returns the network's point estimate of r given its inputs. Training draws t uniformly and weights the per-sample L1 on v by min(SNR, 5)/(SNR + 1) (diffusion_trainer.py:151, 195-204). At t = 999 that weight is 4.0e-5, which is 4.8e-5 of its peak (0.83 at t = 130). In residual units (|Δv| = |Δr|/√(1 − ᾱ)), t = 999 gets 2.0e-5 of the peak, and all of t ≥ 500 carries 1.8 % of the total weight (3.5 % in v units). The estimate also depends on the noise draw, although x_T carries signal amplitude only 0.0064: the two seeds differ by 0.033 points per cube on average (max 0.150), about a third of the 0.109 gain. As an L1 (median) fit scored by RMSE, it is not guaranteed to beat r = 0 cube by cube: 14 of 200 cubes are worse than the baseline, by up to 3.4 %. The 4.6 % gain (ρ² ≈ 9 % of the residual energy) is therefore a lower bound on what (x̂₀, y_d) predict, not a measure of it.
2. **What the y-swap does and does not show for B.**
   - The check is deterministic: x_t is fixed by seed 1234 for every variant (eval_warmstart.py:236-237; 223-224 at a9bb5c9) and the model is in eval mode. A condition-blind network therefore scores exactly 1.000, as A's identical swap_full and swap_sensor_only columns show. B scores 0.995 (t = 100) and 0.985 (t = 300), and the true condition beats swap_full on 8/8 cubes at both t (per-cube values from the CPU run below). B's output depends on which cube's condition it receives.
   - At t = 100 and 300 the noisy input still carries the residual (signal amplitude 0.9461 and 0.6277), so a swap can move the error only by what the condition adds beyond r_t. A shrinkage model calibrated on B's own ddim1 error predicts 0.994 and 0.984, which is what B shows. Under a linear-Gaussian approximation, the ratio reaches 0.9 at t = 100 only if the condition explains ≈ 41 % of the residual energy (a one-step gain of ≈ 23 %; ≈ 8 % at t = 300). The ratio cannot fall below √(v/(2 − v)) ≈ 0.914 at any t when v ≈ 0.91 of the residual energy is unexplained, and B's one-step gain is 4.6 % (v from that 200-cube gain; on the y-swap cubes B's t = 999 gain is larger, 6.3 % on the gate's 8 cubes and 7.1 % on the 32 below, so v ≈ 0.88 and 0.86 and the floor ≈ 0.885 and 0.871: the measured t = 999 ratios 0.913 and 0.910 are consistent with it). These thresholds are model-based approximations. The conclusion holds regardless: the check has no power at B's gain size. (Corrected 2026-10-16, two bullets down: the larger y-swap-cube gains came from a leak in the t = 999 y-swap, not from the cubes.)
   - *Post-hoc CPU check at t = 999*, where B's 1-NFE gain is produced (`scripts/yswap_one_step.py`, seed 1234, same composition with the true x̂₀). Command: `CUDA_VISIBLE_DEVICES= python scripts/yswap_one_step.py --checkpoint <RESULTS_ROOT>/runs/residual_d4_b64/checkpoint_epoch_100.pth --mode residual --t 100 300 999 --n 8` (and `--n 32`) `--out <scratch file>`, run in the fix wave at 6002c26 and 1c60541, whose y-swap code is identical. It reproduces the harness's t = 100 values to 0.0005 (0.7269 / 0.7305 on CPU against 0.7274 / 0.7311 on GPU). On the gate's 8 cubes: true 2.1409, swap_full 2.3453 (ratio 0.913), swap_sensor_only 2.1541, zero 3.5448, baseline 2.2841. On 32 stratified subset cubes (29 source images): true 2.2479, swap_full 2.4704, swap_sensor_only 2.2662, zero 3.7364, baseline 2.4194 (ratio 0.910). Paired per cube: swap_full − true +0.2225 ± 0.0136; true − baseline −0.1715 ± 0.0197; swap_full − baseline +0.0509 ± 0.0149; swap_sensor_only − true +0.0183 ± 0.0037 (clustered by source image: 0.0141 / 0.0189 / 0.0140 / 0.0037). On the same 32 cubes the full-swap effect is also significant at t = 100 (+0.0061 ± 0.0008) and t = 300 (+0.0226 ± 0.0036), and the true condition beats swap_full on 32/32 cubes at every t.
   - *Correction (2026-10-16, Phase 1a′ final review).* The t = 999 values in the two bullets above came from x_t = q_sample(true residual, 999) = 0.00635·r + 0.99998·ε, which leaks the residual to B; B's real one-step call (ddim1) starts from pure noise. With x_t = ε (`yswap_one_step.py --x_t noise`, added in 5739753; record `diagnostics/yswap_residual_d4_b64_t999_n32__xT.json`, same seed 1234) the 32 cubes give true 2.3128, swap_full 2.5022, swap_sensor_only 2.3299, zero 3.8452, baseline 2.4194 (ratio 0.924). Paired per cube: swap_full − true +0.1895 ± 0.0134; true − baseline −0.1067 ± 0.0139; swap_full − baseline +0.0828 ± 0.0121; swap_sensor_only − true +0.0171 ± 0.0039 (clustered 0.0140 / 0.0135 / 0.0113 / 0.0039). The true condition still beats swap_full on 32/32 cubes. B's t = 999 gain on these cubes is 4.4 %, not 7.1 %, consistent with the 200-cube 4.6 % and with B's stored ddim1 on the same cubes (2.3156, 4.3 %). On the gate's 8 cubes B's stored ddim1 gains 2.8 % (2.2199 against 2.2841), not 6.3 %; no leak-free 8-cube y-swap was run. So the larger y-swap-cube gains came from the leak, not from the cube selection. With v = (1 − 0.044)² the floor is ≈ 0.917, and the leak-free 0.924 is consistent with it: the check still has no power at B's gain size. The t = 100 and 300 values are unaffected (q_sample is the right x_t there). Details: `2026-10-12-residual-warm-start-phase1a-prime-RESULTS.md`.
   - B's 1-NFE gain is therefore cube-specific. It disappears under a swapped condition, leaving B worse than the baseline, which rules out a cube-independent mean correction. It comes mainly through the x̂₀ channels; the sensor path is used only weakly. Even at t = 999 the spec's 0.9 rule would fail (0.910; leak-free 0.924, corrected 2026-10-16).
   - Zeroing both inputs hurts all three arms (t = 300: ×2.11 B, ×1.98 A, ×1.91 A′). A zero condition is out of distribution, though, and does not by itself show cube-specific use.
3. **Spec §11 risk checks.**
   - *B copies x̂₀?* No. The paired gain at ddim1 is −0.1086 ± 0.0060 (18.1 cube SE, 15.9 clustered); at ddim2 it is ≈ 0 (−0.0040 ± 0.0076); every other setting is worse than the baseline. σ_d is the Phase 0 value (unit-RMS test passes), and the training loss fell below its start (0.1232 → 0.1060). But the gain is tiny: 4.62 % relative. The Phase 1b target, 1.9985 % (0.85 × 2.3512), needs 15 %. B's selected setting is 0.2441 points above it.
   - *Network ignores the sensor?* Not the full condition. B's one-step output is better with the true condition at t = 100, 300 and 999 on 8/8 cubes, and at t = 999 its gain vanishes under a full swap (Observation 2). The sensor-only path is weakly used (ratios 0.998 / 0.991 at t = 100 / 300; paired +0.0183 ± 0.0037 at t = 999), which spec §11 accepts for B because x̂₀ carries y. (*Correction 2026-10-16:* the t = 999 values here, 8/8 cubes and +0.0183 ± 0.0037, came from the leaky x_t of Observation 2; leak-free, the true condition beats swap_full on 32/32 cubes and the sensor-only swap costs +0.0171 ± 0.0039.) B's y-swap FAIL, at every milestone, comes from the 10 % threshold, which B cannot reach at a 4.6 % gain. A′ does use its x̂₀ channels at t = 300: true 16.9891, sensor-only swap 19.0976, full swap 22.5144 (ratio 0.755). It fails at t = 100 (0.919). A passes at t = 300 (0.885) and fails at t = 100 (0.985).
   - *Gain only on the seen class?* No. B at ddim1 gains −0.1158 ± 0.0125 on P022 and −0.1076 ± 0.0066 on the unseen classes (difference −0.0083 ± 0.0142). The hardest class, P024, gains the most: −0.2116 ± 0.0143, 6.5 % of its baseline.
   - *Few steps lose to many?* The opposite: DDIM-10 beats DDPM-1000 by 0.2621 ± 0.0189 and DDIM-1 is the best row. The risk's follow-up (a shifting chain to rescue few-step sampling) does not apply, because more chain makes B worse; under RMSE that is expected of any stochastic sampler (Observation 1), so the comparison says nothing about chain quality.
4. **A and A′ are far from the baseline at base 64 / 100 epochs.** Both predict the whole cube from scratch.
   - At 1 NFE, A′ reaches 8.1405 and A 13.8290. The aligned x̂₀ channels help from noise (paired A′ − A at ddim1 −5.69 ± 0.26) but hurt at every warm start (warm50_ddim +0.146 ± 0.003, warm100_ddim +0.289 ± 0.006, warm50_ddpm +0.191 ± 0.005; these four SEs are clustered by source image). A′ is badly underfit: copying its own x̂₀ input would score 2.35 %, and its DDPM-1000/DDIM-1 ratio is 3.29 (both rows on the 50 DDPM cubes; A 2.08, paired the same way), against √2 for calibrated sampling. Their best ≤ 10-NFE rows are 7.0798 and 6.9338 (warm50_ddim). That is ≈ 3× the baseline, even though at t₀ = 50 their input is x̂₀ with noise of amplitude 0.17.
   - Their validation losses are not monotone, and A's quick-grid ddim1 got worse from epoch 25 to 50 (13.6210 → 14.2770).
   - B starts from x̂₀ and only moves it by σ_d-scaled residuals, so an imperfect network costs B fractions of a point, while A and A′ must produce the whole cube. B's 4.7–24.8-point margins over A and A′ therefore measure the residual parameterisation's head start, not sampling quality. At this budget neither ablation is a working conditional sampler, so the only licensed conclusion is about optimisation: at base 64 / 100 epochs only the residual target fits.
5. **Compute.**
   - Training: 9.05 / 9.14 / 7.71 h for B / A / A′. With two runs training side by side, the median is 312.0–314.8 s/epoch. A single run takes ≈ 256–261 s/epoch: A′'s median is 260.6 overall and 255.9 after 06:23, and A's epoch 100 after B finished took 256.9 s. Phase 0 timed a lone run at 256.5. The loader wait is ≤ 1.2 % in every epoch (median 0.74–0.84 %). The epochs that ran entirely during a quick evaluation on the same GPU took 453–475 s. The single-GPU record in `timing_phase1.json` (0.2704 s/step, 306 s/epoch; Task 3) was measured without the thread limit and is ≈ 19 % high. With the limit, a lone base-64 run takes 0.227 s/step, equal to Phase 0's 0.228. The FSDP records were taken under torchrun, which sets OMP_NUM_THREADS=1, and stand.
   - Evaluations: quick evaluations took 26.5–27.5 min; full-grid evaluations took 3.153 / 3.129 / 3.121 h (B / A / A′). Of the full grid's ≈ 3.1 h of sampling, the warm*_ddpm rows take 69.5 % and DDPM-1000 23.3 %. The nine rows with ≤ 20 NFE take 809–816 s.
   - The eval batch probe was flat: 0.2457–0.2503 s/cube at batch 8–64, ddim10. Per-cube time does not fall with batch size, so the GPU is already throughput-saturated at batch 8, and larger eval batches would not shorten evaluations.
   - Whole pilot: relaunch 2026-10-11 14:21:22 → gate 2026-10-12 11:58:29 = 21.62 h. From the first launch at 14:03:19 it was 21.92 h.
6. **Phase 1b sizing** (`timing_phase1.json`, `timing.json`).
   - At batch 4/GPU (FSDP, 2 GPUs), the base-256 model needs 0.9227 s/step and 1038.9 s/epoch: 57.72 h for 200 epochs, peaking at 11.58 GB.
   - At batch 8/GPU (Phase 0) it needs 0.8823 s/step and 496.3 s/epoch: 27.57 h, peaking at 16.09 GB of 20 GB.
   - The step time barely changes, so the user's "4/GPU for now" choice doubles the wall-clock (×2.09) and halves the global batch (8 vs 16). With no EMA in the plan, the ≈ 4.5 GB it frees is not used by anything.
   - Either way, the run is above the 24 h cap the pilot used (`launch_run.sh` passes `--max_hours 24`; since the fix wave the trainer's own default is 0, no cap).
7. **Deviations from the plan** (timestamps from `pilot_events.log`).
   - **Restart.** The first launch (14:03:19) ran ≈ 465 s/epoch, against ≈ 312 s/epoch side by side after the fix (the 306 s/epoch reference from Task 3 was itself oversubscribed; Observation 5), because of CPU oversubscription: 16 loader workers × ≈ 34 BLAS threads on 32 cores. Load was ≈ 50 and the GPUs were 45–50 % busy, while the loader wait was only 1 %. The driver and both runs were stopped after 2 epochs and the two run directories deleted. `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1` was added to `launch_run.sh` (7da6b93). `RESTART` was logged at 14:20:55 and the runs relaunched at 14:21:22. The same limits were later added to `eval_milestone.sh` (2fd2bee).
   - The run logs hold only the second launch. Its epoch-1 losses reproduce the first launch to 3 digits (B 0.12311 → 0.12322, A 0.11053 → 0.11051; Task 6 report, part 1b). The worker count stayed at 8 per run.
   - **Manual A′ launch.** At the 23:29:22 pass A had not yet written DONE: it was finishing epoch 100 (checkpoint 23:29:44.71, DONE 23:29:44.85) and still held GPU 1, so the policy (pilot_driver.py:96 in the pilot's code, :121 now, launches A′ only after A's DONE) could not launch A′. The same pass started B's full evaluation on GPU 0. Actions run synchronously by design (plan Task 5), so the driver did not look again until 02:48:33. GPU 1 sat idle for 1.41 h until the controller launched A′ by hand at 00:54:33 (launches/pid mtime; the `MANUAL` line is logged at 00:54:44, pid 646601). Without that launch the driver would have started A′ at 02:48:33, so the manual launch gained ≈ 1.9 h. The driver was inside synchronous evaluations for 5.78 h of A′'s 7.71 h of training, and a crash in those windows would have waited up to ≈ 3 h for a relaunch; none occurred. The driver then ran A′'s quick evaluations (02:48:33, 06:33:47) and full evaluation (08:41:11) by itself.
   - No crash relaunches (`launches` = 1 for each run), no skipped steps, and no wall-clock cap. The quick evaluations shared their own run's GPU as planned (plan deviation 5).
   - **Execution location** (controller ruling Finding A, not one of the six approved deviations; the plan's Global Constraint said master): Tasks 3 and 6 ran from the worktree on the unmerged branch worktree-phase1-plan, not from master in the main checkout. Training ran at 7da6b93 (inferred from the launch time; args.json records no git head) and the evaluations at 2fd2bee, with all artefacts under the absolute RESULTS_ROOT. Both commits are on origin/worktree-phase1-plan.
8. **B vs the 1b threshold.** At its selected setting (ddim1), B scores 2.2426 % against 1.9985 %: short by 0.2441 points, at 4.6 % below the baseline where 15 % is required. At the pre-registered ddim10 it is above the baseline.

## Decision for the user

The verdict is **INVALID** by the spec §7 rule, but it does not show that B ignores its condition. At t = 999, where B's one-step gain is produced, a full-condition swap removes the gain (Observation 2). Two things limit B:
- the size of the gain: 4.6 %, ≈ 9 % of the residual energy, and a lower bound because training gives t ≥ 500 about 2 % of the loss weight (Observation 1);
- the metric: under RMSE any stochastic chain loses to its own point estimate.

**Recommendation (planner):** do **not** scale B as-is to base 256 (Phase 1b: 57.7 h at batch 4/GPU, 27.6 h at batch 8/GPU). DDIM-10 is worse than the baseline, and the best gain is under a third of the 15 % target. In order:

0. **Diagnostics on the existing B epoch-100 checkpoint** (≈ 1.6 h of GPU, no training; corrected 2026-10-16 from "minutes of GPU": DDPM-1000 K = 4 on 50 cubes is 1.46 GPU-hours, and the three shards ran in sequence on one GPU, 1 h 37 min of wall-clock; see the Phase 1a′ note). These need a small script that keeps samples, because the harness stores only per-seed metrics:
   - K = 8 sample means of DDIM-10 and DDPM-1000 on the 50 DDPM cubes. A chain centred on r = 0 predicts ≈ 2.574 % and one centred on the one-step estimate ≈ 2.486 % (gap 0.088 ± 0.009); the H1 value used the single-draw ddim1 energy, which includes the one-step output's own noise energy, and the Phase 1a′ note replaces it with the prediction from the noise-free t = 999 centre, so this settles what the chain is centred on.
   - The per-cube distance between two seeds' samples, against the condition-predicted residual energy.
   - B's t = 999 output averaged over K = 8 noise draws. If the gain grows, the single-draw 4.6 % is depressed by noise sensitivity.
1. **A direct one-step residual regressor** x̂ = x̂₀ + σ_d·f(x̂₀, y_d). This is spec §11's fallback, and the template's FAIL path proposes the same thing, so the two are merged here.
   - Base 64, 100 epochs, same data and harness, evaluated at 1 NFE.
   - Trained at a single input with full weight, it also measures how much (x̂₀, y_d) actually predict.
   - Cost: ≈ 7.1 h alone on one GPU, or ≈ 8.7 h for two runs side by side. The 1-NFE evaluation takes seconds; if a diffusion arm is evaluated alongside, the ≤ 20-NFE rows take ≈ 14 min and the full grid ≈ 3.1 h.
   - **Question for you:** spec §4.3 fixes an L1 loss, but the metric is RMSE and L1 targets the median. Approve L2 for the regressor, or train both?
   - Go criterion for 1b, stated in advance: ≤ 1.9985 % (15 % below the 2.3512 % baseline) on the 200-cube subset at ≤ 10 NFE, confirmed with a second training seed, with the re-specified validity check (step 3) passing. Otherwise stop.
2. **If diffusion is kept**, treat loss weighting as an alternative explanation to the conditioning architecture: the weighting or t-sampling should give t ≥ 500 a real share of the loss. Position-aware sensor conditioning is demoted for B. The x̂₀ channels already carry y_d at aligned positions (A·K has rank 30), and spec §11 accepts an unused sensor path for B.
3. **Re-specify the y-swap and the 1b gate before any re-run.** This changes binding spec §7, §8 and §11, so it needs your approval.
   - As written, a B that exactly meets the 15 % target with full use of its condition is predicted at true/swap_full ≈ 0.95 at t = 100, and would be ruled INVALID. Passing at t = 100 needs ≈ 1.80 %, not 1.9985 %. Even at t = 999 the 0.9 ratio needs a gain above ≈ 5.4 %.
   - Proposed rule for residual models: at the first-call t of the gate-selected sampler (999 for ddim1), on ≥ 32 cubes with per-cube values, require swap_full − true > 2 paired SE, and require the gain over the baseline to vanish under the swap. Keep the t = 100/300 ratios as reported diagnostics.
   - **Question for you:** state whether 1b scores the one-step read-out or a K-sample mean at matched NFE. A calibrated sampled output reaches 1.9985 % only if its posterior-mean RMSE is ≤ 1.9985/√2 ≈ 1.41 %, which is ≈ 40 % below the baseline.

If a full-size run does follow, batch 8/GPU halves its wall-clock (27.6 h vs 57.7 h) and fits without EMA (16.1 GB). If it is launched through `launch_run.sh`, which passes `--max_hours 24`, it must pass a cap above its budget (or 0, no cap, the trainer's default since the fix wave).
