# Phase 1a results — 2026-10-12

**Gate 1a verdict: INVALID.** B fails the y-swap check, so by spec §11 the run is invalid and the comparison numbers are not acted on. B's best setting is one DDIM step: 2.2426 % RMSE against the baseline's 2.3512 %, a gain of 0.1086 ± 0.0060 (4.6 %). Every further sampling step makes B worse; at the pre-registered DDIM-10 it is 0.7349 ± 0.0168 above the baseline. Neither ablation comes within 4.5 points of the baseline.

Sources (all under `results/residual_warmstart/`, git-ignored): `runs/<run>/{train_log.jsonl,args.json,launches,DONE}`, `runs/pilot_events.log`, `runs/pilot_driver.log`, `eval/<mode>/d4/<run>__checkpoint_epoch_{25,50}__quick.json` and `…_epoch_100.json`, `eval/residual/d4/gate_1a__residual_d4_b64.json`, `timing_phase1.json`, `timing.json`. The Phase 0 record is `tests/data/baseline_phase0.json`. Every evaluation carries `meta_checked = true` and `git_head` 2fd2bee. RMSE is in % of each cube's range. "±" is one standard error over cubes. "Paired" means the per-cube difference against the baseline (or another arm) on the same cubes, averaged over the 2 noise seeds.

## Runs (d = 4, base 64, batch 8, 1 GPU each, 100 epochs or 24 h; no EMA)
| run | GPU | launched → DONE | epochs done | wall h | s/epoch median (data %) | loss epoch 1 → 100 | val loss (800 non-gate files) epoch 5 → 100 | peak GB | skipped steps | launches (relaunches) |
|---|---|---|---|---|---|---|---|---|---|---|
| B (residual) | 0 | 10-11 14:21:22 → 10-11 23:24:23 | 100 | 9.05 | 312.0 (0.77 %) | 0.1232 → 0.1060 | 0.1191 → 0.1032 | 3.37 | 0 | 1 (0) |
| A (standard) | 1 | 10-11 14:21:22 → 10-11 23:29:44 | 100 | 9.14 | 314.8 (0.74 %) | 0.1105 → 0.0747 | 0.0847 → 0.0732 | 3.35 | 0 | 1 (0) |
| A′ (standard_xhat) | 1 | 10-12 00:54:44 → 10-12 08:37:26 | 100 | 7.71 | 260.6 (0.84 %) | 0.1110 → 0.0774 | 0.0860 → 0.0756 | 3.37 | 0 | 1 (0) |

- `wall h` is `elapsed_hours` of epoch 100. `data %` is the median of `data_sec / sec` per epoch. The DONE times are the DONE files' mtimes. Parameters: B and A′ 58,942,272; A 58,905,408 (no x̂₀ input channels). All three runs use 8 loader workers, lr 1e-4 → 3.32e-6 (cosine to 0.033·lr), v-prediction, L1, min-SNR 5, seed 42.
- The 24 h cap never fired. Each run finished its full cosine schedule in 100 epochs.
- Losses are not comparable across arms: B's target is the unit-RMS residual (x − x̂₀)/σ_d (σ₄ = 0.051661), while A's and A′'s target is x. B's validation loss falls monotonically and is flat over the last 20 epochs (0.1035 at epoch 80, 0.1032 at 100). A's and A′'s validation losses are not monotone: A has 0.0728 at epoch 20, 0.0826 at 35 and 0.0732 at 100; A′ has 0.0751 at 15, 0.0882 at 55 and 0.0756 at 100.
- **Restart:** the first launch of B and A (2026-10-11 14:03:19) ran 2 epochs at ≈ 465 s/epoch because of CPU oversubscription. It was stopped, the two run directories were deleted and the runs relaunched at 14:21:22 with single-threaded BLAS. The logs above hold only that second launch (Observation 7).
- **Manual launch:** the controller launched A′ at 2026-10-12 00:54:44 (pid 646601), because the driver could not (Observation 7).

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

### Per split (B at its selected setting ddim1 and at the pre-registered ddim10; 200-cube subset)
| split | cubes | baseline | B ddim1 | B ddim1 − baseline (paired) | B ddim10 | B ddim10 − baseline (paired) |
|---|---|---|---|---|---|---|
| all | 200 | 2.3512 ± 0.0529 | 2.2426 ± 0.0507 | −0.1086 ± 0.0060 | 3.0861 ± 0.0413 | +0.7349 ± 0.0168 |
| P022 (seen) | 24 | 2.0699 ± 0.0825 | 1.9541 ± 0.0836 | −0.1158 ± 0.0125 | 2.8475 ± 0.0615 | +0.7776 ± 0.0337 |
| P023 | 152 | 2.2545 ± 0.0546 | 2.1633 ± 0.0527 | −0.0911 ± 0.0064 | 3.0319 ± 0.0436 | +0.7774 ± 0.0174 |
| P024 | 24 | 3.2449 ± 0.1746 | 3.0333 ± 0.1736 | −0.2116 ± 0.0143 | 3.6679 ± 0.1488 | +0.4230 ± 0.0432 |
| unseen (P023 + P024) | 176 | 2.3895 ± 0.0585 | 2.2820 ± 0.0559 | −0.1076 ± 0.0066 | 3.1186 ± 0.0457 | +0.7291 ± 0.0185 |

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

### Speed–quality (`ddim10_minus_ddpm1000` from gate_1a.json, paired on the 50-cube intersection)
| run | DDIM-10 − DDPM-1000 | n | in SE |
|---|---|---|---|
| B | −0.2621 ± 0.0189 | 50 | −13.9 |
| A | −1.0695 ± 0.1374 | 50 | −7.8 |
| A′ | −1.5727 ± 0.0506 | 50 | −31.1 |

B's sec/cube per row is the last column of the full-grid table. One NFE costs 0.0271 s/cube and DDPM-1000 costs 26.2444 s/cube. For B the cheapest setting is also the most accurate.

## Gate 1a
Printed by `scripts/gate_1a.py` in the driver's gate step (`runs/pilot_driver.log`). Re-running it from the worktree against the same three JSONs (written to a scratch file, not over the official one) prints the same lines, and its JSON equals `gate_1a__residual_d4_b64.json` except for the `inputs` paths.

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

**Verdict: INVALID.** The gate returns INVALID whenever B's y-swap fails. This follows spec §11: if the network ignores its condition (a full-condition swap changes nothing), "the run is invalid". The comparisons are reported, not acted on. Without the y-swap condition, the selected-setting comparisons would have read PASS: all three differences are negative by more than 2 SE. At the pre-registered ddim10, however, B is worse than the baseline. Applying B's setting (ddim1) to the ablations gives B − A = −11.5864 ± 0.1928 and B − A′ = −5.8979 ± 0.0642 (`b_setting_applied_to`).

The selected settings (B ddim1, A and A′ warm50_ddim) are selected from the grid (spec §7); ddim10 is the pre-registered setting. Each arm is one training seed. A decisive margin under ≈ 3 SE is within plausible training-seed variance and should be confirmed with a second seed before 1b. B's selected margin over the baseline is 18.1 SE, but it is a single seed and the run is INVALID.

## Observations
1. **B gets worse with every sampling step.**
   - The DDIM ladder rises monotonically: ddim1 2.2426 → ddim2 2.3471 → ddim5 2.8379 → ddim10 3.0861 → ddim20 3.2485. DDPM-1000 reaches 3.4016 on the 50 cubes, whose baseline is 2.4280.
   - The warm starts order the same way: the earlier the start, the worse (warm50_ddim 2.3896 → warm400_ddim 3.0744; warm50_ddpm 2.4230 → warm400_ddpm 3.2813). No setting with more than one NFE beats the baseline: ddim2 ties it (−0.0040 ± 0.0076) and every other setting is worse.
   - DDPM-1000 over the 50-cube baseline gives a ratio of 1.401 (ratio of the means; the mean per-cube ratio is 1.446). √2 = 1.414.
   - *Interpretation:* √2 is the signature of sampling from an essentially unconditional residual distribution of the right (unit-RMS) scale. The baseline's error is σ_d·r. A sample r̂ that is uncorrelated with r and has the same RMS gives an error σ_d·(r − r̂) with twice the energy, which is √2 × the baseline RMSE. The chain draws a plausible residual, not this cube's residual.
   - *Interpretation, continued:* one-step DDIM from t = 999 instead returns the network's point estimate of r given its inputs. At t = 999, v = √ᾱ·ε − √(1−ᾱ)·r ≈ −r, so the L1 loss makes it a conditional median. Predicting r = 0 (the baseline) is one of the candidates the median is chosen over, so such an estimate is at worst about the baseline, and it beats it by as much as the inputs predict the residual: here 4.6 %.
2. **What the y-swap does and does not show for B.**
   - At t = 100 and 300 the noisy input r_t still carries the residual (signal amplitude 0.9461 and 0.6277). The network can denoise it without the condition, and it does: true/swap_full is 0.995 and 0.985.
   - B's 1-NFE gain is produced at t = 999, where the noisy input's signal amplitude is 0.0064 and it carries nothing about the cube. There the gain must come from the conditioning inputs (x̂₀, y_d), or from a cube-independent mean correction. The y-swap at t = 100/300 cannot tell these apart.
   - Zeroing both inputs does hurt B (t = 300: 2.6221 vs 1.2419). The network reacts to an out-of-range input, but not to which cube's condition it receives.
   - *Follow-up (not done):* a y-swap at t = 999, i.e. on the one-step output, for residual models.
3. **Spec §11 risk checks.**
   - *B copies x̂₀?* No. The paired gain at ddim1 is −0.1086 ± 0.0060 (18.1 SE); at ddim2 it is ≈ 0 (−0.0040 ± 0.0076); every other setting is worse than the baseline. σ_d is the Phase 0 value (unit-RMS test passes), and the training loss fell below its start (0.1232 → 0.1060). But the gain is tiny: 4.62 % relative. The Phase 1b target, 1.9985 % (0.85 × 2.3512), needs 15 %. B's selected setting is 0.2441 points above it.
   - *Network ignores the sensor?* Yes for B at t ≤ 300: the full swap ratio is 0.995 / 0.985 and the sensor-only ratio is 0.998 / 0.991, failing at every milestone. A′ does use its x̂₀ channels at t = 300: true 16.9891, sensor-only swap 19.0976, full swap 22.5144 (ratio 0.755). It fails at t = 100 (0.919). A passes at t = 300 (0.885) and fails at t = 100 (0.985).
   - *Gain only on the seen class?* No. B at ddim1 gains −0.1158 ± 0.0125 on P022 and −0.1076 ± 0.0066 on the unseen classes (difference −0.0083 ± 0.0142). The hardest class, P024, gains the most: −0.2116 ± 0.0143, 6.5 % of its baseline.
   - *Few steps lose to many?* The opposite: DDIM-10 beats DDPM-1000 by 0.2621 ± 0.0189 and DDIM-1 is the best row. The risk's follow-up (a shifting chain to rescue few-step sampling) does not apply, because more chain makes B worse.
4. **A and A′ are far from the baseline at base 64 / 100 epochs.** Both predict the whole cube from scratch.
   - At 1 NFE, A′ reaches 8.1405 and A 13.8290: the aligned x̂₀ channels help. Their best ≤ 10-NFE rows are 7.0798 and 6.9338 (warm50_ddim). That is ≈ 3× the baseline, even though at t₀ = 50 their input is x̂₀ with noise of amplitude 0.17.
   - Their validation losses are not monotone, and A's quick-grid ddim1 got worse from epoch 25 to 50 (13.6210 → 14.2770).
   - B starts from x̂₀ and only moves it by σ_d-scaled residuals, so an imperfect network costs B fractions of a point, while A and A′ must produce the whole cube. B's 4.7–24.8-point margins over A and A′ therefore measure the residual parameterisation's head start, not sampling quality.
5. **Compute.**
   - Training: 9.05 / 9.14 / 7.71 h for B / A / A′. With two runs training side by side, the median is 312.0–314.8 s/epoch. A single run takes ≈ 256–261 s/epoch: A′'s median is 260.6 overall and 255.9 after 06:23, and A's epoch 100 after B finished took 256.9 s. Phase 0 timed a lone run at 256.5. The loader wait is < 1 % throughout. The epochs that ran entirely during a quick evaluation on the same GPU took ≈ 455–475 s.
   - Evaluations: quick evaluations took 26.5–27.5 min; full-grid evaluations took 3.153 / 3.129 / 3.121 h (B / A / A′). Of the full grid's ≈ 3.1 h of sampling, the warm*_ddpm rows take 69.5 % and DDPM-1000 23.3 %. The nine rows with ≤ 20 NFE take 809–816 s.
   - The eval batch probe was flat: 0.2457–0.2503 s/cube at batch 8–64, ddim10. Per-cube time does not fall with batch size, so the GPU is already throughput-saturated at batch 8, and larger eval batches would not shorten evaluations.
   - Whole pilot: relaunch 2026-10-11 14:21:22 → gate 2026-10-12 11:58:29 = 21.62 h. From the first launch at 14:03:19 it was 21.92 h.
6. **Phase 1b sizing** (`timing_phase1.json`, `timing.json`).
   - At batch 4/GPU (FSDP, 2 GPUs), the base-256 model needs 0.9227 s/step and 1038.9 s/epoch: 57.72 h for 200 epochs, peaking at 11.58 GB.
   - At batch 8/GPU (Phase 0) it needs 0.8823 s/step and 496.3 s/epoch: 27.57 h, peaking at 16.09 GB of 20 GB.
   - The step time barely changes, so the user's "4/GPU for now" choice doubles the wall-clock (×2.09) and halves the global batch (8 vs 16). With no EMA in the plan, the ≈ 4.5 GB it frees is not used by anything.
   - Either way, the run is above the 24 h `--max_hours` default.
7. **Deviations from the plan** (timestamps from `pilot_events.log`).
   - **Restart.** The first launch (14:03:19) ran ≈ 465 s/epoch, not 306, because of CPU oversubscription: 16 loader workers × ≈ 34 BLAS threads on 32 cores. Load was ≈ 50 and the GPUs were 45–50 % busy, while the loader wait was only 1 %. The driver and both runs were stopped after 2 epochs and the two run directories deleted. `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1` was added to `launch_run.sh` (7da6b93). `RESTART` was logged at 14:20:55 and the runs relaunched at 14:21:22. The same limits were later added to `eval_milestone.sh` (2fd2bee).
   - The run logs hold only the second launch. Its epoch-1 losses reproduce the first launch to 3 digits (B 0.12311 → 0.12322, A 0.11053 → 0.11051; Task 6 report, part 1b). The worker count stayed at 8 per run.
   - **Manual A′ launch.** A wrote DONE at 23:29:44. The driver pass at 23:29:22 still saw GPU 1 busy (A exiting), started B's full evaluation on GPU 0, and blocked in it until 02:38:33. The controller launched A′ by hand at 00:54:44 (`MANUAL` line, pid 646601). The driver then ran A′'s quick evaluations (02:48:33, 06:33:47) and full evaluation (08:41:11) by itself.
   - No crash relaunches (`launches` = 1 for each run), no skipped steps, and no wall-clock cap. The quick evaluations shared their own run's GPU as planned (plan deviation 5).
8. **B vs the 1b threshold.** At its selected setting (ddim1), B scores 2.2426 % against 1.9985 %: short by 0.2441 points, at 4.6 % below the baseline where 15 % is required. At the pre-registered ddim10 it is above the baseline.

## Decision for the user
The verdict is **INVALID**: the condition path is not used where the gate looks. The three paths:
- **Scale B as-is (Phase 1b: base 256, FSDP, 200 epochs; 57.7 h at batch 4/GPU, 27.6 h at batch 8/GPU).** Not supported by the evidence. The gate is INVALID, DDIM-10 is worse than the baseline, and the best pilot gain (4.6 % at 1 NFE) is under a third of the 15 % target.
- **Treat it as FAIL and diagnose** (the template's FAIL path: σ_d scaling, loss trajectory, direct x₀-prediction of the residual). σ_d and the loss trajectory check out (Observation 3), which leaves the spec's fallback: direct x₀-prediction of the residual.
- **Treat it as INVALID (the verdict): change the conditioning, re-run a pilot-size check, then decide on 1b.**

**Recommendation (planner):** do **not** scale B as-is to base 256. The sampling chain hurts, and B's network does not use its condition where the y-swap looks. The evidence favours:
1. a direct one-step residual regressor, x̂ = x̂₀ + σ_d·f(x̂₀, y_d) (spec §11's fallback), trained with the same data and harness and evaluated at 1 NFE;
2. fixing the conditioning path before any full-size diffusion run: position-aware sensor conditioning, e.g. concatenating the grid-aligned lifted y_d as input channels, or positional encodings in the cross-attention (position-blind since Phase 0). For B, the x̂₀ channels already carry y_d at aligned positions, so this mainly targets the sensor path itself;
3. if diffusion sampling is kept, re-specifying the y-swap/validity check for residual models (e.g. at t = 999 / one-step).

If a full-size run does follow, batch 8/GPU halves its wall-clock (27.6 h vs 57.7 h) and fits without EMA (16.1 GB).
