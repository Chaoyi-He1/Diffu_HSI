# Phase 2a — Regressor-Centred Residual Diffusion for Calibrated Uncertainty: Results

Spec: `docs/superpowers/specs/2026-10-22-regressor-centred-uq-diffusion-design.md` (approved 2026-10-22, commit 162c724; as-built note appended by Task 10, commit a36979a). Plan: `docs/superpowers/plans/2026-10-22-regressor-centred-uq-diffusion.md` (commit 850ca6a; the controller's rulings R1–R13 are listed in the plan's as-built block). Build phase (Tasks 1–10) complete 2026-10-22: 429 tests green, CPU smoke green (branch `worktree-phase1-plan`, HEAD a36979a).

Conventions: every number below is on the 200 gate cubes of `tests/data/val_subset_200.json` unless stated; errors in % of the [0, 1] range; CRPS in %; "± " is the iid SE over cubes with the source-image-clustered SE in parentheses where a paired difference is reported; K = 32 samples per cube everywhere (ruling R10 withdrew the spec's K = 16 at d = 4). No d = 8 gate number is looked at before Task 15's single verdict.

## Stage 1 — split and half regressors (d = 8)

- Split (2026-10-22, `scripts/uq/split_halves.py`, seed 0): `results/residual_warmstart/uq/split_halves.json`, sha256 `977cf0d08a1977b6622b5fe920f640c1bfb5bb6ed5739e84455cb6409e3ab421`; 4516 + 4484 training files; images per class within one of each other in all 22 classes (e.g. P001 56/55, P002 53/53, P016 39/39, P022 47/46).
- `regress_l1_half1_d8_b64`: launched 2026-10-22 ≈ 20:56 on GPU 1 (4516 files, 564 steps per epoch, 40 epochs, monitoring every epoch, every epoch saved); DONE = 40 at 22:44 (1.85 h, 155 s/epoch, 0 skipped steps). **Pre-registered selection = epoch 15** (monitoring loss 0.48565; epoch 40 = 0.51020). Pruned to epochs 15 and 40.
- `regress_l1_half2_d8_b64`: launched 2026-10-22 ≈ 21:40 on GPU 0 after Stage 0 finished (4484 files, 561 steps per epoch); DONE = 40 at 23:17 (1.83 h, 119 s/epoch, 0 skipped steps). **Pre-registered selection = epoch 13** (monitoring loss 0.48473; epoch 40 = 0.51470). Pruned to epochs 13 and 40.
- Both halves select early (15 and 13) and both drift up by ≈ 5–6 % of the monitoring loss by epoch 40, the same overfitting shape as the full-data runs (whose minima were at epoch 11 at d = 8 and 17 at d = 4).

## Stage 1 — the centre cache at d = 8 (2026-10-22 23:17–23:23, GPU 0)

`scripts/uq/build_xr_cache.py --d 8 --half1 regress_l1_half1_d8_b64@15 --half2 regress_l1_half2_d8_b64@13 --full regress_l1_sel_d8_b64@11` → `results/residual_warmstart/uq/d8/xr_cache/` (5.0 GB; 9 000 training cubes centred by the opposite half's regressor in 300 s, 1 000 test-split cubes by the full regressor in 338 s; prior sha `1ad9108f…`).

| quantity | value |
|---|---|
| σ_R (RMS of x − x̂_R over the 9 000 cross-fitted training cubes) | **0.089342** |
| σ_full (RMS of x − x̂_R with the full regressor over the 800 monitoring cubes) | 0.086718 |
| σ_full / σ_R | 0.9706 (inside [0.9, 1.1], not flagged) |
| σ_d at d = 8 (the linear baseline's residual scale) for reference | 0.095731 |

Reading: the half-data regressors leave residuals 3 % wider than the full regressor does on unseen cubes, an acceptable cross-fit mismatch; the regressor shrinks the baseline's residual scale only from 0.0957 to 0.087–0.089 (≈ 7–9 %), the same picture as its 7.6 % RMSE gain on the gate.

## Stage 1 — the residual diffusion and the variance head at d = 8

- `residual_reg_d8_b64`: launched 2026-10-22 23:23 on GPU 0 (100 epochs, v-prediction, L1, centre x̂_R from the cache, σ_R = 0.089342); DONE = 100 at 08:21 on 2026-10-23 (8.97 h, 318 s/epoch, 0 skipped steps). Monitoring loss (fixed-RNG v-prediction L1, every 5 epochs) fell monotonically: 0.1124 (epoch 20), 0.1082 (40), 0.1047 (60), 0.1033 (80), 0.1030 (100) — the diffusion does not overfit the way the one-call regressors do, so the pre-registered epoch-100 checkpoint is evaluated. Checkpoints kept: 25, 50, 90, 100.
- `varhead_d8_b64`: launched 2026-10-22 23:23 on GPU 1 (40 epochs, Gaussian NLL, monitoring every epoch, every epoch saved); DONE = 40 at 02:58 on 2026-10-23 (3.58 h, 297 s/epoch, 0 skipped steps). **Pre-registered selection = epoch 10** (monitoring NLL −0.2391; epoch 40 = −0.1262, i.e. the variance head overfits after epoch 10 as the regressors do). Pruned to epochs 10 and 40.
- `regress_l1_sel_s43_d8_b64` (the optional second seed for the ensemble baseline, the B.5 recipe at d = 8): launched 2026-10-23 ≈ 03:00 on GPU 1 once the variance head freed it; DONE = 100 at 12:02 (9.05 h). **Selection = epoch 12** (monitoring loss 0.48175; epoch 100 = 0.54190), next to seed 42's epoch 11 (0.47862). Pruned to 12 / 25 / 50 / 100. The d = 8 ensemble baseline is therefore `regress_l1_sel_d8_b64@11` + `regress_l1_sel_s43_d8_b64@12` (two members; its predictive std has very few degrees of freedom, so it is a reported comparison only, as the spec says).
- Stage 1 evaluation (T15): started 2026-10-23 17:12 on GPU 1 — `uq_eval.py --d 8 --name d8_main`, centre `regress_l1_sel_d8_b64@11`, diffusion `residual_reg_d8_b64@100`, variance head `varhead_d8_b64@10`, ensemble as above, half-centre sensitivity row with `regress_l1_half1_d8_b64@15` + `regress_l1_half2_d8_b64@13`, cache `uq/d8/xr_cache`, K = 32, DDIM-10, followed by the single `uq_verdict.py` call.

## Stage 1 verdict (d = 8) — taken once, 2026-10-23 18:44

Record `results/residual_warmstart/uq/d8/eval/d8_main.json` (200 gate cubes, K = 32 DDIM-10 samples, centre `regress_l1_sel_d8_b64@11`, `coverage_exact` true; 8 stored cubes from the y-swap batch in `d8_main__cubes.npz`); verdict `results/residual_warmstart/uq/d8/verdict__d8_main.json`, printed verbatim:

```
C1: FAIL  +0.1720 ± 0.0058 (cluster 0.0068)   [mean(CRPS_diffusion - CRPS_varhead) + 2 SE_cluster < 0]
C2: FAIL  cov90 0.908 cov50 0.674   [0.85 <= cov90 <= 0.95 and 0.45 <= cov50 <= 0.55]
P1: PASS  -0.3124 ± 0.0525 (cluster 0.0651)   [mean(|hf_single| - |hf_regressor|) + 2 SE_cluster < 0]
P2: FAIL  +1.1695 ± 0.0394 (cluster 0.0442)   [mean(RMSE_single - RMSE_baseline) + 2 SE_cluster < 0]
S1: FAIL  +0.0517 ± 0.0042 (cluster 0.0046)   [|mean(RMSE_meanK - RMSE_regressor)| <= 2 SE_cluster]
S2: FAIL  +0.0125 ± 0.0114   [mean(RMSE_swap_full - RMSE_true) - 2 SE_iid > 0 on the y-swap cubes]
calibration FAIL; perceptual FAIL; diffusion wins (a): False
```

| method (d = 8, gate) | RMSE % | log HF ratio | SAM ° | CRPS % | cov50 | cov90 | AUSE | spread ratio |
|---|---|---|---|---|---|---|---|---|
| baseline x̂₀ | 4.5799 | −3.266 | 2.94 | — | — | — | — | — |
| regressor x̂_R (centre) | 4.2319 | −3.195 | 2.84 | — | — | — | — | — |
| variance head (Gaussian around x̂_R) | 4.2319 | −3.195 | 2.84 | **1.7604** | 0.624 | 0.930 | **0.179** | 1.48 |
| residual diffusion, mean of 32 | 4.2836 | −0.490 | 3.13 | 1.9325 | 0.674 | 0.908 | 0.375 | 1.05 |
| residual diffusion, single sample | 5.7493 | +2.882 | 6.71 | | | | | |
| 2-member ensemble (Gaussian) | 4.1905 | −3.246 | 2.82 | 2.1449 | 0.080 | 0.186 | 0.446 | 0.04 |
| diffusion around the half regressors' average | 4.2824 (mean) | −0.493 | 3.13 | 1.9360 | 0.673 | 0.907 | 0.377 | 1.06 |

**Result: the pre-registered success criterion of part (a) is not met at d = 8.** The variance head is the better uncertainty model: lower CRPS on 193 of 200 cubes, and a per-pixel spread that ranks the errors twice as well (AUSE 0.18 vs 0.37). The diffusion beats only the degenerate two-member ensemble.

### Verification (2026-10-23, CPU; 6 independent verifiers + a synthesis; scripts and the full structured result in `docs/analysis/phase2a_verify_d8/`, large intermediates in `results/residual_warmstart/uq/d8/verify_d8_intermediates/`)

Confirmed:
- Two independent recomputations (one numpy from scratch, one by a 5000-draw cluster bootstrap over the 113 source images) reproduce every clause bit for bit. Calibration, perceptual and "wins" pass in 0 of 5000 bootstrap draws; C1, C2 and P2 fail in all of them; C1 fails in every class (P022 +0.157, P023 +0.186, P024 +0.101).
- Provenance is clean: every checkpoint, the cache meta, the prior and the split re-hash to the record; σ_R = 0.089342 everywhere; the epochs are the selection rule's (centre 11, seed 43 at 12, halves 15 / 13, variance head 10, diffusion 100); the cache is truly cross-fitted.
- C1 compares like with like (same scale, centre and elements; the fair CRPS of 32 draws from the variance head's Gaussian is 1.8641 against its closed form 1.8640 on the stored cubes; the only asymmetry, the [−1, 1] clamp on diffusion samples, favours the diffusion).

Mechanism (diagnostic evidence, not gate evidence):
- **The diffusion's predictive spread is nearly homoscedastic.** Across cubes, log spread rises with log error at slope 0.13 (variance head 0.72). Across the 64 sensor phases the spread varies 1.15× while the error varies 14×; at the measured grid pixels the spread is 3.6 % RMS against an error of 0.43 %. As a predictive distribution it is statistically indistinguishable from one constant-width Gaussian N(x̂_R, (0.5 σ_R)²): CRPS 1.9325 vs 1.9472, difference −0.015 ± 0.010 (cluster). The variance head's whole advantage (0.187 over that Gaussian) is localisation of the spread, which the diffusion lacks.
- **The spread is white, high-frequency noise.** Lag-1 spatial autocorrelation 0.24–0.29 (true residual 0.87), adjacent-band correlation 0.29 (true residual 0.975), 61 % of its energy at high spatial frequency (true residual 10 %), coherence with the true residual ≈ 0 at every frequency.
- **Likely upstream cause: the low-noise denoiser is under-fit.** At t ≤ 299 the network's x₀ error is worse than a linear per-pixel spectral filter fitted on training cubes (e.g. 0.29 vs 0.16 r-units at t = 99), and in calm regions worse than predicting zero; it leaves ≈ 0.29 r-units of noise in every spectral direction. A post hoc hybrid on the 8 stored cubes that replaces the network below t = 399 by that filter removes ≈ 60 % of the C1 gap (+0.161 → +0.061), but the samples then become under-dispersed. More sampling steps do not help (DDIM-50 and DDPM-1000 on one cube are slightly worse).
- **The diffusion does use its condition**, but only to shape the spread, not the mean: under a swapped condition the std map follows the donor cube (Spearman 0.55 with the donor, 0.25 with its own), and the denoiser's x₀ error at t = 900 is 0.74 with the true condition, 0.93 swapped, 7.1 zeroed. The residual's conditional mean is ≈ 0 by construction (training-residual mean +0.0016 pct).

**Three clauses of my pre-registered success criterion could not have been passed by any well-calibrated sampler — a design flaw of the spec, recorded here as such:**
- **P2** (a single sample beats the baseline): a calibrated single draw has RMSE ≈ √2 × the regressor's (5.98 % here; observed 5.75 %), and the baseline/regressor ratio is 1.08 at d = 8 (1.20 at d = 4), below √2; passing needs a spread ratio below ≈ 0.18, which contradicts C2. So "diffusion wins (a)" was unreachable at either stride.
- **S1** (the mean of K samples stays on the regressor): with spread variance v the mean of K draws carries an extra v/K; a perfectly centred sampler with this spread is expected to show +0.058 at K = 32 (observed +0.052). Debiased, S1 is −0.006 ± 0.004 and would pass; the rule as written needs K ≈ 256.
- **S2** (y-swap): the samples are always composed with the TRUE centre and the residual's conditional mean is ≈ 0, so swapping the condition only changes the 1/K noise term; even an oracle-calibrated sampler would score +0.024 ± 0.019 and fail. S2 cannot detect condition use in this design.
- **P1's PASS is an artefact** of the symmetric |log ratio| statistic: the single sample over-sharpens on 200/200 cubes (17.9× the truth's Laplacian energy), the regressor blurs on 200/200 (1/24); gradient W1 (0.0080 vs 0.0054) and SAM (6.71° vs 2.84°) say the sample is worse; x̂_R plus white noise would pass P1 by a wider margin.

The calibration clauses C1 and C2 were attainable and failed on merit (C1 by 25 cluster SEs; C2's cov50 band is strict — the variance head itself has cov50 0.624 — but the diffusion's 0.674 comes from the homoscedastic spread: cov50 0.84 in the calmest decile, 0.33 in the most uncertain).

### Pre-registered fix round (spec §5.7) — predictions recorded before it runs

Ruling R15 (controller, 2026-10-23): the spec allows exactly one fix round when calibration fails; the plan (T15 step 3) runs it. (1) `d8_fix_k64`: K = 64 DDIM-10 on the gate, same thresholds, with the predictions below; (2) the DDPM-1000 arm is run at K = 8 on the 50-cube DDPM subset **report-only** (`uq_verdict --report_only`), because ruling R10 refuses a verdict at K < 19 and a K ≥ 19 DDPM run would cost ≈ 40 GPU-h for a gate it cannot plausibly change; it supplies the DDPM-1000 reference row that d8_main lacks. Both run only after the d = 4 trainings free a GPU. No other re-run.

Predictions for `d8_fix_k64` (from the mechanism above): C1 unchanged within ≈ 1 SE (≈ +0.17; the fair CRPS is K-invariant); cov50 ≈ 0.67 (FAIL); S1 literal ≈ +0.023 (still FAIL; a perfectly centred sampler would show +0.029); P2 ≈ +1.17 (FAIL); S2 still underpowered; P1 ≈ unchanged (PASS, artefact). Prediction for the DDPM-1000 row: CRPS no better than DDIM-10, spread at least as wide and still white.

**K = 64 arm result (2026-10-24 01:15, GPU 1, 2.5 h; record `uq/d8/eval/d8_fix_k64.json`, verdict `uq/d8/verdict__d8_fix_k64.json`) — every prediction held:**

```
C1: FAIL  +0.1733 ± 0.0057 (cluster 0.0066)
C2: FAIL  cov90 0.901 cov50 0.681
P1: PASS  -0.3124 ± 0.0525 (cluster 0.0651)
P2: FAIL  +1.1695 ± 0.0394 (cluster 0.0442)
S1: FAIL  +0.0246 ± 0.0037 (cluster 0.0039)
S2: FAIL  +0.0126 ± 0.0096
calibration FAIL; perceptual FAIL; diffusion wins (a): False
```

C1 moved by +0.0013 (K-invariant, as predicted); S1 halved from +0.0517 to +0.0246, which is the 1/K Monte Carlo term predicted (+0.023), not drift; the mean of 64 samples has RMSE 4.2565 (vs 4.2836 at K = 32 and the regressor's 4.2319) and log HF −1.10 (vs −0.49 at K = 32: the white noise averages away as 1/K); AUSE 0.346 (vs 0.375). P1 and P2 are identical to K = 32 because the single sample is draw 0 with the same seed. Under the spec, the fix round's K = 64 arm leaves part (a) failed at d = 8.

## Stage 2 — the d = 4 contrast (report only)

Same recipe at `--ds 4`; the full centre is `regress_l1_sel_d4_b64@17`; the same split JSON.

- `regress_l1_half1_d4_b64`: launched 2026-10-23 08:25 on GPU 0 (after the d = 8 diffusion finished); DONE = 40 at 10:21 (1.97 h, 155 s/epoch, 0 skipped). **Selection = epoch 12** (monitoring loss 0.43100; epoch 40 = 0.45764). Pruned to 12 and 40.
- `regress_l1_half2_d4_b64`: launched 2026-10-23 ≈ 10:25 on GPU 0; DONE = 40 at ≈ 12:15 (≈ 1.9 h). **Selection = epoch 20** (monitoring loss 0.43155; epoch 40 = 0.44948). Pruned to 20 and 40.
- d = 4 centre cache (2026-10-23 17:12–17:18, GPU 0; `half1@12`, `half2@20`, full `regress_l1_sel_d4_b64@17`) → `uq/d4/xr_cache/` (5.0 GB): **σ_R = 0.040619**, σ_full = 0.040775, ratio **1.0038** (not flagged); σ_d at d = 4 is 0.051661 for reference. The cross-fit mismatch is smaller than at d = 8 (0.3 % against 3 %).
- `residual_reg_d4_b64`: launched 2026-10-23 17:18 on GPU 0 (100 epochs, ≈ 9 h).
- `varhead_d4_b64`: launched 2026-10-23 18:46 on GPU 1; DONE = 40 at 22:45 (3.99 h, 0 skipped). **Selection = epoch 12** (monitoring NLL −0.2016; epoch 40 = −0.1703). Pruned to 12 and 40.
- The d = 8 fix round's K = 64 arm (`d8_fix_k64`) started on GPU 1 at ≈ 22:47.
- `residual_reg_d4_b64`: DONE = 100 at 01:39 on 2026-10-24 (8.33 h; monitoring loss 0.1107, plateaued over epochs 80–100).

### d = 4 contrast result (report only; 2026-10-24 03:44, GPU 0)

Record `uq/d4/eval/d4_contrast.json` (+ `__cubes.npz`), report `uq/d4/verdict__d4_contrast__report.json`; centre `regress_l1_sel_d4_b64@17`, diffusion `residual_reg_d4_b64@100`, variance head `varhead_d4_b64@12`, 3-member ensemble, B `residual_d4_b64@100`, half-centre row, K = 32 (R10), same 200 gate cubes in the same order as d = 8 (so every d4 − d8 difference is paired).

```
C1: FAIL  +0.0876 ± 0.0017 (cluster 0.0021)
C2: FAIL  cov90 0.918 cov50 0.666
P1: PASS  -0.4389 ± 0.0384 (cluster 0.0468)
P2: FAIL  +0.2701 ± 0.0244 (cluster 0.0268)
S1: FAIL  +0.0255 ± 0.0009 (cluster 0.0009)
S2: FAIL  +0.0035 ± 0.0020
calibration FAIL; perceptual FAIL; diffusion wins (a): False   (REPORT ONLY, no gate)
```

| method (d = 4, gate) | RMSE % | log HF ratio | CRPS % | cov50 | cov90 | AUSE | spread ratio |
|---|---|---|---|---|---|---|---|
| baseline x̂₀ | 2.3512 | −1.992 | — | — | — | — | — |
| regressor x̂_R (centre) | 1.9621 | −1.846 | — | — | — | — | — |
| variance head | 1.9621 | −1.846 | **0.8024** | 0.537 | 0.897 | **0.178** | 1.04 |
| residual diffusion, mean of 32 | 1.9876 | −1.204 | 0.8900 | 0.666 | 0.918 | 0.427 | 0.93 |
| residual diffusion, single sample | 2.6213 | +1.407 | | | | | |
| B (linear-centred residual diffusion), mean of 32 | 2.1450 | −1.155 | 0.9478 | 0.697 | 0.933 | 0.329 | 1.32 |
| 3-member ensemble | 1.9244 | −1.938 | 0.9689 | — | 0.241 | 0.319 | — |

Verification (2026-10-24; 2 verifiers + synthesis, scripts and structured result in `docs/analysis/phase2a_verify_d4/`): an independent numpy recomputation reproduces the report and the d = 8 verdict bit for bit. Findings:
- **The diffusion loses C1 on all 200 cubes** (7 went the other way at d = 8), in every class (P022 +0.061, P023 +0.093, P024 +0.082).
- **Dose-response — the spec's prediction holds only in absolute units.** The absolute CRPS gap halves (+0.172 → +0.088; paired d4 − d8 = −0.084 ± 0.007, cluster), but so does the error scale (regressor RMSE 4.23 → 1.96 %). Scale-free, the gap does not shrink: ratio of means 9.8 % ± 0.6 → 10.9 % ± 0.4 of the variance head's CRPS (+1.1 ± 0.5 pp), mean of per-cube ratios 12.4 % → 11.9 % (−0.5 ± 0.5 pp). The error ranking gets worse: AUSE gap +0.195 → +0.249 (paired +0.054 ± 0.005); the diffusion's ranking skill (1 − AUSE/AUSE_random) falls from 52 % to 45 % while the variance head's stays at 77 %.
- **The d = 8 mechanism repeats, if anything stronger:** cube-level spread slope 0.07 (0.14 at d = 8; variance head 0.82); across the 16 sensor phases the spread varies 1.16× while the error varies 8.7×; the spread is white and high-frequency (lag-1 0.14 vs the true residual's 0.70; adjacent-band correlation 0.2 vs 0.94); the diffusion's CRPS equals a constant-width Gaussian N(x̂_R, (0.5 σ_R)²) on all 200 cubes (−0.0017 ± 0.0029); a training-fitted linear spectral filter beats the denoiser at every t ≤ 299, with the crossover at t = 399 at both strides.
- **Texture:** a single sample's gradient-W1 penalty over the regressor falls from +47 % to +24 % and its over-sharpening from 17.9× to 4.1× the truth's high-frequency energy — partly because the true d = 4 residual is itself rougher (32 % of its spatial energy at high frequency vs 11 % at d = 8). The pre-registered P1 statistic moves the other way (−0.31 → −0.44), consistent with P1 being an artefact.
- **The spec-design flaws repeat:** S1's literal fail is the 1/K term (debiased +0.0011 ± 0.0007, a pass); P2 is unattainable for a calibrated sampler (baseline/regressor ratio 1.198 < √2).
- **The variance head becomes nearly calibrated at d = 4** (cov50 0.537 and cov90 0.897, inside both C2 bands; ranking skill 77 %).
- **B vs the regressor-centred diffusion separates centre from spread:** centring on x̂_R lowers the CRPS (diffusion − B = −0.058 ± 0.004) through the better centre, while B's spread ranks errors better (AUSE lower on 197/200 cubes; B's spread slope 0.19). Re-centring the residual on the regressor removed the little error localisation the linear-centred diffusion had.

## Stage 0 — harness validation on the existing d = 4 assets (2026-10-22, GPU 0, ≈ 40 min)

Record `results/residual_warmstart/uq/d4/eval/stage0_d4_assets.json`: centre `regress_l1_sel_d4_b64@17`, ensemble of the three selected d = 4 checkpoints (`regress_l1_sel_d4_b64@17`, `regress_l1_sel_s43_d4_b64@17`, `regress_l2_sel_d4_b64@18`), B `residual_d4_b64@100`, K = 32, DDIM-10, batch 4. No `residual_reg` / `varhead` yet (decision D4).

| method | RMSE % | log HF ratio | CRPS % | cov90 | AUSE |
|---|---|---|---|---|---|
| baseline x̂₀ | **2.3512** | −1.992 | — | — | — |
| regressor x̂_R (centre) | **1.9621** | −1.846 | — | — | — |
| ensemble (3 members, Gaussian) | 1.9244 (member mean) | −1.938 | 0.969 | 0.241 | 0.319 |
| B, mean of 32 DDIM-10 draws | 2.1450 | −1.155 | 0.948 | 0.933 | 0.329 |
| B, single draw | 3.0884 | +1.803 | | | |

**Harness accepted:** the regressor's 1.9621 % and the baseline's 2.3512 % reproduce the B-track gate records to the last digit. Side observations (not gates): the three-member ensemble is sharp but badly under-dispersed (90 % intervals cover 24 % of elements) and its paired mean beats the single regressor; B's 32-draw mean (2.145 %) sits where the Phase 1a′ chain-mean estimate put it (2.11–2.15 %), its single draw is far worse (3.09 %) with 6 × the truth's high-frequency energy (log ratio +1.8: residual noise, not texture), its 90 % coverage is 0.93, and despite its worse mean it already beats the ensemble on CRPS (0.948 vs 0.969) — a calibrated-but-blunt sampler outscores a sharp-but-overconfident one under a proper scoring rule. Every log HF ratio of a mean estimate is strongly negative (−1.85 for the regressor: its texture energy is e^{−1.85} ≈ 16 % of the truth's), which is the blur the perceptual clause P1 is about.

No d = 8 number has been looked at.
