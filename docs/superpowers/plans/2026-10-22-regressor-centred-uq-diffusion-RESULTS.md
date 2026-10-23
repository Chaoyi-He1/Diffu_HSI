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
- Stage 1 evaluation (T15): started 2026-10-23 ≈ 12:20 on GPU 1 — `uq_eval.py --d 8 --name d8_main`, centre `regress_l1_sel_d8_b64@11`, diffusion `residual_reg_d8_b64@100`, variance head `varhead_d8_b64@10`, ensemble as above, half-centre sensitivity row with `regress_l1_half1_d8_b64@15` + `regress_l1_half2_d8_b64@13`, cache `uq/d8/xr_cache`, K = 32, DDIM-10, followed by the single `uq_verdict.py` call.

## Stage 2 — the d = 4 contrast (report only)

Same recipe at `--ds 4`; the full centre is `regress_l1_sel_d4_b64@17`; the same split JSON.

- `regress_l1_half1_d4_b64`: launched 2026-10-23 08:25 on GPU 0 (after the d = 8 diffusion finished); DONE = 40 at 10:21 (1.97 h, 155 s/epoch, 0 skipped). **Selection = epoch 12** (monitoring loss 0.43100; epoch 40 = 0.45764). Pruned to 12 and 40.
- `regress_l1_half2_d4_b64`: launched 2026-10-23 ≈ 10:25 on GPU 0; DONE = 40 at ≈ 12:15 (≈ 1.9 h). **Selection = epoch 20** (monitoring loss 0.43155; epoch 40 = 0.44948). Pruned to 20 and 40.
- d = 4 centre cache: building on GPU 0 since ≈ 12:20 (`half1@12`, `half2@20`, full `regress_l1_sel_d4_b64@17`); `residual_reg_d4_b64` launches on GPU 0 when it is done; `varhead_d4_b64` launches on GPU 1 after the d = 8 evaluation.

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
