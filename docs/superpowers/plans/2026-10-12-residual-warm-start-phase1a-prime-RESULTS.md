# Phase 1a′ results — 2026-10-16

**Verdicts:** both regressors pass the binding gate, and neither meets the 1.9985 % target. R L1 (seed 42) is **PASS** at 2.1422 % RMSE (1 NFE): 0.2090 ± 0.0234 below the baseline (8.9 %) and 0.1004 ± 0.0195 below B's ddim1. It does **not meet** the 1.9985 % target (+0.1437). R L2 (seed 42) is **PASS** at 2.1829 %: 0.1683 ± 0.0257 below the baseline (7.2 %) and 0.0597 ± 0.0220 below B's ddim1. It does **not meet** the target either (+0.1844). The best regressor is R L1 (L1 − L2 −0.0407 ± 0.0123). B under the proposed rule, with the leak-free t = 999 y-swap: **PASS** (valid; the spec 0.9 ratio, 0.924, fails as in Phase 1a). go_1b: **false**. Seed 43 was not run (Deviation 5: the best regressor passes but misses the target, so a second seed cannot make go_1b true).

Step 0 settles what Phase 1a left open. B's chain is conditional, but it is centred on neither r = 0 nor its one-step estimate: its K → ∞ mean is closer to the truth than the noise-free centre of B's one-step output (λ = 1.698 ± 0.022 for DDIM-10). A single chain draw is worse than the one-step read-out because it pays its spread, which is as large as its centre's error energy and barely depends on the cube. In error energy, the L1 regressor is level in one call with DDIM-10's chain centre, which needs K → ∞ draws (5.031 against 4.908 pct², difference +0.1225 ± 0.0866). Both regressors overfit at this size, and L2 more than L1.

Sources (all under `results/residual_warmstart/`, git-ignored):
- `diagnostics/b_diagnostics.json`, merged from the shards `diagnostics/b_fast.npz` (ddim10 and t999, draws 0–7), `b_ddpm_a.npz` (ddpm1000, draws 0 1) and `b_ddpm_b.npz` (ddpm1000, draws 2 3), with their `.log` files, `started_at` and `gpu0_chain.log`;
- `diagnostics/yswap_residual_d4_b64_t999_n32__xT.json`, B's leak-free t = 999 y-swap and the gate's `--b_yswap`, and `diagnostics/yswap_residual_d4_b64_t999_n32.json`, the leaky record kept as a labelled reference (see the correction below);
- `runs/regress_{l1,l2}_d4_b64/{train_log.jsonl,args.json,launches,DONE,run.log}`;
- `eval/regress/d4/regress_{l1,l2}_d4_b64__checkpoint_epoch_100__direct.json`;
- the binding gate record `eval/regress/d4/gate_1a_prime__regress_l1_d4_b64+regress_l2_d4_b64.json`; the early single-model record `eval/regress/d4/gate_1a_prime__regress_l1_d4_b64.json` stays as a reference;
- `timing_phase1.json` (entry `regress_d4_b64_bs8x1_w8`), `timing.json`, and B's Phase 1a evaluation `eval/residual/d4/residual_d4_b64__checkpoint_epoch_100.json`.

The controller's rulings are in `2026-10-12-residual-warm-start-phase1a-prime-LEDGER.md`, a copy of the SDD ledger. Both regressor evaluations carry `meta_checked = true` and `git_head` 5739753. RMSE is in % of each cube's range. "±" is one standard error over cubes (iid). Brackets give the cluster-robust SE (CR1 over source images) where it matters. For λ and the spread slopes the SE is a leave-one-image-out jackknife (the record's `se_cluster` key). "Mean RMSE" is the mean over cubes of the per-cube RMSE; the harness and the gate use it. Step 0 also uses per-cube error energies (RMSE², pct²) and the RMS energy √(mean energy). Each comparison names its aggregation.

## Correction to Phase 1a: B's t = 999 y-swap leaked the residual
- `scripts/yswap_one_step.py` built x_t = q_sample(true residual, 999) = 0.00635·r + 0.99998·ε for every variant (√ᾱ₉₉₉ = 0.006353). B's real one-step call (ddim1) starts from x_T = ε and carries no signal. B reads the leaked residual: with the same ε (seed 1234) and x_t = ε, the true condition scores 2.3128 instead of 2.2479 on the 32 y-swap cubes (leaky − leak-free per cube −0.0649 ± 0.0102, lower on 32/32 cubes). The final review found this. The fix wave 5739753 adds `--x_t noise`, and the gate refuses a `--b_yswap` record made without it.
- Leak-free on the 32 cubes (CPU, 257 s): true 2.3128, swap_full 2.5022, swap_sensor_only 2.3299, zero 3.8452, baseline 2.4194, true / swap_full 0.924. True − baseline is −0.1067 ± 0.0139 (4.4 %), not −0.1715 ± 0.0197 (7.1 %).
- B's stored ddim1, which starts from pure noise, scores 2.3156 on the same cubes (4.3 %; leak-free true − stored ddim1 −0.0028 ± 0.0051), and its 200-cube gain is 4.6 %. The larger y-swap-cube gain that Phase 1a attributed to the cubes came from the leak, not from the cube selection. On the gate's 8 y-swap cubes, B's stored ddim1 gains 2.8 % (2.2199 against 2.2841), not 6.3 %; no leak-free 8-cube y-swap was run.
- What holds: B passes the proposed rule with either record, and fails the spec 0.9 rule at t = 999 with either (leak-free 0.924; with v = (1 − 0.044)² the floor √(v/(2 − v)) is 0.917). The t = 100 and 300 checks are unaffected, since there q_sample is the sampler's own input distribution. The Phase 1a note carries dated corrections in Observations 2 and 3.

## Step 0 — what B's chain does (`b_diagnostics.json`)
Mean RMSE % over cubes. Row baselines: 2.3512 on the 200 cubes (B's stored ddim1 2.2426) and 2.4280 on the 50 DDPM cubes (stored ddim1 2.3206).

| row | cubes | K | single-draw RMSE | K-mean RMSE | K-mean − single (clustered) | pred H0 | pred H1 | λ vs one-step centre (± jk SE) | λ vs single-draw ddim1 (biased reference) | ratio_of_means / self_ratio | spread slope on b² ± jk SE (reference: one-step c1 / own centre c2) | draws 0/1 vs stored |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ddim10 | 200 | 8 | 3.0786 | 2.2566 | −0.8220 ± 0.0074 (0.0080) | 2.4584 | 2.3409 | 1.698 ± 0.022 | 2.160 ± 0.042 | 0.910 / 1.000 | 0.131 ± 0.031 (0.900 / 0.814) | 4.8e-7 / 9.5e-7 |
| t999 | 200 | 8 | 2.2405 | 2.2141 | −0.0265 ± 0.0006 (0.0007) | n/a (the record's 2.3378; o < b² on 187 of 200 cubes) | 2.2141 (= K-mean by construction) | 1 (by construction) | 1.272 ± 0.025 | 0.026 / 0.026 | 0.011 ± 0.003 (0.900 / 0.900) | 4.8e-7 / 4.8e-7 |
| ddpm1000 | 50 | 4 | 3.3966 | 2.5110 | −0.8856 ± 0.0112 (0.0117) | 2.7090 | 2.6142 | 2.055 ± 0.112 | 2.746 ± 0.181 | 1.258 / 1.448 | 0.263 ± 0.047 (0.907 / 0.753) | 4.8e-7 / 4.8e-7 |

Error energies (pct², means over the row's cubes). Per cube, o = c2 + spread exactly, and ebar = c2 + spread / K.

| row | cubes | b² (baseline) | o (single draw) | ebar (K-mean) | spread | c2 (chain centre) | c1 (one-step centre) | c2 / c1 | baseline energy removed by c1 / c2 | √(o / c2) |
|---|---|---|---|---|---|---|---|---|---|---|
| ddim10 | 200 | 6.085 | 9.815 | 5.522 | 4.907 | 4.908 | 5.392 | 0.910 | 11.4 % / 19.3 % | 1.414 |
| t999 | 200 | 6.085 | 5.531 | 5.410 | 0.139 | 5.392 | 5.392 | 1 | 11.4 % / 11.4 % | 1.013 |
| ddpm1000 | 50 | 6.203 | 11.731 | 6.526 | 6.940 | 4.791 | 5.516 | 0.869 | 11.1 % / 22.8 % | 1.565 |

Reading:
- λ = (b² − c2) / (b² − c1). λ ≈ 1 means the chain is centred on the noise-free one-step centre (a conditional posterior sampler built on B's one-step estimate); λ ≈ 0 means it samples an unconditional residual. λ > 1 means the chain's K → ∞ mean is closer to the truth than the one-step centre.
- pred H0 and pred H1 are the K-mean RMSEs each hypothesis predicts from the row's own single-draw energy. For t999, H0 would need a negative spread on 187 cubes (o < b²), so its value is not a prediction; its H1 equals the K-mean identically, because there c1 = c2.
- The spread is calibrated only if `ratio_of_means` (mean spread / mean c1, the plan's rule) ≈ 1 AND the per-cube slope of spread on b² is close to the reference slope of the one-step centre energy c1 on b². A sampler calibrated to its own centre would instead show `self_ratio` (mean spread / mean c2) ≈ 1 and the own-centre slope of c2 on b². A slope ≈ 0 means a cube-independent spread (Phase 1a Observation 1), whatever the ratio.
- Draws 0 and 1 of every row reproduce B's stored evaluation (max |difference| in RMSE %, tolerance 1e-3), so the diagnostics sample exactly the chain that Phase 1a scored.

## Regressor runs (d = 4, base 64, batch 8, one GPU each, 100 epochs; staggered, not side by side)
| run | GPU | launched → DONE | loss | seed | wall h | s/epoch median (data %) | loss epoch 1 → 100 | val loss (800 non-gate files) epoch 5 → 100 (minimum) | peak GB | skipped | launches (relaunches) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R L1 (`regress_l1_d4_b64`) | 0 | 10-15 21:15:08 → 10-16 04:48:57 | L1 | 42 | 7.56 | 261.9 (0.88 %) | 0.4671 → 0.2677 | 0.4290 → 0.4586 (0.4208 at epoch 15) | 3.34 | 0 | 1 (0) |
| R L2 (`regress_l2_d4_b64`) | 1 | 10-16 04:08:03 → 10-16 11:00:28 | L2 | 42 | 6.87 | 237.9 (0.91 %) | 0.7660 → 0.2129 | 0.6722 → 0.7796 (0.6503 at epoch 10) | 3.34 | 0 | 1 (0) |

- `wall h` is `elapsed_hours` of epoch 100. `data %` is the median of `data_sec / sec` per epoch. Launch times are the mtimes of the `launches` files (for L2 the same second as the auto-launcher's log line). DONE times are the DONE files' mtimes.
- Both runs (`args.json`, `run.log`): 58,942,272 parameters (B's architecture), mode regress, prediction_type direct (one call at t = 999 on a zero x_t), lr 1e-4 → 3.32e-6 (cosine to 0.033·lr), weight decay 0, grad clip 1.0, bf16, 8 loader workers, a 24 h cap that never fired. The two `args.json` differ only in loss_type, device and save_dir. L1 was launched with `RUN_NAME=regress_l1_d4_b64 bash scripts/pilot/launch_run.sh regress 0 --prediction_type direct --loss_type l1`. L2 was launched with `RUN_NAME=regress_l2_d4_b64 bash scripts/pilot/launch_run.sh regress 1 --prediction_type direct --loss_type l2` by Ruling 18's auto-launcher (the controller's `l2_auto_launch.sh`).
- Losses are in units of the unit-RMS residual (σ₄ = 0.051661). L1's loss is a mean absolute error and is not comparable with L2's squared error.
- **Both regressors overfit.** L1's training loss falls 0.4671 → 0.2677, but its validation loss is lowest at epoch 15 (0.4208) and rises at every check from epoch 35 (0.4285) to 100 (0.4586, 9.0 % above the minimum). L2's training loss falls 0.7660 → 0.2129, while its validation loss is lowest at epoch 10 (0.6503) and rises at every check from epoch 60 (0.7003) to 100 (0.7796, 19.9 % above the minimum). B's validation loss fell monotonically (Phase 1a). The gate scores the epoch-100 checkpoints, as pre-registered; the epoch-25 and 50 checkpoints are kept but were not evaluated.
- Per-epoch time, L1: the median, 261.9 s, is what Task 5's dry run predicted for a lone run (0.2317 s/step, 262.2 s/epoch, data wait 0.00019 s/step, peak 3.34 GB; `timing_phase1.json` entry 8). Epochs 2–12 (21:19–22:20) took 279–364 s while the final review ran CPU test suites and CPU training runs in scratch copies (its findings were written at 22:21; one lens noted that its niced CPU test run may have added contention). Epochs 93–100 took 295–306 s alongside L2 on GPU 1. Epochs 13–92 took 241–286 s.
- Per-epoch time, L2: epochs 1–9 (04:08–04:52) took 286–302 s alongside L1; epochs 10–100 took 237–241 s alone on GPU 1 (median 237.8 s). That is faster than L1's epochs 13–92 (median 252.1 s) and than the dry run's 262.2 s, so the machine's load differed between the runs. L1 and L2 per-epoch times are not comparable (Ruling 13).

## Comparison on the 200 gate cubes (mean RMSE %; paired per cube; ± iid SE, clustered SE in brackets)
| model | NFE | RMSE % | − baseline | − B ddim1 | − B t999 8-draw mean (report only) | P022 | unseen | target 1.9985 % |
|---|---|---|---|---|---|---|---|---|
| baseline | 0 | 2.3512 ± 0.0529 | — | — | — | 2.0699 | 2.3895 | — |
| B ddim1 | 1 | 2.2426 ± 0.0507 | −0.1086 ± 0.0060 (0.0068) | — | +0.0286 ± 0.0015 (0.0015) | 1.9541 | 2.2820 | not met |
| B t999 mean of K = 8 draws | 8 | 2.2141 ± 0.0505 | −0.1371 ± 0.0059 (0.0069) | −0.0286 ± 0.0015 (0.0015) | — | 1.9111 | 2.2554 | not met |
| R L1 | 1 | 2.1422 ± 0.0471 | −0.2090 ± 0.0234 (0.0244) | −0.1004 ± 0.0195 (0.0197) | −0.0718 ± 0.0196 (0.0196) | 1.7385 | 2.1973 | not met (+0.1437) |
| R L2 | 1 | 2.1829 ± 0.0503 | −0.1683 ± 0.0257 (0.0284) | −0.0597 ± 0.0220 (0.0239) | −0.0312 ± 0.0221 (0.0237) | 1.7049 | 2.2481 | not met (+0.1844) |
| R L1 seed 43 (if run) | 1 | not run (Deviation 5: best PASS, target not met) | — | — | — | — | — | — |
| R L2 seed 43 (if run) | 1 | not run (Deviation 5: best PASS, target not met) | — | — | — | — | — | — |

L1 − L2 (paired, seed 42): −0.0407 ± 0.0123 (0.0126) in mean RMSE; in error energy −0.2368 ± 0.0641 pct². Seed pairs: not run (Deviation 5); go_1b: false.

- B ddim1 is the mean of B's two stored seeds (draws 0 and 1). The B t999 8-draw mean averages draws 0–7 of the same one-step call per cube (`per_cube.kmean_rmse`), at 8 NFE. The R L1 and R L2 cells equal the binding gate record's `vs_baseline`, `vs_b_ddim1` and `vs_b_t999_kmean`. R L2's margin over B's 8-draw mean does not pass (−0.0312 + 2 × 0.0221 > 0); R L1's does.
- Each regressor's two evaluation seeds are identical (seed-std 0.0): the regressors are deterministic.
- The regressors' mean gains are the largest, but they are less uniform than B's. L1 is worse than the baseline on 53 of the 200 cubes (by up to 19.6 %), L2 on 65 (by up to 53.8 %), B's ddim1 on 14. L1's paired SE is 3.9× B's.

Per split (mean RMSE %; paired against the baseline; the regressors' clustered SEs in brackets):

| split | cubes | baseline | B ddim1 − baseline | R L1 | R L1 − baseline | R L1 gain | R L2 | R L2 − baseline | R L2 gain |
|---|---|---|---|---|---|---|---|---|---|
| all | 200 | 2.3512 ± 0.0529 | −0.1086 ± 0.0060 | 2.1422 ± 0.0471 | −0.2090 ± 0.0234 (0.0244) | 8.9 % | 2.1829 ± 0.0503 | −0.1683 ± 0.0257 (0.0284) | 7.2 % |
| P022 (seen) | 24 | 2.0699 ± 0.0825 | −0.1158 ± 0.0125 | 1.7385 ± 0.0643 | −0.3314 ± 0.0459 (0.0568) | 16.0 % | 1.7049 ± 0.0710 | −0.3650 ± 0.0518 (0.0632) | 17.6 % |
| P023 | 152 | 2.2545 ± 0.0546 | −0.0911 ± 0.0064 | 2.1181 ± 0.0525 | −0.1364 ± 0.0252 (0.0245) | 6.0 % | 2.1693 ± 0.0559 | −0.0852 ± 0.0279 (0.0296) | 3.8 % |
| P024 | 24 | 3.2449 ± 0.1746 | −0.2116 ± 0.0143 | 2.6986 ± 0.1440 | −0.5464 ± 0.0630 (0.0600) | 16.8 % | 2.7469 ± 0.1531 | −0.4980 ± 0.0651 (0.0610) | 15.3 % |
| unseen (P023 + P024) | 176 | 2.3895 ± 0.0585 | −0.1076 ± 0.0066 | 2.1973 ± 0.0515 | −0.1923 ± 0.0257 (0.0262) | 8.0 % | 2.2481 ± 0.0545 | −0.1415 ± 0.0277 (0.0298) | 5.9 % |

## Validity — proposed one-step y-swap rule (t = 999, 32 cubes from 29 source images)
Mean RMSE % on the 32 stratified y-swap cubes; paired per cube, ± iid SE (clustered). Valid means (a) swap_full − true > 2 SE and (b) swap_full − baseline + 2 SE ≥ 0. Every variant is composed with the true x̂₀.

| model | x_t at t = 999 | true | swap_full | swap_sensor_only | zero | baseline | swap_full − true | swap_full − baseline | true − baseline | valid (invalid_reason) | spec 0.9 ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B, leak-free (`…__xT.json`, the gate's input) | ε (pure noise) | 2.3128 | 2.5022 | 2.3299 | 3.8452 | 2.4194 | +0.1895 ± 0.0134 (0.0140) | +0.0828 ± 0.0121 (0.0113) | −0.1067 ± 0.0139 (0.0135) | valid | 0.924 (fails) |
| B, leaky reference (not used) | q_sample(r, 999) | 2.2479 | 2.4704 | 2.2662 | 3.7364 | 2.4194 | +0.2225 ± 0.0136 (0.0141) | +0.0509 ± 0.0149 (0.0140) | −0.1715 ± 0.0197 (0.0189) | valid | 0.910 (fails) |
| R L1 | zero placeholder | 2.1888 | 2.9678 | 2.2385 | 2.4624 | 2.4194 | +0.7789 ± 0.0695 (0.0723) | +0.5483 ± 0.0747 (0.0719) | −0.2306 ± 0.0554 (0.0548) | valid | 0.738 (passes) |
| R L2 | zero placeholder | 2.2665 | 3.0344 | 2.3276 | 2.4550 | 2.4194 | +0.7679 ± 0.0890 (0.0892) | +0.6150 ± 0.0802 (0.0776) | −0.1529 ± 0.0667 (0.0679) | valid | 0.747 (passes) |

- B's leak-free row equals the binding gate record's `b.validity`. The regressors always take a zero x_t; B's ddim1 starts from ε.
- Each regressor's y-swap `true` equals its direct row on these 32 cubes exactly (max |difference| 0.0 for both). For both, the true condition beats swap_full on 32/32 cubes, and swap_full is worse than the baseline on 32/32. Sensor-only swap over true: L1 +0.0497 ± 0.0143 (0.0149), L2 +0.0611 ± 0.0275 (0.0260). With both inputs zeroed (still composed with the true x̂₀), L1 scores +0.0429 ± 0.0037 and L2 +0.0356 ± 0.0034 over the baseline.

## Gate 1a′
The binding gate ran at 11:04 with `--reg` L1 and L2, `--b_yswap …__xT.json` and `--b_diag b_diagnostics.json` (DONE required). It wrote `eval/regress/d4/gate_1a_prime__regress_l1_d4_b64+regress_l2_d4_b64.json` and printed:

```
regress_l1_d4_b64 (l1, seed 42, epoch 100): PASS  RMSE 2.1422 ± 0.0471 %  target 1.9985 not met
   - baseline -0.2090 ± 0.0234 (clustered 0.0244)  - B ddim1 -0.1004 ± 0.0195 (clustered 0.0197)
   - B t999 mean of 8 draws (report only) -0.0718 ± 0.0196 (clustered 0.0196)
   validity at t = 999 on 32 cubes: swap_full - true +0.7789 ± 0.0695, swap_full - baseline +0.5483 ± 0.0747 -> valid (spec 0.9 ratio 0.738)
regress_l2_d4_b64 (l2, seed 42, epoch 100): PASS  RMSE 2.1829 ± 0.0503 %  target 1.9985 not met
   - baseline -0.1683 ± 0.0257 (clustered 0.0284)  - B ddim1 -0.0597 ± 0.0220 (clustered 0.0239)
   - B t999 mean of 8 draws (report only) -0.0312 ± 0.0221 (clustered 0.0237)
   validity at t = 999 on 32 cubes: swap_full - true +0.7679 ± 0.0890, swap_full - baseline +0.6150 ± 0.0802 -> valid (spec 0.9 ratio 0.747)
B ddim1 2.2426 %  - baseline -0.1086 ± 0.0060  under the proposed rule: PASS
loss_pair: regress_l1_d4_b64 - regress_l2_d4_b64 -0.0407 ± 0.0123
best regress_l1_d4_b64 (PASS); go_1b False
```

**Verdict: both regressors PASS; best R L1 (PASS); target met by neither; go_1b false.** The early single-model gate on L1 alone (04:53, `gate_1a_prime__regress_l1_d4_b64.json`) printed the same L1 and B lines and is kept only as a reference. Task 7 Step 5 was skipped by its own rule: `best_verdict` is PASS but the best model's `target_met` is false, so seed 43 cannot make go_1b true (Deviation 5).

## Observations
1. **Step 0: B's chain is conditional, and its centre beats its one-step output.**
   - *Centre.* λ is 1.698 ± 0.022 for DDIM-10 (200 cubes) and 2.055 ± 0.112 for DDPM-1000 (50 cubes); an unconditional residual sampler gives 0, a chain centred on the one-step centre 1. The chain's K → ∞ mean has 9.0 % (DDIM-10) and 13.1 % (DDPM-1000) less error energy than the noise-free centre of B's t = 999 output (c2 / c1 0.910 and 0.869). It removes 19.3 % and 22.8 % of the baseline's error energy, against 11.4 % and 11.1 % for the one-step centre. Neither hypothesis holds: in mean RMSE the 8-draw DDIM-10 mean scores 2.2566 against pred H0 2.4584 and pred H1 2.3409, and the 4-draw DDPM-1000 mean 2.5110 against 2.7090 and 2.6142.
   - The single-draw λ reference (2.160 and 2.746) is biased upwards: on the t999 row, whose true λ is 1, it reads 1.272 ± 0.025. Phase 1a's ρ² ≈ 9 % came from single-draw ddim1 RMSEs; the noise-free one-step centre removes 11.4 % of the mean error energy.
   - *Spread.* DDIM-10's mean spread equals its own centre's mean error energy (self_ratio 1.000), but it is 0.910 of the one-step centre's (`ratio_of_means`, the plan's calibration rule). Per cube it barely follows difficulty: its slope on b² is 0.131 ± 0.031, against 0.814 for its own centre energy and 0.900 for the one-step centre's. Its coefficient of variation over cubes is 0.15, against 0.68 for c2. DDPM-1000 is over-dispersed (self_ratio 1.448, `ratio_of_means` 1.258), with slope 0.263 ± 0.047 against 0.753 and 0.907. Both slopes are far below their own-centre slopes, so neither chain is calibrated cube by cube. The spread is nearly cube-independent, as Phase 1a Observation 1 inferred from per-seed RMSE.
   - *The √2 question.* A single draw's error energy is its centre's plus its spread: 9.815 = 4.908 + 4.907 pct² for DDIM-10. In RMS energy a single DDIM-10 draw is therefore √(9.815 / 4.908) = 1.414 × its own centre, the √2 of calibrated sampling, but only in aggregate (see Spread). DDPM-1000 has 11.731 = 4.791 + 6.940, a factor of 1.565. The DDIM-10 and DDPM-1000 losses of Phase 1a are this spread, paid on top of a centre that is better than the one-step read-out.
   - *B's one-step output barely depends on its noise draw.* Its spread is 0.139 pct², 2.6 % of its centre energy (ratio 0.026). The 8-draw mean (8 NFE) gains only 0.0265 ± 0.0006 in mean RMSE over a single draw (2.2141 against 2.2405).
   - *What the chain's centre costs.* DDIM-10's 8-draw mean (80 NFE) is +0.0161 ± 0.0072 worse than a single one-step draw in mean RMSE (2.2566 against 2.2405), but level in RMS energy (2.3498 against 2.3518; per-cube energy difference −0.0093 ± 0.0358 pct²). Extrapolating from the per-cube c2 and spread, the mean-RMSE break-even is at ≈ K = 9 (K = 10 gives 2.2279), and the chain's K → ∞ mean scores 2.1079. Against B's own 8-draw one-step mean, DDIM-10's 8-draw mean is +0.0426 ± 0.0071 worse (80 against 8 NFE).
2. **The one-step regressors beat B's one-step read-out; only L1 also beats B's 8-draw mean.**
   - Direct regression at t = 999 changes three things: the objective (a full-weight loss on the residual at one t, where B's training weighted t = 999 at 4.8e-5 of its peak weight; Phase 1a Observation 1), a single deterministic call, and a zero x_t in place of B's noise.
   - At 1 NFE, L1 scores 2.1422 %. It is 0.2090 ± 0.0234 below the baseline (8.9 %), 0.1004 ± 0.0195 below B's ddim1, and 0.0718 ± 0.0196 below B's 8-draw one-step mean (report only, 8 NFE). Since it also beats the 8-draw mean, L1's gain is credited to the objective, not to B's noise sensitivity, which costs B only 0.0265 (Observation 1).
   - L2 scores 2.1829 %: 0.1683 ± 0.0257 below the baseline (7.2 %) and 0.0597 ± 0.0220 below B's ddim1, but only 0.0312 ± 0.0221 below B's 8-draw mean, which does not pass. By the plan's reading, L2's margin over B's single draw is not separable from B's noise sensitivity (B's 8-draw mean is 0.0286 below its ddim1).
   - In error energy, L1 is level with B's DDIM-10 chain centre: 5.031 pct² against DDIM-10's c2 4.908 (L1 − c2 +0.1225 ± 0.0866 pct²; against the one-step centre −0.3615 ± 0.1082). It removes 17.3 % of the baseline's error energy (one-step centre 11.4 %, DDIM-10 centre 19.3 %). L2 (5.268 pct²) removes 13.4 %; it is +0.3593 ± 0.1031 above the chain centre and level with the one-step centre (−0.1247 ± 0.1194). The target is 15 % in mean RMSE, which is 28 % of the energy if every cube improves by the same factor.
   - Both regressors use their condition strongly. A full-condition swap costs +0.7789 ± 0.0695 (L1) and +0.7679 ± 0.0890 (L2), and leaves them 0.5483 ± 0.0747 and 0.6150 ± 0.0802 worse than the baseline, on 32/32 cubes each. Each regressor applies a confident, cube-specific correction, which is wrong for another cube's condition. The spec 0.9 rule passes as well (0.738 and 0.747, against B's 0.924). As in B, the sensor-only path is used weakly (+0.0497 ± 0.0143 and +0.0611 ± 0.0275).
   - *Interpretation:* both validation losses bottom early (epoch 15 for L1, 10 for L2) and end 9.0 % and 19.9 % higher, while the training losses keep falling. At this size the regressors look limited by generalisation from the 9000 training files, not by optimisation, and more epochs alone are unlikely to help.
3. **L1 vs L2 under RMSE: L1 is better.** L1 beats L2 by 0.0407 ± 0.0123 (clustered 0.0126) in mean RMSE. It also wins in error energy, the aggregation L2's squared loss targets: −0.2368 ± 0.0641 pct², RMS energy 2.2430 against 2.2951. L1 is better on 108 of the 200 cubes. The argument that RMSE favours an L2 fit (L1 targets the median) does not show at this budget. L2 overfits more (above) and is worse than the baseline on 65 cubes against L1's 53. By class, L2 is ahead on the seen class P022 (L1 − L2 +0.0336 ± 0.0237) and behind on the unseen classes (−0.0508 ± 0.0134). *Interpretation:* this fits L2 having fitted the training classes more closely. Each loss is one training seed, so the comparison carries no training-seed variance.
4. **Seen vs unseen classes.** L1 gains 16.0 % on the seen class P022 and 8.0 % on the unseen classes: a difference of −0.1392 ± 0.0526 (clustered 0.0625), where B's ddim1 gained alike (−0.0083 ± 0.0142, Phase 1a). For L2 the gap is wider: 17.6 % against 5.9 %, a difference of −0.2235 ± 0.0588 (clustered 0.0698). But the unseen P024 gains 16.8 % (L1) and 15.3 % (L2), about as much as P022. The shortfall is P023 (6.0 % and 3.8 %), which holds 152 of the 200 cubes and so sets the subset mean. With three classes, a seen/unseen effect cannot be told apart from a class effect.
5. **Compute.**
   - *Diagnostics* (Task 6; Rulings 4 and 8). The three shards ran in sequence on GPU 0 from 10-15 19:33:57 to 21:10:32 (1.61 h wall-clock), because GPU 1 held the user's other project. Shard times: b_fast 522 s (ddim10 ≈ 450 s, t999 ≈ 72 s; the shard's per-row print was cumulative), b_ddpm_a 2629 s, b_ddpm_b 2630 s. That is 5781 s = 1.61 GPU-hours, of which DDPM-1000 K = 4 on 50 cubes is 1.46. The plan expected ≈ 55 min of wall-clock over both GPUs (Deviation 4). The CPU merge wrote `b_diagnostics.json` at 21:12:43.
   - The shards were launched at ce31b21 (b_fast), 24593a8 (b_ddpm_a) and c431a02 (b_ddpm_b), the HEAD when each started. Their recorded `git_head` (24593a8, c431a02, b1b8dce) is the HEAD when each was saved; 5739753 fixed that. The commits in between change only the gate, `launch_run.sh`, `smoke_regress.sh` and tests, so all three shards ran the same sampling code.
   - *B's y-swaps* (CPU): the leaky record took 267 s (ce31b21), the leak-free one 257 s (5739753).
   - *Regressors* (Rulings 13 and 18). L1 ran alone on GPU 0 from b1b8dce for 7.56 h: one launch, no relaunch, no skipped steps. The final fix wave (5739753, 22:48) landed during L1's training and changed nothing under `main_2d_residual_fsdp.py` or `model/`. L2 started on GPU 1 at 04:08:03 (5739753) when the user's other project freed it, so the two runs overlapped only for L1's last 8 epochs and L2's first 9. L2 reached DONE at 11:00:28 after 6.87 h: one launch, no relaunch, no skipped steps.
   - *Evaluations* on CPU (Ruling 19). L1's evaluation (direct grid, 2 seeds, plus the 32-cube y-swap) was written at 04:52:52, 3 min 55 s after L1's DONE; its direct row took 0.344 s/cube. L2's was written at 11:04:02, 3 min 34 s after its DONE (0.274 s/cube). The early L1 gate ran at 04:53:05, the binding gate at 11:04:09.
   - *Totals.* GPU time is 16.04 GPU-hours: diagnostics 1.61, L1 7.56, L2 6.87, plus Task 5's smoke test and dry run. From the diagnostics' start (10-15 19:33:57) to the binding gate, the phase took 15.50 h of wall-clock.
   - Test suite after the runs (`CUDA_VISIBLE_DEVICES= python -m pytest tests -q`, at 5739753): `317 passed, 1 warning in 129.83s (0:02:09)`.

## Decision for the user
The binding gate selects the template's branch **"Best regressor PASS but target not met"**. go_1b is false, and the seed-43 runs did not happen (Deviation 5). The other branches do not apply: neither regressor is INVALID, PASS_NOT_OVER_B or FAIL. **The decision is yours; nothing is running or queued.**

Where things stand:
- The best regressor, R L1, scores 2.1422 %: 8.9 % below the baseline and 0.1437 above the 1.9985 % target. The target needs 15 % in mean RMSE, about 28 % of the error energy. L1 removes 17.3 % of the energy, B's DDIM-10 chain centre 19.3 % (at K → ∞ draws) and B's one-step centre 11.4 %.
- Both regressors overfit at base 64. Their validation minima are at epoch 15 (L1) and 10 (L2), and they are 9.0 % and 19.9 % higher at epoch 100.
- L1 beats L2 by 0.0407 ± 0.0123, one seed each.

Options, with what each costs:
1. **Regularise or stop early at base 64.** Pre-register checkpoint selection on the monitoring loss of the 800 non-gate files (the gate cubes stay unseen), and add weight decay (0 now) or augmentation. A full base-64 run costs 6.9–7.6 h on one GPU (this phase), and two variants fit side by side on the two GPUs. The validation minima were reached after 1.32 h (L1, epoch 15) and 0.82 h (L2, epoch 10; its first 9 epochs ran side by side with L1). This tests whether generalisation, not capacity, limits the gain. The already saved epoch-25 and 50 checkpoints could be scored in minutes on CPU, but only as a labelled post-hoc look, since nothing pre-registered that selection.
2. **More epochs at base 64.** The validation losses do not support it: they are lowest at epoch 15 (L1) and 10 (L2) and higher at every later check.
3. **More capacity: the Phase 1b′ path.** This is the regressor at base 256 with FSDP, batch 8/GPU: ≈ 27.6 h for 200 epochs on both GPUs at Phase 0's timing (16.1 GB peak). `launch_run.sh` passes `--max_hours 24`, so it needs a higher cap. The gate does not license it (go_1b false). The base-256 model has 16× the parameters (940,117,056 against 58,942,272) and would train on the same 9000 files on which base 64 already overfits by epoch 10–15. Nothing measured here predicts its gain.
4. **Change the information the model gets.** The best models here remove 17–19 % of the baseline's error energy from (x̂₀, y_d) at d = 4 (L1 17.3 %, B's DDIM-10 chain centre 19.3 %); the target needs about 28 %. Every model here barely uses the sensor-only path: sensor-only swaps cost +0.0497 (L1), +0.0611 (L2) and +0.0171 (B), against full swaps of +0.7789, +0.7679 and +0.1895. Changing the conditioning path, the training data or d changes what can be predicted. Each is a new spec decision and costs at least one base-64 run (6.9–7.6 h on one GPU) plus any data or prior work.
5. **Use B's chain centre.** Step 0 shows that B's chain centre is better than its one-step read-out (19.3 % against 11.4 % of the energy removed), and its K → ∞ limit (2.1079 %, extrapolated) is near L1. Reaching it by sampling costs ≈ 9 DDIM-10 draws (≈ 90 NFE) just to match the one-step read-out in mean RMSE. Distilling the chain mean into one call is a direction whose cost is not measured here.
6. **A seed-variance estimate**, if you want one. Seed 43 of both losses side by side costs ≈ 8.8 h of both GPUs (the plan's estimate, Deviation 5). It cannot make go_1b true at these RMSEs, but it measures the training-seed spread of L1's 8.9 % and of the 0.0407 L1 − L2 difference.
7. **Stop** the residual line at d = 4 with this result.
