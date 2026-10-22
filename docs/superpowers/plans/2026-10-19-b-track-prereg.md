# B-track pre-registration — 2026-10-19

Follow-up to Phase 1a′ (`2026-10-12-residual-warm-start-phase1a-prime-RESULTS.md`). The user's decision (2026-10-19): keep diffusion as the goal, but first see whether the one-call regressor can reach the 1.9985 % target (the B-track), then return to diffusion (the A-track: distil the chain's mean; a matched warm start at t ≈ 11). This file fixes the rules of the B-track before any gate look, because the monitoring split shares all 113 gate source images with the gate cubes.

## B.2 / B.3 — test-time augmentation and ensembles (no training)

Spike, scored on the 800 monitoring files only: `results/residual_warmstart/diagnostics/regress_tta_ensemble_monitoring.json` (script: a throwaway under the job's scratch dir; the record carries its provenance). It reports, in the gate's metric (mean per-file RMSE %, clamped x̂), every kept checkpoint of the two Phase 1a′ regressors under each of the 8 dihedral transforms, TTA over {identity, transpose} and over all 8, L1 + L2 ensembles, snapshot ensembles over epochs, and their products.

- Known in advance: the loader point-samples the sensor image at the top-left of each stride-4 block and the linear lift anchors sample (i, j) at pixel (4i, 4j), so flips and 90°/270° rotations are not exact symmetries of the data (a 3-pixel grid shift); the transpose is. The spike reports each transform separately so this shows.
- No combination from this spike is evaluated on the gate under this note. If one is worth it, it gets its own pre-registered rule (the gate harness scores single checkpoints today).

## B.1 — checkpoint selection on the monitoring split (one run)

Run `regress_l1_sel_d4_b64`: the Phase 1a′ L1 regressor's exact configuration (B's architecture, base 64, batch 8, lr 1e-4 cosine, weight decay 0, grad clip 1.0, seed 42, 100 epochs, bf16, 8 workers) with every epoch monitored and every checkpoint kept:

```
RUN_NAME=regress_l1_sel_d4_b64 bash scripts/pilot/launch_run.sh regress <gpu> --prediction_type direct --loss_type l1 --val_every 1 --save_every 1 --keep_last 100
```

**Selection rule (fixed now):** the checkpoint evaluated on the gate is the one with the lowest `val_loss` in the run's `train_log.jsonl` (the monitoring loss on the 800 non-gate files, every epoch). Ties go to the earlier epoch. Exactly one checkpoint of this run is evaluated on the gate, with the Phase 1a′ commands (`scripts/eval_warmstart.py --mode regress --prediction_type direct --grid direct`, then `scripts/gate_1a_prime.py` with B's leak-free y-swap and `b_diagnostics.json`; `--no_require_done` is not needed since the run completes). The epoch-100 checkpoint is reported as the pre-registered control, so the run's own epoch-100 result (a replicate of Phase 1a′'s L1 up to GPU non-determinism) can be compared, and that is the only other checkpoint of this run that touches the gate.

Expected from Phase 1a′: the monitoring minimum near epoch 15 (0.4208 there against 0.4586 at epoch 100), and on the monitoring files the mean-RMSE gain rising from 8.2 % to about 14.7 % against the 15 % target. The gate verdict comes from `gate_1a_prime.py` as before (PASS requires beating the baseline and B's ddim1 by 2 SE and validity under the one-step rule; the target is 1.9985 %).

Costs: ≈ 7.6 h on one GPU plus about 7 % for the per-epoch validation; 100 checkpoints of 0.7 GB (70 GB, pruned to the selected, epoch-100 and milestone checkpoints after the gate).

## B.4 — only if B.1 misses the target

A per-cube RMSE loss (the metric itself) and training-step regularisation (weight decay, augmentation inside the training step; the loader stays unchanged), each as one run with the same selection rule, pre-registered in a later section of this file before it starts.

## B.2 / B.3 result (2026-10-19, before any gate look)

`regress_tta_ensemble_monitoring.json` (GPU 1, fp32 with TF32 convolutions, 1 379 s; identity singles reproduce the CPU monitoring record to |ΔMSE| ≤ 0.0014). Mean per-file RMSE % on the 800 monitoring files, baseline 2.3033 there; "gain" is paired against the baseline, "worse" counts files worse than the baseline:

| combination | NFE | RMSE % | gain | worse |
|---|---|---|---|---|
| L1@100 (identity) | 1 | 2.1140 | 8.2 % | — |
| L1@25 (identity) | 1 | 1.9636 | 14.8 % | 75 |
| L2@25 (identity) | 1 | 2.0049 | 13.0 % | — |
| L1@25 + L2@25 | 2 | 1.9260 | 16.4 % | 49 |
| L1@25 + L2@25 + L1@50 + L2@50 | 4 | 1.9121 | 17.0 % | 48 |
| the same + transpose TTA | 8 | 1.9092 | 17.1 % ± 0.5 (SE of the paired gain 0.0111) | 30 |
| all 8 checkpoints + 8-fold TTA | 64 | 1.9580 | 15.0 % | 7 |

- Transforms, L1@25 alone: identity 1.9636, transpose 2.0408, flips/rotations 2.09–2.21. The trained network is not orientation-symmetric, so single transformed inputs are worse; averaging them still helps a little (transpose TTA +0.3 pt) and makes the per-file error far more uniform (8-fold TTA: 9 files worse than the baseline against 75).
- The ensembles across the two losses and across epochs 25/50 are the strong effect: +2 pt over the best single checkpoint, at 2–4 NFE.

## E1 — the ensemble's gate evaluation (pre-registered; awaiting the user's go before the gate look)

Exactly one ensemble is evaluated on the gate: **L1@25 + L2@25 + L1@50 + L2@50 with transpose TTA** (the spike's best; 8 NFE; the four checkpoints exist from Phase 1a′ and were kept as pre-chosen milestones). Its residual prediction is the mean of the eight calls, composed with the true x̂₀ and clamped as before; the y-swap uses the same mean. The gate's verdict rules are unchanged (PASS against the baseline and B's ddim1 by 2 SE, the one-step validity rule, target 1.9985 %). The 4-NFE version without TTA is reported alongside as a secondary row only if the harness scores it in the same run; nothing else from the spike touches the gate. The selection of the four checkpoints and of the transpose came from the monitoring split, which shares all 113 gate source images with the gate, so the gate number after this selection is not an image-independent hold-out; this is stated with the result.

## B.1 result (2026-10-19)

`regress_l1_sel_d4_b64`: DONE = 100 at 10:28, 7.49 h on GPU 1, 0 skipped steps, 100 checkpoints kept. Monitoring loss (every epoch, 800 non-gate files): minimum 0.41920 at epoch 17, 0.45876 at epoch 100 (the Phase 1a′ L1 run: 0.4208 at 15, 0.4586 at 100). The pre-registered rule selects **epoch 17**. Two gate evaluations, as pre-registered (CPU, `--seeds 0 1`):

| checkpoint | record | RMSE % (200 gate cubes) | − baseline | − B ddim1 | − B 8-draw mean | validity (a) / (b) | spec ratio | verdict | target 1.9985 |
|---|---|---|---|---|---|---|---|---|---|
| epoch 17 (selected) | `gate_1a_prime__regress_l1_sel_d4_b64__nodone.json` | **1.9621 ± 0.0439** | −0.3890 ± 0.0223 (cl 0.0242), 16.5 % | −0.2805 ± 0.0182 (cl 0.0193) | −0.2519 ± 0.0182 | +0.7369 ± 0.0734 / +0.3693 ± 0.0683, valid | 0.736 | PASS | **met** |
| epoch 100 (control) | `gate_1a_prime__regress_l1_sel_d4_b64.json` | 2.1338 ± 0.0476 | −0.2174 ± 0.0244 (cl 0.0261), 9.2 % | −0.1088 ± 0.0205 | −0.0802 ± 0.0206 | +0.8029 ± 0.0770 / +0.5847 ± 0.0752, valid | 0.733 | PASS | not met |

- The epoch-17 record is gated with `--no_require_done` (the gate's final-checkpoint guard; the run is DONE at 100 and the selection was fixed above before the look). The epoch-100 control replicates Phase 1a′'s L1 (2.1422) to within 0.008.
- The selection used the monitoring split, which shares all 113 gate source images with the gate cubes, so the 1.9621 % is a post-selection number on a hold-out that is not image-independent. The selection itself used one pre-registered rule and one look.
- `go_1b` is false because the plan's rule requires two training seeds (Phase 1a′ plan, Deviation 5).

## B.5 — the second seed (pre-registered before launch)

Run `regress_l1_sel_s43_d4_b64`: identical to B.1 with `--seed 43`. Selection rule identical (lowest monitoring loss, ties → earlier); exactly one selected checkpoint and the epoch-100 control are gated. `go_1b` is declared true if both seeds' selected checkpoints are PASS and at or below 1.9985 % (the gate's seed-pair logic compares records at one epoch, so the pair verdict is computed from the two single-run gate records under this rule and stated with both numbers). Launch:

```
RUN_NAME=regress_l1_sel_s43_d4_b64 bash scripts/pilot/launch_run.sh regress <gpu> --prediction_type direct --loss_type l1 --seed 43 --val_every 1 --save_every 1 --keep_last 100
```

## B.6 — L2 with the same selection (pre-registered before launch) and E2

Run `regress_l2_sel_d4_b64`: the Phase 1a′ L2 configuration (`--loss_type l2`, seed 42) with `--val_every 1 --save_every 1 --keep_last 100`, on the idle GPU 0, side by side with B.5. Same selection rule; one selected checkpoint and the epoch-100 control are gated. Purpose: the L1-vs-L2 comparison at selected checkpoints, and E2.

**E2 (pre-registered; awaits the user's go like E1):** exactly one ensemble of the two SELECTED checkpoints (L1 sel from B.1, L2 sel from B.6) with transpose TTA (4 calls per cube), scored with the ensemble harness. It replaces E1 as the ensemble of record if both runs finish; E1 (the four Phase 1a′ milestone checkpoints) stays pre-registered as written.

```
RUN_NAME=regress_l2_sel_d4_b64 bash scripts/pilot/launch_run.sh regress 0 --prediction_type direct --loss_type l2 --val_every 1 --save_every 1 --keep_last 100
```

## B.7 — the regressor at d = 8 and d = 2 (pre-registered before launch; user: "看一下" d = 8, 2026-10-19)

Same recipe as B.1 at the other two target strides (the user's targets are d = 2, 4, 8): `regress_l1_sel_d8_b64` (`--ds 8`, launched automatically on the GPU that B.5 frees) and `regress_l1_sel_d2_b64` (`--ds 2`, launched only on the user's go, so one GPU stays free for the A-track or the user's other work), L1, seed 42, base 64, batch 8, 100 epochs, `--val_every 1 --save_every 1 --keep_last 100`. The launcher's fixed `--ds 4` is overridden by the trailing `--ds` argument (argparse keeps the last value); the run's `args.json` records the stride and the evaluation must pass the same `--d`.

Selection rule identical (lowest monitoring loss, ties → earlier); one selected checkpoint and the epoch-100 control are evaluated on the same 200 gate cubes with `scripts/eval_warmstart.py --d <8|2> --mode regress --prediction_type direct --grid direct --seeds 0 1 --device cpu`. There is no B at these strides, so `gate_1a_prime.py` (which compares against B's d = 4 record) is not used; the verdict pieces come from the evaluation record itself, by the same rules: (i) paired gain against the baseline with its iid and clustered SE, the baseline being the record's own `baseline` entry (`eval_warmstart.py` computes x̂₀ on the gate cubes at the record's stride; the 1000-file validation baselines are 1.034 % at d = 2 and 4.491 % at d = 8, against 2.313 % at d = 4); **target = 15 % below that stride's gate-cube baseline**, i.e. ratio ≤ 0.85 with its 2-SE margin; (ii) validity by the proposed one-step rule from the record's 32-cube y-swap at t = 999 ((a) swap_full − true > 2 SE, (b) swap_full − baseline + 2 SE ≥ 0); the spec 0.9 ratio reported. Expected: the relative gain shrinks at d = 8, where 63 of 64 pixels are interpolated and σ₈ = 0.0957 is 4.3 × σ₄, and the spatial-fill share of the error grows; the d = 2 case is spectral-dominated.

Analysis that precedes it (CPU, monitoring files only): `diagnostics/regress_spatial_split_monitoring.json` splits the baseline's, B's and the selected regressor's d = 4 squared error into the 1/16 measured grid pixels and the 15/16 interpolated ones, and by distance to the grid.

**Result of that analysis (2026-10-19, CPU, the 800 monitoring files, d = 4; script `scripts/diagnose_spatial_split.py`, a copy of the scratch script the record's provenance names; errors on the [0, 1] display scale, energies in pct²):**

| model | measured pixels (256 / cube) | interpolated pixels (3 840 / cube) | pooled rms |
|---|---|---|---|
| baseline x̂₀ | 0.071 pct² = 0.266 % rms | 6.207 pct² = 2.491 % rms (5.30 at Chebyshev distance 1 from the grid, 7.25 at distance 2) | 2.413 % |
| regressor `regress_l1_sel_d4_b64` @ 17 | 0.076 = 0.276 % rms (−7.6 ± 0.4 % energy, i.e. slightly worse) | 4.429 = 2.104 % rms (28.7 ± 0.8 % of the energy removed; 3.79 / 5.16 by distance) | 2.039 % |
| B ddim1 | 0.109 = 0.330 % rms (54 % worse) | 5.655 = 2.378 % rms (8.9 ± 0.3 % removed) | 2.304 % |

The baseline's measured-pixel error equals the d = 1 baseline (0.263 % on the 1000 validation files): with noise-free y the 30 → 64 spectral inverse is already solved where a sample exists, and 99.9 % of the baseline's error energy sits on the interpolated pixels. The share of the regressor's energy gain that comes from the measured pixels is 0.0 % (bootstrap over files): **at d = 4 the whole gain is spatial fill-in of a known spectrum, none of it spectral.** By class the regressor keeps 0.61 / 0.75 / 0.64 of the baseline's interpolated energy on P022 / P023 / P024 (98 / 608 / 94 files). Consequence for B.7: at d = 8 the measured-pixel error stays at the d = 1 level while 63 / 64 pixels are interpolated, so the regressor's gain is bounded by how much of an 8 × 8-sample spatial fill-in a conditional-mean estimate can recover; the d = 2 case (3 / 4 interpolated, baseline 1.034 %) is the easy end.

## B.6 result (2026-10-19 19:50, CPU; records `eval/regress/d4/regress_l2_sel_d4_b64__checkpoint_epoch_{18,100}__direct.json`, gates `gate_1a_prime__regress_l2_sel_d4_b64__nodone.json` (epoch 18, the pre-registered non-final exception) and `gate_1a_prime__regress_l2_sel_d4_b64.json` (epoch 100))

Run `regress_l2_sel_d4_b64`: DONE = 100 after 9.1 h on GPU 0 (319 s/epoch, 0 skipped non-finite steps, 100 checkpoints kept). Pre-registered selection = **epoch 18** (monitoring loss 0.64514; epoch 100 = 0.77871; epoch 1 = 0.76492), applied once after DONE.

| checkpoint | gate RMSE (200 cubes) | vs baseline 2.3512 | vs B ddim1 2.2426 | validity (t = 999, 32 cubes) | verdict |
|---|---|---|---|---|---|
| epoch 18 (selected) | **1.9908 ± 0.0437 %** | −0.3604 ± 0.0217 (clustered 0.0245) = 15.3 % | −0.2518 ± 0.0180 | swap_full − true +0.7293 ± 0.0809, swap_full − baseline +0.3686 ± 0.0766 → valid; spec 0.9 ratio 0.738 | **PASS; target 1.9985 met at the point estimate**, by 0.008 (paired m − 0.85 · baseline = −0.0077 ± 0.0181, 0.4 SE: not met with a 2-SE margin) |
| epoch 100 (control) | 2.1852 ± 0.0492 % | −0.1659 ± 0.0248 = 7.1 % | −0.0574 ± 0.0212 | valid; ratio 0.733 | PASS; target not met; replicates the Phase 1a′ L2 2.1829 |

Reading: checkpoint selection gives L2 the same jump as L1 (2.185 → 1.991 against L1's 2.134 → 1.962), so the overfitting-after-epoch-20 picture holds for both losses; the L2 regressor stays a little behind the L1 one at the point estimates (no paired L1-vs-L2 test was run). go_1b is untouched by B.6 (it needs the two L1 seeds). E2 (the selected L1 + selected L2 ensemble with transpose TTA) remains pre-registered and waits for the user's go.

## B.5 result and go_1b (2026-10-19 20:18, CPU; records `eval/regress/d4/regress_l1_sel_s43_d4_b64__checkpoint_epoch_{17,100}__direct.json`, gates `gate_1a_prime__regress_l1_sel_s43_d4_b64__nodone.json` (epoch 17), `gate_1a_prime__regress_l1_sel_s43_d4_b64.json` (epoch 100) and the pre-registered two-seed look `gate_1a_prime__regress_l1_sel_d4_b64+regress_l1_sel_s43_d4_b64__nodone.json`)

Run `regress_l1_sel_s43_d4_b64` (seed 43): DONE = 100 after 9.25 h on GPU 1 (0 skipped non-finite steps, 100 checkpoints kept). Pre-registered selection = **epoch 17** (monitoring loss 0.41918; epoch 100 = 0.45958; epoch 1 = 0.46553), the same epoch as seed 42 (0.41920).

| checkpoint | gate RMSE (200 cubes) | vs baseline 2.3512 | vs B ddim1 2.2426 | validity (t = 999, 32 cubes) | verdict |
|---|---|---|---|---|---|
| seed 43, epoch 17 (selected) | **1.9516 ± 0.0430 %** | −0.3996 ± 0.0233 (clustered 0.0258) = 17.0 % | −0.2910 ± 0.0196 | swap_full − true +0.7854 ± 0.1035, swap_full − baseline +0.3934 ± 0.0865 → valid; spec 0.9 ratio 0.721 | **PASS; target 1.9985 met** (margin 0.047 ≈ 1.1 SE) |
| seed 43, epoch 100 (control) | 2.1286 ± 0.0469 % | −0.2226 ± 0.0258 = 9.5 % | −0.1140 ± 0.0221 | valid; ratio 0.713 | PASS; target not met; replicates 2.1422 (Phase 1a′) and 2.1338 (B.1 control) |

**go_1b = TRUE under the B.5 rule:** both L1 seeds' selected checkpoints PASS at ≤ 1.9985 % (seed 42: 1.9621 ± 0.0439; seed 43: 1.9516 ± 0.0430), both valid, both spec ratios below 0.9; the two-seed gate call reports `seed_pair +0.0105 ± 0.0080` (the seeds agree within 1.3 SE) and prints `go_1b True`. Caveats carried from B.1: the selected checkpoints are non-final (`--no_require_done`, the pre-registered exception); selection used the monitoring split, which shares all 113 gate source images, so gate numbers after selection are not image-independent (each record was looked at once, as pre-registered); the target margin as a paired test (per-cube m − 0.85 · baseline, 200 cubes) is −0.0469 ± 0.0196 for seed 43 (met by 2 iid SE, 2.4 SE) and −0.0364 ± 0.0189 for seed 42 (1.9 SE: met at the point estimate, not quite by 2 SE), so the 15 % target is met by one seed with a 2-SE margin and by the other just short of it; the two-seed replication is what makes the margin more than a seed accident.

What go_1b means here: the Phase 1 gate for "a one-step residual estimator beats the linear baseline by 15 % at d = 4" is met, by a direct regressor rather than by a diffusion read-out; the question of what diffusion still adds (samples, uncertainty, perceptual quality at wide posteriors, a prior reusable across sensors and strides) is with the user, see the spatial-split result above.

## B.7 result — d = 8 (2026-10-20 03:43, CPU; run `regress_l1_sel_d8_b64`, records `eval/regress/d8/regress_l1_sel_d8_b64__checkpoint_epoch_{11,100}__direct.json`; the verdict is read from the records by a stride-agnostic reader that reproduces the d = 4 gate's paired-gain and validity numbers on the d = 4 records)

Run: launched 2026-10-19 20:10 on GPU 1 by the auto-launcher once B.5 had written DONE; DONE = 100 after 7.45 h (239 s/epoch, 0 skipped non-finite steps, 100 checkpoints kept). Pre-registered selection = **epoch 11** (monitoring loss 0.47862; epoch 1 = 0.51624; epoch 100 = 0.53976): the minimum comes earlier than at d = 4 (epoch 17) and the monitoring loss then climbs 13 % by epoch 100.

Gate cubes at d = 8: baseline x̂₀ = 4.5799 ± 0.1130 % (the 15 % target is therefore 3.8929 %).

| checkpoint | gate RMSE (200 cubes) | paired vs baseline | validity (t = 999, 32 cubes) | verdict |
|---|---|---|---|---|
| epoch 11 (selected) | **4.2319 ± 0.1016 %** | −0.3480 ± 0.0284 (clustered 0.0306) = **−7.6 %** | swap_full − true +0.6571 ± 0.1059, swap_full − baseline +0.3902 ± 0.0992 → valid; spec 0.9 ratio 0.865 | PASS (beats the baseline by 2 SE); **target not met**: m − 0.85 · baseline = +0.339 ± 0.025 |
| epoch 100 (control) | 4.7436 ± 0.1185 % | +0.1637 ± 0.0420 = 3.6 % **worse** | valid; ratio 0.845 | FAIL: worse than the linear baseline |

Per class, epoch 11 keeps 0.897 / 0.932 / 0.910 of the baseline on P022 / P023 / P024 (paired −0.44 ± 0.07, −0.29 ± 0.03, −0.61 ± 0.09); PSNR 27.99 dB vs 27.33; SAM 2.84° vs 2.94°; 0.27 s per cube on CPU.

Reading. (i) The absolute paired gain is about the same as at d = 4 (−0.35 against −0.39 / −0.40 pct), so against a baseline twice as large the relative gain halves: 7.6 % against 16.5–17.0 %. With 63 of 64 pixels interpolated, the conditional-mean regressor recovers only a small part of the spatial fill-in error, as the spatial split predicted (all of the gain is spatial, and the posterior is far wider at d = 8). (ii) Overfitting is severe: by epoch 100 the regressor is worse than the linear baseline, so checkpoint selection is not optional at d = 8. (iii) In the y-swap, the `zero` variant (sensor input zeroed) scores 4.4525, better than the baseline 4.5799, whereas at d = 4 it was worse (2.458 against 2.351): at d = 8 roughly 0.13 of the 0.35 gain is y-independent clean-up of the lift's artefacts in x̂₀ (image-prior smoothing), and `swap_sensor_only` 4.329 against `true` 4.204 shows the sensor input contributes the rest. (iv) For the user's question: at d = 8 regression still beats the linear baseline, but the 15 % target is far away (ratio 0.924) and the recoverable part of the error is small next to the width of the spatial posterior; this is the regime where a point estimate stops being informative and samples / uncertainty have a job.

Not launched: `regress_l1_sel_d2_b64` (awaits the user's go). Disk: the d = 8 run adds 100 checkpoints (≈ 70 GB) to the ≈ 210 GB of the three d = 4 selection runs. **Pruned 2026-10-22 on the user's instruction:** each selection run keeps only its selected epoch and epochs 25 / 50 / 100 (`regress_l1_sel_d4_b64` 17, `regress_l1_sel_s43_d4_b64` 17, `regress_l2_sel_d4_b64` 18, `regress_l1_sel_d8_b64` 11); 4 × 96 files, 253 GiB freed; every checkpoint named in a record above is still on disk.
