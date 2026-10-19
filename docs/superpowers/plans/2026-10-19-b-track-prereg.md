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
