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
