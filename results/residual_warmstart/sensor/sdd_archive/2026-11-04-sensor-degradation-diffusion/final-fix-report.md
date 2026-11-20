# Final-review fix wave — implementer report (2026-11-06)

Branch worktree-phase1-plan, on top of a3bc56b. CPU only (`CUDA_VISIBLE_DEVICES=`); nothing written under `results/`; sd_verdict not run on the real manifest; page not rebuilt.

## Commits

- 94ee3db fix(sensor): final-review wave — common-file tie-break, training-device refusals, C sampler provenance, toy/mmse guards, tighter tests (items A, B, G, H, I, J, L)
- bbc3ad6 fix(page): Phase 3 page sources — stale B records out of status.final and the B hours, b_valid guard on the outcome / KPI / count / main-line slots, common-file tie-break when the verdict carries it, typed constants bound or footnoted (items F, K)
- a27757a chore: untrack committed model/__pycache__ bytecode (item M)
- aeda8e3 docs(sensor): Ruling R2, the Minor-1 drop ruling and the two baseline conventions (items C, D, E). It also carries the controller's pending one-line edit to the results note (E4 B re-run launch time "≈ 11:05" → "16:56"), which was already in the working tree when I started.

## Per item

**A. Important 1, tie-break on common files.** `scripts/sensor/sd_verdict.py`:
- new `tie_break(res, settings, tied, min_common)` (l. 240). It takes the files common to every tied line and V at 20 dB, in the first tied line's order, and refuses fewer than MIN_COMMON. Each line's per-cube CRPS is restricted to those files and paired against V with `paired_on_common`, the same path and clustered SE that T2 uses.
- `main_line(res, settings=None, min_common=MIN_COMMON, info=None)` (l. 266) uses it when `settings` is given, which is what `verdict` does. Without `settings` it keeps the old own-mean rule, so the `_fake` unit test is unchanged.
- `verdict` now writes a new top-level field `tie_break` (l. 312). It is None when there is no tie or the verdict is report-only. When used it is `{"metric": "crps_vs_V@20", "n_common": int, "lines": {line: {"mean_own", "n_own", "mean_common", "se_cluster_common"}}, "winner": line}`. Every existing field is unchanged.
- `_print` prints one tie-break line.
- Tests in `tests/test_sd_verdict.py`: `test_tiebreak_pairs_the_tied_lines_on_their_common_files` (l. 247) gives B 50 files and C all 60. C is lower on its own files (−0.333 vs −0.3) and B is lower on the common 50 (−0.3 vs −0.1), and the test asserts winner B, n_common 50 and both means. `test_no_tie_break_when_the_counts_differ` (l. 273) covers the no-tie case.

**B. Important 2, training-device checks.**
- (1) `scripts/sensor/sd_eval.py` `check_models` (l. 92–94): for every flag except `uncond`, it refuses when `meta.get('sensor_device', cfg['device_train']) != cfg['device_train']`. The message is `sensor_device: checkpoint 'device2' vs config device_train 'device1'`, added to the existing "does not fit this evaluation" ValueError.
- (2) `main_2d_residual_fsdp.py`:
  - new flag `--allow_other_train_device` (store_true, default False, l. 98); its help gives the reason (the config sha hashes device_train, not the device used, so the T3 transfer clause would be void).
  - new `check_train_device(args, cfg)` (l. 231), called in `make_datasets` right after `load_config` (l. 165). `check_mode_args` does not load the config, so the check sits where the config is first read. Training and `dry_run` both go through `make_datasets`, and the default device1 launch is unchanged.
- Tests:
  - `tests/test_sd_eval.py::test_refuses_learned_checkpoint_trained_on_another_device` (l. 259) covers D/V/C with device2 refused, device1 accepted, a missing field accepted, and a device2 uncond prior accepted. `save_ckpt` gained a `sensor_device` kwarg.
  - `tests/test_sensor_training.py::test_training_device_must_be_the_configs_unless_allowed` (l. 221) covers the direct check, the flag, and `make_datasets` refusing before any dataset is built.
  - The existing `test_dry_run_handles_dict_batches` still passes.

**C. Important 3, R2.** The ruling text from the ledger, copied verbatim, is a paragraph at the end of `## Phase B — trainings` in the results note (l. 58) and a bullet in the spec's as-built rulings list (l. 159, prefixed "§10 R2 (final review, Important 3):").

**D. Minor 1.** Spec as-built bullet (l. 160): the per-class breakdown, the SE-4 log spread–error correlation and the toy's AUSE were not computed. They are report-only, no clause reads them, and they stay computable post hoc from the per-cube arrays and class labels.

**E. Minor 2 and 3.** The results note already had a "Conventions:" paragraph at the top (l. 5), so I extended it with (i) unclamped L vs clamped D/C/B, with the 0.02 / 0.31 / 0.55 % figures and "no clause moves", and (ii) the Device-1 e2 table used for the Device 2 / PH5 demosaic baselines. A one-line pointer sits under the Stage 0 baseline table (l. 44) and under the E4 table (l. 169).

**F. Minor 4, page extractor and charts.** `docs/analysis/phase3_page/extract_page_data.py`:
- `settings_block` now returns `(settings, final, pending, stale)`. A B record whose prediction type is not x0 goes to the new `status.stale` list and never to `status.final`.
- `eval_hours` counts only x0-prior B records under `eval_hours['B']` and `b_records`. A non-x0 B record still in eval/ counts under the new key `eval_hours['B_stale']`; archived ones stay under `B_vpred`.
- The verdict's `tie_break` is passed through in `verdict_block`, and the extractor prints the stale list.

`docs/analysis/phase3_page/charts.js`:
- New helpers `bInvalid()`, `bInvalidWhy()` and `tieInfo(n)`.
- When `V.b_valid` is false, these slots show a "not current" pill instead of B's numbers: `SL.outcome` (C passes plus the D and C bullets), `SL.kpiB`, `SL.kpiBsub`, `SL.countBC`, `SL.mainline`, and also `SL.countsub` and the clause table's "clauses passed" / "main line" B cells.
- When `V.tie_break` is present, `SL.outcome`, `SL.mainline`, the clause table's main-line row and a new `SL.tbRuling` slot (wrapped around the §9 rulings sentence "B's tie-break CRPS difference is therefore on 50 cubes and C's on 200") show the common-file means ± clustered SE and the own-file means. Without the field they fall back to the previous text.
- `SL.budget` adds `B_stale` to the total.

Checks:
- New `tests/test_phase3_page_extract.py`, 2 tests on synthetic records in a temp EVAL dir: stale → status.stale and not final, and B_stale hours.
- The charts.js syntax check passes with `node -e` (VS Code server's node 18, since `node` is not on PATH).
- A stubbed-D smoke run (scratch script in /tmp, not committed) rendered all changed slots and the clause table in three cases: valid with tie_break, valid without, invalid.

**G. Minor 5, C sampler provenance.**
- `sd_eval.py` has new constants `C_STEPS = 10` and `C1_STEPS, C1_SEED = 1, 0` (l. 47–48), now used by the sampler calls.
- New `c_sampler_provenance(methods, K)` (l. 133) writes `record['provenance']['c_sampler']` (l. 203), next to `b_prediction_type`: `{method: 'ddim', prediction_type: 'v', n_timesteps: 1000, clamp_x0: [-8, 8], C: {n_steps: 10, seeds: [0..K-1], rule}, C1: {n_steps: 1, seed: 0, rule: 'one DDIM call … (spec §5.3)'}}`. It is None without C or C1.
- `test_record_holds_every_requested_method_with_paired_files` is extended to cover it.
- Existing records are not rewritten.

**H. Minor 6, toy denoiser.**
- `scripts/sensor/toy_guidance.py` `GaussianDenoiser` refuses a `trainer.prediction_type` outside ('v', 'x0', 'eps') at construction, and its output branches to x0, eps or v.
- Test `test_gaussian_denoiser_follows_the_trainer_prediction_type`: the same x̂₀ is recovered under v, x0 and eps trainers.

**I. Minor 7.** `data_loader/mmse.py:26`: `assert M.shape[0] <= M.shape[1]` before the thin SVD, with a message naming R's and M's shapes. Test `test_sequential_posterior_refuses_more_channels_than_bands` (tests/test_mmse.py:74).

**J. Minor 8, three tests tightened.**
- `test_pigdm_snapshot_toy_runs_and_beats_the_prior` now asserts `rmse_meas < 0.3 × rmse_meas_prior`: the error of H(x̂ − x) on the measured entries, guided mean vs prior mean; measured 0.095 vs 0.656. It also asserts `rmse ≤ 1.05 × rmse_prior` and keeps `rmse_excess < 0.5`.
  - Comparing against the prior's full-space RMSE was not possible. In snapshot mode the isotropic toy has one scalar measurement per pixel out of 64 dimensions, so the full-space RMSE of a 32-sample mean cannot beat the prior's: measured 1.020 vs 1.011.
  - To support the new assertion, `run_toy` now also returns `rmse`, `rmse_prior`, `rmse_meas` and `rmse_meas_prior`. These are new keys; the old ones are unchanged, and `toy_guidance.json` is not rewritten.
- `test_B_is_unchanged_when_off_mask_entries_are_corrupted` uses `Shrink`, plus an on-mask control: adding 0.5 to the measured entries changes B's CRPS, so the test fails if B ignores y.
- `test_report_only_lists_missing_settings_and_writes_report_name` now uses `pytest.raises(sv.Missing, match='setting device1/sequential/60')`.

**K. Minor 9, typed constants on the page.**
- Bound to data, via new extractor facts:
  - "(T2 passed)" → `(T2 <facts.C_T2_30_word>)`, from the verdict's T2.C["30.0"].passed.
  - "113 images … 45 for B's 50" → `facts.clusters_gate`, the distinct source images of the E1 30 dB main record, and `facts.clusters_B`, the clusters of B's paired rmse_vs_L at E1 30 dB.
  - "seed 42" → `facts.train_seed`: the single seed in the five runs' args.json. `runs_block` now records `seed`; if the seeds differ the value is None and the page shows a pending pill.
- Footnoted: "–13" became "–13*", with a glossary entry "* typed" saying that the upper bound is a typed reading of the singular-value spectrum. The ≥ 1 % count is 11, and the next two values are 0.92 % and 0.86 %. No natural threshold gives 13: 0.5 % gives 14.

**L. Minor 10.** `model/guidance.py:78` `_init` has a docstring: seed k gives the same x_T for every batch of the same shape. This is kept deliberately so that the B records stay reproducible. No behaviour change.

**M. Minor 11.** `git rm --cached` removed the five tracked `model/__pycache__/*.pyc` files from the index (diffusion_trainer, hyperspectral_vae, layers, u2net_2d, u2net_hyperspectral); they are still on disk. `.gitignore` already had `*.pyc` and `__pycache__/`, so it is unchanged.

## Tests

Sensor files (tests/test_sensor_*.py, test_mmse, test_hfd_sensor_sim, test_guidance, test_sd_eval, test_sd_verdict, test_stage0, test_launch_run_sensor, plus the new test_phase3_page_extract):
```
120 passed, 2 warnings in 34.63s
```
Whole suite (`CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 … pytest tests -q -p no:cacheprovider`, after all four commits):
```
561 passed, 3 warnings in 219.67s (0:03:39)
```
(553 before the wave; 8 new tests: 2 for the verdict, 1 each for the sd_eval device check, the trainer device check, the toy denoiser and the mmse assertion, and 2 for the extractor.)

## Concerns

- Scope: beyond the listed slots I also guarded `SL.countsub` and the clause table's count and main-line B cells with `V.b_valid`, and replaced the §9 rulings sentence on the 50-vs-200 tie-break with the `tbRuling` slot. Otherwise they would contradict the guarded or common-file numbers.
- The spec's as-built block does not yet say that the tie-break is now computed on the common files. I left that out because it was not in the scope list; the controller may want one line when it regenerates the verdict.
- The snapshot-toy tightening uses a measured-space error, not the full-space error (see J). The full-space RMSE cannot beat the prior there by construction.
