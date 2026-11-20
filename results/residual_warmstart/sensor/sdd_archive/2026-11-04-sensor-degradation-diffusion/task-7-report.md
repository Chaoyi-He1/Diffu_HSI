# Task 7 report — the trainer learns the sensor modes

Status: DONE. Commit 08bd6f8 (feat(sensor): trainer runs the four sensor modes ... (Phase 3 Task 7)). Not pushed.

## Implemented (main_2d_residual_fsdp.py)
- Imports: HFDSensorSim, load_config, DEVICES, ONE_CALL_MODES, SENSOR_MODES, batch_to_cond. Module docstring gains a Phase 3 usage line.
- get_parser: `--sensor_config` (default ''), `--sensor_device` (choices DEVICES, default 'device1'); help texts of --prediction_type / --loss_type updated.
- exclude_gate(val, data_root): the existing gate-cube block, factored out verbatim (same assertion message).
- make_datasets: SENSOR_MODES -> load_config + HFDSensorSim train/test (val never set_epoch'd), exclude_gate, --limit_files as today; legacy path unchanged (train_half / xr_cache legacy only). Mode read with getattr(args, 'mode', None) because the monitoring scripts pass a namespace without `mode` (tests/test_uq_training.py covers this).
- make_loader: collate_fn = getattr(ds, 'collate_fn', residual_collate_fn); persistent_workers = not hasattr(ds, 'set_epoch') when workers > 0 (prefetch_factor=4 kept).
- check_mode_args: ONE_CALL_MODES <-> direct; varhead/varhead_fam <-> nll; SENSOR_MODES <-> bool(sensor_config); SENSOR_MODES need d == 1; xr_cache / train_half checks unchanged.
- make_trainer: regress_fam -> RegressionTrainer, varhead_fam -> VarheadTrainer.
- check_resume_meta(..., sensor_sha=None): sensor_config_sha mismatch appended to the mismatch list.
- train_one_epoch / validate: batch_to_cond + target_for(mode, x, cond.get('x0_hat'), scale); timing, autocast, skip logic, logging unchanged.
- build_everything: passes train_ds.checkpoint_extra()['sensor_config_sha'] for SENSOR_MODES.
- run_epochs: ds_meta.set_epoch(epoch) before each epoch when the training dataset has it (ds_meta = parts[0] / parts.train_ds).
- checkpoint_meta unchanged.

## Tests (tests/test_sensor_training.py, 10 tests)
StubSensorSet (Task 5 dict items, H=16, 64 bands, 30 channels, 8 cubes, one-hot / all-ones mask, s_post >= 0.02, noise seeded by (epoch, idx), sensor_collate_fn, checkpoint_extra as specified) and StubTupleSet (the Phase 1 tuple stub).
- test_check_mode_args_refusals (the five brief refusals + cond_fam with direct / nll; the four valid sensor launches and legacy residual / regress pass)
- test_make_loader_uses_the_dataset_collate_and_no_persistent_workers_with_set_epoch
- test_two_epochs_train_on_cpu_and_log_finite_losses x4 (base 16, batch 4, num_epochs 2, val_every 1, save_every 1; log 2 records finite, DONE == '2', meta sensor_config_sha == 'cfgsha', sigma_d == 0.02, also P_y / sensor_device / R_sha256, and set_epoch(1) reached the dataset)
- test_set_epoch_changes_training_noise_but_monitoring_is_fixed
- test_resume_refused_when_sensor_config_sha_differs
- test_build_everything_passes_the_sensor_sha_on_resume (extra: wiring check, mirrors the XR test)
- test_legacy_residual_mode_still_trains

## TDD evidence
RED: `CUDA_VISIBLE_DEVICES= .../envs/hsi/bin/python -m pytest tests/test_sensor_training.py -q` -> `error: unrecognized arguments: --sensor_config /cfg.json`; 9 failed, 1 passed (the legacy test).
First GREEN attempt of the four files: 1 failed (test_uq_training::test_make_datasets_takes_the_minimal_namespace_of_the_monitoring_scripts: SimpleNamespace has no `mode`) -> fixed with getattr.
GREEN: `CUDA_VISIBLE_DEVICES= .../envs/hsi/bin/python -m pytest tests/test_sensor_training.py tests/test_train_residual.py tests/test_uq_training.py tests/test_regress_mode.py -q` -> 76 passed, 1 warning (pre-existing: the lr_scheduler-before-optimizer.step warning from test_train_residual::test_nonfinite_loss_is_skipped_without_a_step_then_aborts, which skips the first step on purpose).

## Files changed
- /data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan/main_2d_residual_fsdp.py
- /data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan/tests/test_sensor_training.py (new)

## Self-review
- Legacy path: batch_to_cond on a tuple returns (x, {'sensor', 'x0_hat'}, sigma_d) — identical cond and target to before; validate now uses non_blocking copies (no numerical effect). Legacy loaders keep persistent workers and residual_collate_fn (asserted).
- Exact resume: the dataset's noise is a function of (config seed, epoch, idx) and run_epochs sets the epoch from the loop index, so a resumed epoch e sees the same noise; val stays at epoch 0.
- Meta: sensor keys arrive via the existing `extra`; sigma_d in the meta is the dataset's 0.02.

## Concerns
- dry_run() was not in the brief's Files and still unpacks tuple batches (`x, y_d, xh = next(it)`), so `--dry_run_steps` fails for the sensor modes. If Task 8 launch scripts use dry runs for timing, dry_run needs the same batch_to_cond change (a 3-line edit).
- SENSOR_MODES require an explicit `--ds 1` (the parser default is 4), as the brief specifies; launch scripts must pass it.
