### Task 7: The trainer learns the sensor modes

**Files:**
- Modify: `main_2d_residual_fsdp.py` (`get_parser` 66–101, `make_datasets` 135–160, `make_loader` 163–173, `check_mode_args` 176–189, `make_trainer` 192–199, `checkpoint_meta` 225–243, `check_resume_meta` 279–312, `train_one_epoch` 397–442, `validate` 445–466, `build_everything` 469–486, `run_epochs` 489–553)
- Test: `tests/test_sensor_training.py`

**Interfaces:**
- Consumes: `HFDSensorSim`, `sensor_collate_fn`, `sensor_config.load_config`, `residual_wrapper.{SENSOR_MODES, FAM_MODES, ONE_CALL_MODES, batch_to_cond, target_for}`.
- Produces: flags `--sensor_config` (default '') and `--sensor_device` (choices `DEVICES`, default 'device1'); `check_mode_args`: `(args.mode in SENSOR_MODES) == bool(args.sensor_config)` else ValueError; SENSOR_MODES require `args.d == 1`; `(args.mode in ONE_CALL_MODES) == (args.prediction_type == 'direct')`; `(args.mode in ('varhead', 'varhead_fam')) == (args.loss_type == 'nll')`; `make_datasets`: SENSOR_MODES → `train = HFDSensorSim(args.data_root, 'train', cfg, device=args.sensor_device, prior_path=args.prior)`, `val = HFDSensorSim(args.data_root, 'test', cfg, device=args.sensor_device, prior_path=args.prior)` (family draws with fixed epoch 0), gate cubes removed from `val.img_list` by the existing block (factored into `def exclude_gate(val, data_root)`); `make_loader`: `collate_fn=getattr(ds, 'collate_fn', residual_collate_fn)`, `persistent_workers = args.num_workers > 0 and not hasattr(ds, 'set_epoch')`; `make_trainer`: `regress_fam` → `RegressionTrainer`, `varhead_fam` → `VarheadTrainer`; `train_one_epoch` / `validate`: `x, cond, scale = batch_to_cond(mode, batch, sigma_d, device)`, `target = target_for(mode, x, cond.get('x0_hat'), scale)`; `run_epochs`: `if hasattr(ds_train, 'set_epoch'): ds_train.set_epoch(epoch)` before each epoch (the tuple-`parts` path uses `parts[0]`); `check_resume_meta(..., sensor_sha=None)`: when given, `meta['sensor_config_sha']` must equal it (listed among the mismatches); `build_everything` passes `train_ds.checkpoint_extra()['sensor_config_sha']` for SENSOR_MODES; `checkpoint_meta` unchanged (`extra` already carries the dataset's `checkpoint_extra()`).

- [ ] **Step 1: Write the failing tests** (pattern of `tests/test_uq_training.py`: a `StubSensorSet` whose items are the Task 5 dict with H = 16, 8 cubes, `collate_fn = sensor_collate_fn`, `set_epoch` that re-seeds the noise, `checkpoint_extra()` returning `dict(sensor_config_sha='cfgsha', s_floor01=0.01, P_y=50.0, snr_floor_db=60.0, family={}, device_train='device1', sensor_device='device1', R_sha256='r')`, `sigma_d = 0.02`, `prior = {'s': 3e-7}`, `img_list`)

```python
# tests/test_sensor_training.py  (names and assertions; bodies follow test_uq_training.py's helpers)
def test_check_mode_args_refusals():
    # uncond without --sensor_config; cond_fam with --ds 4; regress_fam without direct; varhead_fam without nll;
    # residual WITH --sensor_config  -> each raises ValueError
def test_make_loader_uses_the_dataset_collate_and_no_persistent_workers_with_set_epoch():
    # loader.collate_fn is sensor_collate_fn; loader.persistent_workers is False for StubSensorSet (num_workers=1), True for a tuple dataset
@pytest.mark.parametrize('mode,pt,loss', [('uncond', 'v', 'l1'), ('cond_fam', 'v', 'l1'), ('regress_fam', 'direct', 'l1'), ('varhead_fam', 'direct', 'nll')])
def test_two_epochs_train_on_cpu_and_log_finite_losses(tmp_path, mode, pt, loss):
    # tr.run_epochs(args, (ds, loader, model, trainer, opt, sched), tr.NO_DIST, CPU) with num_epochs=2, base 16, spectral 64, batch 4
    # -> train_log.jsonl has 2 records, both losses finite, DONE == '2', checkpoint meta has sensor_config_sha == 'cfgsha' and sigma_d == 0.02
def test_set_epoch_changes_training_noise_but_monitoring_is_fixed():
    # first batch 'y' of epoch 0 != epoch 1 on the train stub after set_epoch; the val stub (no set_epoch call) gives identical 'y'
def test_resume_refused_when_sensor_config_sha_differs(tmp_path):
    # check_resume_meta(ck with meta sensor_config_sha='other', ..., sensor_sha='cfgsha') raises ValueError mentioning sensor_config_sha
def test_legacy_residual_mode_still_trains(tmp_path):
    # the Phase 1 tuple dataset stub of tests/test_train_residual.py runs one epoch unchanged
```

- [ ] **Step 2: Run to verify failure.** Run: `... -m pytest tests/test_sensor_training.py -q`. Expected: FAIL (unknown flags / modes).

- [ ] **Step 3: Implement the trainer changes** per Interfaces.

- [ ] **Step 4: Run the new tests and every existing trainer test.** Run: `... -m pytest tests/test_sensor_training.py tests/test_train_residual.py tests/test_uq_training.py tests/test_regress_mode.py -q`. Expected: all pass.

- [ ] **Step 5: Commit** — `git add main_2d_residual_fsdp.py tests/test_sensor_training.py` ; `git commit -m "feat(sensor): trainer runs the four sensor modes — --sensor_config / --sensor_device, dict batches via batch_to_cond, per-epoch noise with non-persistent workers, config sha in the meta and the resume check (Phase 3 Task 7)"`.

