### Task 10: The evaluation harness `sd_eval.py`

**Files:**
- Create: `scripts/sensor/sd_eval.py`
- Test: `tests/test_sd_eval.py`

**Interfaces:**
- Consumes: `uq_eval.{point_metrics, sample_metrics, gaussian_metrics}` (imported from `scripts.uq_eval`), `eval_warmstart.{metrics, load_subset, class_of, make_eval_loader}`, `HFDSensorSim`, `mmse.demosaic_then_mmse`, `residual_wrapper.{build_residual_model, compose, regress_forward, varhead_forward, LOGVAR_RANGE, FAM_MODES}`, `guidance.{LinearMeasurement, draw_guided}`, `sensor_forward.guidance_operator`, `sensor_config.{load_config, training_sha}`, `warmstart_paths.{git_head, file_sha256}`.
- Produces: `METHODS = ('L', 'L_demosaic', 'D', 'V', 'C', 'C1', 'B')`; `EVAL_DIR = os.path.join(SENSOR_DIR, 'eval')`; CLI `--name` (required), `--sensor_device {device1,device2,ph5}`, `--mode {sequential,snapshot}`, `--snr_db` (float), `--subset {gate,ddpm}`, `--methods` (nargs +, default all applicable), `--regress`, `--varhead`, `--cond`, `--uncond` (checkpoint paths; each required iff its method is requested: D ← regress, V ← varhead, C and C1 ← cond, B ← uncond), `--K 32`, `--pigdm_steps 50`, `--guidance {pigdm,dps}`, `--sensor_config CONFIG_PATH`, `--batch_size 4`, `--num_workers 4`, `--limit_files 0`, `--device`, `--data_root`, `--prior`; `def check_models(args, cfg, device) -> tuple[dict, dict]` (loads each requested checkpoint, requires `meta['mode']` to be the method's mode and `meta['sensor_config_sha'] == training_sha(cfg)`, else ValueError naming the file; provenance per checkpoint: path, epoch, sha256, mode, git_head); `def run_eval(models, ds, loader, args, cfg, provenance, device) -> dict` with `per_cube[method]` lists: `L` → `gaussian_metrics(x0_hat, s_raw, x)` (mean_* keys hold the point metrics of x̂; `s_raw` on the trainer scale); `L_demosaic` (snapshot only) → point metrics of `demosaic_then_mmse(...)`'s x̂ plus its Gaussian metrics; `D` → `point_metrics(compose(x0_hat, s_post, regress_forward(models['D'], cond)), x)`; `V` → `gaussian_metrics(x0_hat, s_post · exp(0.5 · logvar.clamp(*LOGVAR_RANGE)), x)`; `C` → `sample_metrics(stack_k compose(x0_hat, s_post, r_k), x)` with r_k from `trainer.sample(models['C'], cond, shape, n_steps=10, method='ddim', clamp_x0=(-8.0, 8.0))` under `torch.manual_seed(k)`; `C1` (spec §5.3's one-call point of C) → `point_metrics(compose(x0_hat, s_post, r1), x)` with r1 from `trainer.sample(models['C'], cond, shape, n_steps=1, method='ddim', clamp_x0=(-8.0, 8.0))` under `torch.manual_seed(0)`; `B` → `sample_metrics(draw_guided(models['B'], trainer, LinearMeasurement(H, b, y, mask if snapshot else None, sigma), shape, K, pigdm_steps, args.guidance), x)`; record keys `name, setting=dict(sensor_device, mode, snr_db, sigma, R_sha256), subset, K, pigdm_steps, guidance, n, files, classes, per_cube, means, coverage_exact=bool(K >= min_k_for(0.9)), provenance=dict(sensor_config_sha, config_path, checkpoints, git_head, created, seconds, seeds, device, batch_size, limit_files)`; `def output_path(name) -> os.path.join(EVAL_DIR, name + '.json')`; refuses to overwrite; prints one line per method with rmse / crps / cov90 / ause.

- [ ] **Step 1: Write the failing tests** (stub dataset yielding the Task 5 dict with H = 8, `sigma_d = 0.02`, `R`, `A`, `img_list`; stub nets `Const` / `Noisy` from `tests/test_uq_eval.py`; checkpoints written with `torch.save(dict(model_state_dict=..., meta=dict(mode=..., sensor_config_sha=...)))`)

```python
def test_record_holds_every_requested_method_with_paired_files(tmp_path)   # methods L D V C C1 B on a sequential stub; len(per_cube[m]['...']) == n for each; 'L_demosaic' absent; per_cube['C1'] has 'rmse_pct'
def test_snapshot_adds_L_demosaic(tmp_path)                                  # snapshot stub -> 'L_demosaic' present with mean_rmse_pct
def test_refuses_checkpoint_with_other_config_sha(tmp_path)                 # meta sensor_config_sha 'other' -> ValueError naming the checkpoint
def test_refuses_wrong_mode_checkpoint(tmp_path)                            # --cond pointing at a regress_fam checkpoint -> ValueError
def test_refuses_overwrite(tmp_path)
def test_B_is_unchanged_when_off_mask_entries_are_corrupted(tmp_path)        # snapshot stub, Const uncond model, K=2, 2 steps: per_cube['B'] identical after adding 1e3 to y where mask == 0
def test_coverage_exact_flag(tmp_path)                                       # K=32 -> True, K=8 -> False
```

- [ ] **Step 2: Run to verify failure.** Expected: FAIL, module missing.

- [ ] **Step 3: Implement `scripts/sensor/sd_eval.py`** per Interfaces; `main()` builds `HFDSensorSim(args.data_root, 'test', cfg, device=args.sensor_device, setting=dict(mode=args.mode, snr_db=args.snr_db), prior_path=args.prior)` restricted to the subset's files (`load_subset(args.data_root, 'files' | 'ddpm1000_subset')`, in that order), `make_eval_loader` with the dataset's `collate_fn`.

- [ ] **Step 4: Run to verify pass.** Expected: 7 passed.

- [ ] **Step 5: Commit** — `git add scripts/sensor/sd_eval.py tests/test_sd_eval.py` ; `git commit -m "feat(sensor): sd_eval — one record per (device, mode, SNR, subset) with L / L_demosaic / D / V / C / B, config-sha and mode checks, never overwrites (Phase 3 Task 10)"`.

