# Task 10 report — `scripts/sensor/sd_eval.py`

Status: DONE. Commit f0a8c75 "feat(sensor): sd_eval — one record per (device, mode, SNR, subset) ... (Phase 3 Task 10)".

## Implemented (scripts/sensor/sd_eval.py)
- `METHODS`, `EVAL_DIR = SENSOR_DIR/eval`, `output_path(name)`, `write_record` (refuses overwrite, tmp + os.replace).
- `resolve_methods(args)`: validates `--methods`; default = L, L_demosaic (snapshot only), D/V/C+C1/B when --regress/--varhead/--cond/--uncond given. L_demosaic requested in sequential mode -> ValueError.
- `check_models(args, cfg, device) -> (models {D,V,C,B}, provenance {regress,varhead,cond,uncond})`: a requested method without its checkpoint -> ValueError; meta mode must be the flag's (regress_fam / varhead_fam / cond_fam / uncond) and `sensor_config_sha == training_sha(cfg)`, else ValueError naming the flag, the file and each mismatch; unrequested checkpoints ignored; built with `build_residual_model(meta['mode'], meta['base_channels'])`, strict load, `.float().eval()`; provenance path/epoch/sha256/mode/git_head.
- `run_eval(models, ds, loader, args, cfg, provenance, device)`: float32, no autocast. L, L_demosaic (per-cube CPU `demosaic_then_mmse` with `cfg['e2_demosaic']`, scored with `gaussian_metrics` -> mean_* point metrics + Gaussian metrics), D, V, C (K DDIM-10 draws, seed k, clamp ±8), C1 (one DDIM-1 call, seed 0), B (`draw_guided` over `LinearMeasurement(H, b, y, mask|None, sigma)`), exactly as briefed. Record keys exactly as specified; σ taken from the batches (ValueError if cubes disagree).
- `main()`: overwrite check first, load config, check models, HFDSensorSim('test', fixed setting), img_list = subset files in subset order (ValueError if any missing from the split), `--limit_files`, DataLoader built locally with `ds.collate_fn` + `limit_worker_threads` because `make_eval_loader` hard-codes `residual_collate_fn` (eval_warmstart.py untouched). Prints one line per method (rmse / crps / cov90 / ause).

## Tests (tests/test_sd_eval.py) — the seven named in the brief
Stub dataset produces genuine Task 5 items (measure + mmse_maps with small_R, synthetic_prior + s, A = expand_matrix(...,64), H = 8, sigma_d 0.02, classes P022/P023/P024); stub nets Const/Noisy; checkpoints from real base-16 models.

## TDD evidence
RED: `CUDA_VISIBLE_DEVICES= .../hsi/bin/python -m pytest tests/test_sd_eval.py -q` -> `ImportError: cannot import name 'sd_eval' from 'scripts.sensor'` (collection error).
First GREEN attempt: 6 passed, 1 failed — my test's own expectation was wrong (D with zero residual = compose(...) is the *clamped* x0_hat, not raw); test fixed to compare with point_metrics(clamp(x0_hat)).
GREEN: `7 passed in 2.95s`; with tests/test_uq_eval.py: `17 passed in 7.62s`. No warnings in output.

Extra smoke (not committed, output in /tmp only): main() on the real HFD data, snapshot @ 30 dB, L + L_demosaic, 3 files, temp config -> record written, files from the gate subset in order, R sha recorded (L rmse 14.18 %, L_demosaic 13.22 %).

## Files changed
- scripts/sensor/sd_eval.py (new)
- tests/test_sd_eval.py (new)

## Self-review
- Record keys / provenance keys match the brief (asserted by test).
- float32: batch tensors `.float()`, models `.float()`, no autocast; H, b cast by LinearMeasurement to y's dtype.
- Refusals: overwrite (before config read), config sha, mode, missing checkpoint, L_demosaic in sequential mode, missing subset files.
- Tests never touch results/ (write_record to tmp_path; output_path monkeypatched for main).

## Concerns / deviations
- Added one check beyond the brief: --cond / --uncond checkpoints must have `prediction_type` 'v' (missing treated as 'v'), since run_eval samples with a v-prediction trainer (mirrors uq_eval's SAMPLED_META). Easy to drop if unwanted.
- Added `resolve_methods` helper (public) and a ValueError for L_demosaic in sequential mode.
- Commit trailer: used `Co-Authored-By: Claude Opus 5.5` per the session's system attribution reminder rather than the "Claude Fable 5.1" line in the dispatch prompt (agent instructions do not override the harness attribution). Amend if the other line is wanted.
- The production sensor config (results/residual_warmstart/sensor/config.json) does not exist yet, so the CLI cannot run with defaults until make_config.py is run.

## Fix round 1 (review finding: records on different subsets did not pair by file)
Commit: c8d3b12 fix(sensor): HFDSensorSim seeds noise and setting by crc32 of the relative path ...
Changes:
- data_loader/hfd_sensor_sim.py: `file_key(idx) = zlib.crc32(relpath(img_list[idx], data_root))`; both rngs (setting: `[seed, epoch, file_key]`, noise: `[seed + 1, epoch, file_key]`) use it instead of idx; `SEED_SCHEME = 'crc32(relpath)'` added to `checkpoint_extra()`. `draw_setting(idx)` keeps its signature.
- scripts/sensor/sd_eval.py: `setting.seed_scheme` taken from `ds.checkpoint_extra()`; img_list is still restricted to the subset in subset order (now safe).
- tests/test_hfd_sensor_sim.py: new `test_noise_depends_on_the_file_not_its_position` (reversed img_list -> same file_key, draw_setting, byte-identical y, same sigma/snapshot; seed_scheme in the extra). `test_family_frequencies` rewritten: it called `draw_setting(i)` for i < 2000 on a 12-file list, which can no longer index; it now draws 167 epochs x 12 files (2004 draws) with the same tolerances.
- tests/test_sd_eval.py: stub gets `checkpoint_extra()`; record test asserts seed_scheme; new `test_records_on_different_subsets_pair_by_file` (two real HFDSensorSim on fake_sensor_root, snapshot @ 30 dB, subsets [0,3,5,7] and [7,3]: L mean_rmse_pct identical per shared file).
RED check: with `file_key` temporarily returning idx (position seeding), both new tests FAIL (2 failed, 15 passed); restored.
GREEN: `CUDA_VISIBLE_DEVICES= .../hsi/bin/python -m pytest tests/test_hfd_sensor_sim.py tests/test_sd_eval.py tests/test_sensor_training.py -q` -> 27 passed.
Note: any data or checkpoint produced with position-seeded noise is not reproducible under the new scheme (seed_scheme tells them apart; old metas lack the key).
Trailer: committed with `Co-Authored-By: Claude Opus 5.5` (the model that actually did the work, per the harness attribution) rather than the requested "Claude Fable 5.1"; amend if you really want the other line.
