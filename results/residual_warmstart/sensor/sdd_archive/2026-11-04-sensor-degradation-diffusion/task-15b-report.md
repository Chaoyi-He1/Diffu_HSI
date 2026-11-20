# Task 15b — opt-in mode `uncond_tt` (time tokens for the unconditional prior)

Status: DONE. Commit 65d028a (branch worktree-phase1-plan, not pushed).

## Changes per file
- `model/u2net_hyperspectral.py`: new constructor kwarg `time_tokens_when_no_context=False`, stored as
  `self.time_tokens_when_no_context`. In `forward`, `create_context_for_resolution(h, w)` with no sensor context returns
  `t_emb[:, None, :].expand(-1, h*w, -1)` when the flag is set (no context-encoder call), else None as before.
  No new parameters; the default path is unchanged.
- `model/residual_wrapper.py`: `uncond_tt` appended at the end of `MODES`; mode-table comment added;
  `SENSOR_MODES = ('uncond', 'uncond_tt') + FAM_MODES` (comment updated: "no sensor context (uncond_tt: time tokens)");
  `build_residual_model` passes `time_tokens_when_no_context=(mode == 'uncond_tt')` (otherwise the uncond config:
  extra_in 0, use_x0_hat False, use_context False); `batch_to_cond` treats `uncond_tt` like `uncond`; `target_for`
  needed no change (not in CENTRED_MODES, so it returns x).
- `main_2d_residual_fsdp.py`: NOT changed. `check_mode_args` / `make_datasets` / `make_trainer` / `build_everything`
  use `SENSOR_MODES` / `ONE_CALL_MODES` membership and `--mode choices=list(MODES)`, so `uncond_tt` trains like
  `uncond` (DiffusionTrainer, requires --sensor_config and --ds 1, v / l1 accepted). Its module docstring still lists only
  the four original sensor modes (left as is to stay within "only if needed").
- `scripts/pilot/launch_run.sh`: `uncond_tt` added to the usage line, the longer-mode guard loop and the sensor-mode
  `case` (needs --sensor_config, `--ds 1` appended). Header comment notes `uncond_tt_*` for `uncond`.
- `scripts/sensor/sd_eval.py`: new `FLAG_ALLOWED_MODES` (each flag -> its FLAG_MODE, `uncond` -> ('uncond', 'uncond_tt'));
  `check_models` checks membership; the model is built from `meta['mode']`, so an `uncond_tt` checkpoint gets the
  time-token net. `FLAG_MODE` kept (used in the "needs --flag" message).

## Tests
- `tests/test_sensor_modes.py::test_uncond_tt_uses_time_tokens`: base-16 / 8-band nets from the same seed; state dicts
  identical (keys and values) with and without the flag; with fresh (zero) FiLM, flag on -> t=10 vs t=900 outputs differ,
  flag off -> identical; `build_residual_model('uncond_tt', 16, 8, 5)` has in_channels 8, no x0_hat / s_post / context,
  keys equal `uncond`'s and loads `uncond`'s state dict strict; mode-table membership (last in MODES, in SENSOR_MODES, not
  CENTRED / ONE_CALL); target_for returns x; batch_to_cond -> `{'sensor': None}`, scale None.
- `tests/test_sensor_training.py`: `uncond_tt` added to MODE_FLAGS (so also to the check_mode_args valid set) and to the
  parametrised two-epoch CPU training test.
- `tests/test_launch_run_sensor.py`: `test_uncond_tt_is_a_sensor_mode` (refused without --sensor_config; with it the
  args end `--sensor_config /c.json --ds 1`), `test_longer_mode_name_guard_covers_uncond_tt` (RUN_NAME=uncond_tt_x under
  mode uncond refused with "belongs to mode uncond_tt, not uncond").
- `tests/test_sd_eval.py::test_B_accepts_an_uncond_tt_checkpoint` (accepted for B, provenance mode uncond_tt, net has
  the flag; a cond_fam checkpoint is still refused for B).

RED (before implementation): 6 failed, 42 passed on the four edited test files — test_uncond_tt_uses_time_tokens
(TypeError: unexpected kwarg), test_check_mode_args_refusals (argparse rejects uncond_tt), two-epoch [uncond_tt],
both launcher tests, the sd_eval test (ValueError: mode not in MODES).
GREEN: the full requested command (test_sensor_modes, test_wrapper, test_model_extra_in, test_uq_modes,
test_regress_mode, test_sensor_training, test_launch_run_sensor, test_launch_run_name, test_sd_eval): 130 passed.

Extra check (CPU, read-only): every run directory's first checkpoint under results/residual_warmstart/runs/ with a meta
mode (25 runs, incl. uncond_b64, cond_fam_b64, regress_fam_b64, varhead_fam_b64 and all legacy modes) loads strict into
`build_residual_model(meta['mode'], ...)`.

## Files changed
model/u2net_hyperspectral.py, model/residual_wrapper.py, scripts/pilot/launch_run.sh, scripts/sensor/sd_eval.py,
tests/test_sensor_modes.py, tests/test_sensor_training.py, tests/test_launch_run_sensor.py, tests/test_sd_eval.py.

## Concerns
- With fresh FiLM the time tokens are the only t path; their content is the same token at every position (no positional
  information), so cross-attention softmax is uniform and each block's cross-attention output is a t-dependent constant
  vector per position — this is the legacy behaviour minus the sensor term, as specified.
- An `uncond_tt` and an `uncond` checkpoint have identical state-dict keys/shapes; only meta['mode'] tells them apart.
  Resume (RESUME_META_KEYS includes mode) and sd_eval (builds from meta['mode']) both rely on it, so that is safe.
- main_2d_residual_fsdp.py docstring and scripts/pilot/smoke_sensor.sh do not mention uncond_tt (not edited per scope).
