### Task 8: Launcher guards for the sensor modes

**Files:**
- Modify: `scripts/pilot/launch_run.sh` (mode list line 22, the regress block 37–63, the varhead line 64–66, the `nohup` line 91)
- Test: `tests/test_launch_run_sensor.py`

**Interfaces:**
- Produces: the mode list becomes `residual standard standard_xhat regress residual_reg varhead uncond cond_fam regress_fam varhead_fam`; for MODE ∈ {uncond, cond_fam, regress_fam, varhead_fam}: `--sensor_config <path>` must appear in the extra args (else exit 1 "needs --sensor_config"), a `--ds N` with N ≠ 1 in the extra args is refused, and the launcher appends `--ds 1` after `"$@"`; the regress guards (loss l1|l2, `_l1_` / `_l2_` consistency, seed) apply to `regress_fam` as to `regress`; `varhead_fam` needs `--loss_type nll` like `varhead`; the longer-mode RUN_NAME guard covers `regress_fam_*` under `regress` and `varhead_fam_*` under `varhead` automatically through the list.

- [ ] **Step 1: Write the failing tests** (the stub pattern of `tests/test_launch_run_name.py`)

```python
def test_sensor_mode_without_config_is_refused(env, tmp_path):        # uncond with no --sensor_config -> returncode 1, 'needs --sensor_config', no run dir
def test_sensor_mode_appends_ds_1_and_passes_config(env, tmp_path):   # RUN_NAME=cond_fam_b64 cond_fam 0 --sensor_config /c.json -> stub args end with '--sensor_config /c.json --ds 1'
def test_sensor_mode_refuses_other_ds(env, tmp_path):                 # uncond ... --sensor_config /c.json --ds 4 -> refused
def test_regress_fam_needs_loss_and_direct_like_regress(env, tmp_path):  # regress_fam without --loss_type -> refused; with --loss_type l1 and RUN_NAME regress_fam_l2_x -> refused
def test_varhead_fam_needs_nll(env, tmp_path)
def test_longer_mode_name_guard_covers_fam(env, tmp_path):            # RUN_NAME=regress_fam_b64 under mode regress -> refused 'belongs to mode regress_fam'
```

- [ ] **Step 2: Run to verify failure.** Expected: FAIL (modes unknown to the list; no `--ds 1` appended).

- [ ] **Step 3: Implement the launcher changes.**

- [ ] **Step 4: Run the new and old launcher tests.** Run: `... -m pytest tests/test_launch_run_sensor.py tests/test_launch_run_name.py -q`. Expected: all pass.

- [ ] **Step 5: Commit** — `git add scripts/pilot/launch_run.sh tests/test_launch_run_sensor.py` ; `git commit -m "feat(sensor): launch_run.sh knows the sensor modes — --sensor_config required, --ds 1 appended, regress / varhead guards extended to the family modes (Phase 3 Task 8)"`.

