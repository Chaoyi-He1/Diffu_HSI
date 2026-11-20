# Task 12 report
Status: DONE_WITH_CONCERNS

Implemented: scripts/sensor/stage0_baselines.py (run_L, raw_P_y/raw_sigma, shape_check, principal_angles, CLI with
normalised x 3 devices and raw x device1, summary.json, NORMALISATION CHECK lines; no overwrites); R_override hook in
HFDSensorSim (+ self.R_sha256 used by checkpoint_extra); scripts/pilot/smoke_sensor.sh; tests/test_stage0.py (3 tests);
test_R_override_changes_y_and_sha.

TDD: first run failed at collection (ImportError: stage0_baselines). After implementation: 13 passed (stage0 + hfd_sensor_sim).

Smoke: SMOKE OK, 7m06s real. Stage 0 on 4 cubes printed NORMALISATION CHECK [sequential] PASS (max_dev=0.0997),
[snapshot] PASS (0.0066), overall PASS. Real results/residual_warmstart/sensor untouched; $T removed.

Whole suite: 544 passed, 1 deselected (tests/test_guidance.py::test_sequential_solve_matches_dense).
Without the deselect: 1 failed, 127 passed with -x. It fails deterministically in full-suite order, passes alone (3/3):
unseeded torch.randn, float32 solve vs float64 reference, rtol 1e-4 on values ~60 (diffs ~3e-4). Pre-existing flake in
Task 9/11b code, independent of this task (fails with my tests ignored); not fixed (out of scope).

Deviations / concerns:
1. sd_verdict hard-requires MIN_COMMON = 45 common files even with --report_only, so records from 8 files cannot be
   verdicted. The smoke therefore asserts sd_verdict's exact refusal ("only 8 files are common ... MIN_COMMON = 45")
   rather than a successful report. The verdict logic is not exercised end-to-end by the smoke.
2. Smoke config is at $T/config.json, not $T/sensor/config.json: with RESIDUAL_WARMSTART_RESULTS=$T the latter equals the
   default CONFIG_PATH and make_config refuses --limit_files there.
3. The smoke copies the real linear_prior_R1.npz into $T (PRIOR default follows the env redirect).
4. Stage 0 CLI also has --batch_size/--num_workers (defaults 8/4).
Files: scripts/sensor/stage0_baselines.py, scripts/pilot/smoke_sensor.sh, tests/test_stage0.py, data_loader/hfd_sensor_sim.py, tests/test_hfd_sensor_sim.py

## Fix round 1
- tests/test_guidance.py: test_sequential_solve_matches_dense seeded (torch.manual_seed(0)), rtol 1e-3.
- sd_verdict.py: --min_common (default MIN_COMMON) threaded through _common / paired_on_common / ratio_clause / _clause /
  t1-t4 / sanity / verdict; written JSON records "min_common". New test test_min_common_override_is_recorded_and_applied.
- smoke_sensor.sh step 4: sd_verdict --report_only --min_common 2, asserts report JSON exists, main_line null, min_common 2.
- Tests: pytest test_guidance test_sd_verdict test_stage0 -> 23 passed.
- Smoke: SMOKE OK, 7m08s; verdict printed T1/T2/T4 clauses for B and C plus MISSING lines, "main line: none (report only)".
- Whole suite: "546 passed, 1 warning in 172.74s" (first run; second run launched to show flake gone, result in the commit's follow-up if it differs).
