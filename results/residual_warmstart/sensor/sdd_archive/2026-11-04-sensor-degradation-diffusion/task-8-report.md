# Task 8 report
Implemented in scripts/pilot/launch_run.sh: mode list extended; sensor-mode guard (needs --sensor_config <path>|=<path>; last --ds / --ds= / --sensor_down_sample_rate[=] must be 1 else refused); DSARG=(--ds 1) appended after "$@" on the nohup line (empty-array-safe under set -u); regress guards apply to regress_fam, varhead nll guard to varhead_fam (messages now use $MODE). Header comment updated.
Tests: tests/test_launch_run_sensor.py (imports env/launch/stub_lines/wait_dead from test_launch_run_name): 69 passed with test_launch_run_name.py; bash -n clean.
TDD: RED = 17 failed (before implementation); GREEN = 69 passed.
Commit f7e33cd. Only the two files staged.
Concerns: none. Extra tests beyond brief: explicit --ds 1 accepted, accepted launches for regress_fam/varhead_fam, last-wins ds. Import uses `from test_launch_run_name import` (tests/ has no package).
