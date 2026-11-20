# Task 11 report: sd_verdict.py

Implemented scripts/sensor/sd_verdict.py (constants verbatim; load_manifest, paired_on_common, ratio_clause, t1-t4, sanity, main_line, verdict, CLI) and tests/test_sd_verdict.py (13 tests: the nine named plus constants, T2 failure, full verdict, 200-vs-50 order check).
Design notes: load_manifest returns setting -> method -> (per_cube[method], files, K) (3-tuple, K kept for T2); setting keys are normalised as "device/mode/30" (float :g); records whose setting does not match the manifest key are a ValueError; ratio_clause takes per-cube metric dicts and `files` as a list or a (num_files, den_files) pair. report_only collects missing settings/methods in `missing`, skips K check, main_line None.
TDD: RED = ImportError (module missing) on first run; GREEN = `13 passed` with `CUDA_VISIBLE_DEVICES= .../envs/hsi/bin/python -m pytest tests/test_sd_verdict.py -q`.
Files: the two above only. Tests write only under tmp_path (SENSOR_DIR monkeypatched).
Concerns: in non-report mode verdict() also requires L at all five E1 SNRs (sanity). A tie in counts and CRPS-vs-V resolves to the first line (B).
