### Task 11: The verdict `sd_verdict.py` — T1–T4 from a manifest

**Files:**
- Create: `scripts/sensor/sd_verdict.py`
- Test: `tests/test_sd_verdict.py`

**Interfaces:**
- Consumes: `paired_stats.cluster_se`, `metrics.min_k_for`, the `sd_eval` record format.
- Produces: thresholds as module constants `T1_RATIO = 1.05`, `T2_COV90 = (0.85, 0.95)`, `T2_SNRS = (30.0, 20.0)`, `T3_RATIO = 1.10`, `T3_SNR = 30.0`, `T4_RATIO = 0.85`, `T4_SNR = 30.0`, `MIN_COMMON = 45`; `MANIFEST_KEYS`: a manifest JSON `{"records": {"<device>/<mode>/<snr>": ["<record name>", ...]}, "name": "<verdict name>"}` (several records per setting are merged by method; a method present twice is an error); `def load_manifest(path) -> dict[str, dict[str, tuple[dict, list[str]]]]` (setting → method → (per_cube dict, files)); `def paired_on_common(a: dict, af: list, b: dict, bf: list, key_a: str, key_b: str) -> dict(mean, se, se_cluster, n, clusters, ratio_of_means)` (restricts both to the common files in `af`'s order; raises ValueError when fewer than `MIN_COMMON` are common); `def ratio_clause(num, den, files, key_num, key_den, limit, direction) -> dict(passed, ratio, ratio_2se, mean, se_cluster, n)` — the paired test m − limit · b with its clustered SE: `direction='le'` passes when mean + 2 se_cluster ≤ 0; `def t1(settings, line) -> dict[snr -> clause]` (line ∈ {'B', 'C'}; `mean_rmse_pct` of the line vs `rmse_pct` of D over E1's five SNRs); `def t2(settings, line) -> dict[snr -> dict(crps_vs_L, crps_vs_V, cov90, passed)]` at T2_SNRS (CRPS 2-SE clauses against `L` and `V`; cov90 of the line's record within T2_COV90; needs `K >= min_k_for(0.9)` of that record else ValueError unless `--report_only`); `def t3(settings, line) -> dict(ratio_dev1, ratio_dev2, passed)` ((RMSE_line / RMSE_L) on `device2/sequential/30` ≤ T3_RATIO × the same on `device1/sequential/30`, each a ratio of means of `mean_rmse_pct` on common files; PH5's ratio reported under `ph5`); `def t4(settings, line) -> clause` (line's `mean_rmse_pct` vs `L_demosaic`'s `mean_rmse_pct` on `device1/snapshot/30`, limit T4_RATIO, plus D's for the report); `def sanity(settings) -> dict(monotone: dict[method -> bool], floor_reference_rmse_pct, floor_pass: dict[method -> bool])` (spec §6.3: over E1's five SNRs every method's RMSE (`mean_rmse_pct` for L, V, C, B; `rmse_pct` for D, C1) is non-increasing in SNR; at 60 dB every learned method's RMSE ≤ L's + 2 clustered SE of the paired difference); `def main_line(res: dict) -> str` (count passes of T1 (all five SNRs count as one pass when all pass), T2 (both SNRs), T3, T4 per line; more passes wins; tie → lower CRPS vs V at 20 dB); `def verdict(settings, report_only=False) -> dict(T1, T2, T3, T4, sanity, counts, main_line, report_only, missing)`; CLI `python scripts/sensor/sd_verdict.py --manifest <path> [--report_only]` writes `os.path.join(SENSOR_DIR, 'verdict__' + name + ('__report' if report_only else '') + '.json')`, refusing to overwrite; with `--report_only` missing settings are listed instead of raising and `main_line` is None.

- [ ] **Step 1: Write the failing tests** (synthetic records built by a helper `_rec(files, **method_arrays)` in the sd_eval format)

```python
def test_paired_on_common_restricts_to_common_files_and_refuses_small_overlap()   # 200 vs 50 files -> n == 50 in the 50's order; 200 vs 30 -> ValueError
def test_ratio_clause_pass_and_fail()                                             # line = 1.02 D everywhere -> T1 passes; 1.08 D -> fails
def test_t2_requires_K_19_unless_report_only()
def test_t3_uses_the_device1_reference_ratio()                                    # ratios 0.8 / 0.86 -> pass (1.075 <= 1.10); 0.8 / 0.9 -> fail
def test_t4_against_demosaic()
def test_main_line_counts_and_tiebreak()                                          # B passes T1-T4, C passes T1,T2 -> 'B'; equal counts -> lower CRPS vs V at 20 dB
def test_sanity_monotone_and_floor()                                               # a method whose RMSE rises from 50 to 60 dB -> monotone False; learned method above L + 2 SE at 60 dB -> floor_pass False
def test_duplicate_method_in_a_setting_is_an_error()
def test_report_only_lists_missing_settings_and_writes_report_name(tmp_path)
```

- [ ] **Step 2: Run to verify failure.** Expected: FAIL, module missing.

- [ ] **Step 3: Implement `scripts/sensor/sd_verdict.py`** per Interfaces.

- [ ] **Step 4: Run to verify pass.** Expected: 9 passed.

- [ ] **Step 5: Commit** — `git add scripts/sensor/sd_verdict.py tests/test_sd_verdict.py` ; `git commit -m "feat(sensor): sd_verdict — pre-registered T1–T4 from a manifest of records, pairing on common files, main-line rule (Phase 3 Task 11)"`.

