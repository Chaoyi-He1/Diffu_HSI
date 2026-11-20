# Final whole-branch review — Phase 3 "diffusion reconstruction from a poor EC sensor" (41462ca..6d2db08)

Reviewer: final-review seat (read-only). Package: `.superpowers/sdd/2026-11-04-sensor-degradation-diffusion/review-41462ca..6d2db08.diff` (43 commits, 44 files, 6 839 insertions). The package was too large for one pass; it was reviewed in six passes, each file read in full at the pinned head `6d2db08` with `git show <sha>:<path>` (HEAD moved to a3bc56b, a docs commit, during the review; the package is what was judged):

1. Forward model and MMSE: `data_loader/sensor_model.py`, `sensor_forward.py`, `mmse.py`, `sensor_config.py`, `hfd_sensor_sim.py`, `scripts/sensor/make_config.py`.
2. Guidance and models: `model/guidance.py`, `scripts/sensor/toy_guidance.py`, `model/residual_wrapper.py`, the `model/u2net_hyperspectral.py` diff, `model/diffusion_trainer.py` (unchanged, read for `_model_to_x0_eps`, the DDIM branch and the min-SNR weights).
3. Trainer and launcher: `main_2d_residual_fsdp.py` diff, `scripts/pilot/launch_run.sh` diff, `scripts/pilot/smoke_sensor.sh`.
4. Harness and verdict: `scripts/sensor/sd_eval.py`, `sd_verdict.py`, `stage0_baselines.py`, with `scripts/uq_eval.py`, `scripts/uq/metrics.py`, `scripts/paired_stats.py`, `scripts/eval_warmstart.py` read for the consumed helpers.
5. Tests: all 16 new/changed test files and the `tests/conftest.py` fixture.
6. Page: `docs/analysis/phase3_page/{extract_page_data.py,build_page.py,body.html,charts.js}`, the pinned `page_data.json` status block.

Against the spec (with its as-built block), the plan, the results note and the ledger's rulings. Shared results were read but not written: `verdict__e123.json`, `config.json`, the E1 20/30 dB records (to recompute the tie-break on the common 50 cubes), the pinned `page_data.json`. Two small CPU checks were run from `/tmp` (unclamped-vs-clamped MMSE on 12 gate cubes; the tie-break recomputation). The already-ruled deviations (ΠGDM coefficient, crc32 seeds, K = 32 / 50-cube B, x₀-prediction prior, PH5 B rows dropped, B cost) were checked for faithful implementation, not re-litigated.

### Strengths

- **The forward model is exactly what the spec says, and the tests prove it with closed forms.** `mosaic_mask` implements k = (i mod 5)·6 + (j mod 6) and the one-full-tile test enumerates all 30 channels; `sigma_from_snr` is σ² = P_y/10^(SNR/10); `measure` zeroes off-mask entries and the measured-SNR test checks 30 dB to 0.2 dB over 50 cubes; `guidance_operator` H = ½RᵀA⁺, b = H·1 is checked against Rᵀx₀₁ to 1e-9 and the adjoint test closes the loop. P_y in the shipped config (32.948) is the noise-free mean y² over the 9 000 training cubes, as §4.3 asks.
- **The MMSE module is correct and tested against brute force.** `posterior_sequential` is the whitened-SVD gain (K = L V diag(S/(S²+1)) Uᵀ D) and is checked against the textbook Gaussian conditioning to 1e-9; `posterior_snapshot` is checked row by row against the sequential form with one column; the low-rank snapshot variance (`lift_mean_std_lowrank`) is pinned to the dense lift at 1e-9, which keeps the Task-3 ruling honest. The s = 3e-7 regulariser conversion ν = (s/2)² reproduces the legacy `linear_inverse` on the trainer scale.
- **Training and evaluation see the same measurement.** One class (`HFDSensorSim`) with one `measure` call serves the family (training), the fixed settings (evaluation) and Stage 0; both rngs are keyed by crc32 of the file's relative path, so the 50-cube B records and the 200-cube records carry identical noise per file. `test_noise_depends_on_the_file_not_its_position` and `test_records_on_different_subsets_pair_by_file` pin this; I confirmed on the real records that B's files are the `ddpm1000_subset` in order and the verdict paired 50 common files with 45 clusters.
- **ΠGDM is derived correctly and licensed before use.** The x₀-coordinate correction x̂₀ + (1−ᾱ_t)/√ᾱ_t·g is the exact likelihood-score correction (ε̂' = ε̂ − √(1−ᾱ)g), the DDIM step recomputes ε from the corrected x̂₀ exactly as `DiffusionTrainer.sample` does, r_t² = 1−ᾱ_t is the exact p(x₀|x_t) variance for the isotropic toy, and the sequential (30×30 solve per cube) and one-hot (scalar per pixel) covariances are right. The isotropic toy with the real Device-1 operator reproduces the exact posterior mean to 0.3 % and variance to 7 % at 50 steps in both modes; `test_pigdm_recovers_the_isotropic_gaussian_posterior` is a real closed-form test. In snapshot mode `y_eff = (y−b)⊙mask`, the residual is masked and the solve ignores unmeasured rows, so B does see the measurement there; the off-mask corruption test confirms nothing leaks.
- **The sampler is built from the checkpoint's own prediction type** (`sd_eval.check_models` → `b_prediction_type`, `trainer_B`), the ruling that made the x₀-prior records valid; `_model_to_x0_eps` is differentiable for both v and x0 and the gradient flows through the clamp as intended. The real E1 records carry `prediction_type: 'x0'` and the checkpoint sha.
- **Provenance refusals are in place and tested**: `sd_eval` refuses a checkpoint whose mode, training-config sha or prediction type does not fit (`test_refuses_checkpoint_with_other_config_sha`, `test_refuses_wrong_mode_checkpoint`, `test_B_refuses_eps_prediction_prior`); the trainer refuses to resume against another config sha or sensor device; `make_config` refuses `--limit_files` at the default path; every record writer refuses to overwrite; the verdict is taken once.
- **The verdict is pre-registered code with its constants pinned by a test** (`test_constants_are_the_preregistered_ones`), the clauses are the spec's (T1 paired m − 1.05·d with clustered SE; T2 CRPS 2-SE against L and V and cov90 in [0.85, 0.95]; T3 ratio-of-means on device2 ≤ 1.1× device1; T4 against `L_demosaic`), T1 is counted once and T2 per SNR, and the K ≥ 19 refusal protects the coverage clause. The printed verdict in the results note matches `verdict__e123.json` line for line.
- **The page is genuinely data-bound.** Every number in `body.html` except a handful of fixed constants is a `data-v` path or a slot computed from `page_data.json`, which the extractor builds from the records; the "stale" detector (`pred_type != 'x0'`) and `b_is_valid_in` (any B record newer than the verdict invalidates its B clauses) are the right guards, and they worked in anger: the E4 B record written with the wrong checkpoint was flagged and archived.
- **The ledger is unusually complete**: every deviation carries a `Ruling:` with a why and a cost-if-wrong, and the as-built block in the spec records them. The B diagnosis (sampler sound via the Wiener-filter control; prior input-blind; three recipe pilots; the x₀ pilot's pre-stated decision rule) is exemplary scientific hygiene.

### Issues

#### Critical (Must Fix)

None found. I looked specifically for anything that would make a published number wrong: the forward model, σ, the mask, H and b, both MMSE posteriors and the lift, the scale conversions, the ΠGDM math and the autograd path, the per-method metric calls in `sd_eval`, the pairing and SE in `sd_verdict`, and the clause arithmetic against the real `verdict__e123.json`. All check out.

#### Important (Should Fix)

1. **`scripts/sensor/sd_verdict.py:247` — the tie-break compares a 50-cube paired mean with a 200-cube paired mean.** `main_line` picks `min(res['T2'][l][20.0]['crps_vs_V']['mean'])`, where B's mean is over its 50 cubes and C's over 200 (spec §6.1 says every clause that *involves B* is paired on the identical 50; the tie-break is the one place the two lines are compared to each other). *Why it matters:* the main-line decision rests on a non-like-for-like comparison. *Effect on the result:* none — recomputed on the common 50 files from the records, C − V at 20 dB is −0.802 ± 0.057 (clustered) against B's −1.601 ± 0.124, so B still wins by a wide margin; at 30 dB it is −0.430 vs −0.635. *Fix:* in `main_line`, recompute `crps_vs_V` for every tied line on the intersection of all tied lines' files (the per-cube arrays are in `settings`), and record both the 200-cube and the common-file values in the verdict JSON; add a test with records of different sizes. The page's rulings bullet already discloses the 50-vs-200 asymmetry, so the write-up only needs the common-file numbers added.

2. **`scripts/sensor/sd_eval.py:83-93` and `main_2d_residual_fsdp.py:check_mode_args` — the training device is never checked against `cfg['device_train']`.** A checkpoint carries `meta['sensor_device']` (from `--sensor_device`), but `check_models` only compares the training-config sha, which hashes the constant `device_train='device1'`, not the device the run actually used; the trainer accepts `--sensor_device device2` with the same config and sha. A D/V/C checkpoint trained on Device 2 would pass every refusal and the E2 "zero-shot transfer" clause (T3) would be void. *Effect on the result:* none — all four trainings went through `launch_run.sh`, which never passes `--sensor_device`, so every checkpoint is Device 1 (the launch commands are in the results note). *Fix:* in `check_models`, refuse when `meta.get('sensor_device') != cfg['device_train']` for the `regress`/`varhead`/`cond` flags (the prior is sensor-free), and in `check_mode_args`/`make_datasets` refuse `args.sensor_device != cfg['device_train']` unless an explicit `--allow_other_train_device` is given; one test each.

3. **Plan Task 14 Step 4 / spec §10 R2 — the oracle-regressor decision is not recorded anywhere.** The spec's risk R2 ("if D's monitoring loss at 30 dB is more than 10 % above a 30 dB-only run's at epoch 25, train the oracle; otherwise the oracle run is unnecessary") and the plan's step ("otherwise record 'oracle run not needed'") have no trace in the results note, the ledger or the as-built block (grep for "oracle" finds only the spec). *Why it matters:* D is the reference of T1 and T4; the spec made "D is a weak baseline" an explicit risk with a decision rule, and a pre-registered risk control that is silently skipped is exactly what the ledger exists to prevent. The data make the answer obvious (D halves the MMSE error at 20 dB and is the best point estimate at 60–40 dB), but it has to be written down. *Fix:* one line in the Phase B section of the results note and in the ledger: "R2: oracle run not needed — D is the best point estimate at 60–40 dB and 0.50× L at 20 dB; the 30 dB-only comparison run was not trained (reason)", plus the as-built block.

#### Minor (Nice to Have)

1. **Spec §6.2 metrics not delivered and not in the ledger:** the per-class (P022/P023/P024) breakdown is not reported by `sd_verdict`, the results note or the page (the records do carry `classes`, so it is a post-hoc computation), and the "log spread–error correlation (SE-4)" is not computed by `sample_metrics`/`gaussian_metrics` at all. Spec §8's toy asks for AUSE of the exact posterior; `toy_guidance.run_toy` reports rmse_excess/var_ratio/cov50/cov90/crps_ratio only. These are report-only items with no effect on the clauses; either add them or record a ruling that drops them.
2. **`scripts/sensor/sd_eval.py:148` — L's point estimate is the unclamped MMSE mean while D, C and B are clamped to [−1, 1] by `compose`.** Measured on 12 gate cubes: the clamp would lower L's RMSE by 0.02 / 0.31 / 0.55 % relative at 60 / 30 / 20 dB (1–2 % of elements out of range). Far too small to move any clause or ratio (T3's ratios shift < 0.6 %), but the spec's anti-goal is "a baseline that was crippled", so either clamp L's point metrics or state the convention in the results note.
3. **`scripts/sensor/sd_eval.py:150`, `scripts/sensor/stage0_baselines.py:63` — `demosaic_then_mmse` uses the Device-1 `e2_demosaic` table for Device 2 and PH5.** The interpolation-error variance depends on the device's R; the spec only says "estimated once on training cubes", so this is spec-compliant, but the report-only E4 baseline (7.73 %) and Stage 0's device2/ph5 demosaic rows use a mismatched noise term. Note it in the results note or compute a per-device e2 in `make_config`.
4. **`docs/analysis/phase3_page/extract_page_data.py:186-198` — `status.final` lists a stale B record (the name is appended before the `pred_type` check), and `eval_hours`/`b_records` count stale records' hours;** `charts.js` `SL.outcome`, `SL.kpiB`, `SL.kpiBsub`, `SL.countBC`, `SL.mainline` read the verdict's B numbers without the `V.b_valid` guard that the clause table and marks use. Harmless today (the verdict is valid), inconsistent if a B record is ever re-run.
5. **Record provenance for C is thinner than for B:** `sd_eval` records `pigdm_steps`/`guidance` for B but not C's sampler (DDIM, n_steps 10, clamp (−8, 8)) or the C1 one-call rule; add them to the record.
6. **`scripts/sensor/toy_guidance.py:28-45` — `GaussianDenoiser` always emits v-prediction regardless of `trainer.prediction_type`;** with an x0 trainer the toy would silently be wrong. Assert `trainer.prediction_type == 'v'` or branch on it.
7. **`data_loader/mmse.py:24` — `np.linalg.svd(M, full_matrices=False)` with the "+1" in `S/(S²+1)` assumes S ≤ L (30 channels ≤ 31 bands);** for S > L the complement of U would be missing and the gain wrong. Add an assertion or `full_matrices=True`.
8. **Tests already recorded as deferred in the ledger that are worth the ten minutes:** `test_pigdm_snapshot_toy_runs_and_beats_the_prior` only asserts `rmse_excess < 0.5`; `test_B_is_unchanged_when_off_mask_entries_are_corrupted` uses a constant model, so it would pass if B ignored y entirely (use the `Shrink` stub, which already exists in the same file); `test_report_only_lists_missing_settings_and_writes_report_name` uses a bare `pytest.raises(Exception)`.
9. **`docs/analysis/phase3_page/body.html` keeps a few typed constants outside the data binding:** "(T2 passed)" in the KPI tile, "–13" informative channels in the lede, "113 images … 45 for B's 50" in the glossary, "seed 42" in §8. All currently true; bind or footnote them.
10. **`model/guidance.py:78-80` — `_init` draws the same x_T for seed k in every batch** (same shape), so cube i of batch 1 and cube i of batch 2 start every chain from identical noise. Harmless for a 50-step chain, but undocumented; mixing the batch index into the seed would remove the correlation.
11. **Repo hygiene:** `model/__pycache__/*.pyc` are tracked and show as modified in the working tree (pre-existing; the plan's Global Constraints say never to stage them). Untrack them in a follow-up.

### Declined to judge

- The ΠGDM coefficient (1−ᾱ_t)/√ᾱ_t, the isotropic r_t² (D5), 50 steps — ruled; I checked that the code implements the ruling (it does).
- crc32(relpath) noise seeds — ruled; implemented as written and tested.
- K = 32 everywhere and B on the 50-cube subset; the 12 GPU-h cap overrun (≈ 21 GPU-h); PH5's B rows dropped by the drop order; three records per setting — ruled; the records match.
- B's prior retrained with x₀-prediction, and `sd_eval` reading the prediction type from the checkpoint — ruled; implemented and tested (`test_B_accepts_x0_prediction_prior_and_records_it`).
- The tie-break's *definition* ("lower paired CRPS − V at 20 dB" as the plan's refinement of the spec's "T2 at 20 dB") — a pre-registered plan decision; only its implementation (Important 1) is judged.
- Reading each line's clauses on its own cubes (B on 50, C on 200), so the two lines' pass counts have different statistical power — spec §6.3's own design.
- T3 as a ratio-of-means with no SE — spec §6.3's wording.
- The legacy stretched wavelength axis, column min–max normalisation and the 30-state down-sampling — the user's decisions (Q1–Q2), verified by Stage 0's normalisation check.
- Zero-initialised FiLM layers making `uncond` t-blind at init — ruled (Task 6), legacy checkpoints depend on it.
- `uncond` computing MMSE maps it never uses, and the per-item MMSE making `regress_fam` data-bound (319 s/epoch) — D4 and a performance matter, not correctness.
- The regressor-centred design handicapping C in snapshot mode (a results interpretation the note records as out of scope).
- Checkpoint pruning (user's say-so) and the smoke script creating its temp root under the real results root (ledger minor).
- `HFD_data` loading Device 1's R inside every `HFDSensorSim` — harmless, noted by the Task 5 review.
- Whether a paper should present B (main line by tie-break, over-confident) or C as its headline — the spec says both are reported in full; the page's outcome panel states the caveat.

### Recommendations

1. Fix Important 1 and 2 before merge (both are small: a common-file recomputation in `main_line` with a test; a `sensor_device` check in `check_models` and `check_mode_args` with a test). Neither changes a published number.
2. Write the R2 line (Important 3) into the results note, the ledger and the as-built block in the same commit.
3. When the E4 B record lands and the page is rebuilt, add the common-file tie-break numbers (−1.60 vs −0.80 at 20 dB) to the outcome panel and §7, and fix the `status.final`/guard inconsistencies (Minor 4) in the same rebuild.
4. Decide on Minor 1 (per-class, SE-4, toy AUSE): either compute them from the existing per-cube arrays (cheap) or record a ruling that drops them from §6.2.
5. Note the two baseline conventions (unclamped L, Device-1 e2 for other devices) in the results note's Conventions paragraph.

### Assessment

**Ready to merge? With fixes.** The science is implemented correctly and defensively: the forward model, MMSE, ΠGDM, harness and verdict all check out against closed forms and against the real records, the pre-registered clauses were applied once and as written, and every deviation was ruled and recorded. The three Important items are a non-paired tie-break comparison (outcome unchanged when paired), an unchecked training-device field that the launch path happened to protect, and an unrecorded pre-registered risk decision; all three are an hour's work and none alters a number on the page.

### Test run

Command (CPU, GPUs hidden, from the worktree root):

```
cd /data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 /home/grads/c/chaoyi_he/Desktop/conda/envs/hsi/bin/python -m pytest tests -q -p no:cacheprovider 2>&1 | tail -15
```

Tail:

```
tests/test_sd_eval.py::test_B_accepts_x0_prediction_prior_and_records_it
  .../scripts/uq_eval.py:170: UserWarning: coverage_samples: K=2 < 3 needed for the 50% interval; empirical coverage is biased low
tests/test_sd_eval.py::test_B_accepts_x0_prediction_prior_and_records_it
  .../scripts/uq_eval.py:170: UserWarning: coverage_samples: K=2 < 19 needed for the 90% interval; empirical coverage is biased low
tests/test_train_residual.py::test_nonfinite_loss_is_skipped_without_a_step_then_aborts
  .../torch/optim/lr_scheduler.py:192: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. ...
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
553 passed, 3 warnings in 216.55s (0:03:36)
```

The whole suite (553 tests, the Phase 1/2a suites included) passed in 3 min 36 s; the dataset-marked tests ran against the real HFD root. Two auxiliary CPU checks were run from `/tmp` (not in the repo): the tie-break recomputation on the common 50 cubes and the unclamped-vs-clamped MMSE comparison; their numbers are quoted under Important 1 and Minor 2.
