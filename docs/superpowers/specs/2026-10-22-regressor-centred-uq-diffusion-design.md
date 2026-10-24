# Regressor-Centred Residual Diffusion for Calibrated Uncertainty — Design Spec (Phase 2a)

Date: 2026-10-22. Branch: `worktree-phase1-plan`. Status: approved in conversation (three design sections, 2026-10-22); awaiting the user's review of this written spec. Predecessor: `docs/superpowers/specs/2026-10-08-residual-warm-start-design.md` (Phase 0/1) and the B-track record `docs/superpowers/plans/2026-10-19-b-track-prereg.md`.

## 1. Motivation

The Phase 1 question "can a one-step residual estimator beat the linear baseline by 15 % at d = 4?" is answered, by a regressor rather than by diffusion: with checkpoint selection the one-call L1 regressor reaches 1.9621 % (seed 42) and 1.9516 % (seed 43) on the 200 gate cubes against the baseline's 2.3512 % (go_1b true, 2026-10-19). Under RMSE a sampler never beats its own mean, and B's chain mean has a ceiling of 2.108 % on the gate, below which the regressor already sits. So for the RMSE goal diffusion adds nothing, and the user has re-aimed it (2026-10-22) at what a point estimate cannot give.

Two measurements say where that is:

- **The task is spatial fill-in of a known spectrum.** On the 800 monitoring files at d = 4 the baseline's error on the measured 1/16 of pixels is 0.266 % rms, the d = 1 figure; 99.9 % of its error energy is on the interpolated 15/16 (2.491 % rms), and 0 % of the regressor's gain is spectral (`diagnostics/regress_spatial_split_monitoring.json`).
- **At d = 8 the point estimate stops being informative.** 63 of 64 pixels are interpolated; the selected regressor (`regress_l1_sel_d8_b64` epoch 11) scores 4.2319 ± 0.1016 % against the baseline's 4.5799 ± 0.1130 %, a 7.6 % gain (half of d = 4's 16.5–17 % in relative terms, the same ≈ 0.35 pct in absolute terms), the epoch-100 model is worse than the baseline, and even the y-zeroed network beats the baseline (4.45 %): the conditional mean recovers little of a wide spatial posterior and blurs the rest.

A sampler can report that width: calibrated per-pixel uncertainty and plausible textures where the mean is blurred. Whether it actually does, against the cheapest competitor that also reports uncertainty (a regressor with a variance head), is the question this spec answers.

## 2. Agreed Decisions (do not re-open)

Carried over from Phase 0/1 (user): no noise added to the sensor data; the HFD_data loader is not modified; strides d = 2/4/8 with d = 4 first; train on this server (2 × RTX A4500) with the existing FSDP trainer; never share a GPU with the user's `dm_snn` sweep; Phase 1a / 1a′ / B-track records are never overwritten.

New (user, 2026-10-22):

- Direction: keep diffusion, re-aimed at samples / uncertainty / perceptual quality in the wide-posterior regime (**part (a)**, this spec), then a prior reusable across sensors and strides (**part (b)**, a later spec).
- Success criteria for (a): **(1) calibrated per-pixel uncertainty that beats a heteroscedastic (variance-head) regressor; (2) perceptual / texture quality of single samples versus the regressor mean**, RMSE being allowed to be worse (perception–distortion). Downstream tasks and qualitative-only were not chosen.
- Approach: **regressor-centred residual diffusion with cross-fitting** (approach 1 of three), d = 8 as the main experiment, d = 4 as a contrast; a zero-training Stage 0 fixes the evaluation harness first.
- K = 32 samples per cube at d = 8 (K = 16 at d = 4) — the one correction to the approved design: with K = 16 the empirical 5 % / 95 % quantiles are the sample minimum and maximum and the 90 % coverage is biased low.
- Checkpoints of the four selection runs were pruned on 2026-10-22 to the selected epoch + 25/50/100 (253 GiB freed); /data has ≈ 400 GB free.

## 3. Notation

x ∈ [−1, 1]^{64×64×64} the true cube (trainer scale; metrics are reported on the [0, 1] display scale in %), y the 30-channel sensor image point-sampled at stride d, x̂₀ the linear baseline (Phase 0), x̂_R the regressor's estimate, σ_d the residual scale of x − x̂₀ (0.051661 at d = 4, 0.095731 at d = 8), σ_R the residual scale of x − x̂_R defined in §4.3. Gate cubes: the 200 files of `tests/data/val_subset_200.json` (113 source images, classes P022/P023/P024); monitoring split: the other 800 validation files. Source image of a file `MatFlower60/Train/P0xx/<img>_<patch>.mat` is `P0xx/<img>`; all statistics are paired per cube and reported with an iid SE and an SE clustered by source image.

## 4. Method

### 4.1 Cross-fitting split

The 9 000 training cubes are split into halves H₁ and H₂ **by source image**: source images are grouped by class, shuffled with seed 0 within each class, and assigned alternately to H₁ and H₂, so no source image has patches in both halves and every class is split within ±1 image. The split is written once to `results/residual_warmstart/uq/split_halves.json` (`{"seed": 0, "half1": [files], "half2": [files], "images_per_class": {...}}`) and is the same for every stride.

### 4.2 Half regressors

R₁ trains on H₁ only, R₂ on H₂ only, with the B.1 recipe: mode `regress`, L1, `--prediction_type direct`, seed 42, base 64, batch 8, `--val_every 1 --save_every 1`, **40 epochs** (every selection minimum so far lies at epoch 11–18), `--keep_last 40`; the trainer gains `--train_half {1,2}` which filters `train.img_list` to the half and records it in `args.json` and in the checkpoint meta. Selection is the B.1 rule: the lowest monitoring loss, ties → earlier. Run names `regress_l1_half1_d8_b64`, `regress_l1_half2_d8_b64` (and `_d4_`); `launch_run.sh` refuses a `_half<N>_` name whose `--train_half` differs, exactly like its `_s<N>_` seed guard.

### 4.3 Cross-fitted centre cache and σ_R

`scripts/uq/build_xr_cache.py --d 8 --half1 <R₁ ckpt> --half2 <R₂ ckpt> --full <selected full regressor ckpt>` writes, for every training cube, x̂_R from the regressor that did **not** see it (H₁ cubes ← R₂, H₂ cubes ← R₁), and for every monitoring cube x̂_R from the **full** selected regressor (the test-time centre, `regress_l1_sel_d8_b64` epoch 11 at d = 8; `regress_l1_sel_d4_b64` epoch 17 at d = 4), as fp16 `.npy` of shape [64, 64, 64] under `uq/d<d>/xr_cache/train/<relpath>.npy` and `uq/d<d>/xr_cache/val/<relpath>.npy` (≈ 4.4 GiB + 0.4 GiB per stride). Its `meta.json` records d, the three checkpoints (path, run, epoch, sha256, their `train_half`), git head, and:

- **σ_R** = the standard deviation of all elements of x − x̂_R over the 9 000 cross-fitted training cubes on the trainer scale, computed exactly as `scripts/compute_residual_scale.py` computes σ_d;
- σ_full = the same statistic for the full regressor on the 800 monitoring cubes, and the ratio σ_full / σ_R. **A ratio outside [0.9, 1.1] is flagged in the record and in the results note** (the training residual distribution would then differ from the test-time one); the pipeline continues.

### 4.4 Residual diffusion `residual_reg`

Same network, wrapper and trainer as B (`U2NetHyperspectral`, `ResidualWrapper`, `DiffusionTrainer`): input `cat([x_t, x̂_R])` (extra_in_channels = 64), sensor y through the existing cross-attention context, **target r = (x − x̂_R) / σ_R**, v-prediction, L1, 1 000 linear-β timesteps, 100 epochs, seed 42, base 64, batch 8, `--save_every 10 --keep_epochs 25 50 100 --val_every 5` (the launcher's defaults). x̂₀ is not an input (the regressor already consumed it). The cond dict gains the key `x_r`; the dataset wrapper `XRCachedData` (§6) supplies it from the cache. The checkpoint meta records mode, d, σ_R, the cache meta's sha256 and the centre checkpoint's run/epoch/sha256. Sampling: `trainer.sample(method='ddim', n_steps=10)` from pure noise, K independent draws with seeds 0 … K−1; composition **x_k = clamp(x̂_R + σ_R · r_k, −1, 1)**. DDPM-1000 on the existing `ddpm1000_subset` is the reference sampler. Run names `residual_reg_d8_b64`, `residual_reg_d4_b64`.

### 4.5 Variance-head baseline `varhead`

Same network and wrapper, input `cat([x̂_R, x̂_R])` (x̂_R in the 64 channels B reserves for x_t, as `regress` puts zeros there) at the constant t = REGRESS_T, one deterministic call; the 64 output channels are **log σ²** of the standardised residual r = (x − x̂_R) / σ_R per pixel-band; loss = Gaussian negative log-likelihood ½ (log σ² + r² e^{−log σ²}) averaged over elements, the mean being fixed at x̂_R. Trained on the **same cross-fitted cache** (so both methods see identical residual distributions), 40 epochs, selection by the lowest monitoring NLL (ties → earlier). Its predictive distribution is N(x̂_R, σ_R² σ²) per element, clamped to [−1, 1] when intervals are formed. Run names `varhead_d8_b64`, `varhead_d4_b64`.

### 4.6 Deep-ensemble baseline

Members at d = 4: the selected `regress_l1_sel_d4_b64`@17, `regress_l1_sel_s43_d4_b64`@17 and `regress_l2_sel_d4_b64`@18 (3 members, already on disk). At d = 8: `regress_l1_sel_d8_b64`@11 plus one new seed-43 selection run (`regress_l1_sel_s43_d8_b64`, the B.5 recipe at `--ds 8`, optional but planned). The ensemble's predictive distribution is N(member mean, member variance) per element, with the member variance's SE acknowledged for 2–3 members. The ensemble is a reported comparison, not a gate.

### 4.7 Test-time centre

At evaluation the centre is always the **full** selected regressor of that stride (§4.3), for the diffusion, the variance head and the perception–distortion table alike; the record names it. A sensitivity row uses the two half regressors' average as the centre instead, to show how much the cross-fit/test mismatch matters.

## 5. Evaluation Protocol

### 5.1 Data and sample budget

All methods are evaluated on the 200 gate cubes with their full y-swap batch (32 cubes, the existing `yswap_files` rule). Model selection only ever reads the monitoring split. K = 32 at d = 8, K = 16 at d = 4, DDIM-10, seeds 0 … K−1 fixed per cube index; the DDPM-1000 reference uses K = 8 on the ddpm subset. Cost at d = 8: 200 × 32 × 10 = 64 000 network calls, ≈ 40 min on one A4500.

### 5.2 Predictive objects

- **Diffusion:** the K samples x_k; derived: per-element mean μ̂ and std ŝ (ddof = 1), empirical quantiles q_α (linear interpolation), the point estimates "mean-of-K" and "single sample" (k = 0).
- **Variance head:** N(x̂_R, σ_R² σ²(x̂_R, y)) per element.
- **Ensemble:** N(member mean, member variance).
- **References:** x̂_R (the full regressor), x̂₀ (linear baseline), and at d = 4 B's `residual_d4_b64`@100 samples (K = 16, DDIM-10) as the "linear-centred" comparison.

### 5.3 Calibration metrics (all per element on the [0, 1] display scale, averaged per cube, then paired across cubes)

- **CRPS.** Sample estimator (fair form): CRPS = (1/K) Σ_k |x_k − x| − (1 / (2 K (K − 1))) Σ_{j ≠ k} |x_j − x_k|. Gaussian closed form for the variance head and the ensemble: CRPS = σ [ z (2 Φ(z) − 1) + 2 φ(z) − 1/√π ], z = (x − μ)/σ. Reported in %.
- **Coverage** of the central 50 % and 90 % intervals: empirical [q_0.25, q_0.75] and [q_0.05, q_0.95] of the K samples for the diffusion (plus the Gaussian-fit intervals μ̂ ± 0.674 ŝ and μ̂ ± 1.645 ŝ as a secondary line), μ ± 0.674 σ and μ ± 1.645 σ for the Gaussian competitors; reported as the fraction of elements covered.
- **AUSE** (area under the sparsification error): per cube, elements are ranked by predicted std (ŝ, or σ) descending; for removal fractions f = 0, 0.02, …, 0.98 the RMSE of the remaining elements is divided by the full RMSE; the oracle ranks by |error|; AUSE = the mean over f of (predicted curve − oracle curve). 0 is perfect.
- **Spread / error energy ratio** = mean predicted variance / mean squared error over the cube's elements (1 for a calibrated predictor), and the per-cube correlation between predicted and actual error energy — the Phase 1a′ `diagnose_b.py` quantities, now per method.

### 5.4 Perceptual / texture metrics (per cube, band-averaged, on the [0, 1] scale)

- **High-frequency energy ratio.** E(u) = mean over bands and pixels of (L ∗ u)², L the 3 × 3 Laplacian [[0, 1, 0], [1, −4, 1], [0, 1, 0]] with reflect padding; r_HF = log E(x̂) / E(x). 0 is perfect; r_HF < 0 means blur, > 0 over-sharpening. The gate statistic is |r_HF|.
- **Gradient-magnitude W1.** Sobel gradient magnitude per band, pooled over bands into a 100-bin histogram on [0, 0.5]; W1 = Σ_bins |CDF(x̂) − CDF(x)| × bin width.
- Both are computed for the regressor mean x̂_R, the diffusion mean-of-K, the single sample, B's single sample (d = 4) and the baseline, and tabulated with RMSE as the **perception–distortion table**.

### 5.5 Point-estimate metrics, sanity and validity

RMSE %, PSNR and SAM as in `eval_warmstart.py` for every point estimate (regressor, mean-of-K, single sample, baseline). Sanity S1: the mean-of-K must not drift from the centre. Validity S2: the y-swap rule of Phase 1a′ applied to the mean-of-K at the sampler's start (`x_t = noise`, the leak-free form): swapping y for another cube's y must worsen RMSE.

### 5.6 Statistics

Every comparison is a paired per-cube difference with its iid SE (n = 200) and its SE clustered by source image (113 clusters, the `paired_stats.py` formula). "By 2 SE" always means the clustered SE unless stated.

### 5.7 Pre-registered thresholds (d = 8; fixed here before any Stage 1 number is seen)

| id | clause | rule |
|---|---|---|
| C1 | CRPS: diffusion beats the variance head | mean(CRPS_diff − CRPS_varhead) + 2 SE_cluster < 0 |
| C2 | coverage | 90 % empirical coverage ∈ [0.85, 0.95] and 50 % ∈ [0.45, 0.55] |
| P1 | texture: a single sample is less blurred than the regressor mean | mean(|r_HF|_sample − |r_HF|_regressor) + 2 SE_cluster < 0 |
| P2 | texture is not bought with garbage | mean(RMSE_sample − RMSE_baseline) + 2 SE_cluster < 0 |
| S1 | the centre is preserved | |mean(RMSE_meanK − RMSE_regressor)| ≤ 2 SE_cluster |
| S2 | validity | on the 32 y-swap cubes, mean(RMSE_swap_full − RMSE_true) − 2 SE > 0 for the mean-of-K |

**Calibration passes = C1 ∧ C2. Perceptual passes = P1 ∧ P2. "Diffusion wins (a)" = both passes ∧ S1 ∧ S2.** AUSE, the ensemble comparison, the spread ratio, the DDPM-1000 reference and the half-regressor-centre sensitivity row are reported without thresholds. If calibration fails, exactly one pre-registered fix round is allowed — K = 64, and DDPM-1000 on the ddpm subset — with the same thresholds; if it still fails, the result is reported as a failure. If the perceptual clause fails while calibration passes, that is the result. The d = 4 contrast uses the same metrics with no thresholds; the expectation to be checked is that the CRPS gap and the texture gap are smaller than at d = 8.

## 6. Components and Interfaces

- `scripts/uq/split_halves.py` → `uq/split_halves.json` (§4.1); refuses to overwrite.
- `main_2d_residual_fsdp.py`: `--train_half {1,2}` (reads the split JSON, filters `train.img_list`, asserts the count); `--xr_cache <dir>` (required for modes `residual_reg` and `varhead`; the dataset is wrapped in `XRCachedData`, σ_R is read from the cache meta and written into the checkpoint meta together with the cache sha256); `check_mode_args` learns the two modes (`residual_reg` needs `--prediction_type v`; `varhead` needs `--prediction_type direct`, its loss is fixed to Gaussian NLL and `--loss_type` is refused); monitoring loss for `varhead` is the NLL.
- `data_loader/xr_cache.py`: `XRCachedData(base, cache_dir, split)` returning `(x, y, x0_hat, x_r)`; `xr_collate_fn`; `read_meta(cache_dir)`; refuses a cache whose meta `d` differs from the dataset's or whose files are missing.
- `model/residual_wrapper.py`: `MODES += ('residual_reg', 'varhead')`; `build_residual_model` gives both extra_in_channels = 64; `target_for('residual_reg', x, x_r, sigma_r) = (x − x_r)/σ_R`; `varhead_forward(model, cond)` (the constant-t call) and `varhead_nll(logvar, r)`; `compose(mode, centre, sigma, r)`.
- `scripts/uq/build_xr_cache.py` (§4.3).
- `scripts/pilot/launch_run.sh`: name regex accepts `residual_reg_…` and `varhead_…`; `_half<N>_` guard.
- `scripts/uq/metrics.py`: pure functions `crps_samples`, `crps_gaussian`, `coverage`, `ause`, `spread_ratio`, `hf_energy_ratio`, `grad_w1`, each unit-tested (§8).
- `scripts/uq_eval.py --d 8 --diffusion <ckpt> --centre <ckpt> --varhead <ckpt> [--ensemble <ckpts…>] [--b <ckpt>] --K 32 --sampler ddim10 [--centre_halves <ckpt> <ckpt>]`: draws the samples (GPU), computes every §5 quantity per cube for every method, and writes one record `uq/d<d>/eval/<name>.json` with per-cube arrays, files, classes, the y-swap block and provenance (every checkpoint's run/epoch/sha256, cache sha, σ_R, σ_full ratio, git head, K, sampler, seeds). Refuses to overwrite. Samples themselves are not stored except for 8 fixed gate cubes (fp16, for figures).
- `scripts/uq_verdict.py <record>`: applies §5.7, prints the table, writes `uq/d<d>/verdict__<name>.json`; refuses a record whose centre is not the pre-registered selected checkpoint of that stride.
- Results tree: `results/residual_warmstart/uq/{split_halves.json, d8/, d4/}` with `xr_cache/`, `eval/`, `verdict__*.json`; runs stay under `runs/` with the names above.

## 7. Phases, Budgets and Gates

- **Stage 0 — harness (no training, ≈ 2 days).** `metrics.py` + tests; `uq_eval.py` + `uq_verdict.py` + tests on stub models; a dry run on d = 4 assets (B@100 samples K = 16, the 3-member ensemble, `regress_l1_sel_d4_b64`@17 as centre; no `residual_reg`/`varhead` yet) written as `uq/d4/eval/stage0_*.json`. Gate: all tests pass; the dry-run record's regressor RMSE reproduces 1.9621 % and its baseline 2.3512 %. Stage 0 looks at no d = 8 gate number.
- **Stage 1 — d = 8 (≈ 2 GPU-days wall).** Split → R₁, R₂ in parallel (≈ 1.5 h each at 4 500 cubes, 40 epochs) → selection → cache + σ_R (≈ 20 min GPU) → `residual_reg_d8_b64` (≈ 8.5 h) ∥ `varhead_d8_b64` (≈ 3 h) → `regress_l1_sel_s43_d8_b64` (≈ 7.5 h, optional, on the free GPU) → `uq_eval.py` (≈ 1 h) → `uq_verdict.py` once. Gate: §5.7. Disk: ≈ 5 GiB cache + ≈ 7 GiB checkpoints.
- **Stage 2 — d = 4 contrast (≈ 1 GPU-day).** Same recipe at `--ds 4`; the ensemble exists; no thresholds.
- **Stage 3 — write-up.** A results note next to this spec and a GDN-style page (the user's preferred report form) with the perception–distortion table, calibration curves, sparsification curves and the 8 stored cubes' sample/mean/std figures.
- Calendar: ≈ 1.5 weeks. GPUs are used only when no `dm_snn` process is on them.

## 8. Validation, Testing and Error Handling

- Split: no source image in both halves; every training file assigned exactly once; per-class image counts differ by ≤ 1; deterministic for seed 0; the JSON refuses overwrite.
- Cache: meta contract (d, three checkpoints with sha256 and `train_half`, σ_R, σ_full, git head); `XRCachedData` refuses a missing file, a wrong `d`, or a cache built from a regressor that saw the cube (checked by `train_half` against the split); a round-trip test writes a 4-cube cache from a stub and reads it back bit-exactly.
- Modes: `target_for` and `compose` round-trip; `varhead_nll` equals the closed form on a known case and is minimised at log σ² = log r²; `build_residual_model` shapes for both modes; `check_mode_args` refuses `residual_reg` without `--xr_cache`, `varhead` with `--loss_type`, and a `--train_half` without a split.
- Trainer: CPU smoke (16 files, 1 epoch) for `regress --train_half 1`, `residual_reg` and `varhead`, checking meta fields and the monitoring NLL; resume keeps σ_R.
- Metrics: `crps_samples` at K = 2 000 agrees with `crps_gaussian` within 1 % on Gaussian data; `coverage` of Gaussian samples is within 0.02 of nominal at K = 32 on 10⁵ elements; `ause` is 0 for the oracle ranking and > 0 for a random ranking; `hf_energy_ratio` is 0 for identical cubes and < 0 for a blurred copy; `grad_w1` is 0 for identical cubes and > 0 for a blurred copy; everything handles the clamp to [−1, 1].
- `uq_eval.py` end-to-end on a stub model (CPU, 8 cubes, K = 4): record schema, refusal to overwrite, provenance; `uq_verdict.py` on synthetic records exercising every clause both ways and the centre-checkpoint refusal.
- `launch_run.sh`: name/arg guards for `_half<N>_`, `residual_reg_`, `varhead_` (tests in `test_launch_run_name.py`).
- Error handling: every script refuses to overwrite a record; a σ_full / σ_R ratio outside [0.9, 1.1] is flagged, not fatal; a `residual_reg` or `varhead` checkpoint whose meta cache sha differs from the cache on disk is refused by `uq_eval.py`.

## 9. Out of Scope

LPIPS / FID (pretrained on RGB natural images), downstream classification or detection, sensor noise, loader changes, d = 2, E1/E2 ensemble gate looks (still awaiting the user's go), and part (b) — the reusable cross-sensor / cross-stride prior, which gets its own spec and reuses this harness as its benchmark.

## 10. Risks

- **Cross-fit mismatch.** Half-data regressors leave larger residuals than the full one; the diffusion then learns a slightly too-wide residual. Mitigation: σ_full / σ_R reported and flagged; the half-regressor-centre sensitivity row; C2 would show over-coverage.
- **The mean drifts.** A residual model whose samples do not average to zero degrades the centre. Mitigation: S1; if S1 fails the diffusion result is reported with the drift, not re-tuned.
- **Too few samples.** K = 32 gives coarse 5 % / 95 % quantiles. Mitigation: the Gaussian-fit intervals as a secondary line; the pre-registered K = 64 fix round.
- **A weak competitor.** An under-trained variance head makes C1 easy. Mitigation: NLL-based selection, its own coverage reported, and the ensemble as a second competitor.
- **Compute.** ≈ 3 GPU-days total; both GPUs are free now, but the `dm_snn` rule stands.

## 11. File Manifest

Create: `scripts/uq/__init__.py`, `scripts/uq/split_halves.py`, `scripts/uq/build_xr_cache.py`, `scripts/uq/metrics.py`, `scripts/uq_eval.py`, `scripts/uq_verdict.py`, `data_loader/xr_cache.py`, `tests/test_uq_split.py`, `tests/test_xr_cache.py`, `tests/test_uq_modes.py`, `tests/test_uq_metrics.py`, `tests/test_uq_eval.py`, `tests/test_uq_verdict.py`, `scripts/pilot/smoke_uq.sh`, `docs/superpowers/plans/2026-10-22-regressor-centred-uq-diffusion.md` (the implementation plan, written after this spec is approved).
Modify: `main_2d_residual_fsdp.py` (`--train_half`, `--xr_cache`, the two modes, meta), `model/residual_wrapper.py`, `scripts/pilot/launch_run.sh`, `tests/test_launch_run_name.py`, `tests/test_regress_mode.py` (mode table), `data_loader/warmstart_paths.py` (`UQ_DIR`).
Never: `data_loader/HFD_dataset.py`, anything under `results/residual_warmstart/{eval,diagnostics,runs}` from earlier phases.

## As built (plan `docs/superpowers/plans/2026-10-22-regressor-centred-uq-diffusion.md`)

- D1: the cached centre x̂_R travels in the existing `x0_hat` condition slot (`XRCachedData` returns `(x, y_d, x_r)` with `sigma_d = σ_R`); no separate `x_r` key. The checkpoint meta carries `sigma_r`, `centre`, `xr_cache_sha256`, `split_sha256`, `train_half`.
- D2: `varhead` takes `--loss_type nll` explicitly (`l1`/`l2` refused for it, `nll` refused for every other mode).
- D3: the cache's `val/` part holds all 1 000 test-split cubes centred by the full regressor; σ_full uses the 800 monitoring files; `uq_eval.py` recomputes x̂_R for the gate cubes in float32.
- D4: `uq_eval.py` runs without `--diffusion` / `--varhead` for the Stage 0 dry run.
- σ_R is the RMS (not the std) of the cross-fitted residual elements, the exact definition of σ_d.
- R6: a half regressor's checkpoint meta records `split_sha256` of the split it was trained with; resume and the cache builder refuse a mismatch.
- R8: the cache builder refuses a regressor whose `sigma_d` differs from the dataset's or whose `prior_sha256` differs from `--prior`; the cache meta records `prior_path` / `prior_sha256`; the split's halves are checked up front.
- R9/R10 (amends §5.2's linear interpolation, §5.3, and the K of §2, §5.1 and §7): empirical coverage uses Weibull quantile positions (k/(K+1)) so a calibrated sampler's expected coverage equals the nominal level; this holds only for K ≥ `min_k_for(level)` (19 for the 90 % interval), below which `coverage_samples` warns and `uq_verdict` refuses a verdict (`--report_only` turns the refusal into a WARNING); therefore **K = 32 at both strides and in the Stage 0 dry run** (the spec's K = 16 at d = 4 is withdrawn); a deterministic ≈ +0.009 interpolation residual remains at the 90 % level for Gaussian posteriors.
- R11 (amends §6): the stored figure cubes are `--store_cubes` evenly spaced cubes of the class-stratified y-swap batch (record key `stored_files`), not the first gate files; `uq_eval` also checks the sampler parameterisation (`prediction_type` v, 1000 steps) of the diffusion and B and the prior sha of every checkpoint, and records the cache's σ_full / σ_R ratio and flag in `provenance.xr_cache`.
- `uq_eval` also refuses a `--centre`, ensemble member or `--b` with `train_half ≠ 0`, a one-member ensemble and an existing `<name>__cubes.npz`; the half regressors are therefore never ensemble members (the CPU smoke `scripts/pilot/smoke_uq.sh` uses a second full-data seed as its second member).
- R12: `uq_verdict` runs the centre and K checks before the exists check; the reported AUSE key is `C3`.
- S2 uses the iid SE over the 32 y-swap cubes, as plan Task 9 specified (an amendment to §5.6's clustered-SE default); the clustered SE is within about 2 % of it, and S2 fails under either.
- Also: the mean CRPS of N(0,1) against N(0,1) is 1/√π (a plan test constant was corrected).
