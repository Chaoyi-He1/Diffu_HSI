# Task 9 report: Guidance samplers (ΠGDM, DPS) and the Gaussian toy

Status: DONE_WITH_CONCERNS (one deliberate deviation from the brief, see "Deviation")
Commit: e4b208f feat(sensor): ΠGDM and DPS guidance over the linear measurement ... (Phase 3 Task 9)

## Implemented (formulas as implemented)

`model/guidance.py`
- `LinearMeasurement(H, b, y, mask, sigma_y)`: casts H, b to y's dtype/device once. Mask None or all-ones -> sequential
  (stored mask None); mask with 0/1 values and `mask.sum(1) == 1` everywhere -> snapshot; anything else ValueError
  (slightly stricter than the brief's `max()` test: checks every pixel). `y_eff = (y - b) ⊙ mask`,
  `apply(x) = apply_H(x, H) ⊙ mask`, `residual(x0) = y_eff - apply(x0)`.
  `solve(resid, r2)`: sequential -> A_i = (r2_i H) Hᵀ + σ² I, one `torch.linalg.solve` per cube (batched over B,
  broadcast over pixels); snapshot -> v = resid / (r2 ‖h_k‖² + σ²) ⊙ mask.
- `pigdm_sample`: x_T = randn(shape, CPU generator seeded `seed`).to(device); grid `range(T-1, -1, -(T // n_steps))`;
  per step: x_in = x.detach().requires_grad_(); x0 = clamp(_model_to_x0_eps(model(x_in, {'sensor': None}, t))[0]);
  under no_grad v = solve(residual(x0), 1 - ᾱ_t), w = apply_HT(v ⊙ mask, H) (v detached);
  g = autograd.grad((x0 · w).sum(), x_in) (the only grad per step);
  **x0g = clamp(x0 + (1 - ᾱ_t)/√ᾱ_t · g)**; last step returns x0g; else eps = (x - √ᾱ_t x0g)/√(1-ᾱ_t),
  x = √ᾱ_prev x0g + √(1-ᾱ_prev) eps (repo DDIM η = 0 form). `trainer.last_nfe = len(ts)`. clamp=None disables both clamps.
- `dps_sample`: x0 = clamp(D(x_t)); nrm = ‖residual(x0)‖₂ per cube; g = grad(nrm.sum(), x_t); unguided eps-consistent
  DDIM step to x'; x_{t-1} = x' - ζ g; last step returns x0.
- `draw_guided(..., method='pigdm'|'dps')` -> stack of K draws with seeds 0..K-1, [K, B, 64, H, W].

`scripts/sensor/toy_guidance.py`
- `GaussianDenoiser(mu, C, trainer)`: x̂0 = μ + √ᾱ C (ᾱC + (1-ᾱ)I)⁻¹ (x_t - √ᾱ μ) (via solve with left=False),
  returns v = (√ᾱ x_t - x̂0)/√(1-ᾱ).
- `exact_posterior`: float64; sequential: K = C Hᵀ (H C Hᵀ + σ²I)⁻¹, mean = μ + K(y - b - Hμ), var = diag(C - K H C);
  snapshot: per pixel row k = mask.argmax(1), gain C h_k / (h_kᵀ C h_k + σ²), var = diag(C) - (C h_k)²/(h_kᵀ C h_k + σ²).
- `run_toy(..., C_kind='isotropic', prior_path=PRIOR)`: x0 ~ N(μ, C) via Cholesky; hfd: μ64 = A μ, C64 = A C Aᵀ + 1e-6 I
  from the prior npz (trainer scale). y_clean = H x0 + b; σ = sigma_from_snr(snr, mean(y_clean²)) over all 30 rows;
  y = (y_clean + n) ⊙ mask. Toy's x0/noise generator is seeded 10000 + seed so it never coincides with draw seed 0's
  initial noise. Metrics exactly as the brief (sample var ddof=1; coverage_samples; CRPS ratio of means).
- CLI: loops isotropic/hfd × pigdm/dps × sequential/snapshot × steps, prints each result, writes JSON with mode 'x'
  (refuses to overwrite), prints `steps_for_B = N` or `FAIL: use DPS` using the gate on ΠGDM isotropic in both modes.
  Added `prior_path` kwarg to run_toy and `--snr_db`/`--seed` CLI options (defaults 30, 0).

## Deviation from the brief (must be reviewed)
The brief's ΠGDM correction x0g = x0 + √ᾱ_t · g fails the isotropic test catastrophically: **rmse_excess = 41.99**
(measured, 50 steps, K=200). Reason: for C = I, x̂0 = √ᾱ x_t so g = √ᾱ Hᵀv and √ᾱ g = ᾱ Hᵀ v, whereas the exact
E[x0 | x_t, y] = x̂0 + r_t² Hᵀ v = x̂0 + (1-ᾱ) Hᵀ v. Tweedie on p(x_t | y) gives the general form
E[x0|x_t,y] = x̂0 + (1-ᾱ_t)/√ᾱ_t · ∇_{x_t} log p(y|x_t), i.e. scale r_t²/√ᾱ_t. (In Song et al.'s Algorithm 1 the
√ᾱ_t g term is added to x_{t-1}, not to x̂0, which is presumably where the brief's scale came from.) I implemented the
Tweedie scale, documented in a code comment and the commit message. The brief itself named "the √ᾱ_t scale of the
guidance term" as the first thing to check.

## Tests (TDD)
RED: `CUDA_VISIBLE_DEVICES= .../hsi/bin/python -m pytest tests/test_guidance.py -q`
-> `ImportError: cannot import name 'guidance' from 'model'` (collection error).
Intermediate: 2 failed / 4 passed: isotropic toy (rmse_excess 41.99 with the brief's scale) and
test_sequential_solve_matches_dense (float32 rounding: computing r2·(HHᵀ) vs the test's (r2·H)@Hᵀ differed by ~1e-4 on
values ~100; fixed by matching the test's operation order).
GREEN: `6 passed in 11.47s`, no warnings.

Isotropic toy (test settings, small_R, sequential, 50 steps, K=200, B=1, 8×8, 30 dB):
rmse_excess 0.0027, var_ratio 0.924, cov50 0.488, cov90 0.882, crps_ratio 1.001.
Same, snapshot: 0.0025 / 0.927 / 0.480 / 0.883. 100 steps: var_ratio 0.960 / 0.962, cov90 0.889 / 0.889.
Extra sanity with the REAL device-1 R (K=64, B=1, 8×8, 50 steps, CPU): isotropic seq 0.0075/0.923/0.485/0.884,
snap 0.0068/0.925/0.481/0.879; hfd seq rmse_excess 0.092, var_ratio 0.81, cov90 0.85; hfd snap 0.012/0.93/0.44/0.855.

## CLI dry run
`python scripts/sensor/toy_guidance.py --steps 5 --K 8 --B 1 --hw 4 --device cpu --out .../tmp/d16/toy_smoke.json`
ran all 8 configs (expected coverage_samples K<19 warning), printed `FAIL: use DPS` (5 steps, var_ratio ~0.46), wrote
the JSON; a second run raised `FileExistsError ... refusing to overwrite`. Smoke file deleted.

## Files
- /data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan/model/guidance.py (new)
- /data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan/scripts/sensor/toy_guidance.py (new)
- /data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan/tests/test_guidance.py (new, verbatim from the brief)

## Self-review
- One autograd.grad per step; v/w computed under no_grad from x0 (detached by no_grad); DDIM update under no_grad on
  x_in.detach(); no graph retained across steps.
- Seeds: CPU generator per draw, seeds 0..K-1; reproducibility test passes.
- No numpy in model/guidance.py.
- DPS is unchanged from the brief (ζ = 1, final step returns x0 without guidance).

## Concerns
1. The guidance-scale deviation above (decisive for Task 10/13 and the paper's method description: say "ΠGDM with
   Tweedie-scaled x̂0 correction r_t²/√ᾱ_t").
2. At 50 steps the isotropic var_ratio is ~0.92 (gate |var_ratio-1| <= 0.10 passes, but with ~0.02 margin; 100 steps
   gives ~0.96). Task 13's B=4, K=250 run decides.
3. test_sequential_solve_matches_dense compares two float32 solves at atol 1e-4 on values ~100; it passes because the
   operation order matches, i.e. it is sensitive to fp ordering.
4. ΠGDM on the HFD prior in sequential mode under-disperses (var_ratio 0.81 at 50 steps): r_t² = 1-ᾱ is not the true
   posterior covariance for a correlated prior — expected, report-only.
