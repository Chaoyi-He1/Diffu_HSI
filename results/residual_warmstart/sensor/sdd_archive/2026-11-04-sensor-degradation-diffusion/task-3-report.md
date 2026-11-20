# Task 3 report: noise-aware MMSE

## Implemented
`data_loader/mmse.py` (numpy/scipy only): prior01, posterior_sequential (Cholesky-whitened SVD, float or [30] noise variance), posterior_snapshot (einsum-vectorised), lift_mean_std (shared or per-pixel Sigma), mmse_maps (mask None/all-ones -> sequential, one-hot -> snapshot via argmax; trainer-scale float32 outputs), demosaic (RegularGridInterpolator per channel lattice), demosaic_then_mmse (noise_var = sigma^2 + e2).

## Tests
`CUDA_VISIBLE_DEVICES= .../hsi/bin/python -m pytest tests/test_mmse.py -q` -> 8 passed.

## TDD evidence
RED: before mmse.py existed, same command -> collection error `ImportError: cannot import name 'mmse' from 'data_loader'` (expected: module missing).
Intermediate: with the brief's verbatim test, 7 passed / 1 failed (regulariser test, max abs diff 0.127 on 8/1550 elements).
GREEN: after the coordinator-approved clip amendment, 8 passed.

## Test-defect finding
The brief's regulariser test claimed the clip was inactive, but the unclipped linear estimate reaches 1.127 on a few synthetic pixels, while `linear_inverse` clips. Posterior matched make_gain to 8.7e-8 unclipped (K rel. diff 2.5e-6). Fix: compare np.clip(2*xhat01-1, -1, 1) to ref; comment on ref line updated. Ruled by coordinator; plan updated separately (bc63d8c).

## Files changed
data_loader/mmse.py, tests/test_mmse.py (new). No .pyc staged.

## Self-review
Formulas follow the brief; Sigma symmetrised in the sequential path; variance clamped at 0 before sqrt; works on ill-conditioned small_R. No over-building.

## Concerns
None. demosaic is O(30) interpolator calls on the full grid (fine for current sizes).

## Fix round 1 (reviewer: snapshot path memory/work)
Changed `data_loader/mmse.py`: added `posterior_snapshot_lowrank` (returns xhat, Cr, den; Sigma_n = C - Cr Cr^T/den) and `lift_mean_std_lowrank` (var = diag(A C A^T) - (A Cr)^2/den, clamped at 0); `mmse_maps` snapshot branch now uses them (no [N,31,31] tensor); sequential branch and dense `posterior_snapshot` unchanged.
Covering test: new `test_lowrank_snapshot_lift_matches_dense` (small_R/synthetic_prior, 10x12 one-hot mask; xhat, mean, std match dense path to 1e-9); `test_mmse_maps_shapes_scales_and_mask_dispatch` still exercises mmse_maps snapshot.
Command: `CUDA_VISIBLE_DEVICES= .../hsi/bin/python -m pytest tests/test_mmse.py -q` -> `9 passed in 2.04s`.
