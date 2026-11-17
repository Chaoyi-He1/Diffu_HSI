# Phase 3b — B⁺ and the zero-shot generality tests: Results

Spec and pre-registration: `docs/superpowers/specs/2026-11-16-b-plus-and-zero-shot-generality-design.md`. Everything
here is **report-only**: the Phase 3 verdict (`sensor/verdict__e123.json`, main line B) stands. Conventions as in the
Phase 3 results note: errors in % of the [0, 1] range, CRPS in %, 50-cube `ddpm1000_subset` for B and B⁺ with every
other method re-averaged on the same cubes, "± SE" the source-image-clustered SE of a paired difference.

## Part A — B⁺ (2026-11-16 / 17)

### Diagnosis (calibration cubes, no test cube)

The prior network's x̂₀ error on 16 calibration cubes is 1.697 / 1.707 / 1.717 / 1.919 % at t = 0 / 5 / 19 / 99, where
the input noise is 0.50 / 1.50 / 3.80 / 16.0 %: a reconstruction floor that does not move with t, above the input noise
for t ≤ 5. ΠGDM returns the network's x̂₀ at its last step, so the K draws converge to one smoothed estimate. Continuing
the grid to t = 0 (55 steps) makes it worse on the same cubes at 30 dB: mean RMSE 2.00 vs 1.73 %, cov90 0.17 vs 0.35.

### Kernel

`sensor/b_kernel.npz` (sha 292580a7e1a5…, code a3a07cf): 68 calibration cubes (gate cubes sharing no source image with
the 50 test cubes, one per image), Device 1 sequential 20 dB, K = 8, ΠGDM 50 steps; B's mean RMSE on them 2.894 %, the
kernel's RMS 2.962 % of the range. One kernel for all settings, τ = 1.

### Test cubes: all 13 settings (records `*_Bp.json`, K = 32, B and B⁺ scored on the same draws; draws in `sensor/draws/`)

| setting | B: RMSE / CRPS / cov50 / cov90 | B⁺: RMSE / single / CRPS / cov50 / cov90 | D | L | V CRPS | C RMSE / CRPS | B⁺ − D ± SE (ratio) | B⁺ − B ± SE |
|---|---|---|---|---|---|---|---|---|
| E1 dev1 seq 60 dB | 2.383 / 1.361 / 0.241 / 0.504 | **1.039** / 1.425 / 0.550 / 0.497 / 0.893 | 1.055 | 1.074 | 0.570 | 1.079 / 0.571 | −0.016 ± 0.014 (0.985) | −1.344 ± 0.104 |
| E1 dev1 seq 50 dB | 2.290 / 1.363 / 0.210 / 0.451 | **1.380** / 1.909 / 0.736 / 0.505 / 0.904 | 1.354 | 1.526 | 0.817 | 1.402 / 0.750 | +0.026 ± 0.018 (1.019) | −0.910 ± 0.094 |
| E1 dev1 seq 40 dB | 2.274 / 1.437 / 0.153 / 0.351 | **1.868** / 2.729 / 1.003 / 0.536 / 0.922 | 1.815 | 2.604 | 1.373 | 1.928 / 1.050 | +0.054 ± 0.025 (1.029) | −0.406 ± 0.071 |
| E1 dev1 seq 30 dB | 2.351 / 1.539 / 0.118 / 0.288 | **2.227** / 3.426 / 1.206 / 0.586 / 0.942 | 2.435 | 4.109 | 2.173 | 3.211 / 1.743 | −0.208 ± 0.040 (0.915) | −0.124 ± 0.035 |
| E1 dev1 seq 20 dB | 2.876 / 1.817 / 0.145 / 0.366 | **2.861** / 4.093 / 1.518 / 0.563 / 0.922 | 3.080 | 6.314 | 3.420 | 4.778 / 2.618 | −0.219 ± 0.075 (0.929) | −0.015 ± 0.008 |
| E2 dev2 seq 40 dB | 2.330 / 1.488 / 0.123 / 0.318 | **2.160** / 3.009 / 1.141 / 0.528 / 0.911 | 3.013 | 3.141 | 1.715 | 3.121 / 1.703 | −0.852 ± 0.074 (0.717) | −0.170 ± 0.029 |
| E2 dev2 seq 30 dB | 2.530 / 1.628 / 0.114 / 0.283 | **2.546** / 3.580 / 1.342 / 0.551 / 0.919 | 3.992 | 5.000 | 2.678 | 4.749 / 2.575 | −1.447 ± 0.124 (0.638) | +0.016 ± 0.016 |
| E2 dev2 seq 20 dB | 2.906 / 1.763 / 0.165 / 0.409 | **2.943** / 4.170 / 1.546 / 0.560 / 0.921 | 6.530 | 7.386 | 4.055 | 8.170 / 4.605 | −3.586 ± 0.312 (0.451) | +0.037 ± 0.009 |
| E3 dev1 snap 60 dB | 5.915 / 3.566 / 0.244 / 0.572 | **5.845** / 7.133 / 3.300 / 0.389 / 0.781 | 2.670 | 5.562 | 5.871 | 6.405 / 3.776 | +3.175 ± 0.254 (2.189) | −0.070 ± 0.006 |
| E3 dev1 snap 40 dB | 5.965 / 3.593 / 0.246 / 0.575 | **5.907** / 7.183 / 3.341 / 0.388 / 0.782 | 2.683 | 5.600 | 5.882 | 6.443 / 3.793 | +3.224 ± 0.257 (2.202) | −0.058 ± 0.006 |
| E3 dev1 snap 30 dB | 6.110 / 3.669 / 0.252 / 0.578 | **6.094** / 7.458 / 3.451 / 0.384 / 0.782 | 2.881 | 5.812 | 5.963 | 6.722 / 3.926 | +3.213 ± 0.259 (2.115) | −0.016 ± 0.004 |
| E3 dev1 snap 20 dB | 6.442 / 3.790 / 0.276 / 0.629 | **6.460** / 7.960 / 3.630 / 0.386 / 0.800 | 3.938 | 6.788 | 6.459 | 7.599 / 4.374 | +2.522 ± 0.261 (1.641) | +0.018 ± 0.002 |
| E4 dev2 snap 30 dB | 8.154 / 4.645 / 0.285 / 0.660 | **8.122** / 9.556 / 4.464 / 0.372 / 0.786 | 13.819 | 7.513 | 6.724 | 16.994 / 10.374 | −5.697 ± 0.431 (0.588) | −0.031 ± 0.006 |

(L is demosaic-then-MMSE in the snapshot rows. V's point estimate is L's. C is the Phase 3 C record on the same cubes.)

**Integrity check.** The new records' B means reproduce the Phase 3 B records to within 0.018 (RMSE), 0.010 (CRPS) and
0.017 (cov90) over the 13 settings. Individual cubes are not bit-identical (largest per-cube differences in the seed-0
draw's RMSE at 60–40 dB): the guided chain amplifies non-deterministic GPU kernels where the guidance is strongest. The
B columns above are the new records'.

### The pre-registered clauses for B⁺ (`sd_verdict.py --report_only`, manifest `e123_bplus` with `rename: {Bp: B}`; `sensor/verdict__e123_bplus__report.json`), verbatim

```
T1 B 60 dB: PASS  ratio 0.9846 (2se 1.0118, limit 1.05)
T1 B 50 dB: PASS  ratio 1.0191 (2se 1.0472, limit 1.05)
T1 B 40 dB: FAIL  ratio 1.0295 (2se 1.0596, limit 1.05)
T1 B 30 dB: PASS  ratio 0.9147 (2se 0.9482, limit 1.05)
T1 B 20 dB: PASS  ratio 0.9288 (2se 0.9771, limit 1.05)
T2 B 30 dB: PASS  crps-L -1.0351  crps-V -0.9666  cov90 0.942
T2 B 20 dB: PASS  crps-L -1.9773  crps-V -1.9017  cov90 0.922
T3 B: PASS  dev2 0.5092 vs 1.1 x dev1 0.5420
T4 B: FAIL  ratio 1.0486 (2se 1.1865, limit 0.85)
passes {'B': 3, 'C': 1} | main line: none (report only)
```

("B" in these lines is B⁺; C's lines are unchanged from the Phase 3 verdict.)

### Reading against the predictions (spec A.5)

- **T2: passes at 30 and 20 dB, as predicted.** cov90 0.942 / 0.922 (B: 0.288 / 0.366), CRPS 1.04 / 1.98 below the MMSE
  posterior and 0.97 / 1.90 below the variance head. In sequential mode on both devices cov50 is 0.50–0.59 and cov90
  0.89–0.94 at every SNR, with one kernel fitted at 20 dB on Device 1 and no scale factor. B⁺ has the lowest CRPS of
  every method at all eight sequential settings (at 60 dB 0.550 against V's 0.570 and C's 0.571).
- **T1: four of five SNRs pass; 40 dB misses by the 2-SE rule** (ratio 1.029, bound 1.060 against 1.05), so the clause
  fails as a whole. Predicted: 30 / 20 / 60 dB pass, 50 / 40 dB open. The conditioning removes B's high-SNR penalty:
  2.38 → 1.04 % at 60 dB (D 1.055, MMSE 1.074), 2.29 → 1.38 % at 50 dB, 2.27 → 1.87 % at 40 dB. B⁺ is within 3 % of the
  regressor's mean at every SNR and 7–9 % below it at 30 and 20 dB.
- **T3: passes, as predicted** (Device 2 ratio to L 0.509 against 1.1 × 0.542). On Device 2 B⁺'s mean equals B's at 30
  and 20 dB (the measurement adds little there) and improves at 40 dB; it is 0.72 / 0.64 / 0.45 × the regressor at
  40 / 30 / 20 dB.
- **T4: fails, as predicted** (ratio 1.049 to demosaic-then-MMSE). One row per pixel leaves the mean where B's was.
- **Snapshot coverage stays low, as predicted from the calibration cubes:** cov90 0.78–0.80 (B: 0.57–0.66), cov50
  0.37–0.39. The kernel was measured in sequential mode and does not contain the demosaicing error.
- **Passes: B⁺ 3 of 5 (T2 at 30 dB, T2 at 20 dB, T3), C 1, B 1.** None of the spec's "what would count against the
  claim" outcomes occurred in Part A: T2's band holds on the test cubes, and B⁺'s mean is never worse than B's by more
  than 0.04 (Device 2 at 20 dB, +0.037 ± 0.009; snapshot 20 dB +0.018 ± 0.002).

What B⁺ is and is not: it is a post-hoc amendment of B's last sampling step, designed after the Phase 3 verdict on
cubes that are not in the test subset, and it uses the measurement twice (in the guidance and in the conditioning). The
calibration it reaches is that of a Gaussian kernel around a collapsed ensemble: single B⁺ draws are noisy
(single-draw RMSE 1.4 × the mean's), not realistic textures. Its uncertainty is calibrated per element, not a model of
the cube's joint structure.
