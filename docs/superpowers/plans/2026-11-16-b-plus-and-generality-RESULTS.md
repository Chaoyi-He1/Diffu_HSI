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

### Test cubes: all 13 settings (records `*_Bp.json`, K = 32, B and B⁺ scored on the same draws; the raw draws, 22 GB in `sensor/draws/`, were deleted on 2026-11-20 on the user's instruction — the records keep every per-cube metric)

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

**Integrity check.** The new records' B means reproduce the Phase 3 B records to within 0.017 (RMSE), 0.010 (CRPS) and
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

## Part B — four zero-shot generality tests (2026-11-17)

Records `g_<scenario>_{main,interp,B,C}.json` for the 24 settings of `scripts/sensor/scenarios_3b.py` (code 11e7d8d for
main / B / C, 4905719 for interp). Nothing was retrained; the changed forward model is known to every method. All numbers
below are RMSE % on the B record's 50 cubes (L, D and the interpolation baselines re-averaged from their 200-cube records;
C, B, B⁺ with K = 20); "± SE" is the clustered SE of the paired difference, the ratio is of means. Reference points on the
same cubes (Part A, K = 32): Device 1 at 60 / 40 / 30 / 20 dB L 1.074 / 2.604 / 4.109 / 6.314, D 1.055 / 1.815 / 2.435 /
3.080, B⁺ 1.039 / 1.868 / 2.227 / 2.861; Device 2 at 30 dB L 5.000, D 3.992, B⁺ 2.546. GPU time: 26.9 h for the 24 B / B⁺
records, 1.8 h for C, 0.4 h for the rest.

**B.1 sensor variation (30 dB)**

| setting | L | D | C | B | B⁺ | B⁺ CRPS / cov50 / cov90 | L CRPS | B⁺ − D ± SE (ratio) |
|---|---|---|---|---|---|---|---|---|
| blend_025 | 4.313 | 2.582 | 3.458 | 2.434 | **2.369** | 1.256 / 0.582 / 0.935 | 2.386 | −0.213 ± 0.048 (0.918) |
| blend_050 | 4.627 | 2.843 | 3.736 | 2.649 | **2.624** | 1.379 / 0.552 / 0.919 | 2.567 | −0.219 ± 0.076 (0.923) |
| blend_075 | 5.188 | 3.583 | 4.438 | 2.502 | **2.532** | 1.321 / 0.572 / 0.924 | 2.843 | −1.052 ± 0.089 (0.706) |
| shift_1 | 4.069 | 2.640 | 3.157 | 2.375 | **2.335** | 1.240 / 0.560 / 0.927 | 2.234 | −0.305 ± 0.043 (0.884) |
| shift_2 | 4.057 | 3.022 | 3.254 | 2.378 | **2.355** | 1.244 / 0.548 / 0.920 | 2.227 | −0.667 ± 0.071 (0.779) |
| shift_4 | 4.052 | 3.401 | 3.495 | 2.335 | **2.285** | 1.215 / 0.544 / 0.915 | 2.233 | −1.116 ± 0.096 (0.672) |
| gain_02 | 4.050 | 2.413 | 3.178 | 2.341 | **2.248** | 1.201 / 0.583 / 0.936 | 2.212 | −0.166 ± 0.038 (0.931) |
| gain_05 | 3.919 | 2.377 | 3.022 | 2.318 | **2.230** | 1.190 / 0.581 / 0.936 | 2.142 | −0.148 ± 0.036 (0.938) |

**B.2 fewer states**

| setting | L | D | C | B | B⁺ | B⁺ CRPS / cov50 / cov90 | L CRPS | B⁺ − D ± SE (ratio) |
|---|---|---|---|---|---|---|---|---|
| states_15_40 | 3.126 | 2.107 | 2.349 | 2.315 | **2.066** | 1.102 / 0.550 / 0.921 | 1.679 | −0.042 ± 0.032 (0.980) |
| states_10_40 | 3.551 | 2.408 | 2.794 | 2.411 | **2.247** | 1.186 / 0.565 / 0.927 | 1.909 | −0.161 ± 0.041 (0.933) |
| states_5_40 | 4.078 | 2.787 | 4.006 | 3.043 | **2.908** | 1.558 / 0.522 / 0.896 | 2.265 | +0.120 ± 0.114 (1.043) |
| states_15_20 | 7.335 | 3.819 | 6.258 | 3.265 | **3.287** | 1.721 / 0.533 / 0.898 | 4.089 | −0.532 ± 0.118 (0.861) |
| states_10_20 | 7.847 | 5.397 | 7.830 | 3.764 | **3.791** | 2.006 / 0.490 / 0.870 | 4.390 | −1.606 ± 0.467 (0.702) |
| states_5_20 | 8.909 | 7.563 | 10.946 | 4.163 | **4.200** | 2.248 / 0.471 / 0.869 | 4.988 | −3.363 ± 0.369 (0.555) |

**B.3 other tasks**

| setting | L | L_interp | L_interp_plain | D | C | B | B⁺ | B⁺ CRPS / cov50 / cov90 | L CRPS | B⁺ − D ± SE (ratio) | B⁺ − best classical ± SE (ratio) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dead_50 | 12.088 | 4.028 | 3.965 | 11.581 | 12.993 | 2.483 | **2.472** | 1.305 / 0.593 / 0.935 | 5.794 | −9.109 ± 0.478 (0.213) | −1.493 ± 0.202 (0.623) |
| dead_90 | 15.740 | 4.824 | 4.496 | 18.332 | 23.049 | 3.685 | **3.698** | 1.836 / 0.565 / 0.922 | 8.611 | −14.634 ± 0.585 (0.202) | −0.797 ± 0.196 (0.823) |
| stride_4_60 | 16.024 | 4.195 | 2.577 | 20.243 | 25.518 | 3.664 | **3.651** | 1.783 / 0.570 / 0.922 | 8.784 | −16.592 ± 0.566 (0.180) | +1.074 ± 0.072 (1.417) |
| stride_8_60 | 16.418 | 6.210 | 4.770 | 23.561 | 31.011 | 5.789 | **5.810** | 2.704 / 0.555 / 0.919 | 9.195 | −17.751 ± 0.586 (0.247) | +1.040 ± 0.095 (1.218) |
| stride_4_30 | 16.057 | 4.709 | 4.380 | 19.499 | 26.876 | 3.978 | **3.993** | 1.982 / 0.552 / 0.915 | 8.888 | −15.506 ± 0.637 (0.205) | −0.387 ± 0.209 (0.912) |

**B.4 non-Gaussian noise**

| setting | L | L_naive | D_naive | D | C | B | B⁺ | B⁺ CRPS / cov50 / cov90 | L CRPS | B⁺ − D ± SE (ratio) |
|---|---|---|---|---|---|---|---|---|---|---|
| poisson_30 | 4.060 | 4.089 | 2.394 | 2.430 | 3.262 | 2.383 | **2.252** | 1.204 / 0.567 / 0.928 | 2.178 | −0.178 ± 0.039 (0.927) |
| poisson_20 | 6.308 | 6.307 | 3.169 | 3.565 | 4.967 | 3.001 | **3.012** | 1.585 / 0.533 / 0.901 | 3.450 | −0.553 ± 0.052 (0.845) |
| sat_90 | 4.248 | 8.342 | 8.608 | 2.528 | 3.460 | 2.377 | **2.305** | 1.236 / 0.581 / 0.933 | 2.311 | −0.223 ± 0.041 (0.912) |
| sat_70 | 5.300 | 15.193 | 15.558 | 3.835 | 4.930 | 2.638 | **2.604** | 1.373 / 0.563 / 0.923 | 2.774 | −1.232 ± 0.128 (0.679) |
| quant_4 | 5.990 | 13.660 | 12.062 | 4.028 | 4.926 | 3.315 | **3.324** | 1.745 / 0.502 / 0.872 | 3.281 | −0.703 ± 0.066 (0.825) |

### Summary

B⁺ has the lowest RMSE in **21 of the 24 settings** (among L, the two classical interpolations, D, C, B and B⁺ on the same
cubes) and its 90 % coverage is 0.87–0.94 in all 24 with the one kernel of Part A. The three exceptions: ×4 and ×8
spatial super-sampling at the 60 dB floor (the plain interpolation is 1.42 × and 1.22 × better) and five states at 40 dB
(the regressor 2.787 against 2.908, +0.120 ± 0.114). B and B⁺ have nearly the same mean wherever the measurement is
weak (20 dB, missing pixels); the conditioning matters at 40 dB and above. C, fed the changed model's statistics, is
behind the regressor everywhere and fails with it on missing pixels.

### Against the predictions (spec Part B; ten predictions: five held, three did not, two held in part)

- **B.1.** *Held:* B⁺ is the best point estimate at all eight settings (every B⁺ − D difference beyond 2.8 SE); the
  regressor's advantage over L shrinks with the change (D / L 0.59 on Device 1; 0.60 / 0.61 / 0.69 for the blends,
  0.65 / 0.74 / 0.84 for shifts of 1 / 2 / 4 bands, 0.60 / 0.61 for the gains). *Did not hold:* "B⁺ within 10 % of its
  Device 1 value everywhere" — shifts and gains stay within 6 % (2.23–2.36 against 2.227), the blends give +6 / +18 /
  +14 % (Device 2 itself: +14 %). B⁺ / D: gain 0.93 / 0.94, shift 0.88 / 0.78 / 0.67, blend 0.92 / 0.92 / 0.71.
- **B.2.** *Held:* every method degrades as states are removed. *Did not hold:* "B⁺ / L falls with S" (0.66 / 0.63 /
  0.71 at 40 dB, 0.45 / 0.48 / 0.47 at 20 dB against 0.72 and 0.45 with 30 states); "D loses its advantage over L by
  S = 5" (D / L 0.68 at 40 dB, 0.85 at 20 dB). At 20 dB B⁺ / D is 0.86 / 0.70 / 0.56 for S = 15 / 10 / 5 (4.20 against
  7.56 % with five states); at 40 dB 0.98 / 0.93 / 1.04, only S = 10 beyond two SE.
- **B.3.** *Held in part:* B⁺ beats the better classical interpolation on dead pixels (0.62 × at 50 %, 0.82 × at 90 %)
  and at ×4 30 dB (0.91 ×, −0.387 ± 0.209, 1.9 SE), not at ×4 or ×8 at the 60 dB floor (1.42 ×, 1.22 ×). Against
  L_interp as the spec first defined it B⁺ wins at ×4 60 dB too (3.651 against 4.195); the ruling of 2026-11-17 made
  the stronger L_interp_plain the reference before any B record of this group existed. *Held:* Phase 1's regressor
  trained for ×4 stays ahead (1.962 % noise-free on the 200 gate cubes, typed from its note; B⁺ 3.651, plain
  interpolation 2.577 at 60 dB on the 50 cubes). The zero-shot regressor and C are useless on missing pixels (D 11.6–
  23.6 %, C 13.0–31.0 %, worse than the exact MMSE that gives an unmeasured pixel the prior mean, 12.1–16.4 %).
- **B.4.** *Held:* B⁺ beats the aware regressor under saturation (0.91 × at 10 %, 0.68 × at 30 %); it is also the best
  point estimate under shot noise (0.93 / 0.85 ×) and with the 4-bit ADC (0.83 ×). *Held in part:* "aware beats
  naive" — yes under saturation and quantisation (naive L 8.3 / 15.2 / 13.7 against 4.2 / 5.3 / 6.0; naive D 8.6 /
  15.6 / 12.1 against 2.5 / 3.8 / 4.0), no under shot noise (L 4.089 / 6.307 naive against 4.060 / 6.308 aware; D
  2.394 / 3.169 naive against 2.430 / 3.565 aware: the per-element variances take D's statistics out of its family).
  Against the better of the two regressors at shot noise 20 dB B⁺ is 0.95 ×.

### What this adds to the paper's claim, and what it does not

One prior and one kernel, with the forward model swapped in at sampling time, give the best or near-best point estimate
and near-nominal per-element coverage across a changed response matrix, fewer states, missing pixels and three
non-Gaussian noise models; the regressor needs retraining to stay competitive once the measurement geometry changes
(missing pixels) or the sensor moves far (shift of 4 bands, 75 % toward Device 2, five states at 20 dB). It does not
show that diffusion wins everywhere: with all 30 states at ≥ 40 dB the regressor matches B⁺ (Part A, B.2 at 40 dB), a
trained-for-the-task regressor beats it at ×4 super-sampling, plain interpolation beats it there at the noise floor,
snapshot mode on the training sensor remains the regressor's (Part A, T4), and nothing here was replicated over
training seeds. The regressor itself transfers better than Phase 3 suggested when the change is mild (gain drift,
25–50 % blend, shot noise): its (x̂, s) interface absorbs those.
