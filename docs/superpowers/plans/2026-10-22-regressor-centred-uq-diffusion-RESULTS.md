# Phase 2a — Regressor-Centred Residual Diffusion for Calibrated Uncertainty: Results

Spec: `docs/superpowers/specs/2026-10-22-regressor-centred-uq-diffusion-design.md` (approved 2026-10-22, commit 162c724; as-built note appended by Task 10, commit a36979a). Plan: `docs/superpowers/plans/2026-10-22-regressor-centred-uq-diffusion.md` (commit 850ca6a; the controller's rulings R1–R13 are listed in the plan's as-built block). Build phase (Tasks 1–10) complete 2026-10-22: 429 tests green, CPU smoke green (branch `worktree-phase1-plan`, HEAD a36979a).

Conventions: every number below is on the 200 gate cubes of `tests/data/val_subset_200.json` unless stated; errors in % of the [0, 1] range; CRPS in %; "± " is the iid SE over cubes with the source-image-clustered SE in parentheses where a paired difference is reported; K = 32 samples per cube everywhere (ruling R10 withdrew the spec's K = 16 at d = 4). No d = 8 gate number is looked at before Task 15's single verdict.

## Stage 1 — split and half regressors (d = 8)

- Split (2026-10-22, `scripts/uq/split_halves.py`, seed 0): `results/residual_warmstart/uq/split_halves.json`, sha256 `977cf0d08a1977b6622b5fe920f640c1bfb5bb6ed5739e84455cb6409e3ab421`; 4516 + 4484 training files; images per class within one of each other in all 22 classes (e.g. P001 56/55, P002 53/53, P016 39/39, P022 47/46).
- `regress_l1_half1_d8_b64`: launched 2026-10-22 ≈ 20:56 on GPU 1 (4516 files, 564 steps per epoch, 40 epochs, monitoring every epoch, every epoch saved). Selection pending.
- `regress_l1_half2_d8_b64`: launched 2026-10-22 ≈ 21:40 on GPU 0 after Stage 0 finished (4484 files). Selection pending.

## Stage 0 — harness validation on the existing d = 4 assets (2026-10-22, GPU 0, ≈ 40 min)

Record `results/residual_warmstart/uq/d4/eval/stage0_d4_assets.json`: centre `regress_l1_sel_d4_b64@17`, ensemble of the three selected d = 4 checkpoints (`regress_l1_sel_d4_b64@17`, `regress_l1_sel_s43_d4_b64@17`, `regress_l2_sel_d4_b64@18`), B `residual_d4_b64@100`, K = 32, DDIM-10, batch 4. No `residual_reg` / `varhead` yet (decision D4).

| method | RMSE % | log HF ratio | CRPS % | cov90 | AUSE |
|---|---|---|---|---|---|
| baseline x̂₀ | **2.3512** | −1.992 | — | — | — |
| regressor x̂_R (centre) | **1.9621** | −1.846 | — | — | — |
| ensemble (3 members, Gaussian) | 1.9244 (member mean) | −1.938 | 0.969 | 0.241 | 0.319 |
| B, mean of 32 DDIM-10 draws | 2.1450 | −1.155 | 0.948 | 0.933 | 0.329 |
| B, single draw | 3.0884 | +1.803 | | | |

**Harness accepted:** the regressor's 1.9621 % and the baseline's 2.3512 % reproduce the B-track gate records to the last digit. Side observations (not gates): the three-member ensemble is sharp but badly under-dispersed (90 % intervals cover 24 % of elements) and its paired mean beats the single regressor; B's 32-draw mean (2.145 %) sits where the Phase 1a′ chain-mean estimate put it (2.11–2.15 %), its single draw is far worse (3.09 %) with 6 × the truth's high-frequency energy (log ratio +1.8: residual noise, not texture), its 90 % coverage is 0.93, and despite its worse mean it already beats the ensemble on CRPS (0.948 vs 0.969) — a calibrated-but-blunt sampler outscores a sharp-but-overconfident one under a proper scoring rule. Every log HF ratio of a mean estimate is strongly negative (−1.85 for the regressor: its texture energy is e^{−1.85} ≈ 16 % of the truth's), which is the blur the perceptual clause P1 is about.

No d = 8 number has been looked at.
