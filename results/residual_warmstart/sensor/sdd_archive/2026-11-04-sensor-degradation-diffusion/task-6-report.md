# Task 6 report — DONE (after coordinator rulings)

## What was implemented (working tree only, uncommitted)
- `model/u2net_hyperspectral.py` `forward`: `context_encoded = self.context_encoder(context) if context is not None else None`;
  `create_context_for_resolution` returns None when `context_encoded` is None; docstring notes context may be None.
  Checked: `U2NetBlock2D.forward` skips cross-attention on None, `bridge_attn` (BasicTransformerBlock -> CrossAttention)
  falls back to self-attention (`context = x`).
- `model/residual_wrapper.py`: `ResidualWrapper(net, use_x0_hat, use_s_post=False, use_context=True)`; forward concatenates
  [x_t, x0_hat?, s_post?] (cast to x_t dtype), channel check against the full concatenation, context = cond['sensor'] only
  when use_context ('sensor' key required only then). `MODES` + 4 new names (first six unchanged), `FAM_MODES`,
  `SENSOR_MODES`, `CENTRED_MODES += FAM_MODES`, `ONE_CALL_MODES`; mode comment table extended; `build_residual_model`
  (uncond: extra 0, no context; FAM: extra 2*L, x0_hat + s_post, no context; legacy unchanged); `target_for(mode, x, x0_hat, scale)`;
  new `batch_to_cond(mode, batch, sigma_d, device)` (tuple / dict, ValueError on mismatched kind, non_blocking moves).
- `tests/test_sensor_modes.py`: the brief's tests verbatim.

## TDD evidence
RED: `CUDA_VISIBLE_DEVICES= .../hsi/bin/python -m pytest tests/test_sensor_modes.py -q` -> 6 failed
(FAM_MODES missing; conv2d got NoneType from context_encoder(None); 'uncond' not in MODES; no use_context; target_for
returned x for cond_fam; no batch_to_cond).

After implementation: full command (5 files) -> **4 failed, 32 passed**. The 4 failures are contradictions in the brief,
not implementation bugs:

### Blocker 1 — existing tests pin MODES exactly
- `tests/test_wrapper.py:67`, `tests/test_uq_modes.py:14` (also `:16` pins `CENTRED_MODES`), `tests/test_regress_mode.py:65`
  assert `MODES == ('residual', 'standard', 'standard_xhat', 'regress', 'residual_reg', 'varhead')` (and the uq test
  `CENTRED_MODES == ('residual', 'regress', 'residual_reg', 'varhead')`).
- The brief says these lists "depend on the names, not the order" — they depend on exact tuple equality. Extending MODES /
  CENTRED_MODES as the Interfaces require necessarily breaks them, and the task forbids modifying any file other than the
  two model files + the new test.
- Suggested fix: allow editing those three lines to `MODES[:6] == (...)` and `CENTRED_MODES[:4] == (...)`
  (or `CENTRED_MODES == (...) + FAM_MODES`).

### Blocker 2 — `test_u2net_runs_without_context_and_depends_on_t` cannot pass at initialisation
- With context=None, t reaches the net only through FiLM (`model/layers.py:236-240`, `:361-365`), and every FiLM linear
  is zero-initialised by design, so a freshly built net's output is exactly t-independent (verified: `allclose` True).
  With context, t also enters via `context_flat + t_emb`, which is why legacy nets depend on t at init.
- Not an architecture defect: all 124 FiLM parameters get non-zero gradients on the first backward, and after
  perturbing them (normal(0, 0.02)) outputs at t=10 vs 900 differ (max |diff| 1.34). The model will learn t-dependence.
- Suggested fix (test only): perturb the FiLM params before the comparison, e.g.
  `with torch.no_grad(): [p.normal_(0, 0.02) for n, p in net.named_parameters() if 'film' in n]`,
  or assert the FiLM grads are non-zero. Changing the zero-init would alter legacy behaviour — not recommended.

## Files changed (uncommitted)
model/residual_wrapper.py, model/u2net_hyperspectral.py, tests/test_sensor_modes.py (new). Nothing staged or committed.

## Self-review
Other 5 new tests pass; test_model_extra_in.py fully green; the only legacy failures are the exact-tuple asserts above.
Legacy-mode state dicts load strict (test_legacy_modes_unchanged and test_regress_mode's strict load pass).

## Concerns
Need the controller's decision on Blockers 1 and 2 (permission to edit the three legacy test lines and the new U-Net test).

## Resolution (coordinator rulings)
1. Legacy MODES pins: `tests/test_wrapper.py:67`, `tests/test_uq_modes.py:14`, `tests/test_regress_mode.py:65` now assert
   `MODES[:6] == (six legacy names)`; `tests/test_uq_modes.py:16` asserts `CENTRED_MODES[:4] == ('residual', 'regress',
   'residual_reg', 'varhead')` — the prefix form, because FAM_MODES is not imported in that file and the ruling said to
   keep every other line unchanged. Nothing else in those files was touched.
2. `test_u2net_runs_without_context_and_depends_on_t` perturbs the FiLM params (`normal_(0, 0.1)`, selected by
   `'film' in n.lower()`) before comparing; the parameter names do contain 'film' (`film1` / `film2`, layers.py:236-240,
   361-365). The model's zero init is unchanged.

## GREEN
`CUDA_VISIBLE_DEVICES= .../hsi/bin/python -m pytest tests/test_sensor_modes.py tests/test_wrapper.py tests/test_model_extra_in.py tests/test_uq_modes.py tests/test_regress_mode.py -q`
-> `36 passed in 11.63s` (no warnings).

## Commit
7c301f9 feat(sensor): modes uncond / cond_fam / regress_fam / varhead_fam — context-free U-Net path, s_post input slot, per-element residual scale, batch_to_cond for tuple and dict batches (Phase 3 Task 6)
Files: model/residual_wrapper.py, model/u2net_hyperspectral.py, tests/test_sensor_modes.py (new), tests/test_wrapper.py, tests/test_uq_modes.py, tests/test_regress_mode.py. Not pushed; .pyc files not staged.
