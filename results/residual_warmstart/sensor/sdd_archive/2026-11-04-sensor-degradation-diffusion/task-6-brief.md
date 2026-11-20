### Task 6: The four new modes in the wrapper and the U-Net's context-free path

**Files:**
- Modify: `model/residual_wrapper.py`, `model/u2net_hyperspectral.py:424-452` (the `context_encoded` / `create_context_for_resolution` lines of `forward`)
- Test: `tests/test_sensor_modes.py`

**Interfaces:**
- Produces: `MODES = ('residual', 'standard', 'standard_xhat', 'regress', 'residual_reg', 'varhead', 'uncond', 'cond_fam', 'regress_fam', 'varhead_fam')`; `FAM_MODES = ('cond_fam', 'regress_fam', 'varhead_fam')`; `SENSOR_MODES = ('uncond',) + FAM_MODES`; `CENTRED_MODES += FAM_MODES`; `ONE_CALL_MODES = ('regress', 'varhead', 'regress_fam', 'varhead_fam')`; `ResidualWrapper.__init__(self, net, use_x0_hat: bool, use_s_post: bool = False, use_context: bool = True)`; `forward(x_t, cond, t)`: inp = cat([x_t] + [cond['x0_hat']] if use_x0_hat + [cond['s_post']] if use_s_post), context = cond['sensor'] if use_context else None (the `'sensor'` key is required only when use_context); `build_residual_model(mode, base_channels, spectral_channels=64, sensor_channels=30)`: `uncond` → extra 0, use_x0_hat False, use_context False; FAM → extra 2 · spectral, use_x0_hat True, use_s_post True, use_context False; legacy modes unchanged; `def target_for(mode, x, x0_hat, scale)` (scale scalar or tensor; `uncond` and `standard*` return x); `def batch_to_cond(mode: str, batch, sigma_d: float, device) -> tuple[torch.Tensor, dict, object]` — tuple batch `(x, y_d, xh)` → `(x, {'sensor': y_d, 'x0_hat': xh}, sigma_d)`; dict batch → `(b['x'], {'sensor': None, 'x0_hat': b['x0_hat'], 's_post': b['s_post']}, b['s_post'])` for FAM and `(b['x'], {'sensor': None}, None)` for `uncond`; all tensors moved to `device` (non_blocking); `regress_forward` / `varhead_forward` unchanged (the zero placeholder has `cond['x0_hat']`'s 64 channels; the wrapper appends s_post); `U2NetHyperspectral.forward`: `context_encoded = self.context_encoder(context) if context is not None else None` and `create_context_for_resolution` returns None when it is None (every block and `bridge_attn` already accept None).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sensor_modes.py
import torch
from model import residual_wrapper as RW
from model.u2net_hyperspectral import U2NetHyperspectral

def test_mode_tables():
    assert set(RW.FAM_MODES) <= set(RW.MODES) and 'uncond' in RW.MODES and set(RW.FAM_MODES) <= set(RW.CENTRED_MODES)

def test_u2net_runs_without_context_and_depends_on_t():
    net = U2NetHyperspectral(spectral_channels=8, sensor_channels=5, base_channels=16)
    x = torch.randn(2, 8, 16, 16)
    a = net(x, None, torch.tensor([10, 10])); b = net(x, None, torch.tensor([900, 900]))
    assert a.shape == (2, 8, 16, 16) and not torch.allclose(a, b)

def test_uncond_and_fam_models_shapes():
    u = RW.build_residual_model('uncond', 16, spectral_channels=8, sensor_channels=5)
    assert u.net.input_proj.in_channels == 8 and u(torch.randn(2, 8, 16, 16), {'sensor': None}, torch.zeros(2, dtype=torch.long)).shape == (2, 8, 16, 16)
    c = RW.build_residual_model('cond_fam', 16, spectral_channels=8, sensor_channels=5)
    cond = {'sensor': None, 'x0_hat': torch.randn(2, 8, 16, 16), 's_post': torch.rand(2, 8, 16, 16) + 0.02}
    assert c.net.input_proj.in_channels == 24 and c(torch.randn(2, 8, 16, 16), cond, torch.zeros(2, dtype=torch.long)).shape == (2, 8, 16, 16)
    r = RW.build_residual_model('regress_fam', 16, spectral_channels=8, sensor_channels=5)
    assert RW.regress_forward(r, cond).shape == (2, 8, 16, 16) and RW.varhead_forward(RW.build_residual_model('varhead_fam', 16, 8, 5), cond).shape == (2, 8, 16, 16)

def test_legacy_modes_unchanged():
    b = RW.build_residual_model('residual', 16, spectral_channels=8, sensor_channels=5)
    assert b.net.input_proj.in_channels == 16 and b.use_context and not b.use_s_post
    b2 = RW.build_residual_model('residual', 16, spectral_channels=8, sensor_channels=5); b2.load_state_dict(b.state_dict(), strict=True)

def test_target_for_accepts_tensor_scale_and_uncond_returns_x():
    x, xh, s = torch.ones(1, 2, 2, 2), torch.zeros(1, 2, 2, 2), torch.full((1, 2, 2, 2), 0.5)
    assert torch.equal(RW.target_for('cond_fam', x, xh, s), 2 * torch.ones_like(x)) and torch.equal(RW.target_for('uncond', x, None, None), x)

def test_batch_to_cond_both_batch_kinds():
    x, cond, scale = RW.batch_to_cond('residual', (torch.zeros(1, 4, 2, 2), torch.zeros(1, 3, 1, 1), torch.zeros(1, 4, 2, 2)), 0.05, torch.device('cpu'))
    assert set(cond) == {'sensor', 'x0_hat'} and scale == 0.05
    b = dict(x=torch.zeros(1, 4, 2, 2), y=torch.zeros(1, 3, 2, 2), x0_hat=torch.zeros(1, 4, 2, 2), s_post=torch.full((1, 4, 2, 2), 0.1), s_raw=torch.zeros(1, 4, 2, 2), mask=torch.ones(1, 3, 2, 2), sigma=torch.zeros(1), snr_db=torch.zeros(1), snapshot=torch.zeros(1))
    x, cond, scale = RW.batch_to_cond('cond_fam', b, 0.02, torch.device('cpu')); assert cond['sensor'] is None and torch.equal(scale, b['s_post'])
    x, cond, scale = RW.batch_to_cond('uncond', b, 0.02, torch.device('cpu')); assert cond == {'sensor': None} and scale is None
```

- [ ] **Step 2: Run to verify failure.** Expected: FAIL (`FAM_MODES` missing; U-Net raises on None context).

- [ ] **Step 3: Implement** the wrapper changes and the two-line U-Net change per Interfaces. Keep `MODES`' first six entries in their current order (the launcher's and the Phase 1/2a tests' lists depend on the names, not the order, but do not rename anything).

- [ ] **Step 4: Run the new tests and the old mode tests.** Run: `... -m pytest tests/test_sensor_modes.py tests/test_wrapper.py tests/test_model_extra_in.py tests/test_uq_modes.py tests/test_regress_mode.py -q`. Expected: all pass.

- [ ] **Step 5: Commit** — `git add model/residual_wrapper.py model/u2net_hyperspectral.py tests/test_sensor_modes.py` ; `git commit -m "feat(sensor): modes uncond / cond_fam / regress_fam / varhead_fam — context-free U-Net path, s_post input slot, per-element residual scale, batch_to_cond for tuple and dict batches (Phase 3 Task 6)"`.

