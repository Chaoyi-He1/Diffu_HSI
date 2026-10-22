"""Phase 2a Task 2: the two new modes share B's architecture (128 input channels); residual_reg has B's target with
the cached centre; varhead's one call sits on x_r at t = REGRESS_T and returns a log-variance scored by the Gaussian NLL."""
import math

import pytest
import torch

from model.residual_wrapper import (CENTRED_MODES, LOGVAR_RANGE, MODES, REGRESS_T, XR_MODES, build_residual_model,
                                    compose, target_for, varhead_forward, varhead_nll)
from model.varhead_trainer import VarheadTrainer


def test_mode_tables():
    assert MODES == ('residual', 'standard', 'standard_xhat', 'regress', 'residual_reg', 'varhead')
    assert XR_MODES == ('residual_reg', 'varhead') and set(XR_MODES) < set(MODES)
    assert CENTRED_MODES == ('residual', 'regress', 'residual_reg', 'varhead')


@pytest.mark.parametrize('mode', ['residual_reg', 'varhead'])
def test_new_modes_take_128_input_channels_like_b(mode):
    m = build_residual_model(mode, base_channels=8)
    assert m.use_x0_hat and m.net.input_proj.in_channels == 128
    x_t, xr, y = torch.zeros(2, 64, 16, 16), torch.rand(2, 64, 16, 16) * 2 - 1, torch.rand(2, 30, 4, 4)
    out = m(x_t, {'sensor': y, 'x0_hat': xr}, torch.full((2,), 10, dtype=torch.long))
    assert out.shape == (2, 64, 16, 16)


def test_residual_reg_and_varhead_targets_are_the_scaled_residual_to_the_centre():
    x, xr = torch.rand(2, 64, 4, 4) * 2 - 1, torch.rand(2, 64, 4, 4) * 2 - 1
    for mode in ('residual_reg', 'varhead'):
        assert torch.equal(target_for(mode, x, xr, 0.09), (x - xr) / 0.09)
    assert torch.equal(target_for('standard', x, xr, 0.09), x)


def test_varhead_forward_puts_the_centre_in_the_x_t_slot_at_regress_t():
    seen = {}

    class Spy(torch.nn.Module):
        def forward(self, x_t, cond, t):
            seen['x_t'], seen['t'] = x_t, t
            return torch.zeros_like(x_t)

    xr = torch.rand(3, 64, 8, 8)
    varhead_forward(Spy(), {'sensor': torch.zeros(3, 30, 2, 2), 'x0_hat': xr})
    assert torch.equal(seen['x_t'], xr) and torch.equal(seen['t'], torch.full((3,), REGRESS_T, dtype=torch.long))


def test_varhead_nll_closed_form_and_minimum():
    r = torch.tensor([2.0])
    assert torch.allclose(varhead_nll(torch.tensor([0.0]), r), torch.tensor([2.0]))          # 0.5 * (0 + 4)
    best = math.log(4.0)
    at_best = varhead_nll(torch.tensor([best]), r)
    assert at_best < varhead_nll(torch.tensor([best - 0.5]), r) and at_best < varhead_nll(torch.tensor([best + 0.5]), r)
    assert torch.allclose(at_best, torch.tensor([0.5 * (best + 1.0)]))


def test_varhead_nll_clamps_the_log_variance():
    r = torch.tensor([1.0])
    assert torch.equal(varhead_nll(torch.tensor([-50.0]), r), varhead_nll(torch.tensor([LOGVAR_RANGE[0]]), r))
    assert torch.equal(varhead_nll(torch.tensor([50.0]), r), varhead_nll(torch.tensor([LOGVAR_RANGE[1]]), r))


def test_compose_adds_the_scaled_residual_and_clamps():
    c = torch.tensor([0.9, -0.5]); r = torch.tensor([3.0, 1.0])
    assert torch.allclose(compose(c, 0.1, r), torch.tensor([1.0, -0.4]))


def test_varhead_trainer_loss_is_the_mean_nll_of_the_networks_logvar():
    class Const(torch.nn.Module):
        def forward(self, x_t, cond, t):
            return torch.full_like(x_t, 0.5)

    tr = VarheadTrainer(device=torch.device('cpu'))
    assert tr.prediction_type == 'direct' and tr.loss_type == 'nll' and tr.last_nfe == 1
    r = torch.full((2, 64, 4, 4), 1.5)
    loss, info = tr.get_loss(Const(), r, {'sensor': torch.zeros(2, 30, 1, 1), 'x0_hat': torch.zeros(2, 64, 4, 4)}, loss_in_fp32=True)
    expected = 0.5 * (0.5 + 1.5 ** 2 * math.exp(-0.5))
    assert abs(loss.item() - expected) < 1e-6 and info['t'] == REGRESS_T and info['logvar'].shape == r.shape
