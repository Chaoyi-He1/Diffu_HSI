import os

import torch
import pytest

from model.diffusion_trainer import DiffusionTrainer

REFERENCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'sampler_reference.pt')


class CountingIdentityX0(torch.nn.Module):
    """x0-prediction model that always predicts the constant x0 it is given, and counts calls."""
    def __init__(self, x0):
        super().__init__()
        self.x0 = x0
        self.calls = 0
        self.seen_t = []

    def forward(self, x_t, cond, t):
        self.calls += 1
        self.seen_t.append(int(t[0]))
        return self.x0.expand_as(x_t).clone()


class ShrinkX0(torch.nn.Module):
    """State-dependent x0 prediction (0.5 * x_t): the output depends on the noise stream and the whole chain."""
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.seen_t = []

    def forward(self, x_t, cond, t):
        self.calls += 1
        self.seen_t.append(int(t[0]))
        return 0.5 * x_t


def _trainer():
    return DiffusionTrainer(device=torch.device('cpu'), prediction_type='x0', clamp_x0=(-1.0, 1.0), snr_gamma=None)


@pytest.mark.parametrize('method', ['ddim', 'ddpm'])
def test_defaults_unchanged_against_reference_run(method):
    """The default path (x_init=None, t_start=None) must reproduce the output of the PRE-change sampler.

    tests/data/sampler_reference.pt was captured from the sampler before the warm-start option existed.
    """
    ref = torch.load(REFERENCE_PATH)
    tr = _trainer()
    shape = (1, 2, 4, 4)

    # constant-x0 stub (the reference run specified by the plan)
    m = CountingIdentityX0(torch.full(shape, 0.3))
    torch.manual_seed(0)
    out = tr.sample(m, None, shape, n_steps=10, method=method, progress=False, x_init=None, t_start=None)
    assert torch.allclose(out, ref[method])
    assert m.calls == ref[f'{method}_calls'] == 10
    assert m.seen_t == ref[f'{method}_t']
    assert tr.last_nfe == m.calls

    # state-dependent stub: its output depends on the noise draws and every update of the chain
    s = ShrinkX0()
    torch.manual_seed(0)
    out_s = tr.sample(s, None, shape, n_steps=10, method=method, progress=False)
    assert torch.allclose(out_s, ref[f'{method}_shrink'])
    assert s.calls == ref[f'{method}_shrink_calls']
    assert s.seen_t == ref[f'{method}_shrink_t']
    assert tr.last_nfe == s.calls


def test_warm_start_first_call_is_t_start_and_ends_above_zero():
    tr = _trainer()
    x0 = torch.full((1, 2, 4, 4), 0.3)
    m = CountingIdentityX0(x0)
    tr.sample(m, None, (1, 2, 4, 4), n_steps=10, method='ddim', progress=False, x_init=torch.zeros(1, 2, 4, 4), t_start=200)
    assert m.seen_t[0] == 200 and m.seen_t[-1] > 0 and m.calls == 10 and tr.last_nfe == 10
    assert all(a > b for a, b in zip(m.seen_t, m.seen_t[1:]))      # strictly descending


def test_warm_start_small_t_start_has_no_repeated_steps():
    tr = _trainer()
    m = CountingIdentityX0(torch.zeros(1, 2, 4, 4))
    tr.sample(m, None, (1, 2, 4, 4), n_steps=10, method='ddpm', progress=False, x_init=torch.zeros(1, 2, 4, 4), t_start=5)
    assert len(set(m.seen_t)) == len(m.seen_t) and m.calls <= 6
    assert m.seen_t == [5, 4, 3, 2, 1]


@pytest.mark.parametrize('method', ['ddim', 'ddpm'])
def test_warm_start_t_start_one_never_calls_model_at_zero(method):
    tr = _trainer()
    m = CountingIdentityX0(torch.zeros(1, 2, 4, 4))
    tr.sample(m, None, (1, 2, 4, 4), n_steps=10, method=method, progress=False, x_init=torch.zeros(1, 2, 4, 4), t_start=1)
    assert m.seen_t == [1] and m.calls == 1 and tr.last_nfe == 1


def test_warm_start_t0_zero_returns_init():
    tr = _trainer()
    m = CountingIdentityX0(torch.zeros(1, 2, 4, 4))
    x_init = torch.full((1, 2, 4, 4), 1.7)
    out = tr.sample(m, None, (1, 2, 4, 4), n_steps=10, method='ddim', progress=False, x_init=x_init, t_start=0)
    assert m.calls == 0 and tr.last_nfe == 0
    assert torch.allclose(out, x_init.clamp(-1, 1))


def test_warm_start_noise_level_matches_schedule():
    tr = _trainer()
    seen = {}

    class Probe(torch.nn.Module):
        def forward(self, x_t, cond, t):
            seen['x'] = x_t.clone(); return torch.zeros_like(x_t)

    x_init = torch.zeros(1, 1, 64, 64)
    torch.manual_seed(1)
    tr.sample(Probe(), None, (1, 1, 64, 64), n_steps=1, method='ddim', progress=False, x_init=x_init, t_start=400)
    expected_std = float(tr.sqrt_one_minus_alpha_bars[400])
    assert abs(seen['x'].std().item() - expected_std) < 0.05 * expected_std


def test_warm_start_signal_term_uses_x_init():
    tr = _trainer()
    seen = {}

    class Probe(torch.nn.Module):
        def forward(self, x_t, cond, t):
            seen['x'] = x_t.clone(); return torch.zeros_like(x_t)

    x_init = torch.full((1, 1, 64, 64), 0.7)
    torch.manual_seed(1)
    tr.sample(Probe(), None, (1, 1, 64, 64), n_steps=1, method='ddim', progress=False, x_init=x_init, t_start=400)
    expected_mean = float(tr.sqrt_alpha_bars[400]) * 0.7
    expected_std = float(tr.sqrt_one_minus_alpha_bars[400])
    assert abs(seen['x'].mean().item() - expected_mean) < 0.05
    assert abs(seen['x'].std().item() - expected_std) < 0.05 * expected_std


def test_warm_start_requires_both_args():
    tr = _trainer()
    m = CountingIdentityX0(torch.zeros(1, 2, 4, 4))
    with pytest.raises(ValueError):
        tr.sample(m, None, (1, 2, 4, 4), n_steps=10, progress=False, x_init=torch.zeros(1, 2, 4, 4))
