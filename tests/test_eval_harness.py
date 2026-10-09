import json
import os
import numpy as np
import pytest
import torch

from scripts.eval_warmstart import metrics, sampler_grid, reconstruct, yswap_check
from model.diffusion_trainer import DiffusionTrainer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_metrics_perfect_and_known_error():
    x = torch.rand(2, 4, 8, 8) * 2 - 1
    m = metrics(x, x)
    assert torch.allclose(m['rmse_pct'], torch.zeros(2)) and torch.all(m['sam_deg'] < 1e-3)
    x_hat = x.clone(); x_hat[0] += 0.2                    # +0.1 on the [0,1] scale for item 0
    m = metrics(x_hat.clamp(-1, 1), x)
    assert abs(m['rmse_pct'][0].item() - 10.0) < 1.5 and m['rmse_pct'][1].item() == 0.0
    assert abs(m['psnr'][0].item() - 20.0) < 1.5


def test_sampler_grid_names_and_nfe_budget():
    g = sampler_grid()
    names = [c['name'] for c in g]
    assert names[:6] == ['ddpm1000', 'ddim1', 'ddim2', 'ddim5', 'ddim10', 'ddim20']
    assert 'warm200_ddim' in names and 'warm400_ddpm' in names
    assert all('t_start' in c for c in g if c['name'].startswith('warm'))


class ZeroModel(torch.nn.Module):
    def forward(self, x_t, cond, t):
        return torch.zeros_like(x_t)


def test_zero_residual_equals_baseline():
    tr = DiffusionTrainer(device=torch.device('cpu'), prediction_type='x0', snr_gamma=None)
    x0_hat = torch.rand(2, 4, 8, 8) * 2 - 1
    batch = (torch.zeros(2, 4, 8, 8), torch.zeros(2, 3, 2, 2), x0_hat)
    for cfg in [dict(name='ddim5', method='ddim', n_steps=5), dict(name='warm50_ddim', method='ddim', n_steps=10, t_start=50)]:
        out = reconstruct(ZeroModel(), tr, 'residual', batch, cfg, sigma_d=0.1, seed=0)
        assert torch.allclose(out, x0_hat, atol=1e-6)


def test_residual_mode_does_not_clamp_residual_to_unit():
    """A residual prediction of 3 sigma must survive sampling: x_hat = x0_hat + sigma_d * 3, not + sigma_d * 1."""
    class ConstResidual(torch.nn.Module):
        def forward(self, x_t, cond, t):
            return torch.full_like(x_t, 3.0)
    tr = DiffusionTrainer(device=torch.device('cpu'), prediction_type='x0', snr_gamma=None)
    x0_hat = torch.zeros(1, 4, 8, 8)
    batch = (torch.zeros(1, 4, 8, 8), torch.zeros(1, 3, 2, 2), x0_hat)
    cfg = dict(name='ddim1', method='ddim', n_steps=1)
    out = reconstruct(ConstResidual(), tr, 'residual', batch, cfg, sigma_d=0.1, seed=0)
    assert torch.allclose(out, torch.full_like(out, 0.3), atol=1e-6)


def test_yswap_requires_batch_ge_2():
    tr = DiffusionTrainer(device=torch.device('cpu'), prediction_type='x0', snr_gamma=None)
    batch = (torch.zeros(1, 4, 8, 8), torch.zeros(1, 3, 2, 2), torch.zeros(1, 4, 8, 8))
    with pytest.raises(ValueError):
        yswap_check(ZeroModel(), tr, 'residual', batch)


def test_yswap_fails_for_condition_blind_model():
    tr = DiffusionTrainer(device=torch.device('cpu'), prediction_type='x0', snr_gamma=None)
    x = torch.rand(4, 4, 8, 8) * 2 - 1
    batch = (x, torch.rand(4, 3, 2, 2), torch.rand(4, 4, 8, 8) * 2 - 1)
    res = yswap_check(ZeroModel(), tr, 'residual', batch)
    assert res['passes'] is False
    assert set(res['t100'].keys()) == {'true', 'swap_full', 'swap_sensor_only', 'zero'}


def test_yswap_passes_for_oracle_model():
    """A model that reads x0_hat (= the truth here) passes: swapping the condition hurts it."""
    class Oracle(torch.nn.Module):
        def forward(self, x_t, cond, t):
            return (x_t * 0 + 0)  # residual-mode x0 prediction of r = 0 → x_hat = cond['x0_hat']
    tr = DiffusionTrainer(device=torch.device('cpu'), prediction_type='x0', snr_gamma=None)
    x = torch.rand(4, 4, 8, 8) * 2 - 1
    batch = (x, torch.rand(4, 3, 2, 2), x.clone())           # x0_hat equals the truth
    res = yswap_check(Oracle(), tr, 'residual', batch)
    assert res['passes'] is True


@pytest.mark.dataset
def test_baseline_eval_reproduces_recorded_numbers(data_root):
    from scripts.eval_warmstart import baseline_eval
    rec_path = os.path.join(REPO, 'tests', 'data', 'baseline_phase0.json')
    prior = os.path.join(REPO, 'results', 'residual_warmstart', 'linear_prior_R1.npz')
    if not (os.path.exists(rec_path) and os.path.exists(prior)):
        pytest.skip('baseline record or prior missing')
    rec = json.load(open(rec_path))
    sub = json.load(open(os.path.join(REPO, 'tests', 'data', 'val_subset_200.json')))
    files = [os.path.join(data_root, f) for f in sub['files'][:40]]
    out = baseline_eval(data_root, prior, d=4, files=files)
    assert abs(out['mean']['rmse_pct'] - rec['d4']['subset40_rmse_pct']) < 0.01
