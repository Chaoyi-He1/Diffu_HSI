"""Harness support for mode 'regress' (Phase 1a-prime Task 2): the one-row 'direct' grid, the one-call reconstruction,
the regressor's y-swap, the mode/grid/prediction_type guards and the y-swap script's diagnostics output directory."""
import argparse
import json
import os

import pytest
import torch

from model.diffusion_trainer import DiffusionTrainer
from model.residual_wrapper import REGRESS_T
from scripts import eval_warmstart as ew

CPU = torch.device('cpu')
DIRECT = dict(name='direct', method='direct', n_steps=1)


class Const(torch.nn.Module):
    """Predicts a constant residual; has a parameter so a missing no_grad shows up as requires_grad."""

    def __init__(self, v):
        super().__init__()
        self.v = v
        self.p = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x_t, cond, t):
        return torch.full_like(cond['x0_hat'], self.v) + 0 * self.p


class Spy(torch.nn.Module):
    """Checks the regressor's call (zero x_t placeholder, t = REGRESS_T) and reads BOTH condition inputs."""

    def forward(self, x_t, cond, t):
        assert t.tolist() == [REGRESS_T] * x_t.shape[0] and float(x_t.abs().max()) == 0.0
        return cond['x0_hat'] + cond['sensor'].mean((1, 2, 3), keepdim=True)


def test_direct_grid_is_one_call():
    assert ew.sampler_grid('direct') == [DIRECT]
    assert 'direct' not in [c['name'] for c in ew.sampler_grid('full')]
    with pytest.raises(ValueError):
        ew.sampler_grid('fast')


def test_reconstruct_direct_is_x0_hat_plus_sigma_times_one_prediction():
    tr = DiffusionTrainer(device=CPU, prediction_type='direct', snr_gamma=None)
    xh = torch.rand(2, 4, 8, 8) * 1.2 - 0.6
    batch = (torch.zeros(2, 4, 8, 8), torch.zeros(2, 3, 2, 2), xh)
    out = ew.reconstruct(Const(3.0), tr, 'regress', batch, DIRECT, sigma_d=0.1, seed=0)
    assert torch.allclose(out, (xh + 0.3).clamp(-1, 1)) and tr.last_nfe == 1 and not out.requires_grad
    y = torch.rand(2, 3, 2, 2)                                     # the model's inputs: zero x_t, t = REGRESS_T, both conds
    out = ew.reconstruct(Spy(), tr, 'regress', (torch.zeros_like(xh), y, xh), DIRECT, sigma_d=0.1, seed=0)
    assert torch.allclose(out, (xh + 0.1 * (xh + y.mean((1, 2, 3), keepdim=True))).clamp(-1, 1))
    with pytest.raises(ValueError, match='regress'):
        ew.reconstruct(Const(3.0), tr, 'residual', batch, DIRECT, 0.1, 0)
    with pytest.raises(ValueError, match='regress'):
        ew.reconstruct(Const(3.0), tr, 'regress', batch, dict(name='ddim1', method='ddim', n_steps=1), 0.1, 0)


def test_reconstruct_direct_clamps_to_the_cube_range():
    tr = DiffusionTrainer(device=CPU, prediction_type='direct', snr_gamma=None)
    xh = torch.rand(2, 4, 8, 8) * 1.2 - 0.6
    out = ew.reconstruct(Const(30.0), tr, 'regress', (torch.zeros_like(xh), torch.zeros(2, 3, 2, 2), xh), DIRECT,
                         sigma_d=0.1, seed=0)
    assert torch.equal(out, torch.ones_like(xh))


def test_yswap_regress_composes_every_variant_with_the_true_x0_hat():
    tr = DiffusionTrainer(device=CPU, prediction_type='direct', snr_gamma=None)
    torch.manual_seed(0)
    x = torch.rand(4, 4, 8, 8) * 1.6 - 0.8
    batch = (x, torch.rand(4, 3, 2, 2), 0.9 * x)                  # x0_hat = 0.9 x and sigma_d = 0.1, so r = x

    class Oracle(torch.nn.Module):                                 # reads its condition: f(x0_hat) = x0_hat / 0.9
        def forward(self, x_t, cond, t):
            assert t.tolist() == [REGRESS_T] * x_t.shape[0] and float(x_t.abs().max()) == 0.0
            return cond['x0_hat'] / 0.9
    files = [f'MatFlower60/Train/P0{22 + i % 3}/f{i}_1.mat' for i in range(4)]
    res = ew.yswap_check(Oracle(), tr, 'regress', batch, t_values=(100, 300), sigma_d=0.1, files=files)
    assert set(res) == {f't{REGRESS_T}', 'passes', 'per_cube', 'baseline_per_cube', 'files'}     # t forced to REGRESS_T
    cubes = res['per_cube'][str(REGRESS_T)]
    assert max(cubes['true']) < 1e-4 and min(cubes['swap_full']) > 1e-2 and res['passes']
    blind = ew.yswap_check(Const(0.0), tr, 'regress', batch, sigma_d=0.1)
    c = blind['per_cube'][str(REGRESS_T)]
    assert c['true'] == c['swap_full'] == c['swap_sensor_only'] and not blind['passes']
    with pytest.raises(ValueError, match='sigma_d'):
        ew.yswap_check(Oracle(), tr, 'regress', batch)


def test_checkpoint_eval_refuses_mode_grid_and_prediction_type_mismatches(tmp_path):
    base = dict(d=4, ckpt=str(tmp_path / 'missing.pth'), base_channels=8, seeds=[0, 1], batch_size=4, device='cpu',
                data_root=str(tmp_path), prior=str(tmp_path / 'prior.npz'), run_name=None, yswap_n=None)
    for kw, msg in ((dict(mode='regress', grid='full', prediction_type='direct'), 'grid'),
                    (dict(mode='residual', grid='direct', prediction_type='v'), 'grid'),
                    (dict(mode='regress', grid='direct', prediction_type='v'), 'prediction_type'),
                    (dict(mode='residual', grid='full', prediction_type='direct'), 'prediction_type')):
        with pytest.raises(ValueError, match=msg):
            ew.checkpoint_eval(argparse.Namespace(**base, **kw))     # refused before anything is loaded


@pytest.mark.parametrize('mode, prediction_type, grid, yswap_n, expected', [
    ('regress', 'direct', 'direct', None, 32),                       # the regressor's default
    ('residual', 'v', 'full', None, 8),                              # every other mode keeps the gate's 8
    ('regress', 'direct', 'direct', 16, 16),                         # an explicit --yswap_n wins
    ('standard', 'v', 'quick', 5, 5),
])
def test_checkpoint_eval_yswap_n_reaches_yswap_batch_and_provenance(tmp_path, monkeypatch, mode, prediction_type,
                                                                    grid, yswap_n, expected):
    """The y-swap cube count flows args -> yswap_batch(n) -> provenance. Every data-touching collaborator is stubbed
    and the grid is emptied, so only the y-swap path runs (no dataset, no real model)."""
    from types import SimpleNamespace

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(()))
    ckpt, prior = tmp_path / 'tiny.pth', tmp_path / 'prior.npz'
    torch.save({'model_state_dict': Tiny().state_dict(), 'meta': dict(args=dict(loss_type='l1', seed=43))}, ckpt)
    prior.write_bytes(b'stub')
    ds = SimpleNamespace(sigma_d=0.1, prior={'s': 1.0}, img_list=[])
    seen = []
    monkeypatch.setattr(ew, 'build_model', lambda *a: Tiny())
    monkeypatch.setattr(ew, 'dataset_for', lambda *a, **k: ds)
    monkeypatch.setattr(ew, 'check_meta', lambda *a: True)
    monkeypatch.setattr(ew, 'baseline_eval', lambda *a, **k: dict(mean={}, per_item={}, files=[]))
    monkeypatch.setattr(ew, 'sampler_grid', lambda which: [])
    monkeypatch.setattr(ew, 'yswap_batch', lambda root, prior_path, d, n=8: (seen.append(n), (ds, None))[1])
    monkeypatch.setattr(ew, 'yswap_check', lambda *a, **k: {})
    monkeypatch.setattr(ew, 'EVAL_DIR', str(tmp_path / 'eval'))
    ew.checkpoint_eval(argparse.Namespace(d=4, ckpt=str(ckpt), mode=mode, grid=grid, run_name='r', base_channels=8,
                                          prediction_type=prediction_type, seeds=[0, 1], batch_size=4, device='cpu',
                                          data_root=str(tmp_path), prior=str(prior), yswap_n=yswap_n))
    suffix = '' if grid == 'full' else f'__{grid}'
    prov = json.load(open(tmp_path / 'eval' / mode / 'd4' / f'r__tiny{suffix}.json'))['provenance']
    assert seen == [expected] and prov['yswap_n'] == expected
    assert prov['loss_type'] == 'l1' and prov['train_seed'] == 43


def test_cli_passes_grid_direct_and_yswap_n_to_checkpoint_eval(monkeypatch):
    import sys
    got = []
    monkeypatch.setattr(ew, 'checkpoint_eval', got.append)
    cmd = ['eval_warmstart.py', '--d', '4', '--ckpt', 'x.pth', '--mode', 'regress', '--prediction_type', 'direct',
           '--grid', 'direct']
    monkeypatch.setattr(sys, 'argv', cmd)
    ew.main()
    monkeypatch.setattr(sys, 'argv', cmd + ['--yswap_n', '16'])
    ew.main()
    a, b = got
    assert (a.mode, a.prediction_type, a.grid, a.yswap_n) == ('regress', 'direct', 'direct', None)   # None -> 32 later
    assert b.yswap_n == 16


def test_yswap_one_step_writes_only_under_the_diagnostics_dir(tmp_path, monkeypatch):
    from scripts import yswap_one_step as ys
    root = tmp_path / 'root'
    monkeypatch.setattr(ys, 'RESULTS_ROOT', str(root))
    monkeypatch.setattr(ys, 'DIAG_DIR', str(root / 'diagnostics'))
    missing = str(tmp_path / 'missing.pth')
    with pytest.raises(SystemExit, match='results tree'):
        ys.main(['--checkpoint', missing, '--mode', 'residual', '--out', str(root / 'eval' / 'x.json')])
    with pytest.raises(FileNotFoundError):                       # passes the guard, then fails on the missing checkpoint
        ys.main(['--checkpoint', missing, '--mode', 'residual', '--out', str(root / 'diagnostics' / 'x.json')])


@pytest.mark.dataset
def test_checkpoint_eval_regress_writes_a_one_row_direct_evaluation(data_root, tmp_path, monkeypatch):
    from data_loader.linear_estimate import load_prior
    from data_loader.warmstart_paths import PRIOR
    if not os.path.exists(PRIOR):
        pytest.skip('prior missing')
    torch.manual_seed(0)
    sd = ew.build_model('regress', 8, CPU).state_dict()
    p = load_prior(PRIOR)
    meta = dict(mode='regress', d=4, base_channels=8, prediction_type='direct',
                sigma_d=float(p['sigma_d'][list(p['d_values']).index(4)]), prior_sha256=ew.file_sha256(PRIOR),
                args=dict(loss_type='l1', seed=43))                  # non-default: the gate pairs runs on these
    ckpt = tmp_path / 'tiny.pth'
    torch.save({'model_state_dict': sd, 'meta': meta}, ckpt)
    real = ew.load_subset
    pick = list(range(0, 3)) + list(range(24, 27)) + list(range(176, 178))              # 3 P022, 3 P023, 2 P024
    monkeypatch.setattr(ew, 'load_subset', lambda root, which='files': (
        [real(root, 'files')[i] for i in pick] if which == 'files' else real(root, which)[:4]))
    monkeypatch.setattr(ew, 'subset_class_counts', lambda: {'P022': 3, 'P023': 3, 'P024': 2})
    monkeypatch.setattr(ew, 'EVAL_DIR', str(tmp_path))
    real_yb, seen_n = ew.yswap_batch, []
    monkeypatch.setattr(ew, 'yswap_batch', lambda root, prior_path, d, n=8: (
        seen_n.append(n), real_yb(root, prior_path, d, n))[1])
    args = argparse.Namespace(d=4, ckpt=str(ckpt), mode='regress', grid='direct', run_name='tiny_run', base_channels=8,
                              prediction_type='direct', seeds=[0, 1], batch_size=4, device='cpu', data_root=data_root,
                              prior=PRIOR, yswap_n=None)
    ew.checkpoint_eval(args)
    out = json.load(open(tmp_path / 'regress' / 'd4' / 'tiny_run__tiny__direct.json'))
    assert [r['name'] for r in out['rows']] == ['direct'] and out['rows'][0]['nfe'] == 1
    assert out['grid'] == 'direct' and out['provenance']['loss_type'] == 'l1' and out['provenance']['train_seed'] == 43
    assert out['provenance']['meta_checked'] is True and out['provenance']['prediction_type'] == 'direct'
    assert out['provenance']['yswap_n'] == 32 and len(out['yswap']['files']) == 8   # regress default; all 8 picked cubes
    assert seen_n == [32]                                                     # the count reached yswap_batch
    ys = out['yswap']
    assert set(ys) >= {f't{REGRESS_T}', 'per_cube', 'baseline_per_cube', 'files'}
    assert len(ys['files']) == len(ys['baseline_per_cube'])
    assert out['rows'][0]['rmse_pct_seed_std'] == 0.0                          # deterministic: both seeds identical
