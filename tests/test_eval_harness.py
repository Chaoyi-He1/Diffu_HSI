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


def _cpu_trainer():
    return DiffusionTrainer(device=torch.device('cpu'), prediction_type='x0', snr_gamma=None)


def test_yswap_requires_batch_ge_2():
    batch = (torch.zeros(1, 4, 8, 8), torch.zeros(1, 3, 2, 2), torch.zeros(1, 4, 8, 8))
    with pytest.raises(ValueError):
        yswap_check(ZeroModel(), _cpu_trainer(), 'residual', batch, sigma_d=0.05)


def test_yswap_residual_mode_requires_sigma_d():
    """A forgotten sigma_d would silently use the wrong residual scale, so residual mode must refuse it."""
    batch = (torch.zeros(4, 4, 8, 8), torch.zeros(4, 3, 2, 2), torch.zeros(4, 4, 8, 8))
    with pytest.raises(ValueError, match='sigma_d'):
        yswap_check(ZeroModel(), _cpu_trainer(), 'residual', batch)


def test_yswap_fails_for_condition_blind_model():
    """A network that ignores cond must fail even with a realistic prior (x0_hat = x + 0.05 noise, sigma_d = 0.05):
    its error is identical under every condition variant, so true / swap_full = 1 > 0.9."""
    class Blind(torch.nn.Module):
        def forward(self, x_t, cond, t):
            return 0.5 * x_t                                   # never reads cond
    torch.manual_seed(0)
    x = torch.rand(4, 4, 8, 8) * 2 - 1
    x0_hat = x + 0.05 * torch.randn_like(x)
    batch = (x, torch.rand(4, 3, 2, 2), x0_hat)
    res = yswap_check(Blind(), _cpu_trainer(), 'residual', batch, sigma_d=0.05)
    assert res['passes'] is False
    assert set(res['t100'].keys()) == {'true', 'swap_full', 'swap_sensor_only', 'zero'}
    for tkey in ('t100', 't300'):
        row = res[tkey]
        assert row['swap_full'] == pytest.approx(row['true'], rel=1e-6)          # swapping the condition changes nothing
        assert row['swap_sensor_only'] == pytest.approx(row['true'], rel=1e-6)
        assert row['zero'] == pytest.approx(row['true'], rel=1e-6)


def test_yswap_passes_for_oracle_model():
    """A model that genuinely reads cond passes. x0_hat_true = 0.9 x and sigma_d = 0.1 give the residual r = x.
    The stub returns cond['x0_hat'] / 0.9: that is r exactly under the true condition and another cube's spectrum
    under swap_full, so swapping hurts it."""
    class Oracle(torch.nn.Module):
        def forward(self, x_t, cond, t):
            return cond['x0_hat'] / 0.9
    torch.manual_seed(0)
    x = torch.rand(4, 4, 8, 8) * 2 - 1
    x0_hat, sigma_d = 0.9 * x, 0.1
    y = torch.rand(4, 3, 2, 2)
    batch = (x, y, x0_hat)
    t = torch.full((4,), 100, dtype=torch.long)
    out_true = Oracle()(x, {'sensor': y, 'x0_hat': x0_hat}, t)
    out_swap = Oracle()(x, {'sensor': y.roll(2, 0), 'x0_hat': x0_hat.roll(2, 0)}, t)
    assert torch.allclose(out_true, (x - x0_hat) / sigma_d, atol=1e-5)           # exact residual under the true condition
    assert not torch.allclose(out_true, out_swap, atol=1e-2)                      # and the output really depends on cond
    res = yswap_check(Oracle(), _cpu_trainer(), 'residual', batch, sigma_d=sigma_d)
    assert res['passes'] is True
    for tkey in ('t100', 't300'):
        assert res[tkey]['true'] < 1e-3 and res[tkey]['swap_full'] > 1.0


def test_yswap_swaps_only_network_inputs_by_half_batch():
    """Every variant feeds the network a rolled-by-B//2 condition (never roll-by-1, which pairs neighbouring
    patches of one source image) and composes with the TRUE x0_hat."""
    seen = []

    class Recorder(torch.nn.Module):
        def forward(self, x_t, cond, t):
            seen.append({k: v.clone() for k, v in cond.items()})
            return torch.zeros_like(x_t)
    torch.manual_seed(0)
    B = 6
    x, y, x0_hat = torch.rand(B, 4, 8, 8) * 2 - 1, torch.rand(B, 3, 2, 2), torch.rand(B, 4, 8, 8) * 2 - 1
    res = yswap_check(Recorder(), _cpu_trainer(), 'residual', (x, y, x0_hat), t_values=(100,), sigma_d=0.1)
    assert len(seen) == 4                                                          # true, swap_full, swap_sensor_only, zero
    assert torch.equal(seen[0]['sensor'], y) and torch.equal(seen[0]['x0_hat'], x0_hat)
    assert torch.equal(seen[1]['sensor'], y.roll(B // 2, 0)) and torch.equal(seen[1]['x0_hat'], x0_hat.roll(B // 2, 0))
    assert torch.equal(seen[2]['sensor'], y.roll(B // 2, 0)) and torch.equal(seen[2]['x0_hat'], x0_hat)
    assert not seen[3]['sensor'].any() and not seen[3]['x0_hat'].any()
    # the stub predicts r = 0, so every variant must give x_hat = clamp(x0_hat_true): identical errors in all four rows
    row = res['t100']
    assert len({round(v, 6) for v in row.values()}) == 1


def test_yswap_standard_mode_uses_prediction_directly():
    class ReadsX0Hat(torch.nn.Module):
        def forward(self, x_t, cond, t):
            return cond['x0_hat']
    x = torch.rand(4, 4, 8, 8) * 2 - 1
    batch = (x, torch.rand(4, 3, 2, 2), x.clone())
    res = yswap_check(ReadsX0Hat(), _cpu_trainer(), 'standard', batch)              # sigma_d is not needed in standard mode
    assert res['passes'] is True and res['t100']['true'] < 1e-3 and res['t100']['swap_full'] > 1.0


def test_summarize_seeds_averages_over_seeds_before_se():
    """Cubes, not cube x seed pairs, are the independent units: per-cube seed means, then mean and SE over cubes."""
    from scripts.eval_warmstart import summarize_seeds
    per = {'rmse_pct': [1.0, 2.0, 3.0, 4.0, 1.2, 2.2, 3.2, 4.2]}                  # seed-major: 2 seeds x 4 cubes
    s = summarize_seeds(per, ['A', 'A', 'B', 'B'], n_seeds=2)
    assert s['rmse_pct'] == pytest.approx(2.6)
    assert s['rmse_pct_se'] == pytest.approx(np.sqrt(5.0 / 3.0) / 2.0)           # std of [1.1, 2.1, 3.1, 4.1] / sqrt(4)
    naive = np.std(per['rmse_pct'], ddof=1) / np.sqrt(8)                          # the old cubes x seeds formula, 0.424
    assert abs(s['rmse_pct_se'] - naive) > 0.1
    assert s['rmse_pct_seed_std'] == pytest.approx(np.std([2.5, 2.7], ddof=1))   # spread of the per-seed means
    assert s['rmse_pct_A'] == pytest.approx(1.6) and s['rmse_pct_B'] == pytest.approx(3.6)


def test_aggregate_row_paired_difference_is_per_cube():
    from scripts.eval_warmstart import aggregate_row
    per = {'rmse_pct': [1.5, 2.5, 2.0, 5.0, 0.7, 1.7, 4.0, 3.0]}                  # seed means: 1.1, 2.1, 3.0, 4.0
    base = [1.0, 2.0, 3.0, 4.0]
    r = aggregate_row(per, base, ['A', 'A', 'B', 'B'], n_seeds=2)
    assert r['paired_rmse_diff'] == pytest.approx(0.05)                           # per-cube diffs 0.1, 0.1, 0, 0
    assert r['paired_rmse_se'] == pytest.approx(np.sqrt(0.01 / 3.0) / 2.0)
    with pytest.raises(ValueError):
        aggregate_row({'rmse_pct': [1.0, 2.0, 3.0]}, [1.0, 2.0], ['A', 'A'], n_seeds=2)   # not seed-major N x S


def test_yswap_files_is_stratified_by_class():
    """The subset file list is grouped P022 (24), P023 (152), P024 (24); the y-swap batch takes evenly spaced files from
    each block (1 / 6 / 1), so the hard class P024 is represented and no two cubes share a source image."""
    from scripts.eval_warmstart import yswap_files, subset_class_counts, class_of
    sub = json.load(open(os.path.join(REPO, 'tests', 'data', 'val_subset_200.json')))
    assert subset_class_counts() == sub['classes']
    picked = yswap_files(sub['files'], subset_class_counts())
    classes = [class_of(f) for f in picked]
    assert len(picked) == 8 and len(set(picked)) == 8
    assert [classes.count(c) for c in ('P022', 'P023', 'P024')] == [1, 6, 1]
    assert len({os.path.basename(f).split('_')[0] for f in picked}) == 8          # eight different source images
    assert picked == yswap_files(sub['files'], subset_class_counts())             # deterministic
    with pytest.raises(ValueError):
        yswap_files(sub['files'][:100], subset_class_counts())                     # counts do not describe the list


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


@pytest.mark.dataset
def test_dataset_for_rejects_files_outside_the_validation_split(data_root):
    from scripts.eval_warmstart import dataset_for
    prior = os.path.join(REPO, 'results', 'residual_warmstart', 'linear_prior_R1.npz')
    if not os.path.exists(prior):
        pytest.skip('prior missing')
    sub = json.load(open(os.path.join(REPO, 'tests', 'data', 'val_subset_200.json')))
    good = os.path.join(data_root, sub['files'][0])
    assert len(dataset_for(data_root, prior, 4, [good]).img_list) == 1
    with pytest.raises(AssertionError, match='1 of 2'):
        dataset_for(data_root, prior, 4, [good, os.path.join(data_root, 'MatFlower60', 'Train', 'P022', 'not_a_file.mat')])


@pytest.mark.dataset
def test_checkpoint_eval_records_per_cube_values(data_root, tmp_path, monkeypatch):
    """Runs the real checkpoint_eval on 8 cubes with a tiny random-weight model and a 2-row sampler grid."""
    import argparse
    from scripts import eval_warmstart as ew
    prior = os.path.join(REPO, 'results', 'residual_warmstart', 'linear_prior_R1.npz')
    if not os.path.exists(prior):
        pytest.skip('prior missing')
    torch.manual_seed(0)
    ckpt = tmp_path / 'tiny.pth'
    torch.save(ew.build_model('residual', 8, torch.device('cpu')).state_dict(), ckpt)   # wrapper state dict, keys 'net.*'
    real_load_subset = ew.load_subset
    pick = list(range(0, 3)) + list(range(24, 27)) + list(range(176, 178))              # 3 x P022, 3 x P023, 2 x P024
    monkeypatch.setattr(ew, 'load_subset', lambda root, which='files': (
        [real_load_subset(root, 'files')[i] for i in pick] if which == 'files' else real_load_subset(root, which)[:4]))
    monkeypatch.setattr(ew, 'subset_class_counts', lambda: {'P022': 3, 'P023': 3, 'P024': 2})
    monkeypatch.setattr(ew, 'sampler_grid', lambda: [dict(name='ddim2', method='ddim', n_steps=2),
                                                     dict(name='warm50_ddim', method='ddim', n_steps=2, t_start=50)])
    monkeypatch.setattr(ew, 'EVAL_DIR', str(tmp_path))
    args = argparse.Namespace(d=4, ckpt=str(ckpt), mode='residual', base_channels=8, prediction_type='v', seeds=[0, 1],
                              batch_size=4, device='cpu', data_root=data_root, prior=prior)
    ew.checkpoint_eval(args)
    out = json.load(open(tmp_path / 'tiny_residual_d4.json'))
    assert len(out['yswap_files']) == 8 and 'passes' in out['yswap']
    assert [f.split('/')[2] for f in out['yswap_files']].count('P024') == 2          # the stratified pick reached P024
    assert len(out['baseline_files']) == 8 and len(out['baseline_per_item']['rmse_pct']) == 8
    assert len(out['rows']) == 2
    for row in out['rows']:
        per = np.asarray(row['per_item']['rmse_pct'])
        assert per.shape == (2, 8) and len(row['files']) == 8                      # [seed][cube]; cube order = row["files"] (dataset sorted order)
        assert row['seeds'] == [0, 1] and row['batch_size'] == 4 and row['nfe'] == 2
        assert set(row['per_item']) == {'rmse_pct', 'psnr', 'sam_deg'}
        assert not np.array_equal(per[0], per[1])                                      # seeds really differ
        cube = per.mean(0)
        assert row['rmse_pct'] == pytest.approx(cube.mean())
        assert row['rmse_pct_se'] == pytest.approx(cube.std(ddof=1) / np.sqrt(8))    # SE over 8 cubes, not 16 pairs
        assert row['rmse_pct_seed_std'] == pytest.approx(np.std(per.mean(1), ddof=1))
        base = np.asarray(out['baseline_per_item']['rmse_pct'])
        assert row['files'] == out['baseline_files']
        assert row['paired_rmse_diff'] == pytest.approx((cube - base).mean())
