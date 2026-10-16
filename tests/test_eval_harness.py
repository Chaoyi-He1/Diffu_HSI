import json
import os
import re
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


VARIANTS = {'true', 'swap_full', 'swap_sensor_only', 'zero'}


class BlindHalf(torch.nn.Module):
    def forward(self, x_t, cond, t):
        return 0.5 * x_t                                       # never reads cond


def _blind_batch():
    torch.manual_seed(0)
    x = torch.rand(4, 4, 8, 8) * 2 - 1
    return x, torch.rand(4, 3, 2, 2), x + 0.05 * torch.randn_like(x)


def test_yswap_check_returns_per_cube_values_at_any_t():
    """Fix item 1 (a), additive: per-cube RMSE per t and variant, the per-cube baseline and the files in batch order;
    the summary rows and `passes` are unchanged. A condition-blind net scores exactly 1.000 at t = 999 too."""
    from scripts.eval_warmstart import metrics
    x, y, x0_hat = _blind_batch()
    files = [f'MatFlower60/Train/P023/{i}_1.mat' for i in range(4)]
    res = yswap_check(BlindHalf(), _cpu_trainer(), 'residual', (x, y, x0_hat), t_values=(999,), sigma_d=0.05, files=files)
    assert set(res) == {'t999', 'passes', 'files', 'per_cube', 'baseline_per_cube'}
    assert set(res['t999']) == VARIANTS and res['passes'] is False and res['files'] == files
    pc = res['per_cube']
    assert set(pc) == {'999'} and set(pc['999']) == VARIANTS and all(len(v) == 4 for v in pc['999'].values())
    assert pc['999']['true'] == pc['999']['swap_full']                                   # elementwise, exactly
    assert res['t999']['true'] / res['t999']['swap_full'] == 1.0
    for v in VARIANTS:
        assert res['t999'][v] == pytest.approx(np.mean(pc['999'][v]), rel=1e-6)          # the summary is the per-cube mean
    assert res['baseline_per_cube'] == pytest.approx(metrics(x0_hat, x)['rmse_pct'].tolist())   # as baseline_eval scores it


def test_yswap_check_default_keeps_the_spec_t_values_and_rule():
    """Without `files` the record is the old one plus per-cube values; t = 100 and 300 and the 0.9 rule are spec §7."""
    from scripts.eval_warmstart import YSWAP_RATIO
    res = yswap_check(BlindHalf(), _cpu_trainer(), 'residual', _blind_batch(), sigma_d=0.05)
    assert YSWAP_RATIO == 0.9
    assert {k for k in res if k.startswith('t')} == {'t100', 't300'} and set(res['per_cube']) == {'100', '300'}
    assert res['files'] is None and res['passes'] is False


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


def test_per_split_statistics_and_unseen_aggregate():
    """Per class and for the unseen aggregate (P023 + P024): mean and SE over cubes (after averaging over seeds),
    and in aggregate_row the paired difference to the baseline with its SE. Existing keys keep their values."""
    from scripts.eval_warmstart import summarize, aggregate_row
    classes = ['P022', 'P022', 'P023', 'P023', 'P024', 'P024']
    v = [1.0, 3.0, 2.0, 4.0, 6.0, 10.0]
    s = summarize({'rmse_pct': v}, classes)
    assert s['rmse_pct'] == pytest.approx(26.0 / 6.0) and s['rmse_pct_se'] == pytest.approx(np.std(v, ddof=1) / np.sqrt(6))
    assert (s['rmse_pct_P022'], s['rmse_pct_P022_se']) == pytest.approx((2.0, 1.0))
    assert (s['rmse_pct_P023'], s['rmse_pct_P023_se']) == pytest.approx((3.0, 1.0))
    assert (s['rmse_pct_P024'], s['rmse_pct_P024_se']) == pytest.approx((8.0, 2.0))
    assert (s['rmse_pct_unseen'], s['rmse_pct_unseen_se']) == pytest.approx((5.5, np.sqrt(35.0 / 3.0) / 2.0))
    assert 'rmse_pct_unseen' not in summarize({'rmse_pct': [1.0, 2.0]}, ['A', 'B'])   # no unseen class, no key

    per = {'rmse_pct': v + [x + 0.2 for x in v]}                                    # 2 seeds; seed means v + 0.1
    base = [0.5, 2.5, 2.0, 3.0, 5.0, 9.0]                                           # per-cube diffs 0.6 0.6 | 0.1 1.1 | 1.1 1.1
    r = aggregate_row(per, base, classes, n_seeds=2)
    assert (r['rmse_pct_unseen'], r['rmse_pct_unseen_se']) == pytest.approx((5.6, np.sqrt(35.0 / 3.0) / 2.0))
    assert r['paired_rmse_diff'] == pytest.approx(4.6 / 6.0)
    assert (r['paired_rmse_diff_P022'], r['paired_rmse_se_P022']) == pytest.approx((0.6, 0.0))
    assert (r['paired_rmse_diff_P023'], r['paired_rmse_se_P023']) == pytest.approx((0.6, 0.5))
    assert (r['paired_rmse_diff_P024'], r['paired_rmse_se_P024']) == pytest.approx((1.1, 0.0))
    assert (r['paired_rmse_diff_unseen'], r['paired_rmse_se_unseen']) == pytest.approx((0.85, 0.25))


def test_source_image_is_class_dir_plus_image_stem():
    """Fix item 10: the cluster of a cube is its source image, i.e. class dir + basename.rsplit('_', 1)[0]."""
    from scripts.paired_stats import source_image
    assert source_image('MatFlower60/Train/P023/34_10.mat') == source_image('/abs/MatFlower60/Train/P023/34_1.mat') == 'P023/34'
    assert source_image('MatFlower60/Train/P022/34_1.mat') != source_image('MatFlower60/Train/P023/34_1.mat')
    assert source_image('MatFlower60/Train/P023/3_4.mat') != source_image('MatFlower60/Train/P023/34_1.mat')


def test_cluster_se_is_cr1_over_source_images():
    from scripts.paired_stats import cluster_se
    # hand-computed CR1: d = [1, 3, 2, 6], images a a b c; dbar = 3, cluster sums of (d - dbar) = -2, -1, 3
    files = ['P023/a_1.mat', 'P023/a_2.mat', 'P023/b_1.mat', 'P023/c_1.mat']
    se, g = cluster_se([1.0, 3.0, 2.0, 6.0], files)
    assert g == 3 and se == pytest.approx(np.sqrt(3 / 2 * (4 + 1 + 9)) / 4, abs=1e-12)
    # one patch per image: CR1 equals the iid SE (ddof = 1)
    rng = np.random.default_rng(0)
    d = rng.standard_normal(40)
    se, g = cluster_se(d, [f'MatFlower60/Train/P023/{i}_1.mat' for i in range(40)])
    assert g == 40 and se == pytest.approx(d.std(ddof=1) / np.sqrt(40), abs=1e-12)
    # two patches of one image share an offset: the clustered SE exceeds the iid SE
    files = [f'MatFlower60/Train/P023/{i // 2}_{i % 2 + 1}.mat' for i in range(40)]
    d = np.repeat(rng.standard_normal(20), 2) + 0.1 * rng.standard_normal(40)
    se, g = cluster_se(d, files)
    assert g == 20 and se > 1.2 * d.std(ddof=1) / np.sqrt(40)
    assert cluster_se([1.0, 2.0], ['P023/a_1.mat', 'P023/a_2.mat']) == (None, 1)          # one cluster: no SE
    with pytest.raises(ValueError):
        cluster_se([1.0, 2.0], ['P023/a_1.mat'])


def test_aggregate_row_reports_cluster_robust_se():
    """Additive: with `files`, aggregate_row also reports the CR1 SE of the paired difference over source images and
    the number of images, overall and per split; every existing key keeps its value."""
    from scripts.eval_warmstart import aggregate_row
    from scripts.paired_stats import cluster_se
    classes = ['P022', 'P022', 'P023', 'P023', 'P023', 'P024']
    files = [f'MatFlower60/Train/{c}/{s}.mat' for c, s in zip(classes, ('1_1', '1_2', '5_1', '5_7', '6_1', '2_1'))]
    per = {'rmse_pct': [1.0, 3.0, 2.0, 4.0, 6.0, 10.0, 1.2, 3.2, 2.2, 4.4, 6.0, 10.2]}
    base = [0.5, 2.5, 2.0, 3.0, 5.0, 9.0]
    plain, r = aggregate_row(per, base, classes, n_seeds=2), aggregate_row(per, base, classes, n_seeds=2, files=files)
    assert 'paired_rmse_se_cluster' not in plain and {k: r[k] for k in plain} == plain
    diff = np.asarray(per['rmse_pct']).reshape(2, 6).mean(0) - np.asarray(base)
    assert (r['paired_rmse_se_cluster'], r['n_clusters']) == cluster_se(diff, files) and r['n_clusters'] == 4
    assert r['n_clusters_P023'] == 2 and r['paired_rmse_se_cluster_P023'] == pytest.approx(cluster_se(diff[2:5], files[2:5])[0])
    assert r['n_clusters_unseen'] == 3 and r['paired_rmse_se_cluster_P024'] is None             # P024 has one image


def test_reconstruct_x_init_per_mode():
    """Warm start: residual mode starts the chain from x_init = 0 (a zero residual), standard mode from x0_hat, so
    the first state's mean is 0 vs sqrt(alpha_bar[t_start]) * mean(x0_hat)."""
    tr = DiffusionTrainer(device=torch.device('cpu'), prediction_type='x0', snr_gamma=None)
    seen = []

    class Probe(torch.nn.Module):
        def forward(self, x_t, cond, t):
            seen.append((int(t[0]), x_t.clone()))
            return torch.zeros_like(x_t)
    x0_hat = torch.full((2, 4, 32, 32), 0.7)
    batch = (torch.zeros_like(x0_hat), torch.zeros(2, 3, 8, 8), x0_hat)
    cfg = dict(name='warm400_ddim', method='ddim', n_steps=1, t_start=400)
    noise_std = float(tr.sqrt_one_minus_alpha_bars[400])
    for mode, expected_mean in (('residual', 0.0), ('standard', float(tr.sqrt_alpha_bars[400]) * 0.7)):
        seen.clear()
        reconstruct(Probe(), tr, mode, batch, cfg, sigma_d=0.05, seed=0)
        t0, x = seen[0]
        assert t0 == 400 and tr.last_nfe == 1
        assert abs(x.mean().item() - expected_mean) < 0.05, mode
        assert abs(x.std().item() - noise_std) < 0.05 * noise_std, mode
    assert float(tr.sqrt_alpha_bars[400]) * 0.7 > 0.2                               # the two modes are 4+ tolerances apart


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


def test_harness_uses_the_shared_paths_and_helpers():
    """Fix item 15: the harness reads the prior, subset and results root from data_loader/warmstart_paths.py (defined
    once) and still exports the names for its importers (scripts/make_baseline_record.py, the tests)."""
    from data_loader import warmstart_paths as wp
    from scripts import eval_warmstart as ew
    assert (ew.DATA_ROOT, ew.RESULTS_ROOT, ew.PRIOR, ew.SUBSET, ew.EVAL_DIR) == \
        (wp.DATA_ROOT, wp.RESULTS_ROOT, wp.PRIOR, wp.SUBSET, wp.EVAL_DIR)
    assert ew.file_sha256 is wp.file_sha256 and ew.git_head is wp.git_head


def _blas_threads():
    from threadpoolctl import threadpool_info
    return max(p['num_threads'] for p in threadpool_info() if p['user_api'] == 'blas')


class BlasProbe(torch.utils.data.Dataset):
    """Each item reports the BLAS thread count of the process that loads it as its x0_hat (x = 0), shaped like the
    (x64, y_d, x0_hat) HWC arrays residual_collate_fn takes. The baseline RMSE of an item is then 50 x that count."""
    def __init__(self, n=8):
        self.img_list = [f'/r/MatFlower60/Train/P022/{i}_1.mat' for i in range(n)]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i):
        z = np.zeros((2, 2, 1), dtype=np.float32)
        return z, z, np.full((2, 2, 1), float(_blas_threads()), dtype=np.float32)


@pytest.fixture
def parent_at_four_blas_threads():
    """The parent runs 4 BLAS threads, so a worker that inherits the parent's setting reports 4 (fix item 5)."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(4, user_api='blas'):
        if _blas_threads() != 4:
            pytest.skip('this BLAS cannot run 4 threads, so inheritance cannot be told apart from the limit')
        yield


def test_eval_loader_workers_use_one_blas_thread(parent_at_four_blas_threads, monkeypatch):
    from data_loader.hfd_residual import limit_worker_threads
    from scripts import eval_warmstart as ew
    loader = ew.make_eval_loader(BlasProbe(), batch_size=2, num_workers=2)
    assert loader.worker_init_fn is limit_worker_threads
    seen = [v for _, _, xh in loader for v in xh.flatten().tolist()]
    assert set(seen) == {1.0}, f'workers reported BLAS thread counts {sorted(set(seen))}'
    monkeypatch.setattr(ew, 'dataset_for', lambda *a, **k: BlasProbe())
    out = ew.baseline_eval('/r', 'prior.npz', 4)                                  # the baseline's own loader
    assert out['per_item']['rmse_pct'] == pytest.approx([50.0] * 8)               # 200 if its workers inherited 4 threads
    assert _blas_threads() == 4                                                   # the limit applies in the workers only


def test_every_harness_loader_goes_through_the_factory():
    """The baseline loader and the per-row checkpoint loader both come from make_eval_loader, so neither can lose the
    worker_init_fn: the module builds exactly one DataLoader, inside the factory."""
    import inspect
    from scripts import eval_warmstart as ew
    src = inspect.getsource(ew)
    assert src.count('DataLoader(') == 1 and 'DataLoader(' in inspect.getsource(ew.make_eval_loader)


def test_file_sha256_hashes_the_bytes(tmp_path):
    import hashlib
    from scripts.eval_warmstart import file_sha256
    p = tmp_path / 'blob.bin'
    data = bytes(range(256)) * 9000                                                 # > one 1 MiB read block
    p.write_bytes(data)
    assert file_sha256(str(p)) == hashlib.sha256(data).hexdigest()


def _meta_args(**kw):
    import argparse
    return argparse.Namespace(**dict(dict(mode='residual', d=4, base_channels=64, prediction_type='v', ckpt='c.pth'), **kw))


def test_check_meta_without_meta_warns_and_trusts_the_args(capsys):
    from scripts.eval_warmstart import check_meta
    assert check_meta({'net.w': torch.zeros(1)}, _meta_args(), 0.05, 'abc') is False   # a bare state dict
    assert check_meta({'model_state_dict': {}}, _meta_args(), 0.05, 'abc') is False
    lines = [l for l in capsys.readouterr().out.splitlines() if l]
    assert len(lines) == 2 and all('no meta' in l and 'CLI args' in l for l in lines)


def test_check_meta_names_every_mismatch():
    from scripts.eval_warmstart import check_meta
    meta = dict(mode='residual', d=4, base_channels=64, prediction_type='v', sigma_d=0.05, prior_sha256='abc')
    assert check_meta({'meta': meta}, _meta_args(), 0.05, 'abc') is True
    assert check_meta({'meta': dict(meta, sigma_d=0.05 + 1e-12)}, _meta_args(), 0.05, 'abc') is True
    for change, word in ((dict(mode='standard'), 'mode'), (dict(base_channels=256), 'base_channels'),
                         (dict(prediction_type='eps'), 'prediction_type'), (dict(sigma_d=0.06), 'sigma_d'),
                         (dict(prior_sha256='def'), 'prior_sha256')):
        with pytest.raises(ValueError, match=word):
            check_meta({'meta': dict(meta, **change)}, _meta_args(), 0.05, 'abc')
    no_sigma = {k: v for k, v in meta.items() if k != 'sigma_d'}
    with pytest.raises(ValueError, match='sigma_d'):
        check_meta({'meta': no_sigma}, _meta_args(), 0.05, 'abc')                     # a missing key is a mismatch


@pytest.mark.dataset
def test_baseline_eval_reproduces_recorded_numbers(data_root):
    from scripts.eval_warmstart import baseline_eval
    rec_path = os.path.join(REPO, 'tests', 'data', 'baseline_phase0.json')
    from data_loader.warmstart_paths import PRIOR as prior                         # the prior training and the harness read
    if not (os.path.exists(rec_path) and os.path.exists(prior)):
        pytest.skip('baseline record or prior missing')
    rec = json.load(open(rec_path))
    sub = json.load(open(os.path.join(REPO, 'tests', 'data', 'val_subset_200.json')))
    files = [os.path.join(data_root, f) for f in sub['files'][:40]]
    # d = 1 is the spec's pin (S9): the only rate whose error is dominated by the solver and s, not by the lift
    for d in (1, 4):
        out = baseline_eval(data_root, prior, d=d, files=files)
        assert abs(out['mean']['rmse_pct'] - rec[f'd{d}']['subset40_rmse_pct']) < 0.01, f'd={d}'


@pytest.mark.dataset
def test_dataset_for_rejects_files_outside_the_validation_split(data_root):
    from scripts.eval_warmstart import dataset_for
    from data_loader.warmstart_paths import PRIOR as prior                         # the prior training and the harness read
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
    from data_loader.warmstart_paths import PRIOR as prior                         # the prior training and the harness read
    if not os.path.exists(prior):
        pytest.skip('prior missing')
    from data_loader.linear_estimate import load_prior
    torch.manual_seed(0)
    sd = ew.build_model('residual', 8, torch.device('cpu')).state_dict()               # wrapper state dict, keys 'net.*'
    p = load_prior(prior)
    meta = dict(mode='residual', d=4, base_channels=8, prediction_type='v',
                sigma_d=float(p['sigma_d'][list(p['d_values']).index(4)]), prior_sha256=ew.file_sha256(prior))
    ckpt = tmp_path / 'tiny.pth'
    torch.save({'model_state_dict': sd, 'meta': meta}, ckpt)                            # matching meta: evaluates
    bad = tmp_path / 'tiny_bad.pth'
    torch.save({'model_state_dict': sd, 'meta': dict(meta, d=2)}, bad)                 # trained at another d: refused
    real_load_subset = ew.load_subset
    pick = list(range(0, 3)) + list(range(24, 27)) + list(range(176, 178))              # 3 x P022, 3 x P023, 2 x P024
    monkeypatch.setattr(ew, 'load_subset', lambda root, which='files': (
        [real_load_subset(root, 'files')[i] for i in pick] if which == 'files' else real_load_subset(root, which)[:4]))
    monkeypatch.setattr(ew, 'subset_class_counts', lambda: {'P022': 3, 'P023': 3, 'P024': 2})
    monkeypatch.setattr(ew, 'sampler_grid', lambda which='full': [dict(name='ddim2', method='ddim', n_steps=2),
                                                                  dict(name='warm50_ddim', method='ddim', n_steps=2, t_start=50)])
    monkeypatch.setattr(ew, 'EVAL_DIR', str(tmp_path))
    made, real_factory = [], ew.make_eval_loader
    monkeypatch.setattr(ew, 'make_eval_loader', lambda ds, *a, **k: made.append(len(ds)) or real_factory(ds, *a, **k))
    args = argparse.Namespace(d=4, ckpt=str(ckpt), mode='residual', base_channels=8, prediction_type='v', seeds=[0, 1],
                              batch_size=4, device='cpu', data_root=data_root, prior=prior, grid='full', run_name='tiny_run')
    with pytest.raises(ValueError, match=r'\bd: checkpoint 2 vs args 4'):
        ew.checkpoint_eval(argparse.Namespace(**dict(vars(args), ckpt=str(bad))))
    assert made == []                                                                   # refused before any loader
    ew.checkpoint_eval(args)
    assert made == [8, 8, 8]                                    # baseline + one per grid row, all via the factory (fix item 5)
    out = json.load(open(tmp_path / 'residual' / 'd4' / 'tiny_run__tiny.json'))       # eval/<mode>/d<d>/<run_name>__<ckpt>.json
    assert out['grid'] == 'full' and out['run_name'] == 'tiny_run'
    prov = out['provenance']
    assert {k: prov[k] for k in ('mode', 'd', 'base_channels', 'prediction_type')} == \
        dict(mode='residual', d=4, base_channels=8, prediction_type='v')
    assert prov['sigma_d'] == meta['sigma_d'] and prov['prior_sha256'] == meta['prior_sha256']
    assert prov['n_timesteps'] == 1000 and prov['prior_s'] == float(p['s'])
    assert prov['prior_path'] == os.path.abspath(prior) and prov['ckpt'] == os.path.abspath(ckpt)
    assert prov['git_head'] is None or len(prov['git_head']) == 40
    assert len(out['yswap_files']) == 8 and 'passes' in out['yswap']
    assert out['yswap']['files'] == out['yswap_files']                                 # fix item 1 (a): batch order
    assert all(len(v) == 8 for t in ('100', '300') for v in out['yswap']['per_cube'][t].values())
    assert len(out['yswap']['baseline_per_cube']) == 8
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
        unseen = np.array([f.split('/')[2] in ('P023', 'P024') for f in row['files']])
        assert unseen.sum() == 5 and row['paired_rmse_diff_unseen'] == pytest.approx((cube - base)[unseen].mean())
        from scripts.paired_stats import cluster_se                                    # fix item 10: CR1 over source images
        assert (row['paired_rmse_se_cluster'], row['n_clusters']) == pytest.approx(cluster_se(cube - base, row['files']))


from scripts.eval_warmstart import build_model, eval_output_path, sampler_grid


def test_sampler_grid_quick_is_the_low_nfe_subset():
    full = {c['name']: c for c in sampler_grid('full')}
    quick = sampler_grid('quick')
    names = [c['name'] for c in quick]
    assert names == ['ddim1', 'ddim2', 'ddim5', 'ddim10', 'ddim20', 'warm50_ddim', 'warm100_ddim', 'warm200_ddim', 'warm400_ddim']
    assert all(c == full[c['name']] for c in quick)
    assert sampler_grid() == sampler_grid('full')
    with pytest.raises(ValueError):
        sampler_grid('fast')


def test_build_model_standard_xhat_matches_wrapper_builder():
    m = build_model('standard_xhat', 8, torch.device('cpu'))
    assert m.use_x0_hat and m.net.input_proj.in_channels == 128 and not m.training
    sd = m.state_dict()
    build_model('standard_xhat', 8, torch.device('cpu')).load_state_dict(sd, strict=True)
    with pytest.raises(RuntimeError):                      # A′ weights do not fit A
        build_model('standard', 8, torch.device('cpu')).load_state_dict(sd, strict=True)


def test_eval_output_path_separates_runs_and_grids():
    a = eval_output_path('standard', 4, 'standard_d4_b64', '/x/runs/standard_d4_b64/checkpoint_epoch_100.pth')
    b = eval_output_path('standard', 4, 'standard_d4_b256', '/x/runs/standard_d4_b256/checkpoint_epoch_100.pth')
    q = eval_output_path('standard', 4, 'standard_d4_b64', '/x/runs/standard_d4_b64/checkpoint_epoch_100.pth', grid='quick')
    assert a != b and a.endswith('eval/standard/d4/standard_d4_b64__checkpoint_epoch_100.json')
    assert q.endswith('eval/standard/d4/standard_d4_b64__checkpoint_epoch_100__quick.json')
    assert eval_output_path('standard_xhat', 4, None, '/x/runs/standard_xhat_d4_b64/checkpoint_epoch_100.pth') \
        .endswith('eval/standard_xhat/d4/standard_xhat_d4_b64__checkpoint_epoch_100.json')


def test_the_module_docstring_names_every_output_suffix():
    """F6: the documented output pattern covers the direct grid's suffix, which eval_output_path writes."""
    from scripts import eval_warmstart as ew
    assert eval_output_path('regress', 4, 'regress_l1_d4_b64', '/x/runs/regress_l1_d4_b64/checkpoint_epoch_100.pth',
                            grid='direct').endswith('eval/regress/d4/regress_l1_d4_b64__checkpoint_epoch_100__direct.json')
    assert '<run_name>__<ckpt>[__quick|__direct].json' in ew.__doc__


# ----------------------------------------------------------------------------- scripts/yswap_one_step.py (fix item 1 (b))
def test_yswap_one_step_defaults_and_refuses_the_results_tree(tmp_path):
    from data_loader.warmstart_paths import RESULTS_ROOT
    from scripts import yswap_one_step as yo
    a = yo.get_parser().parse_args(['--checkpoint', 'c.pth', '--mode', 'residual', '--out', 'o.json'])
    assert (a.t, a.n, a.device) == ([100, 300, 999], 32, 'cpu')
    for out in (os.path.join(RESULTS_ROOT, 'eval', 'residual', 'd4', 'x.json'), RESULTS_ROOT):
        with pytest.raises(SystemExit, match='results tree'):                       # before the checkpoint is even opened
            yo.main(['--checkpoint', str(tmp_path / 'missing.pth'), '--mode', 'residual', '--out', out])
    with pytest.raises(SystemExit):
        yo.get_parser().parse_args(['--checkpoint', 'c.pth', '--mode', 'residual'])     # --out is required


def test_yswap_one_step_paired_contrasts():
    """Per t: the variant means, the baseline, true / swap_full, and paired per-cube mean +- SE (iid and clustered by
    source image) of swap_full - true, true - baseline, swap_full - baseline, swap_sensor_only - true."""
    from scripts.paired_stats import cluster_se
    from scripts.yswap_one_step import summarize_t
    files = ['P023/1_1.mat', 'P023/1_2.mat', 'P023/2_1.mat', 'P024/3_1.mat']
    true, full, sens, zero, base = [1.0, 2.0, 3.0, 4.0], [1.5, 2.1, 3.6, 3.9], [1.1, 2.0, 3.1, 4.2], [5.0] * 4, [1.2, 2.4, 3.1, 4.5]
    res = {'t999': dict(true=2.5, swap_full=2.775, swap_sensor_only=2.6, zero=5.0), 'passes': False, 'files': files,
           'per_cube': {'999': dict(true=true, swap_full=full, swap_sensor_only=sens, zero=zero)}, 'baseline_per_cube': base}
    s = summarize_t(res)['999']
    assert s['means'] == dict(true=2.5, swap_full=2.775, swap_sensor_only=2.6, zero=5.0, baseline=pytest.approx(2.8))
    assert s['ratio_true_over_swap_full'] == pytest.approx(2.5 / 2.775)
    assert s['ratio_true_over_baseline'] == pytest.approx(2.5 / 2.8)
    for name, a, b in (('swap_full_minus_true', full, true), ('true_minus_baseline', true, base),
                       ('swap_full_minus_baseline', full, base), ('swap_sensor_only_minus_true', sens, true)):
        d = np.subtract(a, b)
        c = s['paired'][name]
        assert c['mean'] == pytest.approx(d.mean()) and c['se'] == pytest.approx(d.std(ddof=1) / 2) and c['n'] == 4
        assert (c['se_cluster'], c['n_clusters']) == pytest.approx(cluster_se(d, files)) and c['n_positive'] == int((d > 0).sum())
    assert s['n_true_beats_swap_full'] == 3                                             # cube 4: swap_full 3.9 < true 4.0


@pytest.mark.dataset
def test_yswap_one_step_end_to_end_on_cpu(data_root, tmp_path):
    """The script on a tiny random-weight checkpoint: the batch is the harness's stratified pick, every t is scored,
    and the summary agrees with the per-cube values it stores."""
    from data_loader.linear_estimate import load_prior
    from data_loader.warmstart_paths import PRIOR
    from scripts import eval_warmstart as ew
    from scripts import yswap_one_step as yo
    if not os.path.exists(PRIOR):
        pytest.skip('prior missing')
    torch.manual_seed(0)
    p = load_prior(PRIOR)
    meta = dict(mode='residual', d=4, base_channels=8, prediction_type='v',
                sigma_d=float(p['sigma_d'][list(p['d_values']).index(4)]), prior_sha256=ew.file_sha256(PRIOR))
    ckpt = tmp_path / 'checkpoint_epoch_1.pth'
    torch.save({'model_state_dict': ew.build_model('residual', 8, torch.device('cpu')).state_dict(), 'meta': meta}, ckpt)
    out = tmp_path / 'ys.json'
    with pytest.raises(ValueError, match='base_channels'):                              # the checkpoint's meta is checked
        yo.main(['--checkpoint', str(ckpt), '--mode', 'residual', '--n', '4', '--t', '999', '--out', str(out)])
    yo.main(['--checkpoint', str(ckpt), '--mode', 'residual', '--base_channels', '8', '--n', '4', '--t', '300', '999',
             '--out', str(out)])
    r = json.load(open(out))
    sub = json.load(open(os.path.join(REPO, 'tests', 'data', 'val_subset_200.json')))
    want = sorted(ew.yswap_files(sub['files'], sub['classes'], 4))
    assert r['files'] == r['yswap']['files'] and sorted(r['files']) == want and r['n'] == 5        # 1 + 3 + 1 cubes
    assert r['t_values'] == [300, 999] and set(r['summary']) == {'300', '999'}
    assert r['provenance']['meta_checked'] is True and r['provenance']['device'] == 'cpu'
    for t in ('300', '999'):
        pc, s = r['yswap']['per_cube'][t], r['summary'][t]
        assert s['means']['true'] == r['yswap'][f't{t}']['true']
        d = np.subtract(pc['swap_full'], pc['true'])
        assert s['paired']['swap_full_minus_true']['mean'] == pytest.approx(d.mean())


# ----------------------------------------------------------------------------- final fix wave F1: x_t = 'noise'
CPU = torch.device('cpu')
DDIM1 = dict(name='ddim1', method='ddim', n_steps=1)


def _tiny_b_batch():
    """4 cubes of B's shapes at 16 x 16 (64 bands, 30 sensor channels at d = 4)."""
    torch.manual_seed(0)
    x = torch.rand(4, 64, 16, 16) * 1.6 - 0.8
    return x, torch.rand(4, 30, 4, 4), (x + 0.05 * torch.randn_like(x)).clamp(-1, 1)


def _tiny_b():
    """A tiny random-weight residual (B-architecture) model and a v-prediction CPU trainer, as B is evaluated."""
    torch.manual_seed(0)
    return build_model('residual', 8, CPU), DiffusionTrainer(device=CPU, prediction_type='v', snr_gamma=None)


def test_yswap_noise_x_t_true_is_bs_ddim1_read_out_for_the_same_noise():
    """F1: with x_t='noise' every variant's x_t is the sampler's own x_T (randn after seed 1234), so at t = 999 the
    y-swap's 'true' per-cube RMSE IS reconstruct(mode residual, ddim1) seeded with 1234: B's real one-step read-out,
    which never sees the true residual. The default q_sample x_t = 0.00635 r + 0.99998 eps carries the true residual,
    and a network that reads x_t scores differently on it (the leak)."""
    model, tr = _tiny_b()
    batch, sigma_d = _tiny_b_batch(), 0.05
    assert DDIM1 == next(c for c in sampler_grid('full') if c['name'] == 'ddim1')
    ref = metrics(reconstruct(model, tr, 'residual', batch, DDIM1, sigma_d, seed=1234), batch[0])['rmse_pct'].tolist()
    res = yswap_check(model, tr, 'residual', batch, t_values=(999,), sigma_d=sigma_d, x_t='noise')
    assert res['per_cube']['999']['true'] == ref                                     # exactly, cube by cube
    leaky = yswap_check(model, tr, 'residual', batch, t_values=(999,), sigma_d=sigma_d)
    assert leaky['per_cube']['999']['true'] != ref
    assert set(res) == set(leaky) == {'t999', 'passes', 'files', 'per_cube', 'baseline_per_cube'}   # the same record


def test_yswap_x_t_feeds_every_variant_one_shared_eps_and_the_default_is_q_sample():
    """F1: the default (and x_t='q_sample') is unchanged -- x_t = q_sample(true target, t) with the eps of seed 1234,
    the same for every variant; x_t='noise' feeds every variant that eps itself."""
    seen = []

    class Recorder(torch.nn.Module):
        def forward(self, x_t, cond, t):
            seen.append(x_t.clone())
            return torch.zeros_like(x_t)
    torch.manual_seed(0)
    x, y, x0_hat = torch.rand(4, 4, 8, 8) * 2 - 1, torch.rand(4, 3, 2, 2), torch.rand(4, 4, 8, 8) * 2 - 1
    tr, sigma_d = _cpu_trainer(), 0.1
    target = (x - x0_hat) / sigma_d
    torch.manual_seed(1234)
    eps = torch.randn(x.shape)
    for kw in ({}, {'x_t': 'q_sample'}):
        seen.clear()
        yswap_check(Recorder(), tr, 'residual', (x, y, x0_hat), t_values=(100, 999), sigma_d=sigma_d, **kw)
        assert len(seen) == 8
        for i, tv in enumerate((100, 999)):
            want = tr.sqrt_alpha_bars[tv] * target + tr.sqrt_one_minus_alpha_bars[tv] * eps
            assert all(torch.allclose(s, want, rtol=0, atol=1e-6) for s in seen[4 * i:4 * i + 4]), (kw, tv)
    seen.clear()
    yswap_check(Recorder(), tr, 'residual', (x, y, x0_hat), t_values=(999,), sigma_d=sigma_d, x_t='noise')
    assert len(seen) == 4 and all(torch.equal(s, eps) for s in seen)


def test_yswap_noise_x_t_is_refused_off_the_one_step_t_and_ignored_by_the_regressor():
    """F1: 'noise' is the ddim1 sampler's x_T, defined only at t = 999 (refused at any other t); an unknown x_t is
    refused; mode regress always calls on its zero x_t at REGRESS_T, so x_t does not change its record."""
    from model.residual_wrapper import REGRESS_T
    batch = (torch.zeros(4, 4, 8, 8), torch.zeros(4, 3, 2, 2), torch.zeros(4, 4, 8, 8))
    for tv in ((100,), (300, 999), (100, 300)):
        with pytest.raises(ValueError, match=r"x_t='noise'.*t = 999"):
            yswap_check(ZeroModel(), _cpu_trainer(), 'residual', batch, t_values=tv, sigma_d=0.05, x_t='noise')
    with pytest.raises(ValueError, match='x_t must be one of'):
        yswap_check(ZeroModel(), _cpu_trainer(), 'residual', batch, t_values=(999,), sigma_d=0.05, x_t='zeros')

    class Spy(torch.nn.Module):
        def forward(self, x_t, cond, t):
            assert t.tolist() == [REGRESS_T] * x_t.shape[0] and float(x_t.abs().max()) == 0.0
            return cond['x0_hat'] + cond['sensor'].mean((1, 2, 3), keepdim=True)
    torch.manual_seed(0)
    x = torch.rand(4, 4, 8, 8) * 1.6 - 0.8
    rb = (x, torch.rand(4, 3, 2, 2), 0.9 * x)
    tr = DiffusionTrainer(device=CPU, prediction_type='direct', snr_gamma=None)
    assert yswap_check(Spy(), tr, 'regress', rb, sigma_d=0.1, x_t='noise') == yswap_check(Spy(), tr, 'regress', rb,
                                                                                           sigma_d=0.1)


def _no_work(*args, **kwargs):
    raise AssertionError('work started before the guards')


def _fake_yswap_world(tmp_path, monkeypatch, mode='residual'):
    """scripts/yswap_one_step.py main over a synthetic 4-cube batch (yswap_batch stubbed) and a tiny real model saved
    as a checkpoint (check_meta stubbed True); the real yswap_check, summary and writer run."""
    from types import SimpleNamespace
    from scripts import yswap_one_step as yo
    tmp_path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = build_model(mode, 8, CPU)
    ckpt, prior = tmp_path / 'checkpoint_epoch_100.pth', tmp_path / 'prior.npz'
    torch.save({'model_state_dict': model.state_dict()}, str(ckpt))
    prior.write_bytes(b'stub prior')
    files = [f'MatFlower60/Train/P0{22 + i % 3}/img{i}_1.mat' for i in range(4)]
    batch = _tiny_b_batch()
    ds = SimpleNamespace(sigma_d=0.05, img_list=[str(tmp_path / f) for f in files])
    monkeypatch.setattr(yo, 'yswap_batch', lambda root, prior_path, d, n: (ds, batch))
    monkeypatch.setattr(yo, 'check_meta', lambda *a: True)
    argv = ['--checkpoint', str(ckpt), '--mode', mode, '--base_channels', '8', '--prior', str(prior),
            '--data_root', str(tmp_path), '--n', '4', '--t', '999']
    if mode == 'regress':
        argv += ['--prediction_type', 'direct']
    return yo, model, batch, files, argv


def test_yswap_one_step_records_x_t_in_its_provenance(tmp_path, monkeypatch):
    """F1: --x_t (default q_sample) reaches yswap_check and the record's provenance; with --x_t noise at t = 999 the
    record's 'true' cubes are B's ddim1 read-out; mode regress records its zero x_t whatever --x_t says."""
    yo, model, batch, files, argv = _fake_yswap_world(tmp_path, monkeypatch)
    assert yo.get_parser().parse_args(['--checkpoint', 'c', '--mode', 'residual', '--out', 'o']).x_t == 'q_sample'
    with pytest.raises(SystemExit):
        yo.get_parser().parse_args(['--checkpoint', 'c', '--mode', 'residual', '--out', 'o', '--x_t', 'zero'])
    tr = DiffusionTrainer(device=CPU, prediction_type='v', snr_gamma=None)
    ref = metrics(reconstruct(model, tr, 'residual', batch, DDIM1, 0.05, seed=1234), batch[0])['rmse_pct'].tolist()
    noise = yo.main(argv + ['--x_t', 'noise', '--out', str(tmp_path / 'n.json')])
    assert noise['provenance']['x_t'] == 'noise' and noise['provenance']['noise_seed'] == 1234 and noise['files'] == files
    assert json.load(open(tmp_path / 'n.json'))['yswap']['per_cube']['999']['true'] == ref
    leaky = yo.main(argv + ['--out', str(tmp_path / 'q.json')])
    assert leaky['provenance']['x_t'] == 'q_sample' and leaky['yswap']['per_cube']['999']['true'] != ref
    yo, _, _, _, argv_r = _fake_yswap_world(tmp_path / 'r', monkeypatch, mode='regress')
    r = yo.main(argv_r + ['--x_t', 'noise', '--out', str(tmp_path / 'r.json')])
    assert r['provenance']['x_t'] == 'zero' and set(r['yswap']['per_cube']) == {'999'}


def test_yswap_one_step_refuses_noise_off_999_before_any_work(tmp_path, monkeypatch):
    yo, _, _, _, argv = _fake_yswap_world(tmp_path, monkeypatch)
    monkeypatch.setattr(yo.torch, 'load', _no_work)
    monkeypatch.setattr(yo, 'yswap_batch', _no_work)
    with pytest.raises(ValueError, match=r"x_t='noise'.*t = 999"):
        yo.main(argv + ['--t', '300', '999', '--x_t', 'noise', '--out', str(tmp_path / 'o.json')])
    assert not (tmp_path / 'o.json').exists()


# ----------------------------------------------------------------------------- final fix wave F4 / F5: yswap_one_step
def test_yswap_one_step_never_clobbers_an_output(tmp_path, monkeypatch):
    """F4: an existing --out is refused before any work (its bytes untouched), and the record is written to a
    temporary file first, so a crash while writing leaves no --out behind."""
    yo, _, _, _, argv = _fake_yswap_world(tmp_path, monkeypatch)
    out = tmp_path / 'diag' / 'ys.json'
    out.parent.mkdir()
    out.write_text('{"earlier": "record"}')
    real_load = yo.torch.load
    monkeypatch.setattr(yo.torch, 'load', _no_work)
    with pytest.raises(SystemExit, match=re.escape(str(out)) + ' exists'):
        yo.main(argv + ['--out', str(out)])
    assert out.read_text() == '{"earlier": "record"}'
    monkeypatch.setattr(yo.torch, 'load', real_load)
    fresh = tmp_path / 'diag' / 'fresh.json'

    def crash(obj, f, **kw):
        f.write('{"provenance": ')
        raise OSError('disk full')
    monkeypatch.setattr(yo.json, 'dump', crash)
    with pytest.raises(OSError, match='disk full'):
        yo.main(argv + ['--out', str(fresh)])
    assert not fresh.exists()


def test_yswap_one_step_records_the_git_head_at_launch(tmp_path, monkeypatch):
    """F5: the record's git_head is the HEAD when the command started, not a commit that landed during the run."""
    yo, _, _, _, argv = _fake_yswap_world(tmp_path, monkeypatch)
    head = ['a' * 40]
    monkeypatch.setattr(yo, 'git_head', lambda: head[0])
    real = yo.yswap_check

    def commit_lands(*a, **k):
        head[0] = 'b' * 40
        return real(*a, **k)
    monkeypatch.setattr(yo, 'yswap_check', commit_lands)
    rec = yo.main(argv + ['--out', str(tmp_path / 'o.json')])
    assert rec['provenance']['git_head'] == 'a' * 40 and head[0] == 'b' * 40
