"""Step-0 diagnostics (Phase 1a-prime Task 3): seeding, sums, the centre/spread estimates, lambda, shard handling,
the stored-evaluation consistency check and the merge statistics."""
import json
import os
import re

import numpy as np
import pytest
import torch

from scripts import diagnose_b as db
from scripts.eval_warmstart import sampler_grid


def test_accumulate_uses_the_harness_seeding_and_sums_per_cube():
    x = torch.rand(5, 3, 4, 4) * 2 - 1
    batches = [(x[:2], None, None), (x[2:4], None, None), (x[4:], None, None)]
    seen = []

    def recon(batch, seed):
        seen.append(seed)
        g = torch.Generator().manual_seed(seed)
        return (batch[0] + 0.1 * torch.randn(batch[0].shape, generator=g)).clamp(-1, 1)
    sum_x, sum_x2, err = db.accumulate(batches, recon, [0, 3], 5)
    assert seen == [0, 1, 2, 3000, 3001, 3002]                   # draw k, batch bi -> seed k * 1000 + bi
    xs = [torch.cat([recon(b, k * 1000 + bi) for bi, b in enumerate(batches)]).double().numpy() for k in (0, 3)]
    assert np.allclose(sum_x, xs[0] + xs[1])
    assert np.allclose(sum_x2, sum((v ** 2).reshape(5, -1).mean(1) for v in xs))
    assert np.allclose(err[1], db.metrics(torch.from_numpy(xs[1]), x.double())['rmse_pct'].numpy())
    with pytest.raises(ValueError, match='expected 6'):
        db.accumulate(batches, recon, [0], 6)


def test_row_stats_recovers_centre_and_spread():
    rng = np.random.default_rng(0)
    N, K, shape = 6, 64, (4, 16, 16)
    x = rng.uniform(-0.5, 0.5, (N,) + shape)
    draws = (x + 0.02)[None] + 0.03 * rng.standard_normal((K, N) + shape)    # centre 0.02 off the truth, spread 0.03
    sum_x = draws.sum(0)
    sum_x2 = (draws ** 2).reshape(K, N, -1).mean(2).sum(0)
    err = 50 * np.sqrt(((draws - x[None]) ** 2).reshape(K, N, -1).mean(2))
    s = db.row_stats(sum_x, sum_x2, err, x)
    assert s['K'] == K
    assert np.allclose(s['c2'], 2500 * 0.02 ** 2, rtol=0.05)
    assert np.allclose(s['spread'], 2500 * 0.03 ** 2, rtol=0.05)
    assert np.allclose(s['o'], 2500 * (0.02 ** 2 + 0.03 ** 2), rtol=0.05)


def test_identical_draws_have_zero_spread_and_lambda_endpoints():
    x, d = np.zeros((3, 2, 4, 4)), np.full((3, 2, 4, 4), 0.1)
    s = db.row_stats(4 * d, np.full(3, 4 * 0.01), np.full((4, 3), 50 * 0.1), x)
    assert np.allclose(s['spread'], 0) and np.allclose(s['c2'], 2500 * 0.01) and np.allclose(s['kmean_rmse'], 5.0)
    b, m = np.array([2.0, 3.0]), np.array([1.0, 2.0])
    assert db.centre_lambda(b, b ** 2, m ** 2) == pytest.approx(0.0)                # centred on r = 0
    assert db.centre_lambda(b, m ** 2, m ** 2) == pytest.approx(1.0)                # centred on the one-step centre
    with pytest.raises(ValueError, match='2 draws'):
        db.row_stats(d, np.full(3, 0.01), np.full((1, 3), 5.0), x)


def test_row_stats_spread_is_unbiased_with_two_draws():
    x = np.zeros((1, 1, 1, 2))
    d0, d1 = np.full((1, 1, 1, 2), 0.1), np.full((1, 1, 1, 2), 0.3)
    s = db.row_stats(d0 + d1, (d0 ** 2 + d1 ** 2).reshape(1, -1).mean(1), 50 * np.array([[0.1], [0.3]]), x)
    assert s['spread'][0] == pytest.approx(2500 * 0.02)                   # var({0.1, 0.3}, ddof=1) = 0.02
    assert s['c2'][0] == pytest.approx(2500 * 0.2 ** 2 - 2500 * 0.02 / 2)


def test_jackknife_se_over_source_images():
    files = [f'MatFlower60/Train/P022/{img}_{p}.mat' for img in ('a', 'b', 'c') for p in (1, 2)]
    v = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    # leave-one-image-out means 2.5, 2.0, 1.5 -> jackknife SE = sqrt(2/3 * 0.5)
    assert db.jackknife_se(lambda mask: v[mask].mean(), files) == pytest.approx(np.sqrt(2 / 3 * 0.5))
    assert db.jackknife_se(lambda mask: v[mask].mean(), files[:2]) is None          # one image: no SE


def test_merge_requires_disjoint_draws_and_equal_cubes():
    def shard(draws, files=('a', 'b')):
        K = len(draws)
        return {'ddim10': dict(files=list(files), draws=list(draws), sum_x=np.ones((2, 1, 2, 2)) * K,
                               sum_x2=np.ones(2) * K, err=np.ones((K, 2)))}
    m = db.merge_shards([shard([0, 1]), shard([2])])
    assert m['ddim10']['draws'] == [0, 1, 2] and np.allclose(m['ddim10']['sum_x'], 3) and m['ddim10']['err'].shape == (3, 2)
    with pytest.raises(ValueError, match='two shards'):
        db.merge_shards([shard([0, 1]), shard([1])])
    with pytest.raises(ValueError, match='different cubes'):
        db.merge_shards([shard([0]), shard([1], files=('a', 'c'))])


def test_shard_roundtrip(tmp_path):
    rows = {'t999': dict(files=['MatFlower60/Train/P022/a_1.mat'], draws=[0, 1], sum_x=np.zeros((1, 2, 2, 2)),
                         sum_x2=np.zeros(1), err=np.zeros((2, 1)))}
    p = str(tmp_path / 's.npz')
    db.save_shard(p, rows, dict(device='cpu'))
    back, meta = db.load_shard(p)
    assert back['t999']['files'] == rows['t999']['files'] and back['t999']['draws'] == [0, 1]
    assert np.allclose(back['t999']['err'], 0) and meta == dict(device='cpu')


def test_outputs_refused_inside_the_results_tree_outside_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'RESULTS_ROOT', str(tmp_path / 'root'))
    monkeypatch.setattr(db, 'DIAG_DIR', str(tmp_path / 'root' / 'diagnostics'))
    with pytest.raises(SystemExit, match='results tree'):
        db.check_out(str(tmp_path / 'root' / 'eval' / 'x.json'))
    db.check_out(str(tmp_path / 'root' / 'diagnostics' / 'x.json'))
    db.check_out(str(tmp_path / 'elsewhere.json'))


def test_consistency_reads_the_stored_seed_row_by_file():
    ev_b = {'rows': [dict(name='ddim1', files=['a', 'b'], seeds=[0, 1], per_item={'rmse_pct': [[1.0, 2.0], [3.0, 4.0]]})]}
    d = dict(files=['b', 'a'], draws=[1, 0, 5], err=np.array([[4.0, 3.0], [2.0, 1.5], [9.0, 9.0]]))
    assert db.consistency(d, ev_b, 't999') == {'1': 0.0, '0': 0.5}


def test_merge_end_to_end_on_synthetic_shards(tmp_path, monkeypatch):
    files = [f'MatFlower60/Train/P022/img{i}_1.mat' for i in range(4)]
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.5, 0.5, (4, 2, 2, 2))
    b = np.array([3.0, 3.2, 2.8, 3.1])

    class FakeDS:
        img_list = [os.path.join(str(tmp_path), f) for f in files]

        def __getitem__(self, i):
            return x[i].transpose(1, 2, 0), None, None

        def __len__(self):
            return 4
    monkeypatch.setattr(db, 'dataset_for', lambda root, prior, d, fl: FakeDS())

    def shard(off):
        xs = np.stack([x + off + 0.01 * rng.standard_normal(x.shape) for _ in range(2)])
        return dict(files=files, draws=[0, 1], sum_x=xs.sum(0), sum_x2=(xs ** 2).reshape(2, 4, -1).mean(2).sum(0),
                    err=50 * np.sqrt(((xs - x[None]) ** 2).reshape(2, 4, -1).mean(2)))
    rows = {'ddim10': shard(0.04), 't999': shard(0.02)}        # t999 last: merge must still score it first
    db.save_shard(str(tmp_path / 's.npz'), rows, dict(device='cuda:0'))
    assert list(db.load_shard(str(tmp_path / 's.npz'))[0]) == ['ddim10', 't999']
    stored = [dict(name=n, files=files, seeds=[0, 1], per_item={'rmse_pct': rows[r]['err'].tolist()})
              for n, r in (('ddim1', 't999'), ('ddim10', 'ddim10'))]
    (tmp_path / 'b.json').write_text(json.dumps(dict(baseline_files=files, baseline_per_item={'rmse_pct': list(b)},
                                                     rows=stored)))
    db.main(['merge', '--inputs', str(tmp_path / 's.npz'), '--b_eval', str(tmp_path / 'b.json'), '--data_root',
             str(tmp_path), '--prior', 'unused', '--out', str(tmp_path / 'o.json')])
    out = json.load(open(tmp_path / 'o.json'))['rows']
    assert 'lambda_vs_one_step_centre' in out['ddim10'] and 'lambda_vs_one_step_centre' not in out['t999']
    assert out['t999']['consistency_vs_stored'] == {'0': 0.0, '1': 0.0}
    c1 = db.row_stats(rows['t999']['sum_x'], rows['t999']['sum_x2'], rows['t999']['err'], x)['c2']   # H1 centre energy
    s10 = db.row_stats(rows['ddim10']['sum_x'], rows['ddim10']['sum_x2'], rows['ddim10']['err'], x)
    r10 = out['ddim10']
    assert r10['pred_h1'] == pytest.approx(float(np.mean(np.sqrt(np.clip(c1 + (s10['o'] - c1) / 2, 0, None)))))
    assert r10['lambda_vs_one_step_centre']['value'] == pytest.approx(db.centre_lambda(b, s10['c2'], c1))
    assert r10['calibration']['ratio_of_means'] == pytest.approx(float(s10['spread'].mean() / c1.mean()))
    assert r10['calibration']['per_cube_slope_on_b2'] == pytest.approx(float(np.polyfit(b ** 2, s10['spread'], 1)[0]))
    # fix round 1: the remaining merge statistics (4 cubes from 4 source images, so every jackknife SE exists)
    K = s10['K']
    assert K == 2 and r10['K'] == 2
    assert r10['pred_h0'] == pytest.approx(float(np.mean(np.sqrt(np.clip(b ** 2 + (s10['o'] - b ** 2) / K, 0, None)))))
    diff = s10['kmean_rmse'] - s10['single_rmse']                # K-draw mean minus single draw, per cube
    assert r10['kmean_minus_single']['mean'] == pytest.approx(float(diff.mean()))
    assert r10['kmean_minus_single']['se_cluster'] == pytest.approx(db.cluster_se(diff, files)[0])
    assert r10['kmean_minus_single']['n_clusters'] == 4
    lam_se = db.jackknife_se(lambda m: db.centre_lambda(b[m], s10['c2'][m], c1[m]), files)
    assert lam_se is not None and r10['lambda_vs_one_step_centre']['se_cluster'] == pytest.approx(lam_se)
    cal = r10['calibration']
    assert cal['self_ratio'] == pytest.approx(float(s10['spread'].mean() / s10['c2'].mean()))
    assert cal['reference_slope_c2_one_step_on_b2'] == pytest.approx(float(np.polyfit(b ** 2, c1, 1)[0]))
    assert cal['per_cube_corr'] == pytest.approx(float(np.corrcoef(b ** 2, s10['spread'])[0, 1]))
    slope_se = db.jackknife_se(lambda m: float(np.polyfit(b[m] ** 2, s10['spread'][m], 1)[0]), files)
    assert slope_se is not None and cal['per_cube_slope_se'] == pytest.approx(slope_se)
    ms = rows['t999']['err'].mean(0)                             # stored ddim1 RMSE per cube, mean of its 2 seeds
    assert r10['lambda_vs_ddim1_single_draw']['value'] == pytest.approx(db.centre_lambda(b, s10['c2'], ms ** 2))
    assert out['t999']['mean_c2_one_step'] == pytest.approx(float(c1.mean()))
    assert out['t999']['per_cube']['files'] == files
    assert out['t999']['per_cube']['kmean_rmse'] == pytest.approx(
        db.row_stats(rows['t999']['sum_x'], rows['t999']['sum_x2'], rows['t999']['err'], x)['kmean_rmse'].tolist())


def test_merge_needs_the_t999_row(tmp_path):
    rows = {'ddim10': dict(files=['MatFlower60/Train/P022/a_1.mat'], draws=[0, 1], sum_x=np.zeros((1, 1, 2, 2)),
                           sum_x2=np.zeros(1), err=np.zeros((2, 1)))}
    db.save_shard(str(tmp_path / 's.npz'), rows, dict(device='cuda:0'))
    with pytest.raises(ValueError, match='t999'):
        db.main(['merge', '--inputs', str(tmp_path / 's.npz'), '--b_eval', str(tmp_path / 'missing.json'),
                 '--out', str(tmp_path / 'o.json')])


# ----------------------------------------------------------------------------- fix round 1: sample, guards
def _no_work(*args, **kwargs):
    raise AssertionError('work started before the output guards')


def _tiny_row(draws):
    return dict(files=['MatFlower60/Train/P022/a_1.mat'], draws=list(draws), sum_x=np.zeros((1, 1, 2, 2)),
                sum_x2=np.zeros(1), err=np.zeros((len(draws), 1)))


def _fake_sample_world(tmp_path, monkeypatch):
    """`sample` over fakes: a 10-cube subset (batches of 8 + 2) and a 3-cube DDPM subset of it, a checkpoint and a
    prior file, recorders on every harness call. The real DiffusionTrainer (CPU), make_eval_loader (in-process),
    accumulate, metrics and save_shard run."""
    data_root = tmp_path / 'data'
    subset = {'files': [str(data_root / f'MatFlower60/Train/P022/img{i}_1.mat') for i in range(10)]}
    subset['ddpm1000_subset'] = subset['files'][2:5]
    rng = np.random.default_rng(1)
    cube = {f: rng.uniform(-0.5, 0.5, (4, 4, 3)).astype(np.float32) for f in subset['files']}     # HWC, as the loader
    rec = dict(events=[], subsets=[], datasets=[], loader_bs=[], calls=[])

    class FakeDS:
        sigma_d = 0.05

        def __init__(self, files):
            self.img_list = list(files)

        def __getitem__(self, i):
            xi = cube[self.img_list[i]]
            return xi, np.zeros((2, 2, 1), np.float32), np.zeros_like(xi)

        def __len__(self):
            return len(self.img_list)

    class FakeModel:
        def load_state_dict(self, sd, strict):
            rec['events'].append('load_state_dict')
            rec['state'] = (dict(sd), strict)

    def load_subset(root, which='files'):
        rec['subsets'].append(which)
        return list(subset[which])

    def dataset_for(root, prior, d, files):
        rec['datasets'].append(dict(root=root, prior=prior, d=d))
        return FakeDS(files)

    real_loader = db.make_eval_loader

    def make_eval_loader(ds, batch_size):
        rec['loader_bs'].append(batch_size)
        return real_loader(ds, batch_size, num_workers=0)

    def check_meta(ck, args, sigma_d, prior_sha256):
        rec['events'].append('check_meta')
        rec['meta_call'] = dict(meta=ck.get('meta'), args=vars(args), sigma_d=sigma_d, prior_sha256=prior_sha256)
        return True

    def build_model(mode, base_channels, device):
        rec['events'].append('build_model')
        rec['build'] = (mode, base_channels, str(device))
        rec['model'] = FakeModel()
        return rec['model']

    def reconstruct(model, trainer, mode, batch, cfg, sigma_d, seed):
        rec['calls'].append(dict(model=model, prediction_type=trainer.prediction_type, device=str(trainer.device),
                                 mode=mode, cfg=dict(cfg), sigma_d=sigma_d, seed=seed, n=len(batch[0])))
        return batch[0] + 0.01 * (seed // 1000)                  # draw k shifts every element by 0.01 k
    for name, fn in dict(load_subset=load_subset, dataset_for=dataset_for, make_eval_loader=make_eval_loader,
                         check_meta=check_meta, build_model=build_model, reconstruct=reconstruct).items():
        monkeypatch.setattr(db, name, fn)
    ckpt, prior = tmp_path / 'b.pth', tmp_path / 'prior.npz'
    torch.save({'model_state_dict': {'w': torch.ones(1)}, 'meta': {'tag': 'fake'}}, str(ckpt))
    prior.write_bytes(b'not a real prior')
    return rec, subset, cube, data_root, ckpt, prior


def test_sample_runs_the_harness_rows_seeds_subsets_and_meta_check(tmp_path, monkeypatch):
    rec, subset, cube, data_root, ckpt, prior = _fake_sample_world(tmp_path, monkeypatch)
    out = tmp_path / 'diag' / 's.npz'
    db.main(['sample', '--rows', 'ddim10', 't999', 'ddpm1000', '--draws', '0', '2', '--device', 'cpu',
             '--checkpoint', str(ckpt), '--prior', str(prior), '--data_root', str(data_root), '--out', str(out)])
    # subsets, datasets, loader batch size (default 8, the stored evaluation's)
    assert rec['subsets'] == ['files', 'files', 'ddpm1000_subset']
    assert rec['datasets'] == [dict(root=str(data_root), prior=str(prior), d=4)] * 3
    assert rec['loader_bs'] == [8, 8, 8]
    # the checkpoint's meta is checked once, against mode residual / v-prediction, before the model is built
    assert rec['events'] == ['check_meta', 'build_model', 'load_state_dict']
    mc = rec['meta_call']
    assert mc['meta'] == {'tag': 'fake'} and mc['sigma_d'] == 0.05 and mc['prior_sha256'] == db.file_sha256(str(prior))
    assert mc['args'] == dict(ckpt=str(ckpt), mode='residual', d=4, base_channels=64, prediction_type='v')
    assert rec['build'] == ('residual', 64, 'cpu')
    sd, strict = rec['state']
    assert list(sd) == ['w'] and torch.equal(sd['w'], torch.ones(1)) and strict is True
    # per row: the harness's sampler row, mode residual, a v-prediction trainer, seed k * 1000 + batch index
    cfgs = {'ddim10': dict(name='ddim10', method='ddim', n_steps=10),
            't999': dict(name='ddim1', method='ddim', n_steps=1),
            'ddpm1000': dict(name='ddpm1000', method='ddpm', n_steps=1000)}
    grid = {c['name']: c for c in sampler_grid('full')}
    seeds = {'ddim10': [0, 1, 2000, 2001], 't999': [0, 1, 2000, 2001], 'ddpm1000': [0, 2000]}
    i = 0
    for row in ('ddim10', 't999', 'ddpm1000'):
        assert cfgs[row] == grid[cfgs[row]['name']]
        calls = rec['calls'][i:i + len(seeds[row])]
        i += len(seeds[row])
        assert [c['seed'] for c in calls] == seeds[row], row
        for c in calls:
            assert c['cfg'] == cfgs[row] and c['mode'] == 'residual' and c['sigma_d'] == 0.05, row
            assert c['prediction_type'] == 'v' and c['device'] == 'cpu' and c['model'] is rec['model'], row
    assert i == len(rec['calls'])
    assert [c['n'] for c in rec['calls'][:2]] == [8, 2]
    # the shard: data_root-relative files in dataset order, the draws as given, the sums of the reconstructions
    rows, meta = db.load_shard(str(out))
    assert list(rows) == ['ddim10', 't999', 'ddpm1000']
    for row, d in rows.items():
        want = subset['ddpm1000_subset' if row == 'ddpm1000' else 'files']
        assert d['files'] == [os.path.relpath(f, str(data_root)) for f in want]
        assert d['files'][0] == f"MatFlower60/Train/P022/img{2 if row == 'ddpm1000' else 0}_1.mat"
        assert d['draws'] == [0, 2]
        x = np.stack([cube[f] for f in want]).transpose(0, 3, 1, 2).astype(np.float64)
        n = len(want)
        assert np.allclose(d['sum_x'], 2 * x + 0.02, atol=1e-6)
        assert np.allclose(d['sum_x2'], (x ** 2).reshape(n, -1).mean(1) + ((x + 0.02) ** 2).reshape(n, -1).mean(1),
                           atol=1e-6)
        assert np.allclose(d['err'], [[0.0] * n, [1.0] * n], atol=1e-4)      # RMSE % of a 0.01 k shift = 0.5 k
    assert meta['device'] == 'cpu' and meta['batch_size'] == 8 and meta['checkpoint'] == str(ckpt)
    assert meta['checkpoint_sha256'] == db.file_sha256(str(ckpt)) and meta['prior_sha256'] == db.file_sha256(str(prior))


def test_sample_and_merge_refuse_results_tree_outputs_before_any_work(tmp_path, monkeypatch):
    root = tmp_path / 'root'
    monkeypatch.setattr(db, 'RESULTS_ROOT', str(root))
    monkeypatch.setattr(db, 'DIAG_DIR', str(root / 'diagnostics'))
    for name in ('load_shard', 'DiffusionTrainer', 'dataset_for', 'load_subset'):
        monkeypatch.setattr(db, name, _no_work)
    with pytest.raises(SystemExit, match='results tree'):
        db.main(['merge', '--inputs', str(tmp_path / 's.npz'), '--b_eval', str(tmp_path / 'missing.json'),
                 '--out', str(root / 'eval' / 'x.json')])
    with pytest.raises(SystemExit, match='results tree'):
        db.main(['sample', '--rows', 't999', '--draws', '0', '1', '--device', 'cpu', '--checkpoint',
                 str(tmp_path / 'none.pth'), '--out', str(root / 'runs' / 'x.npz')])
    assert not root.exists()


def test_duplicate_draws_are_refused(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match=re.escape('ddim10: draws [0] repeat within a shard')):
        db.merge_shards([{'ddim10': _tiny_row([0, 1, 0])}])
    with pytest.raises(ValueError, match=re.escape('t999: draws [3] repeat within a shard')):
        db.merge_shards([{'t999': _tiny_row([0, 1])}, {'t999': _tiny_row([3, 3])}])
    monkeypatch.setattr(db, 'DiffusionTrainer', _no_work)
    with pytest.raises(ValueError, match='repeat a draw'):
        db.main(['sample', '--rows', 't999', '--draws', '0', '1', '0', '--device', 'cpu', '--out',
                 str(tmp_path / 's.npz')])
    assert not (tmp_path / 's.npz').exists()


def test_merge_refuses_shards_of_another_checkpoint_prior_or_batch_size(tmp_path):
    def write(name, draws, meta):
        p = str(tmp_path / name)
        db.save_shard(p, {'t999': _tiny_row(draws)}, meta)
        return p

    def merge(*inputs):
        db.main(['merge', '--inputs', *inputs, '--b_eval', str(tmp_path / 'missing.json'), '--out',
                 str(tmp_path / 'o.json')])
    base = dict(device='cuda:0', checkpoint_sha256='c0', prior_sha256='p0', batch_size=8)
    a = write('a.npz', [0, 1], base)
    for field, value in (('checkpoint_sha256', 'c1'), ('prior_sha256', 'p1'), ('batch_size', 16)):
        with pytest.raises(ValueError, match=field):
            merge(a, write(f'b_{field}.npz', [2, 3], dict(base, **{field: value})))
    with pytest.raises(ValueError, match='batch_size'):                 # batch 8 is required, not only agreement
        merge(write('c.npz', [0, 1], dict(base, batch_size=16)))
    # a shard that records none of the fields is compared on none: merge gets past the checks to the missing evaluation
    with pytest.raises(FileNotFoundError, match='missing.json'):
        merge(a, write('d.npz', [4, 5], dict(device='cuda:0')))


def test_outputs_are_never_clobbered(tmp_path, monkeypatch):
    p = str(tmp_path / 's.npz')
    db.save_shard(p, {'t999': _tiny_row([0, 1])}, dict(device='cuda:0'))
    before = open(p, 'rb').read()
    monkeypatch.setattr(db, 'DiffusionTrainer', _no_work)
    with pytest.raises(SystemExit, match=re.escape(p) + ' exists'):
        db.main(['sample', '--rows', 't999', '--draws', '2', '3', '--device', 'cpu', '--out', p])
    os.symlink(str(tmp_path), str(tmp_path / 'link'))
    for out in (p, str(tmp_path) + '/./s.npz', str(tmp_path / 'link' / 's.npz')):
        with pytest.raises(SystemExit, match='one of the --inputs'):
            db.main(['merge', '--inputs', p, '--b_eval', str(tmp_path / 'missing.json'), '--out', out])
    assert open(p, 'rb').read() == before
