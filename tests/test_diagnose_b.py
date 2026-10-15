"""Step-0 diagnostics (Phase 1a-prime Task 3): seeding, sums, the centre/spread estimates, lambda, shard handling,
the stored-evaluation consistency check and the merge statistics."""
import json
import os

import numpy as np
import pytest
import torch

from scripts import diagnose_b as db


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
    rows = {'t999': shard(0.02), 'ddim10': shard(0.04)}
    db.save_shard(str(tmp_path / 's.npz'), rows, dict(device='cuda:0'))
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


def test_merge_needs_the_t999_row(tmp_path):
    rows = {'ddim10': dict(files=['MatFlower60/Train/P022/a_1.mat'], draws=[0, 1], sum_x=np.zeros((1, 1, 2, 2)),
                           sum_x2=np.zeros(1), err=np.zeros((2, 1)))}
    db.save_shard(str(tmp_path / 's.npz'), rows, dict(device='cuda:0'))
    with pytest.raises(ValueError, match='t999'):
        db.main(['merge', '--inputs', str(tmp_path / 's.npz'), '--b_eval', str(tmp_path / 'missing.json'),
                 '--out', str(tmp_path / 'o.json')])
