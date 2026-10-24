"""Phase 2a Task 9: the six pre-registered clauses, each flipped by its own input; the centre check; no overwrite."""
import json
import os

import numpy as np
import pytest

from scripts import uq_verdict as V

N = 40
FILES = [f'MatFlower60/Train/P02{2 + (i // 4) % 3}/{i // 4}_{i % 4}.mat' for i in range(N)]    # 10 images x 4 patches


def record(tmp_path=None):
    rng = np.random.default_rng(0)
    base = 4.5 + rng.normal(0, 0.3, N)
    reg = base - 0.35 + rng.normal(0, 0.05, N)
    pc = dict(
        baseline=dict(rmse_pct=base.tolist(), hf=(-0.9 + rng.normal(0, 0.05, N)).tolist()),
        regressor=dict(rmse_pct=reg.tolist(), hf=(-0.8 + rng.normal(0, 0.05, N)).tolist()),
        diffusion=dict(crps=(1.0 + rng.normal(0, 0.05, N)).tolist(), cov90=(0.90 + rng.normal(0, 0.01, N)).tolist(),
                       cov50=(0.50 + rng.normal(0, 0.01, N)).tolist(), single_hf=(-0.2 + rng.normal(0, 0.05, N)).tolist(),
                       single_rmse_pct=(reg + 0.05 + rng.normal(0, 0.05, N)).tolist(), mean_rmse_pct=(reg + rng.normal(0, 0.01, N)).tolist(),
                       ause=(0.05 + rng.normal(0, 0.005, N)).tolist(), spread_ratio=(1.0 + rng.normal(0, 0.05, N)).tolist(),
                       cov90_gauss=(0.9 + rng.normal(0, 0.01, N)).tolist()),
        varhead=dict(crps=(1.3 + rng.normal(0, 0.05, N)).tolist(), ause=(0.08 + rng.normal(0, 0.005, N)).tolist(),
                     cov90=(0.8 + rng.normal(0, 0.01, N)).tolist(), spread_ratio=(0.7 + rng.normal(0, 0.05, N)).tolist()))
    ys = dict(K=32, files=FILES[:8], true=(4.0 + rng.normal(0, 0.1, 8)).tolist(), swap_full=(4.8 + rng.normal(0, 0.1, 8)).tolist())
    rec = dict(name='t', d=8, K=32, sampler='ddim10', n=N, files=FILES, per_cube=pc, yswap=ys,
               provenance=dict(centre=dict(run='regress_l1_sel_d8_b64', epoch=11)))
    return rec


def test_everything_passes_on_the_reference_record():
    c = V.clauses(record())
    assert all(c[k]['passed'] for k in ('C1', 'C2', 'P1', 'P2', 'S1', 'S2'))
    assert c['calibration_pass'] and c['perceptual_pass'] and c['wins']
    assert c['C1']['clusters'] == 10 and c['C1']['se_cluster'] > 0 and 'C3' in c['reported']


def test_each_clause_fails_on_its_own_input():
    r = record(); r['per_cube']['varhead']['crps'] = (np.array(r['per_cube']['diffusion']['crps']) - 0.01).tolist()
    assert not V.clauses(r)['C1']['passed'] and V.clauses(r)['C2']['passed']
    r = record(); r['per_cube']['diffusion']['cov90'] = [0.8] * N
    assert not V.clauses(r)['C2']['passed'] and not V.clauses(r)['calibration_pass']
    r = record(); r['per_cube']['diffusion']['single_hf'] = r['per_cube']['regressor']['hf']
    assert not V.clauses(r)['P1']['passed'] and not V.clauses(r)['perceptual_pass']
    r = record(); r['per_cube']['diffusion']['single_rmse_pct'] = (np.array(r['per_cube']['baseline']['rmse_pct']) + 0.1).tolist()
    assert not V.clauses(r)['P2']['passed']
    r = record(); r['per_cube']['diffusion']['mean_rmse_pct'] = (np.array(r['per_cube']['regressor']['rmse_pct']) + 0.3).tolist()
    assert not V.clauses(r)['S1']['passed'] and not V.clauses(r)['wins']
    r = record(); r['yswap']['swap_full'] = r['yswap']['true']
    assert not V.clauses(r)['S2']['passed']


def test_clauses_need_every_method():
    r = record(); del r['per_cube']['varhead']
    with pytest.raises(ValueError, match='varhead'):
        V.clauses(r)


def test_centre_check_and_paths():
    V.check_centre(record())
    r = record(); r['provenance']['centre']['epoch'] = 100
    with pytest.raises(ValueError, match='pre-registered'):
        V.check_centre(r)
    assert V.verdict_path(record(), False, False).endswith('/uq/d8/verdict__t.json')
    assert V.verdict_path(record(), True, True).endswith('/uq/d8/verdict__t__nocentre__report.json')


def test_main_writes_once_and_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(V, 'UQ_DIR', str(tmp_path))
    p = tmp_path / 'rec.json'
    p.write_text(json.dumps(record()))
    V.main([str(p)])
    out = tmp_path / 'd8' / 'verdict__t.json'
    v = json.load(open(out))
    assert v['wins'] is True and v['record'] == str(p) and v['report_only'] is False
    with pytest.raises(FileExistsError):
        V.main([str(p)])
    r = record(); r['provenance']['centre']['epoch'] = 3
    q = tmp_path / 'rec2.json'; q.write_text(json.dumps(r))
    with pytest.raises(ValueError):
        V.main([str(q)])
    V.main([str(q), '--no_centre_check', '--report_only'])
    assert (tmp_path / 'd8' / 'verdict__t__nocentre__report.json').exists()


def test_check_k_refuses_inexact_coverage_unless_report_only(tmp_path, monkeypatch, capsys):
    """R10: C2's empirical coverage is exact only for K >= min_k_for(0.9) = 19; below it the verdict is refused, and a
    --report_only verdict says so instead."""
    V.check_k(record())                                           # K = 32, no coverage_exact key
    r = record(); r['K'] = 8; r['coverage_exact'] = False
    with pytest.raises(ValueError, match='coverage_exact'):
        V.check_k(r)
    r2 = record(); r2['K'] = 8                                   # K alone is enough, whatever the flag says
    with pytest.raises(ValueError, match='min_k_for'):
        V.check_k(r2)
    r3 = record(); r3['coverage_exact'] = False                  # the flag alone is enough, whatever K says
    with pytest.raises(ValueError, match='coverage_exact'):
        V.check_k(r3)
    r4 = record(); r4['K'] = 19; V.check_k(r4)                   # the boundary is inclusive
    monkeypatch.setattr(V, 'UQ_DIR', str(tmp_path))
    p = tmp_path / 'rec.json'
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError, match='coverage_exact'):
        V.main([str(p)])
    assert not (tmp_path / 'd8').exists()                         # refused before anything is written
    capsys.readouterr()
    V.main([str(p), '--report_only'])
    assert 'WARNING' in capsys.readouterr().out
    assert json.load(open(tmp_path / 'd8' / 'verdict__t__report.json'))['report_only'] is True


def test_optional_methods_are_reported_and_the_verdict_is_plain_json():
    r = record()
    r['per_cube']['ensemble'] = dict(crps=[1.1] * N, ause=[0.06] * N, cov90=[0.7] * N)
    r['per_cube']['diffusion_halfcentre'] = dict(crps=[1.05] * N, cov90=[0.88] * N)
    c = V.clauses(r)
    rep = c['reported']
    assert 'ensemble' not in V.clauses(record())['reported'] and 'halfcentre' not in V.clauses(record())['reported']
    assert rep['ensemble']['cov90'] == pytest.approx(0.7) and rep['ensemble']['crps']['n'] == N
    assert rep['halfcentre']['cov90'] == pytest.approx(0.88) and rep['halfcentre']['crps']['clusters'] == 10
    assert json.loads(json.dumps(c)) == c                          # plain types only
    r = record(); r['per_cube']['diffusion']['cov50'] = [0.6] * N   # C2 also fails on the 50 % interval alone
    assert not V.clauses(r)['C2']['passed'] and V.clauses(r)['C1']['passed']
