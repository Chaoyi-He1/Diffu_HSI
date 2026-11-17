"""Phase 3 page extractor: a stale B record (not x0-prior) is listed under status.stale, never final, and its hours are
counted under B_stale, not B (final-review Minor 4). Synthetic records in a temp EVAL dir; no shared results are read."""
import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location(
    'extract_page_data', os.path.join(os.path.dirname(HERE), 'docs', 'analysis', 'phase3_page', 'extract_page_data.py'))
X = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(X)

FILES = [f'MatFlower60/Test/P022/{i // 2}_{i}.mat' for i in range(6)]          # 3 source images, 2 cubes each


def _rec(methods, seconds, pred_type=None, snr=30.0):
    pc = {m: {'mean_rmse_pct': [1.0] * 6, 'rmse_pct': [1.0] * 6, 'crps': [0.5] * 6} for m in methods}
    prov = dict(seconds=seconds, created='2026-11-06T10:00:00')
    if pred_type:
        prov['b_prediction_type'] = pred_type
    return dict(n=6, K=32, pigdm_steps=50, files=FILES, setting=dict(sigma=0.1, snr_db=snr, sensor_device='device1', mode='sequential'),
                per_cube=pc, means={m: {k: v[0] for k, v in d.items()} for m, d in pc.items()}, provenance=prov)


@pytest.fixture
def evaldir(tmp_path, monkeypatch):
    monkeypatch.setattr(X, 'EVAL', str(tmp_path))
    monkeypatch.setattr(X, 'EXPERIMENTS', [('E1', 'device1', 'sequential', [30, 20], 'e1_dev1_seq_{snr}')])
    for snr in (30, 20):
        for kind, ms in (('main', ('L', 'D', 'V')), ('C', ('C', 'C1'))):
            (tmp_path / f'e1_dev1_seq_{snr}_{kind}.json').write_text(json.dumps(_rec(ms, 3600, snr=snr)))
    (tmp_path / 'e1_dev1_seq_30_B.json').write_text(json.dumps(_rec(('B',), 7200, 'x0')))
    (tmp_path / 'e1_dev1_seq_20_B.json').write_text(json.dumps(_rec(('B',), 1800, 'v', snr=20.0)))
    return tmp_path


def test_stale_B_record_is_listed_as_stale_not_final(evaldir):
    S, final, pending, stale = X.settings_block()
    assert 'e1_dev1_seq_30_B' in final and 'e1_dev1_seq_20_B' not in final
    assert stale == ['e1_dev1_seq_20_B'] and S['device1_sequential_20']['B']['status'] == 'stale'
    assert S['device1_sequential_30']['B']['status'] == 'final' and S['device1_sequential_30']['clusters'] == 3


def test_eval_hours_count_only_x0_B_records(evaldir):
    hours, nb = X.eval_hours()
    assert nb == 1 and hours['B'] == pytest.approx(2.0) and hours['B_stale'] == pytest.approx(0.5)
    assert hours['main'] == pytest.approx(2.0) and hours['C'] == pytest.approx(2.0)


# ---------------------------------------------------------------- Phase 3b block (B⁺ and the zero-shot scenarios)
def _vrec(methods, files=FILES, seconds=3600, scale=None):
    """per-cube values 1, 2, ..., n (times a per-method scale) so re-averaging on a subset is checkable."""
    n = len(files)
    pc = {}
    for m in methods:
        f = (scale or {}).get(m, 1.0)
        pc[m] = {'mean_rmse_pct': [f * (i + 1) for i in range(n)], 'rmse_pct': [f * (i + 1) for i in range(n)],
                 'single_rmse_pct': [2 * f * (i + 1) for i in range(n)], 'crps': [0.5 * f * (i + 1) for i in range(n)],
                 'cov90': [0.9] * n}
        if m in X.POINT_ONLY:
            pc[m] = {k: pc[m][k] for k in ('rmse_pct',)}
    means = {m: {k: sum(v) / n for k, v in d.items()} for m, d in pc.items()}
    return dict(n=n, K=20, files=list(files), per_cube=pc, means=means,
                setting=dict(sigma=0.1, snr_db=30.0, state_indices=None),
                provenance=dict(seconds=seconds, created='2026-11-17T01:00:00'))


def test_norm_means_reads_the_point_key_for_one_call_methods():
    assert X.norm_means({'rmse_pct': 2.0, 'mean_rmse_pct': 9.0}, 'D')['rmse'] == 2.0
    assert X.norm_means({'rmse_pct': 2.0}, 'D_naive') == {'rmse': 2.0}
    out = X.norm_means({'mean_rmse_pct': 3.0, 'single_rmse_pct': 4.0, 'crps': 1.0, 'cov90': 0.9}, 'Bp')
    assert out == {'rmse': 3.0, 'single': 4.0, 'crps': 1.0, 'cov90': 0.9}


def test_sub_means_reaverages_on_the_common_files():
    rec = _vrec(('L', 'D'))
    sub = X.sub_means(rec, ('L', 'D', 'V'), [FILES[0], FILES[2], 'not/in/record.mat'])
    assert set(sub) == {'L', 'D'} and sub['L']['rmse'] == pytest.approx(2.0) and sub['L']['n'] == 2
    assert sub['D']['rmse'] == pytest.approx(2.0) and sub['L']['crps'] == pytest.approx(1.0)
    assert X.sub_means(rec, ('L',), [FILES[0]]) == {} and X.sub_means(None, ('L',), FILES) == {}


def test_best_of_and_integrity():
    assert X.best_of({'L_interp': 4.1, 'L_interp_plain': 3.9}) == 'L_interp_plain'
    assert X.best_of({'L_interp': 4.1, 'L_interp_plain': None}) == 'L_interp' and X.best_of({}) is None
    got = X.integrity([({'rmse': 2.0, 'crps': 1.0, 'cov90': 0.30}, {'mean_rmse_pct': 2.01, 'crps': 1.0, 'cov90': 0.28}),
                       ({'rmse': 3.0, 'crps': 1.5, 'cov90': 0.40}, {'mean_rmse_pct': 2.97, 'crps': 1.52, 'cov90': 0.40})])
    assert got['rmse'] == pytest.approx(0.03) and got['crps'] == pytest.approx(0.02) and got['cov90'] == pytest.approx(0.02)
    assert got['n'] == 2 and X.integrity([])['rmse'] is None


def test_bplus_setting_pairs_on_the_B_records_cubes():
    sub_files = FILES[:4]
    bprec = _vrec(('B', 'Bp'), files=sub_files, scale={'B': 1.0, 'Bp': 0.5})
    main = _vrec(('L', 'D', 'V'), scale={'L': 2.0, 'D': 1.0, 'V': 2.0})
    crec = _vrec(('C', 'C1'), files=sub_files)
    blk = X.bplus_setting(bprec, main, crec, 'sequential')
    assert blk['status'] == 'final' and blk['L_ref'] == 'L' and blk['hours'] == pytest.approx(1.0)
    assert blk['means']['Bp']['rmse'] == pytest.approx(1.25) and blk['sub']['D']['rmse'] == pytest.approx(2.5)
    p = blk['paired']['rmse_vs_D']
    assert p['n'] == 4 and p['ratio'] == pytest.approx(0.5) and p['diff'] == pytest.approx(-1.25)
    assert blk['paired']['rmse_vs_B']['ratio'] == pytest.approx(0.5) and blk['paired']['rmse_vs_L']['ratio'] == pytest.approx(0.25)
    assert X.bplus_setting(bprec, main, None, 'snapshot')['L_ref'] == 'L_demosaic'


def test_scenario_block_pending_then_final(tmp_path, monkeypatch):
    monkeypatch.setattr(X, 'EVAL', str(tmp_path))
    cfg = dict(sensor_device='device1', snr_db=30.0, scenario=dict(name='dead_50', R={'kind': 'device'}, states=None,
                                                                   pixels={'kind': 'dead', 'p': 0.5}, noise={'kind': 'gaussian'}))
    main = _vrec(('L', 'L_interp', 'D', 'V'), scale={'L': 3.0, 'L_interp': 1.2, 'D': 2.0, 'V': 3.0})
    (tmp_path / 'g_dead_50_main.json').write_text(json.dumps(main))
    (tmp_path / 'g_dead_50_interp.json').write_text(json.dumps(_vrec(('L_interp_plain',), scale={'L_interp_plain': 1.1})))
    canon = FILES[:4]
    s = X.scenario_block('dead_50', cfg, 'B3', canon)
    assert s['status'] == 'pending' and s['c_status'] == 'pending' and s['sub_files'] == 'subset'
    assert 'B' not in s and 'Bp' not in s and s['paired'] == {}
    assert s['sub']['L_interp_plain']['rmse'] == pytest.approx(2.75) and s['best_classical'] == 'L_interp_plain'
    assert s['main']['L']['rmse'] == pytest.approx(10.5) and s['main']['L_interp_plain']['rmse'] == pytest.approx(3.85)
    (tmp_path / 'g_dead_50_B.json').write_text(json.dumps(_vrec(('B', 'Bp'), files=FILES[2:], scale={'B': 1.0, 'Bp': 1.0})))
    (tmp_path / 'g_dead_50_C.json').write_text(json.dumps(_vrec(('C', 'C1'), files=FILES[2:])))
    s = X.scenario_block('dead_50', cfg, 'B3', canon)
    assert s['status'] == 'final' and s['c_status'] == 'final' and s['sub_files'] == 'B' and s['K'] == 20
    assert s['sub']['D']['rmse'] == pytest.approx(9.0) and s['Bp']['rmse'] == pytest.approx(2.5) and s['C1']['rmse'] == pytest.approx(2.5)
    pb = s['paired']['rmse_vs_best_classical']
    assert pb['n'] == 4 and pb['b'] == pytest.approx(1.1 * 4.5) and pb['ratio'] == pytest.approx(2.5 / (1.1 * 4.5))
    assert s['paired']['rmse_vs_D']['ratio'] == pytest.approx(2.5 / 9.0)


def test_eval_hours_ignore_phase3b_records(evaldir):
    (evaldir / 'e1_dev1_seq_30_Bp.json').write_text(json.dumps(_rec(('B', 'Bp'), 7200, 'x0')))
    (evaldir / 'g_blend_050_B.json').write_text(json.dumps(_rec(('B', 'Bp'), 3600, 'x0')))
    (evaldir / 'g_blend_050_main.json').write_text(json.dumps(_rec(('L', 'D', 'V'), 3600)))
    hours, nb = X.eval_hours()
    assert nb == 1 and hours['B'] == pytest.approx(2.0) and hours['main'] == pytest.approx(2.0) and 'Bp' not in hours
