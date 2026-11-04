import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from scripts.sensor import sd_verdict as sv  # noqa: E402

N = 60
E1 = (60.0, 50.0, 40.0, 30.0, 20.0)
POINT = ('D', 'C1')


def _files(n=N, offset=0):
    return [f'MatFlower60/Test/P0{22 + i % 3}/{(i + offset) // 2}_{i + offset}.mat' for i in range(n)]


BASE = 1.0 + 0.3 * np.random.RandomState(0).rand(N)


def _arr(scale, seed, n=N):
    return (BASE[:n] * scale + 0.001 * np.random.RandomState(seed).randn(n)).tolist()


def _m(method, rmse=None, seed=1, **extra):
    d = {}
    if rmse is not None:
        d['rmse_pct' if method in POINT else 'mean_rmse_pct'] = _arr(rmse, seed)
    for k, v in extra.items():
        d[k] = _arr(v, seed + 7) if k == 'crps' else [float(v)] * N
    return d


def _rec(files, name='r', K=32, dev='device1', mode='sequential', snr=30.0, **method_arrays):
    return dict(name=name, setting=dict(sensor_device=dev, mode=mode, snr_db=float(snr)), subset='test', K=K, n=len(files),
                files=list(files), classes=[f.split('/')[-2] for f in files], per_cube=method_arrays,
                coverage_exact=bool(K >= 19))


def _dscale(s):
    return 1.0 + (60.0 - s) / 20.0


DEFAULT = dict(t1=1.02, crps=0.9, cov=0.9, K=32, r1=0.8, r2=0.8, t4=0.8)


def _world(B=None, C=None, override=None, snrs_e1=E1, extra=True):
    """-> {setting key: [record, ...]}; B/C override DEFAULT; override[(snr, method)] = RMSE scale at device1/sequential."""
    spec = {'B': dict(DEFAULT, **(B or {})), 'C': dict(DEFAULT, **(C or {}))}
    override = override or {}
    files = _files()
    recs = {}

    def put(dev, mode, snr, *rs):
        recs[f'{dev}/{mode}/{snr:g}'] = list(rs)

    for s in snrs_e1:
        ds = _dscale(s)
        sc = lambda m, v: override.get((s, m), v)
        base = _rec(files, name=f'base{s:g}', snr=s,
                    L=_m('L', sc('L', 1.1 * ds), 2, crps=1.0 * ds), D=_m('D', sc('D', ds), 3),
                    V=_m('V', sc('V', 1.05 * ds), 4, crps=1.0 * ds, cov90=0.9))
        rb = _rec(files, name=f'b{s:g}', snr=s, K=spec['B']['K'],
                  B=_m('B', sc('B', spec['B']['t1'] * ds), 5, crps=spec['B']['crps'] * ds, cov90=spec['B']['cov']))
        rc = _rec(files, name=f'c{s:g}', snr=s, K=spec['C']['K'],
                  C=_m('C', sc('C', spec['C']['t1'] * ds), 6, crps=spec['C']['crps'] * ds, cov90=spec['C']['cov']))
        put('device1', 'sequential', s, base, rb, rc)
    if extra:
        for dev, k in (('device2', 'r2'), ('ph5', 'r2')):
            put(dev, 'sequential', 30.0, _rec(files, name=f'{dev}L', dev=dev, L=_m('L', 1.0, 8)),
                _rec(files, name=f'{dev}B', dev=dev, B=_m('B', spec['B'][k], 9)), _rec(files, name=f'{dev}C', dev=dev, C=_m('C', spec['C'][k], 10)))
        put('device1', 'snapshot', 30.0, _rec(files, name='snapL', mode='snapshot', L_demosaic=_m('L_demosaic', 1.0, 11),
                                                  D=_m('D', 1.0, 12)),
            _rec(files, name='snapB', mode='snapshot', B=_m('B', spec['B']['t4'], 13)),
            _rec(files, name='snapC', mode='snapshot', C=_m('C', spec['C']['t4'], 14)))
    return recs


def _load(tmp_path, monkeypatch, recs, name='w'):
    monkeypatch.setattr(sv, 'SENSOR_DIR', str(tmp_path))
    os.makedirs(tmp_path / 'eval', exist_ok=True)
    manifest = dict(name=name, records={})
    for key, rs in recs.items():
        manifest['records'][key] = []
        for r in rs:
            with open(tmp_path / 'eval' / (r['name'] + '.json'), 'w') as f:
                json.dump(r, f)
            manifest['records'][key].append(r['name'])
    mp = tmp_path / f'{name}_manifest.json'
    mp.write_text(json.dumps(manifest))
    return str(mp), sv.load_manifest(str(mp))


def test_constants_are_the_preregistered_ones():
    assert (sv.T1_RATIO, sv.T2_COV90, sv.T2_SNRS, sv.T3_RATIO, sv.T3_SNR, sv.T4_RATIO, sv.T4_SNR, sv.MIN_COMMON) == \
        (1.05, (0.85, 0.95), (30.0, 20.0), 1.10, 30.0, 0.85, 30.0, 45)


def test_paired_on_common_restricts_to_common_files_and_refuses_small_overlap():
    f200 = _files(200)
    f50 = f200[100:150][::-1]                          # a different order: the result follows the first list's order
    a = {'x': np.arange(200, dtype=float).tolist()}
    b = {'x': (np.arange(50, dtype=float) * 0 + 1).tolist()}
    r = sv.paired_on_common(a, f200, b, f50, 'x', 'x')
    assert r['n'] == 50 and r['clusters'] >= 2
    assert r['mean'] == pytest.approx(np.mean(np.arange(100, 150)) - 1.0)
    r2 = sv.paired_on_common(b, f50, a, f200, 'x', 'x')                      # 50's order
    assert r2['n'] == 50 and r2['mean'] == pytest.approx(1.0 - np.mean(np.arange(100, 150)))
    assert r['ratio_of_means'] == pytest.approx(np.mean(np.arange(100, 150)) / 1.0)
    with pytest.raises(ValueError, match='common'):
        sv.paired_on_common(a, f200, {'x': [1.0] * 30}, f200[:30], 'x', 'x')


def test_ratio_clause_pass_and_fail(tmp_path, monkeypatch):
    for ratio, ok in ((1.02, True), (1.08, False)):
        (tmp_path / str(ratio)).mkdir()
        _, st = _load(tmp_path / str(ratio), monkeypatch, _world(B=dict(t1=ratio)))
        res = sv.t1(st, 'B')
        assert set(res) == set(E1)
        assert all(c['passed'] == ok for c in res.values())
        c = res[30.0]
        assert c['ratio'] == pytest.approx(ratio, rel=1e-2) and c['n'] == N
        assert (c['mean'] + 2 * c['se_cluster'] <= 0) is ok


def test_t2_requires_K_19_unless_report_only(tmp_path, monkeypatch):
    _, st = _load(tmp_path, monkeypatch, _world(C=dict(K=8)))
    with pytest.raises(ValueError, match='19'):
        sv.t2(st, 'C')
    r = sv.t2(st, 'C', report_only=True)
    assert set(r) == set(sv.T2_SNRS)
    ok = sv.t2(st, 'B')
    assert all(v['passed'] and v['crps_vs_L']['passed'] and v['crps_vs_V']['passed'] for v in ok.values())
    assert ok[30.0]['cov90'] == pytest.approx(0.9)


def test_t2_fails_on_coverage_or_crps(tmp_path, monkeypatch):
    _, st = _load(tmp_path, monkeypatch, _world(B=dict(cov=0.8), C=dict(crps=1.1)))
    rb, rc = sv.t2(st, 'B'), sv.t2(st, 'C')
    assert not rb[30.0]['passed'] and rb[30.0]['crps_vs_L']['passed']
    assert not rc[20.0]['passed'] and not rc[20.0]['crps_vs_V']['passed']


@pytest.mark.parametrize('r2,ok', [(0.86, True), (0.9, False)])
def test_t3_uses_the_device1_reference_ratio(tmp_path, monkeypatch, r2, ok):
    # device1 reference: B / L = 1.02 * ds / (1.1 * ds) -> pin it at 0.8 through the device1 B scale
    _, st = _load(tmp_path, monkeypatch, _world(B=dict(t1=0.8 * 1.1, r2=r2)))
    r = sv.t3(st, 'B')
    assert r['ratio_dev1'] == pytest.approx(0.8, rel=1e-2) and r['ratio_dev2'] == pytest.approx(r2, rel=1e-2)
    assert r['passed'] is ok
    assert r['ph5'] == pytest.approx(r2, rel=1e-2)


def test_t4_against_demosaic(tmp_path, monkeypatch):
    _, st = _load(tmp_path, monkeypatch, _world(B=dict(t4=0.8), C=dict(t4=0.9)))
    b, c = sv.t4(st, 'B'), sv.t4(st, 'C')
    assert b['passed'] and not c['passed']
    assert b['ratio'] == pytest.approx(0.8, rel=1e-2) and c['ratio'] == pytest.approx(0.9, rel=1e-2)
    assert b['D']['ratio'] == pytest.approx(1.0, rel=1e-2) and not b['D']['passed']


def _fake(b_pass, c_pass, crps20=(0.5, 0.4)):
    def t1(p):
        return {s: dict(passed=p) for s in E1}

    def t2(p, v):
        return {s: dict(passed=p, crps_vs_V=dict(mean=v)) for s in sv.T2_SNRS}
    return dict(T1=dict(B=t1(b_pass[0]), C=t1(c_pass[0])), T2=dict(B=t2(b_pass[1], crps20[0]), C=t2(c_pass[1], crps20[1])),
                T3=dict(B=dict(passed=b_pass[2]), C=dict(passed=c_pass[2])), T4=dict(B=dict(passed=b_pass[3]), C=dict(passed=c_pass[3])))


def test_main_line_counts_and_tiebreak():
    assert sv.main_line(_fake((1, 1, 1, 1), (1, 1, 0, 0))) == 'B'
    assert sv.main_line(_fake((1, 0, 0, 0), (1, 1, 0, 0))) == 'C'
    assert sv.main_line(_fake((1, 1, 0, 0), (1, 1, 0, 0), crps20=(0.5, 0.4))) == 'C'       # lower CRPS - V paired mean at 20 dB
    assert sv.main_line(_fake((1, 1, 0, 0), (1, 1, 0, 0), crps20=(0.3, 0.4))) == 'B'
    f = _fake((1, 1, 0, 0), (1, 1, 0, 0))
    f['T1']['B'][40.0] = dict(passed=False)                                                # T1 counts once, only when all five pass
    assert sv.main_line(f) == 'C'                                                          # B: 2 of 3, C: 3 -> C wins on count alone


def test_sanity_monotone_and_floor(tmp_path, monkeypatch):
    _, st = _load(tmp_path, monkeypatch, _world())
    s = sv.sanity(st)
    assert all(s['monotone'].values()) and all(s['floor_pass'].values())
    assert set(s['monotone']) >= {'L', 'D', 'V', 'B', 'C'}
    assert s['floor_reference_rmse_pct'] == pytest.approx(1.1 * float(np.mean(BASE)), rel=1e-2)
    (tmp_path / 'a').mkdir()
    _, st = _load(tmp_path / 'a', monkeypatch, _world(override={(60.0, 'V'): 2.0}))
    s = sv.sanity(st)
    assert not s['monotone']['V'] and not s['floor_pass']['V'] and s['monotone']['D']
    (tmp_path / 'b').mkdir()
    _, st = _load(tmp_path / 'b', monkeypatch, _world(override={(60.0, 'D'): 1.2}))
    s = sv.sanity(st)
    assert s['monotone']['D'] and not s['floor_pass']['D'] and s['floor_pass']['B']


def test_duplicate_method_in_a_setting_is_an_error(tmp_path, monkeypatch):
    recs = _world()
    recs['device1/sequential/30'].append(_rec(_files(), name='dup', B=_m('B', 1.0, 3)))
    monkeypatch.setattr(sv, 'SENSOR_DIR', str(tmp_path))
    with pytest.raises(ValueError, match='B'):
        _load(tmp_path, monkeypatch, recs)


def test_full_verdict_and_main_line(tmp_path, monkeypatch):
    _, st = _load(tmp_path, monkeypatch, _world(C=dict(t4=0.95)))
    v = sv.verdict(st)
    assert v['counts'] == {'B': 5, 'C': 4} and v['main_line'] == 'B' and v['missing'] == [] and not v['report_only']


def test_report_only_lists_missing_settings_and_writes_report_name(tmp_path, monkeypatch):
    recs = _world()
    for k in ('device2/sequential/30', 'ph5/sequential/30', 'device1/snapshot/30', 'device1/sequential/60'):
        recs.pop(k)
    mp, st = _load(tmp_path, monkeypatch, recs, name='part')
    with pytest.raises(Exception):
        sv.verdict(st)
    v = sv.verdict(st, report_only=True)
    assert v['report_only'] and v['main_line'] is None and v['missing']
    assert any('device2/sequential/30' in m for m in v['missing'])
    sv.main(['--manifest', mp, '--report_only'])
    path = tmp_path / 'verdict__part__report.json'
    assert path.exists() and json.loads(path.read_text())['main_line'] is None
    with pytest.raises(FileExistsError):
        sv.main(['--manifest', mp, '--report_only'])
