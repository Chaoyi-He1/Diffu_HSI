# tests/test_scenarios_3b.py — the 24 pre-registered Phase 3b settings (spec 2026-11-16 Part B)
import json

import numpy as np

from data_loader import scenarios as SC
from scripts.sensor.scenarios_3b import GROUPS, SCENARIOS


def test_exactly_the_24_preregistered_settings():
    assert len(SCENARIOS) == 24
    assert {g: len(v) for g, v in GROUPS.items()} == {'B1': 8, 'B2': 6, 'B3': 5, 'B4': 5}
    assert sorted(sum(GROUPS.values(), [])) == sorted(SCENARIOS)
    for name, s in SCENARIOS.items():
        assert set(s) == {'sensor_device', 'snr_db', 'scenario'} and s['sensor_device'] == 'device1'
        assert s['scenario']['name'] == name and set(s['scenario']) == {'name', 'R', 'states', 'pixels', 'noise'}
        json.dumps(s)
    g = {n: SCENARIOS[n] for n in SCENARIOS}
    assert [g[n]['scenario']['R']['alpha'] for n in ('blend_025', 'blend_050', 'blend_075')] == [0.25, 0.5, 0.75]
    assert [g[n]['scenario']['R']['bands'] for n in ('shift_1', 'shift_2', 'shift_4')] == [1, 2, 4]
    assert [(g[n]['scenario']['R']['delta'], g[n]['scenario']['R']['seed']) for n in ('gain_02', 'gain_05')] == [(0.2, 0), (0.5, 0)]
    assert all(g[n]['snr_db'] == 30.0 for n in GROUPS['B1'])
    assert [(g[n]['scenario']['states'], g[n]['snr_db']) for n in GROUPS['B2']] == \
        [(15, 40.0), (10, 40.0), (5, 40.0), (15, 20.0), (10, 20.0), (5, 20.0)]
    assert [(g[n]['scenario']['pixels'], g[n]['snr_db']) for n in GROUPS['B3']] == [
        ({'kind': 'dead', 'p': 0.5}, 30.0), ({'kind': 'dead', 'p': 0.9}, 30.0), ({'kind': 'stride', 'd': 4}, 60.0),
        ({'kind': 'stride', 'd': 8}, 60.0), ({'kind': 'stride', 'd': 4}, 30.0)]
    assert [(g[n]['scenario']['noise'], g[n]['snr_db']) for n in GROUPS['B4']] == [
        ({'kind': 'poisson'}, 30.0), ({'kind': 'poisson'}, 20.0), ({'kind': 'saturation', 'q': 0.9}, 30.0),
        ({'kind': 'saturation', 'q': 0.7}, 30.0), ({'kind': 'quantisation', 'bits': 4}, 40.0)]
    for n in GROUPS['B1'] + GROUPS['B2'] + GROUPS['B3']:
        assert g[n]['scenario']['noise'] == {'kind': 'gaussian'}
    assert len({SC.scenario_sha(s['scenario']) for s in SCENARIOS.values()}) == 24


def test_every_setting_builds(small_R):
    R2 = np.flipud(small_R).copy()
    x = np.random.default_rng(0).random((12, 12, 31))
    consts = SC.scenario_consts(x @ small_R)
    for name, s in SCENARIOS.items():
        sc = s['scenario']
        R = SC.scenario_R(small_R, R2, sc)
        S = sc['states'] or 30
        assert R.shape == (31, S), name
        meas = SC.pixel_measured(12, 12, sc['pixels'], np.random.default_rng(0))
        assert meas.any()
        m = SC.measure_scenario(x, R, 0.05, sc, np.random.default_rng(1), consts)
        assert m['y'].shape == m['mask'].shape == m['noise_var'].shape == (12, 12, S), name
        assert np.isfinite(m['y']).all() and (m['noise_var'] > 0).all()
        assert ('y_naive' in m) == SC.has_naive(sc) and ('measured' in m) == SC.has_pixels(sc)
