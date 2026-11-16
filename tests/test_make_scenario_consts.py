# tests/test_make_scenario_consts.py — the Phase 3b scenario constants from the clean Device-1 readings
import json
import os

import numpy as np
import pytest

from data_loader import scenarios as SC
from data_loader.hfd_sensor_sim import load_cube01
from data_loader.sensor_model import load_sensor
from scripts.sensor import make_scenario_consts as M


def _files(root):
    out = []
    for dp, _, fs in os.walk(root):
        out += [os.path.relpath(os.path.join(dp, f), root) for f in fs if f.endswith('.mat') and '_' in f and 'R_' not in f
                and 'PH5' not in f]
    return sorted(out)[:3]


def test_consts_record_from_the_clean_readings(fake_sensor_root):
    files = _files(fake_sensor_root)
    sensor = load_sensor('device1', fake_sensor_root)
    rec = M.build_consts(files, fake_sensor_root, sensor)
    clean = [load_cube01(os.path.join(fake_sensor_root, f)) @ sensor['R'] for f in files]
    ref = SC.scenario_consts(clean)
    assert rec['mean_y'] == ref['mean_y'] and rec['y_max'] == ref['y_max'] and rec['quantiles'] == ref['quantiles']
    assert rec['files'] == files and rec['n_cubes'] == 3 and rec['R_sha256'] == sensor['sha256']
    assert rec['sensor_device'] == 'device1' and rec['mode'] == 'sequential' and rec['readings'] == 'clean'
    assert {'git_head', 'created'} <= set(rec)
    json.dumps(rec)


def test_written_once(tmp_path):
    p = str(tmp_path / 'scenario_consts.json')
    M.write_consts(p, dict(mean_y=1.0, y_max=2.0, quantiles={'0.9': 1.5}))
    assert SC.load_consts(p)['y_max'] == 2.0
    with pytest.raises(FileExistsError):
        M.write_consts(p, dict(mean_y=3.0, y_max=4.0, quantiles={}))
    assert SC.load_consts(p)['mean_y'] == 1.0
    assert M.DEFAULT_OUT.endswith('/sensor/scenario_consts.json')
