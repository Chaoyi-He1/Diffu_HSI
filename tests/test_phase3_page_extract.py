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
