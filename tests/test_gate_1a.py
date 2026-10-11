import numpy as np
import pytest

from scripts.gate_1a import best_row, check_comparable, gate, paired_on_intersection, paired_vs_model

FILES = [f'f{i}' for i in range(6)]
PROV = dict(d=4, base_channels=64, prediction_type='v', prior_sha256='p' * 64, meta_checked=True, ckpt='/x/runs/r/checkpoint_epoch_100.pth')


def row(name, nfe, vals, base=None, files=FILES):
    vals = np.asarray(vals, dtype=float)                     # [seed][cube]
    cube = vals.mean(0)
    base = np.asarray(base if base is not None else cube + 0.5)
    diff = cube - base
    return dict(name=name, nfe=nfe, rmse_pct=float(cube.mean()), rmse_pct_se=float(cube.std(ddof=1) / np.sqrt(len(cube))),
                paired_rmse_diff=float(diff.mean()), paired_rmse_se=float(diff.std(ddof=1) / np.sqrt(len(diff))),
                per_item={'rmse_pct': vals.tolist()}, files=list(files))


def ev(mode, rows, passes=True, **prov):
    return dict(provenance=dict(PROV, mode=mode, **prov), grid='full', seeds=[0, 1], d=4, rows=rows, yswap=dict(passes=passes), baseline=dict(rmse_pct=2.35))


def test_best_row_respects_nfe_and_excludes_ddpm1000():
    rows = [row('ddpm1000', 1000, [[0.1] * 6]), row('ddim20', 20, [[0.2] * 6]), row('ddim10', 10, [[0.5] * 6]), row('warm100_ddim', 10, [[0.4] * 6])]
    assert best_row(ev('residual', rows), 10)['name'] == 'warm100_ddim'
    assert best_row(ev('residual', rows), 20)['name'] == 'ddim20'
    assert best_row(ev('residual', rows), 1000)['name'] == 'ddim20'      # ddpm1000 is never eligible


def test_paired_vs_model_uses_per_cube_differences_and_same_cubes():
    b = row('ddim10', 10, [[1.0, 1.1, 1.2, 1.3, 1.4, 1.5]] * 2)
    a = row('ddim5', 5, [[2.0, 2.1, 2.2, 2.3, 2.4, 2.5]] * 2)
    r = paired_vs_model(b, a)
    assert r['diff'] == pytest.approx(-1.0) and r['se'] == pytest.approx(0.0, abs=1e-12) and r['n'] == 6
    with pytest.raises(ValueError):
        paired_vs_model(b, row('ddim10', 10, [[1.0] * 6], files=FILES[::-1]))


def test_paired_on_intersection_pairs_by_file():
    x = row('ddim10', 10, [[1.0, 1.1, 1.2, 1.3, 1.4, 1.5]] * 2)
    y = row('ddpm1000', 1000, [[0.9, 1.0, 1.3]] * 2, files=['f0', 'f1', 'f5'])
    r = paired_on_intersection(x, y)
    assert r['n'] == 3 and r['diff'] == pytest.approx(np.mean([0.1, 0.1, 0.2]))


def test_gate_threshold_uses_k_paired_se():
    base = np.full(6, 2.3)
    a_cube = np.array([2.0, 2.2, 2.4, 2.6, 2.8, 3.0]); b_cube = a_cube + np.array([-0.3, 0.1, -0.2, 0.0, -0.25, 0.05])
    b = ev('residual', [row('ddim10', 10, [b_cube, b_cube], base)])
    a = ev('standard', [row('ddim10', 10, [a_cube, a_cube], base)])
    ap = ev('standard_xhat', [row('ddim10', 10, [a_cube + 0.5, a_cube + 0.5], base)])
    g2, g1 = gate(b, a, ap, k=2.0, require_done=False), gate(b, a, ap, k=1.0, require_done=False)
    assert g2['vs_a']['diff'] == pytest.approx(-0.1) and g2['vs_a']['se'] == pytest.approx(0.0695, abs=1e-3)
    assert g2['vs_a']['pass'] is False and g2['verdict'] == 'FAIL'
    assert g1['vs_a']['pass'] is True


def test_gate_verdict_matrix():
    base = np.full(6, 2.3); n = np.linspace(-0.05, 0.05, 6)
    b = ev('residual', [row('ddim10', 10, [1.8 + n, 1.9 + n], base)])
    a = ev('standard', [row('ddim10', 10, [2.6 + n, 2.7 + n], base)])
    ap = ev('standard_xhat', [row('ddim10', 10, [2.4 + n, 2.5 + n], base)])
    g = gate(b, a, ap, require_done=False)
    assert g['verdict'] == 'PASS' and g['b']['selected_from_grid'] == 'ddim10' and g['prereg']['setting'] == 'ddim10'
    ap_equal = ev('standard_xhat', [row('ddim10', 10, [1.8 + n[::-1], 1.9 + n[::-1]], base)])   # same mean as B, per-cube spread
    assert gate(b, a, ap_equal, require_done=False)['verdict'] == 'PASS_NOT_OVER_APRIME'
    a_better = ev('standard', [row('ddim10', 10, [1.7 + n, 1.7 + n], base)])
    assert gate(b, a_better, ap, require_done=False)['verdict'] == 'FAIL'
    assert gate(ev('residual', b['rows'], passes=False), a, ap, require_done=False)['verdict'] == 'INVALID'
    assert gate(b, ev('standard', a['rows'], passes=False), ap, require_done=False)['vs_a']['ablation_blind'] is True


def test_gate_refuses_incomparable_inputs(tmp_path):
    base = np.full(6, 2.3)
    b = ev('residual', [row('ddim10', 10, [[1.8] * 6] * 2, base)]); a = ev('standard', [row('ddim10', 10, [[2.6] * 6] * 2, base)])
    ap = ev('standard_xhat', [row('ddim10', 10, [[2.4] * 6] * 2, base)])
    with pytest.raises(ValueError, match='mode'):
        gate(a, b, ap, require_done=False)
    for bad in (dict(a, grid='quick'), dict(a, seeds=[0]), dict(a, d=2), ev('standard', a['rows'], base_channels=256),
                ev('standard', a['rows'], prior_sha256='q' * 64), ev('standard', a['rows'], meta_checked=False)):
        with pytest.raises(ValueError):
            check_comparable([b, bad, ap])
    run = tmp_path / 'r'; run.mkdir(); (run / 'DONE').write_text('100\n')
    ok = ev('residual', b['rows'], ckpt=str(run / 'checkpoint_epoch_100.pth'))
    stale = ev('residual', b['rows'], ckpt=str(run / 'checkpoint_epoch_50.pth'))
    assert gate(ok, a, ap, require_done=False)['verdict'] in ('PASS', 'PASS_NOT_OVER_APRIME', 'FAIL')
    with pytest.raises(ValueError, match='DONE'):
        gate(stale, a, ap, require_done=True)
