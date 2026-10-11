import json

import numpy as np
import pytest

from scripts.gate_1a import best_row, check_comparable, default_out_name, gate, main, paired_on_intersection, paired_vs_model

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
    """A full-grid evaluation; like the real ones it carries a ddpm1000 row on a subset of the cubes."""
    rows = list(rows)
    if not any(r['name'] == 'ddpm1000' for r in rows):
        rows.append(row('ddpm1000', 1000, [[2.0, 2.1, 2.2]] * 2, files=FILES[:3]))
    return dict(provenance=dict(PROV, mode=mode, **prov), grid='full', seeds=[0, 1], d=4, rows=rows, yswap=dict(passes=passes), baseline=dict(rmse_pct=2.35))


def make_run(tmp_path, name, epoch=100, done=True, final=None):
    """A run directory with a DONE file (holding `final`, default `epoch`); returns the checkpoint path inside it."""
    run = tmp_path / name; run.mkdir()
    if done:
        (run / 'DONE').write_text(f'{epoch if final is None else final}\n')
    return str(run / f'checkpoint_epoch_{epoch}.pth')


def trio(vals=(1.8, 2.6, 2.4), **ckpts):
    """B / A / A' evaluations with constant per-cube RMSE `vals` against a 2.3 baseline (B beats everything with SE 0)."""
    base = np.full(6, 2.3)
    return [ev(mode, [row('ddim10', 10, [[v] * 6] * 2, base)], **({'ckpt': ckpts[slot]} if slot in ckpts else {}))
            for slot, mode, v in zip(('b', 'a', 'aprime'), ('residual', 'standard', 'standard_xhat'), vals)]


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


def test_paired_needs_at_least_two_cubes():
    one = dict(files=['f0'], per_item={'rmse_pct': [[1.0]]})
    with pytest.raises(ValueError, match='at least 2 paired cubes'):
        paired_vs_model(one, one)
    two = dict(files=['f0', 'f9'], per_item={'rmse_pct': [[1.0, 1.0]]})
    with pytest.raises(ValueError, match='at least 2 paired cubes'):
        paired_on_intersection(row('ddim10', 10, [[1.0] * 6]), two)               # only f0 is common


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


def test_gate_baseline_threshold_uses_k_paired_se():
    b_cube = np.array([2.0, 2.2, 2.4, 2.6, 2.8, 3.0])
    base = b_cube + 0.12 + np.array([-0.2, 0.2, -0.1, 0.1, -0.2, 0.2])          # B beats the baseline by 0.12 = 1.55 paired SE
    b = ev('residual', [row('ddim10', 10, [b_cube, b_cube], base)])
    a = ev('standard', [row('ddim10', 10, [b_cube + 0.5, b_cube + 0.5], base)])
    ap = ev('standard_xhat', [row('ddim10', 10, [b_cube + 0.5, b_cube + 0.5], base)])
    g2, g1 = gate(b, a, ap, k=2.0, require_done=False), gate(b, a, ap, k=1.0, require_done=False)
    assert g2['vs_baseline']['diff'] == pytest.approx(-0.12) and g2['vs_baseline']['se'] == pytest.approx(0.0775, abs=1e-3)
    assert g2['vs_a']['pass'] is True and g2['vs_aprime']['pass'] is True       # only the baseline comparison decides
    assert g2['vs_baseline']['pass'] is False and g2['verdict'] == 'FAIL'
    assert g1['vs_baseline']['pass'] is True and g1['verdict'] == 'PASS'


def test_gate_verdict_matrix():
    base = np.full(6, 2.3); n = np.linspace(-0.05, 0.05, 6)
    b = ev('residual', [row('ddim10', 10, [1.8 + n, 1.9 + n], base)])
    a = ev('standard', [row('ddim10', 10, [2.6 + n, 2.7 + n], base)])
    ap = ev('standard_xhat', [row('ddim10', 10, [2.4 + n, 2.5 + n], base)])
    g = gate(b, a, ap, require_done=False)
    assert g['verdict'] == 'PASS' and g['b']['selected_from_grid'] == 'ddim10' and g['prereg']['setting'] == 'ddim10'
    assert isinstance(g['prereg']['vs_a']['pass'], bool) and g['prereg']['vs_a']['pass'] is True
    ap_equal = ev('standard_xhat', [row('ddim10', 10, [1.8 + n[::-1], 1.9 + n[::-1]], base)])   # same mean as B, per-cube spread
    assert gate(b, a, ap_equal, require_done=False)['verdict'] == 'PASS_NOT_OVER_APRIME'
    a_better = ev('standard', [row('ddim10', 10, [1.7 + n, 1.7 + n], base)])
    assert gate(b, a_better, ap, require_done=False)['verdict'] == 'FAIL'
    assert gate(ev('residual', b['rows'], passes=False), a, ap, require_done=False)['verdict'] == 'INVALID'
    assert gate(b, ev('standard', a['rows'], passes=False), ap, require_done=False)['vs_a']['ablation_blind'] is True


def test_prereg_ddim10_is_reported_as_it_is_when_it_differs_from_the_selected_row():
    base = np.full(6, 2.3); n = np.linspace(-0.05, 0.05, 6)
    b = ev('residual', [row('ddim10', 10, [2.45 + n, 2.45 + n], base), row('warm100_ddim', 10, [1.8 + n, 1.8 + n], base)])
    a = ev('standard', [row('ddim10', 10, [2.4 + n, 2.4 + n], base)])
    ap = ev('standard_xhat', [row('ddim10', 10, [2.5 + n, 2.5 + n], base)])
    g = gate(b, a, ap, require_done=False)
    assert g['b']['selected_from_grid'] == 'warm100_ddim' and g['verdict'] == 'PASS'              # wins at its own best row ...
    pr = g['prereg']
    assert pr['setting'] == 'ddim10'
    assert pr['rmse_pct'] == dict(b=pytest.approx(2.45), a=pytest.approx(2.4), aprime=pytest.approx(2.5))
    assert pr['vs_baseline']['diff'] == pytest.approx(0.15) and pr['vs_baseline']['pass'] is False   # ... but not at ddim10
    assert pr['vs_a']['diff'] == pytest.approx(0.05) and pr['vs_a']['pass'] is False
    assert pr['vs_aprime']['diff'] == pytest.approx(-0.05) and pr['vs_aprime']['pass'] is True
    assert g['vs_a']['diff'] == pytest.approx(-0.6) and g['vs_baseline']['pass'] is True


def test_gate_refuses_incomparable_inputs():
    b, a, ap = trio()
    with pytest.raises(ValueError, match='mode'):
        gate(a, b, ap, require_done=False)
    no_grid = {k: v for k, v in a.items() if k != 'grid'}
    no_sha = ev('standard', a['rows']); del no_sha['provenance']['prior_sha256']
    for bad in (dict(a, grid='quick'), no_grid, dict(a, seeds=[0]), dict(a, d=2), ev('standard', a['rows'], base_channels=256),
                ev('standard', a['rows'], prediction_type='eps'), ev('standard', a['rows'], prior_sha256='q' * 64),
                ev('standard', a['rows'], meta_checked=False)):
        with pytest.raises(ValueError):
            check_comparable([b, bad, ap])
    with pytest.raises(ValueError, match='prior_sha256'):                      # a missing key is a refusal that names it, not a KeyError
        check_comparable([b, no_sha, ap])
    with pytest.raises(ValueError, match='prediction_type'):
        check_comparable([b, ev('standard', a['rows'], prediction_type='eps'), ap])


@pytest.mark.parametrize('slot', ['b', 'a', 'aprime'])
def test_gate_requires_each_model_to_be_at_its_runs_final_checkpoint(tmp_path, slot):
    modes = dict(b='residual', a='standard', aprime='standard_xhat')
    b, a, ap = trio(**{s: make_run(tmp_path, s) for s in modes})                  # every run dir has a DONE file saying 100
    assert gate(b, a, ap, require_done=True)['verdict'] == 'PASS'
    good = dict(b=b, a=a, aprime=ap)
    stale = dict(good, **{slot: dict(good[slot], provenance=dict(good[slot]['provenance'], ckpt=make_run(tmp_path, 'stale', epoch=50, final=100)))})
    with pytest.raises(ValueError, match='not the final checkpoint'):
        gate(stale['b'], stale['a'], stale['aprime'], require_done=True)
    unfinished = dict(good, **{slot: dict(good[slot], provenance=dict(good[slot]['provenance'], ckpt=make_run(tmp_path, 'unfinished', done=False)))})
    with pytest.raises(ValueError, match='no DONE file'):
        gate(unfinished['b'], unfinished['a'], unfinished['aprime'], require_done=True)
    assert gate(unfinished['b'], unfinished['a'], unfinished['aprime'], require_done=False)['require_done'] is False      # the peek


def test_gate_unequal_training_lengths(tmp_path):
    b, a, ap = trio(b=make_run(tmp_path, 'b', epoch=60), a=make_run(tmp_path, 'a'), aprime=make_run(tmp_path, 'ap'))
    for require_done in (True, False):
        with pytest.raises(ValueError, match='different numbers of epochs') as exc:
            gate(b, a, ap, require_done=require_done)
        assert 'B=60' in str(exc.value) and 'A=100' in str(exc.value)
        g = gate(b, a, ap, require_done=require_done, allow_unequal_epochs=True)
        assert g['verdict'] == 'PASS_UNEQUAL_EPOCHS' and g['final_epochs'] == dict(b=60, a=100, aprime=100)
    b_inv = dict(b, yswap=dict(passes=False))
    assert gate(b_inv, a, ap, allow_unequal_epochs=True)['verdict'] == 'INVALID_UNEQUAL_EPOCHS'
    b, a, ap = trio(b=make_run(tmp_path, 'b2'), a=make_run(tmp_path, 'a2'), aprime=make_run(tmp_path, 'ap2'))       # equal: no suffix
    g = gate(b, a, ap, allow_unequal_epochs=True)
    assert g['verdict'] == 'PASS' and g['final_epochs'] == dict(b=100, a=100, aprime=100)


@pytest.mark.parametrize('slot, missing', [('b', 'ddim10'), ('a', 'ddpm1000'), ('aprime', 'ddpm1000'), ('aprime', 'ddim10')])
def test_gate_refuses_a_full_grid_evaluation_without_ddim10_or_ddpm1000(slot, missing):
    evs = dict(zip(('b', 'a', 'aprime'), trio()))
    evs[slot] = dict(evs[slot], rows=[r for r in evs[slot]['rows'] if r['name'] != missing])
    if missing == 'ddim10':                                                      # keep another eligible row so only the pre-registered one is missing
        evs[slot]['rows'].append(row('ddim5', 5, [[2.0] * 6] * 2, np.full(6, 2.3)))
    with pytest.raises(ValueError, match=rf'--{slot} .*{missing}'):
        gate(evs['b'], evs['a'], evs['aprime'], require_done=False)


def test_default_out_name_distinguishes_peeks_and_other_thresholds():
    assert default_out_name('res_d4_b64', True, 10, 2.0) == 'gate_1a__res_d4_b64.json'
    assert default_out_name('res_d4_b64', False, 10, 2.0) == 'gate_1a__res_d4_b64__nodone.json'
    assert default_out_name('res_d4_b64', True, 5, 2.0) == 'gate_1a__res_d4_b64__nfe5.json'
    assert default_out_name('res_d4_b64', True, 10, 1.5) == 'gate_1a__res_d4_b64__k1.5.json'
    assert default_out_name('res_d4_b64', False, 20, 1.0) == 'gate_1a__res_d4_b64__nodone__nfe20__k1.json'


def test_cli_records_the_decision_context(tmp_path, capsys):
    ckpts = {s: make_run(tmp_path, s) for s in ('b', 'a', 'aprime')}
    paths = {}
    for (slot, e) in zip(('b', 'a', 'aprime'), trio(**ckpts)):
        paths[slot] = str(tmp_path / f'{slot}.json')
        with open(paths[slot], 'w') as f:
            json.dump(dict(e, run_name=f'run_{slot}'), f)
    argv = ['--b', paths['b'], '--a', paths['a'], '--aprime', paths['aprime']]
    main(argv)
    printed = capsys.readouterr().out
    assert ': PASS' in printed and 'pre-registered ddim10:' in printed
    g = json.load(open(tmp_path / 'gate_1a__run_b.json'))
    assert g['require_done'] is True and g['max_nfe'] == 10 and g['k'] == 2.0
    assert g['inputs'] == paths
    assert g['run_names'] == dict(b='run_b', a='run_a', aprime='run_aprime')
    assert g['final_epochs'] == dict(b=100, a=100, aprime=100)
    assert g['prereg']['setting'] == 'ddim10' and g['prereg']['vs_a']['pass'] is True
    main(argv + ['--no_require_done', '--max_nfe', '20', '--k', '1.5'])       # a peek with other thresholds writes elsewhere
    g = json.load(open(tmp_path / 'gate_1a__run_b__nodone__nfe20__k1.5.json'))
    assert g['require_done'] is False and g['max_nfe'] == 20 and g['k'] == 1.5
    assert json.load(open(tmp_path / 'gate_1a__run_b.json'))['require_done'] is True       # the official file was not touched
