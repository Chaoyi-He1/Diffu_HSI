"""Phase 1a-prime gate (Task 4): proposed one-step validity rule, verdicts, refusals, loss and seed pairs, best, the
report-only comparison with B's K-draw mean, CLI."""
import json

import numpy as np
import pytest

from scripts import gate_1a_prime as gp

FILES = [f'MatFlower60/Train/P0{22 + i % 3}/img{i // 2}_{i % 2}.mat' for i in range(12)]
BASE = np.linspace(2.0, 3.0, 12)
SPREAD = np.array([-0.05, 0.03, -0.02, 0.04, -0.01, 0.02, -0.03, 0.01, 0.0, 0.05, -0.04, 0.0])


def row(name, vals):
    vals = np.asarray(vals, float)
    d = vals - BASE
    return dict(name=name, nfe=1, files=list(FILES), rmse_pct=float(vals.mean()),
                rmse_pct_se=float(vals.std(ddof=1) / np.sqrt(12)), per_item={'rmse_pct': [list(vals), list(vals)]},
                paired_rmse_diff=float(d.mean()), paired_rmse_se=float(d.std(ddof=1) / np.sqrt(12)))


def yswap(true, swap):
    return {'t999': dict(true=float(np.mean(true)), swap_full=float(np.mean(swap))), 'passes': False,
            'per_cube': {'999': dict(true=list(true), swap_full=list(swap), swap_sensor_only=list(true), zero=list(swap))},
            'baseline_per_cube': list(BASE[:8]), 'files': FILES[:8]}


GOOD_YS = yswap(BASE[:8] - 0.15 + SPREAD[:8], BASE[:8] + 0.05 + 0.5 * SPREAD[:8])    # uses its condition, gain vanishes;
#                                                                                     spec 0.9 ratio fails (0.915)
TARGET_YS = yswap(BASE[:8] - 0.45 + SPREAD[:8], BASE[:8] + 0.05 + 0.5 * SPREAD[:8])  # the same at a target-size gain;
#                                                                                     spec 0.9 ratio passes (0.789)
BLIND_YS = yswap(BASE[:8] - 0.3 + SPREAD[:8], BASE[:8] - 0.3 + SPREAD[:8])           # swap changes nothing
STILL_YS = yswap(BASE[:8] - 0.3 + SPREAD[:8], BASE[:8] - 0.2 + SPREAD[:8])           # still beats the baseline swapped


def reg_eval(vals, ys, loss='l2', seed=42, ckpt='/r/regress_l2_d4_b64/checkpoint_epoch_100.pth', run='regress_l2_d4_b64',
             **prov):
    p = dict(mode='regress', d=4, base_channels=64, prediction_type='direct', prior_sha256='p' * 64, meta_checked=True,
             ckpt=ckpt, loss_type=loss, train_seed=seed)
    p.update(prov)
    return dict(provenance=p, grid='direct', run_name=run, rows=[row('direct', vals)], yswap=ys,
                baseline=dict(rmse_pct=float(BASE.mean())), baseline_files=list(FILES),
                baseline_per_item={'rmse_pct': list(BASE)})


def b_eval(vals, **prov):
    p = dict(mode='residual', d=4, base_channels=64, prediction_type='v', prior_sha256='p' * 64, meta_checked=True,
             ckpt='/r/residual_d4_b64/checkpoint_epoch_100.pth')
    p.update(prov)
    return dict(provenance=p, grid='full', run_name='residual_d4_b64', rows=[row('ddim1', vals)],
                baseline=dict(rmse_pct=float(BASE.mean())), baseline_files=list(FILES),
                baseline_per_item={'rmse_pct': list(BASE)})


B = b_eval(BASE - 0.1 + SPREAD)                                                       # B: 0.1 better than the baseline


def test_proposed_validity_three_cases():
    good = gp.proposed_validity(GOOD_YS)
    assert good['valid'] and good['t'] == 999 and good['n'] == 8 and not good['spec_rule_passes']
    assert good['invalid_reason'] is None and gp.proposed_validity(TARGET_YS)['spec_rule_passes']
    blind = gp.proposed_validity(BLIND_YS)
    assert not blind['valid'] and not blind['swap_full_minus_true']['pass'] and blind['invalid_reason'] == 'ab'
    still = gp.proposed_validity(STILL_YS)
    assert not still['valid'] and still['swap_full_minus_true']['pass'] and not still['swap_full_minus_baseline']['pass']
    assert still['invalid_reason'] == 'b'
    two_t = dict(GOOD_YS, per_cube={'100': GOOD_YS['per_cube']['999'], '300': GOOD_YS['per_cube']['999']})
    with pytest.raises(ValueError, match='single t'):
        gp.proposed_validity(two_t)


def test_proposed_validity_refuses_a_single_t_other_than_999():
    with pytest.raises(ValueError, match='t = 999'):
        gp.proposed_validity(dict(GOOD_YS, per_cube={'100': GOOD_YS['per_cube']['999']}))


def test_proposed_validity_part_b_tolerates_a_swap_within_k_se_below_the_baseline():
    near = yswap(BASE[:8] - 0.15 + SPREAD[:8], BASE[:8] - 0.005 + 0.5 * SPREAD[:8])
    v = gp.proposed_validity(near)
    b = v['swap_full_minus_baseline']
    assert b['diff'] < 0 <= b['diff'] + 2 * b['se'] and b['pass'] and v['valid']


def test_verdicts():
    m = gp.gate([reg_eval(BASE - 0.3 + SPREAD, GOOD_YS)], B, require_done=False)['models'][0]
    assert m['verdict'] == 'PASS' and m['invalid_reason'] is None
    m = gp.gate([reg_eval(BASE - 0.05 + 0.2 * SPREAD, GOOD_YS)], B, require_done=False)['models'][0]
    assert m['verdict'] == 'PASS_NOT_OVER_B' and m['vs_baseline']['pass'] and not m['vs_b_ddim1']['pass']
    assert gp.gate([reg_eval(BASE + 0.1 + SPREAD, GOOD_YS)], B, require_done=False)['models'][0]['verdict'] == 'FAIL'
    m = gp.gate([reg_eval(BASE - 0.3 + SPREAD, BLIND_YS)], B, require_done=False)['models'][0]
    assert m['verdict'] == 'INVALID' and m['verdict_if_valid'] == 'PASS' and m['invalid_reason'] == 'ab'
    m = gp.gate([reg_eval(BASE - 0.3 + SPREAD, STILL_YS)], B, require_done=False)['models'][0]
    assert m['verdict'] == 'INVALID' and m['invalid_reason'] == 'b'
    m = gp.gate([reg_eval(BASE - 0.4 + SPREAD, GOOD_YS)], B, require_done=False)['models'][0]
    assert m['target_met'] and m['target_rmse_pct'] == pytest.approx(0.85 * BASE.mean())


def test_pass_rule_is_diff_plus_k_iid_se():
    files = [f'MatFlower60/Train/P022/img{i // 2}_{i % 2}.mat' for i in range(12)]       # 6 images x 2 patches
    noise = np.repeat([0.03, -0.03, 0.02, -0.02, 0.01, -0.01], 2)                       # patches of an image move together

    def relabel(ev):
        ev = json.loads(json.dumps(ev))
        ev['baseline_files'] = list(files)
        for r in ev['rows']:
            r['files'] = list(files)
        return ev
    b = relabel(B)
    verdicts = {}
    for delta in (-0.016, -0.005):            # iid: -0.016 + 2 * 0.0065 < 0 (pass); the clustered SE 0.0097 would fail it
        m = gp.gate([relabel(reg_eval(BASE + delta + noise, GOOD_YS))], b, require_done=False)['models'][0]
        assert m['vs_baseline']['se_cluster'] > 1.3 * m['vs_baseline']['se']
        verdicts[delta] = m['vs_baseline']['pass']
    assert verdicts == {-0.016: True, -0.005: False}            # -0.005 < 0 but within 2 SE: a bare sign rule passes it


def test_b_valid_but_not_better_than_the_baseline_is_fail():
    g = gp.gate([reg_eval(BASE - 0.3 + SPREAD, GOOD_YS)], b_eval(BASE + 0.05 + SPREAD), {'yswap': GOOD_YS},
                require_done=False)
    assert g['b']['validity']['valid'] and g['b']['verdict_under_proposed_rule'] == 'FAIL'


def test_refusals():
    def good(**p):
        return reg_eval(BASE - 0.3 + SPREAD, GOOD_YS, **p)
    for bad in (dict(mode='residual'), dict(prediction_type='v'), dict(meta_checked=False), dict(prior_sha256='q' * 64),
                dict(d=2)):
        with pytest.raises(ValueError):
            gp.gate([good(**bad)], B, require_done=False)
    wrong_grid = good()
    wrong_grid['grid'] = 'full'
    with pytest.raises(ValueError, match='direct'):
        gp.gate([wrong_grid], B, require_done=False)
    shifted = good()
    shifted['baseline_per_item'] = {'rmse_pct': list(BASE + 0.01)}
    with pytest.raises(ValueError, match='baselines differ'):
        gp.gate([shifted], B, require_done=False)
    with pytest.raises(ValueError, match='different epochs'):
        gp.gate([good(), good(ckpt='/r/x/checkpoint_epoch_90.pth', loss_type='l1')], B, require_done=False)
    with pytest.raises(ValueError, match='full-grid residual'):
        gp.gate([good()], dict(B, grid='quick'), require_done=False)


def test_done_is_required(tmp_path):
    run, brun = tmp_path / 'regress_l2_d4_b64', tmp_path / 'residual_d4_b64'
    for d in (run, brun):
        d.mkdir()
        (d / 'DONE').write_text('100\n')
    b = b_eval(BASE - 0.1 + SPREAD, ckpt=str(brun / 'checkpoint_epoch_100.pth'))
    ok = reg_eval(BASE - 0.3 + SPREAD, GOOD_YS, ckpt=str(run / 'checkpoint_epoch_100.pth'))
    assert gp.gate([ok], b)['models'][0]['verdict'] == 'PASS'
    (run / 'DONE').write_text('90\n')
    with pytest.raises(ValueError, match='not the final checkpoint'):
        gp.gate([ok], b)


def test_loss_pair_seed_pair_and_best():
    l1 = reg_eval(BASE - 0.3 + SPREAD, GOOD_YS, loss='l1', run='regress_l1_d4_b64')
    l2 = reg_eval(BASE - 0.4 + SPREAD, TARGET_YS, loss='l2', run='regress_l2_d4_b64')
    g = gp.gate([l1, l2], B, require_done=False)
    assert g['pairs'][0]['kind'] == 'loss_pair' and g['pairs'][0]['a_minus_b']['diff'] == pytest.approx(0.1)
    assert g['best'] == 'regress_l2_d4_b64' and g['best_verdict'] == 'PASS' and g['go_1b'] is False
    s43 = reg_eval(BASE - 0.45 + SPREAD, TARGET_YS, loss='l2', seed=43, run='regress_l2_s43_d4_b64')
    g = gp.gate([l2, s43], B, require_done=False)
    assert g['pairs'][0]['kind'] == 'seed_pair' and g['pairs'][0]['go_1b'] is True and g['go_1b'] is True
    low_ratio = [reg_eval(BASE - 0.4 + SPREAD, GOOD_YS, loss='l2', run='regress_l2_d4_b64'),
                 reg_eval(BASE - 0.45 + SPREAD, GOOD_YS, loss='l2', seed=43, run='regress_l2_s43_d4_b64')]
    g = gp.gate(low_ratio, B, require_done=False)       # PASS and target met, but the spec 0.9 ratio fails at t = 999
    assert [m['verdict'] for m in g['models']] == ['PASS', 'PASS'] and all(m['target_met'] for m in g['models'])
    assert g['go_1b'] is False
    blind_l2 = reg_eval(BASE - 0.4 + SPREAD, BLIND_YS, loss='l2', run='regress_l2_d4_b64')
    g = gp.gate([l1, blind_l2], B, require_done=False)  # the lower-RMSE L2 is INVALID: best is the passing L1
    assert g['best'] == 'regress_l1_d4_b64' and g['best_verdict'] == 'PASS'


def test_seed_pair_needs_both_targets_met():
    l2 = reg_eval(BASE - 0.4 + SPREAD, TARGET_YS, loss='l2', run='regress_l2_d4_b64')
    s43 = reg_eval(BASE - 0.3 + SPREAD, TARGET_YS, loss='l2', seed=43, run='regress_l2_s43_d4_b64')   # 2.2 > 2.125
    g = gp.gate([l2, s43], B, require_done=False)
    assert [m['verdict'] for m in g['models']] == ['PASS', 'PASS'] and not g['models'][1]['target_met']
    assert g['pairs'][0]['kind'] == 'seed_pair' and g['pairs'][0]['go_1b'] is False and g['go_1b'] is False


def test_vs_b_t999_kmean_is_reported_but_not_binding():
    diag = {'rows': {'t999': {'K': 8, 'per_cube': {'files': list(FILES), 'kmean_rmse': list(BASE - 0.35 + SPREAD)}}}}
    reg = reg_eval(BASE - 0.3 + SPREAD, GOOD_YS)
    m = gp.gate([reg], B, require_done=False, b_diag=diag)['models'][0]
    assert m['verdict'] == 'PASS' and m['vs_b_t999_kmean']['nfe'] == 8          # B's 8-draw mean is 0.05 better
    assert m['vs_b_t999_kmean']['diff'] == pytest.approx(0.05) and not m['vs_b_t999_kmean']['pass']
    assert gp.gate([reg], B, require_done=False)['models'][0]['vs_b_t999_kmean'] is None
    diag['rows']['t999']['per_cube']['kmean_rmse'] = list(BASE - 0.3 + 2 * SPREAD + 0.002)    # R 0.002 better, within 2 SE
    vk = gp.gate([reg], B, require_done=False, b_diag=diag)['models'][0]['vs_b_t999_kmean']
    assert vk['diff'] == pytest.approx(-0.002) and vk['diff'] + 2 * vk['se'] > 0 and not vk['pass']
    diag['rows']['t999']['per_cube']['files'] = FILES[:11] + ['MatFlower60/Train/P022/other_1.mat']
    with pytest.raises(ValueError, match='different cubes'):
        gp.gate([reg], B, require_done=False, b_diag=diag)


def test_main_writes_the_named_record(tmp_path):
    pr, pb, pys, pd = tmp_path / 'reg.json', tmp_path / 'b.json', tmp_path / 'bys.json', tmp_path / 'diag.json'
    pr.write_text(json.dumps(reg_eval(BASE - 0.3 + SPREAD, GOOD_YS)))
    pb.write_text(json.dumps(B))
    pys.write_text(json.dumps({'yswap': GOOD_YS}))
    pd.write_text(json.dumps({'rows': {'t999': {'K': 8, 'per_cube': {'files': list(FILES),
                                                                     'kmean_rmse': list(BASE - 0.2 + SPREAD)}}}}))
    g = gp.main(['--reg', str(pr), '--b', str(pb), '--no_require_done'])
    out = tmp_path / 'gate_1a_prime__regress_l2_d4_b64__nodone.json'
    assert out.exists() and json.loads(out.read_text())['models'][0]['verdict'] == 'PASS'
    assert g['inputs']['b'] == str(pb) and g['b']['verdict_under_proposed_rule'] is None
    g = gp.main(['--reg', str(pr), '--b', str(pb), '--b_yswap', str(pys), '--b_diag', str(pd), '--no_require_done',
                 '--out', str(tmp_path / 'x.json')])
    assert g['b']['verdict_under_proposed_rule'] == 'PASS' and (tmp_path / 'x.json').exists()
    assert g['inputs']['b_diag'] == str(pd) and g['models'][0]['vs_b_t999_kmean']['pass']
