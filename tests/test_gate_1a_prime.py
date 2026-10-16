"""Phase 1a-prime gate (Task 4): proposed one-step validity rule, verdicts, refusals, loss and seed pairs, best, the
report-only comparison with B's K-draw mean, CLI; fix round 1: the k-SE margins and the iid SE of every rule, the pair
kinds, go_1b's PASS condition, the spec ratio's value, the input refusals and the B-record ownership checks."""
import json

import numpy as np
import pytest

from scripts import gate_1a_prime as gp

FILES = [f'MatFlower60/Train/P0{22 + i % 3}/img{i // 2}_{i % 2}.mat' for i in range(12)]   # 12 images, 1 cube each
BASE = np.linspace(2.0, 3.0, 12)
SPREAD = np.array([-0.05, 0.03, -0.02, 0.04, -0.01, 0.02, -0.03, 0.01, 0.0, 0.05, -0.04, 0.0])
CLUSTERED = [f'MatFlower60/Train/P022/img{i // 2}_{i % 2}.mat' for i in range(12)]          # 6 images x 2 patches
IMG_NOISE = np.repeat([0.03, -0.03, 0.02, -0.02, 0.01, -0.01], 2)                          # patches of an image move together
B_CKPT = '/r/residual_d4_b64/checkpoint_epoch_100.pth'


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
STILL_RATIO_YS = yswap(BASE[:8] - 0.5 + SPREAD[:8], BASE[:8] - 0.2 + SPREAD[:8])     # the same (b fails), yet the spec
#                                                                                     0.9 ratio passes (0.858)
RATIO_YS = yswap(BASE[:8] - 0.1 + SPREAD[:8], BASE[:8] + 0.3 + 0.5 * SPREAD[:8])     # true / swap_full 0.847 <= 0.9 <
#                                                                                     true / baseline 0.956


def clustered_yswap(true, swap):
    return dict(yswap(true, swap), files=CLUSTERED[:8])


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
             ckpt=B_CKPT)
    p.update(prov)
    return dict(provenance=p, grid='full', run_name='residual_d4_b64', rows=[row('ddim1', vals)],
                baseline=dict(rmse_pct=float(BASE.mean())), baseline_files=list(FILES),
                baseline_per_item={'rmse_pct': list(BASE)})


def b_yswap_rec(ys=GOOD_YS, **prov):
    """B's y-swap record as scripts/yswap_one_step.py writes it: the provenance the gate checks plus the y-swap."""
    p = dict(checkpoint=B_CKPT, checkpoint_sha256='s' * 64, mode='residual', d=4, base_channels=64, prediction_type='v',
             prior_sha256='p' * 64, meta_checked=True)
    p.update(prov)
    return dict(provenance=p, yswap=ys)


def diag_rec(kmean, files=FILES, b_eval_path='/e/residual_d4_b64__checkpoint_epoch_100.json'):
    """b_diagnostics.json as scripts/diagnose_b.py merge writes it (the parts the gate reads)."""
    return {'provenance': {'b_eval': b_eval_path},
            'rows': {'t999': {'K': 8, 'per_cube': {'files': list(files), 'kmean_rmse': list(kmean)}}}}


def relabel(ev, files=CLUSTERED):
    """The same evaluation with cube i renamed files[i] (rows, baseline and, the first 8, its y-swap's cubes)."""
    ev = json.loads(json.dumps(ev))
    ev['baseline_files'] = list(files)
    for r in ev['rows']:
        r['files'] = list(files)
    if 'yswap' in ev:
        ev['yswap']['files'] = list(files[:len(ev['yswap']['files'])])
    return ev


B = b_eval(BASE - 0.1 + SPREAD)                                                       # B: 0.1 better than the baseline


def test_proposed_validity_three_cases():
    good = gp.proposed_validity(GOOD_YS)
    assert good['valid'] and good['t'] == 999 and good['n'] == 8 and not good['spec_rule_passes']
    assert good['spec_ratio'] == pytest.approx(GOOD_YS['t999']['true'] / GOOD_YS['t999']['swap_full'])
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


def test_proposed_validity_part_a_needs_more_than_k_se():
    swap = BASE[:8] + 0.05 + 0.5 * SPREAD[:8]
    weak = yswap(swap - 0.018 - SPREAD[:8], swap)          # swap_full - true = 0.018 + noise: 1 to 2 SE above 0
    v = gp.proposed_validity(weak)
    a = v['swap_full_minus_true']
    assert a['se'] < a['diff'] < 2 * a['se'] and not a['pass'] and v['swap_full_minus_baseline']['pass']
    assert not v['valid'] and v['invalid_reason'] == 'a'
    m = gp.gate([reg_eval(BASE - 0.3 + SPREAD, weak)], B, require_done=False)['models'][0]
    assert m['verdict'] == 'INVALID' and m['verdict_if_valid'] == 'PASS' and m['invalid_reason'] == 'a'


def test_proposed_validity_uses_the_iid_se_in_both_parts():
    base, noise = BASE[:8], IMG_NOISE[:8]                 # 4 images x 2 patches: the CR1 SE is 1.5 x the iid SE
    v = gp.proposed_validity(clustered_yswap(base + 0.05 - 0.025 - noise, base + 0.05))
    a = v['swap_full_minus_true']                          # 0.025 + noise: above 2 iid SE, not above 2 CR1 SE
    assert a['diff'] - 2 * a['se'] > 0 >= a['diff'] - 2 * a['se_cluster'] and a['pass'] and v['valid']
    v = gp.proposed_validity(clustered_yswap(base - 0.025 + noise - 0.3, base - 0.025 + noise))
    b = v['swap_full_minus_baseline']                      # -0.025 + noise: below -2 iid SE, not below -2 CR1 SE
    assert b['diff'] + 2 * b['se'] < 0 <= b['diff'] + 2 * b['se_cluster'] and not b['pass']
    assert v['swap_full_minus_true']['pass'] and v['invalid_reason'] == 'b'


def test_spec_ratio_is_true_over_swap_full():
    v = gp.proposed_validity(RATIO_YS)
    assert v['spec_ratio'] == pytest.approx(RATIO_YS['t999']['true'] / RATIO_YS['t999']['swap_full'])
    assert v['spec_ratio'] <= 0.9 < RATIO_YS['t999']['true'] / np.mean(RATIO_YS['baseline_per_cube'])
    assert v['spec_rule_passes'] and v['spec_threshold'] == 0.9


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
    b = relabel(B)
    verdicts = {}
    for delta in (-0.016, -0.005):            # iid: -0.016 + 2 * 0.0065 < 0 (pass); the clustered SE 0.0097 would fail it
        m = gp.gate([relabel(reg_eval(BASE + delta + IMG_NOISE, GOOD_YS))], b, require_done=False)['models'][0]
        assert m['vs_baseline']['se_cluster'] > 1.3 * m['vs_baseline']['se']
        verdicts[delta] = m['vs_baseline']['pass']
    assert verdicts == {-0.016: True, -0.005: False}            # -0.005 < 0 but within 2 SE: a bare sign rule passes it


def test_vs_b_ddim1_needs_more_than_k_iid_se():
    m = gp.gate([reg_eval(BASE - 0.107 + SPREAD + 0.5 * SPREAD[::-1], GOOD_YS)], B, require_done=False)['models'][0]
    vb = m['vs_b_ddim1']                                   # R - B = -0.007 + noise: 1 to 2 SE below 0
    assert vb['diff'] == pytest.approx(-0.007) and vb['diff'] + vb['se'] < 0 < vb['diff'] + 2 * vb['se']
    assert m['vs_baseline']['pass'] and not vb['pass'] and m['verdict'] == 'PASS_NOT_OVER_B'
    m = gp.gate([relabel(reg_eval(BASE - 0.116 + SPREAD + IMG_NOISE, GOOD_YS))], relabel(B),
                require_done=False)['models'][0]
    vb = m['vs_b_ddim1']                                   # R - B = -0.016 + image noise: iid pass, CR1 would fail
    assert vb['diff'] + 2 * vb['se'] < 0 < vb['diff'] + 2 * vb['se_cluster'] and vb['pass'] and m['verdict'] == 'PASS'


def test_b_valid_but_not_better_than_the_baseline_is_fail():
    g = gp.gate([reg_eval(BASE - 0.3 + SPREAD, GOOD_YS)], b_eval(BASE + 0.05 + SPREAD), b_yswap_rec(),
                require_done=False)
    assert g['b']['validity']['valid'] and g['b']['verdict_under_proposed_rule'] == 'FAIL'


def test_refusals():
    def good(**p):
        return reg_eval(BASE - 0.3 + SPREAD, GOOD_YS, **p)
    for bad, msg in ((dict(mode='residual'), "--reg must be a direct-grid regress evaluation, got mode='residual'"),
                     (dict(prediction_type='v'), "--reg must be a direct-grid .* prediction_type='v'"),
                     (dict(meta_checked=False), r'regress_l2_d4_b64/checkpoint_epoch_100\.pth: checkpoint meta was not'),
                     (dict(prior_sha256='q' * 64), r'differ in provenance\.prior_sha256'),
                     (dict(d=2), r'differ in provenance\.d:'),
                     (dict(base_channels=256), r'differ in provenance\.base_channels')):
        with pytest.raises(ValueError, match=msg):
            gp.gate([good(**bad)], B, require_done=False)
    wrong_grid = good()
    wrong_grid['grid'] = 'full'
    with pytest.raises(ValueError, match='direct'):
        gp.gate([wrong_grid], B, require_done=False)
    shifted = good()
    shifted['baseline_per_item'] = {'rmse_pct': list(BASE + 0.01)}
    with pytest.raises(ValueError, match='baselines differ'):
        gp.gate([shifted], B, require_done=False)
    subset = good()                                        # the same values on 11 of B's 12 cubes
    subset['baseline_files'], subset['baseline_per_item'] = FILES[:11], {'rmse_pct': list(BASE[:11])}
    with pytest.raises(ValueError, match="per-cube baselines differ from B's"):
        gp.gate([subset], B, require_done=False)
    with pytest.raises(ValueError, match='different epochs'):
        gp.gate([good(), good(ckpt='/r/x/checkpoint_epoch_90.pth', loss_type='l1')], B, require_done=False)
    with pytest.raises(ValueError, match='full-grid residual'):
        gp.gate([good()], dict(B, grid='quick'), require_done=False)
    with pytest.raises(ValueError, match="full-grid residual evaluation, got mode='standard'"):
        gp.gate([good()], b_eval(BASE - 0.1 + SPREAD, mode='standard'), require_done=False)
    with pytest.raises(ValueError, match=r'residual_d4_b64/checkpoint_epoch_100\.pth: checkpoint meta was not checked'):
        gp.gate([good()], b_eval(BASE - 0.1 + SPREAD, meta_checked=False), require_done=False)
    on_11 = dict(row('direct', BASE), files=FILES[:11], per_item={'rmse_pct': [list(BASE[:11])] * 2})
    with pytest.raises(ValueError, match='rows score different cubes'):
        gp.paired_by_file(on_11, row('ddim1', BASE))


def test_done_is_required(tmp_path):
    run, brun = tmp_path / 'regress_l2_d4_b64', tmp_path / 'residual_d4_b64'
    for d in (run, brun):
        d.mkdir()
        (d / 'DONE').write_text('100\n')
    b = b_eval(BASE - 0.1 + SPREAD, ckpt=str(brun / 'checkpoint_epoch_100.pth'))
    ok = reg_eval(BASE - 0.3 + SPREAD, GOOD_YS, ckpt=str(run / 'checkpoint_epoch_100.pth'))
    assert gp.gate([ok], b)['models'][0]['verdict'] == 'PASS'
    (run / 'DONE').write_text('90\n')
    with pytest.raises(ValueError, match=r'regress_l2_d4_b64/checkpoint_epoch_100\.pth is not the final checkpoint'):
        gp.gate([ok], b)
    (run / 'DONE').write_text('100\n')
    (brun / 'DONE').write_text('90\n')
    with pytest.raises(ValueError, match=r'residual_d4_b64/checkpoint_epoch_100\.pth is not the final checkpoint'):
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


def test_pair_kinds_need_the_same_seed_or_the_same_loss():
    good, bad = BASE - 0.45 + SPREAD, BASE + 0.1 + SPREAD   # only the mixed pair (L1 s42, L2 s43) has two qualifiers
    regs = [reg_eval(good, TARGET_YS, loss='l1', seed=42, run='regress_l1_d4_b64'),
            reg_eval(bad, TARGET_YS, loss='l2', seed=42, run='regress_l2_d4_b64'),
            reg_eval(bad, TARGET_YS, loss='l1', seed=43, run='regress_l1_s43_d4_b64'),
            reg_eval(good, TARGET_YS, loss='l2', seed=43, run='regress_l2_s43_d4_b64')]
    g = gp.gate(regs, B, require_done=False)
    assert [m['verdict'] for m in g['models']] == ['PASS', 'FAIL', 'FAIL', 'PASS']
    assert [(p['a'], p['b']) for p in g['pairs']][2] == ('regress_l1_d4_b64', 'regress_l2_s43_d4_b64')
    assert [p['kind'] for p in g['pairs']] == ['loss_pair', 'seed_pair', 'other', 'other', 'seed_pair', 'loss_pair']
    assert [p.get('go_1b') for p in g['pairs']] == [None, False, None, None, False, None] and g['go_1b'] is False


def test_seed_pair_needs_both_targets_met():
    l2 = reg_eval(BASE - 0.4 + SPREAD, TARGET_YS, loss='l2', run='regress_l2_d4_b64')
    s43 = reg_eval(BASE - 0.3 + SPREAD, TARGET_YS, loss='l2', seed=43, run='regress_l2_s43_d4_b64')   # 2.2 > 2.125
    g = gp.gate([l2, s43], B, require_done=False)
    assert [m['verdict'] for m in g['models']] == ['PASS', 'PASS'] and not g['models'][1]['target_met']
    assert g['pairs'][0]['kind'] == 'seed_pair' and g['pairs'][0]['go_1b'] is False and g['go_1b'] is False


def test_go_1b_needs_both_models_pass():
    s43 = reg_eval(BASE - 0.45 + SPREAD, TARGET_YS, loss='l2', seed=43, run='regress_l2_s43_d4_b64')
    invalid = reg_eval(BASE - 0.45 + SPREAD, STILL_RATIO_YS, loss='l2', run='regress_l2_d4_b64')
    g = gp.gate([invalid, s43], B, require_done=False)
    m = g['models'][0]                                     # INVALID ('b') though target met and the spec ratio passes
    assert m['verdict'] == 'INVALID' and m['invalid_reason'] == 'b' and m['verdict_if_valid'] == 'PASS'
    assert m['target_met'] and m['validity']['spec_rule_passes'] and g['models'][1]['verdict'] == 'PASS'
    assert g['pairs'][0]['kind'] == 'seed_pair' and g['pairs'][0]['go_1b'] is False and g['go_1b'] is False
    b_low = b_eval(BASE - 0.45 + SPREAD)                   # a B as good as the first regressor
    pair = [reg_eval(BASE - 0.45 + SPREAD, TARGET_YS, loss='l2', run='regress_l2_d4_b64'),
            reg_eval(BASE - 0.5 + SPREAD, TARGET_YS, loss='l2', seed=43, run='regress_l2_s43_d4_b64')]
    g = gp.gate(pair, b_low, require_done=False)
    assert [m['verdict'] for m in g['models']] == ['PASS_NOT_OVER_B', 'PASS'] and all(m['target_met'] for m in g['models'])
    assert all(m['validity']['spec_rule_passes'] for m in g['models']) and g['go_1b'] is False


def test_vs_b_t999_kmean_is_reported_but_not_binding():
    diag = diag_rec(BASE - 0.35 + SPREAD)
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


def test_b_yswap_must_be_b_s_own():
    reg = reg_eval(BASE - 0.3 + SPREAD, GOOD_YS)
    assert gp.gate([reg], B, b_yswap_rec(), require_done=False)['b']['validity']['valid']
    respelled = b_yswap_rec(checkpoint='/r/./residual_d4_b64/checkpoint_epoch_100.pth')      # the same file
    assert gp.gate([reg], B, respelled, require_done=False)['b']['validity']['valid']
    rounding = dict(GOOD_YS, baseline_per_cube=list(BASE[:8] + 3e-6))      # float32 device rounding of the baseline
    assert gp.gate([reg], B, b_yswap_rec(rounding), require_done=False)['b']['validity']['valid']
    for bad, field in ((dict(mode='regress'), 'mode'), (dict(checkpoint='/r/residual_d4_b64/checkpoint_epoch_90.pth'),
                                                        'checkpoint'),
                       (dict(d=2), 'd'), (dict(prior_sha256='q' * 64), 'prior_sha256'),
                       (dict(meta_checked=False), 'meta_checked')):
        with pytest.raises(ValueError, match=rf"--b_yswap is not B's y-swap: provenance\.{field} is"):
            gp.gate([reg], B, b_yswap_rec(**bad), require_done=False)
    with pytest.raises(ValueError, match="--b_yswap record has no 'provenance' key"):
        gp.gate([reg], B, {'yswap': GOOD_YS}, require_done=False)
    for ys in (dict(GOOD_YS, baseline_per_cube=list(BASE[:8] + 0.01)),                       # another prior
               dict(GOOD_YS, files=FILES[1:9]),                                               # other cubes of B's
               dict(GOOD_YS, files=FILES[:7] + ['MatFlower60/Train/P022/other_1.mat'])):      # a cube outside them
        with pytest.raises(ValueError, match="--b_yswap: its per-cube baselines differ from its evaluation's"):
            gp.gate([reg], B, b_yswap_rec(ys), require_done=False)


def test_regressor_yswap_baseline_must_be_its_evaluations():
    for ys in (dict(GOOD_YS, baseline_per_cube=list(BASE[:8] + 0.01)), dict(GOOD_YS, files=FILES[1:9]),
               dict(GOOD_YS, files=FILES[:7] + ['MatFlower60/Train/P022/other_1.mat'])):
        with pytest.raises(ValueError, match="regressor's y-swap: its per-cube baselines differ from its evaluation's"):
            gp.gate([reg_eval(BASE - 0.3 + SPREAD, ys)], B, require_done=False)


def test_b_diag_must_name_the_b_evaluation():
    reg, diag = reg_eval(BASE - 0.3 + SPREAD, GOOD_YS), diag_rec(BASE - 0.35 + SPREAD, b_eval_path='/e/b.json')
    assert gp.gate([reg], B, require_done=False, b_diag=diag, b_path='/e/./b.json')['models'][0]['vs_b_t999_kmean']
    with pytest.raises(ValueError, match=r'--b_diag was made from /e/b\.json, not from --b /e/other\.json'):
        gp.gate([reg], B, require_done=False, b_diag=diag, b_path='/e/other.json')
    with pytest.raises(ValueError, match="b_diagnostics provenance has no 'b_eval' key"):
        gp.gate([reg], B, require_done=False, b_diag=dict(diag, provenance={}))
    with pytest.raises(ValueError, match="b_diagnostics has no 'provenance' key"):
        gp.gate([reg], B, require_done=False, b_diag={'rows': diag['rows']})


def test_main_writes_the_named_record(tmp_path):
    pr, pb, pys, pd = tmp_path / 'reg.json', tmp_path / 'b.json', tmp_path / 'bys.json', tmp_path / 'diag.json'
    pr.write_text(json.dumps(reg_eval(BASE - 0.3 + SPREAD, GOOD_YS)))
    pb.write_text(json.dumps(B))
    pys.write_text(json.dumps(b_yswap_rec()))
    pd.write_text(json.dumps(diag_rec(BASE - 0.2 + SPREAD, b_eval_path=str(pb))))
    g = gp.main(['--reg', str(pr), '--b', str(pb), '--no_require_done'])
    out = tmp_path / 'gate_1a_prime__regress_l2_d4_b64__nodone.json'
    assert out.exists() and json.loads(out.read_text())['models'][0]['verdict'] == 'PASS'
    assert g['inputs']['b'] == str(pb) and g['b']['verdict_under_proposed_rule'] is None
    g = gp.main(['--reg', str(pr), '--b', str(pb), '--b_yswap', str(pys), '--b_diag', str(pd), '--no_require_done',
                 '--out', str(tmp_path / 'x.json')])
    assert g['b']['verdict_under_proposed_rule'] == 'PASS' and (tmp_path / 'x.json').exists()
    assert g['inputs']['b_diag'] == str(pd) and g['models'][0]['vs_b_t999_kmean']['pass']
    other = tmp_path / 'b_copy.json'                       # B's content under another name: the diagnostics name pb
    other.write_text(pb.read_text())
    with pytest.raises(ValueError, match='--b_diag was made from'):
        gp.main(['--reg', str(pr), '--b', str(other), '--b_diag', str(pd), '--no_require_done',
                 '--out', str(tmp_path / 'y.json')])
    assert not (tmp_path / 'y.json').exists()
