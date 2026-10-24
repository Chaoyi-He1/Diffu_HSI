"""Dose-response d = 8 (d8_main) vs d = 4 (d4_contrast): absolute and relative CRPS / AUSE gaps, coverage, texture gaps,
cube-level heteroscedasticity slope; S1 1/K debiasing; P2 feasibility. CR1 cluster SEs (own implementation), delta
method for ratios, and a joint 5000-draw cluster bootstrap over the 113 source images (same 200 cubes at both strides)."""
import json
import math

import numpy as np
from scipy.stats import norm

R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart/uq/'
OUT = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d4/recompute-d4/'
rec = {4: json.load(open(R + 'd4/eval/d4_contrast.json')), 8: json.load(open(R + 'd8/eval/d8_main.json')),
       '8k64': json.load(open(R + 'd8/eval/d8_fix_k64.json'))}
files = rec[4]['files']
assert files == rec[8]['files'] == rec['8k64']['files']


def key(f):
    parts = f.split('/')
    return parts[-2] + '/' + parts[-1][:-4].rsplit('_', 1)[0]


keys = [key(f) for f in files]
uk = sorted(set(keys))
cid = np.array([uk.index(k) for k in keys])
G, n = len(uk), len(files)


def cse(d):
    d = np.asarray(d, dtype=np.float64)
    s = np.bincount(cid, weights=d - d.mean(), minlength=G)
    return float(math.sqrt(G / (G - 1) * np.sum(s ** 2)) / n)


def st(d):
    d = np.asarray(d, dtype=np.float64)
    return dict(mean=float(d.mean()), se=float(d.std(ddof=1) / math.sqrt(n)), se_cluster=cse(d))


def A(s, m, k):
    return np.asarray(rec[s]['per_cube'][m][k], dtype=np.float64)


def ratio_gap(a, b):
    """rel = mean(a)/mean(b) - 1 with delta-method CR1 SE; also returns the linearised per-cube z (for paired diffs)."""
    ra = a.mean() / b.mean()
    z = (a - ra * b) / b.mean()
    return dict(rel=float(ra - 1), se_cluster=cse(z)), z


# bootstrap weights
rng = np.random.default_rng(0)
NB = 5000
counts = rng.multinomial(G, np.full(G, 1.0 / G), size=NB)       # [NB, G]
W = counts[:, cid].astype(np.float64)                             # [NB, n]
Wsum = W.sum(1)


def bm(a):
    return (W @ np.asarray(a, dtype=np.float64)) / Wsum


def bsum(vals):
    v = np.asarray(vals)
    return dict(se=float(v.std(ddof=1)), lo=float(np.quantile(v, 0.025)), hi=float(np.quantile(v, 0.975)))


def bslope(x, y):
    mx, my = bm(x), bm(y)
    cxy = bm(x * y) - mx * my
    vx = bm(x * x) - mx * mx
    return cxy / vx


res = {}
boot = {}
for s in (8, 4, '8k64'):
    r = {}
    cd, cv = A(s, 'diffusion', 'crps'), A(s, 'varhead', 'crps')
    r['crps_diff'] = st(cd)
    r['crps_vh'] = st(cv)
    r['crps_gap_abs'] = st(cd - cv)
    rg, z = ratio_gap(cd, cv)
    r['crps_gap_rel'] = rg
    r['crps_gap_rel_percube'] = st((cd - cv) / cv)
    r['_z_crps'] = z
    if s == '8k64':
        res[s] = r
        continue
    ad, av = A(s, 'diffusion', 'ause'), A(s, 'varhead', 'ause')
    r['ause_diff'] = st(ad)
    r['ause_vh'] = st(av)
    r['ause_gap_abs'] = st(ad - av)
    rg, z = ratio_gap(ad, av)
    r['ause_gap_rel'] = rg
    r['_z_ause'] = z
    for m in ('diffusion', 'varhead'):
        for c in ('cov50', 'cov90'):
            r[f'{c}_{m}'] = st(A(s, m, c))
    r['cov50_gauss_diffusion'] = st(A(s, 'diffusion', 'cov50_gauss'))
    r['cov90_gauss_diffusion'] = st(A(s, 'diffusion', 'cov90_gauss'))
    r['spread_ratio_diffusion'] = st(A(s, 'diffusion', 'spread_ratio'))
    r['spread_ratio_varhead'] = st(A(s, 'varhead', 'spread_ratio'))
    # texture
    hs, hr, hm = A(s, 'diffusion', 'single_hf'), A(s, 'regressor', 'hf'), A(s, 'diffusion', 'mean_hf')
    ws, wr, wm = A(s, 'diffusion', 'single_w1'), A(s, 'regressor', 'w1'), A(s, 'diffusion', 'mean_w1')
    r['hf_single'] = st(hs)
    r['hf_regressor'] = st(hr)
    r['hf_meanK'] = st(hm)
    r['P1_abs'] = st(np.abs(hs) - np.abs(hr))
    rg, z = ratio_gap(np.abs(hs), np.abs(hr))
    r['P1_rel'] = rg
    r['_z_p1'] = z
    r['hf_signed_gap'] = st(hs - hr)
    r['w1_single'] = st(ws)
    r['w1_regressor'] = st(wr)
    r['w1_meanK'] = st(wm)
    r['w1_gap_abs'] = st(ws - wr)
    rg, z = ratio_gap(ws, wr)
    r['w1_gap_rel'] = rg
    r['_z_w1'] = z
    r['single_rmse'] = st(A(s, 'diffusion', 'single_rmse_pct'))
    r['regressor_rmse'] = st(A(s, 'regressor', 'rmse_pct'))
    r['baseline_rmse'] = st(A(s, 'baseline', 'rmse_pct'))
    r['single_over_reg_rmse'] = float(r['single_rmse']['mean'] / r['regressor_rmse']['mean'])
    # heteroscedasticity: cube-level slope of log spread on log RMSE of the method's own mean
    sl = {}
    for m in ('diffusion', 'varhead'):
        rm = A(s, m, 'mean_rmse_pct') / 100
        sp = np.sqrt(A(s, m, 'spread_ratio')) * rm
        x, y = np.log(rm), np.log(sp)
        sl[m] = float(np.polyfit(x, y, 1)[0])
        boot[(s, 'slope_' + m)] = bslope(x, y)
        sl[m + '_spread_cv'] = float(sp.std() / sp.mean())
        sl[m + '_err_cv'] = float(rm.std() / rm.mean())
        sl[m + '_spread_rms_pct'] = float(100 * np.sqrt(np.mean(sp ** 2)))
    r['slope'] = sl
    # bootstrap draws
    boot[(s, 'crps_abs')] = bm(cd) - bm(cv)
    boot[(s, 'crps_rel')] = bm(cd) / bm(cv) - 1
    boot[(s, 'ause_abs')] = bm(ad) - bm(av)
    boot[(s, 'ause_rel')] = bm(ad) / bm(av) - 1
    boot[(s, 'p1_abs')] = bm(np.abs(hs) - np.abs(hr))
    boot[(s, 'p1_rel')] = bm(np.abs(hs)) / bm(np.abs(hr)) - 1
    boot[(s, 'w1_abs')] = bm(ws - wr)
    boot[(s, 'w1_rel')] = bm(ws) / bm(wr) - 1
    boot[(s, 'cov50')] = bm(A(s, 'diffusion', 'cov50'))
    boot[(s, 'cov90')] = bm(A(s, 'diffusion', 'cov90'))
    boot[(s, 'hf_single')] = bm(hs)
    res[s] = r

# paired d4 - d8 differences (same cubes)
diff = {}
for nm, k4, k8 in (('crps_gap_abs', None, None), ('ause_gap_abs', None, None), ('P1_abs', None, None),
                   ('w1_gap_abs', None, None)):
    pass
cd4, cv4, cd8, cv8 = A(4, 'diffusion', 'crps'), A(4, 'varhead', 'crps'), A(8, 'diffusion', 'crps'), A(8, 'varhead', 'crps')
diff['crps_gap_abs_d4_minus_d8'] = st((cd4 - cv4) - (cd8 - cv8))
diff['crps_gap_rel_d4_minus_d8'] = dict(mean=res[4]['crps_gap_rel']['rel'] - res[8]['crps_gap_rel']['rel'],
                                        se_cluster_delta=cse(res[4]['_z_crps'] - res[8]['_z_crps']))
diff['crps_gap_rel_percube_d4_minus_d8'] = st((cd4 - cv4) / cv4 - (cd8 - cv8) / cv8)
ad4, av4, ad8, av8 = A(4, 'diffusion', 'ause'), A(4, 'varhead', 'ause'), A(8, 'diffusion', 'ause'), A(8, 'varhead', 'ause')
diff['ause_gap_abs_d4_minus_d8'] = st((ad4 - av4) - (ad8 - av8))
diff['ause_gap_rel_d4_minus_d8'] = dict(mean=res[4]['ause_gap_rel']['rel'] - res[8]['ause_gap_rel']['rel'],
                                        se_cluster_delta=cse(res[4]['_z_ause'] - res[8]['_z_ause']))
diff['cov50_diff_d4_minus_d8'] = st(A(4, 'diffusion', 'cov50') - A(8, 'diffusion', 'cov50'))
diff['cov90_diff_d4_minus_d8'] = st(A(4, 'diffusion', 'cov90') - A(8, 'diffusion', 'cov90'))
diff['cov50_excess_d4_minus_d8_vs_varhead'] = st((A(4, 'diffusion', 'cov50') - A(4, 'varhead', 'cov50')) -
                                                 (A(8, 'diffusion', 'cov50') - A(8, 'varhead', 'cov50')))
diff['P1_abs_d4_minus_d8'] = st((np.abs(A(4, 'diffusion', 'single_hf')) - np.abs(A(4, 'regressor', 'hf'))) -
                                (np.abs(A(8, 'diffusion', 'single_hf')) - np.abs(A(8, 'regressor', 'hf'))))
diff['P1_rel_d4_minus_d8'] = dict(mean=res[4]['P1_rel']['rel'] - res[8]['P1_rel']['rel'],
                                  se_cluster_delta=cse(res[4]['_z_p1'] - res[8]['_z_p1']))
diff['w1_gap_abs_d4_minus_d8'] = st((A(4, 'diffusion', 'single_w1') - A(4, 'regressor', 'w1')) -
                                    (A(8, 'diffusion', 'single_w1') - A(8, 'regressor', 'w1')))
diff['w1_gap_rel_d4_minus_d8'] = dict(mean=res[4]['w1_gap_rel']['rel'] - res[8]['w1_gap_rel']['rel'],
                                      se_cluster_delta=cse(res[4]['_z_w1'] - res[8]['_z_w1']))
diff['hf_single_d4_minus_d8'] = st(A(4, 'diffusion', 'single_hf') - A(8, 'diffusion', 'single_hf'))
diff['abs_hf_regressor_d4_minus_d8'] = st(np.abs(A(4, 'regressor', 'hf')) - np.abs(A(8, 'regressor', 'hf')))
diff['crps_gap_rel_k64_minus_k32_d8'] = dict(mean=res['8k64']['crps_gap_rel']['rel'] - res[8]['crps_gap_rel']['rel'],
                                             se_cluster_delta=cse(res['8k64']['_z_crps'] - res[8]['_z_crps']))
# bootstrap of differences
bd = {}
for q in ('crps_abs', 'crps_rel', 'ause_abs', 'ause_rel', 'p1_abs', 'p1_rel', 'w1_abs', 'w1_rel', 'cov50', 'cov90',
          'hf_single', 'slope_diffusion', 'slope_varhead'):
    bd[q] = dict(d8=bsum(boot[(8, q)]), d4=bsum(boot[(4, q)]), d4_minus_d8=bsum(boot[(4, q)] - boot[(8, q)]),
                 frac_d4_smaller=float(np.mean(boot[(4, q)] < boot[(8, q)])),
                 frac_absd4_smaller=float(np.mean(np.abs(boot[(4, q)]) < np.abs(boot[(8, q)]))))

# ---------------------------------------------------------------- S1 debias and P2 feasibility
s1 = {}
p2 = {}
for s in (4, 8, '8k64'):
    K = rec[s]['K']
    mk = A(s, 'diffusion', 'mean_rmse_pct')
    rr = A(s, 'regressor', 'rmse_pct')
    sr = A(s, 'diffusion', 'spread_ratio')
    vbar = sr * mk ** 2                              # mean per-element sample variance (ddof 1), %^2
    deb = np.sqrt(np.maximum(mk ** 2 - vbar / K, 0))
    lit = st(mk - rr)
    dd = st(deb - rr)
    exp_centred = st(np.sqrt(rr ** 2 + vbar / K) - rr)
    s1[str(s)] = dict(K=K, literal=lit, literal_pass=abs(lit['mean']) <= 2 * lit['se_cluster'],
                      debiased=dd, debiased_pass=abs(dd['mean']) <= 2 * dd['se_cluster'],
                      expected_literal_for_perfectly_centred=exp_centred,
                      literal_minus_expected=st(mk - rr - (np.sqrt(rr ** 2 + vbar / K) - rr)),
                      mean_sample_sd_pct=float(np.sqrt(vbar.mean())),
                      K_needed_for_literal_pass=None)
    # K needed so the expected 1/K term <= 2 SE_cluster (using the debiased SE as a proxy)
    for Kt in (32, 64, 128, 256, 512, 1024, 2048):
        e = float(np.mean(np.sqrt(rr ** 2 + vbar / Kt) - rr))
        if e <= 2 * dd['se_cluster']:
            s1[str(s)]['K_needed_for_literal_pass'] = Kt
            break
    if s == '8k64':
        continue
    rb = A(s, 'baseline', 'rmse_pct')
    rs = A(s, 'diffusion', 'single_rmse_pct')
    ratio = rb / rr
    sstar = (rb.mean() ** 2 - rr.mean() ** 2) / rr.mean() ** 2
    p2[str(s)] = dict(
        ratio_of_means=float(rb.mean() / rr.mean()),
        ratio_percube_mean=float(ratio.mean()), ratio_percube_median=float(np.median(ratio)),
        ratio_percube_max=float(ratio.max()),
        frac_cubes_ratio_gt_sqrt2=float(np.mean(ratio > math.sqrt(2))),
        calibrated_single_rmse_sqrt2_reg=float(math.sqrt(2) * rr.mean()),
        observed_single_rmse=float(rs.mean()),
        predicted_single_rmse_from_spread=float(np.mean(np.sqrt(rr ** 2 + vbar))),
        P2_for_calibrated_sampler=st(math.sqrt(2) * rr - rb),
        P2_observed=st(rs - rb),
        spread_ratio_needed_to_pass_P2=float(sstar),
        gaussian_cov90_at_that_spread=float(2 * norm.cdf(1.6448536 * math.sqrt(sstar)) - 1),
        gaussian_cov50_at_that_spread=float(2 * norm.cdf(0.6744898 * math.sqrt(sstar)) - 1),
        observed_spread_ratio_vs_regressor_mse=float(np.mean(vbar) / np.mean(rr ** 2)))


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items() if not str(k).startswith('_')}
    return o


result = dict(per_stride=clean(res), paired_d4_minus_d8=diff, bootstrap=bd, S1=s1, P2=p2, NB=NB, G=G, n=n)
json.dump(result, open(OUT + 'dose.json', 'w'), indent=1)


def f(x, k=4):
    return f'{x:+.{k}f}'


for s in (8, 4):
    r = res[s]
    print(f'\n===== d = {s} =====')
    print(f"CRPS diff {r['crps_diff']['mean']:.4f} varhead {r['crps_vh']['mean']:.4f}  abs gap {f(r['crps_gap_abs']['mean'])} ± {r['crps_gap_abs']['se_cluster']:.4f} (cl)  "
          f"rel gap {100*r['crps_gap_rel']['rel']:+.2f}% ± {100*r['crps_gap_rel']['se_cluster']:.2f}% (delta, cl)  boot SE {100*bd['crps_rel']['d%d'%s]['se']:.2f}% CI [{100*bd['crps_rel']['d%d'%s]['lo']:.2f}, {100*bd['crps_rel']['d%d'%s]['hi']:.2f}]  per-cube mean rel {100*r['crps_gap_rel_percube']['mean']:+.2f}% ± {100*r['crps_gap_rel_percube']['se_cluster']:.2f}")
    print(f"AUSE diff {r['ause_diff']['mean']:.4f} varhead {r['ause_vh']['mean']:.4f}  abs gap {f(r['ause_gap_abs']['mean'])} ± {r['ause_gap_abs']['se_cluster']:.4f}  rel gap {100*r['ause_gap_rel']['rel']:+.1f}% ± {100*r['ause_gap_rel']['se_cluster']:.1f}%")
    print(f"cov50 diff {r['cov50_diffusion']['mean']:.4f}±{r['cov50_diffusion']['se_cluster']:.4f} vh {r['cov50_varhead']['mean']:.4f}  cov90 diff {r['cov90_diffusion']['mean']:.4f}±{r['cov90_diffusion']['se_cluster']:.4f} vh {r['cov90_varhead']['mean']:.4f}  gauss-fit diff cov50 {r['cov50_gauss_diffusion']['mean']:.4f} cov90 {r['cov90_gauss_diffusion']['mean']:.4f}")
    print(f"spread ratio diff {r['spread_ratio_diffusion']['mean']:.3f} vh {r['spread_ratio_varhead']['mean']:.3f}")
    print(f"hf single {r['hf_single']['mean']:+.3f}  regressor {r['hf_regressor']['mean']:+.3f}  meanK {r['hf_meanK']['mean']:+.3f}  P1 abs {f(r['P1_abs']['mean'])} ± {r['P1_abs']['se_cluster']:.4f}  P1 rel {100*r['P1_rel']['rel']:+.1f}% ± {100*r['P1_rel']['se_cluster']:.1f}%")
    print(f"W1 single {r['w1_single']['mean']:.5f} regressor {r['w1_regressor']['mean']:.5f} meanK {r['w1_meanK']['mean']:.5f}  gap abs {r['w1_gap_abs']['mean']:+.5f} ± {r['w1_gap_abs']['se_cluster']:.5f}  rel {100*r['w1_gap_rel']['rel']:+.1f}% ± {100*r['w1_gap_rel']['se_cluster']:.1f}%")
    print(f"RMSE single {r['single_rmse']['mean']:.4f} reg {r['regressor_rmse']['mean']:.4f} base {r['baseline_rmse']['mean']:.4f} single/reg {r['single_over_reg_rmse']:.3f}")
    print('slope', r['slope'], ' boot SE diff', bd['slope_diffusion']['d%d' % s]['se'], 'vh', bd['slope_varhead']['d%d' % s]['se'])
r = res['8k64']
print(f"\nd8 K=64: CRPS gap abs {r['crps_gap_abs']['mean']:+.4f} ± {r['crps_gap_abs']['se_cluster']:.4f}, rel {100*r['crps_gap_rel']['rel']:+.2f}% ± {100*r['crps_gap_rel']['se_cluster']:.2f}%")
print('\n===== paired d4 - d8 =====')
for k, v in diff.items():
    print(k, {a: round(b, 5) for a, b in v.items()})
print('\n===== bootstrap =====')
for k, v in bd.items():
    print(k, 'd4-d8', {a: round(b, 5) for a, b in v['d4_minus_d8'].items()}, 'P(d4<d8)', v['frac_d4_smaller'], 'P(|d4|<|d8|)', v['frac_absd4_smaller'])
print('\n===== S1 =====')
print(json.dumps(s1, indent=1))
print('\n===== P2 =====')
print(json.dumps(p2, indent=1))
