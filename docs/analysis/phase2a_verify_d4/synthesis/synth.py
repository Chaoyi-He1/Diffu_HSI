"""Synthesis re-checks for the d = 4 contrast (CPU, numpy only).

Re-derives the scale-free dose-response numbers on which the two verifiers frame things differently
(relative CRPS gap by several normalisations, AUSE share, constant-Gaussian share), with one CR1 cluster SE
(paired_stats formula, re-implemented) and a joint 5000-draw cluster bootstrap over the 113 source images.
"""
import json, os
import numpy as np

R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart/uq/'
OUT = os.path.dirname(os.path.abspath(__file__))
D4 = json.load(open(R + 'd4/eval/d4_contrast.json'))
D8 = json.load(open(R + 'd8/eval/d8_main.json'))
D8K = json.load(open(R + 'd8/eval/d8_fix_k64.json'))
REF4 = json.load(open('/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d4/mechanism-d4/ref200_d4.json'))
REF8 = json.load(open('/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan/docs/analysis/phase2a_verify_d8/condition-blindness/ref200.json'))
assert D4['files'] == D8['files'] == D8K['files']
files = D4['files']
n = len(files)


def key(f):
    return os.path.basename(os.path.dirname(f)) + '/' + os.path.basename(f).rsplit('_', 1)[0]


keys = [key(f) for f in files]
uk = sorted(set(keys)); pos = {k: i for i, k in enumerate(uk)}
gid = np.array([pos[k] for k in keys]); G = len(uk)
print('n', n, 'G', G)


def cse(d):
    d = np.asarray(d, float)
    s = np.bincount(gid, weights=d - d.mean(), minlength=G)
    return float(np.sqrt(G / (G - 1) * np.sum(s ** 2)) / n)


def A(rec, m, k):
    return np.asarray(rec['per_cube'][m][k], float)


def stats_for(rec, ref):
    S = {}
    cd, cv = A(rec, 'diffusion', 'crps'), A(rec, 'varhead', 'crps')
    rr = A(rec, 'regressor', 'rmse_pct')
    ad, av = A(rec, 'diffusion', 'ause'), A(rec, 'varhead', 'ause')
    S['crps_gap'] = (cd - cv)
    S['crps_d'] = cd; S['crps_v'] = cv; S['rmse_r'] = rr; S['ause_d'] = ad; S['ause_v'] = av
    S['ause_gap'] = ad - av
    S['ause_rand'] = np.asarray(ref['per_cube']['ause_random'], float)
    S['homo05'] = np.asarray(ref['per_cube']['crps_homo_0.5'], float)
    S['w1_s'] = A(rec, 'diffusion', 'single_w1'); S['w1_r'] = A(rec, 'regressor', 'w1')
    S['hf_s'] = A(rec, 'diffusion', 'single_hf'); S['hf_r'] = A(rec, 'regressor', 'hf')
    S['p1'] = np.abs(S['hf_s']) - np.abs(S['hf_r'])
    S['cov50_d'] = A(rec, 'diffusion', 'cov50'); S['cov90_d'] = A(rec, 'diffusion', 'cov90')
    S['cov50_v'] = A(rec, 'varhead', 'cov50'); S['cov90_v'] = A(rec, 'varhead', 'cov90')
    S['sr_d'] = A(rec, 'diffusion', 'spread_ratio')
    S['mrmse'] = A(rec, 'diffusion', 'mean_rmse_pct')
    S['srmse'] = A(rec, 'diffusion', 'single_rmse_pct')
    S['brmse'] = A(rec, 'baseline', 'rmse_pct')
    return S


def scalars(S, idx=None, K=32):
    """All scale-free / absolute summaries for a (possibly resampled) index set."""
    if idx is None:
        idx = np.arange(n)
    g = lambda k: S[k][idx]
    o = {}
    o['crps_gap_abs'] = g('crps_gap').mean()
    o['crps_rel_ratio_of_means'] = g('crps_d').mean() / g('crps_v').mean() - 1
    o['crps_rel_mean_percube'] = (g('crps_d') / g('crps_v') - 1).mean()
    o['crps_gap_over_rmse_r_ratio'] = g('crps_gap').mean() / g('rmse_r').mean()
    o['crps_gap_over_rmse_r_percube'] = (g('crps_gap') / g('rmse_r')).mean()
    # share of the variance head's localisation gain over N(x_r, (0.5 sigma_r)^2) that the diffusion achieves
    o['loc_share'] = (g('homo05').mean() - g('crps_d').mean()) / (g('homo05').mean() - g('crps_v').mean())
    o['ause_gap_abs'] = g('ause_gap').mean()
    o['ause_rel_ratio'] = g('ause_d').mean() / g('ause_v').mean() - 1
    # fraction of the random-ranking AUSE removed (1 = oracle, 0 = random)
    o['ause_skill_d'] = 1 - g('ause_d').mean() / g('ause_rand').mean()
    o['ause_skill_v'] = 1 - g('ause_v').mean() / g('ause_rand').mean()
    o['w1_pen_rel_ratio'] = g('w1_s').mean() / g('w1_r').mean() - 1
    o['w1_pen_abs'] = (g('w1_s') - g('w1_r')).mean()
    o['p1'] = g('p1').mean()
    o['p1_rel'] = g('p1').mean() / np.abs(g('hf_r')).mean()
    o['hf_single'] = g('hf_s').mean()
    o['cov50_d'] = g('cov50_d').mean(); o['cov90_d'] = g('cov90_d').mean()
    o['cov50_v'] = g('cov50_v').mean(); o['cov90_v'] = g('cov90_v').mean()
    # S1 literal and debiased
    vbar = g('sr_d') * g('mrmse') ** 2
    o['s1_lit'] = (g('mrmse') - g('rmse_r')).mean()
    o['s1_deb'] = (np.sqrt(np.maximum(g('mrmse') ** 2 - vbar / K, 0)) - g('rmse_r')).mean()
    o['p2'] = (g('srmse') - g('brmse')).mean()
    o['p2_calibrated'] = (np.sqrt(2) * g('rmse_r') - g('brmse')).mean()
    o['base_over_reg'] = g('brmse').mean() / g('rmse_r').mean()
    return o


S4, S8 = stats_for(D4, REF4), stats_for(D8, REF8)
S8K = stats_for(D8K, REF8)
o4, o8, o8k = scalars(S4), scalars(S8), scalars(S8K, K=64)

# delta-method cluster SEs for a few ratio statistics, paired difference via influence functions
def infl_ratio(a, b):
    R_ = a.mean() / b.mean()
    return (a - R_ * b) / b.mean()

infl = {}
for tag, S in (('d4', S4), ('d8', S8)):
    infl[(tag, 'crps_gap_abs')] = S['crps_gap']
    infl[(tag, 'crps_rel_ratio_of_means')] = infl_ratio(S['crps_d'], S['crps_v'])
    infl[(tag, 'crps_rel_mean_percube')] = S['crps_d'] / S['crps_v'] - 1
    infl[(tag, 'crps_gap_over_rmse_r_ratio')] = infl_ratio(S['crps_gap'], S['rmse_r'])
    infl[(tag, 'ause_gap_abs')] = S['ause_gap']
    infl[(tag, 'ause_rel_ratio')] = infl_ratio(S['ause_d'], S['ause_v'])
    infl[(tag, 'w1_pen_rel_ratio')] = infl_ratio(S['w1_s'], S['w1_r'])
    infl[(tag, 'p1')] = S['p1']
    infl[(tag, 'cov50_d')] = S['cov50_d']
    infl[(tag, 'cov90_d')] = S['cov90_d']
    infl[(tag, 'cov50_v')] = S['cov50_v']
    infl[(tag, 'cov90_v')] = S['cov90_v']
    infl[(tag, 'diff_minus_homo05')] = S['crps_d'] - S['homo05']

res = {'d4': {k: float(v) for k, v in o4.items()}, 'd8': {k: float(v) for k, v in o8.items()},
       'd8_k64': {k: float(v) for k, v in o8k.items()}, 'delta_cluster_se': {}}
for name in sorted(set(k[1] for k in infl)):
    a, b = infl[('d4', name)], infl[('d8', name)]
    res['delta_cluster_se'][name] = dict(d4=cse(a), d8=cse(b), paired_d4_minus_d8=cse(a - b),
                                         d4_mean=float(a.mean()) if name in ('diff_minus_homo05',) else None,
                                         d8_mean=float(b.mean()) if name in ('diff_minus_homo05',) else None)

# joint cluster bootstrap
rng = np.random.default_rng(20261024)
members = [np.where(gid == g_)[0] for g_ in range(G)]
B = 5000
boot = {k: [] for k in o4}
for _ in range(B):
    pick = rng.integers(0, G, G)
    idx = np.concatenate([members[p] for p in pick])
    b4, b8 = scalars(S4, idx), scalars(S8, idx)
    for k in o4:
        boot[k].append((b4[k], b8[k]))
res['bootstrap'] = {}
for k, v in boot.items():
    v = np.array(v)
    d = v[:, 0] - v[:, 1]
    res['bootstrap'][k] = dict(se_d4=float(v[:, 0].std(ddof=1)), se_d8=float(v[:, 1].std(ddof=1)),
                               diff=float(o4[k] - o8[k]), se_diff=float(d.std(ddof=1)),
                               ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                               frac_d4_lt_d8=float((d < 0).mean()))

# cube counts
res['counts'] = dict(
    d4_diff_crps_lt_vh=int((S4['crps_d'] < S4['crps_v']).sum()), d8_diff_crps_lt_vh=int((S8['crps_d'] < S8['crps_v']).sum()),
    d4_diff_ause_lt_vh=int((S4['ause_d'] < S4['ause_v']).sum()), d8_diff_ause_lt_vh=int((S8['ause_d'] < S8['ause_v']).sum()),
    d4_single_w1_worse=int((S4['w1_s'] > S4['w1_r']).sum()), d8_single_w1_worse=int((S8['w1_s'] > S8['w1_r']).sum()),
    d4_base_over_reg_gt_sqrt2=int((S4['brmse'] / S4['rmse_r'] > np.sqrt(2)).sum()),
    d8_base_over_reg_gt_sqrt2=int((S8['brmse'] / S8['rmse_r'] > np.sqrt(2)).sum()),
)
# S1 debiased cluster SEs
for tag, S, K in (('d4', S4, 32), ('d8', S8, 32), ('d8_k64', S8K, 64)):
    vbar = S['sr_d'] * S['mrmse'] ** 2
    deb = np.sqrt(np.maximum(S['mrmse'] ** 2 - vbar / K, 0)) - S['rmse_r']
    lit = S['mrmse'] - S['rmse_r']
    exp_ = np.sqrt(S['rmse_r'] ** 2 + vbar / K) - S['rmse_r']
    res.setdefault('s1', {})[tag] = dict(lit=float(lit.mean()), lit_cse=cse(lit), deb=float(deb.mean()), deb_cse=cse(deb),
                                         expected_from_1_over_K=float(exp_.mean()))
    p2c = np.sqrt(2) * S['rmse_r'] - S['brmse']
    res.setdefault('p2_calibrated', {})[tag] = dict(mean=float(p2c.mean()), cse=cse(p2c))
# provenance notes
prov = D4.get('provenance', {})
res['prov'] = {k: {kk: prov[k].get(kk) for kk in ('run', 'epoch', 'K', 'sampler')} for k in prov if isinstance(prov[k], dict)}
res['yswap_keys'] = list(D4.get('yswap', {}).keys())
json.dump(res, open(os.path.join(OUT, 'synth.json'), 'w'), indent=1)

def fmt(k, sc=1.0, p=4):
    b = res['bootstrap'][k]
    return (f"{k:32s} d8 {o8[k]*sc:+.{p}f} (bs {b['se_d8']*sc:.{p}f})  d4 {o4[k]*sc:+.{p}f} (bs {b['se_d4']*sc:.{p}f})  "
            f"d4-d8 {b['diff']*sc:+.{p}f} ± {b['se_diff']*sc:.{p}f} CI [{b['ci95'][0]*sc:+.{p}f}, {b['ci95'][1]*sc:+.{p}f}] P(d4<d8) {b['frac_d4_lt_d8']:.3f}")
for k in o4:
    sc = 100 if k in ('crps_rel_ratio_of_means', 'crps_rel_mean_percube', 'loc_share', 'ause_rel_ratio', 'w1_pen_rel_ratio', 'p1_rel', 'ause_skill_d', 'ause_skill_v') else 1
    print(fmt(k, sc))
print('d8_k64', {k: round(v, 4) for k, v in o8k.items() if k.startswith('crps') or k.startswith('s1')})
print(json.dumps(res['delta_cluster_se'], indent=0))
print(json.dumps(res['counts']))
print(json.dumps(res['s1']))
print(json.dumps(res['p2_calibrated']))
print(json.dumps(res['prov']))
print(res['yswap_keys'])
