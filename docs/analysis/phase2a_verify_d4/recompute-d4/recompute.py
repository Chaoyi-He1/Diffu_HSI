"""Independent recomputation of the d = 4 contrast clauses + reported block + B comparison, and the d = 8 vs d = 4
dose-response comparison. numpy only; does NOT import scripts/uq_verdict.py or scripts/paired_stats.py."""
import json
import math
import os

import numpy as np

R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart/uq/'
OUT = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d4/recompute-d4/'
rec4 = json.load(open(R + 'd4/eval/d4_contrast.json'))
rec8 = json.load(open(R + 'd8/eval/d8_main.json'))
rec8k = json.load(open(R + 'd8/eval/d8_fix_k64.json'))
ver4 = json.load(open(R + 'd4/verdict__d4_contrast__report.json'))
ver8 = json.load(open(R + 'd8/verdict__d8_main.json'))


def key(f):
    # P0xx/<img> : class dir + basename stem before the last '_'
    parts = f.split('/')
    return parts[-2] + '/' + parts[-1][:-4].rsplit('_', 1)[0]


def stats(d, files):
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / math.sqrt(n)
    keys = [key(f) for f in files]
    uk = sorted(set(keys))
    idx = {k: i for i, k in enumerate(uk)}
    G = len(uk)
    sums = np.zeros(G)
    for k, v in zip(keys, d - m):
        sums[idx[k]] += v
    sec = math.sqrt(G / (G - 1) * np.sum(sums ** 2)) / n
    return dict(mean=float(m), se=float(se), se_cluster=float(sec), n=n, clusters=G)


def A(rec, m, k):
    return np.asarray(rec['per_cube'][m][k], dtype=np.float64)


out = {}
f4 = rec4['files']
f8 = rec8['files']
print('files identical d4 vs d8:', f4 == f8, ' d8k64 files identical:', rec8k['files'] == f8)
print('n clusters d4', len(set(key(f) for f in f4)))
K4 = rec4['K']

# ------------------------------------------------------------------ clauses at d = 4
def clauses(rec):
    f = rec['files']
    c = {}
    c['C1'] = stats(A(rec, 'diffusion', 'crps') - A(rec, 'varhead', 'crps'), f)
    c['C1']['passed'] = c['C1']['mean'] + 2 * c['C1']['se_cluster'] < 0
    cov90, cov50 = A(rec, 'diffusion', 'cov90').mean(), A(rec, 'diffusion', 'cov50').mean()
    c['C2'] = dict(cov90=float(cov90), cov50=float(cov50), passed=bool(0.85 <= cov90 <= 0.95 and 0.45 <= cov50 <= 0.55),
                   cov90_se_cluster=stats(A(rec, 'diffusion', 'cov90'), f)['se_cluster'],
                   cov50_se_cluster=stats(A(rec, 'diffusion', 'cov50'), f)['se_cluster'])
    c['P1'] = stats(np.abs(A(rec, 'diffusion', 'single_hf')) - np.abs(A(rec, 'regressor', 'hf')), f)
    c['P1']['passed'] = c['P1']['mean'] + 2 * c['P1']['se_cluster'] < 0
    c['P2'] = stats(A(rec, 'diffusion', 'single_rmse_pct') - A(rec, 'baseline', 'rmse_pct'), f)
    c['P2']['passed'] = c['P2']['mean'] + 2 * c['P2']['se_cluster'] < 0
    c['S1'] = stats(A(rec, 'diffusion', 'mean_rmse_pct') - A(rec, 'regressor', 'rmse_pct'), f)
    c['S1']['passed'] = abs(c['S1']['mean']) <= 2 * c['S1']['se_cluster']
    ys = rec['yswap']
    dsw = np.asarray(ys['swap_full']) - np.asarray(ys['true'])
    c['S2'] = stats(dsw, ys['files'])
    c['S2']['passed'] = c['S2']['mean'] - 2 * c['S2']['se'] > 0
    rep = {}
    rep['C3'] = stats(A(rec, 'diffusion', 'ause') - A(rec, 'varhead', 'ause'), f)
    rep['spread_ratio'] = dict(diffusion=float(A(rec, 'diffusion', 'spread_ratio').mean()),
                               varhead=float(A(rec, 'varhead', 'spread_ratio').mean()))
    rep['cov90_varhead'] = float(A(rec, 'varhead', 'cov90').mean())
    rep['cov50_varhead'] = float(A(rec, 'varhead', 'cov50').mean())
    rep['cov90_gauss_diffusion'] = float(A(rec, 'diffusion', 'cov90_gauss').mean())
    rep['ensemble'] = dict(crps=stats(A(rec, 'diffusion', 'crps') - A(rec, 'ensemble', 'crps'), f),
                           ause=stats(A(rec, 'diffusion', 'ause') - A(rec, 'ensemble', 'ause'), f),
                           cov90=float(A(rec, 'ensemble', 'cov90').mean()))
    rep['halfcentre'] = dict(crps=stats(A(rec, 'diffusion_halfcentre', 'crps') - A(rec, 'diffusion', 'crps'), f),
                             cov90=float(A(rec, 'diffusion_halfcentre', 'cov90').mean()))
    if 'b' in rec['per_cube']:
        rep['b'] = dict(crps=stats(A(rec, 'diffusion', 'crps') - A(rec, 'b', 'crps'), f),
                        ause=stats(A(rec, 'diffusion', 'ause') - A(rec, 'b', 'ause'), f),
                        crps_b_minus_varhead=stats(A(rec, 'b', 'crps') - A(rec, 'varhead', 'crps'), f),
                        single_rmse=stats(A(rec, 'diffusion', 'single_rmse_pct') - A(rec, 'b', 'single_rmse_pct'), f),
                        single_abs_hf=stats(np.abs(A(rec, 'diffusion', 'single_hf')) - np.abs(A(rec, 'b', 'single_hf')), f),
                        single_w1=stats(A(rec, 'diffusion', 'single_w1') - A(rec, 'b', 'single_w1'), f),
                        cov90=float(A(rec, 'b', 'cov90').mean()), cov50=float(A(rec, 'b', 'cov50').mean()),
                        spread_ratio=float(A(rec, 'b', 'spread_ratio').mean()),
                        crps_b_lower_on=int(np.sum(A(rec, 'b', 'crps') < A(rec, 'diffusion', 'crps'))),
                        ause_b_lower_on=int(np.sum(A(rec, 'b', 'ause') < A(rec, 'diffusion', 'ause'))))
    c['calibration_pass'] = bool(c['C1']['passed'] and c['C2']['passed'])
    c['perceptual_pass'] = bool(c['P1']['passed'] and c['P2']['passed'])
    c['wins'] = bool(c['calibration_pass'] and c['perceptual_pass'] and c['S1']['passed'] and c['S2']['passed'])
    c['crps_varhead_lower_on'] = int(np.sum(A(rec, 'varhead', 'crps') < A(rec, 'diffusion', 'crps')))
    return c, rep


c4, rep4 = clauses(rec4)
c8, rep8 = clauses(rec8)
out['d4_clauses'] = c4
out['d4_reported'] = rep4
out['d8_clauses'] = c8


# compare with verdict file
def cmp(a, b, path=''):
    rows = []
    if isinstance(b, dict):
        for k, v in b.items():
            if isinstance(a, dict) and k in a:
                rows += cmp(a[k], v, path + '.' + k)
    elif isinstance(b, bool) or b is None:
        rows.append((path, a, b, a == b))
    elif isinstance(b, (int, float)):
        rows.append((path, a, b, abs(float(a) - float(b))))
    return rows


print('\n=== d4 verdict comparison (recomputed vs verdict file) ===')
maxdiff = 0.0
mism = []
for sect in ('C1', 'C2', 'P1', 'P2', 'S1', 'S2'):
    for p, a, b, dd in cmp(c4[sect], ver4[sect], sect):
        flag = ''
        if isinstance(dd, bool):
            flag = '' if dd else '  <-- MISMATCH'
            if not dd:
                mism.append(p)
        else:
            maxdiff = max(maxdiff, dd)
            if dd > 1e-12:
                flag = '  <-- diff %.3g' % dd
        print(f'  {p:40s} {a!r:>24} {b!r:>24}{flag}')
for p, a, b, dd in cmp(rep4, ver4['reported'], 'reported'):
    flag = ''
    if isinstance(dd, bool):
        if not dd:
            mism.append(p)
            flag = '  <-- MISMATCH'
    else:
        maxdiff = max(maxdiff, dd)
        if dd > 1e-12:
            flag = '  <-- diff %.3g' % dd
    print(f'  {p:40s} {a!r:>24} {b!r:>24}{flag}')
for k in ('calibration_pass', 'perceptual_pass', 'wins'):
    print(f'  {k:40s} {c4[k]!r:>24} {ver4[k]!r:>24}')
    if c4[k] != ver4[k]:
        mism.append(k)
print('max abs numeric diff vs d4 verdict:', maxdiff, ' boolean mismatches:', mism)
out['d4_verdict_maxdiff'] = maxdiff
out['d4_verdict_mismatch'] = mism

print('\n=== d8 verdict comparison ===')
md8 = 0.0
for sect in ('C1', 'C2', 'P1', 'P2', 'S1', 'S2'):
    for p, a, b, dd in cmp(c8[sect], ver8[sect], sect):
        if not isinstance(dd, bool):
            md8 = max(md8, dd)
        elif not dd:
            print('MISMATCH', p)
print('max abs diff d8 clauses', md8)

print('\n=== d4 clause summary ===')
for s in ('C1', 'P1', 'P2', 'S1', 'S2'):
    v = c4[s]
    print(f"{s}: mean {v['mean']:+.4f} se {v['se']:.4f} se_cl {v['se_cluster']:.4f} n {v['n']} G {v['clusters']} passed {v['passed']}")
print('C2', c4['C2'])
print('C3', rep4['C3'])
print('ensemble', rep4['ensemble'])
print('halfcentre', rep4['halfcentre'])
print('B', json.dumps(rep4['b'], indent=1))
print('varhead lower CRPS on', c4['crps_varhead_lower_on'], '/200 at d4;', c8['crps_varhead_lower_on'], '/200 at d8')

json.dump(out, open(OUT + 'clauses.json', 'w'), indent=1)
