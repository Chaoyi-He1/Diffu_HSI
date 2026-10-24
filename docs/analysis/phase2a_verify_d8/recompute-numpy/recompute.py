import json, os, re
import numpy as np
from collections import Counter

R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
rec = json.load(open(f'{R}/uq/d8/eval/d8_main.json'))
ver = json.load(open(f'{R}/uq/d8/verdict__d8_main.json'))
gate = json.load(open('/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan/tests/data/val_subset_200.json'))

files = rec['files']
print('n files', len(files), 'unique', len(set(files)))
gfiles = gate['files'] if isinstance(gate, dict) else gate
print('gate list len', len(gfiles), 'same order as record:', list(gfiles) == list(files), 'same set:', set(gfiles) == set(files))

def image_of(f):
    # MatFlower60/Train/P0xx/<img>_<patch>.mat -> P0xx/<img>
    parts = f.split('/')
    cls = parts[-2]
    stem = os.path.splitext(parts[-1])[0]
    m = re.match(r'^(.*)_(\d+)$', stem)
    assert m, f
    return f'{cls}/{m.group(1)}'

groups = np.array([image_of(f) for f in files])
G = len(set(groups))
print('clusters', G)
# check classes key consistency
assert all(g.split('/')[0] == c for g, c in zip(groups, rec['classes']))

def iid_se(d):
    d = np.asarray(d, np.float64)
    return d.std(ddof=1) / np.sqrt(len(d))

def cluster_se(d, g):
    d = np.asarray(d, np.float64)
    n = len(d)
    r = d - d.mean()
    labs = sorted(set(g))
    Gc = len(labs)
    s = np.array([r[g == l].sum() for l in labs])
    return np.sqrt(Gc / (Gc - 1) * np.sum(s ** 2)) / n

pc = rec['per_cube']
A = lambda m, k: np.asarray(pc[m][k], np.float64)

out = {}
def paired(name, d):
    d = np.asarray(d, np.float64)
    out[name] = dict(mean=d.mean(), se=iid_se(d), se_cluster=cluster_se(d, groups), n=len(d))
    return out[name]

# C1
c1 = paired('C1', A('diffusion', 'crps') - A('varhead', 'crps'))
c1['passed'] = bool(c1['mean'] + 2 * c1['se_cluster'] < 0)
# C2
cov90 = A('diffusion', 'cov90').mean(); cov50 = A('diffusion', 'cov50').mean()
out['C2'] = dict(cov90=cov90, cov50=cov50, passed=bool(0.85 <= cov90 <= 0.95 and 0.45 <= cov50 <= 0.55))
# P1
p1 = paired('P1', np.abs(A('diffusion', 'single_hf')) - np.abs(A('regressor', 'hf')))
p1['passed'] = bool(p1['mean'] + 2 * p1['se_cluster'] < 0)
# P2
p2 = paired('P2', A('diffusion', 'single_rmse_pct') - A('baseline', 'rmse_pct'))
p2['passed'] = bool(p2['mean'] + 2 * p2['se_cluster'] < 0)
# S1
s1 = paired('S1', A('diffusion', 'mean_rmse_pct') - A('regressor', 'rmse_pct'))
s1['passed'] = bool(abs(s1['mean']) <= 2 * s1['se_cluster'])
# S2
ys = rec['yswap']
d2 = np.asarray(ys['swap_full'], np.float64) - np.asarray(ys['true'], np.float64)
out['S2'] = dict(mean=d2.mean(), se=iid_se(d2), n=len(d2))
out['S2']['passed'] = bool(out['S2']['mean'] - 2 * out['S2']['se'] > 0)
# C3 reported
paired('C3', A('diffusion', 'ause') - A('varhead', 'ause'))
# ensemble / halfcentre reported
paired('ens_crps', A('ensemble', 'crps') - A('varhead', 'crps'))
paired('ens_crps_vs_diff', A('ensemble', 'crps') - A('diffusion', 'crps'))
paired('ens_ause', A('ensemble', 'ause') - A('varhead', 'ause'))
paired('ens_ause_vs_diff', A('ensemble', 'ause') - A('diffusion', 'ause'))
paired('half_crps', A('diffusion_halfcentre', 'crps') - A('diffusion', 'crps'))
paired('half_crps_vs_varhead', A('diffusion_halfcentre', 'crps') - A('varhead', 'crps'))

def cmp(a, b):
    return 'MATCH' if b is not None and abs(a - b) <= 1e-12 * max(1, abs(b)) else f'DIFF (abs {abs(a-b) if b is not None else None:.3e})'

print('\n=== clause comparison vs verdict file ===')
for k in ['C1', 'P1', 'P2', 'S1']:
    for f in ['mean', 'se', 'se_cluster', 'n']:
        print(k, f, repr(out[k][f]), repr(ver[k][f]), cmp(out[k][f], ver[k][f]))
    print(k, 'clusters', G, ver[k]['clusters'], 'passed', out[k]['passed'], ver[k]['passed'])
for f in ['cov90', 'cov50']:
    print('C2', f, repr(out['C2'][f]), repr(ver['C2'][f]), cmp(out['C2'][f], ver['C2'][f]))
print('C2 passed', out['C2']['passed'], ver['C2']['passed'])
for f in ['mean', 'se', 'n']:
    print('S2', f, repr(out['S2'][f]), repr(ver['S2'][f]), cmp(out['S2'][f], ver['S2'][f]))
print('S2 passed', out['S2']['passed'], ver['S2']['passed'])
for f in ['mean', 'se', 'se_cluster', 'n']:
    print('C3', f, repr(out['C3'][f]), repr(ver['reported']['C3'][f]), cmp(out['C3'][f], ver['reported']['C3'][f]))

print('\n=== reported ensemble / halfcentre (identify which pairing the verdict used) ===')
for k in ['ens_crps', 'ens_crps_vs_diff', 'ens_ause', 'ens_ause_vs_diff', 'half_crps', 'half_crps_vs_varhead']:
    print(k, {kk: round(v, 6) if isinstance(v, float) else v for kk, v in out[k].items()})
print('verdict ensemble', ver['reported']['ensemble'])
print('verdict halfcentre', ver['reported']['halfcentre'])
print('spread ratio diff', A('diffusion', 'spread_ratio').mean(), ver['reported']['spread_ratio']['diffusion'])
print('spread ratio varhead', A('varhead', 'spread_ratio').mean(), ver['reported']['spread_ratio']['varhead'])
print('cov90 varhead', A('varhead', 'cov90').mean(), ver['reported']['cov90_varhead'])
print('cov90 gauss diff', A('diffusion', 'cov90_gauss').mean(), ver['reported']['cov90_gauss_diffusion'])
print('ens cov90', A('ensemble', 'cov90').mean(), ver['reported']['ensemble']['cov90'])
print('half cov90', A('diffusion_halfcentre', 'cov90').mean(), ver['reported']['halfcentre']['cov90'])

print('\n=== flags ===')
cal = out['C1']['passed'] and out['C2']['passed']
per = out['P1']['passed'] and out['P2']['passed']
wins = cal and per and out['S1']['passed'] and out['S2']['passed']
print('calibration', cal, ver['calibration_pass'], 'perceptual', per, ver['perceptual_pass'], 'wins', wins, ver['wins'])

print('\n=== margins (distance to flipping) ===')
for k in ['C1', 'P1', 'P2']:
    o = out[k]; print(k, 'mean+2SEc =', o['mean'] + 2 * o['se_cluster'], ' (pass if <0); with iid SE:', o['mean'] + 2 * o['se'])
o = out['S1']; print('S1 |mean| - 2SEc =', abs(o['mean']) - 2 * o['se_cluster'], ' (pass if <=0); |mean|/SEc =', abs(o['mean']) / o['se_cluster'])
o = out['S2']; print('S2 mean - 2SE =', o['mean'] - 2 * o['se'], ' (pass if >0); t =', o['mean'] / o['se'])
print('C2 cov90 margin to 0.95:', 0.95 - cov90, ' cov50 above 0.55 by:', cov50 - 0.55)

print('\n=== printed-table means cross-check ===')
pm = dict(
 baseline_rmse=A('baseline','rmse_pct').mean(), baseline_hf=A('baseline','hf').mean(),
 regressor_rmse=A('regressor','rmse_pct').mean(), regressor_hf=A('regressor','hf').mean(), regressor_sam=A('regressor','sam_deg').mean(),
 diff_mean_rmse=A('diffusion','mean_rmse_pct').mean(), diff_mean_hf=A('diffusion','mean_hf').mean(),
 diff_single_rmse=A('diffusion','single_rmse_pct').mean(), diff_single_hf=A('diffusion','single_hf').mean(), diff_single_sam=A('diffusion','single_sam_deg').mean(),
 diff_crps=A('diffusion','crps').mean(), diff_cov90=cov90, diff_cov50=cov50, diff_ause=A('diffusion','ause').mean(), diff_spread=A('diffusion','spread_ratio').mean(),
 vh_crps=A('varhead','crps').mean(), vh_cov90=A('varhead','cov90').mean(), vh_cov50=A('varhead','cov50').mean(), vh_ause=A('varhead','ause').mean(), vh_spread=A('varhead','spread_ratio').mean(),
 ens_crps=A('ensemble','crps').mean(), ens_cov90=A('ensemble','cov90').mean(), half_crps=A('diffusion_halfcentre','crps').mean())
for k, v in pm.items(): print(k, round(v, 4))
# record means block vs per-cube
bad = []
for m, dd in rec['means'].items():
    for k, v in dd.items():
        mm = np.mean(np.asarray(pc[m][k], np.float64))
        if abs(mm - v) > 1e-9: bad.append((m, k, mm, v))
print('record means block mismatches:', bad)

print('\n=== sanity: NaN/inf, varhead==regressor point estimate, yswap overlap ===')
for m, dd in pc.items():
    for k, v in dd.items():
        a = np.asarray(v, np.float64)
        if not np.all(np.isfinite(a)) or len(a) != 200: print('BAD', m, k)
print('varhead mean_rmse == regressor rmse:', np.array_equal(A('varhead','mean_rmse_pct'), A('regressor','rmse_pct')))
yf = ys['files']
print('yswap n', len(yf), 'all in gate:', set(yf) <= set(files), 'unique:', len(set(yf)))
idx = [files.index(f) for f in yf]
print('yswap.true vs per_cube diffusion mean_rmse on same files, max abs diff:', np.max(np.abs(np.asarray(ys['true']) - A('diffusion','mean_rmse_pct')[idx])))
print('yswap.regressor vs per_cube regressor rmse max abs diff:', np.max(np.abs(np.asarray(ys['regressor']) - A('regressor','rmse_pct')[idx])))
print('yswap.baseline vs per_cube baseline rmse max abs diff:', np.max(np.abs(np.asarray(ys['baseline']) - A('baseline','rmse_pct')[idx])))
print('yswap classes', Counter(f.split('/')[-2] for f in yf))
print('yswap swap_full - regressor mean', np.mean(np.asarray(ys['swap_full']) - np.asarray(ys['regressor'])))
print('stored files subset of yswap', set(rec['stored_files']) <= set(yf))
print('cluster size dist', Counter(Counter(groups).values()))
json.dump({k: {kk: (float(vv) if isinstance(vv, (float, np.floating)) else vv) for kk, vv in v.items()} for k, v in out.items()},
          open('/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d8/recompute-numpy/recomputed.json', 'w'), indent=1)
