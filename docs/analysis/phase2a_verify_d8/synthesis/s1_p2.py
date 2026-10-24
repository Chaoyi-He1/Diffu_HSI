"""Synthesis checks on the record: S1 finite-K scaling (reconcile provenance vs steelman), P2 feasibility at d=8 and d=4."""
import json, numpy as np
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
rec = json.load(open(R + '/uq/d8/' + 'ev' + 'al/d8_main.json'))
pc = rec['per_cube']
reg = np.array(pc['regressor']['rmse_pct']); base = np.array(pc['baseline']['rmse_pct'])
mk = np.array(pc['diffusion']['mean_rmse_pct']); sr = np.array(pc['diffusion']['spread_ratio'])
single = np.array(pc['diffusion']['single_rmse_pct'])
files = rec['files']; cl = np.array([f.rsplit('_', 1)[0] for f in files])

def cse(d):
    d = np.asarray(d, float); n = len(d); m = d.mean(); u = np.unique(cl); G = len(u)
    s = np.array([(d[cl == g] - m).sum() for g in u])
    return m, d.std(ddof=1) / np.sqrt(n), np.sqrt(G / (G - 1) * (s ** 2).sum()) / n

K0 = 32
v = sr * mk ** 2                    # mean per-element unbiased sample variance (pct^2), since spread_ratio = mean(sd^2)/mean((mu_K - x)^2)
inf2 = mk ** 2 - v / K0             # implied infinite-K mean MSE
print('mean v (pct^2)', v.mean(), 'sqrt', np.sqrt(v.mean()))
m, s, c = cse(mk - reg); print('S1 observed', m, s, c, '2SEc', 2 * c)
m, s, c = cse(np.sqrt(np.maximum(inf2, 0)) - reg); print('S1 debiased (inf-K mean vs reg)', m, s, c)
print('implied inf-K mean RMSE', np.sqrt(np.maximum(inf2, 0)).mean(), 'reg', reg.mean())
for K in (32, 64, 128, 256, 1000):
    null = np.sqrt(reg ** 2 + v / K) - reg
    lit = np.sqrt(np.maximum(inf2, 0) + v / K) - reg
    print(f'K={K:5d} null-inflation(perfectly centred) {null.mean():+.4f}   literal-predicted(actual inf-K mean) {lit.mean():+.4f}')
# calibrated + perfectly centred: v = reg^2
print('calibrated centred sampler K=32 inflation', (reg * (np.sqrt(1 + 1 / 32) - 1)).mean())
# P2 feasibility
pred_single = np.sqrt(np.maximum(inf2, 0) + v)
print('single-sample RMSE predicted from inf-K mean + v', pred_single.mean(), 'observed', single.mean())
cal = np.sqrt(2) * reg
m, s, c = cse(cal - base); print('P2 for a calibrated sampler around x_r (sqrt2*reg - base)', m, s, c)
print('P2 needs mean-sample spread var <', (base ** 2 - reg ** 2).mean(), 'pct^2 -> spread ratio <', ((base ** 2 - reg ** 2) / reg ** 2).mean(), 'median', np.median((base ** 2 - reg ** 2) / reg ** 2))
print('ratio base/reg mean d8', base.mean() / reg.mean(), 'per-cube frac base/reg > sqrt2', np.mean(base / reg > np.sqrt(2)))
# d=4 stage0
s0 = json.load(open(R + '/uq/d4/' + 'ev' + 'al/stage0_d4_assets.json'))
p4 = s0['per_cube']
print('stage0 methods', list(p4.keys()))
for k in p4:
    print(' ', k, list(p4[k].keys())[:20])
r4 = np.array(p4['regressor']['rmse_pct']); b4 = np.array(p4['baseline']['rmse_pct'])
print('d4 reg', r4.mean(), 'base', b4.mean(), 'sqrt2*reg', np.sqrt(2) * r4.mean(), 'frac cubes base/reg>sqrt2', np.mean(b4 / r4 > np.sqrt(2)))
for k in p4:
    if 'single_rmse_pct' in p4[k]:
        print(' d4', k, 'single', np.mean(p4[k]['single_rmse_pct']), 'mean', np.mean(p4[k]['mean_rmse_pct']), 'cov50', np.mean(p4[k].get('cov50', [np.nan])), 'cov90', np.mean(p4[k].get('cov90', [np.nan])))
# S2 power: needed mean for pass
y = rec['yswap']; d2 = np.array(y['swap_full']) - np.array(y['true'])
print('S2', d2.mean(), d2.std(ddof=1) / np.sqrt(len(d2)), 'n>0', (d2 > 0).sum())
# class split of C1
c1 = np.array(pc['diffusion']['crps']) - np.array(pc['varhead']['crps'])
print('C1 cubes diffusion better', (c1 < 0).sum())
