import json
import math

import numpy as np

R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart/uq/'
rec = {4: json.load(open(R + 'd4/eval/d4_contrast.json')), 8: json.load(open(R + 'd8/eval/d8_main.json'))}
files = rec[4]['files']


def key(f):
    p = f.split('/')
    return p[-2] + '/' + p[-1][:-4].rsplit('_', 1)[0]


keys = [key(f) for f in files]
uk = sorted(set(keys))
cid = np.array([uk.index(k) for k in keys])


def cse(d, mask=None):
    d = np.asarray(d, float)
    c = cid
    if mask is not None:
        d, c = d[mask], cid[mask]
    u = np.unique(c)
    G = len(u)
    s = np.array([np.sum(d[c == g] - d.mean()) for g in u])
    return float(math.sqrt(G / (G - 1) * np.sum(s ** 2)) / len(d)), G


def A(s, m, k):
    return np.asarray(rec[s]['per_cube'][m][k], float)


cls = np.array(rec[4]['classes'])
for s in (8, 4):
    g = A(s, 'diffusion', 'crps') - A(s, 'varhead', 'crps')
    rel = A(s, 'diffusion', 'crps').sum() / A(s, 'varhead', 'crps').sum() - 1
    print(f'd={s} C1 by class:', {c: (round(float(g[cls == c].mean()), 4), round(cse(g, cls == c)[0], 4), int((cls == c).sum()),
                                      round(float(A(s, 'diffusion', 'crps')[cls == c].mean() / A(s, 'varhead', 'crps')[cls == c].mean() - 1), 4))
                                  for c in sorted(set(cls))})
# slopes for B, ensemble, halfcentre at d4
for m in ('diffusion', 'varhead', 'b', 'ensemble', 'diffusion_halfcentre'):
    rm = A(4, m, 'mean_rmse_pct') / 100
    sp = np.sqrt(A(4, m, 'spread_ratio')) * rm
    print(f'd4 {m:22s} slope log spread on log own-RMSE {np.polyfit(np.log(rm), np.log(sp), 1)[0]:.3f}  spread CV {sp.std() / sp.mean():.3f}  '
          f'Spearman-ish corr(log) {np.corrcoef(np.log(rm), np.log(sp))[0, 1]:.3f}  RMS spread {100 * math.sqrt(np.mean(sp ** 2)):.3f}%')
# slope against the regressor RMSE (common x) for diffusion / varhead at both strides
for s in (8, 4):
    rr = A(s, 'regressor', 'rmse_pct') / 100
    for m in ('diffusion', 'varhead'):
        rm = A(s, m, 'mean_rmse_pct') / 100
        sp = np.sqrt(A(s, m, 'spread_ratio')) * rm
        print(f'd={s} {m:10s} slope log spread on log RMSE_regressor {np.polyfit(np.log(rr), np.log(sp), 1)[0]:.3f}')
# CRPS normalised by the regressor's mean |error| scale: crps / rmse_r (a scale-free per-cube CRPS)
for s in (8, 4):
    rr = A(s, 'regressor', 'rmse_pct')
    for m in ('diffusion', 'varhead'):
        print(f'd={s} {m:10s} mean CRPS/RMSE_r {np.mean(A(s, m, "crps") / rr):.4f}')
    g = (A(s, 'diffusion', 'crps') - A(s, 'varhead', 'crps')) / rr
    print(f'd={s} gap/RMSE_r {g.mean():.4f} ± {cse(g)[0]:.4f}')
g4 = (A(4, 'diffusion', 'crps') - A(4, 'varhead', 'crps')) / A(4, 'regressor', 'rmse_pct')
g8 = (A(8, 'diffusion', 'crps') - A(8, 'varhead', 'crps')) / A(8, 'regressor', 'rmse_pct')
print('gap/RMSE_r d4 - d8', (g4 - g8).mean(), cse(g4 - g8)[0])
print('d4: diffusion CRPS > varhead on', int(np.sum(A(4, 'diffusion', 'crps') > A(4, 'varhead', 'crps'))), '/200;',
      'diffusion AUSE > varhead on', int(np.sum(A(4, 'diffusion', 'ause') > A(4, 'varhead', 'ause'))))
print('d4 varhead C2 would be', A(4, 'varhead', 'cov50').mean(), A(4, 'varhead', 'cov90').mean())
print('d4 B C2', A(4, 'b', 'cov50').mean(), A(4, 'b', 'cov90').mean())
# fraction of cubes where single sample over-sharpens / regressor blurs
for s in (8, 4):
    print(f'd={s}: single hf > 0 on {int(np.sum(A(s, "diffusion", "single_hf") > 0))}/200, regressor hf < 0 on {int(np.sum(A(s, "regressor", "hf") < 0))}/200; '
          f'exp(mean single hf) {math.exp(A(s, "diffusion", "single_hf").mean()):.2f}x truth, exp(mean reg hf) 1/{math.exp(-A(s, "regressor", "hf").mean()):.1f}; '
          f'single W1 > regressor W1 on {int(np.sum(A(s, "diffusion", "single_w1") > A(s, "regressor", "w1")))}/200; '
          f'SAM single {A(s, "diffusion", "single_sam_deg").mean():.2f} reg {A(s, "regressor", "sam_deg").mean():.2f}')
