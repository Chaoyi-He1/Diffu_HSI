"""Weighting variants of each clause (cube-weighted as pre-registered, image-weighted, class-balanced) and a check that
the samples' Laplacian energy behaves like draw-independent noise around the regressor."""
import json
import math
import os
import re

import numpy as np

R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart/uq/d8'
E = os.path.join(R, 'ev' + 'al')
rec = json.load(open(os.path.join(E, 'd8_main.json')))
pc, files, K = rec['per_cube'], rec['files'], rec['K']
pat = re.compile(r'.*?/(P\d{3})/(\d+)_\d+\.mat$')
keys = [pat.match(f).group(1) + '/' + pat.match(f).group(2) for f in files]
uniq = sorted(set(keys)); G = len(uniq)
gid = np.array([uniq.index(k) for k in keys])
cls = np.array(rec['classes'])
A = lambda m, k: np.asarray(pc[m][k], dtype=np.float64)
D = {'C1': A('diffusion', 'crps') - A('varhead', 'crps'),
     'P1': np.abs(A('diffusion', 'single_hf')) - np.abs(A('regressor', 'hf')),
     'P2': A('diffusion', 'single_rmse_pct') - A('baseline', 'rmse_pct'),
     'S1': A('diffusion', 'mean_rmse_pct') - A('regressor', 'rmse_pct'),
     'cov90': A('diffusion', 'cov90'), 'cov50': A('diffusion', 'cov50')}
rng = np.random.default_rng(11)
NB = 5000
g_cls = np.array([u.split('/')[0] for u in uniq])


def img_means(d):
    return np.bincount(gid, weights=d, minlength=G) / np.bincount(gid, minlength=G)


for k, d in D.items():
    im = img_means(d)
    # image-weighted: mean of per-image means; SE = sd of image means / sqrt(G)
    iw, iw_se = im.mean(), im.std(ddof=1) / math.sqrt(G)
    # class-balanced: mean of the three class means; SE from within-class cluster SEs
    cm, cv = [], []
    for c in ('P022', 'P023', 'P024'):
        s = cls == c
        dc, gl = d[s], gid[s]
        mu = dc.mean(); u = np.unique(gl)
        se = math.sqrt(len(u) / (len(u) - 1) * sum(((dc[gl == g] - mu).sum()) ** 2 for g in u)) / len(dc)
        cm.append(mu); cv.append(se ** 2)
    cb, cb_se = float(np.mean(cm)), math.sqrt(sum(cv)) / 3
    # stratified (by class) image bootstrap of the class-balanced mean
    bs = np.zeros(NB)
    for c in ('P022', 'P023', 'P024'):
        gs = np.where(g_cls == c)[0]
        S = np.bincount(gid, weights=d, minlength=G)[gs]; n_ = np.bincount(gid, minlength=G)[gs].astype(float)
        Wc = np.stack([np.bincount(rng.integers(0, len(gs), len(gs)), minlength=len(gs)) for _ in range(NB)]).astype(float)
        bs += (Wc @ S) / (Wc @ n_) / 3
    if k in ('C1', 'P1', 'P2'):
        rule = lambda m, s: m + 2 * s < 0
    elif k == 'S1':
        rule = lambda m, s: abs(m) <= 2 * s
    else:
        rule = None
    tag = (lambda m, s: '' if rule is None else ('PASS' if rule(m, s) else 'FAIL'))
    print(f"{k:5s} cube-weighted {d.mean():+.4f} | image-weighted {iw:+.4f} +- {iw_se:.4f} {tag(iw, iw_se)} | class-balanced {cb:+.4f} +- {cb_se:.4f} {tag(cb, cb_se)} "
          f"(stratified boot 95% [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}])")

# Laplacian-energy additivity: E_L(sample) = E_L(x_r) + E_L(noise); E_L(mean_K) = E_L(x_r) + E_L(noise) / K if the HF part is draw-independent
hr, hs, hm = A('regressor', 'hf'), A('diffusion', 'single_hf'), A('diffusion', 'mean_hf')
pred = np.log(np.exp(hr) + np.maximum(np.exp(hs) - np.exp(hr), 0) / K)
print(f"\nhf of mean-of-K predicted from (hf_single, hf_regressor) under draw-independent HF noise: mean {pred.mean():+.4f} vs recorded {hm.mean():+.4f}; "
      f"per-cube corr {np.corrcoef(pred, hm)[0, 1]:.3f}, mean abs diff {np.mean(np.abs(pred - hm)):.3f}")
print(f"single-sample hf > 0 in {(hs > 0).sum()}/200 cubes; regressor hf < 0 in {(hr < 0).sum()}/200; |hf_single| < |hf_reg| in {(np.abs(hs) < np.abs(hr)).sum()}/200")
print(f"single-sample Laplacian energy / truth: geometric mean {math.exp(hs.mean()):.1f}x; regressor {math.exp(hr.mean()):.3f}x")
