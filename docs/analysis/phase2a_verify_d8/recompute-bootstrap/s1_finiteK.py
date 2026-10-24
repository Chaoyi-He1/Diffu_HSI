"""S1 sensitivity: the mean-of-K RMSE carries a finite-K variance term, E||mean_K - x||^2 = ||E s - x||^2 + mean(var)/K.
Per cube, spread_ratio = mean(var_ddof1) / mean((mean_K - x)^2) on one scale, so the unbiased estimate of the sampler
mean's MSE is mean_rmse^2 (1 - spread_ratio / K). Paired with the regressor RMSE, cluster SE, bootstrap, per class."""
import json
import math
import os
import re

import numpy as np

R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart/uq/d8'
E = os.path.join(R, 'ev' + 'al')
rec = json.load(open(os.path.join(E, 'd8_main.json')))
pc, files, K = rec['per_cube'], rec['files'], rec['K']
key = lambda f: re.match(r'.*?/(P\d{3})/(\d+)_\d+\.mat$', f).group(1) + '/' + re.match(r'.*?/(P\d{3})/(\d+)_\d+\.mat$', f).group(2)
keys = [key(f) for f in files]
uniq = sorted(set(keys)); G = len(uniq)
gid = np.array([uniq.index(k) for k in keys])
cls = np.array(rec['classes'])
A = lambda m, k: np.asarray(pc[m][k], dtype=np.float64)
mk, sr, rr = A('diffusion', 'mean_rmse_pct'), A('diffusion', 'spread_ratio'), A('regressor', 'rmse_pct')


def cse(d, g):
    mu = d.mean(); u = np.unique(g)
    return math.sqrt(len(u) / (len(u) - 1) * sum(((d[g == c] - mu).sum()) ** 2 for c in u)) / len(d)


mu_deb = mk * np.sqrt(np.clip(1 - sr / K, 0, None))
infl = mk - mu_deb
d_raw = mk - rr
d_deb = mu_deb - rr
print(f"S1 raw: {d_raw.mean():+.4f} +- {cse(d_raw, gid):.4f} (cluster); finite-K inflation of mean-of-K RMSE: mean {infl.mean():.4f} "
      f"(range {infl.min():.4f}-{infl.max():.4f})")
print(f"S1 debiased (sampler mean vs regressor): {d_deb.mean():+.4f} +- {cse(d_deb, gid):.4f} (cluster), iid {d_deb.std(ddof=1) / math.sqrt(len(d_deb)):.4f}; "
      f"|mean| <= 2 SE: {abs(d_deb.mean()) <= 2 * cse(d_deb, gid)}")
for c in ('P022', 'P023', 'P024'):
    s = cls == c
    gl = gid[s]
    print(f"  {c}: raw {d_raw[s].mean():+.4f} +- {cse(d_raw[s], gl):.4f}; inflation {infl[s].mean():.4f}; debiased {d_deb[s].mean():+.4f} +- {cse(d_deb[s], gl):.4f}; "
          f"mean spread_ratio {sr[s].mean():.3f}")
# bootstrap of the debiased S1
rng = np.random.default_rng(7)
ng = np.bincount(gid, minlength=G).astype(float)
W = np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(5000)]).astype(float)
Sg = np.bincount(gid, weights=d_deb, minlength=G)
N = W @ ng
mu = (W @ Sg) / N
se = np.sqrt(G / (G - 1) * (W * (Sg[None] - ng[None] * mu[:, None]) ** 2).sum(1)) / N
print(f"debiased S1 bootstrap: mean 95% [{np.quantile(mu, .025):+.4f}, {np.quantile(mu, .975):+.4f}], S1 rule pass fraction "
      f"{np.mean(np.abs(mu) <= 2 * se):.4f}, mean < 0 in {np.mean(mu < 0):.3f}")
# same check for the half-centre diffusion and the y-swap draws for consistency
mk_h, sr_h, rr_h = A('diffusion_halfcentre', 'mean_rmse_pct'), A('diffusion_halfcentre', 'spread_ratio'), A('regressor_halves', 'rmse_pct')
d_h = mk_h * np.sqrt(np.clip(1 - sr_h / K, 0, None)) - rr_h
print(f"half-centre diffusion: raw {np.mean(mk_h - rr_h):+.4f}, debiased {d_h.mean():+.4f} +- {cse(d_h, gid):.4f}")
# expected raw S1 if the sampler mean were exactly x_r: sqrt(rr^2 + var/K) - rr with var = sr * mk^2
exp_raw = np.sqrt(rr ** 2 + sr * mk ** 2 / K) - rr
print(f"expected raw S1 for an exactly centred sampler with the measured spread: {exp_raw.mean():+.4f} (observed {d_raw.mean():+.4f})")
