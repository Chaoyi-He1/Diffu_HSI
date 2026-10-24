"""CRPS of the 8 analysed cubes: diffusion under each condition (CPU samples), the stored GPU K = 32, the variance head,
and the best constant-width homoscedastic Gaussian (c = 0.5, from ref_c.json)."""
import json
import os
import sys

REPO = '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan'
sys.path.insert(0, REPO)
import numpy as np
from scripts.uq import metrics as M

OUT = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d8/condition-blindness'
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
b = np.load(f'{OUT}/base.npz')
x, xr, svh = (b[k].astype(np.float64) for k in ('x', 'xr', 'sig_vh'))
sigma_r = float(b['sigma_r'])
files = [str(f) for f in b['files']]
rec = json.load(open(f'{R}/uq/d8/eval/d8_main.json'))
idx = [rec['files'].index(f) for f in files]
rc = json.load(open(f'{OUT}/ref_c.json'))
x01 = M.to01(x)
N = x.shape[0]


def crps_of(r):
    s = np.clip(xr[None] + sigma_r * r, -1, 1)
    return np.array([100 * float(M.crps_samples(M.to01(s[:, i]), x01[i]).mean()) for i in range(N)])


rows = {}
for c in ('true', 'swap', 'zero', 'swap_y', 'swap_xr'):
    p = f'{OUT}/samples_{c}.npz'
    if os.path.exists(p):
        r = np.load(p)['r'].astype(np.float64)
        rows[f'{c} K={r.shape[0]}'] = crps_of(r)
        if c == 'true':
            rows['true K=4'] = crps_of(r[:4])
rows['varhead'] = np.array([rec['per_cube']['varhead']['crps'][j] for j in idx])
rows['homo c=0.5'] = np.array([rc['0.5']['crps'][j] for j in idx])
rows['homo c=1.0'] = np.array([rc['0.5']['crps'][j] for j in idx]) * 0 + np.nan
for k, v in rows.items():
    if np.all(np.isfinite(v)):
        print(f'{k:14s} mean {v.mean():.4f}  per cube ' + ' '.join(f'{u:.3f}' for u in v))
t = rows['true K=8']
for k, v in rows.items():
    if k != 'true K=8' and np.all(np.isfinite(v)):
        d = v - t
        print(f'{k:14s} - true K=8: {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(N):.4f} (iid SE, 8 cubes)')
