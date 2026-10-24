"""Post-hoc on the saved hybrid samples (8 stored cubes): paired CRPS vs the variance head per branch, and the effect
of one global spread inflation c about the sample mean (in-sample diagnostic, NOT a method)."""
import json, math, sys
import numpy as np
sys.path.insert(0, '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan')
from scripts.uq import metrics as M

OUT = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d8/steelman/'
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
z = np.load(R + '/uq/d8/' + 'ev' + 'al/d8_main__cubes.npz')
x01, xr01 = (z['x'].astype(np.float64) + 1) / 2, (z['x_r'].astype(np.float64) + 1) / 2
sv01 = np.load(OUT + 'varhead_sigma_stored.npy').astype(np.float64) / 2
S0 = (z['samples'].astype(np.float64) + 1) / 2
rows = [json.loads(l) for l in open(OUT + 'hybrid2_tsw399_pk8.jsonl')]
vh = np.array([100 * M.crps_gaussian(xr01[i], sv01[i], x01[i]).mean() for i in range(8)])
stored = np.array([100 * M.crps_samples(S0[:, i], x01[i]).mean() for i in range(8)])
print('varhead CRPS per cube', np.round(vh, 4), 'mean', vh.mean().round(4))
print(f'stored DDIM-10 (record run) mean {stored.mean():.4f}  gap vs vh {np.mean(stored - vh):+.4f} +- {np.std(stored - vh, ddof=1) / math.sqrt(8):.4f}')
for br in ('net', 'wpos', 'proj'):
    rr = sorted([r for r in rows if r['branch'] == br], key=lambda r: r['cube'])
    if len(rr) < 8:
        print(br, 'incomplete', len(rr)); continue
    c = np.array([r['crps'] for r in rr]); d = c - vh
    keys = ('cov50', 'cov90', 'ause', 'spread_ratio', 'sp_sd_err', 'rmse_meanK', 'rmse_single', 'hf_single', 'lag1_dev', 'bandc_dev', 'sd_grid')
    print(f'{br:5s} CRPS {c.mean():.4f}  diff vs vh {d.mean():+.4f} +- {d.std(ddof=1) / math.sqrt(8):.4f} (iid over 8 cubes); cubes better than vh: {int((d < 0).sum())}/8; '
          + ' '.join(f'{k}={np.mean([r[k] for r in rr]):.4f}' for k in keys))
for br in ('wpos', 'proj'):
    S = [(np.load(OUT + f'hyb_{br}_tsw399_cube{i}.npy').astype(np.float64) + 1) / 2 for i in range(8)]
    for cfac in (1.0, 1.25, 1.5, 1.75, 2.0):
        cr, c50, c90 = [], [], []
        for i in range(8):
            mu = S[i].mean(0)
            Si = np.clip(mu[None] + cfac * (S[i] - mu[None]), 0, 1)
            cr.append(100 * M.crps_samples(Si, x01[i]).mean()); cv = M.coverage_samples(Si, x01[i]); c50.append(cv[0.5]); c90.append(cv[0.9])
        d = np.array(cr) - vh
        print(f'  {br} inflate x{cfac:.2f}: CRPS {np.mean(cr):.4f}  diff vs vh {d.mean():+.4f} +- {d.std(ddof=1) / math.sqrt(8):.4f}  cov50 {np.mean(c50):.3f} cov90 {np.mean(c90):.3f}')
