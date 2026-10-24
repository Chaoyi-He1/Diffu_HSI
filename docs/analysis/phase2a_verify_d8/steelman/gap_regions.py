import math, sys
import numpy as np
sys.path.insert(0, '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan')
from scripts.uq import metrics as M
OUT = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d8/steelman/'
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
z = np.load(R + '/uq/d8/' + 'ev' + 'al/d8_main__cubes.npz')
S = (z['samples'].astype(np.float64) + 1) / 2
x01, xr01 = (z['x'].astype(np.float64) + 1) / 2, (z['x_r'].astype(np.float64) + 1) / 2
sv01 = np.load(OUT + 'varhead_sigma_stored.npy').astype(np.float64) / 2
sigma_r = 0.08934239400551727
r = (z['x'].astype(np.float64) - z['x_r'].astype(np.float64)) / sigma_r
print('max |r_true| over stored cubes', np.abs(r).max(), ' 99.99% quantile', np.quantile(np.abs(r), 0.9999), ' x0 clamp bound', 2 / sigma_r)
rs = (z['samples'].astype(np.float64) - z['x_r'].astype(np.float64)[None]) / sigma_r
print('max |r_sample|', np.abs(rs).max())
grid = np.zeros((64, 64), bool); grid[::8, ::8] = True
edge = np.zeros((64, 64), bool); edge[57:, :] = True; edge[:, 57:] = True
inner = ~grid & ~edge
cd = np.stack([M.crps_samples(S[:, i], x01[i]) for i in range(8)])
cv = np.stack([M.crps_gaussian(xr01[i], sv01[i], x01[i]) for i in range(8)])
tot = 100 * (cd - cv).mean()
print(f'C1 gap on 8 stored cubes: {tot:+.4f} (diffusion {100 * cd.mean():.4f}, varhead {100 * cv.mean():.4f})')
for nm, m in (('grid', grid), ('inner', inner), ('edge', edge)):
    frac = m.mean()
    contrib = 100 * (cd - cv)[..., m].mean() * frac
    print(f'  {nm:5s}: element fraction {frac:.4f}  CRPS diff {100 * cd[..., m].mean():.4f} vh {100 * cv[..., m].mean():.4f}  contribution to gap {contrib:+.4f} ({contrib / tot:.1%})')
