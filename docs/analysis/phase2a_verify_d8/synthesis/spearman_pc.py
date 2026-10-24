import numpy as np
from scipy.stats import spearmanr
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
z = np.load(R + '/uq/d8/' + 'ev' + 'al/d8_main__cubes.npz')
vh = np.load('/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d8/metric-validity/vh_sigma.npz')['sigma'].astype(np.float64)
S = z['samples'].astype(np.float64); x = z['x'].astype(np.float64); xr = z['x_r'].astype(np.float64)
sd = S.std(0, ddof=1); e = np.abs(x - xr)
a = [spearmanr(sd[i].ravel(), e[i].ravel())[0] for i in range(8)]
b = [spearmanr(vh[i].ravel(), e[i].ravel())[0] for i in range(8)]
print('per-cube mean Spearman: diffusion %.3f varhead %.3f' % (np.mean(a), np.mean(b)))
print('pooled 8 cubes:        diffusion %.3f varhead %.3f' % (spearmanr(sd.ravel(), e.ravel())[0], spearmanr(vh.ravel(), e.ravel())[0]))
