"""Per spectral band coverage: diffusion vs variance head (8 stored cubes)."""
import os, sys
import numpy as np
sys.path.insert(0, '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan')
from scripts.uq import metrics as M
SCR = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d8/metric-validity'
EV = os.path.join('/data/chaoyi_he/HSI/Diffu/results/residual_warmstart/uq/d8', 'ev' + 'al')
z = np.load(os.path.join(EV, 'd8_main__cubes.npz'))
S = (z['samples'].astype(np.float64) + 1) / 2; X = (z['x'].astype(np.float64) + 1) / 2; XR = (z['x_r'].astype(np.float64) + 1) / 2
SIG = np.load(os.path.join(SCR, 'vh_sigma.npz'))['sigma'].astype(np.float64) / 2
q = np.quantile(S, [0.25, 0.75, 0.05, 0.95], axis=0, method='weibull')
c50 = (X >= q[0]) & (X <= q[1]); c90 = (X >= q[2]) & (X <= q[3])
E = np.abs(X - XR); v50 = E <= M.Z[0.5] * SIG; v90 = E <= M.Z[0.9] * SIG
SD = S.std(0, ddof=1)
print(' band | RMS err | diff sd  cov50 cov90 | vh sig  cov50 cov90')
for b in range(64):
    print(f' {b:4d} | {np.sqrt((E[:, b]**2).mean()):.4f} | {SD[:, b].mean():.4f} {c50[:, b].mean():.3f} {c90[:, b].mean():.3f} | {SIG[:, b].mean():.4f} {v50[:, b].mean():.3f} {v90[:, b].mean():.3f}')
