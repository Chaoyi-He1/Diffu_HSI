"""How much S2 can move at all: RMSE of a mean of K draws around x_r depends on the condition only through the cube's
total predictive variance, E RMSE^2 = RMSE_reg^2 + V / K (no spatial alignment enters). Predict the S2 statistic
(swap_full - true on the 32 y-swap cubes, roll 16) from per-cube V of (a) the diffusion itself, (b) a sampler with the
variance head's spread, (c) an oracle sampler whose V equals the cube's own squared error."""
import json
import os
import sys

REPO = '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan'
sys.path.insert(0, REPO)
import numpy as np
from scripts.paired_stats import cluster_se

R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
rec = json.load(open(f'{R}/uq/d8/eval/d8_main.json'))
pc, ys = rec['per_cube'], rec['yswap']
files = ys['files']
idx = [rec['files'].index(f) for f in files]
B, K = len(files), int(ys['K'])
reg = np.array(ys['regressor'])
print('y-swap regressor RMSE matches per_cube:', np.allclose(reg, np.array(pc['regressor']['rmse_pct'])[idx], atol=0.02))


def V_of(m):
    sr = np.array(pc[m]['spread_ratio'])[idx]
    rm = np.array(pc[m]['mean_rmse_pct'])[idx]
    return sr * rm ** 2                                          # mean predictive variance per element, (% of range)^2


src = [(i - B // 2) % B for i in range(B)]
obs = np.array(ys['swap_full']) - np.array(ys['true'])
print(f'observed S2: mean {obs.mean():+.4f} SE_iid {obs.std(ddof=1) / np.sqrt(B):.4f}; true mean-of-32 {np.mean(ys["true"]):.4f}')
for name, V in (('diffusion V', V_of('diffusion')), ('varhead V', V_of('varhead')),
                ('oracle V = RMSE_reg^2', reg ** 2)):
    pred_true = np.sqrt(reg ** 2 + V / K)
    pred_swap = np.sqrt(reg ** 2 + V[src] / K)
    d = pred_swap - pred_true
    print(f'{name:24s}: predicted S2 mean {d.mean():+.4f} SE_iid {d.std(ddof=1) / np.sqrt(B):.4f}  '
          f'(pass needs mean - 2 SE > 0: {d.mean() - 2 * d.std(ddof=1) / np.sqrt(B):+.4f}); predicted true mean-of-32 {pred_true.mean():.4f}')
