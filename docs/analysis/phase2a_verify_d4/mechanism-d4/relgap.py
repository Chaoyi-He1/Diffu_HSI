import json, os, sys
import numpy as np
sys.path.insert(0, '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan')
from scripts.paired_stats import cluster_se
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
for tag, p in (('d4', os.path.join(R, 'uq/d4', 'ev' + 'al', 'd4_contrast.json')), ('d8', os.path.join(R, 'uq/d8', 'ev' + 'al', 'd8_main.json'))):
    rec = json.load(open(p)); pc = rec['per_cube']; f = rec['files']
    cd, cv = np.array(pc['diffusion']['crps']), np.array(pc['varhead']['crps'])
    rr = cd / cv - 1
    reg = np.array(pc['regressor']['rmse_pct']); sing = np.array(pc['diffusion']['single_rmse_pct'])
    w1s, w1r = np.array(pc['diffusion']['single_w1']), np.array(pc['regressor']['w1'])
    print(tag, f'rel CRPS gap {rr.mean():.4f} se_cl {cluster_se(rr, f)[0]:.4f}; gap/reg RMSE {np.mean((cd - cv) / reg):.4f}; '
          f'single/reg RMSE {np.mean(sing / reg):.3f}; W1 single/reg {np.mean(w1s) / np.mean(w1r):.3f}, single worse in {np.mean(w1s > w1r):.3f}; '
          f'diff cov50 {np.mean(pc["diffusion"]["cov50"]):.3f} cov90 {np.mean(pc["diffusion"]["cov90"]):.3f} sr {np.mean(pc["diffusion"]["spread_ratio"]):.3f}')
