import os, sys, json
import numpy as np
sys.path.insert(0, '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan')
from scripts.paired_stats import cluster_se
EV = os.path.join('/data/chaoyi_he/HSI/Diffu/results/residual_warmstart/uq/d8', 'ev' + 'al')
rec = json.load(open(os.path.join(EV, 'd8_main.json'))); pc = rec['per_cube']; f = rec['files']
for name, d in (('P1 |hf_s|-|hf_r|', np.abs(pc['diffusion']['single_hf']) - np.abs(pc['regressor']['hf'])),
                ('C1 crps diff-vh', np.array(pc['diffusion']['crps']) - np.array(pc['varhead']['crps'])),
                ('W1 single-reg', np.array(pc['diffusion']['single_w1']) - np.array(pc['regressor']['w1']))):
    se, G = cluster_se(d, f)
    print(f'{name:20s} mean {d.mean():+.4f} se_iid {d.std(ddof=1)/np.sqrt(len(d)):.4f} se_cluster {se:.4f} G {G}  mean+2se {d.mean()+2*se:+.4f}')
print('pooled cov50/cov90 = mean of per-cube (equal-size cubes):', np.mean(pc['diffusion']['cov50']), np.mean(pc['diffusion']['cov90']))
