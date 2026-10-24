"""Best constant-width homoscedastic Gaussian N(x_r, (c sigma_r)^2) on the 200 gate cubes (fine grid of c); the oracle-tuned
condition-blind reference for the diffusion's CRPS, coverage and spread ratio."""
import json
import os
import sys

REPO = '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan'
sys.path.insert(0, REPO)
import numpy as np
import torch

torch.set_num_threads(int(os.environ.get('NT', '8')))
from data_loader.warmstart_paths import DATA_ROOT, PRIOR
from model.residual_wrapper import build_residual_model, compose, regress_forward
from scripts.eval_warmstart import dataset_for, load_subset, make_eval_loader
from scripts.uq import metrics as M
from scripts.paired_stats import cluster_se

OUT = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d8/condition-blindness'
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
ck = torch.load(f'{R}/runs/regress_l1_sel_d8_b64/checkpoint_epoch_11.pth', map_location='cpu', weights_only=False)
centre = build_residual_model(ck['meta']['mode'], int(ck['meta']['base_channels']))
centre.load_state_dict(ck['model_state_dict'], strict=True)
centre.eval()
sigma_r = 0.08934239400551727
ds = dataset_for(DATA_ROOT, PRIOR, 8, load_subset(DATA_ROOT, 'files'))
loader = make_eval_loader(ds, 4, num_workers=2)
CS = (0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55)
acc = {c: dict(crps=[], cov50=[], cov90=[]) for c in CS}
with torch.no_grad():
    for x, y, xh in loader:
        xr = compose(xh, float(ds.sigma_d), regress_forward(centre, {'sensor': y, 'x0_hat': xh}))
        mu01, x01 = M.to01(xr), M.to01(x)
        for i in range(x.shape[0]):
            for c in CS:
                s = np.full_like(mu01[i], c * sigma_r / 2)
                acc[c]['crps'].append(100.0 * float(M.crps_gaussian(mu01[i], s, x01[i]).mean()))
                g = M.coverage_gaussian(mu01[i], s, x01[i]); acc[c]['cov50'].append(g[0.5]); acc[c]['cov90'].append(g[0.9])
rec = json.load(open(f'{R}/uq/d8/eval/d8_main.json'))
dc = np.array(rec['per_cube']['diffusion']['crps'])
for c in CS:
    d = dc - np.array(acc[c]['crps'])
    print(f'c={c:.2f}: crps {np.mean(acc[c]["crps"]):.4f} cov50 {np.mean(acc[c]["cov50"]):.3f} cov90 {np.mean(acc[c]["cov90"]):.3f} | '
          f'diffusion - homo: {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f} (iid) {cluster_se(d, rec["files"])[0]:.4f} (cluster), frac cubes diffusion better {np.mean(d < 0):.2f}')
json.dump({str(c): v for c, v in acc.items()}, open(f'{OUT}/ref_c.json', 'w'))
