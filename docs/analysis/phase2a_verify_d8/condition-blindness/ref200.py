"""Condition-blind reference on all 200 gate cubes: a homoscedastic Gaussian N(x_r, (c sigma_r)^2) around the same centre,
scored with the evaluator's metric functions; plus varhead sigma vs |x - x_r| rank correlations; regressor RMSE check
against the record."""
import json
import os
import sys
import time

REPO = '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan'
sys.path.insert(0, REPO)
import numpy as np
import torch
from scipy.stats import spearmanr

torch.set_num_threads(int(os.environ.get('NT', '4')))
from data_loader.warmstart_paths import DATA_ROOT, PRIOR
from model.residual_wrapper import LOGVAR_RANGE, build_residual_model, compose, regress_forward, varhead_forward
from scripts.eval_warmstart import dataset_for, load_subset, make_eval_loader, metrics
from scripts.uq import metrics as M

OUT = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d8/condition-blindness'
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
t0 = time.time()


def load(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    meta = ck['meta']
    m = build_residual_model(meta['mode'], int(meta['base_channels']))
    m.load_state_dict(ck['model_state_dict'], strict=True)
    return m.eval(), meta


centre, _ = load(f'{R}/runs/regress_l1_sel_d8_b64/checkpoint_epoch_11.pth')
vh, mv = load(f'{R}/runs/varhead_d8_b64/checkpoint_epoch_10.pth')
sigma_r = float(mv['sigma_r'])
files = load_subset(DATA_ROOT, 'files')
lim = int(os.environ.get('LIMIT', '0'))
if lim:
    files = files[:lim]
ds = dataset_for(DATA_ROOT, PRIOR, 8, files)
loader = make_eval_loader(ds, 4, num_workers=2)
CS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)
acc = {k: [] for k in ['reg_rmse', 'vh_crps', 'vh_cov50', 'vh_cov90', 'ause_random', 'ause_vh', 'ause_oracle_abs',
                       'rho_vh_abs_elem', 'rho_vh_abs_pix', 'e_rms', 'vh_rms', 'cov50_homo', 'cov90_homo', 'sr_homo']
       + [f'crps_homo_{c}' for c in CS]}
rng = np.random.default_rng(0)
with torch.no_grad():
    for bi, (x, y, xh) in enumerate(loader):
        xr = compose(xh, float(ds.sigma_d), regress_forward(centre, {'sensor': y, 'x0_hat': xh}))
        lv = varhead_forward(vh, {'sensor': y, 'x0_hat': xr}).clamp(*LOGVAR_RANGE)
        s_vh = sigma_r * torch.exp(0.5 * lv)
        acc['reg_rmse'] += metrics(xr, x)['rmse_pct'].tolist()
        mu01, x01, svh01 = M.to01(xr), M.to01(x), s_vh.double().numpy() / 2.0
        for i in range(x.shape[0]):
            m_, t = mu01[i], x01[i]
            err = m_ - t
            acc['e_rms'].append(float(np.sqrt((err ** 2).mean()) * 2 / sigma_r))
            acc['vh_rms'].append(float(np.sqrt((svh01[i] ** 2).mean()) * 2 / sigma_r))
            acc['vh_crps'].append(100.0 * float(M.crps_gaussian(m_, svh01[i], t).mean()))
            c = M.coverage_gaussian(m_, svh01[i], t); acc['vh_cov50'].append(c[0.5]); acc['vh_cov90'].append(c[0.9])
            for cc in CS:
                acc[f'crps_homo_{cc}'].append(100.0 * float(M.crps_gaussian(m_, np.full_like(m_, cc * sigma_r / 2), t).mean()))
            c = M.coverage_gaussian(m_, np.full_like(m_, sigma_r / 2), t); acc['cov50_homo'].append(c[0.5]); acc['cov90_homo'].append(c[0.9])
            acc['sr_homo'].append(M.spread_ratio(np.full_like(m_, (sigma_r / 2) ** 2), err))
            acc['ause_random'].append(M.ause(rng.random(m_.shape), err))
            acc['ause_vh'].append(M.ause(svh01[i], err))
            acc['ause_oracle_abs'].append(M.ause(np.abs(err), err))
            acc['rho_vh_abs_elem'].append(float(spearmanr(svh01[i].ravel(), np.abs(err).ravel()).correlation))
            acc['rho_vh_abs_pix'].append(float(spearmanr(np.sqrt((svh01[i] ** 2).mean(0)).ravel(),
                                                         np.sqrt((err ** 2).mean(0)).ravel()).correlation))
        print(f'batch {bi + 1}/{len(loader)} {time.time() - t0:.0f}s', flush=True)
rec = json.load(open(f'{R}/uq/d8/eval/d8_main.json'))
pc = rec['per_cube']
n = len(acc['reg_rmse'])
out = {k: float(np.mean(v)) for k, v in acc.items()}
out['n'] = n
out['max_abs_reg_rmse_vs_record'] = float(np.max(np.abs(np.array(acc['reg_rmse']) - np.array(pc['regressor']['rmse_pct'][:n]))))
out['max_abs_vh_crps_vs_record'] = float(np.max(np.abs(np.array(acc['vh_crps']) - np.array(pc['varhead']['crps'][:n]))))
d = np.array(pc['diffusion']['crps'][:n]) - np.array(acc['crps_homo_1.0'])
out['diff_crps_minus_homo1_mean'] = float(d.mean()); out['diff_crps_minus_homo1_se'] = float(d.std(ddof=1) / np.sqrt(n))
out['diff_crps_minus_homo1_frac_cubes_better'] = float(np.mean(d < 0))
best = min(CS, key=lambda c: out[f'crps_homo_{c}'])
d2 = np.array(pc['diffusion']['crps'][:n]) - np.array(acc[f'crps_homo_{best}'])
out['best_c'] = best; out['diff_crps_minus_homo_best_mean'] = float(d2.mean()); out['diff_crps_minus_homo_best_se'] = float(d2.std(ddof=1) / np.sqrt(n))
out['record_diffusion_crps_mean'] = float(np.mean(pc['diffusion']['crps'][:n]))
out['record_diffusion_ause_mean'] = float(np.mean(pc['diffusion']['ause'][:n]))
out['record_diffusion_cov50_mean'] = float(np.mean(pc['diffusion']['cov50'][:n]))
out['record_diffusion_cov90_mean'] = float(np.mean(pc['diffusion']['cov90'][:n]))
json.dump(dict(means=out, per_cube=acc), open(f'{OUT}/ref200.json', 'w'))
for k, v in out.items():
    print(f'{k:40s} {v}')
