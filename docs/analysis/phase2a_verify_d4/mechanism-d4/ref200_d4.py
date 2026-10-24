"""d = 4 adaptation of docs/analysis/phase2a_verify_d8/condition-blindness/ref200.py: on all 200 gate cubes, the
homoscedastic Gaussian N(x_r, (c sigma_r)^2) around the same centre, the varhead (CPU re-forward, checked against the
record), and the sensor-phase (i mod 4, j mod 4) error / varhead-sigma profile over 200 cubes. No diffusion sampling."""
import json, os, sys, time
REPO = '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan'
sys.path.insert(0, REPO)
import numpy as np
import torch
from scipy.stats import spearmanr

torch.set_num_threads(8)
from data_loader.warmstart_paths import DATA_ROOT, PRIOR
from model.residual_wrapper import LOGVAR_RANGE, build_residual_model, compose, regress_forward, varhead_forward
from scripts.eval_warmstart import dataset_for, load_subset, make_eval_loader, metrics
from scripts.uq import metrics as M
from scripts.paired_stats import cluster_se

OUT = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d4/mechanism-d4'
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
D = 4
t0 = time.time()


def load(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    meta = ck['meta']
    m = build_residual_model(meta['mode'], int(meta['base_channels']))
    m.load_state_dict(ck['model_state_dict'], strict=True)
    return m.eval(), meta


centre, _ = load(f'{R}/runs/regress_l1_sel_d4_b64/checkpoint_epoch_17.pth')
vh, mv = load(f'{R}/runs/varhead_d4_b64/checkpoint_epoch_12.pth')
sigma_r = float(mv['sigma_r'])
rec = json.load(open(os.path.join(R, 'uq/d4', 'ev' + 'al', 'd4_contrast.json')))
files = load_subset(DATA_ROOT, 'files')
ds = dataset_for(DATA_ROOT, PRIOR, D, files)
assert [os.path.relpath(f, DATA_ROOT) for f in ds.img_list] == rec['files'], 'order mismatch'
loader = make_eval_loader(ds, 4, num_workers=2)
CS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
acc = {k: [] for k in ['reg_rmse', 'vh_crps', 'vh_cov50', 'vh_cov90', 'ause_random', 'ause_vh',
                       'rho_vh_abs_elem', 'rho_vh_abs_pix'] + [f'crps_homo_{c}' for c in CS]}
H = W = 64
pos = (np.arange(H)[:, None] % D) * D + (np.arange(W)[None, :] % D)
interior = np.zeros((H, W), bool); interior[:61, :61] = True     # last sensor sample at pixel 60 (top-left-anchored grid)
ph_err = np.zeros(D * D); ph_vh = np.zeros(D * D); ph_n = np.zeros(D * D)
ph_err_i = np.zeros(D * D); ph_vh_i = np.zeros(D * D); ph_n_i = np.zeros(D * D)
rng = np.random.default_rng(0)
seen = []
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
            acc['vh_crps'].append(100.0 * float(M.crps_gaussian(m_, svh01[i], t).mean()))
            c = M.coverage_gaussian(m_, svh01[i], t); acc['vh_cov50'].append(c[0.5]); acc['vh_cov90'].append(c[0.9])
            for cc in CS:
                acc[f'crps_homo_{cc}'].append(100.0 * float(M.crps_gaussian(m_, np.full_like(m_, cc * sigma_r / 2), t).mean()))
            acc['ause_random'].append(M.ause(rng.random(m_.shape), err))
            acc['ause_vh'].append(M.ause(svh01[i], err))
            acc['rho_vh_abs_elem'].append(float(spearmanr(svh01[i].ravel(), np.abs(err).ravel()).correlation))
            acc['rho_vh_abs_pix'].append(float(spearmanr(np.sqrt((svh01[i] ** 2).mean(0)).ravel(),
                                                         np.sqrt((err ** 2).mean(0)).ravel()).correlation))
            e2 = (err ** 2).mean(0); v2 = (svh01[i] ** 2).mean(0)
            for p in range(D * D):
                mk = pos == p
                ph_err[p] += e2[mk].sum(); ph_vh[p] += v2[mk].sum(); ph_n[p] += mk.sum()
                mk = mk & interior
                ph_err_i[p] += e2[mk].sum(); ph_vh_i[p] += v2[mk].sum(); ph_n_i[p] += mk.sum()
        print(f'batch {bi + 1}/{len(loader)} {time.time() - t0:.0f}s', flush=True)
pc = rec['per_cube']
n = len(acc['reg_rmse'])
out = {k: float(np.mean(v)) for k, v in acc.items()}
out['n'] = n
out['max_abs_reg_rmse_vs_record'] = float(np.max(np.abs(np.array(acc['reg_rmse']) - np.array(pc['regressor']['rmse_pct'][:n]))))
out['max_abs_vh_crps_vs_record'] = float(np.max(np.abs(np.array(acc['vh_crps']) - np.array(pc['varhead']['crps'][:n]))))
best = min(CS, key=lambda c: out[f'crps_homo_{c}'])
d2 = np.array(pc['diffusion']['crps'][:n]) - np.array(acc[f'crps_homo_{best}'])
se_c, G = cluster_se(d2, rec['files'][:n])
out['best_c'] = best; out['diff_minus_homo_best_mean'] = float(d2.mean()); out['diff_minus_homo_best_se_iid'] = float(d2.std(ddof=1) / np.sqrt(n))
out['diff_minus_homo_best_se_cluster'] = float(se_c); out['diff_better_than_homo_frac'] = float(np.mean(d2 < 0))
dvh = np.array(acc[f'crps_homo_{best}']) - np.array(pc['varhead']['crps'][:n])
out['homo_best_minus_vh_mean'] = float(dvh.mean()); out['homo_best_minus_vh_se_cluster'] = float(cluster_se(dvh, rec['files'][:n])[0])
out['record_diffusion_crps_mean'] = float(np.mean(pc['diffusion']['crps'][:n]))
out['record_varhead_crps_mean'] = float(np.mean(pc['varhead']['crps'][:n]))
out['record_b_crps_mean'] = float(np.mean(pc['b']['crps'][:n]))
db = np.array(pc['b']['crps'][:n]) - np.array(acc[f'crps_homo_{best}'])
out['b_minus_homo_best_mean'] = float(db.mean()); out['b_minus_homo_best_se_cluster'] = float(cluster_se(db, rec['files'][:n])[0])
e_ph = np.sqrt(ph_err / ph_n); v_ph = np.sqrt(ph_vh / ph_n); e_phi = np.sqrt(ph_err_i / ph_n_i); v_phi = np.sqrt(ph_vh_i / ph_n_i)
out['phase_err_rms_all'] = e_ph.tolist(); out['phase_vh_rms_all'] = v_ph.tolist()
out['phase_err_rms_interior'] = e_phi.tolist(); out['phase_vh_rms_interior'] = v_phi.tolist()
json.dump(dict(means=out, per_cube=acc), open(f'{OUT}/ref200_d4.json', 'w'))
for k, v in out.items():
    if isinstance(v, list):
        print(f'{k:40s} ' + ' '.join(f'{a:.4f}' for a in v))
    else:
        print(f'{k:40s} {v}')
print('phase err max/min all %.2f interior %.2f; vh max/min all %.2f interior %.2f' % (e_ph.max() / e_ph.min(), e_phi.max() / e_phi.min(), v_ph.max() / v_ph.min(), v_phi.max() / v_phi.min()))
