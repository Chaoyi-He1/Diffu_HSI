"""CPU forward of the variance head (and the live centre) on the 8 stored cubes; saves sigma on [-1,1] scale."""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '6')
import numpy as np, torch
torch.set_num_threads(6)
REPO = '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan'
sys.path.insert(0, REPO)
from data_loader.hfd_residual import residual_collate_fn
from data_loader.warmstart_paths import DATA_ROOT, PRIOR
from model.residual_wrapper import LOGVAR_RANGE, build_residual_model, compose, regress_forward, varhead_forward
from scripts.eval_warmstart import dataset_for
OUT = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d8/metric-validity'
RR = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
z = np.load(os.path.join(RR, 'uq/d8', 'ev' + 'al', 'd8_main__cubes.npz'))
files = [os.path.join(DATA_ROOT, f) for f in z['files']]
ds = dataset_for(DATA_ROOT, PRIOR, 8, files)
order = [os.path.relpath(f, DATA_ROOT) for f in ds.img_list]
print('ds order', order); print('npz order', list(z['files']))
x, y, xh = residual_collate_fn([ds[i] for i in range(len(ds))])
idx = [order.index(f) for f in z['files']]
x, y, xh = x[idx], y[idx], xh[idx]
print('x match', float(np.abs(x.numpy() - z['x']).max()), 'xh match', float(np.abs(xh.numpy() - z['x0_hat']).max()))
def load(p):
    ck = torch.load(p, map_location='cpu', weights_only=False); meta = ck.get('meta') or {}
    m = build_residual_model(meta['mode'], int(meta['base_channels'])); m.load_state_dict(ck['model_state_dict'], strict=True)
    return m.eval(), meta
t0 = time.time()
cen, mc = load(os.path.join(RR, 'runs/regress_l1_sel_d8_b64/checkpoint_epoch_11.pth'))
vh, mv = load(os.path.join(RR, 'runs/varhead_d8_b64/checkpoint_epoch_10.pth'))
print('meta centre', mc.get('mode'), mc.get('sigma_d'), 'varhead', mv.get('mode'), mv.get('sigma_r'), mv.get('base_channels'))
sig_d = float(mc['sigma_d']); sig_r = float(mv['sigma_r'])
with torch.no_grad():
    xr_live, lv_all = [], []
    for i in range(x.shape[0]):
        xr_i = compose(xh[i:i+1], sig_d, regress_forward(cen, {'sensor': y[i:i+1], 'x0_hat': xh[i:i+1]}))
        xr_s = torch.from_numpy(z['x_r'][i:i+1])
        lv = varhead_forward(vh, {'sensor': y[i:i+1], 'x0_hat': xr_s}).clamp(*LOGVAR_RANGE)
        xr_live.append(xr_i.numpy()); lv_all.append(lv.numpy())
        print(i, 'xr live vs stored max abs', float(np.abs(xr_i.numpy() - z['x_r'][i:i+1]).max()), f'{time.time()-t0:.0f}s', flush=True)
xr_live = np.concatenate(xr_live); lv_all = np.concatenate(lv_all)
sigma = sig_r * np.exp(0.5 * lv_all)
np.savez(os.path.join(OUT, 'vh_sigma.npz'), sigma=sigma.astype(np.float32), xr_live=xr_live.astype(np.float32), sigma_r=sig_r)
print('saved; sigma range', sigma.min(), sigma.max(), 'median', np.median(sigma))
