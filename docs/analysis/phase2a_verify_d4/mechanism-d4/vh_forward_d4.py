"""CPU forward of the d = 4 variance head (varhead_d4_b64@12) on the 8 stored d = 4 cubes; saves sigma on the [-1,1] scale.
Adapted from docs/analysis/phase2a_verify_d8/metric-validity/vh_forward.py (stride 8 -> 4, d8 -> d4 paths)."""
import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '8')
import numpy as np, torch
torch.set_num_threads(8)
REPO = '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan'
sys.path.insert(0, REPO)
from data_loader.hfd_residual import residual_collate_fn
from data_loader.warmstart_paths import DATA_ROOT, PRIOR
from model.residual_wrapper import LOGVAR_RANGE, build_residual_model, varhead_forward
from scripts.eval_warmstart import dataset_for
OUT = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d4/mechanism-d4'
RR = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
z = np.load(os.path.join(RR, 'uq/d4', 'ev' + 'al', 'd4_contrast__cubes.npz'))
files = [os.path.join(DATA_ROOT, f) for f in z['files']]
ds = dataset_for(DATA_ROOT, PRIOR, 4, files)
order = [os.path.relpath(f, DATA_ROOT) for f in ds.img_list]
idx = [order.index(f) for f in z['files']]
x, y, xh = residual_collate_fn([ds[i] for i in idx])
print('y shape', tuple(y.shape))
print('x match', float(np.abs(x.numpy() - z['x']).max()), 'xh match', float(np.abs(xh.numpy() - z['x0_hat']).max()))
ck = torch.load(os.path.join(RR, 'runs/varhead_d4_b64/checkpoint_epoch_12.pth'), map_location='cpu', weights_only=False)
meta = ck.get('meta') or {}
print('meta', {k: meta.get(k) for k in ('mode', 'd', 'sigma_r', 'sigma_d', 'base_channels', 'epoch')})
vh = build_residual_model(meta['mode'], int(meta['base_channels']))
vh.load_state_dict(ck['model_state_dict'], strict=True); vh.eval()
sig_r = float(meta['sigma_r'])
t0 = time.time()
lv_all = []
with torch.no_grad():
    for i in range(x.shape[0]):
        xr_s = torch.from_numpy(z['x_r'][i:i + 1]).float()
        lv = varhead_forward(vh, {'sensor': y[i:i + 1], 'x0_hat': xr_s}).clamp(*LOGVAR_RANGE)
        lv_all.append(lv.numpy())
        print(i, f'{time.time() - t0:.0f}s', flush=True)
lv_all = np.concatenate(lv_all)
sigma = sig_r * np.exp(0.5 * lv_all)
np.savez(os.path.join(OUT, 'vh_sigma_d4.npz'), sigma=sigma.astype(np.float32), sigma_r=sig_r, y=y.numpy().astype(np.float32))
print('saved; sigma range', sigma.min(), sigma.max(), 'median', np.median(sigma))
