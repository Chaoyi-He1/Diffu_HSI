"""Linear reference denoisers for the standardised residual r = (x - x_r)/sigma_r, fitted on cross-fitted TRAINING cubes
from the cache (no network, no gate data): (1) per-pixel spectral Wiener with the global 64x64 band covariance C;
(2) the same with a position-dependent variance g(i mod 8, j mod 8, edge) learnt from training cubes. Compared on the
8 stored gate cubes with the network at the same x_t (same eps), x0 RMS error in r units by region."""
import os, sys, time
import numpy as np
import torch

torch.set_num_threads(int(os.environ.get('NT', '5')))
REPO = '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan'
sys.path.insert(0, REPO)
from data_loader.warmstart_paths import DATA_ROOT, PRIOR
from data_loader.hfd_residual import HFDResidualData, residual_collate_fn
from data_loader.xr_cache import cache_path
from model.residual_wrapper import build_residual_model
from model.diffusion_trainer import DiffusionTrainer
from scripts.eval_warmstart import dataset_for

OUT = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d8/steelman/'
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
CACHE = R + '/uq/d8/xr_cache'
sigma_r = 0.08934239400551727

# ---- fit C and the position variance map on training cubes
tr_ds = HFDResidualData(DATA_ROOT, split='train', sensor_down_sample_rate=8, prior_path=PRIOR)
rng = np.random.default_rng(0)
idx = rng.choice(len(tr_ds.img_list), size=int(os.environ.get('NTRAIN', '200')), replace=False)
C = np.zeros((64, 64)); vmap = np.zeros((64, 64)); n = 0
t0 = time.time()
for j in idx:
    f = tr_ds.img_list[j]
    x64, _ = tr_ds.raw_item(j)                                             # [H, W, 64] on [-1, 1]
    xr = np.load(cache_path(CACHE, 'train', os.path.relpath(f, DATA_ROOT))).astype(np.float64)
    r = (x64.astype(np.float64) - xr) / sigma_r                            # [H, W, 64]
    p = r.reshape(-1, 64)
    C += p.T @ p; vmap += (r ** 2).mean(-1); n += p.shape[0]
C /= n; vmap /= len(idx)
print(f'fitted on {len(idx)} training cubes in {time.time() - t0:.0f} s; trace(C)/64 = {np.trace(C) / 64:.3f}')
lam, U = np.linalg.eigh(C); lam, U = lam[::-1].clip(min=1e-8), U[:, ::-1]
print('top-5 eigen fractions', np.round(lam[:5] / lam.sum(), 4), ' top-3 cum', round(float(lam[:3].sum() / lam.sum()), 4))
# position variance: phase (i mod 8, j mod 8) for the interior, own value on the extrapolated edge
g = np.zeros((64, 64))
ph = np.zeros((8, 8)); cnt = np.zeros((8, 8))
for i in range(57):
    for jj in range(57):
        ph[i % 8, jj % 8] += vmap[i, jj]; cnt[i % 8, jj % 8] += 1
ph /= cnt
for i in range(64):
    for jj in range(64):
        g[i, jj] = vmap[i, jj] if (i >= 57 or jj >= 57) else ph[i % 8, jj % 8]
g /= g.mean()
print('position variance: grid phase (0,0) =', round(float(ph[0, 0] / ph.mean()), 4), ' max phase', round(float(ph.max() / ph.mean()), 3),
      ' edge mean', round(float(g[57:, :].mean()), 3))

# ---- stored gate cubes and the network
z = np.load(R + '/uq/d8/' + 'ev' + 'al/d8_main__cubes.npz')
files = list(z['files'])
ds = dataset_for(DATA_ROOT, PRIOR, 8, [os.path.join(DATA_ROOT, f) for f in files])
order = [os.path.relpath(f, DATA_ROOT) for f in ds.img_list]
_, Y, _ = residual_collate_fn([ds[order.index(f)] for f in files])
X, XR = torch.from_numpy(z['x']).float(), torch.from_numpy(z['x_r']).float()
rt = ((X - XR) / sigma_r).double()                                         # [B, 64, H, W]
ck = torch.load(f'{R}/runs/residual_reg_d8_b64/checkpoint_epoch_100.pth', map_location='cpu', weights_only=False)
net = build_residual_model('residual_reg', 64); net.load_state_dict(ck['model_state_dict'], strict=True); net.eval()
trn = DiffusionTrainer(device=torch.device('cpu'), loss_type='l1', prediction_type='v', snr_gamma=None, n_timesteps=1000)
grid = np.zeros((64, 64), bool); grid[::8, ::8] = True
edge = np.zeros((64, 64), bool); edge[57:, :] = True; edge[:, 57:] = True
inner = ~grid & ~edge
Ut = torch.from_numpy(U.copy()); lamt = torch.from_numpy(lam.copy()); gt = torch.from_numpy(g)


def wiener(xt, a, s, pos):
    c = torch.einsum('bchw,ck->bkhw', xt, Ut)                               # coefficients in the eigenbasis
    L = lamt.view(1, -1, 1, 1) * (gt.view(1, 1, 64, 64) if pos else 1.0)
    c = c * (a * L / (a * a * L + s * s))
    return torch.einsum('bkhw,ck->bchw', c, Ut)


def rms(e, m):
    return float(e[..., torch.from_numpy(m)].pow(2).mean().sqrt())


gen = torch.Generator().manual_seed(123)
print('   t | region:  net / spectral-Wiener / position-aware spectral-Wiener / iid')
for t in (49, 99, 149, 199, 299, 399, 499):
    ab = trn.alpha_bars[t].double(); a, s = ab.sqrt(), (1 - ab).sqrt()
    eps = torch.randn(rt.shape, generator=gen, dtype=torch.float64)
    xt = a * rt + s * eps
    with torch.no_grad():
        v = net(xt.float(), {'sensor': Y, 'x0_hat': XR}, torch.full((8,), t, dtype=torch.long)).double()
    outs = dict(net=a * xt - s * v, w=wiener(xt, a, s, False), wp=wiener(xt, a, s, True), iid=a * xt)
    line = []
    for nm, m in (('all', np.ones((64, 64), bool)), ('grid', grid), ('inner', inner), ('edge', edge)):
        line.append(f'{nm} ' + '/'.join(f'{rms(o - rt, m):.3f}' for o in outs.values()))
    print(f'{t:4d} | ' + ' | '.join(line), flush=True)
