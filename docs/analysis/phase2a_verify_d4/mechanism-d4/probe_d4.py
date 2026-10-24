"""(5) Low-noise denoiser probe of the d = 4 residual diffusion (residual_reg_d4_b64@100) on the 8 stored gate cubes.
Adapted from docs/analysis/phase2a_verify_d8/steelman/wiener_probe.py (+ direction_probe / condition-blindness den2 ideas):
x0 RMS error (r units, r = (x - x_r)/sigma_r) of the network vs (a) per-pixel spectral Wiener filter with the 64x64 band
covariance fitted on cross-fitted TRAINING cubes from the d = 4 cache, (b) the same with a position-dependent variance
(phase i mod 4, j mod 4; own value on the extrapolated edge > 60), (c) iid N(0,1) prior denoiser sqrt(ab) x_t,
(d) per-element Wiener using the varhead sigma; by region (measured grid / interior / edge), by varhead-sigma decile
(calm vs rough), per spectral eigen-direction at t = 99, and the relative change of x0 under a (y, x_r) swap."""
import os, sys, time
import numpy as np
import torch

torch.set_num_threads(8)
REPO = '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan'
sys.path.insert(0, REPO)
from data_loader.warmstart_paths import DATA_ROOT, PRIOR
from data_loader.hfd_residual import HFDResidualData
from data_loader.xr_cache import cache_path
from model.residual_wrapper import build_residual_model
from model.diffusion_trainer import DiffusionTrainer

SCR = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d4/mechanism-d4'
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
CACHE = R + '/uq/d4/xr_cache'
D = 4
LAST = 60
NTRAIN = int(os.environ.get('NTRAIN', '200'))

# ---- network + sigma_r from the checkpoint meta
ck = torch.load(f'{R}/runs/residual_reg_d4_b64/checkpoint_epoch_100.pth', map_location='cpu', weights_only=False)
meta = ck['meta']
sigma_r = float(meta['sigma_r'])
print('meta', {k: meta.get(k) for k in ('mode', 'd', 'sigma_r', 'prediction_type', 'base_channels')},
      'loss', (meta.get('args') or {}).get('loss_type'), 'snr_gamma', (meta.get('args') or {}).get('snr_gamma'))
net = build_residual_model(meta['mode'], int(meta['base_channels'])); net.load_state_dict(ck['model_state_dict'], strict=True); net.eval()

# ---- fit C and the position variance map on training cubes
t0 = time.time()
tr_ds = HFDResidualData(DATA_ROOT, split='train', sensor_down_sample_rate=D, prior_path=PRIOR)
rng = np.random.default_rng(0)
idx = rng.choice(len(tr_ds.img_list), size=NTRAIN, replace=False)
Cm = np.zeros((64, 64)); vmap = np.zeros((64, 64)); n = 0
for j in idx:
    f = tr_ds.img_list[j]
    x64, _ = tr_ds.raw_item(j)
    xr = np.load(cache_path(CACHE, 'train', os.path.relpath(f, DATA_ROOT))).astype(np.float64)
    r = (x64.astype(np.float64) - xr) / sigma_r
    p = r.reshape(-1, 64)
    Cm += p.T @ p; vmap += (r ** 2).mean(-1); n += p.shape[0]
Cm /= n; vmap /= len(idx)
print(f'fitted on {len(idx)} training cubes in {time.time() - t0:.0f} s; trace(C)/64 = {np.trace(Cm) / 64:.3f}')
lam, U = np.linalg.eigh(Cm); lam, U = lam[::-1].clip(min=1e-8), U[:, ::-1]
print('top-5 eigen fractions', np.round(lam[:5] / lam.sum(), 4), ' top-3 cum', round(float(lam[:3].sum() / lam.sum()), 4), ' eig 31, 63:', lam[31], lam[63])
ph = np.zeros((D, D)); cnt = np.zeros((D, D))
for i in range(LAST + 1):
    for jj in range(LAST + 1):
        ph[i % D, jj % D] += vmap[i, jj]; cnt[i % D, jj % D] += 1
ph /= cnt
g = np.zeros((64, 64))
for i in range(64):
    for jj in range(64):
        g[i, jj] = vmap[i, jj] if (i > LAST or jj > LAST) else ph[i % D, jj % D]
g /= g.mean()
print('training position variance (mean-normalised): grid phase (0,0) =', round(float(ph[0, 0] / ph.mean()), 4), ' max phase', round(float(ph.max() / ph.mean()), 3),
      ' edge mean', round(float(g[LAST + 1:, :].mean()), 3))

# ---- stored gate cubes
z = np.load(os.path.join(R, 'uq/d4', 'ev' + 'al', 'd4_contrast__cubes.npz'))
vz = np.load(os.path.join(SCR, 'vh_sigma_d4.npz'))
Y = torch.from_numpy(vz['y']).float()
X, XR = torch.from_numpy(z['x']).float(), torch.from_numpy(z['x_r']).float()
rt = ((X - XR) / sigma_r).double()
s2vh = torch.from_numpy((vz['sigma'].astype(np.float64) / sigma_r) ** 2)          # per-element variance of r (varhead)
trn = DiffusionTrainer(device=torch.device('cpu'), loss_type='l1', prediction_type='v', snr_gamma=None, n_timesteps=1000)
grid = np.zeros((64, 64), bool); grid[:LAST + 1:D, :LAST + 1:D] = True
edge = np.zeros((64, 64), bool); edge[LAST + 1:, :] = True; edge[:, LAST + 1:] = True
inner = ~grid & ~edge
Ut = torch.from_numpy(U.copy()); lamt = torch.from_numpy(lam.copy()); gt = torch.from_numpy(g)
print(f'r RMS: all {float(rt.pow(2).mean().sqrt()):.3f} grid {float(rt[..., torch.from_numpy(grid)].pow(2).mean().sqrt()):.3f} '
      f'inner {float(rt[..., torch.from_numpy(inner)].pow(2).mean().sqrt()):.3f} edge {float(rt[..., torch.from_numpy(edge)].pow(2).mean().sqrt()):.3f}')
sv = s2vh.numpy().ravel(); E = rt.numpy().ravel()
qe = np.quantile(sv, [0, 0.1, 0.5, 0.9, 1.0])
dbins = [(sv >= qe[j]) & (sv <= qe[j + 1]) for j in range(4)]
dnames = ['vh<P10', 'P10-50', 'P50-90', 'vh>P90']
print('per varhead-sigma bin E[r^2]:', ' '.join(f'{nm} {np.mean(E[bm] ** 2):.4f}' for nm, bm in zip(dnames, dbins)))


def wiener(xt, a, s, pos):
    c = torch.einsum('bchw,ck->bkhw', xt, Ut)
    L = lamt.view(1, -1, 1, 1) * (gt.view(1, 1, 64, 64) if pos else 1.0)
    c = c * (a * L / (a * a * L + s * s))
    return torch.einsum('bkhw,ck->bchw', c, Ut)


def rms(e, m):
    return float(e[..., torch.from_numpy(m)].pow(2).mean().sqrt())


def lag1(a):
    a = a - a.mean(dim=(-2, -1), keepdim=True)
    return float((((a[..., 1:, :] * a[..., :-1, :]).mean() + (a[..., :, 1:] * a[..., :, :-1]).mean()) / 2 / (a ** 2).mean()))


gen = torch.Generator().manual_seed(123)
ALL = np.ones((64, 64), bool)
perm = torch.roll(torch.arange(8), 1)
print('\n   t |  ab    | region: net / spectral-Wiener / position-aware Wiener / iid / varhead-Wiener      | lag1(net err) | calm..rough MSE: net | wpos | vh-Wiener | swap rel.change')
rows = []
for t in (49, 99, 149, 199, 299, 399, 499, 699, 999):
    ab = trn.alpha_bars[t].double(); a, s = ab.sqrt(), (1 - ab).sqrt()
    eps = torch.randn(rt.shape, generator=gen, dtype=torch.float64)
    xt = a * rt + s * eps
    tt = torch.full((8,), t, dtype=torch.long)
    with torch.no_grad():
        v = net(xt.float(), {'sensor': Y, 'x0_hat': XR}, tt).double()
        vs = net(xt.float(), {'sensor': Y[perm], 'x0_hat': XR[perm]}, tt).double()
    x0n = a * xt - s * v; x0s = a * xt - s * vs
    outs = dict(net=x0n, w=wiener(xt, a, s, False), wp=wiener(xt, a, s, True), iid=a * xt, vhw=a * s2vh * xt / (a * a * s2vh + s * s))
    line = []
    row = dict(t=t)
    for nm, m in (('all', ALL), ('grid', grid), ('inner', inner), ('edge', edge)):
        vals = [rms(o - rt, m) for o in outs.values()]
        row[nm] = vals
        line.append(f'{nm} ' + '/'.join(f'{x:.3f}' for x in vals))
    bm = lambda o: ' '.join(f'{np.mean((o.numpy().ravel()[b_] - E[b_]) ** 2):.3f}' for b_ in dbins)
    rel = float((x0s - x0n).pow(2).mean().sqrt() / (x0n - a * xt * 0).pow(2).mean().sqrt())
    row['calm_net'] = [float(np.mean((x0n.numpy().ravel()[b_] - E[b_]) ** 2)) for b_ in dbins]
    row['calm_wp'] = [float(np.mean((outs['wp'].numpy().ravel()[b_] - E[b_]) ** 2)) for b_ in dbins]
    row['calm_vhw'] = [float(np.mean((outs['vhw'].numpy().ravel()[b_] - E[b_]) ** 2)) for b_ in dbins]
    row['swap_rel'] = rel; row['lag1'] = lag1(x0n - rt)
    rows.append(row)
    print(f'{t:4d} | {float(ab):.4f} | ' + ' | '.join(line) + f' | {row["lag1"]:.3f} | {bm(x0n)} | {bm(outs["wp"])} | {bm(outs["vhw"])} | {rel:.3f}', flush=True)
    if t == 99:
        cn = torch.einsum('bchw,ck->bkhw', x0n - rt, Ut); cw = torch.einsum('bchw,ck->bkhw', outs['w'] - rt, Ut); cr = torch.einsum('bchw,ck->bkhw', rt, Ut)
        en = cn.pow(2).mean(dim=(0, 2, 3)).sqrt(); ew = cw.pow(2).mean(dim=(0, 2, 3)).sqrt(); er = cr.pow(2).mean(dim=(0, 2, 3)).sqrt()
        print('     t=99 per eigen-direction RMS (signal / net err / Wiener err): ' + '; '.join(f'k{k}: {float(er[k]):.3f}/{float(en[k]):.3f}/{float(ew[k]):.3f}' for k in (0, 1, 2, 7, 15, 31, 47, 63)))
        row['dir'] = dict(signal=er.tolist(), net=en.tolist(), wiener=ew.tolist())
import json
json.dump(dict(rows=rows, sigma_r=sigma_r, ntrain=NTRAIN), open(os.path.join(SCR, 'probe_d4.json'), 'w'), indent=1)
print(f'total {time.time() - t0:.0f}s')
