"""Hybrid DDIM-10 on the 8 stored gate cubes: the trained network for t >= T_SWITCH (where it beats linear denoisers),
then three branches from the same state: (net) the network as evaluated; (w) a per-pixel spectral Wiener x0-denoiser;
(wpos) the position-aware spectral Wiener. Both Wiener denoisers are fitted on 200 cross-fitted TRAINING cubes only.
Same DDIM update, grid (999, 899, ..., 99), clamp and final-x0 rule as DiffusionTrainer.sample. CPU only."""
import json, math, os, sys, time
import numpy as np
import torch

torch.set_num_threads(int(os.environ.get('NT', '6')))
REPO = '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan'
sys.path.insert(0, REPO)
from data_loader.warmstart_paths import DATA_ROOT, PRIOR
from data_loader.hfd_residual import HFDResidualData, residual_collate_fn
from data_loader.xr_cache import cache_path
from model.residual_wrapper import build_residual_model
from model.diffusion_trainer import DiffusionTrainer
from scripts.eval_warmstart import dataset_for
from scripts.uq import metrics as M
from scipy.stats import spearmanr

OUT = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d8/steelman/'
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
CACHE = R + '/uq/d8/xr_cache'
sigma_r = 0.08934239400551727
T_SWITCH = int(os.environ.get('TSW', '399'))
K, BS = int(os.environ.get('K', '32')), int(os.environ.get('BS', '16'))
cubes = [int(a) for a in sys.argv[1:]] or list(range(8))

# ---- linear denoisers from training cubes (identical to wiener_probe.py)
tr_ds = HFDResidualData(DATA_ROOT, split='train', sensor_down_sample_rate=8, prior_path=PRIOR)
rng = np.random.default_rng(0)
idx = rng.choice(len(tr_ds.img_list), size=200, replace=False)
C = np.zeros((64, 64)); vmap = np.zeros((64, 64)); n = 0
for j in idx:
    f = tr_ds.img_list[j]
    x64, _ = tr_ds.raw_item(j)
    xr = np.load(cache_path(CACHE, 'train', os.path.relpath(f, DATA_ROOT))).astype(np.float64)
    r = (x64.astype(np.float64) - xr) / sigma_r
    p = r.reshape(-1, 64); C += p.T @ p; vmap += (r ** 2).mean(-1); n += p.shape[0]
C /= n; vmap /= len(idx)
lam, U = np.linalg.eigh(C); lam, U = lam[::-1].clip(min=1e-8).copy(), U[:, ::-1].copy()
ph = np.zeros((8, 8)); cnt = np.zeros((8, 8))
for i in range(57):
    for jj in range(57):
        ph[i % 8, jj % 8] += vmap[i, jj]; cnt[i % 8, jj % 8] += 1
ph /= cnt
g = np.array([[vmap[i, jj] if (i >= 57 or jj >= 57) else ph[i % 8, jj % 8] for jj in range(64)] for i in range(64)])
g /= g.mean()
Ut, lamt, gt = torch.from_numpy(U).float(), torch.from_numpy(lam).float(), torch.from_numpy(g).float()


def wiener(xt, a, s, pos):
    c = torch.einsum('bchw,ck->bkhw', xt, Ut)
    L = lamt.view(1, -1, 1, 1) * (gt.view(1, 1, 64, 64) if pos else 1.0)
    return torch.einsum('bkhw,ck->bchw', c * (a * L / (a * a * L + s * s)), Ut)


# ---- data, network, sampler constants
z = np.load(R + '/uq/d8/' + 'ev' + 'al/d8_main__cubes.npz')
files = list(z['files'])
ds = dataset_for(DATA_ROOT, PRIOR, 8, [os.path.join(DATA_ROOT, f) for f in files])
order = [os.path.relpath(f, DATA_ROOT) for f in ds.img_list]
_, Y, _ = residual_collate_fn([ds[order.index(f)] for f in files])
X, XR = torch.from_numpy(z['x']).float(), torch.from_numpy(z['x_r']).float()
SVH = np.load(OUT + 'varhead_sigma_stored.npy').astype(np.float64)
ck = torch.load(f'{R}/runs/residual_reg_d8_b64/checkpoint_epoch_100.pth', map_location='cpu', weights_only=False)
net = build_residual_model('residual_reg', 64); net.load_state_dict(ck['model_state_dict'], strict=True); net.eval()
trn = DiffusionTrainer(device=torch.device('cpu'), loss_type='l1', prediction_type='v', snr_gamma=None, n_timesteps=1000)
AB = trn.alpha_bars
ts = list(range(999, -1, -100))                 # the evaluated DDIM-10 grid: 999 ... 99
lo, hi = -2.0 / sigma_r, 2.0 / sigma_r
grid = np.zeros((64, 64), bool); grid[::8, ::8] = True
edge = np.zeros((64, 64), bool); edge[57:, :] = True; edge[:, 57:] = True
inner = ~grid & ~edge


def x0_of(kind, x, t, cond):
    a, s = AB[t].sqrt(), (1 - AB[t]).sqrt()
    if kind == 'net':
        with torch.no_grad():
            v = net(x, cond, torch.full((x.shape[0],), t, dtype=torch.long))
        x0 = a * x - s * v
    else:
        x0 = wiener(x, a, s, kind == 'wpos')
    return x0.clamp(lo, hi)


def ddim_step(x, x0, t, t_prev):
    if t_prev < 0:
        return x0
    abt, abp = AB[t], AB[t_prev]
    eps = (x - abt.sqrt() * x0) / (1 - abt).sqrt()
    return abp.sqrt() * x0 + (1 - abp).sqrt() * eps


def lag1(a):
    a = a - a.mean(axis=(-2, -1), keepdims=True)
    return float(((a[..., 1:, :] * a[..., :-1, :]).mean() + (a[..., :, 1:] * a[..., :, :-1]).mean()) / 2 / (a ** 2).mean())


def band_corr(a):
    a = a.reshape(a.shape[0], -1); a = a - a.mean(1, keepdims=True); nn_ = np.sqrt((a ** 2).sum(1))
    return float(((a[1:] * a[:-1]).sum(1) / (nn_[1:] * nn_[:-1] + 1e-30)).mean())


def metrics_for(s01, x01, xr01, sv01):
    mu, sd = s01.mean(0), s01.std(0, ddof=1)
    e = xr01 - x01; dev = s01 - mu[None]
    cv = M.coverage_samples(s01, x01)
    return dict(crps=100 * float(M.crps_samples(s01, x01).mean()), cov50=cv[0.5], cov90=cv[0.9], ause=M.ause(sd, mu - x01),
                spread_ratio=M.spread_ratio(sd ** 2, mu - x01), rms_sd=100 * math.sqrt((sd ** 2).mean()),
                sd_grid=100 * math.sqrt((sd[..., grid] ** 2).mean()), sd_inner=100 * math.sqrt((sd[..., inner] ** 2).mean()),
                sd_edge=100 * math.sqrt((sd[..., edge] ** 2).mean()), sp_sd_err=float(spearmanr(sd.ravel(), np.abs(e).ravel())[0]),
                rmse_meanK=100 * math.sqrt(((mu - x01) ** 2).mean()), rmse_single=100 * math.sqrt(((s01[0] - x01) ** 2).mean()),
                hf_single=M.hf_energy_ratio(s01[0], x01), lag1_dev=float(np.mean([lag1(dev[k]) for k in range(4)])),
                bandc_dev=float(np.mean([band_corr(dev[k]) for k in range(4)])), sam_single=None)


rows = []
for ci in cubes:
    t0 = time.time()
    torch.manual_seed(5000 + ci)
    branches = {'net': [], 'w': [], 'wpos': []}
    for k0 in range(0, K, BS):
        b = min(BS, K - k0)
        cond = {'sensor': Y[ci:ci + 1].expand(b, -1, -1, -1).contiguous(), 'x0_hat': XR[ci:ci + 1].expand(b, -1, -1, -1).contiguous()}
        x = torch.randn(b, 64, 64, 64)
        state = None
        for i, t in enumerate(ts):
            tp = ts[i + 1] if i + 1 < len(ts) else -1
            if t >= T_SWITCH:
                x = ddim_step(x, x0_of('net', x, t, cond), t, tp)
                continue
            if state is None:
                state = {kname: x.clone() for kname in branches}
            for kname in branches:
                state[kname] = ddim_step(state[kname], x0_of(kname, state[kname], t, cond), t, tp)
        for kname in branches:
            branches[kname].append((XR[ci:ci + 1] + sigma_r * state[kname]).clamp(-1, 1))
    x01, xr01, sv01 = (z['x'][ci].astype(np.float64) + 1) / 2, (z['x_r'][ci].astype(np.float64) + 1) / 2, SVH[ci] / 2
    crps_vh = 100 * float(M.crps_gaussian(xr01, sv01, x01).mean())
    for kname, lst in branches.items():
        s01 = (torch.cat(lst, 0).double().numpy() + 1) / 2
        m = metrics_for(s01, x01, xr01, sv01)
        m.update(cube=ci, branch=kname, crps_vh=crps_vh, rmse_R=100 * math.sqrt(((xr01 - x01) ** 2).mean()), tsw=T_SWITCH)
        rows.append(m)
        print(f"cube {ci} {kname:5s} " + ' '.join(f'{k}={v:.4f}' for k, v in m.items() if isinstance(v, float)), flush=True)
    print(f'  cube {ci} done in {time.time() - t0:.0f} s', flush=True)
    with open(OUT + f'hybrid_tsw{T_SWITCH}.jsonl', 'a') as f:
        for r_ in rows[-3:]:
            f.write(json.dumps(r_) + '\n')

print('\nmeans over cubes', cubes)
for kname in ('net', 'w', 'wpos'):
    rr = [r_ for r_ in rows if r_['branch'] == kname]
    print(f'{kname:5s} ' + ' '.join(f'{k}={np.mean([r_[k] for r_ in rr]):.4f}' for k in rr[0] if isinstance(rr[0][k], float)))
