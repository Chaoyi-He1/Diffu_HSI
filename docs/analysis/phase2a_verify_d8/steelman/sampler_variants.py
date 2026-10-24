"""Draw K samples for selected stored cubes with a given sampler (CPU) and report the structure / calibration metrics,
paired with the variance head on the same cube.  usage: sampler_variants.py <method> <n_steps> <K> <batch> <cube idx...>"""
import math, os, sys, time
import numpy as np
import torch

torch.set_num_threads(int(os.environ.get('NT', '6')))
REPO = '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan'
sys.path.insert(0, REPO)
from data_loader.warmstart_paths import DATA_ROOT, PRIOR
from data_loader.hfd_residual import residual_collate_fn
from model.residual_wrapper import build_residual_model, compose
from model.diffusion_trainer import DiffusionTrainer
from scripts.eval_warmstart import dataset_for
from scripts.uq import metrics as M
from scipy.stats import spearmanr

OUT = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d8/steelman/'
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
method, n_steps, K, bs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
cubes = [int(a) for a in sys.argv[5:]]
ep = os.environ.get('EP', '100')
z = np.load(R + '/uq/d8/' + 'ev' + 'al/d8_main__cubes.npz')
files = list(z['files'])
ds = dataset_for(DATA_ROOT, PRIOR, 8, [os.path.join(DATA_ROOT, f) for f in files])
order = [os.path.relpath(f, DATA_ROOT) for f in ds.img_list]
_, Y, _ = residual_collate_fn([ds[order.index(f)] for f in files])
X, XR = torch.from_numpy(z['x']).float(), torch.from_numpy(z['x_r']).float()
SVH = np.load(OUT + 'varhead_sigma_stored.npy').astype(np.float64)
sigma_r = 0.08934239400551727
ck = torch.load(f'{R}/runs/residual_reg_d8_b64/checkpoint_epoch_{ep}.pth', map_location='cpu', weights_only=False)
net = build_residual_model('residual_reg', int(ck['meta']['base_channels']))
net.load_state_dict(ck['model_state_dict'], strict=True); net.eval()
tr = DiffusionTrainer(device=torch.device('cpu'), loss_type='l1', prediction_type='v', snr_gamma=None, n_timesteps=1000)
grid = np.zeros((64, 64), bool); grid[::8, ::8] = True
edge = np.zeros((64, 64), bool); edge[57:, :] = True; edge[:, 57:] = True
inner = ~grid & ~edge


def lag1(a):
    a = a - a.mean(axis=(-2, -1), keepdims=True)
    return float(((a[..., 1:, :] * a[..., :-1, :]).mean() + (a[..., :, 1:] * a[..., :, :-1]).mean()) / 2 / (a ** 2).mean())


def band_corr(a):
    a = a.reshape(a.shape[0], -1); a = a - a.mean(1, keepdims=True); n = np.sqrt((a ** 2).sum(1))
    return float(((a[1:] * a[:-1]).sum(1) / (n[1:] * n[:-1] + 1e-30)).mean())


for i in cubes:
    t0 = time.time()
    torch.manual_seed(1000 + i)
    out = []
    for k0 in range(0, K, bs):
        b = min(bs, K - k0)
        cond = {'sensor': Y[i:i + 1].expand(b, -1, -1, -1).contiguous(), 'x0_hat': XR[i:i + 1].expand(b, -1, -1, -1).contiguous()}
        with torch.no_grad():
            r = tr.sample(net, cond, (b, 64, 64, 64), n_steps=n_steps, method=method, progress=False, clamp_x0=(-2.0 / sigma_r, 2.0 / sigma_r))
        out.append(compose(XR[i:i + 1], sigma_r, r))
        print(f'  cube {i}: {k0 + b}/{K} samples, {time.time() - t0:.0f} s', flush=True)
    s = torch.cat(out, 0).double().numpy()
    np.save(OUT + f'samples_{method}{n_steps}_ep{ep}_cube{i}_K{K}.npy', s.astype(np.float16))
    s01, x01, xr01, sv01 = (s + 1) / 2, (z['x'][i].astype(np.float64) + 1) / 2, (z['x_r'][i].astype(np.float64) + 1) / 2, SVH[i] / 2
    mu, sd = s01.mean(0), s01.std(0, ddof=1)
    e = xr01 - x01
    dev = s01 - mu[None]
    cov = M.coverage_samples(s01, x01) if K >= 19 else {0.5: float('nan'), 0.9: float('nan')}
    res = dict(cube=i, file=files[i], sampler=f'{method}{n_steps}', ep=ep, K=K, sec=time.time() - t0,
               crps=100 * float(M.crps_samples(s01, x01).mean()), crps_vh=100 * float(M.crps_gaussian(xr01, sv01, x01).mean()),
               cov50=cov[0.5], cov90=cov[0.9], rmse_R=100 * math.sqrt((e ** 2).mean()), rmse_meanK=100 * math.sqrt(((mu - x01) ** 2).mean()),
               rmse_single=100 * math.sqrt(((s01[0] - x01) ** 2).mean()),
               rms_sd=100 * math.sqrt((sd ** 2).mean()), sd_grid=100 * math.sqrt((sd[..., grid] ** 2).mean()),
               sd_inner=100 * math.sqrt((sd[..., inner] ** 2).mean()), sd_edge=100 * math.sqrt((sd[..., edge] ** 2).mean()),
               err_grid=100 * math.sqrt((e[..., grid] ** 2).mean()), err_inner=100 * math.sqrt((e[..., inner] ** 2).mean()), err_edge=100 * math.sqrt((e[..., edge] ** 2).mean()),
               sp_sd_err=float(spearmanr(sd.ravel(), np.abs(e).ravel())[0]),
               lag1_dev=float(np.mean([lag1(dev[k]) for k in range(min(K, 4))])), bandc_dev=float(np.mean([band_corr(dev[k]) for k in range(min(K, 4))])),
               hf_single=M.hf_energy_ratio(s01[0], x01), ause=M.ause(sd, mu - x01))
    print(' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in res.items()), flush=True)
    with open(OUT + 'sampler_variants.jsonl', 'a') as f:
        import json
        f.write(json.dumps(res) + '\n')
