"""Probe the residual diffusion's denoiser on the 8 stored cubes at fixed t (training scale, r = (x - x_r) / sigma_r):
x0 error of the network vs (a) the zero predictor, (b) the exact denoiser of an iid N(0,1) residual (sqrt(ab) x_t,
i.e. v = 0), (c) a per-element Gaussian shrinkage that uses the variance head's sigma (no spatial structure at all).
Also the min-SNR-weighted L1 v-loss of each, per region (measured grid pixels / interior / extrapolated edge)."""
import math, os, sys, time
import numpy as np
import torch

torch.set_num_threads(int(os.environ.get('NT', '6')))
REPO = '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan'
sys.path.insert(0, REPO)
from data_loader.warmstart_paths import DATA_ROOT, PRIOR
from data_loader.hfd_residual import residual_collate_fn
from model.residual_wrapper import build_residual_model
from model.diffusion_trainer import DiffusionTrainer
from scripts.eval_warmstart import dataset_for

OUT = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d8/steelman/'
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
z = np.load(R + '/uq/d8/' + 'ev' + 'al/d8_main__cubes.npz')
X, XR, files = torch.from_numpy(z['x']).float(), torch.from_numpy(z['x_r']).float(), list(z['files'])
ds = dataset_for(DATA_ROOT, PRIOR, 8, [os.path.join(DATA_ROOT, f) for f in files])
order = [os.path.relpath(f, DATA_ROOT) for f in ds.img_list]
_, Y, _ = residual_collate_fn([ds[order.index(f)] for f in files])
sigma_r = 0.08934239400551727
SVH = torch.from_numpy(np.load(OUT + 'varhead_sigma_stored.npy')).float()     # [-1,1] scale
s2 = (SVH / sigma_r) ** 2                                                           # per-element variance of r (varhead)
r = (X - XR) / sigma_r
tr = DiffusionTrainer(device=torch.device('cpu'), loss_type='l1', prediction_type='v', snr_gamma=None, n_timesteps=1000)
ab_all = tr.alpha_bars
grid = torch.zeros(64, 64, dtype=torch.bool); grid[::8, ::8] = True
edge = torch.zeros(64, 64, dtype=torch.bool); edge[57:, :] = True; edge[:, 57:] = True
inner = ~grid & ~edge
regions = dict(all=torch.ones(64, 64, dtype=torch.bool), grid=grid, inner=inner, edge=edge)


def lag1(a):
    a = a - a.mean(dim=(-2, -1), keepdim=True)
    return float((((a[..., 1:, :] * a[..., :-1, :]).mean() + (a[..., :, 1:] * a[..., :, :-1]).mean()) / 2 / (a ** 2).mean()))


def wfac(t):
    snr = float(ab_all[t] / (1 - ab_all[t]))
    return min(snr, 5.0) / (snr + 1.0)


ckpts = sys.argv[1:] or ['100']
for ep in ckpts:
    ck = torch.load(f'{R}/runs/residual_reg_d8_b64/checkpoint_epoch_{ep}.pth', map_location='cpu', weights_only=False)
    net = build_residual_model('residual_reg', int(ck['meta']['base_channels']))
    net.load_state_dict(ck['model_state_dict'], strict=True); net.eval()
    print(f'\n=== epoch {ep}: RMS x0 error in r units (truth r has RMS {float(r.pow(2).mean().sqrt()):.3f}; grid {float(r[..., grid].pow(2).mean().sqrt()):.3f}, '
          f'inner {float(r[..., inner].pow(2).mean().sqrt()):.3f}, edge {float(r[..., edge].pow(2).mean().sqrt()):.3f})')
    print(f"{'t':>4s} {'snr':>8s} {'w':>6s} | {'pred':>6s} " + ' '.join(f'{k:>22s}' for k in ('all net/iid/vh/zero', 'grid net/iid/vh', 'inner net/iid/vh', 'edge net/iid/vh'))
          + ' | wL1v net / v=0 / vh | lag1 err net / x0 net / r')
    g = torch.Generator().manual_seed(123)
    tot = {'net': 0.0, 'v0': 0.0, 'vh': 0.0}
    for t in (49, 99, 149, 199, 299, 399, 499, 599, 699, 799, 899, 999):
        ab = ab_all[t]
        eps = torch.randn(r.shape, generator=g)
        xt = ab.sqrt() * r + (1 - ab).sqrt() * eps
        with torch.no_grad():
            v = net(xt, {'sensor': Y, 'x0_hat': XR}, torch.full((r.shape[0],), t, dtype=torch.long))
        x0_net = ab.sqrt() * xt - (1 - ab).sqrt() * v
        x0_iid = ab.sqrt() * xt
        x0_vh = ab.sqrt() * s2 * xt / (ab * s2 + 1 - ab)
        v_true = ab.sqrt() * eps - (1 - ab).sqrt() * r
        v_vh = (ab.sqrt() * xt - x0_vh) / (1 - ab).sqrt()
        cells = []
        for nm, m in regions.items():
            e = lambda x0: float((x0 - r)[..., m].pow(2).mean().sqrt())
            cells.append('/'.join(f'{e(x):.3f}' for x in ((x0_net, x0_iid, x0_vh, torch.zeros_like(r)) if nm == 'all' else (x0_net, x0_iid, x0_vh))))
        w = wfac(t)
        l_net, l_v0, l_vh = (w * float((vv - v_true).abs().mean()) for vv in (v, torch.zeros_like(v), v_vh))
        print(f'{t:4d} {float(ab / (1 - ab)):8.4f} {w:6.3f} |        ' + ' '.join(f'{c:>22s}' for c in cells)
              + f' | {l_net:.4f} {l_v0:.4f} {l_vh:.4f} | {lag1(x0_net - r):.3f} {lag1(x0_net):.3f} {lag1(r):.3f}')
    del net
