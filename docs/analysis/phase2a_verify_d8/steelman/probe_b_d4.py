"""Same denoiser probe for B (residual_d4_b64@100, linear-centred, d = 4) on the 8 stored files at d = 4: is the
low-t white-error / no-grid-awareness pattern generic to the recipe?"""
import os, sys
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

R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
z = np.load(R + '/uq/d8/' + 'ev' + 'al/d8_main__cubes.npz')
files = list(z['files'])
ds = dataset_for(DATA_ROOT, PRIOR, 4, [os.path.join(DATA_ROOT, f) for f in files])
order = [os.path.relpath(f, DATA_ROOT) for f in ds.img_list]
X, Y, XH = residual_collate_fn([ds[order.index(f)] for f in files])
sd4 = float(ds.sigma_d)
r = (X - XH) / sd4
ck = torch.load(f'{R}/runs/residual_d4_b64/checkpoint_epoch_100.pth', map_location='cpu', weights_only=False)
print('B meta', {k: ck['meta'].get(k) for k in ('mode', 'd', 'prediction_type', 'sigma_d')}, 'loss', (ck['meta'].get('args') or {}).get('loss_type'), 'snr', (ck['meta'].get('args') or {}).get('snr_gamma'))
net = build_residual_model('residual', int(ck['meta']['base_channels']))
net.load_state_dict(ck['model_state_dict'], strict=True); net.eval()
tr = DiffusionTrainer(device=torch.device('cpu'), loss_type='l1', prediction_type='v', snr_gamma=None, n_timesteps=1000)
grid = torch.zeros(64, 64, dtype=torch.bool); grid[::4, ::4] = True
edge = torch.zeros(64, 64, dtype=torch.bool); edge[61:, :] = True; edge[:, 61:] = True
inner = ~grid & ~edge


def lag1(a):
    a = a - a.mean(dim=(-2, -1), keepdim=True)
    return float((((a[..., 1:, :] * a[..., :-1, :]).mean() + (a[..., :, 1:] * a[..., :, :-1]).mean()) / 2 / (a ** 2).mean()))


print(f'sigma_d4 {sd4:.6f}; r RMS all {float(r.pow(2).mean().sqrt()):.3f} grid {float(r[..., grid].pow(2).mean().sqrt()):.3f} '
      f'inner {float(r[..., inner].pow(2).mean().sqrt()):.3f} edge {float(r[..., edge].pow(2).mean().sqrt()):.3f}; lag1(r) {lag1(r):.3f}')
g = torch.Generator().manual_seed(123)
print('   t | all net/iid | grid net/iid | inner net/iid | edge net/iid | lag1 err net')
for t in (49, 99, 149, 199, 299, 499, 699, 999):
    ab = tr.alpha_bars[t]
    eps = torch.randn(r.shape, generator=g)
    xt = ab.sqrt() * r + (1 - ab).sqrt() * eps
    with torch.no_grad():
        v = net(xt, {'sensor': Y, 'x0_hat': XH}, torch.full((r.shape[0],), t, dtype=torch.long))
    x0n, x0i = ab.sqrt() * xt - (1 - ab).sqrt() * v, ab.sqrt() * xt
    cell = lambda m: f'{float((x0n - r)[..., m].pow(2).mean().sqrt()):.3f}/{float((x0i - r)[..., m].pow(2).mean().sqrt()):.3f}'
    print(f'{t:4d} | {cell(torch.ones(64, 64, dtype=torch.bool))} | {cell(grid)} | {cell(inner)} | {cell(edge)} | {lag1(x0n - r):.3f}')
