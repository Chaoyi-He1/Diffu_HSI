"""Is the network's x0 prediction shrunk (L1 -> conditional median)? For an MMSE denoiser E[r | r_hat] = r_hat (slope 1).
Regress the true r on the network's x0 prediction at fixed t on the 8 stored cubes; binned conditional means."""
import os, sys
import numpy as np
import torch

torch.set_num_threads(int(os.environ.get('NT', '5')))
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
ds = dataset_for(DATA_ROOT, PRIOR, 8, [os.path.join(DATA_ROOT, f) for f in files])
order = [os.path.relpath(f, DATA_ROOT) for f in ds.img_list]
_, Y, _ = residual_collate_fn([ds[order.index(f)] for f in files])
X, XR = torch.from_numpy(z['x']).float(), torch.from_numpy(z['x_r']).float()
sigma_r = 0.08934239400551727
r = (X - XR) / sigma_r
ck = torch.load(f'{R}/runs/residual_reg_d8_b64/checkpoint_epoch_100.pth', map_location='cpu', weights_only=False)
net = build_residual_model('residual_reg', 64); net.load_state_dict(ck['model_state_dict'], strict=True); net.eval()
trn = DiffusionTrainer(device=torch.device('cpu'), loss_type='l1', prediction_type='v', snr_gamma=None, n_timesteps=1000)
gen = torch.Generator().manual_seed(321)
print('t | slope of r on x0_net (OLS through 0) | corr | var(x0_net)/var(r) | E[r | x0_net bin] for bins of x0_net (quantiles 1,5,25,50,75,95,99 %)')
for t in (199, 299, 399, 499, 599, 699, 799):
    ab = trn.alpha_bars[t]
    eps = torch.randn(r.shape, generator=gen)
    xt = ab.sqrt() * r + (1 - ab).sqrt() * eps
    with torch.no_grad():
        v = net(xt, {'sensor': Y, 'x0_hat': XR}, torch.full((8,), t, dtype=torch.long))
    p = (ab.sqrt() * xt - (1 - ab).sqrt() * v).double().numpy().ravel()
    y = r.double().numpy().ravel()
    slope = float((p * y).sum() / (p * p).sum())
    qs = np.quantile(p, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    edges = np.quantile(p, np.linspace(0, 1, 21))
    b = np.clip(np.searchsorted(edges, p) - 1, 0, 19)
    pm = np.array([p[b == k].mean() for k in range(20)]); ym = np.array([y[b == k].mean() for k in range(20)])
    print(f'{t:4d} | slope {slope:.3f} | corr {np.corrcoef(p, y)[0, 1]:.3f} | var ratio {p.var() / y.var():.3f} | '
          f'outer bins (x0_net -> E r): ' + ' '.join(f'{a:+.2f}->{c:+.2f}' for a, c in zip(pm[[0, 1, 2, 17, 18, 19]], ym[[0, 1, 2, 17, 18, 19]])))
