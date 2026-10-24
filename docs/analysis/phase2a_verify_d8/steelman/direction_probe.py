"""x0 error of the network at t = 99 / 199 per spectral eigen-direction of the training residual covariance."""
import os, sys
import numpy as np
import torch

torch.set_num_threads(int(os.environ.get('NT', '4')))
REPO = '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan'
sys.path.insert(0, REPO)
from data_loader.warmstart_paths import DATA_ROOT, PRIOR
from data_loader.hfd_residual import HFDResidualData, residual_collate_fn
from data_loader.xr_cache import cache_path
from model.residual_wrapper import build_residual_model
from model.diffusion_trainer import DiffusionTrainer
from scripts.eval_warmstart import dataset_for

R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
CACHE = R + '/uq/d8/xr_cache'
sigma_r = 0.08934239400551727
tr_ds = HFDResidualData(DATA_ROOT, split='train', sensor_down_sample_rate=8, prior_path=PRIOR)
rng = np.random.default_rng(0)
C = np.zeros((64, 64)); n = 0
for j in rng.choice(len(tr_ds.img_list), size=200, replace=False):
    f = tr_ds.img_list[j]
    x64, _ = tr_ds.raw_item(j)
    xr = np.load(cache_path(CACHE, 'train', os.path.relpath(f, DATA_ROOT))).astype(np.float64)
    p = ((x64.astype(np.float64) - xr) / sigma_r).reshape(-1, 64); C += p.T @ p; n += p.shape[0]
C /= n
lam, U = np.linalg.eigh(C); lam, U = lam[::-1].copy(), U[:, ::-1].copy()
z = np.load(R + '/uq/d8/' + 'ev' + 'al/d8_main__cubes.npz')
files = list(z['files'])
ds = dataset_for(DATA_ROOT, PRIOR, 8, [os.path.join(DATA_ROOT, f) for f in files])
order = [os.path.relpath(f, DATA_ROOT) for f in ds.img_list]
_, Y, _ = residual_collate_fn([ds[order.index(f)] for f in files])
X, XR = torch.from_numpy(z['x']).float(), torch.from_numpy(z['x_r']).float()
r = (X - XR) / sigma_r
ck = torch.load(f'{R}/runs/residual_reg_d8_b64/checkpoint_epoch_100.pth', map_location='cpu', weights_only=False)
net = build_residual_model('residual_reg', 64); net.load_state_dict(ck['model_state_dict'], strict=True); net.eval()
trn = DiffusionTrainer(device=torch.device('cpu'), loss_type='l1', prediction_type='v', snr_gamma=None, n_timesteps=1000)
Ut = torch.from_numpy(U).float()
gen = torch.Generator().manual_seed(7)
for t in (99, 199):
    ab = trn.alpha_bars[t]; a, s = ab.sqrt(), (1 - ab).sqrt()
    eps = torch.randn(r.shape, generator=gen)
    xt = a * r + s * eps
    with torch.no_grad():
        v = net(xt, {'sensor': Y, 'x0_hat': XR}, torch.full((8,), t, dtype=torch.long))
    err = torch.einsum('bchw,ck->bkhw', a * xt - s * v - r, Ut)             # error coefficients per eigen-direction
    sig = torch.einsum('bchw,ck->bkhw', r, Ut)
    e_k = err.pow(2).mean(dim=(0, 2, 3)).sqrt().numpy(); s_k = sig.pow(2).mean(dim=(0, 2, 3)).sqrt().numpy()
    noise = float(s / a)
    print(f't={t}: noise level in x0 units {noise:.3f}')
    for k in (0, 1, 2, 3, 4, 7, 15, 31, 63):
        wiener = np.sqrt(lam[k] * noise ** 2 / (lam[k] + noise ** 2))
        print(f'   direction {k:2d}: train var {lam[k]:8.4f}  gate signal rms {s_k[k]:.4f}  net error rms {e_k[k]:.4f}  (Gaussian-Wiener error {wiener:.4f})')
    print(f'   directions 8-63 pooled: signal rms {np.sqrt((s_k[8:] ** 2).mean()):.4f}  net error rms {np.sqrt((e_k[8:] ** 2).mean()):.4f}')
