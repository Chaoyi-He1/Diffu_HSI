"""Dense-t one-call denoiser test of residual_reg_d8_b64 (epoch from env EP, default 100), true condition, the same fixed
noise as main.py; x0 MSE vs the true residual e overall and per decile of the variance head's sigma, against two
x_t-only Wiener references: homoscedastic (global E[e^2]) and heteroscedastic (per-element varhead sigma^2)."""
import os
import sys
import time

REPO = '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan'
sys.path.insert(0, REPO)
import numpy as np
import torch

torch.set_num_threads(int(os.environ.get('NT', '6')))
from model.diffusion_trainer import DiffusionTrainer
from model.residual_wrapper import build_residual_model

OUT = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d8/condition-blindness'
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
EP = int(os.environ.get('EP', '100'))
t0 = time.time()
b = np.load(f'{OUT}/base.npz')
y, xr, e, svh = (torch.from_numpy(b[k]).float() for k in ('y', 'xr', 'e', 'sig_vh'))
trainer = DiffusionTrainer(device=torch.device('cpu'), loss_type='l1', snr_gamma=None, prediction_type='v', n_timesteps=1000)
gen = torch.Generator().manual_seed(1234)
noise = torch.randn(e.shape, generator=gen)
ck = torch.load(f'{R}/runs/residual_reg_d8_b64/checkpoint_epoch_{EP}.pth', map_location='cpu', weights_only=False)
m = build_residual_model(ck['meta']['mode'], int(ck['meta']['base_channels']))
m.load_state_dict(ck['model_state_dict'], strict=True)
m.eval()
cond = {'sensor': y, 'x0_hat': xr}
sv = svh.numpy().ravel()
edges = np.quantile(sv, [0, 0.1, 0.5, 0.9, 1.0])
bins = [(sv >= edges[j]) & (sv <= edges[j + 1]) for j in range(4)]
names = ['vh<P10', 'P10-50', 'P50-90', 'vh>P90']
E = e.numpy().ravel()
s2g = float((e ** 2).mean())
print(f'epoch {EP}; E[e^2] {s2g:.4f}; per-bin E[e^2]', ' '.join(f'{n} {np.mean(E[bm] ** 2):.4f}' for n, bm in zip(names, bins)))
print('t     ab      | model MSE all / per bin                         | Wiener homo all / bins            | Wiener varhead all / bins')
with torch.no_grad():
    for t_ in (999, 900, 800, 700, 600, 500, 400, 350, 300, 250, 200, 150, 99, 50, 20):
        ab = trainer.alpha_bars[t_]
        xt = ab.sqrt() * e + (1 - ab).sqrt() * noise
        tt = torch.full((e.shape[0],), t_, dtype=torch.long)
        v = m(xt, cond, tt)
        x0 = (ab.sqrt() * xt - (1 - ab).sqrt() * v).numpy().ravel()
        a = float(ab)
        X = xt.numpy().ravel()
        w_h = np.sqrt(a) * s2g / (a * s2g + 1 - a) * X
        s2v = sv ** 2
        w_v = np.sqrt(a) * s2v / (a * s2v + 1 - a) * X
        f = lambda z: f'{np.mean((z - E) ** 2):.4f} / ' + ' '.join(f'{np.mean((z[bm] - E[bm]) ** 2):.4f}' for bm in bins)
        print(f'{t_:4d} {a:.5f} | {f(x0)} | {f(w_h)} | {f(w_v)}', flush=True)
print(f'{time.time() - t0:.0f}s')
