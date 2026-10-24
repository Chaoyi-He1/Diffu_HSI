"""Does the diffusion's heteroscedasticity grow with training? K = 4 DDIM-10 draws (seeds 0-3, true condition) of the same
8 cubes from residual_reg_d8_b64 at epochs 25 / 50 / 90 (epoch 100 is in samples_true.npz), plus the one-call denoiser
MSE at t = 300 and 500 under the true and the swapped condition."""
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
t0 = time.time()
b = np.load(f'{OUT}/base.npz')
y, xr, e = (torch.from_numpy(b[k]).float() for k in ('y', 'xr', 'e'))
sigma_r = float(b['sigma_r'])
trainer = DiffusionTrainer(device=torch.device('cpu'), loss_type='l1', snr_gamma=None, prediction_type='v', n_timesteps=1000)
conds = {'true': {'sensor': y, 'x0_hat': xr}, 'swap': {'sensor': y.roll(4, 0), 'x0_hat': xr.roll(4, 0)}}
gen = torch.Generator().manual_seed(1234)
noise = torch.randn(e.shape, generator=gen)              # the same draw as main.py's denoiser test
eps = [int(a) for a in os.environ.get('EPOCHS', '25,50,90').split(',')]
for ep in eps:
    ck = torch.load(f'{R}/runs/residual_reg_d8_b64/checkpoint_epoch_{ep}.pth', map_location='cpu', weights_only=False)
    m = build_residual_model(ck['meta']['mode'], int(ck['meta']['base_channels']))
    m.load_state_dict(ck['model_state_dict'], strict=True)
    m.eval()
    den = {}
    with torch.no_grad():
        for t_ in (500, 300):
            ab = trainer.alpha_bars[t_]
            xt = ab.sqrt() * e + (1 - ab).sqrt() * noise
            tt = torch.full((e.shape[0],), t_, dtype=torch.long)
            for c, cond in conds.items():
                v = m(xt, cond, tt)
                den[f'{t_}_{c}'] = (ab.sqrt() * xt - (1 - ab).sqrt() * v).half().numpy()
        out = []
        for k in range(int(os.environ.get('K', '4'))):
            torch.manual_seed(k)
            out.append(trainer.sample(m, conds['true'], tuple(xr.shape), n_steps=10, method='ddim', progress=False,
                                      clamp_x0=(-2.0 / sigma_r, 2.0 / sigma_r)))
            print(f'[{time.time() - t0:.0f}s] epoch {ep} seed {k}', flush=True)
    np.savez(f'{OUT}/trend_ep{ep}.npz', r=torch.stack(out, 0).half().numpy(), **den)
print('done', flush=True)
