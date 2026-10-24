"""H1 (condition blindness) test of residual_reg_d8_b64@100 on CPU.

8 gate cubes (the 8 stored y-swap cubes of d8_main), centre = regress_l1_sel_d8_b64@11 exactly as uq_eval.regress_centre,
variance head varhead_d8_b64@10, DDIM-10 v-prediction samples of the standardised residual with torch.manual_seed(k)
before each draw (one draw = the whole 8-cube batch, so seed k gives the same initial noise at batch index i under every
condition), clamp_x0 = (-2/sigma_r, 2/sigma_r). Conditions: true, swap (sensor and x_r of cube (i+4) mod 8), zero
(sensor and x_r zeroed), swap_y (sensor only), swap_xr (x_r only). Also a one-call denoiser sensitivity test on x_t
built from the true residual.
"""
import os
import sys
import time

REPO = '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan'
sys.path.insert(0, REPO)
import numpy as np
import torch

torch.set_num_threads(int(os.environ.get('NT', '10')))
from data_loader.hfd_residual import residual_collate_fn
from data_loader.warmstart_paths import DATA_ROOT, PRIOR
from model.diffusion_trainer import DiffusionTrainer
from model.residual_wrapper import LOGVAR_RANGE, build_residual_model, compose, regress_forward, varhead_forward
from scripts.eval_warmstart import dataset_for

OUT = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d8/condition-blindness'
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
K_MAIN = int(os.environ.get('K_MAIN', '8'))
K_PART = int(os.environ.get('K_PART', '4'))
t0 = time.time()


def log(*a):
    print(f'[{time.time() - t0:7.1f}s]', *a, flush=True)


def load(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    meta = ck['meta']
    m = build_residual_model(meta['mode'], int(meta['base_channels']))
    m.load_state_dict(ck['model_state_dict'], strict=True)
    return m.eval(), meta


st = np.load(f'{R}/uq/d8/eval/d8_main__cubes.npz')
files = [str(f) for f in st['files']]
log('stored files', files, 'samples', st['samples'].shape)
ds = dataset_for(DATA_ROOT, PRIOR, 8, [os.path.join(DATA_ROOT, f) for f in files])
rel = [os.path.relpath(f, DATA_ROOT) for f in ds.img_list]
order = [rel.index(f) for f in files]
x, y, xh = residual_collate_fn([ds[i] for i in order])
log('x', tuple(x.shape), 'y', tuple(y.shape), 'xh', tuple(xh.shape), 'sigma_d', ds.sigma_d)
log('max |x - stored x|', float((x - torch.from_numpy(st['x'])).abs().max()))

centre, mc = load(f'{R}/runs/regress_l1_sel_d8_b64/checkpoint_epoch_11.pth')
diff, md = load(f'{R}/runs/residual_reg_d8_b64/checkpoint_epoch_100.pth')
vh, mv = load(f'{R}/runs/varhead_d8_b64/checkpoint_epoch_10.pth')
sigma_r = float(md['sigma_r'])
assert abs(sigma_r - float(mv['sigma_r'])) < 1e-12
log('modes', mc['mode'], md['mode'], mv['mode'], 'sigma_r', sigma_r)

with torch.no_grad():
    xr = compose(xh, float(ds.sigma_d), regress_forward(centre, {'sensor': y, 'x0_hat': xh}))
    log('max |x_r - stored x_r|', float((xr - torch.from_numpy(st['x_r'])).abs().max()))
    logvar = varhead_forward(vh, {'sensor': y, 'x0_hat': xr}).clamp(*LOGVAR_RANGE)
sig_vh = torch.exp(0.5 * logvar)                 # standardised units (multiply by sigma_r for [-1, 1])
e = (x - xr) / sigma_r                           # true standardised residual
log('e rms', float(e.pow(2).mean().sqrt()), 'sig_vh rms', float(sig_vh.pow(2).mean().sqrt()))

np.savez(f'{OUT}/base.npz', x=x.numpy(), y=y.numpy(), xh=xh.numpy(), xr=xr.numpy(), e=e.numpy(), sig_vh=sig_vh.numpy(),
         files=np.array(files), sigma_r=sigma_r, sigma_d=float(ds.sigma_d))
z = torch.zeros
conds = {
    'true': {'sensor': y, 'x0_hat': xr},
    'swap': {'sensor': y.roll(4, 0), 'x0_hat': xr.roll(4, 0)},
    'zero': {'sensor': z(y.shape), 'x0_hat': z(xr.shape)},
    'swap_y': {'sensor': y.roll(4, 0), 'x0_hat': xr},
    'swap_xr': {'sensor': y, 'x0_hat': xr.roll(4, 0)},
}
trainer = DiffusionTrainer(device=torch.device('cpu'), loss_type='l1', snr_gamma=None, prediction_type='v', n_timesteps=1000)

# ---- one-call denoiser sensitivity on x_t built from the true residual
den = {}
gen = torch.Generator().manual_seed(1234)
noise = torch.randn(e.shape, generator=gen)
with torch.no_grad():
    for t_ in (999, 900, 700, 500, 300, 100, 20):
        ab = trainer.alpha_bars[t_]
        xt = ab.sqrt() * e + (1 - ab).sqrt() * noise
        tt = torch.full((e.shape[0],), t_, dtype=torch.long)
        for c, cond in conds.items():
            v = diff(xt, cond, tt)
            x0 = ab.sqrt() * xt - (1 - ab).sqrt() * v
            den[f'{t_}_{c}'] = x0.half().numpy()
        log('denoiser t', t_)
np.savez(f'{OUT}/denoiser.npz', **den)

# ---- samples
res = {}
plan = [('true', K_MAIN), ('swap', K_MAIN), ('zero', K_MAIN), ('swap_y', K_PART), ('swap_xr', K_PART)]
with torch.no_grad():
    for c, Kc in plan:
        out = []
        for k in range(Kc):
            torch.manual_seed(k)
            r = trainer.sample(diff, conds[c], tuple(xr.shape), n_steps=10, method='ddim', progress=False,
                               clamp_x0=(-2.0 / sigma_r, 2.0 / sigma_r))
            out.append(r)
            log('cond', c, 'seed', k, 'nfe', trainer.last_nfe)
        res[c] = torch.stack(out, 0).half().numpy()
        np.savez(f'{OUT}/samples_{c}.npz', r=res[c])
log('done')
