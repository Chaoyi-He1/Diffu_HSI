"""Where does the d = 4 gain come from? Split each model's squared error on the 800 MONITORING files into the measured
grid pixels (rows and columns that are multiples of d: 1/16 of the pixels at d = 4) and the interpolated pixels (15/16),
for the baseline x0_hat, the selected L1 regressor (regress_l1_sel_d4_b64 epoch 17) and B's one-step read-out (ddim1).
CPU only, never the gate cubes. Writes one JSON under RESULTS_ROOT/diagnostics (refuses an existing file)."""
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

REPO = '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan'
sys.path.insert(0, REPO)
from data_loader.warmstart_paths import RESULTS_ROOT, SUBSET, file_sha256, git_head  # noqa: E402

OUT = os.path.join(RESULTS_ROOT, 'diagnostics', 'regress_spatial_split_monitoring.json')
if os.path.exists(OUT):
    raise SystemExit(f'{OUT} exists')
torch.set_num_threads(8)
from torch.utils.data import DataLoader
from data_loader.hfd_residual import limit_worker_threads, residual_collate_fn
from main_2d_residual_fsdp import make_datasets
from model.diffusion_trainer import DiffusionTrainer
from model.residual_wrapper import build_residual_model, regress_forward
from scripts.eval_warmstart import reconstruct, sampler_grid

head = git_head()
a0 = json.load(open(os.path.join(RESULTS_ROOT, 'runs', 'regress_l1_sel_d4_b64', 'args.json')))
_, val = make_datasets(SimpleNamespace(data_root=a0['data_root'], d=a0['d'], prior=a0['prior'], limit_files=0))
files = [os.path.relpath(f, a0['data_root']) for f in val.img_list]
assert len(files) == 800 and not (set(files) & set(json.load(open(SUBSET))['files']))
classes = [f.split('/')[-2] for f in files]
d, sd = a0['d'], float(val.sigma_d)
loader = DataLoader(val, batch_size=8, shuffle=False, num_workers=4, collate_fn=residual_collate_fn, worker_init_fn=limit_worker_threads)

dev = torch.device('cpu')
reg = build_residual_model('regress', 64).eval()
reg.load_state_dict(torch.load(os.path.join(RESULTS_ROOT, 'runs', 'regress_l1_sel_d4_b64', 'checkpoint_epoch_17.pth'), map_location='cpu', weights_only=False)['model_state_dict'], strict=True)
bnet = build_residual_model('residual', 64).eval()
bnet.load_state_dict(torch.load(os.path.join(RESULTS_ROOT, 'runs', 'residual_d4_b64', 'checkpoint_epoch_100.pth'), map_location='cpu', weights_only=False)['model_state_dict'], strict=True)
trainer = DiffusionTrainer(device=dev, prediction_type='v', loss_type='l1', snr_gamma=None)
ddim1 = next(c for c in sampler_grid('full') if c['name'] == 'ddim1')

H = W = 64
grid = torch.zeros(H, W, dtype=torch.bool)
grid[::d, ::d] = True                                   # the lift anchors sample (i, j) at pixel (i*d, j*d)
masks = {'measured': grid, 'interpolated': ~grid}
# distance class of an interpolated pixel to the nearest grid line (for the gradient of error with distance)
ii = torch.arange(H) % d
dist = torch.minimum(ii, (d - ii) % d)                  # 0,1,2,1 at d = 4
dist2 = torch.maximum(dist[:, None], dist[None, :])     # Chebyshev distance to the nearest grid node line
models = ['baseline', 'regressor_l1_sel17', 'B_ddim1']
acc = {m: {k: [] for k in masks} for m in models}       # per-file mean squared error (on the [0,1] display scale) per mask
accd = {m: {int(k): [] for k in dist2.unique().tolist()} for m in models}
t0 = time.time()
with torch.no_grad():
    for bi, (x, y, xh) in enumerate(loader):
        preds = {'baseline': xh,
                 'regressor_l1_sel17': (xh + sd * regress_forward(reg, {'sensor': y, 'x0_hat': xh})).clamp(-1, 1),
                 'B_ddim1': reconstruct(bnet, trainer, 'residual', (x, y, xh), ddim1, sd, seed=0)}
        for m, p in preds.items():
            se = (((p.float() + 1) / 2 - (x + 1) / 2) ** 2).mean(1)          # [B, H, W], mean over bands, [0,1] scale
            for k, mk in masks.items():
                acc[m][k].extend((se[:, mk].mean(1) * 1e4).tolist())        # pct² (×100²)
            for dv in accd[m]:
                acc_mask = dist2 == dv
                accd[m][dv].extend((se[:, acc_mask].mean(1) * 1e4).tolist())
        if bi % 25 == 0:
            print(f'batch {bi} ({time.time() - t0:.0f} s)', flush=True)


def summ(v):
    v = np.asarray(v)
    return dict(mean_energy_pct2=float(v.mean()), rms_pct=float(np.sqrt(v.mean())))


out = dict(files=files, classes=classes, d=d, sigma_d=sd, n_measured=int(grid.sum()), n_pixels=H * W,
           energy={m: {k: summ(v) for k, v in acc[m].items()} for m in models},
           energy_by_distance={m: {str(k): summ(v) for k, v in accd[m].items()} for m in models},
           per_file={m: {k: acc[m][k] for k in masks} for m in models},
           provenance=dict(script=os.path.abspath(__file__), git_head=head, seconds=time.time() - t0,
                           regressor='runs/regress_l1_sel_d4_b64/checkpoint_epoch_17.pth', B='runs/residual_d4_b64/checkpoint_epoch_100.pth',
                           note='mean over bands of the squared error on the [0,1] display scale, ×1e4 = pct²; measured = pixels (i*d, j*d)'))
tmp = OUT + '.tmp'
json.dump(out, open(tmp, 'w'))
os.replace(tmp, OUT)
for m in models:
    e = out['energy'][m]
    print(f"{m:22s} measured {e['measured']['mean_energy_pct2']:.3f} pct² (rms {e['measured']['rms_pct']:.3f})   interpolated {e['interpolated']['mean_energy_pct2']:.3f} (rms {e['interpolated']['rms_pct']:.3f})")
for m in models:
    print(m, 'by distance to the grid:', {k: round(v['mean_energy_pct2'], 3) for k, v in out['energy_by_distance'][m].items()})
b = out['energy']['baseline']
for m in models[1:]:
    e = out['energy'][m]
    print(f"{m}: removes {100 * (1 - e['measured']['mean_energy_pct2'] / b['measured']['mean_energy_pct2']):.1f} % of the baseline energy on measured pixels, "
          f"{100 * (1 - e['interpolated']['mean_energy_pct2'] / b['interpolated']['mean_energy_pct2']):.1f} % on interpolated pixels")
print('wrote', OUT, f'({time.time() - t0:.0f} s)')
