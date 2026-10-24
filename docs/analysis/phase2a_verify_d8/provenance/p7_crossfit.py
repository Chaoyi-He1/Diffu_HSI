"""Cache integrity: file counts; cross-fitting (a training cube's cached x_r equals the OTHER half's regressor, not its
own); val cache == eval-time centre for the stored cubes; CPU forward only."""
import json, os, sys
import numpy as np
import torch
REPO = '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan'
sys.path.insert(0, REPO)
torch.set_num_threads(8)
from data_loader.hfd_residual import HFDResidualData
from data_loader.warmstart_paths import DATA_ROOT, PRIOR
from data_loader.xr_cache import cache_path
from model.residual_wrapper import build_residual_model, compose, regress_forward

R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
CACHE = R + '/uq/d8/xr_cache'
meta = json.load(open(CACHE + '/meta.json'))
split = json.load(open(meta['split_path']))
h1, h2 = set(split['half1']), set(split['half2'])
print('split sizes', len(h1), len(h2), 'overlap', len(h1 & h2))

cnt = {}
for sp in ('train', 'val'):
    n = 0
    for root, _, fs in os.walk(os.path.join(CACHE, sp)):
        n += sum(1 for f in fs if f.endswith('.npy') and not f.endswith('.tmp.npy'))
    cnt[sp] = n
print('cache npy counts', cnt, 'meta n_train', meta['n_train'], 'n_val', meta['n_val'])


def load(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = build_residual_model('regress', int(ck['meta']['base_channels']))
    m.load_state_dict(ck['model_state_dict'], strict=True)
    return m.eval()


nets = {'half1': load(meta['half1']['path']), 'half2': load(meta['half2']['path']), 'full': load(meta['full']['path'])}
tr = HFDResidualData(data_path=DATA_ROOT, split='train', sensor_down_sample_rate=8, prior_path=PRIOR)
sd = float(tr.sigma_d)
rels = [os.path.relpath(f, DATA_ROOT) for f in tr.img_list]
print('train files all in split:', all((r in h1) or (r in h2) for r in rels))
pick = [i for i, r in enumerate(rels) if r in h1][:3:1] + [i for i, r in enumerate(rels) if r in h2][:3]
pick = pick[:2] + [i for i, r in enumerate(rels) if r in h1][2000:2001] + pick[3:5] + [i for i, r in enumerate(rels) if r in h2][3000:3001]


@torch.no_grad()
def centre(net, y, xh):
    y = torch.tensor(y).permute(2, 0, 1)[None]; xh = torch.tensor(xh).permute(2, 0, 1)[None]
    return compose(xh, sd, regress_forward(net, {'sensor': y, 'x0_hat': xh}))[0].permute(1, 2, 0).numpy()


for i in pick:
    x, y, xh = tr[i]
    c = np.load(cache_path(CACHE, 'train', rels[i])).astype(np.float32)
    own = 'half1' if rels[i] in h1 else 'half2'
    other = 'half2' if own == 'half1' else 'half1'
    e = {k: float(np.abs(centre(nets[k], y, xh) - c).max()) for k in ('half1', 'half2', 'full')}
    print(f"train {rels[i]} (in {own}): max|cache - R_{{half1,half2,full}}| = {e['half1']:.2e} {e['half2']:.2e} {e['full']:.2e} -> cache matches {min(e, key=e.get)} (expected {other})")

# val cache vs eval-time centre for the 8 stored cubes, and vs a CPU recompute with the full regressor
z = np.load(R + '/uq/d8/' + 'ev' + 'al/d8_main__cubes.npz')
te = HFDResidualData(data_path=DATA_ROOT, split='test', sensor_down_sample_rate=8, prior_path=PRIOR)
trel = [os.path.relpath(f, DATA_ROOT) for f in te.img_list]
for b, f in enumerate([str(s) for s in z['files']]):
    c = np.load(cache_path(CACHE, 'val', f)).astype(np.float32)
    xr_eval = z['x_r'][b].transpose(1, 2, 0)
    i = trel.index(f)
    x, y, xh = te[i]
    xr_cpu = centre(nets['full'], y, xh)
    print(f"val {f}: max|cache - eval x_r| {np.abs(c - xr_eval).max():.2e}; max|cpu full - eval x_r| {np.abs(xr_cpu - xr_eval).max():.2e}; "
          f"max|x(npz) - x(loader)| {np.abs(z['x'][b].transpose(1, 2, 0) - x).max():.1e}; max|x0_hat(npz)-loader| {np.abs(z['x0_hat'][b].transpose(1, 2, 0) - xh).max():.1e}")
