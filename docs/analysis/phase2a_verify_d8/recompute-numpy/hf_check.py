import json
import numpy as np
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
rec = json.load(open(f'{R}/uq/d8/eval/d8_main.json')); pc = rec['per_cube']; files = rec['files']
z = np.load(rec['stored_cubes']); sf = [str(s) for s in z['files']]
def E(u):   # u [C,H,W] on [0,1]
    p = np.pad(u, [(0, 0), (1, 1), (1, 1)], mode='reflect')
    lap = p[:, :-2, 1:-1] + p[:, 2:, 1:-1] + p[:, 1:-1, :-2] + p[:, 1:-1, 2:] - 4 * p[:, 1:-1, 1:-1]
    return np.mean(lap ** 2)
for j, f in enumerate(sf):
    i = files.index(f)
    x = (z['x'][j].astype(np.float64) + 1) / 2
    xr = (z['x_r'][j].astype(np.float64) + 1) / 2
    xh = (z['x0_hat'][j].astype(np.float64) + 1) / 2
    s0 = (z['samples'][0, j].astype(np.float64) + 1) / 2
    mu = ((z['samples'][:, j].astype(np.float64) + 1) / 2).mean(0)
    Ex = E(x)
    print(f'{f}: reg hf {np.log(E(xr)/Ex):+.4f} rec {pc["regressor"]["hf"][i]:+.4f} | base hf {np.log(E(xh)/Ex):+.4f} rec {pc["baseline"]["hf"][i]:+.4f} | '
          f'single hf {np.log(E(s0)/Ex):+.4f} rec {pc["diffusion"]["single_hf"][i]:+.4f} | meanK hf {np.log(E(mu)/Ex):+.4f} rec {pc["diffusion"]["mean_hf"][i]:+.4f}')
hs = np.abs(np.asarray(pc['diffusion']['single_hf'])); hr = np.abs(np.asarray(pc['regressor']['hf']))
print('P1 per-cube: |hf_single| < |hf_reg| in', int((hs < hr).sum()), 'of 200; sign(single_hf)>0 in', int((np.asarray(pc['diffusion']['single_hf']) > 0).sum()),
      '; sign(reg hf)<0 in', int((np.asarray(pc['regressor']['hf']) < 0).sum()))
print('median |hf_single|', np.median(hs), 'median |hf_reg|', np.median(hr))
