import json, numpy as np
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
rec = json.load(open(R + '/uq/d8/' + 'ev' + 'al/d8_main.json'))
print(rec.keys())
print('K', rec['K'], 'sampler', rec['sampler'], 'n', rec['n'], 'sigma_r', rec['sigma_r'], 'sigma_d', rec['sigma_d'])
for m, d in rec['per_cube'].items():
    print(m, sorted(d.keys()))
for m, d in rec['means'].items():
    print(m, {k: round(v, 4) for k, v in d.items()})
print('yswap keys', rec['yswap'].keys())
ys = rec['yswap']
for k in ('true', 'swap_full', 'baseline', 'regressor'):
    print(k, np.mean(ys[k]))
print('stored', rec['stored_files'])
print(json.dumps(rec['provenance'], indent=1)[:3000])
v = json.load(open(R + '/uq/d8/verdict__d8_main.json'))
print(json.dumps({k: v[k] for k in v if k not in ('centre',)}, indent=1)[:4000])
