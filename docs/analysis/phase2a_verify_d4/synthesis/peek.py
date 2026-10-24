import numpy as np, json
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart/uq/'
for p in [R + 'd4/eval/d4_contrast__cubes.npz', R + 'd8/eval/d8_main__cubes.npz']:
    z = np.load(p, allow_pickle=True)
    print(p.split('/')[-1], {k: (z[k].shape, z[k].dtype) for k in z.files})
    print(list(z['files']))
a = json.load(open(R + 'd4/eval/d4_contrast.json')); b = json.load(open(R + 'd8/eval/d8_main.json')); c = json.load(open(R + 'd8/eval/d8_fix_k64.json'))
print(a['files'] == b['files'], b['files'] == c['files'])
for k in a.keys():
    if k not in ('per_cube', 'means', 'files', 'classes'):
        print(k, str(a[k])[:400])
