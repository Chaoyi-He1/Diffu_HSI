import json, os
import numpy as np
RR = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
EV = os.path.join(RR, 'uq/d4', 'ev' + 'al')
print(sorted(os.listdir(EV)))
print(sorted(os.listdir(os.path.join(RR, 'uq/d4'))))
rec = json.load(open(os.path.join(EV, 'd4_contrast.json')))
def show(o, pre='', depth=0):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, (dict,)) and depth < 3:
                print(pre + str(k) + ':')
                show(v, pre + '  ', depth + 1)
            elif isinstance(v, list):
                print(pre + f'{k}: list[{len(v)}] ' + (str(v[:3]) if len(v) and not isinstance(v[0], (list, dict)) else ''))
            else:
                s = str(v)
                print(pre + f'{k}: {s[:200]}')
show(rec)
z = np.load(os.path.join(EV, 'd4_contrast__cubes.npz'))
for k in z.files:
    print(k, z[k].shape, z[k].dtype)
print(list(z['files']))
