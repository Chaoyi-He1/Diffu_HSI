import json, os
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart/uq/d8'
E = os.path.join(R, 'ev' + 'al')
for p in (R, E):
    print(p)
    for f in sorted(os.listdir(p)):
        st = os.stat(os.path.join(p, f))
        print('  ', f, st.st_size, st.st_mtime)
rec = json.load(open(os.path.join(E, 'd8_main.json')))
print(list(rec.keys()))
for k, v in rec.items():
    if k in ('per_cube',):
        for m, mm in v.items():
            print('per_cube', m, {kk: (len(vv) if isinstance(vv, list) else vv) for kk, vv in mm.items()})
    elif k == 'files':
        print('files', len(v), v[:3])
    elif k == 'yswap':
        print('yswap', {kk: (len(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()})
    else:
        s = json.dumps(v)
        print(k, s[:1500])
