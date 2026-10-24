import json, numpy as np
CB = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d8/condition-blindness/'
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
rec = json.load(open(R + '/uq/d8/' + 'ev' + 'al/d8_main.json'))
pc = rec['per_cube']; cl = np.array([f.rsplit('_', 1)[0] for f in rec['files']])
def cse(d):
    d = np.asarray(d, float); n = len(d); m = d.mean(); u = np.unique(cl); G = len(u)
    s = np.array([(d[cl == g] - m).sum() for g in u])
    return m, d.std(ddof=1) / np.sqrt(n), np.sqrt(G / (G - 1) * (s ** 2).sum()) / n
ref = json.load(open(CB + 'ref200.json'))['per_cube']; rc = json.load(open(CB + 'ref_c.json'))
vh_rec = np.array(pc['varhead']['crps']); vh_cb = np.array(ref['vh_crps'])
print('order check: max |vh_crps(cb) - record varhead crps|', np.abs(vh_cb - vh_rec).max())
print('reg rmse check', np.abs(np.array(ref['reg_rmse']) - np.array(pc['regressor']['rmse_pct'])).max())
dif = np.array(pc['diffusion']['crps'])
for c in sorted(rc, key=float):
    h = np.array(rc[c]['crps'])
    m, s, sc = cse(dif - h)
    print(f'c={c}: homo CRPS {h.mean():.4f} cov50 {np.mean(rc[c]["cov50"]):.3f} cov90 {np.mean(rc[c]["cov90"]):.3f}  diff-homo {m:+.4f} iid {s:.4f} cl {sc:.4f}  frac diffusion better {np.mean(dif < h):.3f}')
for c in ('0.5', '0.6', '0.7', '1.0'):
    print('ref200 crps_homo', c, np.mean(ref['crps_homo_' + c]))
m, s, sc = cse(vh_rec - np.array(rc['0.5']['crps'])); print('varhead - homo(0.5)', m, s, sc)
