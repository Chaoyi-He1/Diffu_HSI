import json, os, re
import numpy as np
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
rec = json.load(open(f'{R}/uq/d8/eval/d8_main.json'))
files = rec['files']; ys = rec['yswap']; pc = rec['per_cube']
def image_of(f):
    p = f.split('/'); stem = os.path.splitext(p[-1])[0]
    return p[-2] + '/' + re.match(r'^(.*)_(\d+)$', stem).group(1)
def cse(d, g):
    d = np.asarray(d, float); g = np.asarray(g); r = d - d.mean(); labs = sorted(set(g)); G = len(labs)
    return np.sqrt(G / (G - 1) * sum(r[g == l].sum() ** 2 for l in labs)) / len(d), G
yf = ys['files']; idx = [files.index(f) for f in yf]
d = np.asarray(ys['swap_full']) - np.asarray(ys['true'])
se_iid = d.std(ddof=1) / np.sqrt(len(d)); se_c, G = cse(d, [image_of(f) for f in yf])
print(f'S2 as recorded: mean {d.mean():.6f} iid SE {se_iid:.6f} cluster SE {se_c:.6f} (G={G}); mean-2SE_iid {d.mean()-2*se_iid:.6f}; mean-2SE_c {d.mean()-2*se_c:.6f}')
print('  SE needed to pass at this mean:', d.mean() / 2)
d2 = np.asarray(ys['swap_full']) - np.asarray(pc['diffusion']['mean_rmse_pct'])[idx]
print(f'S2 with main-run mean-of-K as true: mean {d2.mean():.6f} iid SE {d2.std(ddof=1)/np.sqrt(32):.6f} -> pass={d2.mean()-2*d2.std(ddof=1)/np.sqrt(32)>0}')
dt = np.asarray(ys['true']) - np.asarray(pc['diffusion']['mean_rmse_pct'])[idx]
print(f'yswap.true - main mean-of-K on same 32 cubes: mean {dt.mean():.6f} sd {dt.std(ddof=1):.6f} maxabs {np.abs(dt).max():.6f}')
d3 = np.asarray(ys['swap_full']) - np.asarray(ys['regressor'])
print(f'swap_full - regressor (same batch): mean {d3.mean():.6f} iid SE {d3.std(ddof=1)/np.sqrt(32):.6f}')
d4 = np.asarray(ys['true']) - np.asarray(ys['regressor'])
print(f'true - regressor (same batch): mean {d4.mean():.6f} iid SE {d4.std(ddof=1)/np.sqrt(32):.6f}')
print('per-cube S2 diffs > 0 count:', int((d > 0).sum()), 'of 32')

# ---- stored cubes cross-check (8 of the y-swap cubes; float16 samples)
z = np.load(rec['stored_cubes'])
print('\nnpz keys', z.files, {k: (z[k].shape, z[k].dtype) for k in z.files})
sf = [str(s) for s in z['files']]
print('npz files == record stored_files:', sf == rec['stored_files'])
S = z['samples']; X = z['x']; XH = z['x0_hat']; XR = z['x_r']
def rmse_pct(a, b):
    a = (np.asarray(a, np.float64) + 1) / 2; b = (np.asarray(b, np.float64) + 1) / 2
    return np.sqrt(np.mean((a - b) ** 2)) * 100
def wq(s_sorted, pos):          # 1-based Weibull position, linear interpolation between order statistics
    lo = int(np.floor(pos)); fr = pos - lo
    return s_sorted[lo - 1] + fr * (s_sorted[lo] - s_sorted[lo - 1])
K = S.shape[0]
for j, f in enumerate(sf):
    i_ys = yf.index(f); i_pc = files.index(f)
    s = (S[:, j].astype(np.float64) + 1) / 2; x = (X[j].astype(np.float64) + 1) / 2
    mu = s.mean(0)
    rm_true = np.sqrt(np.mean((mu - x) ** 2)) * 100
    rm_r = rmse_pct(XR[j], X[j]); rm_b = rmse_pct(XH[j], X[j])
    rm_single = np.sqrt(np.mean((s[0] - x) ** 2)) * 100
    t1 = np.abs(s - x[None]).mean(0)
    pair = np.zeros_like(x)
    for a in range(K):
        pair += np.abs(s[a][None] - s[a + 1:]).sum(0)
    crps = (t1 - pair / (K * (K - 1))).mean() * 100     # fair form: (1/(2K(K-1))) * sum_{j!=k} = sum_{j<k}/(K(K-1))
    ss = np.sort(s, 0)
    cov = {}
    for lv in (0.5, 0.9):
        lo = wq(ss, (1 - lv) / 2 * (K + 1)); hi = wq(ss, (1 + lv) / 2 * (K + 1))
        cov[lv] = np.mean((x >= lo) & (x <= hi))
    sd = s.std(0, ddof=1)
    spread = np.mean(sd ** 2) / np.mean((mu - x) ** 2)
    print(f'{f}: meanK {rm_true:.4f} vs ys.true {ys["true"][i_ys]:.4f} | reg {rm_r:.4f} vs ys.reg {ys["regressor"][i_ys]:.4f} | base {rm_b:.4f} vs ys.base {ys["baseline"][i_ys]:.4f}')
    print(f'    own (ys draws)  crps {crps:.4f} cov90 {cov[0.9]:.4f} cov50 {cov[0.5]:.4f} single {rm_single:.4f} spread {spread:.3f}')
    print(f'    record main-run crps {pc["diffusion"]["crps"][i_pc]:.4f} cov90 {pc["diffusion"]["cov90"][i_pc]:.4f} cov50 {pc["diffusion"]["cov50"][i_pc]:.4f} single {pc["diffusion"]["single_rmse_pct"][i_pc]:.4f} spread {pc["diffusion"]["spread_ratio"][i_pc]:.3f} meanK {pc["diffusion"]["mean_rmse_pct"][i_pc]:.4f}')
