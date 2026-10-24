"""Different-route recomputation of the per-cube metrics from the stored figure cubes (8 y-swap cubes, 32 samples each),
with implementations written here (scipy correlate for the Laplacian, explicit pairwise CRPS, hand-rolled Weibull order
statistics). Compares with (a) the y-swap 'true' RMSE (same draws) and (b) the gate per-cube arrays (other draws of the
same cubes: the gate batch is 4 cubes, the y-swap batch 32, so the noise differs). Also the finite-K bias of S1."""
import json
import math
import os

import numpy as np
from scipy.ndimage import correlate

R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart/uq/d8'
E = os.path.join(R, 'ev' + 'al')
OUT = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d8/recompute-bootstrap'
rec = json.load(open(os.path.join(E, 'd8_main.json')))
pc = rec['per_cube']
idx_of = {f: i for i, f in enumerate(rec['files'])}
z = np.load(os.path.join(E, 'd8_main__cubes.npz'))
print({k: (z[k].shape, z[k].dtype) for k in z.files})
files = [str(f) for f in z['files']]
assert files == rec['stored_files'], (files, rec['stored_files'])
S = z['samples'].astype(np.float64)          # [K, B, C, H, W] on [-1, 1]
X = z['x'].astype(np.float64)
XH = z['x0_hat'].astype(np.float64)
XR = z['x_r'].astype(np.float64)
K = S.shape[0]
to01 = lambda a: (a + 1.0) / 2.0
LAP = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)[None]   # [1, 3, 3]: per band


def rmse_pct(a, b):
    return 100.0 * math.sqrt(np.mean((to01(a) - to01(b)) ** 2))


def hf(a, b):
    ea = np.mean(correlate(to01(a), LAP, mode='mirror') ** 2)
    eb = np.mean(correlate(to01(b), LAP, mode='mirror') ** 2)
    return math.log(ea / eb)


def crps_pairwise(s, x):
    """s [K, ...], x [...] on [0, 1]: mean |s_k - x| - sum_{j<k} |s_j - s_k| / (K (K - 1)), pairwise loop."""
    Kk = s.shape[0]
    t1 = np.mean(np.abs(s - x[None]), axis=0)
    t2 = np.zeros_like(x)
    for j in range(Kk):
        t2 += np.sum(np.abs(s[j + 1:] - s[j][None]), axis=0)
    return float(np.mean(t1 - t2 / (Kk * (Kk - 1))))


def weibull_interval(s, level):
    """Order statistics at positions p (K + 1), linear interpolation, 1-based."""
    ss = np.sort(s, axis=0)
    Kk = ss.shape[0]
    out = []
    for p in ((1 - level) / 2, 1 - (1 - level) / 2):
        h = p * (Kk + 1)
        lo_i = int(math.floor(h))
        frac = h - lo_i
        assert 1 <= lo_i <= Kk - 1 or (lo_i == Kk and frac == 0)
        a = ss[lo_i - 1]
        b = ss[min(lo_i, Kk - 1)]
        out.append(a + frac * (b - a))
    return out


rows = []
for i, f in enumerate(files):
    gi = idx_of[f]
    s01, x01 = to01(S[:, i]), to01(X[i])
    meanK = S[:, i].mean(0)
    r = dict(file=f)
    r['rmse_reg'], r['rmse_reg_rec'] = rmse_pct(XR[i], X[i]), pc['regressor']['rmse_pct'][gi]
    r['rmse_base'], r['rmse_base_rec'] = rmse_pct(XH[i], X[i]), pc['baseline']['rmse_pct'][gi]
    r['hf_reg'], r['hf_reg_rec'] = hf(XR[i], X[i]), pc['regressor']['hf'][gi]
    r['hf_base'], r['hf_base_rec'] = hf(XH[i], X[i]), pc['baseline']['hf'][gi]
    r['rmse_meanK'] = rmse_pct(meanK, X[i])
    r['rmse_meanK_yswap_true'] = rec['yswap']['true'][rec['yswap']['files'].index(f)]
    r['rmse_meanK_gate'] = pc['diffusion']['mean_rmse_pct'][gi]
    r['rmse_single0'] = rmse_pct(S[0, i], X[i])
    r['rmse_single_all'] = [rmse_pct(S[k, i], X[i]) for k in range(K)]
    r['rmse_single_gate'] = pc['diffusion']['single_rmse_pct'][gi]
    r['hf_single0'] = hf(S[0, i], X[i])
    r['hf_single_all'] = [hf(S[k, i], X[i]) for k in range(K)]
    r['hf_single_gate'] = pc['diffusion']['single_hf'][gi]
    r['hf_meanK'], r['hf_meanK_gate'] = hf(meanK, X[i]), pc['diffusion']['mean_hf'][gi]
    r['crps'], r['crps_gate'] = 100.0 * crps_pairwise(s01, x01), pc['diffusion']['crps'][gi]
    r['crps_varhead_gate'] = pc['varhead']['crps'][gi]
    lo90, hi90 = weibull_interval(s01, 0.9)
    lo50, hi50 = weibull_interval(s01, 0.5)
    r['cov90'], r['cov90_gate'] = float(np.mean((x01 >= lo90) & (x01 <= hi90))), pc['diffusion']['cov90'][gi]
    r['cov50'], r['cov50_gate'] = float(np.mean((x01 >= lo50) & (x01 <= hi50))), pc['diffusion']['cov50'][gi]
    var = s01.var(0, ddof=1)
    err2 = (s01.mean(0) - x01) ** 2
    r['spread_ratio'], r['spread_ratio_gate'] = float(var.mean() / err2.mean()), pc['diffusion']['spread_ratio'][gi]
    # finite-K bias of the mean-of-K: E||meanK - x||^2 = ||E s - x||^2 + mean(var)/K
    r['rmse_mu_debiased'] = 100.0 * math.sqrt(max(err2.mean() - var.mean() / K, 0.0))
    # mean residual relative to the centre: is the sampler mean offset from x_r?
    r['rms_meanK_minus_xr_pct'] = 100.0 * math.sqrt(np.mean((to01(meanK) - to01(XR[i])) ** 2))
    r['expected_rms_meanK_minus_xr_if_centred_pct'] = 100.0 * math.sqrt(var.mean() / K)
    rows.append(r)

print('\n%-34s %8s %8s | %8s %8s | %8s %8s | %8s %8s' % ('file', 'rmseR', 'rec', 'hfR', 'rec', 'rmseB', 'rec', 'hfB', 'rec'))
for r in rows:
    print('%-34s %8.4f %8.4f | %8.4f %8.4f | %8.4f %8.4f | %8.4f %8.4f' % (r['file'], r['rmse_reg'], r['rmse_reg_rec'], r['hf_reg'], r['hf_reg_rec'],
                                                                        r['rmse_base'], r['rmse_base_rec'], r['hf_base'], r['hf_base_rec']))
print('\n%-34s %8s %8s %8s | %8s %8s %8s %8s | %8s %8s' % ('file', 'meanK', 'ys_true', 'gate', 'single0', 'k-range', '', 'gate', 'hf_s0', 'gate'))
for r in rows:
    print('%-34s %8.4f %8.4f %8.4f | %8.4f %8.4f %8.4f %8.4f | %8.4f %8.4f' % (
        r['file'], r['rmse_meanK'], r['rmse_meanK_yswap_true'], r['rmse_meanK_gate'], r['rmse_single0'], min(r['rmse_single_all']),
        max(r['rmse_single_all']), r['rmse_single_gate'], r['hf_single0'], r['hf_single_gate']))
print('\n%-34s %8s %8s %8s | %7s %7s | %7s %7s | %7s %7s' % ('file', 'crps', 'gate', 'vh_gate', 'cov90', 'gate', 'cov50', 'gate', 'sprd', 'gate'))
for r in rows:
    print('%-34s %8.4f %8.4f %8.4f | %7.4f %7.4f | %7.4f %7.4f | %7.3f %7.3f' % (
        r['file'], r['crps'], r['crps_gate'], r['crps_varhead_gate'], r['cov90'], r['cov90_gate'], r['cov50'], r['cov50_gate'],
        r['spread_ratio'], r['spread_ratio_gate']))
print('\nfinite-K check on the stored cubes:')
for r in rows:
    print('%-34s meanK %.4f  debiased mu %.4f  regressor %.4f  | rms(meanK - x_r) %.4f vs %.4f expected if exactly centred' % (
        r['file'], r['rmse_meanK'], r['rmse_mu_debiased'], r['rmse_reg'], r['rms_meanK_minus_xr_pct'], r['expected_rms_meanK_minus_xr_if_centred_pct']))
hf_all = np.array([r['hf_single_all'] for r in rows])
print(f"\nsingle-sample hf over all 32 x 8 stored draws: mean {hf_all.mean():+.3f}, min {hf_all.min():+.3f}, max {hf_all.max():+.3f}; "
      f"stored-cube hf_reg mean {np.mean([r['hf_reg'] for r in rows]):+.3f}")
print('max |rmse_reg - rec|', max(abs(r['rmse_reg'] - r['rmse_reg_rec']) for r in rows),
      ' max |hf_reg - rec|', max(abs(r['hf_reg'] - r['hf_reg_rec']) for r in rows),
      ' max |rmse_base - rec|', max(abs(r['rmse_base'] - r['rmse_base_rec']) for r in rows),
      ' max |hf_base - rec|', max(abs(r['hf_base'] - r['hf_base_rec']) for r in rows),
      ' max |meanK - yswap true|', max(abs(r['rmse_meanK'] - r['rmse_meanK_yswap_true']) for r in rows))
json.dump(rows, open(os.path.join(OUT, 'stored_cubes.json'), 'w'), indent=1)
