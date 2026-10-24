"""Stored d = 4 cubes (8 cubes of the y-swap batch, K = 32 true-condition draws; NOT the same draws as the gate loop,
which ran in batches of 4): own metric implementations, check of the record's per-cube values for the same files
(regressor / baseline should match up to fp nondeterminism; sample metrics only up to Monte Carlo noise), the per-element
variance behind the S1 debias, a constant-width-Gaussian CRPS comparison and whiteness of the spread."""
import json
import math

import numpy as np
from scipy.special import erf

R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart/uq/'
rec = json.load(open(R + 'd4/eval/d4_contrast.json'))
z = np.load(R + 'd4/eval/d4_contrast__cubes.npz')
S = z['samples']
X, XH, XR = z['x'], z['x0_hat'], z['x_r']
files = [str(f) for f in z['files']]
print('samples', S.shape, S.dtype, 'x', X.shape, X.dtype, 'x_r', XR.shape, XR.dtype, 'files', files)
assert files == rec['stored_files']
idx = [rec['files'].index(f) for f in files]
pc = rec['per_cube']
K = S.shape[0]
sig_r01 = rec['sigma_r'] / 2.0


def to01(a):
    return (a.astype(np.float64) + 1.0) / 2.0


def lap(u):
    p = np.pad(u, [(0, 0), (1, 1), (1, 1)], mode='reflect')
    return p[:, :-2, 1:-1] + p[:, 2:, 1:-1] + p[:, 1:-1, :-2] + p[:, 1:-1, 2:] - 4 * u


def hf(a, b):
    return math.log((lap(a) ** 2).mean() / (lap(b) ** 2).mean())


def sobel(u):
    p = np.pad(u, [(0, 0), (1, 1), (1, 1)], mode='reflect')
    # rows = H (axis 1), cols = W (axis 2)
    gx = (p[:, :-2, 2:] + 2 * p[:, 1:-1, 2:] + p[:, 2:, 2:]) - (p[:, :-2, :-2] + 2 * p[:, 1:-1, :-2] + p[:, 2:, :-2])
    gy = (p[:, 2:, :-2] + 2 * p[:, 2:, 1:-1] + p[:, 2:, 2:]) - (p[:, :-2, :-2] + 2 * p[:, :-2, 1:-1] + p[:, :-2, 2:])
    return np.sqrt(gx ** 2 + gy ** 2) / 8


def w1(a, b, bins=100, vmax=0.5):
    e = np.linspace(0, vmax, bins + 1)
    h1 = np.histogram(np.clip(sobel(a), 0, vmax), e)[0]
    h2 = np.histogram(np.clip(sobel(b), 0, vmax), e)[0]
    return float(np.abs(np.cumsum(h1) / h1.sum() - np.cumsum(h2) / h2.sum()).sum() * vmax / bins)


def crps_pairwise(s, x):
    Kk = s.shape[0]
    t1 = np.abs(s - x[None]).mean(0)
    acc = np.zeros_like(x)
    for j in range(Kk):
        acc += np.abs(s - s[j:j + 1]).sum(0)          # sum over k of |s_k - s_j|, includes k == j (zero)
    t2 = acc / (2 * Kk * (Kk - 1))
    return t1 - t2


def weibull_q(srt, p):
    Kk = srt.shape[0]
    h = p * (Kk + 1)                                  # 1-indexed position
    lo = int(math.floor(h))
    fr = h - lo
    return srt[lo - 1] + fr * (srt[lo] - srt[lo - 1])


def crps_gauss(mu, sd, x):
    zz = (x - mu) / sd
    Phi = 0.5 * (1 + erf(zz / math.sqrt(2)))
    phi = np.exp(-0.5 * zz * zz) / math.sqrt(2 * math.pi)
    return sd * (zz * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))


def ause(sd, err):
    e2 = (err ** 2).ravel()
    n = e2.size
    out = []
    for score in (sd.ravel(), np.abs(err).ravel()):
        o = np.argsort(-score, kind='stable')
        ee = e2[o]
        tail = np.cumsum(ee[::-1])[::-1]
        full = ee.mean()
        curve = []
        for f in np.arange(0, 1.0, 0.02):
            i = min(int(round(f * n)), n - 1)
            curve.append(math.sqrt(tail[i] / (n - i) / full))
        out.append(np.array(curve))
    return float(np.mean(out[0] - out[1]))


def corr(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))


rows = []
all_sd2, all_err_r = [], []
for c in range(len(files)):
    s = to01(S[:, c])                      # [K, C, H, W]
    x, xr, xh = to01(X[c]), to01(XR[c]), to01(XH[c])
    i = idx[c]
    mu = s.mean(0)
    sd = s.std(0, ddof=1)
    srt = np.sort(s, axis=0)
    r = dict(file=files[c])
    r['rmse_r_stored'] = 100 * math.sqrt(((xr - x) ** 2).mean())
    r['rmse_r_record'] = pc['regressor']['rmse_pct'][i]
    r['rmse_b_stored'] = 100 * math.sqrt(((xh - x) ** 2).mean())
    r['rmse_b_record'] = pc['baseline']['rmse_pct'][i]
    r['hf_r_stored'] = hf(xr, x)
    r['hf_r_record'] = pc['regressor']['hf'][i]
    r['w1_r_stored'] = w1(xr, x)
    r['w1_r_record'] = pc['regressor']['w1'][i]
    r['crps_stored'] = 100 * float(crps_pairwise(s, x).mean())
    r['crps_record'] = pc['diffusion']['crps'][i]
    lo50, hi50 = weibull_q(srt, 0.25), weibull_q(srt, 0.75)
    lo90, hi90 = weibull_q(srt, 0.05), weibull_q(srt, 0.95)
    r['cov50_stored'] = float(np.mean((x >= lo50) & (x <= hi50)))
    r['cov50_record'] = pc['diffusion']['cov50'][i]
    r['cov90_stored'] = float(np.mean((x >= lo90) & (x <= hi90)))
    r['cov90_record'] = pc['diffusion']['cov90'][i]
    # check own Weibull vs numpy
    q = np.quantile(s, [0.05, 0.95], axis=0, method='weibull')
    r['weibull_maxdiff'] = float(max(np.abs(q[0] - lo90).max(), np.abs(q[1] - hi90).max()))
    r['ause_stored'] = ause(sd, mu - x)
    r['ause_record'] = pc['diffusion']['ause'][i]
    r['spread_stored'] = float((sd ** 2).mean() / ((mu - x) ** 2).mean())
    r['spread_record'] = pc['diffusion']['spread_ratio'][i]
    r['meanK_rmse_stored'] = 100 * math.sqrt(((mu - x) ** 2).mean())
    r['meanK_rmse_record'] = pc['diffusion']['mean_rmse_pct'][i]
    r['single_rmse_stored'] = 100 * math.sqrt(((s[0] - x) ** 2).mean())
    r['single_rmse_record'] = pc['diffusion']['single_rmse_pct'][i]
    r['single_hf_stored'] = hf(s[0], x)
    r['single_hf_record'] = pc['diffusion']['single_hf'][i]
    r['single_w1_stored'] = w1(s[0], x)
    r['single_w1_record'] = pc['diffusion']['single_w1'][i]
    vbar = float((sd ** 2).mean()) * 1e4                      # %^2
    r['vbar_stored'] = vbar
    r['vbar_record_derived'] = pc['diffusion']['spread_ratio'][i] * pc['diffusion']['mean_rmse_pct'][i] ** 2
    r['S1_literal_stored'] = r['meanK_rmse_stored'] - r['rmse_r_stored']
    r['S1_debiased_stored'] = math.sqrt(r['meanK_rmse_stored'] ** 2 - vbar / K) - r['rmse_r_stored']
    # is the mean-of-K centred on x_r? RMS of (mu - x_r) vs the 1/K expectation sqrt(vbar/K)
    r['rms_mu_minus_xr'] = 100 * math.sqrt(((mu - xr) ** 2).mean())
    r['expected_rms_mu_minus_xr'] = math.sqrt(vbar / K)
    # whiteness of the spread vs the true residual
    e = s - mu[None]
    tr = x - xr
    r['lag1_spatial_spread'] = corr(e[..., :, :-1], e[..., :, 1:])
    r['lag1_spatial_trueres'] = corr(tr[..., :, :-1], tr[..., :, 1:])
    r['adjband_spread'] = corr(e[:, :-1], e[:, 1:])
    r['adjband_trueres'] = corr(tr[:-1], tr[1:])
    # constant-width Gaussian around x_r
    r['crps_gauss_own_rms_sd'] = 100 * float(crps_gauss(xr, math.sqrt((sd ** 2).mean()), x).mean())
    r['crps_gauss_half_sigma_r'] = 100 * float(crps_gauss(xr, 0.5 * sig_r01, x).mean())
    r['rms_sd_over_sigma_r01'] = math.sqrt((sd ** 2).mean()) / sig_r01
    all_sd2.append((sd ** 2).mean())
    all_err_r.append(((xr - x) ** 2).mean())
    rows.append(r)
    print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)

# global constant Gaussian (one width for all 8 cubes) and optimal one
gsd = math.sqrt(np.mean(all_sd2))
print('global RMS sample sd / sigma_r(0-1 scale) =', gsd / sig_r01)
cs = []
for c in range(len(files)):
    x, xr = to01(X[c]), to01(XR[c])
    cs.append(100 * float(crps_gauss(xr, gsd, x).mean()))
best = None
for f in np.linspace(0.2, 1.2, 51):
    v = np.mean([100 * float(crps_gauss(to01(XR[c]), f * sig_r01, to01(X[c])).mean()) for c in range(len(files))])
    if best is None or v < best[1]:
        best = (float(f), float(v))
summ = dict(
    crps_diffusion_stored_mean=float(np.mean([r['crps_stored'] for r in rows])),
    crps_diffusion_record_mean_same_files=float(np.mean([r['crps_record'] for r in rows])),
    crps_const_gauss_global_rms_sd=float(np.mean(cs)),
    crps_const_gauss_best=best,
    crps_gauss_half_sigma_r=float(np.mean([r['crps_gauss_half_sigma_r'] for r in rows])),
    crps_varhead_record_same_files=float(np.mean([pc['varhead']['crps'][i] for i in idx])),
    max_abs_rmse_r_diff=float(max(abs(r['rmse_r_stored'] - r['rmse_r_record']) for r in rows)),
    max_abs_rmse_b_diff=float(max(abs(r['rmse_b_stored'] - r['rmse_b_record']) for r in rows)),
    max_abs_hf_r_diff=float(max(abs(r['hf_r_stored'] - r['hf_r_record']) for r in rows)),
    max_abs_w1_r_diff=float(max(abs(r['w1_r_stored'] - r['w1_r_record']) for r in rows)),
    vbar_stored_mean=float(np.mean([r['vbar_stored'] for r in rows])),
    vbar_record_mean=float(np.mean([r['vbar_record_derived'] for r in rows])),
    S1_literal_stored_mean=float(np.mean([r['S1_literal_stored'] for r in rows])),
    S1_debiased_stored_mean=float(np.mean([r['S1_debiased_stored'] for r in rows])),
    S1_literal_record_same_files=float(np.mean([pc['diffusion']['mean_rmse_pct'][i] - pc['regressor']['rmse_pct'][i] for i in idx])),
    rms_mu_minus_xr=float(np.mean([r['rms_mu_minus_xr'] for r in rows])),
    expected_rms_mu_minus_xr=float(np.mean([r['expected_rms_mu_minus_xr'] for r in rows])),
    lag1_spatial_spread=float(np.mean([r['lag1_spatial_spread'] for r in rows])),
    lag1_spatial_trueres=float(np.mean([r['lag1_spatial_trueres'] for r in rows])),
    adjband_spread=float(np.mean([r['adjband_spread'] for r in rows])),
    adjband_trueres=float(np.mean([r['adjband_trueres'] for r in rows])),
    weibull_maxdiff=float(max(r['weibull_maxdiff'] for r in rows)),
)
for k in ('crps', 'cov50', 'cov90', 'ause', 'spread', 'meanK_rmse', 'single_rmse', 'single_hf', 'single_w1'):
    summ[k + '_stored_vs_record_mean'] = (float(np.mean([r[k + '_stored'] for r in rows])), float(np.mean([r[k + '_record'] for r in rows])))
print(json.dumps(summ, indent=1))
json.dump(dict(rows=rows, summary=summ), open('/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d4/recompute-d4/stored.json', 'w'), indent=1)
