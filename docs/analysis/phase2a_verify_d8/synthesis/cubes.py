"""Stored-cube checks to settle verifier disagreements: grid-phase spread (3.36 vs 3.56 %), varhead grid sigma (0.40 vs 0.49 %),
Spearman(sd, |e|) for diffusion and varhead (0.589 vs 0.63), band anomaly vs clamp, homoscedastic Gaussian CRPS, and whether the
'mean-of-K drift' CRPS cost is finite-K noise."""
import json, math, sys, numpy as np
from scipy.stats import spearmanr, norm
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
z = np.load(R + '/uq/d8/' + 'ev' + 'al/d8_main__cubes.npz')
for k in z.files:
    print(k, z[k].shape, z[k].dtype)
vh = np.load('/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d8/metric-validity/vh_sigma.npz')
print('vh keys', vh.files, vh['sigma'].shape)
rec = json.load(open(R + '/uq/d8/' + 'ev' + 'al/d8_main.json'))
files = list(rec['files']); sf = list(z['files'])
idx = [files.index(f) for f in sf]
S = z['samples'].astype(np.float32)          # [K, B, C, H, W] on [-1, 1]
x = z['x'].astype(np.float64); xr = z['x_r'].astype(np.float64)
s01 = (S.astype(np.float64) + 1) / 2; x01 = (x + 1) / 2; xr01 = (xr + 1) / 2
sig_vh01 = vh['sigma'].astype(np.float64) / 2
sigma_r = 0.08934239400551727
K = S.shape[0]
mu = s01.mean(0); sd = s01.std(0, ddof=1)
e_r = x01 - xr01
# --- grid phase
print('\n== per-phase RMS error / RMS sd (diffusion) / RMS sigma_vh, % of range; first 6 phases sorted by error')
rows = []
for a in range(8):
    for b in range(8):
        sl = (slice(None), slice(None), slice(a, None, 8), slice(b, None, 8))
        rows.append((a, b, 100 * np.sqrt((e_r[sl] ** 2).mean()), 100 * np.sqrt((sd[sl] ** 2).mean()), 100 * sd[sl].mean(),
                     100 * np.sqrt((sig_vh01[sl] ** 2).mean()), 100 * sig_vh01[sl].mean()))
rows.sort(key=lambda r: r[2])
for r in rows[:4] + rows[-2:]:
    print('phase', r[:2], 'err_rms %.3f  diff_sd rms %.3f mean %.3f  vh rms %.3f mean %.3f' % r[2:])
err_ph = np.array([r[2] for r in rows]); sd_ph = np.array([r[3] for r in rows]); vh_ph = np.array([r[5] for r in rows])
print('phase max/min: err %.2f diff_sd %.2f vh %.2f' % (err_ph.max() / err_ph.min(), sd_ph.max() / sd_ph.min(), vh_ph.max() / vh_ph.min()))
print('excluding grid phase: err %.2f diff_sd %.2f vh %.2f' % (err_ph[1:].max() / err_ph[1:].min(), sd_ph[1:].max() / sd_ph[1:].min(), vh_ph[1:].max() / vh_ph[1:].min()))
# --- Spearman
e_mu = np.abs(x01 - mu)
rng = np.random.default_rng(0); sel = rng.choice(e_r.size, 400000, replace=False)
f = lambda a: a.ravel()[sel]
print('\n== Spearman (400k random elements)')
print('diff sd vs |x - x_r|', spearmanr(f(sd), f(np.abs(e_r)))[0], ' vs |x - mu_K|', spearmanr(f(sd), f(e_mu))[0])
print('vh sigma vs |x - x_r|', spearmanr(f(sig_vh01), f(np.abs(e_r)))[0])
pix = lambda a: np.sqrt((a ** 2).mean(1)).ravel()
print('pixel level (RMS over bands): diff', spearmanr(pix(sd), pix(e_r))[0], 'vh', spearmanr(pix(sig_vh01), pix(e_r))[0])
# --- bands
print('\n== per band: diff sd rms, err rms, clamp fraction (samples at +-1), truth mean, varhead sigma rms (%)')
clamp = (np.abs(S) >= 0.9995).mean(axis=(0, 1, 3, 4))
lo = (S <= -0.9995).mean(axis=(0, 1, 3, 4))
sd_b = 100 * np.sqrt((sd ** 2).mean(axis=(0, 2, 3))); er_b = 100 * np.sqrt((e_r ** 2).mean(axis=(0, 2, 3)))
vh_b = 100 * np.sqrt((sig_vh01 ** 2).mean(axis=(0, 2, 3))); xm_b = x01.mean(axis=(0, 2, 3))
xlow_b = (x <= -0.98).mean(axis=(0, 2, 3)); xrlow_b = (xr <= -0.98).mean(axis=(0, 2, 3))
for c in list(range(0, 4)) + [10, 11, 12, 30] + list(range(55, 64)):
    print(f'band {c:2d} sd {sd_b[c]:.3f} err {er_b[c]:.3f} clamp {clamp[c]:.4f} (low {lo[c]:.4f}) truth_mean {xm_b[c]:.3f} x<=-0.98 {xlow_b[c]:.3f} xr<=-0.98 {xrlow_b[c]:.3f} vh {vh_b[c]:.3f}')
print('corr over bands: sd vs clamp', np.corrcoef(sd_b, clamp)[0, 1], ' sd vs truth_mean', np.corrcoef(sd_b, xm_b)[0, 1])
# raw (unclamped-proxy) residual spread: sd of r_k = (s - x_r)/sigma_r only on elements with no clamped draw
noclamp = (np.abs(S) < 0.9995).all(0)
sd_nc_b = np.array([100 * np.sqrt((sd[:, c][noclamp[:, c]] ** 2).mean()) for c in range(64)])
print('band sd on clamp-free elements, bands 0,1,2,11,56,57,62,63:', np.round(sd_nc_b[[0, 1, 2, 11, 56, 57, 62, 63]], 3), ' bands 10,12,30:', np.round(sd_nc_b[[10, 12, 30]], 3))
print('clamp-free fraction per band 0,1,2,11,56,57,62,63:', np.round(noclamp.mean(axis=(0, 2, 3))[[0, 1, 2, 11, 56, 57, 62, 63]], 3))
# --- CRPS comparisons on 8 cubes (on [0,1], %)
def crps_s(s, t):
    s = np.sort(s, axis=0); Kk = s.shape[0]
    w = 2.0 * np.arange(Kk) - Kk + 1.0
    return np.abs(s - t[None]).mean(0) - np.tensordot(w, s, axes=(0, 0)) / (Kk * (Kk - 1))
def crps_g(m, sg, t):
    zz = (t - m) / sg
    return sg * (zz * (2 * norm.cdf(zz) - 1) + 2 * norm.pdf(zz) - 1 / math.sqrt(math.pi))
print('\n== per-cube CRPS (%): record diff, stored diff, record vh, recomputed vh, N(x_r,(0.5 sigma_r)^2), N(mu32, vh), N(mu16a, vh), N(x_r, vh)')
acc = []
for i in range(S.shape[1]):
    t = x01[i]
    cd = 100 * crps_s(s01[:, i], t).mean()
    cvh = 100 * crps_g(xr01[i], sig_vh01[i], t).mean()
    ch = 100 * crps_g(xr01[i], 0.5 * sigma_r / 2, t).mean()
    c32 = 100 * crps_g(mu[i], sig_vh01[i], t).mean()
    c16 = 100 * crps_g(s01[:16, i].mean(0), sig_vh01[i], t).mean()
    c16b = 100 * crps_g(s01[16:, i].mean(0), sig_vh01[i], t).mean()
    c8 = 100 * crps_g(s01[:8, i].mean(0), sig_vh01[i], t).mean()
    acc.append((rec['per_cube']['diffusion']['crps'][idx[i]], cd, rec['per_cube']['varhead']['crps'][idx[i]], cvh, ch, c32, 0.5 * (c16 + c16b), c8))
acc = np.array(acc)
print(np.round(acc, 4))
print('means', np.round(acc.mean(0), 4))
print('cost of using mu_K instead of x_r under the vh sigma: K=8 %.4f, K=16 %.4f, K=32 %.4f' % (acc[:, 7].mean() - acc[:, 3].mean(), acc[:, 6].mean() - acc[:, 3].mean(), acc[:, 5].mean() - acc[:, 3].mean()))
