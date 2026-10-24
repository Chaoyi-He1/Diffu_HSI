"""S1 investigation on the stored figure cubes: mean residual of the 32 samples around x_r, clamp effects, and the
finite-K noise floor."""
import json
import numpy as np

R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
EV = R + '/uq/d8/' + 'ev' + 'al/'
rec = json.load(open(EV + 'd8_main.json'))
z = np.load(EV + 'd8_main__cubes.npz')
print({k: (z[k].shape, z[k].dtype) for k in z.files})
S = z['samples'].astype(np.float64)       # [K, B, C, H, W]
x = z['x'].astype(np.float64)
xr = z['x_r'].astype(np.float64)
xh = z['x0_hat'].astype(np.float64)
files = [str(f) for f in z['files']]
K, B = S.shape[:2]
sigma_r = rec['sigma_r']
ys = rec['yswap']
idx = [ys['files'].index(f) for f in files]
print('stored files == record stored_files:', files == rec['stored_files'], 'yswap idx', idx)
print('stored files in gate set:', all(f in set(rec['files']) for f in files))

def rmse_pct(a, b):  # [-1,1] scale -> % of [0,1] range
    return 100 * np.sqrt((((a - b) / 2) ** 2).mean())

m = S.mean(0)                               # [B, C, H, W]
print('\n-- per cube (RMSE %, [0,1] scale) --')
rows = []
for i in range(B):
    r_mk, r_r, r_b = rmse_pct(m[i], x[i]), rmse_pct(xr[i], x[i]), rmse_pct(xh[i], x[i])
    var = S[:, i].var(0, ddof=1)               # per element sample variance, [-1,1] scale
    # expected extra MSE of the mean of K draws around its own expectation: var / K (per element)
    mse_r = (((xr[i] - x[i]) / 2) ** 2).mean()
    extra_mse_noise = (var / 4).mean() / K
    pred_drms = 100 * (np.sqrt(mse_r + extra_mse_noise) - np.sqrt(mse_r))
    dvec = m[i] - xr[i]
    ms_d = ((dvec / 2) ** 2).mean()               # observed mean square of the drift
    cross = 2 * ((dvec / 2) * ((xr[i] - x[i]) / 2)).mean()
    rows.append(dict(i=i, file=files[i], rmse_meanK=r_mk, rmse_meanK_rec=ys['true'][idx[i]], rmse_r=r_r,
                     rmse_r_rec=ys['regressor'][idx[i]], d_obs=r_mk - r_r, d_pred_noise=pred_drms,
                     ms_drift=ms_d, ms_noise=extra_mse_noise, ratio=ms_d / extra_mse_noise, cross=cross,
                     spread_pct=100 * np.sqrt((var / 4).mean()), bias_mean=dvec.mean() / 2 * 100))
for r in rows:
    print(f"{r['i']} {r['file']:40s} meanK {r['rmse_meanK']:.4f} (rec {r['rmse_meanK_rec']:.4f}) reg {r['rmse_r']:.4f} (rec {r['rmse_r_rec']:.4f}) "
          f"dObs {r['d_obs']:+.4f} dPredNoise {r['d_pred_noise']:+.4f} | ms_drift/ms_noise {r['ratio']:.3f} "
          f"cross {r['cross']*1e4:+.3f}e-4 spread {r['spread_pct']:.3f}% mean drift {r['bias_mean']:+.4f} pct")
d_obs = np.array([r['d_obs'] for r in rows]); d_pred = np.array([r['d_pred_noise'] for r in rows])
print(f"mean dObs {d_obs.mean():+.4f} (sd {d_obs.std(ddof=1):.4f}); mean dPredNoise {d_pred.mean():+.4f}; "
      f"mean ratio ms_drift/ms_noise {np.mean([r['ratio'] for r in rows]):.3f}")

# record-wide: yswap block S1-like numbers on 32 cubes
t, rg = np.array(ys['true']), np.array(ys['regressor'])
print(f"\nyswap 32 cubes: mean(meanK - regressor) {np.mean(t - rg):+.4f} se {np.std(t - rg, ddof=1)/np.sqrt(32):.4f}")
# gate: S1 and predicted noise floor from the record's per-cube spread? use record cov metrics: spread_ratio etc
pc = rec['per_cube']
dg = np.array(pc['diffusion']['mean_rmse_pct']) - np.array(pc['regressor']['rmse_pct'])
print(f"gate 200: S1 mean {dg.mean():+.4f}, frac cubes worse {np.mean(dg > 0):.3f}, median {np.median(dg):+.4f}")
dh = np.array(pc['diffusion_halfcentre']['mean_rmse_pct']) - np.array(pc['regressor_halves']['rmse_pct'])
print(f"gate 200 halfcentre: mean(meanK_h - regressor_halves) {dh.mean():+.4f}, frac worse {np.mean(dh > 0):.3f}")

# -- band-dependent / global offset of the drift
D = (m - xr)                                  # [B, C, H, W] on [-1,1]
band_mean = D.mean(axis=(2, 3))                # [B, C]
print('\n-- drift m - x_r on [-1,1] scale --')
print(f"global mean {D.mean():+.5f}  (in [0,1]-% units {D.mean()/2*100:+.4f})  RMS {np.sqrt((D**2).mean()):.5f}")
print('per-cube global mean (pct of [0,1]):', np.round(D.mean(axis=(1, 2, 3)) / 2 * 100, 4).tolist())
bm = band_mean.mean(0) / 2 * 100
bse = band_mean.std(0, ddof=1) / np.sqrt(B) / 2 * 100
print('per-band mean over cubes (pct) min/max', bm.min().round(4), bm.max().round(4), ' |t|>3 bands:', int(np.sum(np.abs(bm / bse) > 3)))
print('per-band means (pct), 8-band blocks:', [round(float(bm[j:j+8].mean()), 4) for j in range(0, 64, 8)])
# noise floor for a per-band, per-cube mean of 4096 pixels: sd_k/sqrt(32*4096) -- ignore spatial correlation
sdk = S.std(0, ddof=1)
print('expected |per-cube band mean| from sampling noise only (pct, iid pixels):',
      round(float((sdk.mean() / np.sqrt(K * 4096)) / 2 * 100), 5))
# the fraction of the drift energy that is the per-cube per-band offset
off = band_mean[:, :, None, None]
print(f"share of drift mean-square in per-cube per-band offsets: {float((off**2).mean() / (D**2).mean()):.4f}")

# -- clamp
at_lo = (S <= -1 + 1e-6); at_hi = (S >= 1 - 1e-6)
print('\n-- clamp --')
print(f"fraction of sample elements at -1: {at_lo.mean():.6f}, at +1: {at_hi.mean():.6f}")
print(f"fraction of x_r elements at -1: {(xr <= -1 + 1e-6).mean():.6f}; x at -1: {(x <= -1 + 1e-6).mean():.6f}; x range [{x.min():.3f}, {x.max():.3f}]; x_r range [{xr.min():.3f}, {xr.max():.3f}]")
anyclamp = (at_lo | at_hi).any(0)
print(f"elements with any of the 32 samples clamped: {anyclamp.mean():.6f}")
print(f"drift mean on clamp-free elements (pct): {D[~anyclamp].mean()/2*100:+.5f}; on clamped elements: "
      f"{(D[anyclamp].mean()/2*100) if anyclamp.any() else float('nan'):+.5f}")
# meanK RMSE on clamp-free elements vs regressor
for nm, sel in (('clamp-free', ~anyclamp), ('clamped', anyclamp)):
    if sel.any():
        e_m = (((m - x) / 2) ** 2)[sel].mean(); e_r = (((xr - x) / 2) ** 2)[sel].mean()
        print(f"  {nm}: MSE meanK {e_m*1e4:.4f}e-4 vs regressor {e_r*1e4:.4f}e-4 (n {int(sel.sum())})")
# contribution of clamped elements to the total MSE difference
tot = ((((m - x) / 2) ** 2) - (((xr - x) / 2) ** 2))
print(f"  total MSE diff {tot.mean()*1e6:.4f}e-6; from clamped elements {tot[anyclamp].sum()/tot.size*1e6 if anyclamp.any() else 0:.4f}e-6")

# -- decomposition of the MSE increase: E||m-x||^2 - ||xr-x||^2 = ||m-xr||^2 + 2<m-xr, xr-x>
msd = ((D / 2) ** 2).mean(); cr = 2 * ((D / 2) * ((xr - x) / 2)).mean(); noise = ((sdk ** 2) / 4).mean() / K
print('\n-- MSE decomposition over the 8 cubes ([0,1] scale) --')
print(f"||m - x_r||^2 {msd:.4e}; of which expected finite-K noise var/K {noise:.4e} (ratio {msd/noise:.3f}); cross 2<m-x_r, x_r-x> {cr:+.4e}")
# split drift into smooth/low-pass and noise: compare drift to residual direction
corr = np.corrcoef(D.ravel(), (x - xr).ravel())[0, 1]
print(f"corr(m - x_r, x - x_r) = {corr:+.4f}")
# per-sample residual statistics: r_k = (x_k - x_r)/sigma_r
rk = (S - xr[None]) / sigma_r
print(f"standardised sample residual: mean {rk.mean():+.5f}, std {rk.std():.4f}; truth (x - x_r)/sigma_r: mean {((x - xr)/sigma_r).mean():+.5f}, std {((x - xr)/sigma_r).std():.4f}")
# split the 32 samples into two halves of 16 to test the finite-K explanation directly: if drift were pure noise,
# the two half-means' drifts would be uncorrelated
m1, m2 = S[:16].mean(0), S[16:].mean(0)
c12 = np.corrcoef((m1 - xr).ravel(), (m2 - xr).ravel())[0, 1]
print(f"corr(drift of samples 0-15, drift of samples 16-31) = {c12:+.4f}  (0 if the drift were finite-K noise)")
# variance-based estimate of the systematic part: E[(m1-xr)(m2-xr)] = ||bias||^2
sys_ms = (((m1 - xr) / 2) * ((m2 - xr) / 2)).mean()
print(f"systematic drift mean square (cross-moment of halves) {sys_ms:.4e} vs total ||m-x_r||^2 {msd:.4e}")
# low-frequency check: drift averaged over 8x8 blocks (the sensor grid) and over bands
Db = D.reshape(B, 64, 8, 8, 8, 8).mean(axis=(3, 5))
print(f"drift RMS after 8x8 block mean {np.sqrt((Db**2).mean()):.5f} vs pixel RMS {np.sqrt((D**2).mean()):.5f}; noise-only expectation for block mean {np.sqrt(((sdk**2).mean())/K/64):.5f}")
Ds = D.mean(1)  # spectral mean per pixel
print(f"drift RMS after spectral mean {np.sqrt((Ds**2).mean()):.5f}; noise-only (bands independent) {np.sqrt(((sdk**2).mean())/K/64):.5f}")
json.dump(rows, open('/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d8/provenance/p4_rows.json', 'w'), indent=1, default=float)
