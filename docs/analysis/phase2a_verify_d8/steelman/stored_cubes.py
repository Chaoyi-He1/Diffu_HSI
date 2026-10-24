"""Element-level diagnostics on the 8 stored y-swap cubes (32 DDIM-10 samples each, true condition):
per-cube spread scaling, within-cube localisation, grid/edge structure, spatial and spectral structure of the samples,
PIT histogram, CRPS decomposition (sample vs Gaussian fits, oracle per-cube rescales), K-invariance of the fair CRPS.
CPU only; reads results, writes only into the scratch dir."""
import json, math, os, sys, time
import numpy as np
import torch

torch.set_num_threads(6)
REPO = '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan'
sys.path.insert(0, REPO)
from data_loader.warmstart_paths import DATA_ROOT, PRIOR
from data_loader.hfd_residual import residual_collate_fn
from model.residual_wrapper import build_residual_model, varhead_forward, LOGVAR_RANGE
from scripts.eval_warmstart import dataset_for
from scripts.uq import metrics as M
from scipy.stats import spearmanr, kurtosis

OUT = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d8/steelman/'
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
z = np.load(R + '/uq/d8/' + 'ev' + 'al/d8_main__cubes.npz')
S = z['samples'].astype(np.float64)          # [K, B, C, H, W] on [-1, 1]
X, XR, XH, files = z['x'].astype(np.float64), z['x_r'].astype(np.float64), z['x0_hat'].astype(np.float64), list(z['files'])
K, B = S.shape[:2]
print('samples', S.shape, 'files', files)
sigma_r = 0.08934239400551727

# --- varhead sigma for these cubes (needs y)
ds = dataset_for(DATA_ROOT, PRIOR, 8, [os.path.join(DATA_ROOT, f) for f in files])
order = [os.path.relpath(f, DATA_ROOT) for f in ds.img_list]
idx = [order.index(f) for f in files]
x_t, y_t, xh_t = residual_collate_fn([ds[i] for i in idx])
print('max |x - stored x|', float((x_t.double() - torch.from_numpy(X)).abs().max()))
ck = torch.load(R + '/runs/varhead_d8_b64/checkpoint_epoch_10.pth', map_location='cpu', weights_only=False)
vh = build_residual_model('varhead', int(ck['meta']['base_channels']))
vh.load_state_dict(ck['model_state_dict'], strict=True); vh.eval()
t0 = time.time()
with torch.no_grad():
    lv = varhead_forward(vh, {'sensor': y_t, 'x0_hat': torch.from_numpy(XR).float()}).clamp(*LOGVAR_RANGE)
print(f'varhead forward on CPU, batch {B}: {time.time() - t0:.1f} s')
SVH = (sigma_r * torch.exp(0.5 * lv)).double().numpy()     # [-1,1] scale std
np.save(OUT + 'varhead_sigma_stored.npy', SVH.astype(np.float32))

# all on [0,1]
s01, x01, xr01, sv01 = (S + 1) / 2, (X + 1) / 2, (XR + 1) / 2, SVH / 2
mu = s01.mean(0); sd = s01.std(0, ddof=1)
e = xr01 - x01                     # regressor error
res_true = x01 - xr01              # true residual
dev = s01 - mu[None]               # sample deviations about the sample mean
res_samp = s01 - xr01[None]        # sample residuals about the centre

H = W = 64
grid = np.zeros((H, W), bool); grid[::8, ::8] = True
edge = np.zeros((H, W), bool); edge[57:, :] = True; edge[:, 57:] = True
inner = ~grid & ~edge


def rms(a, m=None):
    a = a if m is None else a[..., m]
    return float(np.sqrt(np.mean(a ** 2)))


def lag1(a):
    """mean lag-1 spatial autocorrelation over (band, axis) of a [C, H, W] field (mean-removed per band)."""
    a = a - a.mean(axis=(-2, -1), keepdims=True)
    num = (a[..., 1:, :] * a[..., :-1, :]).mean() + (a[..., :, 1:] * a[..., :, :-1]).mean()
    return float(num / 2 / (a ** 2).mean())


def band_corr(a):
    """mean correlation between adjacent bands, and between band 0 and band 63, over pixels."""
    a = a.reshape(a.shape[0], -1)
    a = a - a.mean(1, keepdims=True)
    n = np.sqrt((a ** 2).sum(1))
    c = (a[1:] * a[:-1]).sum(1) / (n[1:] * n[:-1] + 1e-30)
    return float(c.mean()), float((a[0] * a[-1]).sum() / (n[0] * n[-1] + 1e-30))


def top3_energy(a):
    a = a.reshape(a.shape[0], -1)            # [C, P]
    sv = np.linalg.svd(a, compute_uv=False)
    return float((sv[:3] ** 2).sum() / (sv ** 2).sum())


rows = []
for i in range(B):
    r = dict(file=files[i])
    r['rmse_R'] = 100 * rms(e[i]); r['rms_sd'] = 100 * rms(sd[i]); r['rms_vh'] = 100 * rms(sv01[i])
    r['rmse_meanK'] = 100 * rms(mu[i] - x01[i])
    # localisation
    r['sp_sd_err'] = spearmanr(sd[i].ravel(), np.abs(e[i]).ravel())[0]
    r['sp_vh_err'] = spearmanr(sv01[i].ravel(), np.abs(e[i]).ravel())[0]
    r['sp_sd_vh'] = spearmanr(sd[i].ravel(), sv01[i].ravel())[0]
    # block-level (8x8 pixels x all bands) RMS, removes K-noise in sd
    blk = lambda a: np.sqrt((a.reshape(64, 8, 8, 8, 8) ** 2).mean(axis=(0, 2, 4)))
    r['sp_blk_sd_err'] = spearmanr(blk(sd[i]).ravel(), blk(e[i]).ravel())[0]
    r['sp_blk_vh_err'] = spearmanr(blk(sv01[i]).ravel(), blk(e[i]).ravel())[0]
    # grid / edge
    for nm, m in (('grid', grid), ('edge', edge), ('inner', inner)):
        r[f'err_{nm}'] = 100 * rms(e[i], m); r[f'sd_{nm}'] = 100 * rms(sd[i], m); r[f'vh_{nm}'] = 100 * rms(sv01[i], m)
    # structure
    r['lag1_true'] = lag1(res_true[i]); r['lag1_dev'] = float(np.mean([lag1(dev[k, i]) for k in range(4)]))
    r['bandc_true'], r['band0_63_true'] = band_corr(res_true[i])
    bc = [band_corr(dev[k, i]) for k in range(4)]
    r['bandc_dev'], r['band0_63_dev'] = float(np.mean([b[0] for b in bc])), float(np.mean([b[1] for b in bc]))
    r['top3_true'] = top3_energy(res_true[i]); r['top3_dev'] = float(np.mean([top3_energy(dev[k, i]) for k in range(4)]))
    # Laplacian energy relative to variance: white noise = 20
    r['lapvar_true'] = float((M.laplacian(res_true[i]) ** 2).mean() / (res_true[i] ** 2).mean())
    r['lapvar_dev'] = float(np.mean([(M.laplacian(dev[k, i]) ** 2).mean() / (dev[k, i] ** 2).mean() for k in range(4)]))
    # CRPS decomposition (%)
    c = lambda m_, s_: 100 * float(M.crps_gaussian(m_, s_, x01[i]).mean())
    r['crps_samples'] = 100 * float(M.crps_samples(s01[:, i], x01[i]).mean())
    r['crps_samples_K16'] = 100 * float(M.crps_samples(s01[:16, i], x01[i]).mean())
    r['crps_samples_K16b'] = 100 * float(M.crps_samples(s01[16:, i], x01[i]).mean())
    r['crps_g_mu_sd'] = c(mu[i], sd[i])
    r['crps_g_xr_sd'] = c(xr01[i], sd[i])
    r['crps_g_xr_vh'] = c(xr01[i], sv01[i])
    cs = rms(e[i]) / rms(sd[i]); cv = rms(e[i]) / rms(sv01[i])
    r['crps_g_xr_sd_oraclecube'] = c(xr01[i], cs * sd[i])
    r['crps_g_xr_vh_oraclecube'] = c(xr01[i], cv * sv01[i])
    r['crps_g_xr_const_oraclecube'] = c(xr01[i], np.full_like(sd[i], rms(e[i])))
    r['crps_samples_oraclecube'] = 100 * float(M.crps_samples(xr01[i][None] + cs * (s01[:, i] - xr01[i][None]), x01[i]).mean())
    # samples' spread replaced by the varhead's geometry but keeping sample shape: standardise sample deviations by sd, rescale by vh
    shape_vh = mu[i][None] * 0 + xr01[i][None] + (dev[:, i] / np.maximum(sd[i], 1e-9)[None]) * sv01[i][None] / math.sqrt(1.0)
    r['crps_samples_reshaped_to_vh'] = 100 * float(M.crps_samples(shape_vh, x01[i]).mean())
    # tails: standardised errors
    r['kurt_e_over_vh'] = float(kurtosis((e[i] / sv01[i]).ravel(), fisher=False))
    r['kurt_e_over_sd'] = float(kurtosis((e[i] / sd[i]).ravel(), fisher=False))
    r['kurt_dev_over_sd'] = float(kurtosis((dev[0, i] / sd[i]).ravel(), fisher=False))
    r['kurt_res_true'] = float(kurtosis(res_true[i].ravel(), fisher=False))
    r['kurt_res_samp'] = float(kurtosis(res_samp[0, i].ravel(), fisher=False))
    # coverage / AUSE
    cvs = M.coverage_samples(s01[:, i], x01[i]); r['cov50'] = cvs[0.5]; r['cov90'] = cvs[0.9]
    cvv = M.coverage_gaussian(xr01[i], sv01[i], x01[i]); r['cov50_vh'] = cvv[0.5]; r['cov90_vh'] = cvv[0.9]
    cvo = M.coverage_gaussian(xr01[i], cs * sd[i], x01[i]); r['cov50_sd_oraclecube'] = cvo[0.5]; r['cov90_sd_oraclecube'] = cvo[0.9]
    r['ause_sd'] = M.ause(sd[i], mu[i] - x01[i]); r['ause_vh'] = M.ause(sv01[i], e[i])
    r['ause_sd_vs_xr'] = M.ause(sd[i], e[i])
    rows.append(r)

keys = list(rows[0].keys())[1:]
print(f"{'metric':32s} " + ' '.join(f'{f.split("/")[-2][-2:]}/{f.split("/")[-1][:-4]:>6s}' for f in files) + '     mean')
for k in keys:
    v = [r[k] for r in rows]
    print(f'{k:32s} ' + ' '.join(f'{x:9.4f}' for x in v) + f'  {np.mean(v):8.4f}')
print('corr across 8 cubes: log rms_sd vs log rmse_R', np.corrcoef(np.log([r['rms_sd'] for r in rows]), np.log([r['rmse_R'] for r in rows]))[0, 1],
      ' slope', np.polyfit(np.log([r['rmse_R'] for r in rows]), np.log([r['rms_sd'] for r in rows]), 1)[0])
print('corr across 8 cubes: log rms_vh vs log rmse_R', np.corrcoef(np.log([r['rms_vh'] for r in rows]), np.log([r['rmse_R'] for r in rows]))[0, 1],
      ' slope', np.polyfit(np.log([r['rmse_R'] for r in rows]), np.log([r['rms_vh'] for r in rows]), 1)[0])

# PIT histogram of the truth's rank among the 32 samples (pooled)
rank = (s01 < x01[None]).sum(0)            # 0..K
hist = np.bincount(rank.ravel(), minlength=K + 1) / rank.size
print('PIT (rank of truth among 32 samples), 11 bins of 3 ranks:', np.round([hist[j:j + 3].sum() for j in range(0, 33, 3)], 4))
print('expected per bin of 3 ranks', round(3 / 33, 4))
json.dump(rows, open(OUT + 'stored_cubes_rows.json', 'w'), indent=1)
