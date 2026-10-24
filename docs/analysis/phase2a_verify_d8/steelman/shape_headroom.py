"""How much CRPS could a predictive with the variance head's localisation but a better shape gain over the Gaussian
variance head?  Closed-form CRPS for Gaussian and Laplace predictives with scale c * sigma_vh (global c), on the 8
stored cubes; plus a check that the 'diffusion shape reshaped to sigma_vh' gain is not a fair-CRPS artefact."""
import math, sys
import numpy as np
from scipy.special import erf
sys.path.insert(0, '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan')
from scripts.uq import metrics as M

OUT = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d8/steelman/'
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
z = np.load(R + '/uq/d8/' + 'ev' + 'al/d8_main__cubes.npz')
x01, xr01 = (z['x'].astype(np.float64) + 1) / 2, (z['x_r'].astype(np.float64) + 1) / 2
sv01 = np.load(OUT + 'varhead_sigma_stored.npy').astype(np.float64) / 2
S = (z['samples'].astype(np.float64) + 1) / 2
e = x01 - xr01


def crps_laplace(mu, b, x):
    # CRPS of Laplace(mu, b) at x: b * ( |z| + exp(-|z|) - 3/4 ), z = (x - mu)/b
    zz = np.abs(x - mu) / b
    return b * (zz + np.exp(-zz) - 0.75)


def crps_gauss(mu, s, x):
    return M.crps_gaussian(mu, s, x)


base = 100 * crps_gauss(xr01, sv01, x01).mean()
print(f'Gaussian varhead (as evaluated) CRPS {base:.4f}')
for c in (0.6, 0.7, 0.8, 0.9, 1.0):
    print(f'  Gaussian c={c:.1f}: {100 * crps_gauss(xr01, c * sv01, x01).mean():.4f}')
for c in (0.4, 0.5, 0.6, 0.7071, 0.8):
    print(f'  Laplace  b = {c:.3f} sigma_vh: {100 * crps_laplace(xr01, c * sv01, x01).mean():.4f}   (0.7071 = variance-matched)')

# per-cube Laplace (variance matched) vs Gaussian
for i in range(8):
    g = 100 * crps_gauss(xr01[i], sv01[i], x01[i]).mean(); l = 100 * crps_laplace(xr01[i], sv01[i] / math.sqrt(2), x01[i]).mean()
    print(f'  cube {i}: gauss {g:.4f}  laplace(var-matched) {l:.4f}  rel {l / g - 1:+.3%}')

# artefact check: iid Gaussian samples normalised per element to unit sample std, rescaled to sigma_vh, fair CRPS
rng = np.random.default_rng(0)
K = 32
tot_g, tot_n = [], []
for i in range(8):
    g = rng.standard_normal((K,) + x01[i].shape)
    gn = (g - g.mean(0)) / g.std(0, ddof=1)
    tot_n.append(100 * M.crps_samples(xr01[i][None] + gn * sv01[i][None], x01[i]).mean())
    tot_g.append(100 * M.crps_samples(xr01[i][None] + g * sv01[i][None], x01[i]).mean())
print(f'iid Gaussian samples at sigma_vh: fair CRPS {np.mean(tot_g):.4f};  per-element normalised: {np.mean(tot_n):.4f}  (closed form {base:.4f})')
# diffusion deviations standardised by their own sd, rescaled to sigma_vh (computed in stored_cubes.py as 1.8205)
mu, sd = S.mean(0), S.std(0, ddof=1)
v = []
for i in range(8):
    dv = (S[:, i] - mu[i][None]) / np.maximum(sd[i], 1e-9)[None]
    v.append(100 * M.crps_samples(xr01[i][None] + dv * sv01[i][None], x01[i]).mean())
print(f'diffusion-shape samples rescaled to sigma_vh: {np.mean(v):.4f}')
# hybrid without normalisation artefact: varhead sigma times the diffusion's sd ratio to its cube mean (keeps diffusion's within-cube pattern)
