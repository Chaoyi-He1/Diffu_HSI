"""P1 lens: is a single diffusion sample's high-frequency content realistic texture or HF noise?
All on [0, 1]. Stored cubes: samples [K=32, B=8, C=64, H, W] on [-1, 1]."""
import os, sys, json, math
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np
REPO = '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan'
sys.path.insert(0, REPO)
from scripts.uq import metrics as M
RR = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
EV = os.path.join(RR, 'uq/d8', 'ev' + 'al')
rec = json.load(open(os.path.join(EV, 'd8_main.json')))
z = np.load(os.path.join(EV, 'd8_main__cubes.npz'))
S = (z['samples'].astype(np.float64) + 1) / 2          # [K, B, C, H, W]
X = (z['x'].astype(np.float64) + 1) / 2                # [B, C, H, W]
XR = (z['x_r'].astype(np.float64) + 1) / 2
XH = (z['x0_hat'].astype(np.float64) + 1) / 2
K, B, C, H, W = S.shape
MU = S.mean(0)
files = list(z['files'])
gi = [rec['files'].index(f) for f in files]
print('K B C H W', S.shape)

# ---------------------------------------------------------------- 0. reproduce record metrics on the 8 cubes
def rmse(a, b): return 100 * math.sqrt(((a - b) ** 2).mean())
print('\n== 0. per-cube hf / w1 / rmse: recomputed from stored cubes vs record (record single/mean are from a different noise draw / batch)')
rows = []
for b in range(B):
    r = dict(f=files[b],
             hf_reg=M.hf_energy_ratio(XR[b], X[b]), hf_reg_rec=rec['per_cube']['regressor']['hf'][gi[b]],
             hf_s0=M.hf_energy_ratio(S[0, b], X[b]), hf_s0_rec=rec['per_cube']['diffusion']['single_hf'][gi[b]],
             hf_mu=M.hf_energy_ratio(MU[b], X[b]), hf_mu_rec=rec['per_cube']['diffusion']['mean_hf'][gi[b]],
             w1_reg=M.grad_w1(XR[b], X[b]), w1_reg_rec=rec['per_cube']['regressor']['w1'][gi[b]],
             w1_s0=M.grad_w1(S[0, b], X[b]), w1_s0_rec=rec['per_cube']['diffusion']['single_w1'][gi[b]],
             w1_mu=M.grad_w1(MU[b], X[b]),
             rmse_reg=rmse(XR[b], X[b]), rmse_s0=rmse(S[0, b], X[b]), rmse_mu=rmse(MU[b], X[b]))
    rows.append(r)
    print(' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in r.items()))
mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'f'}
print('MEAN over 8:', {k: round(v, 4) for k, v in mean.items()})
# all 32 samples' hf
hf_all = np.array([[M.hf_energy_ratio(S[k, b], X[b]) for b in range(B)] for k in range(K)])
print('hf over all 32 samples: mean %.3f, sd across samples (avg per cube) %.3f, min %.3f max %.3f' % (hf_all.mean(), hf_all.std(0).mean(), hf_all.min(), hf_all.max()))
w1_all = np.array([[M.grad_w1(S[k, b], X[b]) for b in range(B)] for k in range(K)])
print('w1 over all 32 samples: mean %.5f (regressor %.5f)' % (w1_all.mean(), mean['w1_reg']))

# ---------------------------------------------------------------- 1. radially averaged power spectrum
hann = np.outer(np.hanning(H), np.hanning(W))
ky, kx = np.meshgrid(np.fft.fftfreq(H) * H, np.fft.fftfreq(W) * W, indexing='ij')
rad = np.sqrt(kx ** 2 + ky ** 2)
rbin = np.rint(rad).astype(int)
NB = 33                                                     # radii 0..32 (cycles per 64 px); corners > 32 dropped
mask = rbin < NB
def F(u):
    u = u - u.mean(axis=(-2, -1), keepdims=True)
    return np.fft.fft2(u * hann)
def radial(P):                                              # P [..., H, W] -> [NB] averaged over leading dims
    P = P.reshape(-1, H, W).mean(0)
    return np.array([P[mask & (rbin == r)].mean() for r in range(NB)])
def psd(u): return radial(np.abs(F(u)) ** 2)
P_x, P_xr, P_xh = psd(X), psd(XR), psd(XH)
P_s0 = psd(S[0]); P_mu = psd(MU)
P_sall = np.mean([psd(S[k]) for k in range(K)], 0)
# residual spectra
FX_res = F(X - XR)
P_res_true = radial(np.abs(FX_res) ** 2)
P_res_samp = np.zeros(NB); P_err_samp = np.zeros(NB); cross = np.zeros(NB)
for k in range(K):
    Fs = F(S[k] - XR)
    P_res_samp += radial(np.abs(Fs) ** 2) / K
    P_err_samp += radial(np.abs(F(S[k] - X)) ** 2) / K
    cross += radial((Fs * np.conj(FX_res)).real) / K
coh = cross / np.sqrt(P_res_samp * P_res_true)
P_err_reg = radial(np.abs(FX_res) ** 2)                      # error of regressor = residual
P_err_mu = psd(MU - X)
print('\n== 1. radially averaged PSD (band- and cube-averaged, Hann window), ratios to truth; r in cycles per 64 px')
print(' r | P_x       | xr/x    s0/x    sAll/x  mu/x    | true_res  samp_res/true_res  coh(samp_res,true_res) | err_s/err_reg')
for r in range(1, NB):
    print(f'{r:2d} | {P_x[r]:.3e} | {P_xr[r]/P_x[r]:7.3f} {P_s0[r]/P_x[r]:7.3f} {P_sall[r]/P_x[r]:7.3f} {P_mu[r]/P_x[r]:7.3f} | {P_res_true[r]:.3e} {P_res_samp[r]/P_res_true[r]:9.3f} {coh[r]:+8.3f} | {P_err_samp[r]/P_err_reg[r]:7.3f}')
bands = {'low 1-4': range(1, 5), 'mid 5-12': range(5, 13), 'high 13-24': range(13, 25), 'top 25-32': range(25, 33)}
print('\nbanded energy ratios (sum of radial-bin power, unweighted by annulus area)')
for name, rr in bands.items():
    rr = list(rr)
    sx = P_x[rr].sum()
    print(f'  {name:11s} xr/x {P_xr[rr].sum()/sx:.3f}  s/x {P_sall[rr].sum()/sx:.3f}  mu/x {P_mu[rr].sum()/sx:.3f}  samp_res/true_res {P_res_samp[rr].sum()/P_res_true[rr].sum():.3f}  coh {cross[rr].sum()/math.sqrt(P_res_samp[rr].sum()*P_res_true[rr].sum()):+.3f}')

# grid artefacts at multiples of 64/8 = 8 along the axes (stride-8 sensor)
Pres2d = np.zeros((H, W))
for k in range(K): Pres2d += (np.abs(F(S[k] - XR)) ** 2).reshape(-1, H, W).mean(0) / K
Ptrue2d = (np.abs(FX_res) ** 2).reshape(-1, H, W).mean(0)
print('\ngrid check: sample-residual power at axis frequencies (kx=8m,ky=0) vs neighbours, and same for true residual')
for m in (1, 2, 3, 4):
    f = 8 * m if 8 * m < 32 else 32
    on = Pres2d[0, f % W]; nb = 0.5 * (Pres2d[0, (f - 1) % W] + Pres2d[0, (f + 1) % W])
    on_t = Ptrue2d[0, f % W]; nb_t = 0.5 * (Ptrue2d[0, (f - 1) % W] + Ptrue2d[0, (f + 1) % W])
    print(f'  kx={f:2d}: sample on/nb {on/nb:6.2f}   true on/nb {on_t/nb_t:6.2f}')

# ---------------------------------------------------------------- 2. noise controls: x_r + iid Gaussian noise
rng = np.random.default_rng(0)
SD = S.std(0, ddof=1)
print('\n== 2. noise controls (no learned texture): x_r + iid N(0, sd^2)')
def ctrl_stats(name, Y):
    hf = np.mean([M.hf_energy_ratio(Y[b], X[b]) for b in range(B)])
    ahf = np.mean([abs(M.hf_energy_ratio(Y[b], X[b])) for b in range(B)])
    w1 = np.mean([M.grad_w1(Y[b], X[b]) for b in range(B)])
    rm = np.mean([rmse(Y[b], X[b]) for b in range(B)])
    print(f'  {name:48s} hf {hf:+.3f} |hf| {ahf:.3f} w1 {w1:.5f} rmse {rm:.3f}')
    return ahf
ctrl_stats('regressor x_r', XR)
ctrl_stats('single sample s0', S[0])
ctrl_stats('x_r + N(0, sd_diffusion^2) per element (iid)', np.clip(XR + SD * rng.standard_normal(XR.shape), 0, 1))
# per-cube scalar noise sigma that makes r_HF = 0 (bisection)
def hf_noise(b, s, seed=1):
    g = np.random.default_rng(seed).standard_normal(XR[b].shape)
    return M.hf_energy_ratio(np.clip(XR[b] + s * g, 0, 1), X[b])
Ymatch = np.empty_like(XR); smatch = []
for b in range(B):
    lo, hi = 0.0, 0.2
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        (lo, hi) = (mid, hi) if hf_noise(b, mid) < 0 else (lo, mid)
    smatch.append(mid)
    Ymatch[b] = np.clip(XR[b] + mid * np.random.default_rng(1).standard_normal(XR[b].shape), 0, 1)
print('  per-cube white-noise sigma giving r_HF = 0 (on [0,1]):', np.round(smatch, 4), ' vs mean sample sd', np.round(SD.mean(axis=(1, 2, 3)), 4))
ctrl_stats('x_r + white noise tuned to r_HF = 0', Ymatch)
ctrl_stats('baseline x0_hat', XH)

# ---------------------------------------------------------------- 3. structure of the sampled residual vs the true residual
print('\n== 3. structure of residuals: sampled (s_k - x_r) vs true (x - x_r)')
def lag1_spatial(u):                                         # u [..., H, W]; mean corr of horizontally/vertically adjacent pixels
    u = u - u.mean(axis=(-2, -1), keepdims=True)
    a = (u[..., :, 1:] * u[..., :, :-1]).mean() / (u ** 2).mean()
    b_ = (u[..., 1:, :] * u[..., :-1, :]).mean() / (u ** 2).mean()
    return 0.5 * (a + b_)
def lag1_spectral(u):                                        # u [B, C, H, W]; corr of adjacent bands
    u = u - u.mean(axis=(-2, -1), keepdims=True)
    return float((u[:, 1:] * u[:, :-1]).mean() / (u ** 2).mean())
rt = X - XR
rs = S[0] - XR
print(f'  spatial lag-1 autocorr: true res {lag1_spatial(rt):.3f}  sample res {lag1_spatial(rs):.3f}  truth x {lag1_spatial(X):.3f}  sample s0 {lag1_spatial(S[0]):.3f}')
print(f'  spectral (adjacent-band) corr: true res {lag1_spectral(rt):.3f}  sample res {lag1_spectral(rs):.3f}')
# spectral PCA of true residual: energy fraction in top-q PCs, for true vs sampled residual
def pc_frac(u_fit, u_eval, q=3):
    A = u_fit.transpose(0, 2, 3, 1).reshape(-1, C); A = A - A.mean(0)
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    E = u_eval.transpose(0, 2, 3, 1).reshape(-1, C); E = E - E.mean(0)
    return float(((E @ Vt[:q].T) ** 2).sum() / (E ** 2).sum())
print(f'  energy fraction in top-3 spectral PCs of the TRUE residual: true res {pc_frac(rt, rt):.3f}  sample res {pc_frac(rt, rs):.3f}  white noise {pc_frac(rt, rng.standard_normal(rt.shape)):.3f}')
print(f'  energy fraction in top-3 spectral PCs of the SAMPLE residual (own PCs): {pc_frac(rs, rs):.3f}')
# spectral roughness: mean squared 2nd difference along bands, relative to x
def spec_rough(u): return float((np.diff(u, 2, axis=1) ** 2).mean())
print(f'  spectral roughness (mean sq 2nd diff over bands): x {spec_rough(X):.3e}  x_r {spec_rough(XR):.3e}  s0 {spec_rough(S[0]):.3e}  mu {spec_rough(MU):.3e}')
# where is the HF energy? correlation of local Laplacian energy maps with truth's
def lmap(u):
    L2 = (M.laplacian(u) ** 2).mean(1)                       # band-mean, [B, H, W]
    # 5x5 box smoothing
    k = 5; p = np.pad(L2, ((0, 0), (2, 2), (2, 2)), mode='reflect'); out = np.zeros_like(L2)
    for i in range(k):
        for j in range(k): out += p[:, i:i + H, j:j + W]
    return out / k / k
Lx, Lxr, Ls0 = lmap(X), lmap(XR), lmap(S[0])
Lres_t, Lres_s = lmap(rt), lmap(rs)
def pc(a, b):
    a, b = np.log(a.ravel() + 1e-12), np.log(b.ravel() + 1e-12)
    return float(np.corrcoef(a, b)[0, 1])
print(f'  corr of log local-HF maps with truth: x_r {pc(Lxr, Lx):.3f}  s0 {pc(Ls0, Lx):.3f}   sampled-res vs true-res {pc(Lres_s, Lres_t):.3f}')
# HF energy ratio by truth's local HF tercile (does the sample put HF where the truth has it?)
q = np.quantile(Lx, [1/3, 2/3])
for name, sel in (('calm', Lx <= q[0]), ('mid', (Lx > q[0]) & (Lx <= q[1])), ('textured', Lx > q[1])):
    print(f'  truth-HF {name:8s}: local HF energy ratio s0/x {Ls0[sel].mean()/Lx[sel].mean():8.2f}   x_r/x {Lxr[sel].mean()/Lx[sel].mean():6.3f}')

# ---------------------------------------------------------------- 4. gradient-magnitude distribution
print('\n== 4. Sobel gradient magnitude: mean and quantiles')
for name, u in (('x', X), ('x_r', XR), ('s0', S[0]), ('mu32', MU), ('noise r_HF=0', Ymatch)):
    g = M.sobel_mag(u)
    print(f'  {name:12s} mean {g.mean():.5f} p50 {np.median(g):.5f} p90 {np.quantile(g, .9):.5f} p99 {np.quantile(g, .99):.5f}')
json.dump(dict(P_x=P_x.tolist(), P_xr=P_xr.tolist(), P_s0=P_s0.tolist(), P_sall=P_sall.tolist(), P_mu=P_mu.tolist(),
               P_res_true=P_res_true.tolist(), P_res_samp=P_res_samp.tolist(), coh=coh.tolist()),
          open('/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d8/metric-validity/psd.json', 'w'))
