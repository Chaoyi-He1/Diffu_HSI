"""C2 lens (coverage by strata) and CRPS apples-to-apples check on the 8 stored cubes. All on [0, 1]."""
import os, sys, json, math
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np
REPO = '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan'
sys.path.insert(0, REPO)
from scripts.uq import metrics as M
SCR = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d8/metric-validity'
RR = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
EV = os.path.join(RR, 'uq/d8', 'ev' + 'al')
rec = json.load(open(os.path.join(EV, 'd8_main.json')))
z = np.load(os.path.join(EV, 'd8_main__cubes.npz'))
vh = np.load(os.path.join(SCR, 'vh_sigma.npz'))
S = (z['samples'].astype(np.float64) + 1) / 2
X = (z['x'].astype(np.float64) + 1) / 2
XR = (z['x_r'].astype(np.float64) + 1) / 2
SIG = vh['sigma'].astype(np.float64) / 2.0                       # varhead std on [0, 1]
K, B, C, H, W = S.shape
files = list(z['files']); gi = [rec['files'].index(f) for f in files]
MU = S.mean(0); SD = S.std(0, ddof=1)

# ---------------------------------------------------------------- per-element interval ends (Weibull, as coverage_samples)
lo50 = np.empty_like(X); hi50 = np.empty_like(X); lo90 = np.empty_like(X); hi90 = np.empty_like(X)
for b in range(B):
    q = np.quantile(S[:, b], [0.25, 0.75, 0.05, 0.95], axis=0, method='weibull')
    lo50[b], hi50[b], lo90[b], hi90[b] = q
c50 = (X >= lo50) & (X <= hi50); c90 = (X >= lo90) & (X <= hi90)
v50 = np.abs(X - XR) <= M.Z[0.5] * SIG; v90 = np.abs(X - XR) <= M.Z[0.9] * SIG
print('== overall on the 8 stored cubes (pooled elements)')
print(f'  diffusion cov50 {c50.mean():.4f} cov90 {c90.mean():.4f}   (record, same 8 cubes, main-loop draw: cov50 {np.mean([rec["per_cube"]["diffusion"]["cov50"][i] for i in gi]):.4f} cov90 {np.mean([rec["per_cube"]["diffusion"]["cov90"][i] for i in gi]):.4f}; 200-cube: 0.6741 / 0.9082)')
print(f'  varhead   cov50 {v50.mean():.4f} cov90 {v90.mean():.4f}   (record, same 8 cubes: cov50 {np.mean([rec["per_cube"]["varhead"]["cov50"][i] for i in gi]):.4f} cov90 {np.mean([rec["per_cube"]["varhead"]["cov90"][i] for i in gi]):.4f})')
# check vs metrics.coverage_samples on one cube
cs = M.coverage_samples(S[:, 0], X[0]); print(f'  metrics.coverage_samples cube 0: {cs}, mine {c50[0].mean():.4f} {c90[0].mean():.4f}')
print(f'  fraction of sample values at the clamp (0 or 1): {np.mean((S <= 0) | (S >= 1)):.2e}; truth at bounds {np.mean((X <= 0) | (X >= 1)):.2e}')

# PIT rank histogram of x among the K samples
rank = (S < X[None]).sum(0)                                     # 0..K
hist = np.bincount(rank.ravel(), minlength=K + 1) / rank.size
print('  PIT rank histogram (uniform = %.4f per rank): ' % (1 / (K + 1)) + ' '.join(f'{h:.3f}' for h in hist))
zz = (X - MU) / SD
print(f'  z = (x - mu)/sd: var {zz.var():.3f}, excess kurtosis {((zz - zz.mean())**4).mean()/zz.var()**2 - 3:.2f}, P(|z|<0.674) {np.mean(np.abs(zz)<0.674):.3f}, P(|z|<1.645) {np.mean(np.abs(zz)<1.645):.3f}')
zv = (X - XR) / SIG
print(f'  varhead z: var {zv.var():.3f}, excess kurtosis {((zv - zv.mean())**4).mean()/zv.var()**2 - 3:.2f}')
# dispersion of the predicted spread
print(f'  sd of log predicted std across elements: diffusion {np.log(SD).std():.3f}  varhead {np.log(SIG).std():.3f}  |x-x_r| (log, +1e-4) {np.log(np.abs(X-XR)+1e-4).std():.3f}')
def spearman(a, b):
    ra = np.argsort(np.argsort(a.ravel())); rb = np.argsort(np.argsort(b.ravel()))
    return float(np.corrcoef(ra, rb)[0, 1])
print(f'  Spearman(pred std, |error|): diffusion (sd vs |x-mu|) {spearman(SD, np.abs(X-MU)):.3f}; diffusion sd vs |x-x_r| {spearman(SD, np.abs(X-XR)):.3f}; varhead {spearman(SIG, np.abs(X-XR)):.3f}')

# ---------------------------------------------------------------- stratified coverage
SOB = M.sobel_mag(XR)
ERR = np.abs(X - XR)
# homoscedastic oracle reference: N(x_r, sigma_cube^2), sigma_cube = RMS(x - x_r) of the cube (one scalar per cube)
sc = np.sqrt(((X - XR) ** 2).mean(axis=(1, 2, 3)))[:, None, None, None] * np.ones_like(X)
h50 = ERR <= M.Z[0.5] * sc; h90 = ERR <= M.Z[0.9] * sc
def strat(name, v, nq=5):
    edges = np.quantile(v, np.linspace(0, 1, nq + 1))
    idx = np.clip(np.searchsorted(edges, v, side='right') - 1, 0, nq - 1)
    print(f'\n== coverage by quintile of {name}')
    print('  q |  range                | diff cov50 cov90 | vh cov50 cov90 | homosc-oracle cov50 cov90 | mean sd_diff  sd_vh  RMS(x-x_r) RMS(x-mu) | RMS(x-mu)/rms sd_diff  RMS(x-x_r)/rms sd_vh')
    out = []
    for qi in range(nq):
        s = idx == qi
        r = dict(lo=float(edges[qi]), hi=float(edges[qi + 1]), d50=float(c50[s].mean()), d90=float(c90[s].mean()), v50=float(v50[s].mean()), v90=float(v90[s].mean()),
                 h50=float(h50[s].mean()), h90=float(h90[s].mean()), sd=float(SD[s].mean()), sv=float(SIG[s].mean()),
                 er=float(np.sqrt((ERR[s] ** 2).mean())), em=float(np.sqrt(((X - MU)[s] ** 2).mean())),
                 ratio_d=float(np.sqrt(((X - MU)[s] ** 2).mean() / (SD[s] ** 2).mean())), ratio_v=float(np.sqrt((ERR[s] ** 2).mean() / (SIG[s] ** 2).mean())))
        out.append(r)
        print(f"  {qi} | {r['lo']:.5f}-{r['hi']:.5f} | {r['d50']:.3f} {r['d90']:.3f} | {r['v50']:.3f} {r['v90']:.3f} | {r['h50']:.3f} {r['h90']:.3f} | {r['sd']:.4f} {r['sv']:.4f} {r['er']:.4f} {r['em']:.4f} | {r['ratio_d']:.3f} {r['ratio_v']:.3f}")
    return out
res = {}
res['err'] = strat('|x - x_r| (outcome-conditioned; compare columns, not with nominal)', ERR)
res['sobel_xr'] = strat('Sobel gradient magnitude of x_r (observable)', SOB)
res['sd_diff'] = strat('diffusion predicted sd (reliability)', SD)
res['sig_vh'] = strat('varhead predicted sigma (reliability)', SIG)
# position in the stride-8 sensor grid
pos = (np.arange(H)[:, None] % 8) * 8 + (np.arange(W)[None, :] % 8)
posb = np.broadcast_to(pos, X.shape)
e_pos = np.array([np.sqrt((ERR[posb == p] ** 2).mean()) for p in range(64)])
s_pos = np.array([SD[posb == p].mean() for p in range(64)])
v_pos = np.array([SIG[posb == p].mean() for p in range(64)])
print(f'\n== sensor-grid phase (64 positions): RMS err range {e_pos.min():.4f}-{e_pos.max():.4f}; diff sd range {s_pos.min():.4f}-{s_pos.max():.4f}; vh sigma range {v_pos.min():.4f}-{v_pos.max():.4f}')
print(f'   corr(err_pos, sd_pos) {np.corrcoef(e_pos, s_pos)[0,1]:.3f}; corr(err_pos, vh_pos) {np.corrcoef(e_pos, v_pos)[0,1]:.3f}')

# ---------------------------------------------------------------- error budget by spatial frequency: true residual vs sampled residual
ky, kx = np.meshgrid(np.fft.fftfreq(H) * H, np.fft.fftfreq(W) * W, indexing='ij')
rad = np.sqrt(kx ** 2 + ky ** 2)
def energy_split(u):                                            # Parseval energy per frequency band, no window
    P = (np.abs(np.fft.fft2(u)) ** 2).reshape(-1, H, W).sum(0)
    tot = P.sum()
    return {nm: float(P[(rad >= a) & (rad < b_)].sum() / tot) for nm, (a, b_) in
            {'DC+r<4.5': (0, 4.5), '4.5-12.5': (4.5, 12.5), '12.5-24.5': (12.5, 24.5), '>=24.5': (24.5, 99)}.items()}
print('\n== energy fraction by spatial frequency (Parseval, DC included in first band)')
print('  true residual x - x_r     ', {k: round(v, 3) for k, v in energy_split(X - XR).items()})
print('  sampled residual s0 - x_r ', {k: round(v, 3) for k, v in energy_split(S[0] - XR).items()})
print(f'  total energy: true res {((X - XR) ** 2).mean():.3e}  sampled res {((S[0] - XR) ** 2).mean():.3e}  sampled spread mean sd^2 {(SD ** 2).mean():.3e}')
# low- vs high-pass structure of the sampled residual
def bandpass(u, a, b_):
    Fu = np.fft.fft2(u); m = (rad >= a) & (rad < b_)
    return np.fft.ifft2(Fu * m).real
for nm, (a, b_) in (('low/mid r<12.5', (0, 12.5)), ('high r>=12.5', (12.5, 99))):
    rt = bandpass(X - XR, a, b_); rs = bandpass(S[0] - XR, a, b_)
    def adj(u):
        u = u - u.mean(axis=(-2, -1), keepdims=True); return float((u[:, 1:] * u[:, :-1]).mean() / (u ** 2).mean())
    print(f'  {nm:15s}: adjacent-band corr true res {adj(rt):.3f}  sampled res {adj(rs):.3f};  energy sampled/true {(rs**2).mean()/(rt**2).mean():.3f}')

# ---------------------------------------------------------------- CRPS apples-to-apples
print('\n== CRPS (%) per cube, [0,1] scale, all C*H*W elements')
rows = []
rng = np.random.default_rng(0)
LP = (rad < 12.5)
for b in range(B):
    s, t, xr, sg = S[:, b], X[b], XR[b], SIG[b]
    r = {}
    r['diff_fair'] = 100 * M.crps_samples(s, t).mean()
    r['diff_rec'] = rec['per_cube']['diffusion']['crps'][gi[b]]
    r['vh_closed'] = 100 * M.crps_gaussian(xr, sg, t).mean()
    r['vh_rec'] = rec['per_cube']['varhead']['crps'][gi[b]]
    g = xr[None] + sg[None] * rng.standard_normal((K,) + t.shape)
    r['vh_fair32'] = 100 * M.crps_samples(g, t).mean()
    r['vh_fair32_clip'] = 100 * M.crps_samples(np.clip(g, 0, 1), t).mean()
    r['diff_gaussfit'] = 100 * M.crps_gaussian(MU[b], SD[b], t).mean()
    r['N(xr,sd_diff)'] = 100 * M.crps_gaussian(xr, SD[b], t).mean()
    r['N(mu,sig_vh)'] = 100 * M.crps_gaussian(MU[b], sg, t).mean()
    # diffusion samples with the residual's high band (r >= 12.5) removed
    res_lp = np.fft.ifft2(np.fft.fft2(s - xr[None]) * LP).real
    r['diff_lowpass'] = 100 * M.crps_samples(np.clip(xr[None] + res_lp, 0, 1), t).mean()
    r['point_regressor_MAE'] = 100 * np.abs(xr - t).mean()
    rows.append(r)
keys = list(rows[0])
print('  ' + ' '.join(f'{k:>16s}' for k in keys))
for r in rows: print('  ' + ' '.join(f'{r[k]:16.4f}' for k in keys))
print('  ' + ' '.join(f'{np.mean([r[k] for r in rows]):16.4f}' for k in keys) + '   <- mean of 8')
json.dump(dict(strata=res, crps=rows), open(os.path.join(SCR, 'coverage_crps.json'), 'w'))
