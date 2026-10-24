"""Element-level mechanism checks on the 8 stored cubes (32 DDIM-10 samples each), identical code for d = 4 and d = 8
so the two strides are compared apples to apples. Adapted from docs/analysis/phase2a_verify_d8/metric-validity/
{coverage_crps,texture,extra}.py and steelman/stored_cubes.py (stride and paths parameterised). All on [0, 1].
usage: python stored_analysis.py d4|d8"""
import json, math, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '8')
import numpy as np
from scipy.stats import spearmanr, kurtosis
REPO = '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan'
sys.path.insert(0, REPO)
from scripts.uq import metrics as M

TAG = sys.argv[1]
SCR = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d4/mechanism-d4'
RR = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
CFG = {'d4': dict(D=4, npz=os.path.join(RR, 'uq/d4', 'ev' + 'al', 'd4_contrast__cubes.npz'), rec=os.path.join(RR, 'uq/d4', 'ev' + 'al', 'd4_contrast.json'),
                  vh=os.path.join(SCR, 'vh_sigma_d4.npz')),
       'd8': dict(D=8, npz=os.path.join(RR, 'uq/d8', 'ev' + 'al', 'd8_main__cubes.npz'), rec=os.path.join(RR, 'uq/d8', 'ev' + 'al', 'd8_main.json'),
                  vh='/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d8/metric-validity/vh_sigma.npz')}[TAG]
D = CFG['D']
rec = json.load(open(CFG['rec']))
sigma_r = float(rec['sigma_r'])
z = np.load(CFG['npz'])
S = (z['samples'].astype(np.float64) + 1) / 2          # [K, B, C, H, W]
X = (z['x'].astype(np.float64) + 1) / 2
XR = (z['x_r'].astype(np.float64) + 1) / 2
SIG = np.load(CFG['vh'])['sigma'].astype(np.float64) / 2.0
K, B, C, H, W = S.shape
files = list(z['files']); gi = [rec['files'].index(f) for f in files]
MU = S.mean(0); SD = S.std(0, ddof=1)
ERR = np.abs(X - XR)
OUT = {}
print(f'===== {TAG}: D {D} K {K} B {B} sigma_r {sigma_r:.5f} files {files}')

# ---------------------------------------------------------------- intervals (Weibull, as metrics.coverage_samples)
lo50 = np.empty_like(X); hi50 = np.empty_like(X); lo90 = np.empty_like(X); hi90 = np.empty_like(X)
for b in range(B):
    lo50[b], hi50[b], lo90[b], hi90[b] = np.quantile(S[:, b], [0.25, 0.75, 0.05, 0.95], axis=0, method='weibull')
c50 = (X >= lo50) & (X <= hi50); c90 = (X >= lo90) & (X <= hi90)
v50 = ERR <= M.Z[0.5] * SIG; v90 = ERR <= M.Z[0.9] * SIG
cs = M.coverage_samples(S[:, 0], X[0])
print(f'check vs metrics.coverage_samples cube 0: {cs[0.5]:.4f}/{cs[0.9]:.4f} mine {c50[0].mean():.4f}/{c90[0].mean():.4f}')
print(f'pooled 8 cubes: diffusion cov50 {c50.mean():.4f} cov90 {c90.mean():.4f} (record same cubes {np.mean([rec["per_cube"]["diffusion"]["cov50"][i] for i in gi]):.4f}/'
      f'{np.mean([rec["per_cube"]["diffusion"]["cov90"][i] for i in gi]):.4f}); varhead {v50.mean():.4f}/{v90.mean():.4f} (record {np.mean([rec["per_cube"]["varhead"]["cov50"][i] for i in gi]):.4f}/'
      f'{np.mean([rec["per_cube"]["varhead"]["cov90"][i] for i in gi]):.4f})')
OUT['pooled'] = dict(d50=c50.mean(), d90=c90.mean(), v50=v50.mean(), v90=v90.mean())

# ---------------------------------------------------------------- (1b) cube-level slope on the stored cubes
rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
rmse_R = np.array([rms(X[b] - XR[b]) for b in range(B)]); rsd = np.array([rms(SD[b]) for b in range(B)]); rvh = np.array([rms(SIG[b]) for b in range(B)])
sl_d = np.polyfit(np.log(rmse_R), np.log(rsd), 1)[0]; sl_v = np.polyfit(np.log(rmse_R), np.log(rvh), 1)[0]
print(f'(1b) stored cubes: RMSE_R % {np.round(100*rmse_R, 3)}; RMS sd % {np.round(100*rsd, 3)}; RMS vh % {np.round(100*rvh, 3)}')
print(f'     slope log RMS sd on log RMSE_R {sl_d:.3f}; varhead {sl_v:.3f}; corr {np.corrcoef(np.log(rmse_R), np.log(rsd))[0,1]:.3f} / {np.corrcoef(np.log(rmse_R), np.log(rvh))[0,1]:.3f}')
OUT['stored_slope'] = dict(diff=sl_d, vh=sl_v, rmse_R=rmse_R.tolist(), rms_sd=rsd.tolist(), rms_vh=rvh.tolist())

# ---------------------------------------------------------------- (2) sensor phase
pos = (np.arange(H)[:, None] % D) * D + (np.arange(W)[None, :] % D)
last = (H - 1) // D * D                                   # last sensor-sample pixel on the top-left-anchored grid
interior = np.zeros((H, W), bool); interior[:last + 1, :last + 1] = True
posb = np.broadcast_to(pos, X.shape); intb = np.broadcast_to(interior, X.shape)
ph = {}
for nm, msk in (('all', np.ones_like(intb)), ('interior', intb)):
    rows = []
    for p in range(D * D):
        s = (posb == p) & msk
        rows.append(dict(err=rms(ERR[s]), sd_mean=float(SD[s].mean()), sd_rms=rms(SD[s]), vh_mean=float(SIG[s].mean()), vh_rms=rms(SIG[s]),
                         d50=float(c50[s].mean()), d90=float(c90[s].mean()), v50=float(v50[s].mean()), v90=float(v90[s].mean())))
    e = np.array([r['err'] for r in rows]); sdm = np.array([r['sd_mean'] for r in rows]); vhm = np.array([r['vh_mean'] for r in rows])
    best = int(np.argmin(e))
    r0 = rows[0]
    ph[nm] = dict(rows=rows, err_ratio=float(e.max() / e.min()), sd_ratio=float(sdm.max() / sdm.min()), vh_ratio=float(vhm.max() / vhm.min()),
                  argmin_err=divmod(best, D), corr_err_sd=float(np.corrcoef(e, sdm)[0, 1]), corr_err_vh=float(np.corrcoef(e, vhm)[0, 1]),
                  median_err=float(np.median(e)), phase00=r0,
                  err_ratio_excl00=float(np.delete(e, 0).max() / np.delete(e, 0).min()), sd_ratio_excl00=float(np.delete(sdm, 0).max() / np.delete(sdm, 0).min()))
    print(f'(2) phase [{nm}] {D*D} phases: RMS err {e.min():.4f}-{e.max():.4f} ({e.max()/e.min():.2f}x); diffusion mean sd {sdm.min():.4f}-{sdm.max():.4f} ({sdm.max()/sdm.min():.3f}x); '
          f'varhead mean sigma {vhm.min():.4f}-{vhm.max():.4f} ({vhm.max()/vhm.min():.2f}x); argmin err phase {divmod(best, D)}; corr(err, sd) {ph[nm]["corr_err_sd"]:.3f} corr(err, vh) {ph[nm]["corr_err_vh"]:.3f}')
    print(f'    excluding (0,0): err {ph[nm]["err_ratio_excl00"]:.2f}x, sd {ph[nm]["sd_ratio_excl00"]:.3f}x')
    print(f'    phase (0,0): RMS err {r0["err"]:.4f} (median phase {np.median(e):.4f}); diffusion sd mean {r0["sd_mean"]:.4f} (= {r0["sd_mean"]/r0["err"]:.1f}x err) cov50 {r0["d50"]:.3f} cov90 {r0["d90"]:.3f}; '
          f'varhead sigma {r0["vh_mean"]:.4f} cov50 {r0["v50"]:.3f} cov90 {r0["v90"]:.3f}')
    print('    RMS err by phase:\n' + str(np.round(e.reshape(D, D), 4)))
    print('    diffusion mean sd by phase:\n' + str(np.round(sdm.reshape(D, D), 4)))
    print('    varhead mean sigma by phase:\n' + str(np.round(vhm.reshape(D, D), 4)))
# edge (extrapolated) strip
edge = ~interior; edgeb = np.broadcast_to(edge, X.shape)
print(f'    extrapolated edge (pixels > {last}): RMS err {rms(ERR[edgeb]):.4f} sd {SD[edgeb].mean():.4f} vh {SIG[edgeb].mean():.4f}; interior non-grid err {rms(ERR[intb & (posb != 0)]):.4f} sd {SD[intb & (posb != 0)].mean():.4f}')
OUT['phase'] = ph

# ---------------------------------------------------------------- (3) structure of sample deviations vs true residual
def lag1(a):                      # [..., H, W], mean-removed per map
    a = a - a.mean(axis=(-2, -1), keepdims=True)
    return float(((a[..., 1:, :] * a[..., :-1, :]).mean() + (a[..., :, 1:] * a[..., :, :-1]).mean()) / 2 / (a ** 2).mean())
def band_corr(a):                 # [C, H, W] -> mean corr of adjacent bands (per-band normalised)
    a = a.reshape(a.shape[0], -1); a = a - a.mean(1, keepdims=True); n = np.sqrt((a ** 2).sum(1))
    return float(((a[1:] * a[:-1]).sum(1) / (n[1:] * n[:-1] + 1e-30)).mean())
def top3(a):
    sv = np.linalg.svd(a.reshape(a.shape[0], -1), compute_uv=False); return float((sv[:3] ** 2).sum() / (sv ** 2).sum())
def lapvar(a): return float((M.laplacian(a) ** 2).mean() / (a ** 2).mean())
rt = X - XR; dev = S[:4] - MU[None]; rs0 = S[0] - XR
st = dict(lag1_true=np.mean([lag1(rt[b]) for b in range(B)]), lag1_dev=np.mean([lag1(dev[k, b]) for k in range(4) for b in range(B)]),
          lag1_sres=np.mean([lag1(rs0[b]) for b in range(B)]),
          band_true=np.mean([band_corr(rt[b]) for b in range(B)]), band_dev=np.mean([band_corr(dev[k, b]) for k in range(4) for b in range(B)]),
          band_sres=np.mean([band_corr(rs0[b]) for b in range(B)]),
          top3_true=np.mean([top3(rt[b]) for b in range(B)]), top3_dev=np.mean([top3(dev[k, b]) for k in range(4) for b in range(B)]),
          lapvar_true=np.mean([lapvar(rt[b]) for b in range(B)]), lapvar_dev=np.mean([lapvar(dev[k, b]) for k in range(4) for b in range(B)]))
print(f'(3) spatial lag-1: true residual {st["lag1_true"]:.3f}, sample deviation (s-mu) {st["lag1_dev"]:.3f}, sampled residual (s0-x_r) {st["lag1_sres"]:.3f}')
print(f'    adjacent-band corr: true {st["band_true"]:.3f}, deviation {st["band_dev"]:.3f}, sampled residual {st["band_sres"]:.3f}')
print(f'    top-3 spectral PC energy (own PCs): true {st["top3_true"]:.3f}, deviation {st["top3_dev"]:.3f};  Laplacian energy/variance: true {st["lapvar_true"]:.2f}, deviation {st["lapvar_dev"]:.2f} (white noise 20)')
# Parseval energy split and coherence
ky, kx = np.meshgrid(np.fft.fftfreq(H) * H, np.fft.fftfreq(W) * W, indexing='ij'); rad = np.sqrt(kx ** 2 + ky ** 2)
BANDS = {'r<4.5': (0, 4.5), '4.5-12.5': (4.5, 12.5), '12.5-24.5': (12.5, 24.5), '>=24.5': (24.5, 99)}
def esplit(u):
    P = (np.abs(np.fft.fft2(u)) ** 2).reshape(-1, H, W).sum(0); tot = P.sum()
    return {nm: float(P[(rad >= a) & (rad < b_)].sum() / tot) for nm, (a, b_) in BANDS.items()}
et, es = esplit(rt), esplit(rs0)
Ft = np.fft.fft2(rt)
Psamp = np.zeros((H, W)); Pt = (np.abs(Ft) ** 2).reshape(-1, H, W).sum(0); cross = np.zeros((H, W))
for k in range(K):
    Fs = np.fft.fft2(S[k] - XR)
    Psamp += (np.abs(Fs) ** 2).reshape(-1, H, W).sum(0) / K; cross += (Fs * np.conj(Ft)).real.reshape(-1, H, W).sum(0) / K
lo, hi = rad < 12.5, rad >= 12.5
st.update(esplit_true=et, esplit_samp=es, samp_over_true_lo=float(Psamp[lo].sum() / Pt[lo].sum()), samp_over_true_hi=float(Psamp[hi].sum() / Pt[hi].sum()),
          coh_lo=float(cross[lo].sum() / math.sqrt(Psamp[lo].sum() * Pt[lo].sum())), coh_hi=float(cross[hi].sum() / math.sqrt(Psamp[hi].sum() * Pt[hi].sum())))
print('    energy fraction by radial frequency: true residual', {k: round(v, 3) for k, v in et.items()}, ' sampled residual', {k: round(v, 3) for k, v in es.items()})
print(f'    sampled/true residual energy: r<12.5 {st["samp_over_true_lo"]:.3f}, r>=12.5 {st["samp_over_true_hi"]:.3f}; coherence lo {st["coh_lo"]:+.3f} hi {st["coh_hi"]:+.3f}')
grid_f = [H // D * m for m in range(1, D) if H // D * m <= H // 2]
for f in grid_f:
    on = Psamp[0, f]; nb = 0.5 * (Psamp[0, f - 1] + Psamp[0, (f + 1) % W]); on_t = Pt[0, f]; nb_t = 0.5 * (Pt[0, f - 1] + Pt[0, (f + 1) % W])
    print(f'    grid freq kx={f}: sampled-res on/nb {on/nb:.2f}  true-res on/nb {on_t/nb_t:.2f}')
hf_s0 = np.mean([M.hf_energy_ratio(S[0, b], X[b]) for b in range(B)]); hf_r = np.mean([M.hf_energy_ratio(XR[b], X[b]) for b in range(B)])
rng = np.random.default_rng(0)
Yn = np.clip(XR + SD * rng.standard_normal(XR.shape), 0, 1)
hf_n = np.mean([M.hf_energy_ratio(Yn[b], X[b]) for b in range(B)])
w1 = lambda Y: np.mean([M.grad_w1(Y[b], X[b]) for b in range(B)])
st.update(hf_s0=hf_s0, hf_reg=hf_r, hf_xr_plus_iid=hf_n, w1_s0=w1(S[0]), w1_reg=w1(XR), w1_xr_plus_iid=w1(Yn))
print(f'    hf: single sample {hf_s0:+.3f}, regressor {hf_r:+.3f}, x_r + iid N(0, sd^2) {hf_n:+.3f};  grad W1: sample {st["w1_s0"]:.5f} regressor {st["w1_reg"]:.5f} x_r+iid {st["w1_xr_plus_iid"]:.5f}')
OUT['structure'] = st

# ---------------------------------------------------------------- (4) coverage by strata
sc = np.sqrt((rt ** 2).mean(axis=(1, 2, 3)))[:, None, None, None] * np.ones_like(X)
h50 = ERR <= M.Z[0.5] * sc; h90 = ERR <= M.Z[0.9] * sc
def strat(name, v, nq=5):
    edges = np.quantile(v, np.linspace(0, 1, nq + 1))
    idx = np.clip(np.searchsorted(edges, v, side='right') - 1, 0, nq - 1)
    out = []
    print(f'(4) coverage by {nq}-tile of {name}:  diff cov50 cov90 | vh cov50 cov90 | per-cube-homosc cov50 cov90 | RMS sd_diff  RMS sig_vh  RMS err | err/sd_diff  err/sig_vh  (sigma_r units: sd, err)')
    for q in range(nq):
        s = idx == q
        r = dict(d50=float(c50[s].mean()), d90=float(c90[s].mean()), v50=float(v50[s].mean()), v90=float(v90[s].mean()), h50=float(h50[s].mean()), h90=float(h90[s].mean()),
                 sd=rms(SD[s]), sv=rms(SIG[s]), er=rms(ERR[s]))
        r['rd'] = r['er'] / r['sd']; r['rv'] = r['er'] / r['sv']
        out.append(r)
        print(f'    q{q}: {r["d50"]:.3f} {r["d90"]:.3f} | {r["v50"]:.3f} {r["v90"]:.3f} | {r["h50"]:.3f} {r["h90"]:.3f} | {r["sd"]:.4f} {r["sv"]:.4f} {r["er"]:.4f} | {r["rd"]:.2f} {r["rv"]:.2f}  ({2*r["sd"]/sigma_r:.2f}, {2*r["er"]/sigma_r:.2f})')
    return out
OUT['strata'] = {'sig_vh': strat('varhead sigma', SIG), 'err': strat('|x - x_r| (outcome-conditioned)', ERR),
                 'sobel_xr': strat('Sobel |grad x_r|', M.sobel_mag(XR)), 'sig_vh_dec': strat('varhead sigma (deciles)', SIG, 10)}
# pixel-level dynamic range (RMS over bands)
pix = lambda a: np.sqrt((a ** 2).mean(1)).ravel()
p9010 = lambda a: float(np.quantile(a, .9) / np.quantile(a, .1))
dr = dict(pix_sd=p9010(pix(SD)), pix_vh=p9010(pix(SIG)), pix_err=p9010(pix(ERR)), el_sd=p9010(SD.ravel()), el_vh=p9010(SIG.ravel()), el_err=p9010(ERR.ravel()))
print(f'    P90/P10 pixel-level: diffusion sd {dr["pix_sd"]:.2f}, varhead {dr["pix_vh"]:.2f}, error {dr["pix_err"]:.2f}; element-level {dr["el_sd"]:.2f} / {dr["el_vh"]:.2f} / {dr["el_err"]:.2f}')
zz = (X - MU) / SD; zv = (X - XR) / SIG
summ = dict(sdlog_sd=float(np.log(SD).std()), sdlog_vh=float(np.log(SIG).std()), sp_sd_err=float(spearmanr(SD.ravel(), ERR.ravel()).correlation),
            sp_vh_err=float(spearmanr(SIG.ravel(), ERR.ravel()).correlation), sp_sd_vh=float(spearmanr(SD.ravel(), SIG.ravel()).correlation),
            exkurt_z=float(kurtosis(zz.ravel())), exkurt_zv=float(kurtosis(zv.ravel())),
            ause_sd=float(np.mean([M.ause(SD[b], MU[b] - X[b]) for b in range(B)])), ause_vh=float(np.mean([M.ause(SIG[b], XR[b] - X[b]) for b in range(B)])))
rank = (S < X[None]).sum(0); hist = np.bincount(rank.ravel(), minlength=K + 1) / rank.size
summ['pit_center'] = float(hist[K // 2 - 1:K // 2 + 2].mean()); summ['pit_ends'] = (float(hist[0]), float(hist[-1])); summ['pit_uniform'] = 1 / (K + 1)
print(f'    SD of log std: diffusion {summ["sdlog_sd"]:.3f} varhead {summ["sdlog_vh"]:.3f}; Spearman(std, |err|): diffusion {summ["sp_sd_err"]:.3f} varhead {summ["sp_vh_err"]:.3f}; '
      f'Spearman(sd, sig_vh) {summ["sp_sd_vh"]:.3f}; excess kurtosis z: diffusion {summ["exkurt_z"]:.1f} varhead {summ["exkurt_zv"]:.1f}; AUSE {summ["ause_sd"]:.3f} / {summ["ause_vh"]:.3f}; '
      f'PIT centre {summ["pit_center"]:.3f} ends {summ["pit_ends"][0]:.3f}/{summ["pit_ends"][1]:.3f} (uniform {summ["pit_uniform"]:.3f})')
OUT['dyn_range'] = dr; OUT['summary'] = summ
# per-band spread vs error; collapsed-band check
bsd = SD.mean(axis=(0, 2, 3)); berr = np.sqrt((ERR ** 2).mean(axis=(0, 2, 3))); bvh = SIG.mean(axis=(0, 2, 3))
b50 = c50.mean(axis=(0, 2, 3)); b90 = c90.mean(axis=(0, 2, 3))
med = np.median(bsd)
coll = [int(i) for i in range(C) if bsd[i] < 0.6 * med]
print(f'    per band: diffusion mean sd {bsd.min():.4f}-{bsd.max():.4f} (median {med:.4f}); RMS err {berr.min():.4f}-{berr.max():.4f}; varhead {bvh.min():.4f}-{bvh.max():.4f}')
print(f'    bands with sd < 0.6 x median: {coll}; their sd {np.round(bsd[coll], 4).tolist()} err {np.round(berr[coll], 4).tolist()} cov50 {np.round(b50[coll], 3).tolist()} cov90 {np.round(b90[coll], 3).tolist()}')
nc = [i for i in range(C) if i not in coll]
print(f'    normal bands: sd {bsd[nc].min():.4f}-{bsd[nc].max():.4f} vs err {berr[nc].min():.4f}-{berr[nc].max():.4f}; cov50 {b50[nc].min():.3f}-{b50[nc].max():.3f}; corr(sd, err) across bands {np.corrcoef(bsd[nc], berr[nc])[0,1]:.3f}')
OUT['bands'] = dict(sd=bsd.tolist(), err=berr.tolist(), vh=bvh.tolist(), cov50=b50.tolist(), cov90=b90.tolist(), collapsed=coll)

# ---------------------------------------------------------------- (E) CRPS decomposition on the 8 cubes
CS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
rows = []
LP = rad < 12.5
for b in range(B):
    s, t, xr, sg = S[:, b], X[b], XR[b], SIG[b]
    r = dict(diff=100 * M.crps_samples(s, t).mean(), diff_rec=rec['per_cube']['diffusion']['crps'][gi[b]], vh=100 * M.crps_gaussian(xr, sg, t).mean(),
             vh_rec=rec['per_cube']['varhead']['crps'][gi[b]], g_xr_sd=100 * M.crps_gaussian(xr, SD[b], t).mean(),
             g_xr_sdcube=100 * M.crps_gaussian(xr, np.full_like(xr, rms(SD[b])), t).mean(), g_mu_vh=100 * M.crps_gaussian(MU[b], sg, t).mean())
    for c in CS:
        r[f'homo_{c}'] = 100 * M.crps_gaussian(xr, np.full_like(xr, c * sigma_r / 2), t).mean()
    res_lp = np.fft.ifft2(np.fft.fft2(s - xr[None]) * LP).real
    r['diff_lowpass'] = 100 * M.crps_samples(np.clip(xr[None] + res_lp, 0, 1), t).mean()
    rows.append(r)
mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
bestc = min(CS, key=lambda c: mean[f'homo_{c}'])
print('(E) CRPS (%) mean of 8 cubes: ' + ', '.join(f'{k} {v:.4f}' for k, v in mean.items()))
print(f'    best constant c (on these 8) {bestc}: {mean[f"homo_{bestc}"]:.4f}; diffusion - homo_best {mean["diff"] - mean[f"homo_{bestc}"]:+.4f}; vh - homo_best {mean["vh"] - mean[f"homo_{bestc}"]:+.4f}')
OUT['crps8'] = dict(rows=rows, mean=mean, bestc=bestc)
json.dump(OUT, open(os.path.join(SCR, f'stored_{TAG}.json'), 'w'), indent=1, default=float)
