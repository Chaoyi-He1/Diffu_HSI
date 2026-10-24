"""Analysis of main.py outputs (CPU samples, denoiser test) and of the stored K = 32 GPU samples of d8_main."""
import os
import sys

REPO = '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan'
sys.path.insert(0, REPO)
import numpy as np
from scipy.stats import spearmanr, pearsonr
from scripts.uq import metrics as M

OUT = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/verify_d8/condition-blindness'
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
b = np.load(f'{OUT}/base.npz')
x, xr, e, svh = (b[k].astype(np.float64) for k in ('x', 'xr', 'e', 'sig_vh'))
sigma_r = float(b['sigma_r'])
N = x.shape[0]
SRC = [(i - 4) % N for i in range(N)]            # torch.roll(4, 0): rolled[i] = orig[(i - 4) mod N]
files = [str(f) for f in b['files']]
print('cubes', N, 'sigma_r', sigma_r, 'src', SRC)


def sp(a, c):
    return float(spearmanr(np.ravel(a), np.ravel(c)).correlation)


def pix(a):                                      # [C, H, W] -> per-pixel RMS over bands [H, W]
    return np.sqrt((np.asarray(a) ** 2).mean(0))


def fmt(v):
    return ' '.join(f'{u:+.3f}' for u in v) + f'  | mean {np.mean(v):+.3f}'


PARTS = os.environ.get('PARTS', 'ABC')
# ------------------------------------------------------------------ A. stored K = 32 GPU samples (true condition)
st = np.load(f'{R}/uq/d8/eval/d8_main__cubes.npz')
assert [str(f) for f in st['files']] == files
s32 = st['samples'].astype(np.float64)                   # [32, 8, C, H, W] composed + clamped, [-1, 1]
r32 = (s32 - xr[None]) / sigma_r
std32 = r32.std(0, ddof=1)
if 'A' in PARTS:
    print('\n=== A. stored GPU K = 32 samples, true condition (standardised units) ===')
    print('frac samples clamped at +-1: %.4f; frac truth at +-1: %.4f' % (np.mean(np.abs(s32) >= 1.0 - 1e-3), np.mean(np.abs(x) >= 1.0 - 1e-6)))
    print('rms e per cube      ', fmt([np.sqrt((e[i] ** 2).mean()) for i in range(N)]))
    print('rms std32 per cube  ', fmt([np.sqrt((std32[i] ** 2).mean()) for i in range(N)]))
    print('rms sig_vh per cube ', fmt([np.sqrt((svh[i] ** 2).mean()) for i in range(N)]))
    print('Spearman elem std32 vs |e|   ', fmt([sp(std32[i], np.abs(e[i])) for i in range(N)]))
    print('Spearman elem sig_vh vs |e|  ', fmt([sp(svh[i], np.abs(e[i])) for i in range(N)]))
    print('Spearman elem std32 vs sig_vh', fmt([sp(std32[i], svh[i]) for i in range(N)]))
    print('Spearman pix  std32 vs |e|   ', fmt([sp(pix(std32[i]), pix(e[i])) for i in range(N)]))
    print('Spearman pix  sig_vh vs |e|  ', fmt([sp(pix(svh[i]), pix(e[i])) for i in range(N)]))
    print('Spearman pix  std32 vs sig_vh', fmt([sp(pix(std32[i]), pix(svh[i])) for i in range(N)]))
    q = lambda a: np.percentile(a, 90) / np.percentile(a, 10)
    print('P90/P10 elem std32 ', fmt([q(std32[i]) for i in range(N)]))
    print('P90/P10 elem sig_vh', fmt([q(svh[i]) for i in range(N)]))
    print('P90/P10 elem |e|   ', fmt([q(np.abs(e[i])) for i in range(N)]))
    print('P90/P10 pix std32  ', fmt([q(pix(std32[i])) for i in range(N)]))
    print('P90/P10 pix sig_vh ', fmt([q(pix(svh[i])) for i in range(N)]))
    print('P90/P10 pix rms e  ', fmt([q(pix(e[i])) for i in range(N)]))
    # reliability: bin all elements by decile of the varhead sigma, report realised rms e, diffusion rms std, varhead rms sigma
    print('\nreliability by varhead-sigma decile (pooled 8 cubes): realised RMS e | diffusion RMS std32 | varhead RMS sigma | diffusion cov50 | varhead cov50')
    sv, ee, sd = svh.ravel(), e.ravel(), std32.ravel()
    lo50, hi50 = np.quantile(r32, [0.25, 0.75], axis=0, method='weibull')
    cov_d = ((e >= lo50) & (e <= hi50)).ravel()
    cov_v = (np.abs(e) <= M.Z[0.5] * svh).ravel()
    edges = np.quantile(sv, np.linspace(0, 1, 11))
    for j in range(10):
        m = (sv >= edges[j]) & (sv <= edges[j + 1])
        print(f'  d{j}: e {np.sqrt((ee[m] ** 2).mean()):.3f} | diff {np.sqrt((sd[m] ** 2).mean()):.3f} | vh {np.sqrt((sv[m] ** 2).mean()):.3f} | '
              f'cov50 diff {cov_d[m].mean():.3f} vh {cov_v[m].mean():.3f}')
    print('reliability by diffusion-std32 decile: realised RMS e | diffusion RMS std32')
    edges = np.quantile(sd, np.linspace(0, 1, 11))
    for j in range(10):
        m = (sd >= edges[j]) & (sd <= edges[j + 1])
        print(f'  d{j}: e {np.sqrt((ee[m] ** 2).mean()):.3f} | diff {np.sqrt((sd[m] ** 2).mean()):.3f} | cov50 diff {cov_d[m].mean():.3f}')
    print('pooled kurtosis of e / sqrt(std32^2) (normalised error; Gaussian 3): diffusion %.2f, varhead %.2f, raw e %.2f' % (
        np.mean((ee / sd) ** 4) / np.mean((ee / sd) ** 2) ** 2, np.mean((ee / sv) ** 4) / np.mean((ee / sv) ** 2) ** 2,
        np.mean(ee ** 4) / np.mean(ee ** 2) ** 2))
    # sample-level kurtosis: are the diffusion's per-element predictive distributions heavy-tailed?
    rc = r32 - r32.mean(0, keepdims=True)
    print('pooled kurtosis of diffusion samples about their per-element mean: %.2f' % (np.mean(rc ** 4) / np.mean(rc ** 2) ** 2))

if 'B' in PARTS:
    # ------------------------------------------------------------------ B. CPU same-seed condition tests
    print('\n=== B. CPU DDIM-10 samples, same seeds across conditions ===')
    S = {}
    for c in ('true', 'swap', 'zero', 'swap_y', 'swap_xr'):
        p = f'{OUT}/samples_{c}.npz'
        if os.path.exists(p):
            S[c] = np.load(p)['r'].astype(np.float64)
    print('available', {c: v.shape[0] for c, v in S.items()})
    if 'true' in S:
        rt = S['true']
        K = rt.shape[0]
        pairs = [(a, c) for a in range(K) for c in range(a + 1, K)]
        seed_rms = [np.mean([np.sqrt(((rt[a, i] - rt[c, i]) ** 2).mean()) for a, c in pairs]) for i in range(N)]
        print('rms std (CPU, K=%d) per cube ' % K, fmt([np.sqrt((rt[:, i].std(0, ddof=1) ** 2).mean()) for i in range(N)]))
        print('seed-to-seed RMS diff, true cond', fmt(seed_rms))
        std_t = rt.std(0, ddof=1)
        for c, rc_ in S.items():
            if c == 'true':
                continue
            Kc = rc_.shape[0]
            d_same = [np.mean([np.sqrt(((rc_[k, i] - rt[k, i]) ** 2).mean()) for k in range(Kc)]) for i in range(N)]
            print(f'\n[{c}] K={Kc}')
            print('  same-seed RMS diff vs true      ', fmt(d_same))
            print('  ratio to seed-to-seed RMS diff  ', fmt([d_same[i] / seed_rms[i] for i in range(N)]))
            print('  Pearson r_c^k[i] vs r_true^k[i] (same noise)        ',
                  fmt([np.mean([pearsonr(rc_[k, i].ravel(), rt[k, i].ravel())[0] for k in range(Kc)]) for i in range(N)]))
            if c in ('swap', 'swap_y', 'swap_xr'):
                print('  Pearson r_c^k[i] vs r_true^j[src] (same cond, k!=j)  ',
                      fmt([np.mean([pearsonr(rc_[k, i].ravel(), rt[(k + 1) % K, SRC[i]].ravel())[0] for k in range(Kc)]) for i in range(N)]))
                print('  Pearson r_true^k[i] vs r_true^j[i] (same cube, k!=j)',
                      fmt([np.mean([pearsonr(rt[k, i].ravel(), rt[(k + 1) % K, i].ravel())[0] for k in range(Kc)]) for i in range(N)]))
            if Kc >= 3:
                std_c = rc_.std(0, ddof=1)
                std_t_m = rt[:Kc].std(0, ddof=1)
                print('  rms std per cube                ', fmt([np.sqrt((std_c[i] ** 2).mean()) for i in range(N)]))
                print('  Spearman pix std_c[i] vs std_true[i]   (same noise) ', fmt([sp(pix(std_c[i]), pix(std_t_m[i])) for i in range(N)]))
                print('  Spearman pix std_c[i] vs std_true[src] (same cond)  ', fmt([sp(pix(std_c[i]), pix(std_t_m[SRC[i]])) for i in range(N)]))
                print('  Spearman pix std_c[i] vs std32_GPU[i]               ', fmt([sp(pix(std_c[i]), pix(std32[i])) for i in range(N)]))
                print('  Spearman pix std_c[i] vs std32_GPU[src]             ', fmt([sp(pix(std_c[i]), pix(std32[SRC[i]])) for i in range(N)]))
                print('  Spearman elem std_c[i] vs |e[i]|                    ', fmt([sp(std_c[i], np.abs(e[i])) for i in range(N)]))
                print('  Spearman pix  std_c[i] vs rms e[i]                  ', fmt([sp(pix(std_c[i]), pix(e[i])) for i in range(N)]))
                print('  Spearman pix  std_c[i] vs rms e[src]                ', fmt([sp(pix(std_c[i]), pix(e[SRC[i]])) for i in range(N)]))
                print('  Spearman pix  std_c[i] vs sig_vh[i]                 ', fmt([sp(pix(std_c[i]), pix(svh[i])) for i in range(N)]))
                print('  Spearman pix  std_c[i] vs sig_vh[src]               ', fmt([sp(pix(std_c[i]), pix(svh[SRC[i]])) for i in range(N)]))
        Kt = rt.shape[0]
        print(f'\n[true] K={Kt} (CPU)')
        print('  Spearman pix std_true vs std32_GPU (same cube)   ', fmt([sp(pix(std_t[i]), pix(std32[i])) for i in range(N)]))
        print('  Spearman pix std_true vs std32_GPU (other cube)  ', fmt([sp(pix(std_t[i]), pix(std32[SRC[i]])) for i in range(N)]))
        print('  Spearman elem std_true vs |e|                    ', fmt([sp(std_t[i], np.abs(e[i])) for i in range(N)]))
        print('  Spearman pix  std_true vs rms e                  ', fmt([sp(pix(std_t[i]), pix(e[i])) for i in range(N)]))
        print('  Spearman pix  std_true vs sig_vh                 ', fmt([sp(pix(std_t[i]), pix(svh[i])) for i in range(N)]))
        print('  Spearman elem std_true vs sig_vh                 ', fmt([sp(std_t[i], svh[i]) for i in range(N)]))

        # CRPS on the evaluator's scale ([0, 1], %), composed with the TRUE centre like uq_eval's y-swap
        print('\nCRPS (%) per cube, samples composed with the true x_r and clamped; varhead and homoscedastic Gaussians for reference')
        x01, mu01 = M.to01(x), M.to01(xr)
        def crps_of(r):
            s = np.clip(xr[None] + sigma_r * r, -1, 1)
            return [100 * float(M.crps_samples(M.to01(s[:, i]), x01[i]).mean()) for i in range(N)]
        def cov50_of(r):
            s = np.clip(xr[None] + sigma_r * r, -1, 1)
            return [M.coverage_samples(M.to01(s[:, i]), x01[i], levels=(0.5,))[0.5] for i in range(N)]
        rows = {}
        for c, rc_ in S.items():
            rows[c] = crps_of(rc_)
            print(f'  {c:8s} K={rc_.shape[0]}: ', fmt(rows[c]), '| cov50', fmt(cov50_of(rc_)))
        rows['true_K4'] = crps_of(rt[:4]); print('  true     K=4: ', fmt(rows['true_K4']))
        rows['gpu32'] = crps_of(r32); print('  GPU32    K=32:', fmt(rows['gpu32']), '| cov50', fmt(cov50_of(r32)))
        rows['varhead'] = [100 * float(M.crps_gaussian(mu01[i], svh[i] * sigma_r / 2, x01[i]).mean()) for i in range(N)]
        print('  varhead      ', fmt(rows['varhead']))
        rows['homo'] = [100 * float(M.crps_gaussian(mu01[i], np.full_like(mu01[i], sigma_r / 2), x01[i]).mean()) for i in range(N)]
        print('  homo N(x_r, sigma_r^2)', fmt(rows['homo']))
        rec_files = None
        import json
        rec = json.load(open(f'{R}/uq/d8/eval/d8_main.json'))
        idx = [rec['files'].index(f) for f in files]
        print('  record diffusion crps (these cubes)', fmt([rec['per_cube']['diffusion']['crps'][j] for j in idx]))
        print('  record varhead crps (these cubes)  ', fmt([rec['per_cube']['varhead']['crps'][j] for j in idx]))
        for c in ('swap', 'zero', 'swap_y', 'swap_xr'):
            if c in rows:
                ref = rows['true'] if S[c].shape[0] == Kt else rows['true_K4']
                dd = np.array(rows[c]) - np.array(ref)
                print(f'  CRPS {c} - true (matched K): mean {dd.mean():+.4f}  SE {dd.std(ddof=1) / np.sqrt(N):.4f}  per cube {fmt(dd)}')
        dd = np.array(rows['true']) - np.array(rows['homo'])
        print(f'  CRPS diffusion(true) - homo: mean {dd.mean():+.4f} SE {dd.std(ddof=1)/np.sqrt(N):.4f}')
        # sample mean information
        print('\nPearson(mean over seeds of r, e): true', fmt([pearsonr(rt[:, i].mean(0).ravel(), e[i].ravel())[0] for i in range(N)]))
        if 'swap' in S:
            print('                                  swap', fmt([pearsonr(S['swap'][:, i].mean(0).ravel(), e[i].ravel())[0] for i in range(N)]))
        print('Pearson(mean of 32 GPU r, e)            ', fmt([pearsonr(r32[:, i].mean(0).ravel(), e[i].ravel())[0] for i in range(N)]))

if 'C' in PARTS:
    # ------------------------------------------------------------------ C. denoiser sensitivity
    p = f'{OUT}/denoiser.npz'
    if os.path.exists(p):
        print('\n=== C. one-call denoiser on x_t = sqrt(ab) e + sqrt(1 - ab) n (fixed n), x0 MSE vs the true residual e ===')
        D = np.load(p)
        E2 = float((e ** 2).mean())
        print('E[e^2] =', round(E2, 4))
        for t_ in (999, 900, 700, 500, 300, 100, 20):
            line = f't={t_:4d}: '
            x0t = D[f'{t_}_true'].astype(np.float64)
            for c in ('true', 'swap', 'zero', 'swap_y', 'swap_xr'):
                x0 = D[f'{t_}_{c}'].astype(np.float64)
                line += f'{c} {((x0 - e) ** 2).mean():.4f}  '
            rel = {c: float(np.sqrt(((D[f"{t_}_{c}"].astype(np.float64) - x0t) ** 2).mean()) / np.sqrt((x0t ** 2).mean()))
                   for c in ('swap', 'zero', 'swap_y', 'swap_xr')}
            print(line, '| RMS(x0_c - x0_true)/RMS(x0_true):', ' '.join(f'{c} {v:.3f}' for c, v in rel.items()))
        for t_ in (500, 300, 100):
            x0t = D[f'{t_}_true'].astype(np.float64); x0s = D[f'{t_}_swap'].astype(np.float64)
            print(f't={t_}: per-cube MSE true', fmt([((x0t[i] - e[i]) ** 2).mean() for i in range(N)]))
            print(f't={t_}: per-cube MSE swap', fmt([((x0s[i] - e[i]) ** 2).mean() for i in range(N)]))
