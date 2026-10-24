"""Supplementary checks: record-level P1 statistics, clamp effects, sensor-phase coverage."""
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
pc = rec['per_cube']
hs = np.array(pc['diffusion']['single_hf']); hr = np.array(pc['regressor']['hf']); hm = np.array(pc['diffusion']['mean_hf'])
print('== record, 200 cubes')
print(f'  hf_single > 0 in {np.mean(hs > 0):.3f} of cubes (min {hs.min():.3f}, max {hs.max():.3f}); hf_regressor < 0 in {np.mean(hr < 0):.3f}')
d = np.abs(hs) - np.abs(hr)
print(f'  mean(|hf_s|-|hf_r|) {d.mean():.4f}; fraction of cubes where |hf_s|<|hf_r|: {np.mean(d < 0):.3f}')
print(f'  energy multiples: single e^mean = {math.exp(hs.mean()):.2f}x truth; regressor e^mean = {math.exp(hr.mean()):.4f}x truth (1/{1/math.exp(hr.mean()):.1f})')
print(f'  P1 restated on mean-of-K: mean(|hf_mean|-|hf_r|) {np.mean(np.abs(hm)-np.abs(hr)):.4f}')
w1s = np.array(pc['diffusion']['single_w1']); w1r = np.array(pc['regressor']['w1'])
print(f'  W1 single {w1s.mean():.5f} vs regressor {w1r.mean():.5f}; single worse in {np.mean(w1s > w1r):.3f} of cubes')
sam_s = np.array(pc['diffusion']['single_sam_deg']); sam_r = np.array(pc['regressor']['sam_deg'])
print(f'  SAM single {sam_s.mean():.3f} vs regressor {sam_r.mean():.3f} deg')
# P1 under a one-sided reading: sample must not over-sharpen beyond truth (r_HF in [hf_r, 0]) -> fraction of cubes over-sharpened
print(f'  single sample over-sharpened (r_HF > 0) in {np.mean(hs > 0):.3f} of cubes; by more than the regressor under-sharpens (|hf_s| > |hf_r|) in {np.mean(d > 0):.3f}')
# paired SE cluster from scripts.paired_stats
try:
    from scripts import paired_stats as P
    print('  paired_stats functions:', [n for n in dir(P) if not n.startswith('_')])
except Exception as e:
    print('  paired_stats import failed', e)

z = np.load(os.path.join(EV, 'd8_main__cubes.npz'))
S = (z['samples'].astype(np.float64) + 1) / 2
X = (z['x'].astype(np.float64) + 1) / 2
XR = (z['x_r'].astype(np.float64) + 1) / 2
K, B, C, H, W = S.shape
print('\n== clamp: fraction of sample values at 0 and at 1:', f'{np.mean(S <= 0):.4f}', f'{np.mean(S >= 1):.4f}')
anyc = ((S <= 0) | (S >= 1)).any(0)
print(f'  elements with >=1 clamped sample: {anyc.mean():.4f}; mean x there {X[anyc].mean():.4f} (overall {X.mean():.4f}); mean x_r there {XR[anyc].mean():.4f}')
q = np.quantile(S, [0.25, 0.75, 0.05, 0.95], axis=0, method='weibull')
c50 = (X >= q[0]) & (X <= q[1]); c90 = (X >= q[2]) & (X <= q[3])
print(f'  coverage on clamped elements: cov50 {c50[anyc].mean():.3f} cov90 {c90[anyc].mean():.3f}; on the rest: cov50 {c50[~anyc].mean():.3f} cov90 {c90[~anyc].mean():.3f}')
# by band: clamp rate and coverage per spectral band (dark bands?)
cr = ((S <= 0) | (S >= 1)).mean(axis=(0, 1, 3, 4))
print('  clamp rate by band (every 8th band):', ' '.join(f'{cr[i]:.3f}' for i in range(0, C, 8)))
print('  cov50 by band (every 8th):', ' '.join(f'{c50[:, i].mean():.3f}' for i in range(0, C, 8)))
print('  cov90 by band (every 8th):', ' '.join(f'{c90[:, i].mean():.3f}' for i in range(0, C, 8)))
# sensor phase: coverage at the lowest-error phase vs the rest
pos = (np.arange(H)[:, None] % 8) * 8 + (np.arange(W)[None, :] % 8)
posb = np.broadcast_to(pos, X.shape)
SD = S.std(0, ddof=1)
vh = np.load(os.path.join(SCR, 'vh_sigma.npz')); SIG = vh['sigma'].astype(np.float64) / 2
ERR = np.abs(X - XR)
e_pos = np.array([np.sqrt((ERR[posb == p] ** 2).mean()) for p in range(64)])
best = int(np.argmin(e_pos)); sel = posb == best
print(f'\n== sensor phase with lowest error: (row%8, col%8) = {divmod(best, 8)}; RMS err {e_pos[best]:.4f} vs median phase {np.median(e_pos):.4f}')
print(f'  diffusion mean sd {SD[sel].mean():.4f} cov50 {c50[sel].mean():.3f} cov90 {c90[sel].mean():.3f};  varhead mean sigma {SIG[sel].mean():.4f} cov50 {np.mean(ERR[sel] <= M.Z[0.5]*SIG[sel]):.3f} cov90 {np.mean(ERR[sel] <= M.Z[0.9]*SIG[sel]):.3f}')
print('  RMS err by phase (8x8):'); print(np.round(e_pos.reshape(8, 8), 4))
print('  diffusion mean sd by phase (8x8):'); print(np.round(np.array([SD[posb == p].mean() for p in range(64)]).reshape(8, 8), 4))
