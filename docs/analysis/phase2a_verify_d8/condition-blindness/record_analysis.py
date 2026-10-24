"""Cube-level heteroscedasticity from the d8_main record (no model calls)."""
import json
import numpy as np
from scipy.stats import spearmanr, pearsonr

R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
rec = json.load(open(f'{R}/uq/d8/eval/d8_main.json'))
pc = rec['per_cube']
print('methods', list(pc.keys()))
for m in pc:
    print(m, sorted(pc[m].keys()))
print('n', rec['n'], 'K', rec['K'], 'sigma_r', rec['sigma_r'], 'sigma_d', rec['sigma_d'])
print('stored_files', rec.get('stored_files'))
ys = rec['yswap']
print('yswap keys', list(ys.keys()), 'n', len(ys['true']))
d = np.array(ys['swap_full']) - np.array(ys['true'])
print('yswap swap-true mean %.4f sd %.4f; true mean %.4f regressor mean %.4f' % (d.mean(), d.std(ddof=1), np.mean(ys['true']), np.mean(ys['regressor'])))

def cube_level(m, rmse_key):
    sr = np.array(pc[m]['spread_ratio'])
    rm = np.array(pc[m][rmse_key]) / 100.0          # RMSE of the predictive mean on [0,1]
    spread = np.sqrt(sr) * rm                        # RMS predicted std on [0,1]
    rho = spearmanr(spread, rm).correlation
    r = pearsonr(np.log(spread), np.log(rm))[0]
    slope = np.polyfit(np.log(rm), np.log(spread), 1)[0]
    print(f'{m:22s} cube-level: spread RMS mean {100*spread.mean():.3f}% (CoV {spread.std()/spread.mean():.3f}), '
          f'err RMS mean {100*rm.mean():.3f}% (CoV {rm.std()/rm.mean():.3f}); Spearman(spread, err) {rho:.3f}; '
          f'Pearson log-log {r:.3f}; log-log slope {slope:.3f}; spread_ratio range [{sr.min():.2f}, {sr.max():.2f}] '
          f'Spearman(spread_ratio, err) {spearmanr(sr, rm).correlation:.3f}')
    return spread, rm

sd_diff, rm_diff = cube_level('diffusion', 'mean_rmse_pct')
sd_vh, rm_vh = cube_level('varhead', 'mean_rmse_pct')
if 'ensemble' in pc:
    cube_level('ensemble', 'mean_rmse_pct')
cube_level('diffusion_halfcentre', 'mean_rmse_pct')
print('Spearman(diffusion spread, varhead spread) across cubes %.3f' % spearmanr(sd_diff, sd_vh).correlation)
# single-sample variance check: single^2 - reg^2 vs implied sample var
reg = np.array(pc['regressor']['rmse_pct']); single = np.array(pc['diffusion']['single_rmse_pct']); meanK = np.array(pc['diffusion']['mean_rmse_pct'])
pred_meanK = np.sqrt(reg ** 2 + (single ** 2 - reg ** 2) / 32.0)
print('mean-of-32 RMSE: observed %.4f, predicted if samples = x_r + independent noise %.4f' % (meanK.mean(), pred_meanK.mean()))
print('per-cube corr(observed, predicted) %.4f; mean diff %.4f' % (pearsonr(meanK, pred_meanK)[0], (meanK - pred_meanK).mean()))
for m in ('diffusion', 'varhead'):
    print(m, 'ause mean %.4f, cov50 %.4f cov90 %.4f crps %.4f' % tuple(np.mean(pc[m][k]) for k in ('ause', 'cov50', 'cov90', 'crps')))
print('diffusion cov50_gauss %.4f cov90_gauss %.4f' % (np.mean(pc['diffusion']['cov50_gauss']), np.mean(pc['diffusion']['cov90_gauss'])))
