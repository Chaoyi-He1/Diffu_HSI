"""Record-level steelman checks: S1 debiasing for finite-K Monte-Carlo variance, P2 feasibility bound, S2 noise floor."""
import json, math, sys
import numpy as np
sys.path.insert(0, '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan')
from scripts.paired_stats import cluster_se

R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
rec = json.load(open(R + '/uq/d8/' + 'ev' + 'al/d8_main.json'))
pc, files, K = rec['per_cube'], rec['files'], rec['K']


def paired(a, b, name):
    d = np.asarray(a, float) - np.asarray(b, float)
    se_c, G = cluster_se(d, files)
    print(f'{name:60s} mean {d.mean():+.4f}  se {d.std(ddof=1)/math.sqrt(len(d)):.4f}  se_cl {se_c:.4f}')
    return d.mean(), se_c


dif = pc['diffusion']
rm_meanK = np.asarray(dif['mean_rmse_pct'])
sr = np.asarray(dif['spread_ratio'])          # mean s^2 / mean (muhat - x)^2 per cube
rm_reg = np.asarray(pc['regressor']['rmse_pct'])
rm_base = np.asarray(pc['baseline']['rmse_pct'])
rm_single = np.asarray(dif['single_rmse_pct'])

# unbiased MSE of the infinite-K sampler mean: E (muhat - x)^2 = (mu_inf - x)^2 + s^2 / K
mse_meanK = (rm_meanK / 100) ** 2
var_pred = sr * mse_meanK                              # mean predicted variance per cube on [0,1]^2
mse_inf = mse_meanK - var_pred / K
rm_inf = 100 * np.sqrt(np.maximum(mse_inf, 0))
print('mean RMSE meanK', rm_meanK.mean(), ' debiased infinite-K mean RMSE', rm_inf.mean(), ' regressor', rm_reg.mean())
print('S1 literal:')
paired(rm_meanK, rm_reg, 'RMSE_meanK - RMSE_regressor (K=32, literal S1)')
print('S1 debiased (MC term removed):')
m, se = paired(rm_inf, rm_reg, 'RMSE_inf - RMSE_regressor (debiased)')
print('   |mean| <= 2 se_cl ?', abs(m) <= 2 * se)
# predicted literal S1 at K = 64 and K = 1000
for KK in (64, 128, 1000):
    rm_K = 100 * np.sqrt(mse_inf + var_pred / KK)
    m, se = paired(rm_K, rm_reg, f'predicted RMSE_meanK(K={KK}) - RMSE_regressor')
    print('   passes literal S1 ?', abs(m) <= 2 * se)
# Expected MC inflation for a perfectly centred calibrated sampler (mu_inf = x_R, spread = MSE)
print('MC-only inflation if mu_inf == x_R and spread ratio 1: RMSE_R*(sqrt(1+1/K)-1) =', (rm_reg * (np.sqrt(1 + 1 / K) - 1)).mean())

print('\nP2 feasibility (single sample vs baseline):')
pred_single = 100 * np.sqrt(mse_inf + var_pred)        # E (x_k - x)^2 = (mu_inf - x)^2 + s^2
print('observed single RMSE', rm_single.mean(), ' predicted from mean+spread', pred_single.mean())
# a calibrated sampler (spread = MSE of its own mean) has single MSE = 2 MSE(mean)
print('calibrated sampler around the regressor: sqrt(2)*RMSE_reg =', (np.sqrt(2) * rm_reg).mean(), ' baseline', rm_base.mean())
print('needed mean RMSE for a calibrated sampler to pass P2 ~ baseline/sqrt2 =', (rm_base / np.sqrt(2)).mean())
# largest spread ratio compatible with P2 at the current mean
rho_max = (rm_base ** 2 - rm_inf ** 2) / rm_inf ** 2
print('largest spread ratio (pred var / MSE(mean)) compatible with RMSE_single < RMSE_baseline: mean', rho_max.mean(), 'median', np.median(rho_max))
# any calibrated sampler
paired(np.sqrt(2) * rm_reg, rm_base, 'P2 for an ideal calibrated sampler centred on x_R (sqrt2 RMSE_R - base)')
paired(np.sqrt(2) * np.asarray(pc['ensemble']['mean_rmse_pct']), rm_base, 'same, centred on the 2-member ensemble mean')

print('\nS2 noise floor:')
ys = rec['yswap']
d = np.asarray(ys['swap_full']) - np.asarray(ys['true'])
print('S2 mean', d.mean(), 'se', d.std(ddof=1) / math.sqrt(len(d)), 'needed mean > 2 se =', 2 * d.std(ddof=1) / math.sqrt(len(d)))
ysr = np.asarray(ys['regressor']); yst = np.asarray(ys['true'])
print('y-swap cubes: true-cond meanK', yst.mean(), ' regressor', ysr.mean(), ' baseline', np.mean(ys['baseline']))
# the composition centre is the TRUE x_R in both arms; only the conditioning of the residual model is swapped,
# so the swap can only move the RMSE through the residual's conditional mean, which S1 asks to be ~0.
print('fraction of y-swap cubes where swap is worse', np.mean(d > 0))

print('\nC1 / C2 per-cube distribution:')
crps_d, crps_v = np.asarray(dif['crps']), np.asarray(pc['varhead']['crps'])
print('fraction of cubes where diffusion CRPS < varhead', np.mean(crps_d < crps_v))
print('relative CRPS gap (diff/varhead - 1): mean', (crps_d / crps_v - 1).mean(), ' median', np.median(crps_d / crps_v - 1))
print('ratio of CRPS to RMSE: diffusion', (crps_d / rm_reg).mean(), ' varhead', (crps_v / rm_reg).mean(), ' ideal calibrated gaussian ~ 1/sqrt(pi)*sqrt? ')
print('cov50 diffusion per cube quantiles', np.quantile(dif['cov50'], [0.05, 0.25, 0.5, 0.75, 0.95]))
print('cov90 diffusion per cube quantiles', np.quantile(dif['cov90'], [0.05, 0.25, 0.5, 0.75, 0.95]))
print('cov50 varhead per cube quantiles', np.quantile(pc['varhead']['cov50'], [0.05, 0.25, 0.5, 0.75, 0.95]))
print('spread ratio diffusion per cube quantiles', np.quantile(sr, [0.05, 0.25, 0.5, 0.75, 0.95]))
print('spread ratio varhead per cube quantiles', np.quantile(pc['varhead']['spread_ratio'], [0.05, 0.25, 0.5, 0.75, 0.95]))
print('corr(log spread ratio diffusion, log RMSE_reg) across cubes', np.corrcoef(np.log(sr), np.log(rm_reg))[0, 1])
print('corr(log spread ratio varhead, log RMSE_reg) across cubes', np.corrcoef(np.log(pc['varhead']['spread_ratio']), np.log(rm_reg))[0, 1])
# Laplace-error reference: a gaussian of matched variance on Laplace errors covers 1-exp(-z*sqrt2)
for z, lv in ((0.6744897501960817, 0.5), (1.6448536269514722, 0.9)):
    print(f'Gaussian(matched var) on Laplace errors, nominal {lv}: coverage {1 - math.exp(-z * math.sqrt(2)):.3f}')
# per class
cl = np.asarray(rec['classes'])
for c in sorted(set(cl)):
    mk = cl == c
    print(c, mk.sum(), 'CRPS diff', crps_d[mk].mean().round(4), 'varhead', crps_v[mk].mean().round(4), 'cov50', np.mean(np.asarray(dif['cov50'])[mk]).round(3), 'cov90', np.mean(np.asarray(dif['cov90'])[mk]).round(3), 'reg rmse', rm_reg[mk].mean().round(3))
