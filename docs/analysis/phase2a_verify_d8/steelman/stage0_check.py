import json, math, sys
import numpy as np
sys.path.insert(0, '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan')
from scripts.paired_stats import cluster_se
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
rec = json.load(open(R + '/uq/d4/' + 'ev' + 'al/stage0_d4_assets.json'))
pc = rec['per_cube']
print('methods', list(pc.keys()), 'K', rec['K'])
b = pc['b']
rm_mean = np.asarray(b['mean_rmse_pct']); sr = np.asarray(b['spread_ratio'])
var_pred = sr * (rm_mean / 100) ** 2
rm_inf = 100 * np.sqrt(np.maximum((rm_mean / 100) ** 2 - var_pred / rec['K'], 0))
print('B d4: mean RMSE meanK', rm_mean.mean(), ' debiased inf-K', rm_inf.mean(), ' baseline', np.mean(pc['baseline']['rmse_pct']), ' regressor', np.mean(pc['regressor']['rmse_pct']))
print('B d4: corr(log spread ratio, log RMSE_meanK)', np.corrcoef(np.log(sr), np.log(rm_mean))[0, 1])
rms_sd = 100 * np.sqrt(var_pred)
print('B d4: slope log rms_sd on log RMSE_inf', np.polyfit(np.log(rm_inf), np.log(rms_sd), 1)[0], ' rms_sd range', np.quantile(rms_sd, [0.05, 0.5, 0.95]), ' rmse range', np.quantile(rm_inf, [0.05, 0.5, 0.95]))
print('B d4: cov50', np.mean(b['cov50']), 'cov90', np.mean(b['cov90']), 'crps', np.mean(b['crps']), 'ause', np.mean(b['ause']))
cl = np.asarray(rec['classes'])
for c in sorted(set(cl)):
    m = cl == c
    print('  ', c, m.sum(), 'cov50', np.mean(np.asarray(b['cov50'])[m]).round(3), 'cov90', np.mean(np.asarray(b['cov90'])[m]).round(3), 'rmse', rm_mean[m].mean().round(3), 'spread', sr[m].mean().round(3))
# d=8 same slope from the record
r8 = json.load(open(R + '/uq/d8/' + 'ev' + 'al/d8_main.json'))
d8 = r8['per_cube']['diffusion']; rm8 = np.asarray(d8['mean_rmse_pct']); sr8 = np.asarray(d8['spread_ratio'])
vp8 = sr8 * (rm8 / 100) ** 2; inf8 = 100 * np.sqrt((rm8 / 100) ** 2 - vp8 / 32)
print('RR d8: slope log rms_sd on log RMSE_inf', np.polyfit(np.log(inf8), np.log(100 * np.sqrt(vp8)), 1)[0], ' rms_sd range', np.quantile(100 * np.sqrt(vp8), [0.05, 0.5, 0.95]))
vh = r8['per_cube']['varhead']; vpv = np.asarray(vh['spread_ratio']) * (np.asarray(r8['per_cube']['regressor']['rmse_pct']) / 100) ** 2
print('VH d8: slope log rms_sigma on log RMSE_R', np.polyfit(np.log(r8['per_cube']['regressor']['rmse_pct']), np.log(100 * np.sqrt(vpv)), 1)[0])
