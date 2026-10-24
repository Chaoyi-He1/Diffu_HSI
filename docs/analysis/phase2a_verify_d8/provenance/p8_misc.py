import json, sys, collections
import numpy as np
sys.path.insert(0, '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan')
from scripts.paired_stats import cluster_se
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
rec = json.load(open(R + '/uq/d8/' + 'ev' + 'al/d8_main.json'))
sub = json.load(open('/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan/tests/data/val_subset_200.json'))
print('subset classes', sub['classes'], 'record class counts', dict(collections.Counter(rec['classes'])))
print('record files sorted == record files:', rec['files'] == sorted(rec['files']))
pc = rec['per_cube']
d = np.array(pc['diffusion']['mean_rmse_pct']) - np.array(pc['regressor']['rmse_pct'])
se = d.std(ddof=1) / np.sqrt(len(d)); sec, G = cluster_se(d, rec['files'])
print(f'S1 recomputed mean {d.mean():+.6f} se {se:.6f} cluster {sec:.6f} G {G}')
for run in ('regress_l1_sel_d8_b64', 'regress_l1_sel_s43_d8_b64', 'regress_l1_half1_d8_b64', 'regress_l1_half2_d8_b64'):
    a = json.load(open(f'{R}/runs/{run}/args.json'))
    print(run, {k: a.get(k) for k in ('mode', 'd', 'ds', 'limit_files', 'train_half', 'seed', 'loss_type', 'prediction_type', 'num_epochs', 'base_channels', 'batch_size', 'prior')})
ys = rec['yswap']
print('yswap files subset of gate:', set(ys['files']) <= set(rec['files']), 'n', len(ys['files']))
