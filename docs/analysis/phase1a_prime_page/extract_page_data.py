"""Extract the chart data for the Phase 1a-prime results page from the result files (read-only) into one JSON."""
import json
import os
import re

import numpy as np

R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
OUT = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/page/page_data.json'


def load(p):
    with open(p) as f:
        return json.load(f)


def row(ev, name):
    return next(r for r in ev['rows'] if r['name'] == name)


def per_cube(r):
    return np.asarray(r['per_item']['rmse_pct'], float).mean(0)


b = load(f'{R}/eval/residual/d4/residual_d4_b64__checkpoint_epoch_100.json')
l1 = load(f'{R}/eval/regress/d4/regress_l1_d4_b64__checkpoint_epoch_100__direct.json')
l2 = load(f'{R}/eval/regress/d4/regress_l2_d4_b64__checkpoint_epoch_100__direct.json')
gate = load(f'{R}/eval/regress/d4/gate_1a_prime__regress_l1_d4_b64+regress_l2_d4_b64.json')
diag = load(f'{R}/diagnostics/b_diagnostics.json')
ysB = load(f'{R}/diagnostics/yswap_residual_d4_b64_t999_n32__xT.json')
ysB_leaky = load(f'{R}/diagnostics/yswap_residual_d4_b64_t999_n32.json')
mon = load(f'{R}/diagnostics/regress_monitoring_energy.json')

files = b['baseline_files']
base = dict(zip(files, b['baseline_per_item']['rmse_pct']))
bd = dict(zip(row(b, 'ddim1')['files'], per_cube(row(b, 'ddim1'))))
r1 = dict(zip(row(l1, 'direct')['files'], per_cube(row(l1, 'direct'))))
r2 = dict(zip(row(l2, 'direct')['files'], per_cube(row(l2, 'direct'))))
cls = lambda f: re.search(r'/(P0\d\d)/', f).group(1)
cubes = [dict(f=f, c=cls(f), base=round(base[f], 4), b=round(bd[f], 4), l1=round(r1[f], 4), l2=round(r2[f], 4)) for f in files]

# B's full grid summary (RMSE vs NFE) for the sampling-hurts chart
grid = [dict(name=r['name'], nfe=r['nfe'], rmse=round(r['rmse_pct'], 4), se=round(r['rmse_pct_se'], 4), n=len(r['files']))
        for r in b['rows']]


def train_log(run):
    recs = [json.loads(l) for l in open(f'{R}/runs/{run}/train_log.jsonl')]
    return [dict(e=x['epoch'], loss=round(x['loss'], 5), val=(round(x['val_loss'], 5) if x.get('val_loss') is not None else None),
                 sec=round(x.get('epoch_sec', x.get('seconds', 0)), 1)) for x in recs]


logs = {k: train_log(k) for k in ('regress_l1_d4_b64', 'regress_l2_d4_b64', 'residual_d4_b64')}
print({k: (len(v), v[0], v[-1]) for k, v in logs.items()})

# y-swap summaries
def ys_summary(ys, label):
    pc = ys['per_cube']['999']
    out = dict(label=label, n=len(ys['files']))
    for k in ('true', 'swap_full', 'swap_sensor_only', 'zero'):
        out[k] = round(float(np.mean(pc[k])), 5)
    out['baseline'] = round(float(np.mean(ys['baseline_per_cube'])), 5)
    return out


yswaps = [ys_summary(l1['yswap'], 'L1 regressor'), ys_summary(l2['yswap'], 'L2 regressor'),
          ys_summary(ysB['yswap'] if 'yswap' in ysB else ysB, 'B (leak-free x_T)'),
          ys_summary(ysB_leaky['yswap'] if 'yswap' in ysB_leaky else ysB_leaky, 'B (leaky q_sample, reference)')]

# diagnostics rows
drows = {}
for k, v in diag['rows'].items():
    drows[k] = dict(K=v['K'], n=v['n'], baseline=v['baseline_rmse'], ddim1=v['ddim1_rmse'], single=v['single_rmse']['mean'],
                    kmean=v['kmean_rmse']['mean'], pred_h0=v['pred_h0'], pred_h1=v['pred_h1'],
                    lam=(v.get('lambda_vs_one_step_centre') or {}).get('value'), lam_se=(v.get('lambda_vs_one_step_centre') or {}).get('se_cluster'),
                    c2=v['mean_c2'], c1=v['mean_c2_one_step'], spread=v['mean_spread'], o=v['mean_o'], ebar=v['mean_ebar'],
                    cal=v['calibration'], per_cube=dict(b=[base[f] for f in v['per_cube']['files']], spread=v['per_cube']['spread'], c2=v['per_cube']['c2']))

# monitoring energy record: keep the summary parts
print(list(mon.keys()))
mon_small = {k: v for k, v in mon.items() if k != 'per_file'}
for k in list(mon_small):
    if isinstance(mon_small[k], dict):
        for kk in list(mon_small[k]):
            if isinstance(mon_small[k][kk], (list,)) and len(mon_small[k][kk]) > 50:
                mon_small[k][kk] = f'<{len(mon_small[k][kk])} values>'
print(json.dumps(mon_small, indent=0)[:3000])

json.dump(dict(cubes=cubes, grid=grid, logs=logs, yswaps=yswaps, diag=drows, gate=gate, monitoring=mon), open(OUT, 'w'))
print('wrote', OUT, os.path.getsize(OUT))
