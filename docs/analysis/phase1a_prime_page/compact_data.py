"""Build the compact JSON the results page embeds (from page_data.json + the monitoring record); print its size."""
import json

P = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/page/'
d = json.load(open(P + 'page_data.json'))
mon = d['monitoring']
r4 = lambda x: None if x is None else round(float(x), 4)
r6 = lambda x: None if x is None else round(float(x), 6)

logs = {}
for run, recs in d['logs'].items():
    logs[run] = [[x['e'], r4(x['loss']), r4(x['val'])] for x in recs]
# re-read sec from the raw logs (the extractor used the wrong key)
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart'
secs = {}
for run in logs:
    secs[run] = [round(json.loads(l)['sec'], 1) for l in open(f'{R}/runs/{run}/train_log.jsonl')]

diag = {}
for k, v in d['diag'].items():
    diag[k] = dict(K=v['K'], n=v['n'], baseline=r4(v['baseline']), ddim1=r4(v['ddim1']), single=r4(v['single']), kmean=r4(v['kmean']),
                   pred_h0=r4(v['pred_h0']), pred_h1=r4(v['pred_h1']), lam=r4(v['lam']), lam_se=r4(v['lam_se']),
                   c2=r4(v['c2']), c1=r4(v['c1']), spread=r4(v['spread']), o=r4(v['o']), ebar=r4(v['ebar']),
                   cal={kk: r4(vv) for kk, vv in v['cal'].items()},
                   pc=dict(b=[r4(x) for x in v['per_cube']['b']], spread=[r4(x) for x in v['per_cube']['spread']], c2=[r4(x) for x in v['per_cube']['c2']]))

ck = {}
for k, v in mon['checkpoints'].items():
    ck[k] = dict(run=v['run'], epoch=v['epoch'], loss_type=v['loss_type'], logged_val=r4(v['logged_val_loss']),
                 mse={a: r6(b) for a, b in v['mse'].items()}, mae={a: r6(b) for a, b in v['mae'].items()})
monc = dict(n=mon['n'], zero=dict(mse={a: r6(b) for a, b in mon['zero']['mse'].items()}, mae={a: r6(b) for a, b in mon['zero']['mae'].items()}),
            checkpoints=ck)

g = d['gate']
gate = dict(models=[dict(run=m['run_name'], loss=m['loss_type'], seed=m['train_seed'], verdict=m['verdict'], rmse=r4(m['rmse_pct']), se=r4(m['rmse_pct_se']),
                         target=r4(m['target_rmse_pct']), target_met=m['target_met'],
                         vs_base={k: r4(m['vs_baseline'][k]) for k in ('diff', 'se', 'se_cluster')} | {'pass': m['vs_baseline']['pass']},
                         vs_b={k: r4(m['vs_b_ddim1'][k]) for k in ('diff', 'se', 'se_cluster')} | {'pass': m['vs_b_ddim1']['pass']},
                         vs_k={k: r4(m['vs_b_t999_kmean'][k]) for k in ('diff', 'se', 'se_cluster')} | {'pass': m['vs_b_t999_kmean']['pass'], 'nfe': m['vs_b_t999_kmean']['nfe']},
                         validity=dict(a={k: r4(m['validity']['swap_full_minus_true'][k]) for k in ('diff', 'se', 'se_cluster')},
                                       b={k: r4(m['validity']['swap_full_minus_baseline'][k]) for k in ('diff', 'se', 'se_cluster')},
                                       g={k: r4(m['validity']['true_minus_baseline'][k]) for k in ('diff', 'se', 'se_cluster')},
                                       valid=m['validity']['valid'], ratio=r4(m['validity']['spec_ratio']), n=m['validity']['n']))
                    for m in g['models']],
            b=dict(rmse=r4(g['b']['rmse_pct']), vs_base={k: r4(g['b']['vs_baseline'][k]) for k in ('diff', 'se', 'se_cluster')},
                   verdict=g['b']['verdict_under_proposed_rule'],
                   validity=dict(a={k: r4(g['b']['validity']['swap_full_minus_true'][k]) for k in ('diff', 'se', 'se_cluster')},
                                 b={k: r4(g['b']['validity']['swap_full_minus_baseline'][k]) for k in ('diff', 'se', 'se_cluster')},
                                 g={k: r4(g['b']['validity']['true_minus_baseline'][k]) for k in ('diff', 'se', 'se_cluster')},
                                 valid=g['b']['validity']['valid'], ratio=r4(g['b']['validity']['spec_ratio']))),
            pairs=[dict(kind=p['kind'], a=p['a'], b=p['b'], diff=r4(p['a_minus_b']['diff']), se=r4(p['a_minus_b']['se']), se_cluster=r4(p['a_minus_b']['se_cluster'])) for p in g['pairs']],
            best=g['best'], best_verdict=g['best_verdict'], go_1b=g['go_1b'])

out = dict(cubes=d['cubes'], grid=d['grid'], logs=logs, secs=secs, yswaps=d['yswaps'], diag=diag, mon=monc, gate=gate)
s = json.dumps(out, separators=(',', ':'))
open(P + 'compact.json', 'w').write(s)
print(len(s), 'bytes')
print(gate['models'][0]['vs_base'], gate['pairs'])
