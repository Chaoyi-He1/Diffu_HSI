import json
R = '/data/chaoyi_he/HSI/Diffu/results/residual_warmstart/runs/'
for run in ('residual_reg_d4_b64', 'residual_reg_d8_b64'):
    rows = [json.loads(l) for l in open(R + run + '/train_log.jsonl') if l.strip()]
    print(run, len(rows), 'keys', sorted(rows[-1].keys()))
    for r in rows:
        ep = r.get('epoch')
        if ep in (1, 5, 10, 25, 50, 75, 80, 90, 95, 100) or r is rows[-1]:
            print('  ', {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if k in ('epoch', 'train_loss', 'val_loss', 'lr', 'monitor_loss', 'val')})
